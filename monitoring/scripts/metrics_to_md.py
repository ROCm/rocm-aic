#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Generate a GitHub-flavoured Markdown metrics reference from raw Prometheus
# /metrics text files.  Reuses the same parsing logic as metrics_to_html.py.
#
# Usage:
#   python3 metrics_to_md.py \
#     --source vllm:/tmp/metrics_vllm.txt \
#     --source lmcache:/tmp/metrics_lmcache.txt \
#     --source nixl:/tmp/metrics_nixl.txt \
#     [--sha abc1234] \
#     [--title "AIC Prometheus Metrics"] \
#     > docs/prometheus-dump.md

from __future__ import annotations

import argparse
import datetime
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MetricEntry:
    name: str
    type: str = "untyped"
    help: str = ""
    source: str = ""


def parse_metrics(text: str, source: str) -> list[MetricEntry]:
    entries: dict[str, MetricEntry] = {}
    for line in text.splitlines():
        m = re.match(r"^# HELP (\S+) (.+)$", line)
        if m:
            name, desc = m.group(1), m.group(2)
            entry = entries.setdefault(name, MetricEntry(name=name, source=source))
            entry.help = desc
        m = re.match(r"^# TYPE (\S+) (\S+)$", line)
        if m:
            name, typ = m.group(1), m.group(2)
            entry = entries.setdefault(name, MetricEntry(name=name, source=source))
            entry.type = typ
    return sorted(entries.values(), key=lambda e: e.name)


def _badge(typ: str) -> str:
    colors = {
        "counter": "blue",
        "gauge": "green",
        "histogram": "orange",
        "summary": "yellow",
        "untyped": "lightgrey",
    }
    color = colors.get(typ, "lightgrey")
    return f"![{typ}](https://img.shields.io/badge/{typ}-{color})"


def render_md(
    sources: list[tuple[str, list[MetricEntry]]],
    title: str,
    sha: str,
    generated_at: str,
    versions: list[tuple[str, str]] | None = None,
    containers: list[tuple[str, str, str]] | None = None,
) -> str:
    total = sum(len(entries) for _, entries in sources)
    lines: list[str] = []

    lines.append(f"# {title}")
    lines.append("")
    meta_parts = [f"**Generated:** {generated_at}", f"**Total metrics:** {total}"]
    if sha:
        meta_parts.append(f"**SHA:** `{sha}`")
    lines.append(" · ".join(meta_parts))
    lines.append("")

    # Component versions table
    if versions:
        lines.append("## Component Versions")
        lines.append("")
        lines.append("| Component | Version |")
        lines.append("|-----------|---------|")
        for component, version in versions:
            lines.append(f"| {component} | `{version}` |")
        lines.append("")

    # Running containers table
    if containers:
        lines.append("## Running Containers")
        lines.append("")
        lines.append("| Container | Image | Status |")
        lines.append("|-----------|-------|--------|")
        for name, image, status in containers:
            lines.append(f"| `{name}` | `{image}` | {status} |")
        lines.append("")

    # Table of contents
    lines.append("## Metrics Sources")
    lines.append("")
    for source_label, entries in sources:
        if not entries:
            continue
        anchor = source_label.lower().replace(" ", "-").replace("_", "-")
        lines.append(f"- [{source_label}](#{anchor}) — {len(entries)} metrics")
    lines.append("")

    # Per-source tables
    for source_label, entries in sources:
        if not entries:
            continue
        anchor = source_label.lower().replace(" ", "-").replace("_", "-")
        lines.append(f"## {source_label}")
        lines.append("")
        lines.append("| Metric | Type | Description |")
        lines.append("|--------|------|-------------|")
        for e in entries:
            name = f"`{e.name}`"
            typ = e.type
            desc = e.help.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {name} | {typ} | {desc} |")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="LABEL:PATH",
        help="Scraped metrics file; may be repeated (e.g. vllm:/tmp/vllm.txt)",
    )
    parser.add_argument("--sha", default="", help="Git SHA for the metadata line")
    parser.add_argument(
        "--title",
        default="AIC Prometheus Metrics Reference",
        help="Document title",
    )
    parser.add_argument(
        "--version",
        action="append",
        default=[],
        metavar="COMPONENT:VALUE",
        help="Component version for the header table (e.g. vllm:v0.29.0); may be repeated",
    )
    parser.add_argument(
        "--containers-tsv",
        metavar="PATH",
        help="Tab-separated file with columns: Name, Image, Status (one container per line)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="-",
        help="Output path (default: stdout)",
    )
    args = parser.parse_args()

    sources: list[tuple[str, list[MetricEntry]]] = []
    for spec in args.source:
        if ":" not in spec:
            print(f"error: --source must be LABEL:PATH, got {spec!r}", file=sys.stderr)
            return 1
        label, path_str = spec.split(":", 1)
        path = Path(path_str)
        if not path.exists():
            print(f"warning: {path} not found; skipping {label}", file=sys.stderr)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        sources.append((label, parse_metrics(text, label)))

    if not sources:
        print("error: no sources with data", file=sys.stderr)
        return 1

    versions: list[tuple[str, str]] = []
    for spec in args.version:
        if ":" not in spec:
            print(f"error: --version must be COMPONENT:VALUE, got {spec!r}", file=sys.stderr)
            return 1
        component, value = spec.split(":", 1)
        versions.append((component, value))

    containers: list[tuple[str, str, str]] = []
    if args.containers_tsv:
        tsv_path = Path(args.containers_tsv)
        if tsv_path.exists():
            for line in tsv_path.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t", 2)
                if len(parts) == 3 and parts[0]:
                    containers.append((parts[0], parts[1], parts[2]))

    generated_at = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    md = render_md(
        sources, args.title, args.sha, generated_at,
        versions or None, containers or None,
    )

    if args.output == "-":
        sys.stdout.write(md)
    else:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"Written to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
