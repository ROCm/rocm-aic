#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Parse vLLM APIServer engine log lines into CSV + SVG time series."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
LINE_RE = re.compile(
    r"\(APIServer pid=\d+\) INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[loggers\.py:\d+\] "
    r"Engine (\d+): Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s,.*?External prefix cache hit rate: ([\d.]+)%"
)


@dataclass(frozen=True)
class Sample:
    ts: datetime
    engine: str
    prompt_tps: float
    generation_tps: float
    external_prefix_hit_pct: float


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def parse_lines(lines: Iterable[str], year: int, engine: str | None) -> list[Sample]:
    out: list[Sample] = []
    for raw in lines:
        line = strip_ansi(raw)
        m = LINE_RE.search(line)
        if not m:
            continue
        ts_str, eng, pt, gt, ex = m.groups()
        if engine is not None and eng != engine:
            continue
        date_part, time_part = ts_str.split(" ", 1)
        month, day = date_part.split("-")
        ts = datetime(year, int(month), int(day), *map(int, time_part.split(":")))
        out.append(
            Sample(
                ts=ts,
                engine=eng,
                prompt_tps=float(pt),
                generation_tps=float(gt),
                external_prefix_hit_pct=float(ex),
            )
        )
    out.sort(key=lambda s: s.ts)
    return out


def write_csv(path: Path, rows: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "timestamp_iso",
                "engine",
                "prompt_throughput_tokens_per_s",
                "generation_throughput_tokens_per_s",
                "external_prefix_cache_hit_pct",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.ts.isoformat(sep=" ", timespec="seconds"),
                    r.engine,
                    f"{r.prompt_tps:.6g}",
                    f"{r.generation_tps:.6g}",
                    f"{r.external_prefix_hit_pct:.6g}",
                ]
            )


def write_svg(path: Path, rows: list[Sample], title: str) -> None:
    """Stacked three-band SVG: prompt TPS, generation TPS, external hit %."""
    if not rows:
        raise ValueError("no rows to plot")

    w, h = 960, 520
    pad_l, pad_r, pad_t, pad_b = 72, 72, 48, 56
    band = (h - pad_t - pad_b) // 3

    t0 = rows[0].ts.timestamp()
    t1 = rows[-1].ts.timestamp()
    span = max(t1 - t0, 1.0)

    def x_of(r: Sample) -> float:
        return pad_l + (r.ts.timestamp() - t0) / span * (w - pad_l - pad_r)

    pts_p = [(x_of(r), r.prompt_tps) for r in rows]
    pts_g = [(x_of(r), r.generation_tps) for r in rows]
    pts_e = [(x_of(r), r.external_prefix_hit_pct) for r in rows]

    ymax_p = max(p for _, p in pts_p) * 1.05 or 1.0
    ymax_g = max(p for _, p in pts_g) * 1.05 or 1.0

    def polyline_d(xs_y: list[tuple[float, float]], y0: float, y1: float, ymax: float) -> str:
        parts: list[str] = []
        for x, v in xs_y:
            yy = y1 - (v / ymax) * (y1 - y0)
            parts.append(f"{x:.2f},{yy:.2f}")
        return "M " + " L ".join(parts) if parts else ""

    svg_parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif" font-size="12">',
        f'<text x="{pad_l}" y="28" font-size="16" font-weight="600">{_xml_esc(title)}</text>',
        f'<text x="{pad_l}" y="44" fill="#555" font-size="11">Source: vLLM APIServer log lines (loggers.py engine snapshot). X = sample time.</text>',
    ]

    bands = [
        ("Prompt throughput (tokens/s)", pts_p, ymax_p, "#1a6cb8"),
        ("Generation throughput (tokens/s)", pts_g, ymax_g, "#2a8f4a"),
        ("External prefix cache hit rate (%)", pts_e, 100.0, "#b85c1a"),
    ]

    for bi, (blabel, pts, ymax, color) in enumerate(bands):
        y_top = pad_t + bi * band
        y_bot = y_top + band - 28
        svg_parts.append(
            f'<rect x="{pad_l - 8}" y="{y_top}" width="{w - pad_l - pad_r + 16}" '
            f'height="{band - 16}" fill="#fafafa" stroke="#ddd"/>'
        )
        svg_parts.append(
            f'<text x="{pad_l}" y="{y_top + 14}" font-weight="600">{_xml_esc(blabel)}</text>'
        )
        d = polyline_d(pts, y_top + 24, y_bot, ymax)
        if d:
            svg_parts.append(
                f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linejoin="round"/>'
            )
        svg_parts.append(
            f'<text x="{w - pad_r}" y="{y_bot + 6}" text-anchor="end" fill="#666" font-size="10">max≈{ymax:.4g}</text>'
        )

    svg_parts.append(
        f'<text x="{w // 2}" y="{h - 12}" text-anchor="middle" fill="#666" font-size="10">'
        f"{rows[0].ts.isoformat(timespec='seconds')} → {rows[-1].ts.isoformat(timespec='seconds')}"
        f" ({len(rows)} points)</text>"
    )
    svg_parts.append("</svg>")
    path.write_text("\n".join(svg_parts), encoding="utf-8")


def _xml_esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "log_file",
        type=Path,
        help="Path to vLLM server tee log (e.g. recipies/vllm-lmcache-hipfile/logs/server.txt).",
    )
    p.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Calendar year for MM-DD timestamps in the log (default: 2026).",
    )
    p.add_argument(
        "--engine",
        type=str,
        default="000",
        help='Engine id substring from log (default: "000").',
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write CSV here (default: <log_dir>/engine_timeseries.csv).",
    )
    p.add_argument(
        "--svg",
        type=Path,
        default=None,
        help="Write SVG chart here (default: <log_dir>/engine_timeseries.svg).",
    )
    args = p.parse_args()
    if not args.log_file.is_file():
        print(f"error: not a file: {args.log_file}", file=sys.stderr)
        return 1

    log_dir = args.log_file.parent
    csv_path = args.csv or (log_dir / "engine_timeseries.csv")
    svg_path = args.svg or (log_dir / "engine_timeseries.svg")

    rows = parse_lines(args.log_file.read_text(encoding="utf-8", errors="replace").splitlines(), args.year, args.engine)
    if not rows:
        print("error: no matching APIServer engine log lines found", file=sys.stderr)
        return 1

    write_csv(csv_path, rows)
    write_svg(svg_path, rows, title=f"vLLM engine {args.engine} — throughput & external prefix hit")

    print(csv_path)
    print(svg_path)
    print(f"points: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
