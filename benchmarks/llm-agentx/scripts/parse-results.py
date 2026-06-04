#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Aggregate llm-agentx run JSONL files into a summary JSON."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }


def summarize_run(path: Path) -> dict[str, Any]:
    rows = load_jsonl_rows(path)
    ttfts = [
        float(r["client_ttft_seconds"])
        for r in rows
        if isinstance(r.get("client_ttft_seconds"), (int, float))
    ]
    walls = [
        float(r["client_wall_time_seconds"])
        for r in rows
        if isinstance(r.get("client_wall_time_seconds"), (int, float))
    ]
    tasks = {str(r.get("trace_id")) for r in rows if r.get("trace_id") is not None}
    summary: dict[str, Any] = {
        "file": path.name,
        "request_count": len(rows),
        "trace_count": len(tasks),
        "all_http_ok": all(r.get("http_status") == 200 for r in rows) if rows else False,
    }
    if ttfts:
        summary["ttft_seconds"] = stats(ttfts)
    if walls:
        summary["wall_time_seconds"] = stats(walls)
    return summary


def aggregate(run_dir: Path) -> dict[str, Any]:
    workers = sorted(run_dir.glob("worker-*.jsonl"))
    serial = run_dir / "run.jsonl"
    files = workers if workers else ([serial] if serial.is_file() else [])
    per_file = [summarize_run(p) for p in files]

    all_rows: list[dict[str, Any]] = []
    for p in files:
        all_rows.extend(load_jsonl_rows(p))

    ttfts = [
        float(r["client_ttft_seconds"])
        for r in all_rows
        if isinstance(r.get("client_ttft_seconds"), (int, float))
    ]
    walls = [
        float(r["client_wall_time_seconds"])
        for r in all_rows
        if isinstance(r.get("client_wall_time_seconds"), (int, float))
    ]

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "file_count": len(files),
        "step_count": len(all_rows),
        "trace_count": len({str(r.get("trace_id")) for r in all_rows if r.get("trace_id")}),
        "runs": per_file,
        "all_http_ok": all(r.get("http_status") == 200 for r in all_rows) if all_rows else False,
    }
    if ttfts:
        summary["ttft_seconds"] = stats(ttfts)
    if walls:
        summary["wall_time_seconds"] = stats(walls)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "run_dir",
        type=Path,
        help="run-agent/<timestamp>/ or run-agent-parallel/<timestamp>/",
    )
    p.add_argument("-o", "--output", type=Path, help="Write summary JSON (default: stdout)")
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"error: not a directory: {run_dir}", file=sys.stderr)
        return 1
    data = aggregate(run_dir)
    text = json.dumps(data, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
