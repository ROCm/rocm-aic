#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Aggregate run-long-parallel worker JSONL files into a summary JSON."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def summarize_worker(path: Path) -> dict[str, Any]:
    data = load_jsonl(path) or {}
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    ttft = data.get("client_ttft_seconds")
    wall = data.get("client_wall_time_seconds")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    prefill_tok_s = None
    if isinstance(prompt_tokens, int) and isinstance(ttft, (int, float)) and ttft > 0:
        prefill_tok_s = prompt_tokens / ttft
    return {
        "file": path.name,
        "worker": data.get("run_long_worker"),
        "seed": data.get("run_long_seed"),
        "book": data.get("run_long_book"),
        "http_status": data.get("http_status"),
        "client_ttft_seconds": ttft,
        "client_wall_time_seconds": wall,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prefill_tok_per_s": prefill_tok_s,
    }


def aggregate(run_dir: Path) -> dict[str, Any]:
    workers = sorted(run_dir.glob("worker-*.jsonl"))
    rows = [summarize_worker(p) for p in workers]
    ttfts = [
        r["client_ttft_seconds"]
        for r in rows
        if isinstance(r.get("client_ttft_seconds"), (int, float))
    ]
    walls = [
        r["client_wall_time_seconds"]
        for r in rows
        if isinstance(r.get("client_wall_time_seconds"), (int, float))
    ]
    prefill_rates = [
        r["prefill_tok_per_s"]
        for r in rows
        if isinstance(r.get("prefill_tok_per_s"), (int, float))
    ]
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "worker_count": len(rows),
        "workers": rows,
        "all_http_ok": all(r.get("http_status") == 200 for r in rows) if rows else False,
    }
    if ttfts:
        summary["ttft_seconds"] = {
            "min": min(ttfts),
            "max": max(ttfts),
            "mean": statistics.fmean(ttfts),
        }
    if walls:
        summary["wall_time_seconds"] = {
            "min": min(walls),
            "max": max(walls),
            "mean": statistics.fmean(walls),
        }
    if prefill_rates:
        summary["prefill_tok_per_s"] = {
            "min": min(prefill_rates),
            "max": max(prefill_rates),
            "mean": statistics.fmean(prefill_rates),
        }
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "run_dir",
        type=Path,
        help="run-long-parallel/<timestamp>/ directory with worker-*.jsonl",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write summary JSON (default: stdout)",
    )
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"error: not a directory: {run_dir}", file=sys.stderr)
        return 1
    data = aggregate(run_dir)
    text = json.dumps(data, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
