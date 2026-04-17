#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Generate a summary table and graphs from TTFT benchmark results.

Reads a JSON-lines results file produced by run-bench.sh, detects
the GPU and model, and produces:
  1. A summary table (printed and saved as CSV)
  2. A bar chart of mean TTFT by context size and phase
  3. A speedup chart showing warm/cold ratio vs context size
  4. A disk IO chart showing read MiB by phase
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def detect_gpu() -> str:
    """Detect the GPU name via rocminfo or rocm-smi."""
    for cmd in [
        ["rocm-smi", "--showproductname"],
        ["rocminfo"],
    ]:
        try:
            out = subprocess.check_output(
                cmd, stderr=subprocess.DEVNULL, text=True, timeout=10)
            for line in out.splitlines():
                if "Instinct" in line or "Radeon" in line or "gfx" in line:
                    name = line.split(":")[-1].strip() if ":" in line else line.strip()
                    if name:
                        return name
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    return "unknown"


def load_results(path: Path) -> list[dict]:
    records = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_context_chars(tag: str) -> str | None:
    m = re.search(r"(\d+)c-", tag)
    return m.group(1) if m else None


def classify_phase(tag: str) -> str:
    if "tmpfs" in tag:
        return "warm-tmpfs"
    if "disk" in tag:
        return "warm-disk"
    if tag.startswith("cold"):
        return "cold"
    return "other"


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def build_summary(records: list[dict]) -> list[dict]:
    """Build per-(context_size, phase) summary rows."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        ctx = extract_context_chars(r["tag"])
        phase = classify_phase(r["tag"])
        if ctx:
            groups[(ctx, phase)].append(r)

    sizes = sorted(set(k[0] for k in groups), key=int)
    phases = ["cold", "warm-tmpfs", "warm-disk"]

    rows = []
    for sz in sizes:
        cold_recs = groups.get((sz, "cold"), [])
        cold_mean = mean([r["ttft_ms"] for r in cold_recs])

        for phase in phases:
            recs = groups.get((sz, phase), [])
            if not recs:
                continue
            ttfts = [r["ttft_ms"] for r in recs]
            m = mean(ttfts)
            speedup = cold_mean / m if m > 0 and phase != "cold" else 0
            rows.append({
                "ctx_chars": int(sz),
                "phase": phase,
                "n": len(ttfts),
                "mean_ms": round(m, 1),
                "min_ms": round(min(ttfts), 1),
                "max_ms": round(max(ttfts), 1),
                "speedup": round(speedup, 1) if speedup else None,
                "disk_read_mib": round(
                    mean([r.get("disk_read_mib", 0) for r in recs]), 1),
                "disk_write_mib": round(
                    mean([r.get("disk_write_mib", 0) for r in recs]), 1),
                "load_ms": round(
                    mean([r.get("startup_total_ms", 0) for r in recs])),
            })
    return rows


def print_table(rows: list[dict], gpu: str, model: str) -> None:
    print()
    print(f"GPU:   {gpu}")
    print(f"Model: {model}")
    print()
    hdr = (f"{'ctx_chars':<10} {'phase':<14} {'n':>3} {'mean_ms':>10}"
           f" {'min_ms':>10} {'max_ms':>10} {'speedup':>8}"
           f" {'rd_MiB':>8} {'wr_MiB':>8} {'load_ms':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        sp = f"{r['speedup']:.1f}x" if r["speedup"] else ""
        print(f"{r['ctx_chars']:<10} {r['phase']:<14} {r['n']:>3}"
              f" {r['mean_ms']:>10.1f} {r['min_ms']:>10.1f}"
              f" {r['max_ms']:>10.1f} {sp:>8}"
              f" {r['disk_read_mib']:>8.1f} {r['disk_write_mib']:>8.1f}"
              f" {r['load_ms']:>9}")
    print()


def save_csv(rows: list[dict], path: Path, gpu: str, model: str) -> None:
    cols = ["ctx_chars", "phase", "n", "mean_ms", "min_ms", "max_ms",
            "speedup", "disk_read_mib", "disk_write_mib", "load_ms"]
    with path.open("w") as fh:
        fh.write(f"# gpu: {gpu}\n")
        fh.write(f"# model: {model}\n")
        fh.write(",".join(cols) + "\n")
        for r in rows:
            vals = [str(r.get(c, "")) for c in cols]
            fh.write(",".join(vals) + "\n")
    print(f"CSV saved: {path}")


def plot_ttft_bars(rows: list[dict], out: Path,
                   gpu: str, model: str) -> None:
    if not HAS_MPL:
        return

    sizes = sorted(set(r["ctx_chars"] for r in rows))
    phases = ["cold", "warm-tmpfs", "warm-disk"]
    colors = {"cold": "#e74c3c", "warm-tmpfs": "#2ecc71",
              "warm-disk": "#3498db"}
    labels = {"cold": "Cold (no cache)", "warm-tmpfs": "Warm (tmpfs/RAM)",
              "warm-disk": "Warm (disk)"}

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = range(len(sizes))
    width = 0.25

    for i, phase in enumerate(phases):
        vals = []
        for sz in sizes:
            match = [r for r in rows
                     if r["ctx_chars"] == sz and r["phase"] == phase]
            vals.append(match[0]["mean_ms"] if match else 0)
        offset = (i - 1) * width
        bars = ax.bar([x + offset for x in x_pos], vals, width,
                      label=labels[phase], color=colors[phase])
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Context Size (chars)")
    ax.set_ylabel("Mean TTFT (ms)")
    ax.set_title(f"TTFT by Context Size and Cache Phase\n{gpu} | {model}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_yscale("log")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"TTFT bar chart saved: {out}")


def plot_speedup(rows: list[dict], out: Path,
                 gpu: str, model: str) -> None:
    if not HAS_MPL:
        return

    sizes = sorted(set(r["ctx_chars"] for r in rows))
    phases = ["warm-tmpfs", "warm-disk"]
    colors = {"warm-tmpfs": "#2ecc71", "warm-disk": "#3498db"}
    labels = {"warm-tmpfs": "tmpfs (RAM)", "warm-disk": "disk"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for phase in phases:
        vals = []
        for sz in sizes:
            match = [r for r in rows
                     if r["ctx_chars"] == sz and r["phase"] == phase]
            vals.append(match[0]["speedup"] if match and match[0]["speedup"]
                        else 0)
        ax.plot(sizes, vals, "o-", label=labels[phase],
                color=colors[phase], linewidth=2, markersize=8)
        for sz, val in zip(sizes, vals):
            if val > 0:
                ax.annotate(f"{val:.0f}x", (sz, val),
                            textcoords="offset points", xytext=(0, 10),
                            ha="center", fontsize=9)

    ax.set_xlabel("Context Size (chars)")
    ax.set_ylabel("Speedup vs Cold (x)")
    ax.set_title(f"Slot Restore Speedup vs Cold Prefill\n{gpu} | {model}")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Speedup chart saved: {out}")


def plot_disk_io(rows: list[dict], out: Path,
                 gpu: str, model: str) -> None:
    if not HAS_MPL:
        return

    sizes = sorted(set(r["ctx_chars"] for r in rows))
    phases = ["cold", "warm-tmpfs", "warm-disk"]
    colors = {"cold": "#e74c3c", "warm-tmpfs": "#2ecc71",
              "warm-disk": "#3498db"}

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = range(len(sizes))
    width = 0.25

    for i, phase in enumerate(phases):
        vals = []
        for sz in sizes:
            match = [r for r in rows
                     if r["ctx_chars"] == sz and r["phase"] == phase]
            vals.append(match[0]["disk_read_mib"] if match else 0)
        offset = (i - 1) * width
        ax.bar([x + offset for x in x_pos], vals, width,
               label=phase, color=colors[phase])

    ax.set_xlabel("Context Size (chars)")
    ax.set_ylabel("Disk Read (MiB)")
    ax.set_title(f"Disk Read IO by Phase\n{gpu} | {model}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Disk IO chart saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TTFT benchmark report")
    parser.add_argument("results", type=Path,
                        help="Path to results.jsonl file")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for output files "
                        "(default: same as results)")
    parser.add_argument("--gpu", default=None,
                        help="GPU name (auto-detected if omitted)")
    parser.add_argument("--format", choices=["table", "csv", "all"],
                        default="all",
                        help="Output format (default: all)")
    args = parser.parse_args()

    if not args.results.exists():
        print(f"ERROR: {args.results} not found", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output_dir or args.results.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_results(args.results)
    if not records:
        print("ERROR: no records found", file=sys.stderr)
        sys.exit(1)

    gpu = args.gpu or detect_gpu()
    model = records[0].get("model", "unknown")

    rows = build_summary(records)

    print_table(rows, gpu, model)

    if args.format in ("csv", "all"):
        save_csv(rows, out_dir / "summary.csv", gpu, model)

    if args.format == "all" and HAS_MPL:
        plot_ttft_bars(rows, out_dir / "ttft_bars.png", gpu, model)
        plot_speedup(rows, out_dir / "speedup.png", gpu, model)
        plot_disk_io(rows, out_dir / "disk_io.png", gpu, model)
    elif args.format == "all" and not HAS_MPL:
        print("WARNING: matplotlib not installed, skipping graphs")
        print("  pip install matplotlib")

    meta = {
        "gpu": gpu,
        "model": model,
        "n_records": len(records),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_file": str(args.results),
    }
    meta_path = out_dir / "report_meta.json"
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
