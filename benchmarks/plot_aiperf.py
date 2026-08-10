# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

"""aiperf chart plotter — AIC vs non-AIC generative-AI performance comparison.

Reads one or more CSV files produced by run_aiperf.py and generates:
  1. aiperf-throughput.png — output token throughput (tok/s) vs concurrency
  2. aiperf-ttft.png       — TTFT p50 and p95 (ms) vs concurrency
  3. aiperf-latency.png    — end-to-end request latency p50 and p95 (ms) vs concurrency

Multiple CSVs (one per arm) are overlaid on each chart so VRAM-only and AIC
arms can be compared directly.

Usage:
    python plot_aiperf.py --input logs/manual/results/ --output-dir logs/manual/plots/aiperf/
    python plot_aiperf.py --input aiperf-vram.csv aiperf-kvd.csv --output-dir logs/manual/plots/aiperf/

Arm display names (override with --arm-labels key=label,...):
    vram_only  → "VRAM only (no AIC)"
    vram_dram  → "VRAM + DRAM"
    kvd_v2     → "AIC (NVMe/NFS via LMCache)"
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Arm styling
# ---------------------------------------------------------------------------

_ARM_STYLE: dict[str, dict] = {
    "vram_only": {"color": "#c0392b", "linestyle": "--", "label": "VRAM only (no AIC)"},
    "vram_dram": {"color": "#e67e22", "linestyle": "-.", "label": "VRAM + DRAM"},
    "kvd_v2":    {"color": "#2980b9", "linestyle": "-",  "label": "AIC (NVMe/NFS via LMCache)"},
}

_FALLBACK_COLORS = ["#8e44ad", "#27ae60", "#16a085", "#d35400"]


def _arm_style(arm: str, arm_labels: dict[str, str], color_pool: list) -> dict:
    base = _ARM_STYLE.get(arm, {})
    label = arm_labels.get(arm, base.get("label", arm))
    color = base.get("color") or (color_pool.pop(0) if color_pool else "#555555")
    return {
        "color": color,
        "linestyle": base.get("linestyle", "-"),
        "label": label,
        "marker": "o",
        "markersize": 5,
        "linewidth": 2,
    }


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _safe_float(v: str) -> float | None:
    try:
        f = float(v)
        return f if f == f else None  # drop NaN
    except (ValueError, TypeError):
        return None


def _load_csvs(paths: list[Path]) -> dict[str, dict[int, dict[str, list[float]]]]:
    """Return {arm: {concurrency: {col: [values]}}}."""
    arm_data: dict[str, dict[int, dict[str, list[float]]]] = {}
    for p in paths:
        try:
            with p.open(newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    arm = row.get("arm", "").strip()
                    if not arm:
                        continue
                    try:
                        c = int(row["concurrency"])
                    except (KeyError, ValueError):
                        continue
                    arm_data.setdefault(arm, {})
                    arm_data[arm].setdefault(c, {})
                    for col, val in row.items():
                        if col in ("arm", "concurrency"):
                            continue
                        f = _safe_float(val)
                        if f is not None:
                            arm_data[arm][c].setdefault(col, []).append(f)
        except OSError as exc:
            print(f"WARN: could not read {p}: {exc}", file=sys.stderr)
    return arm_data


def _median_col(
    arm_data: dict[str, dict[int, dict[str, list[float]]]],
    col: str,
) -> dict[str, tuple[list[int], list[float]]]:
    """Return {arm: (sorted concurrencies, median values for col)}."""
    result: dict[str, tuple[list[int], list[float]]] = {}
    for arm, cmap in arm_data.items():
        xs, ys = [], []
        for c in sorted(cmap):
            vals = cmap[c].get(col, [])
            if vals:
                xs.append(c)
                ys.append(statistics.median(vals))
        if xs:
            result[arm] = (xs, ys)
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_chart(
    arms: list[str],
    series: dict[str, tuple[list[int], list[float]]],
    arm_labels: dict[str, str],
    color_pool: list,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    secondary_series: dict[str, tuple[list[int], list[float]]] | None = None,
    secondary_suffix: str = " (p95)",
) -> None:
    """Render one chart and save as PNG.

    ``secondary_series`` overlays a dashed variant (e.g. p95) on the same
    axes as the primary (p50) series.
    """
    import matplotlib.pyplot as plt  # type: ignore[import]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for arm in arms:
        style = _arm_style(arm, arm_labels, color_pool)
        if arm in series:
            xs, ys = series[arm]
            ax.plot(xs, ys, **style)
        if secondary_series and arm in secondary_series:
            xs2, ys2 = secondary_series[arm]
            secondary_style = dict(style)
            secondary_style["linestyle"] = ":"
            secondary_style["label"] = style["label"] + secondary_suffix
            secondary_style["alpha"] = 0.7
            ax.plot(xs2, ys2, **secondary_style)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot aiperf concurrency-sweep CSVs (throughput / TTFT / latency).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", "-i", nargs="+", required=True,
        help="CSV files produced by run_aiperf.py, or a directory containing them",
    )
    p.add_argument(
        "--output-dir", "-o", required=True,
        help="Directory for the generated PNG charts",
    )
    p.add_argument(
        "--arm-labels",
        help="Override arm display names: key=label,key=label  (e.g. kvd_v2='AIC NVMe')",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Resolve CSV paths
    csv_paths: list[Path] = []
    for inp in args.input:
        p = Path(inp)
        if p.is_dir():
            csv_paths.extend(sorted(p.glob("aiperf-*.csv")))
        elif p.is_file():
            csv_paths.append(p)
        else:
            print(f"WARN: {p} does not exist", file=sys.stderr)
    if not csv_paths:
        print("ERROR: no aiperf-*.csv files found", file=sys.stderr)
        return 1

    # Parse arm-label overrides
    arm_labels: dict[str, str] = {}
    if args.arm_labels:
        for kv in args.arm_labels.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                arm_labels[k.strip()] = v.strip().strip("'\"")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_data = _load_csvs(csv_paths)
    if not arm_data:
        print("ERROR: no rows parsed from input CSVs", file=sys.stderr)
        return 1

    arms = sorted(arm_data.keys(), key=lambda a: (a != "vram_only", a))
    color_pool = list(_FALLBACK_COLORS)

    # Try to import matplotlib; give a helpful error if missing
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "ERROR: matplotlib not installed.  Run: pip install 'rocm-aic[plot]'",
            file=sys.stderr,
        )
        return 1

    # --- Chart 1: output token throughput ---
    tput_series = _median_col(arm_data, "output_tok_tput_total")
    if tput_series:
        _plot_chart(
            arms, tput_series, arm_labels, list(color_pool),
            title="AIC Serving — Output Token Throughput vs Concurrency",
            xlabel="Concurrent Requests",
            ylabel="Output Throughput (tokens/sec)",
            out_path=out_dir / "aiperf-throughput.png",
        )

    # --- Chart 2: TTFT p50 + p95 ---
    ttft_p50 = _median_col(arm_data, "ttft_ms_p50")
    ttft_p95 = _median_col(arm_data, "ttft_ms_p95")
    if ttft_p50 or ttft_p95:
        _plot_chart(
            arms, ttft_p50, arm_labels, list(color_pool),
            title="AIC Serving — Time to First Token vs Concurrency",
            xlabel="Concurrent Requests",
            ylabel="TTFT (ms)",
            out_path=out_dir / "aiperf-ttft.png",
            secondary_series=ttft_p95,
            secondary_suffix=" (p95)",
        )

    # --- Chart 3: E2E request latency p50 + p95 ---
    e2e_p50 = _median_col(arm_data, "e2e_ms_p50")
    e2e_p95 = _median_col(arm_data, "e2e_ms_p95")
    if e2e_p50 or e2e_p95:
        _plot_chart(
            arms, e2e_p50, arm_labels, list(color_pool),
            title="AIC Serving — End-to-End Request Latency vs Concurrency",
            xlabel="Concurrent Requests",
            ylabel="Request Latency (ms)",
            out_path=out_dir / "aiperf-latency.png",
            secondary_series=e2e_p95,
            secondary_suffix=" (p95)",
        )

    print(f"Charts written to {out_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
