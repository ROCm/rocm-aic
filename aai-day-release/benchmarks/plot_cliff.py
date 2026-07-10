"""Cliff chart plotter — AIC vs non-AIC KV cache comparison.

Reads one or more CSV files produced by run_cliff.py and generates:
  1. cliff-throughput.png — total throughput (tok/s) vs concurrency, one line per arm
  2. cliff-latency.png    — p50 and p95 latency (ms) vs concurrency, one line per arm

The "cliff" is the throughput drop that occurs when the KV working set
exceeds VRAM and the engine starts evicting + re-prefilling. AIC arms
(kvd_v2) avoid this by spilling to NVMe/NFS via LMCache+NIXL+hipFile.

Usage:
    python plot_cliff.py --input results/ --output-dir plots/
    python plot_cliff.py --input cliff-vram.csv cliff-kvd.csv --output-dir plots/

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
# Arm styling — colour + display name
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

def _load_csvs(paths: list[Path]) -> dict[str, dict[int, list[float]]]:
    """Return {arm: {concurrency: [throughput_tok_s_total, ...]}}."""
    arm_data: dict[str, dict[int, list]] = {}
    for p in paths:
        with p.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                arm = row.get("arm", "").strip()
                try:
                    c = int(row["concurrency"])
                    thr = float(row["throughput_tok_s_total"])
                    p50 = float(row["p50_latency_s"]) * 1000  # → ms
                    p95 = float(row["p95_latency_s"]) * 1000
                except (KeyError, ValueError):
                    continue
                arm_data.setdefault(arm, {})
                arm_data[arm].setdefault(c, [])
                arm_data[arm][c].append((thr, p50, p95))
    return arm_data


def _median_series(
    arm_data: dict[str, dict[int, list]], col: int
) -> dict[str, tuple[list[int], list[float]]]:
    """Collapse per-iter measurements to median. Returns {arm: (xs, ys)}."""
    result: dict[str, tuple[list[int], list[float]]] = {}
    for arm, cmap in arm_data.items():
        xs = sorted(cmap)
        ys = [statistics.median(v[col] for v in cmap[x]) for x in xs]
        result[arm] = (xs, ys)
    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _detect_cliff(xs: list[int], ys: list[float]) -> int | None:
    """Return the concurrency at which throughput starts declining (the cliff).
    Uses a simple heuristic: first point where throughput drops >10% vs the
    rolling max. Returns None if no clear cliff is found."""
    peak = 0.0
    for x, y in zip(xs, ys):
        if y > peak:
            peak = y
        elif peak > 0 and (peak - y) / peak > 0.10:
            return x
    return None


def _plot_chart(
    arms: list[str],
    series: dict[str, tuple[list[int], list[float]]],
    arm_labels: dict[str, str],
    ylabel: str,
    title: str,
    out_path: Path,
    show_cliff: bool = False,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(10, 6))
    color_pool = list(_FALLBACK_COLORS)

    for arm in arms:
        if arm not in series:
            continue
        xs, ys = series[arm]
        style = _arm_style(arm, arm_labels, color_pool)
        ax.plot(xs, ys, **style)
        if show_cliff:
            cliff_x = _detect_cliff(xs, ys)
            if cliff_x is not None:
                ax.axvline(
                    x=cliff_x, color=style["color"],
                    linestyle=":", linewidth=1.2, alpha=0.7,
                    label=f"{style['label']} cliff @ c={cliff_x}",
                )

    ax.set_xlabel("Concurrency (simultaneous requests)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="CSV files or a directory containing CSV files produced by run_cliff.py",
    )
    parser.add_argument(
        "--output-dir", default="plots",
        help="Directory for output PNG files (default: plots/)",
    )
    parser.add_argument(
        "--arm-labels", default="",
        help="Override arm display names, e.g. 'kvd_v2=AIC NVMe,vram_only=Baseline'",
    )
    parser.add_argument(
        "--arm-order", default="vram_only,vram_dram,kvd_v2",
        help="Comma-separated arm render order (front to back in legend)",
    )
    parser.add_argument(
        "--title-suffix", default="",
        help="Appended to chart titles, e.g. 'gpt-oss-120b MI300X'",
    )
    args = parser.parse_args()

    # Collect CSV paths
    csv_paths: list[Path] = []
    for spec in args.input:
        p = Path(spec)
        if p.is_dir():
            csv_paths.extend(sorted(p.glob("*.csv")))
        elif p.is_file():
            csv_paths.append(p)
        else:
            print(f"WARN: {spec} not found, skipping", file=sys.stderr)

    if not csv_paths:
        print("ERROR: no CSV files found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(csv_paths)} CSV file(s)...")
    arm_data = _load_csvs(csv_paths)
    if not arm_data:
        print("ERROR: no rows parsed from CSVs", file=sys.stderr)
        sys.exit(1)

    arm_labels: dict[str, str] = {}
    if args.arm_labels:
        for pair in args.arm_labels.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                arm_labels[k.strip()] = v.strip()

    arm_order = [a.strip() for a in args.arm_order.split(",")]
    # Add any arms in data but not in the order list (preserves discovery order)
    arm_order += [a for a in arm_data if a not in arm_order]

    out_dir = Path(args.output_dir)
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""

    # Chart 1: throughput cliff
    thr_series = _median_series(arm_data, col=0)
    _plot_chart(
        arms=arm_order,
        series=thr_series,
        arm_labels=arm_labels,
        ylabel="Throughput (tok/s total)",
        title=f"KV Cache Cliff — Throughput vs Concurrency{suffix}",
        out_path=out_dir / "cliff-throughput.png",
        show_cliff=True,
    )

    # Chart 2: latency — p50
    p50_series = _median_series(arm_data, col=1)
    _plot_chart(
        arms=arm_order,
        series=p50_series,
        arm_labels=arm_labels,
        ylabel="p50 Latency (ms)",
        title=f"KV Cache Cliff — p50 Latency vs Concurrency{suffix}",
        out_path=out_dir / "cliff-latency-p50.png",
        show_cliff=False,
    )

    # Chart 3: latency — p95
    p95_series = _median_series(arm_data, col=2)
    _plot_chart(
        arms=arm_order,
        series=p95_series,
        arm_labels=arm_labels,
        ylabel="p95 Latency (ms)",
        title=f"KV Cache Cliff — p95 Latency vs Concurrency{suffix}",
        out_path=out_dir / "cliff-latency-p95.png",
        show_cliff=False,
    )

    print(f"Done. Charts written to {out_dir}/")


if __name__ == "__main__":
    main()
