#!/usr/bin/env python3
"""
Plot a metric from one or more benchmark CSV files.

Each CSV must have been produced by json_to_csv.py and contain at minimum
the columns: concurrency, run_label, and whatever metric column is requested.

Usage:
  python3 plot_metrics.py [options] csv1 [csv2 ...]

Required:
  csv1 [csv2 ...]       One CSV file per series to plot.

Options:
  --metric COLUMN       Column name to plot on Y axis.
                        Default: total_token_throughput
  --run-label LABEL     Which sub-run to plot (run1=cold, run2=warm).
                        Default: run2
  --labels L1 [L2 ...] Legend label for each CSV (in order).
                        Default: CSV filename stems.
  --title TEXT          Plot title.
  --xlabel TEXT         X axis label. Default: "Number of Concurrent Clients"
  --ylabel TEXT         Y axis label. Default: metric column name.
  --xscale linear|log   X axis scale. Default: linear
  --yscale linear|log   Y axis scale. Default: linear
  --ydiv FLOAT          Divide Y values by this factor (e.g. 1000 to show k-tokens/s).
  --yunit TEXT          Unit suffix appended to ylabel (e.g. "k tok/s").
  --out FILE            Output PNG path. Default: plot.png
  --colors C1 [C2 ...]  Color per series (any matplotlib color string).
                        Defaults to 3 preset colors; extra series get random colors.
  --gap-annotation X I J
                        Draw a double-headed arrow at concurrency X between series I
                        and J (0-based indices), labelled with the floor factor "Nx".
                        Example: --gap-annotation 250 0 2
  --width FLOAT         Figure width in inches. Default: 8
  --height FLOAT        Figure height in inches. Default: 5
  --dpi INT             Output DPI. Default: 150

Examples:
  # Throughput cliff (warm run, tokens/s → k tok/s)
  python3 plot_metrics.py \\
    --metric total_token_throughput --run-label run2 \\
    --labels "GPU HBM" "GPU+DRAM" "GPU+AIC" \\
    --title "KV Cache Tier Throughput Cliff — MI300X" \\
    --ylabel "Throughput" --yunit "k tok/s" --ydiv 1000 \\
    --out kv-throughput-cliff-mi300x.png \\
    results/lmcache_gpu_tp1_isl60k_cliff_mi300_gpt_oss_120b.csv \\
    results/lmcache_gpu_cpu_tp1_isl60k_cliff_mi300_gpt_oss_120b.csv \\
    results/lmcache_gpu_aic_tp1_isl60k_cliff_mi300_gpt_oss_120b.csv

  # TTFT climb (warm run, ms → s)
  python3 plot_metrics.py \\
    --metric ttft_mean_ms --run-label run2 \\
    --labels "GPU HBM" "GPU+DRAM" "GPU+AIC" \\
    --title "KV Cache Tier TTFT Climb — MI300X" \\
    --ylabel "TTFT" --yunit "s" --ydiv 1000 \\
    --out kv-ttft-climb-mi300x.png \\
    results/lmcache_gpu_tp1_isl60k_cliff_mi300_gpt_oss_120b.csv \\
    results/lmcache_gpu_cpu_tp1_isl60k_cliff_mi300_gpt_oss_120b.csv \\
    results/lmcache_gpu_aic_tp1_isl60k_cliff_mi300_gpt_oss_120b.csv
"""

import argparse
import csv
import random
import sys
from pathlib import Path

PRESET_COLORS = ["#E8703A", "#4C72B0", "#8B5EA4"]


def load_series(csv_path: Path, metric: str, run_label: str) -> tuple[list, list]:
    x, y = [], []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["run_label"] != run_label:
                continue
            val = row.get(metric, "")
            if val == "":
                continue
            x.append(int(row["concurrency"]))
            y.append(float(val))
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark metrics from CSV files.")
    parser.add_argument("csvfiles", nargs="+", help="Input CSV files (one per series)")
    parser.add_argument("--metric", default="total_token_throughput", help="Column to plot on Y axis")
    parser.add_argument("--run-label", default="run2", help="Sub-run to plot (run1=cold, run2=warm)")
    parser.add_argument("--labels", nargs="+", help="Legend label per CSV")
    parser.add_argument("--title", default="", help="Plot title")
    parser.add_argument("--xlabel", default="Number of Concurrent Clients", help="X axis label")
    parser.add_argument("--ylabel", default="", help="Y axis label")
    parser.add_argument("--xscale", default="linear", choices=["linear", "log"], help="X axis scale")
    parser.add_argument("--yscale", default="linear", choices=["linear", "log"], help="Y axis scale")
    parser.add_argument("--ydiv", type=float, default=1.0, help="Divide Y values by this factor")
    parser.add_argument("--yunit", default="", help="Unit suffix appended to ylabel")
    parser.add_argument("--out", default="plot.png", help="Output PNG path")
    parser.add_argument("--width", type=float, default=8.0, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=5.0, help="Figure height in inches")
    parser.add_argument("--colors", nargs="+", help="Color per series (matplotlib color strings)")
    parser.add_argument("--gap-annotation", nargs=3, metavar=("X", "I", "J"),
                        help="Draw gap arrow at concurrency X between series I and J (0-based)")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    labels = args.labels or [Path(f).stem for f in args.csvfiles]
    if len(labels) < len(args.csvfiles):
        labels += [Path(f).stem for f in args.csvfiles[len(labels):]]

    n = len(args.csvfiles)
    base_colors = args.colors if args.colors else PRESET_COLORS
    colors = base_colors[:n] + [
        "#{:06x}".format(random.randint(0, 0xFFFFFF)) for _ in range(max(0, n - len(base_colors)))
    ]

    ylabel = args.ylabel or args.metric
    if args.yunit:
        ylabel = f"{ylabel} ({args.yunit})"

    fig, ax = plt.subplots(figsize=(args.width, args.height))

    all_series = []
    for csv_path, label, color in zip(args.csvfiles, labels, colors):
        x, y = load_series(Path(csv_path), args.metric, args.run_label)
        if not x:
            print(f"WARNING: no data found in {csv_path} for run_label={args.run_label}", file=sys.stderr)
            all_series.append(([], []))
            continue
        if args.ydiv != 1.0:
            y = [v / args.ydiv for v in y]
        ax.plot(x, y, marker="o", linewidth=2, markersize=5, label=label, color=color)
        all_series.append((x, y))

    ax.set_xscale(args.xscale)
    ax.set_yscale(args.yscale)

    if args.gap_annotation:
        ann_x, idx_i, idx_j = int(args.gap_annotation[0]), int(args.gap_annotation[1]), int(args.gap_annotation[2])

        def y_at(series_idx, target_x):
            xs, ys = all_series[series_idx]
            for xi, yi in zip(xs, ys):
                if xi == target_x:
                    return yi
            return None

        yi = y_at(idx_i, ann_x)
        yj = y_at(idx_j, ann_x)

        if yi is not None and yj is not None:
            y_lo, y_hi = min(yi, yj), max(yi, yj)
            factor = int(y_hi / y_lo)
            y_mid = (y_lo + y_hi) / 2
            ax.annotate(
                "", xy=(ann_x, y_hi), xytext=(ann_x, y_lo),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
            )
            ax.text(
                ann_x + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01,
                y_mid, f"{factor}x",
                va="center", ha="left", fontsize=11, fontweight="bold",
            )
            x_lo, x_hi = ax.get_xlim()
            ax.set_xlim(x_lo, x_hi + (x_hi - x_lo) * 0.08)
        else:
            print(f"WARNING: could not find concurrency={ann_x} in series {idx_i} or {idx_j}", file=sys.stderr)

    ax.set_xlabel(args.xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if args.title:
        ax.set_title(args.title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
