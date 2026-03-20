#!/usr/bin/env python3
"""
Analyze Worker_TP performance statistics from LMCache logs.

Extracts per-worker statistics for Retrieved (CPU→GPU) and Stored (GPU→CPU) operations,
computing mean, median, P90, P95, P99 for cost and throughput metrics, and generates
distribution plots to visualize outliers.
"""

import re
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError as e:
    print(f"Error: Missing required dependencies: {e}", file=sys.stderr)
    print("\nPlease install required packages:", file=sys.stderr)
    print("  pip install -r scripts/requirements-analyze.txt", file=sys.stderr)
    print("\nOr install manually:", file=sys.stderr)
    print("  pip install numpy matplotlib", file=sys.stderr)
    sys.exit(1)


class WorkerStats:
    """Statistics for a single worker."""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        # Data structures: operation_type -> token_count -> list of measurements
        self.retrieved_data = defaultdict(lambda: {'cost': [], 'throughput': []})
        self.stored_data = defaultdict(lambda: {'cost': [], 'throughput': []})

    def add_retrieved(self, token_count: int, cost_ms: float, throughput_gbs: float):
        """Add a Retrieved (CPU→GPU) measurement."""
        self.retrieved_data[token_count]['cost'].append(cost_ms)
        self.retrieved_data[token_count]['throughput'].append(throughput_gbs)

    def add_stored(self, token_count: int, cost_ms: float, throughput_gbs: float):
        """Add a Stored (GPU→CPU) measurement."""
        self.stored_data[token_count]['cost'].append(cost_ms)
        self.stored_data[token_count]['throughput'].append(throughput_gbs)

    def compute_stats(self, values: List[float]) -> Dict[str, float]:
        """Compute statistical measures for a list of values."""
        if not values:
            return {}

        arr = np.array(values)
        return {
            'count': len(values),
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'p90': float(np.percentile(arr, 90)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99))
        }

    def get_summary(self) -> Dict:
        """Get summary statistics for this worker."""
        summary = {
            'worker_id': self.worker_id,
            'retrieved': {},
            'stored': {}
        }

        # Process Retrieved data
        for token_count, metrics in sorted(self.retrieved_data.items()):
            summary['retrieved'][token_count] = {
                'cost_ms': self.compute_stats(metrics['cost']),
                'throughput_gbs': self.compute_stats(metrics['throughput'])
            }

        # Process Stored data
        for token_count, metrics in sorted(self.stored_data.items()):
            summary['stored'][token_count] = {
                'cost_ms': self.compute_stats(metrics['cost']),
                'throughput_gbs': self.compute_stats(metrics['throughput'])
            }

        return summary


def parse_log_file(log_file: Path) -> Dict[str, WorkerStats]:
    """
    Parse log file and extract worker statistics.

    Returns:
        Dictionary mapping worker_id to WorkerStats object
    """
    workers = {}

    # Regex patterns for parsing
    # Example: (Worker_TP0 pid=831) [32;20m[2026-04-11 14:22:22,207] LMCache INFO:[0m [req_id=...]
    #          Stored 768 out of total 768 tokens. size: 0.0264 GB, cost 11.0666 ms, throughput: 2.3826 GB/s; ...

    worker_pattern = re.compile(r'\(Worker_TP(\d+)')
    stored_pattern = re.compile(
        r'Stored (\d+) out of total \d+ tokens\. '
        r'size: ([\d.]+) GB, cost ([\d.]+) ms, throughput: ([\d.]+) GB/s'
    )
    retrieved_pattern = re.compile(
        r'Retrieved (\d+) out of \d+ required tokens.*?'
        r'size: ([\d.]+) gb, cost ([\d.]+) ms, throughput: ([\d.]+) GB/s'
    )

    with open(log_file, 'r', errors='ignore') as f:
        for line in f:
            # Extract worker ID
            worker_match = worker_pattern.search(line)
            if not worker_match:
                continue

            worker_id = f"Worker_TP{worker_match.group(1)}"

            # Initialize worker if needed
            if worker_id not in workers:
                workers[worker_id] = WorkerStats(worker_id)

            # Check for Stored operations (GPU → CPU)
            stored_match = stored_pattern.search(line)
            if stored_match:
                token_count = int(stored_match.group(1))
                cost_ms = float(stored_match.group(3))
                throughput = float(stored_match.group(4))
                workers[worker_id].add_stored(token_count, cost_ms, throughput)
                continue

            # Check for Retrieved operations (CPU → GPU)
            retrieved_match = retrieved_pattern.search(line)
            if retrieved_match:
                token_count = int(retrieved_match.group(1))
                cost_ms = float(retrieved_match.group(3))
                throughput = float(retrieved_match.group(4))
                workers[worker_id].add_retrieved(token_count, cost_ms, throughput)

    return workers


def print_summary(workers: Dict[str, WorkerStats], output_file: Path = None):
    """Print summary statistics to console and optionally to file."""

    # Collect all summaries
    summaries = {}
    for worker_id, worker in sorted(workers.items()):
        summaries[worker_id] = worker.get_summary()

    # Print to console
    print("=" * 80)
    print("WORKER STATISTICS SUMMARY")
    print("=" * 80)
    print()

    for worker_id, summary in summaries.items():
        print(f"\n{worker_id}")
        print("-" * 80)

        # Retrieved operations (CPU → GPU)
        if summary['retrieved']:
            print("\n  RETRIEVED (CPU → GPU):")
            for token_count, stats in sorted(summary['retrieved'].items()):
                print(f"\n    {token_count} tokens:")
                cost_stats = stats['cost_ms']
                tp_stats = stats['throughput_gbs']

                print(f"      Cost (ms):        count={cost_stats['count']:<6} "
                      f"mean={cost_stats['mean']:>8.3f} median={cost_stats['median']:>8.3f} "
                      f"P90={cost_stats['p90']:>8.3f} P95={cost_stats['p95']:>8.3f} "
                      f"P99={cost_stats['p99']:>8.3f}")
                print(f"      Throughput (GB/s): count={tp_stats['count']:<6} "
                      f"mean={tp_stats['mean']:>8.3f} median={tp_stats['median']:>8.3f} "
                      f"P90={tp_stats['p90']:>8.3f} P95={tp_stats['p95']:>8.3f} "
                      f"P99={tp_stats['p99']:>8.3f}")

        # Stored operations (GPU → CPU)
        if summary['stored']:
            print("\n  STORED (GPU → CPU):")
            for token_count, stats in sorted(summary['stored'].items()):
                print(f"\n    {token_count} tokens:")
                cost_stats = stats['cost_ms']
                tp_stats = stats['throughput_gbs']

                print(f"      Cost (ms):        count={cost_stats['count']:<6} "
                      f"mean={cost_stats['mean']:>8.3f} median={cost_stats['median']:>8.3f} "
                      f"P90={cost_stats['p90']:>8.3f} P95={cost_stats['p95']:>8.3f} "
                      f"P99={cost_stats['p99']:>8.3f}")
                print(f"      Throughput (GB/s): count={tp_stats['count']:<6} "
                      f"mean={tp_stats['mean']:>8.3f} median={tp_stats['median']:>8.3f} "
                      f"P90={tp_stats['p90']:>8.3f} P95={tp_stats['p95']:>8.3f} "
                      f"P99={tp_stats['p99']:>8.3f}")

    print("\n" + "=" * 80)

    # Save to JSON file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(summaries, f, indent=2)
        print(f"\nStatistics saved to: {output_file}")


def plot_distributions(workers: Dict[str, WorkerStats], output_file: Path):
    """
    Create distribution plots for cost values.

    One subplot per worker, showing both Retrieved and Stored distributions.
    """
    num_workers = len(workers)
    if num_workers == 0:
        print("No worker data to plot")
        return

    # Determine grid layout
    cols = min(2, num_workers)
    rows = (num_workers + cols - 1) // cols

    # Create figure with subplots
    fig = plt.figure(figsize=(12 * cols, 8 * rows))
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.3)

    for idx, (worker_id, worker) in enumerate(sorted(workers.items())):
        row = idx // cols
        col = idx % cols
        ax = fig.add_subplot(gs[row, col])

        # Collect all cost values for this worker
        retrieved_costs = []
        stored_costs = []

        for token_count in worker.retrieved_data:
            retrieved_costs.extend(worker.retrieved_data[token_count]['cost'])

        for token_count in worker.stored_data:
            stored_costs.extend(worker.stored_data[token_count]['cost'])

        # Create scatter plot
        positions = []
        colors = []
        costs = []
        labels_set = set()

        if retrieved_costs:
            positions.extend([1] * len(retrieved_costs))
            colors.extend(['blue'] * len(retrieved_costs))
            costs.extend(retrieved_costs)
            labels_set.add('Retrieved (CPU→GPU)')

        if stored_costs:
            positions.extend([2] * len(stored_costs))
            colors.extend(['red'] * len(stored_costs))
            costs.extend(stored_costs)
            labels_set.add('Stored (GPU→CPU)')

        if costs:
            # Scatter plot with jitter
            jitter = np.random.normal(0, 0.04, len(positions))
            ax.scatter(np.array(positions) + jitter, costs, alpha=0.3, s=20, c=colors)

            # Add box plots overlay
            if retrieved_costs:
                bp1 = ax.boxplot([retrieved_costs], positions=[1], widths=0.4,
                                 patch_artist=True, showfliers=False,
                                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                                 medianprops=dict(color='darkblue', linewidth=2))

            if stored_costs:
                bp2 = ax.boxplot([stored_costs], positions=[2], widths=0.4,
                                 patch_artist=True, showfliers=False,
                                 boxprops=dict(facecolor='lightcoral', alpha=0.7),
                                 medianprops=dict(color='darkred', linewidth=2))

            # Styling
            ax.set_xlabel('Operation Type', fontsize=12)
            ax.set_ylabel('Cost (ms)', fontsize=12)
            ax.set_title(f'{worker_id} - Cost Distribution', fontsize=14, fontweight='bold')
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Retrieved\n(CPU→GPU)', 'Stored\n(GPU→CPU)'])
            ax.grid(True, alpha=0.3, axis='y')

            # Add statistics text
            stats_text = []
            if retrieved_costs:
                stats_text.append(f'Retrieved: n={len(retrieved_costs)}, '
                                f'median={np.median(retrieved_costs):.2f}ms, '
                                f'P95={np.percentile(retrieved_costs, 95):.2f}ms')
            if stored_costs:
                stats_text.append(f'Stored: n={len(stored_costs)}, '
                                f'median={np.median(stored_costs):.2f}ms, '
                                f'P95={np.percentile(stored_costs, 95):.2f}ms')

            ax.text(0.5, 0.98, '\n'.join(stats_text),
                   transform=ax.transAxes,
                   fontsize=10,
                   verticalalignment='top',
                   horizontalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nDistribution plots saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Worker_TP performance statistics from LMCache logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single log file
  %(prog)s logs/llm-d-model-server-xxx.log

  # Analyze and save statistics to JSON
  %(prog)s logs/llm-d-model-server-xxx.log --output-stats stats.json

  # Generate distribution plots
  %(prog)s logs/llm-d-model-server-xxx.log --output-plot distributions.png

  # Both stats and plots
  %(prog)s logs/llm-d-model-server-xxx.log \\
      --output-stats stats.json \\
      --output-plot distributions.png
        """
    )

    parser.add_argument('log_file', type=Path,
                       help='Path to log file to analyze')
    parser.add_argument('--output-stats', type=Path,
                       help='Output file for statistics JSON (optional)')
    parser.add_argument('--output-plot', type=Path,
                       help='Output file for distribution plots (optional)')

    args = parser.parse_args()

    # Validate input
    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}")
        return 1

    print(f"Analyzing log file: {args.log_file}")
    print()

    # Parse log file
    workers = parse_log_file(args.log_file)

    if not workers:
        print("No worker statistics found in log file")
        return 1

    print(f"Found {len(workers)} worker(s)")

    # Print summary
    print_summary(workers, args.output_stats)

    # Generate plots if requested
    if args.output_plot:
        try:
            plot_distributions(workers, args.output_plot)
        except Exception as e:
            print(f"Error generating plots: {e}")
            import traceback
            traceback.print_exc()
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
