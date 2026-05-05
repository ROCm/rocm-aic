#!/usr/bin/env python3
"""
Generate summary.json from completed sweep runs.

Can be run standalone to regenerate summary from state.json,
or imported and called programmatically.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from sweep_state import RunState, read_state_file, get_sweep_results_dir


def generate_summary_from_states(completed_states: List[RunState]) -> List[Dict[str, Any]]:
    """
    Generate summary data structure from completed run states.

    Args:
        completed_states: List of completed RunState objects

    Returns:
        List of dictionaries containing run results suitable for JSON serialization
    """
    summary = []

    for run_state in completed_states:
        run_result = {
            'run_id': run_state.run_id,
            'namespace': run_state.namespace,
            'parameters': run_state.parameters,
            'gpu_claim': run_state.gpu_claim,
            'status': run_state.status.value,
            'start_time': run_state.start_time,
            'end_time': run_state.end_time,
            'duration': run_state.end_time - run_state.start_time if run_state.end_time and run_state.start_time else None,
        }

        if run_state.benchmark_results:
            run_result['benchmark'] = run_state.benchmark_results

        if run_state.error:
            run_result['error'] = run_state.error

        summary.append(run_result)

    return summary


def write_summary_file(summary_data: List[Dict[str, Any]], output_file: Path):
    """
    Write summary data to summary.json file.

    Args:
        summary_data: List of run result dictionaries
        output_file: Path to output summary.json file
    """
    with open(output_file, 'w') as f:
        json.dump(summary_data, f, indent=2)


def generate_summary(sweep_dir: Path, verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Generate summary.json from a sweep directory.

    Reads state.json and generates summary.json with all completed runs.

    Args:
        sweep_dir: Path to sweep results directory
        verbose: If True, print progress messages

    Returns:
        Summary data (list of run result dictionaries)

    Raises:
        FileNotFoundError: If state.json doesn't exist
    """
    if verbose:
        print(f"Generating summary for: {sweep_dir}")

    # Read state file
    state_file = sweep_dir / "state.json"
    state = read_state_file(state_file)

    # Generate summary from completed states
    completed_states = state['completed']
    summary_data = generate_summary_from_states(completed_states)

    # Write summary file
    summary_file = sweep_dir / "summary.json"
    write_summary_file(summary_data, summary_file)

    if verbose:
        print(f"✓ Generated summary with {len(summary_data)} runs")
        print(f"  Written to: {summary_file}")

    return summary_data


def print_summary_stats(summary_data: List[Dict[str, Any]]):
    """
    Print statistics about the summary.

    Args:
        summary_data: List of run result dictionaries
    """
    if not summary_data:
        print("No runs in summary")
        return

    successful = sum(1 for r in summary_data if r['status'] in ['success', 'completed'])
    failed = sum(1 for r in summary_data if r['status'] == 'failed')
    cancelled = sum(1 for r in summary_data if r['status'] == 'cancelled')

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total runs: {len(summary_data)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if cancelled > 0:
        print(f"Cancelled: {cancelled}")

    if failed > 0:
        print("\nFailed runs:")
        for run in summary_data:
            if run['status'] == 'failed':
                print(f"  Run {run['run_id']}: {run.get('error', 'Unknown error')}")

    if cancelled > 0:
        print("\nCancelled runs:")
        for run in summary_data:
            if run['status'] == 'cancelled':
                print(f"  Run {run['run_id']}")


def main():
    """CLI entry point for generating summary."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate summary.json from a completed sweep",
        epilog="""
Examples:
  # Generate summary from sweep directory name:
  %(prog)s my-sweep_2026-05-04

  # Generate from full path:
  %(prog)s results/sweeps/my-sweep_2026-05-04

  # Generate with statistics:
  %(prog)s my-sweep_2026-05-04 --stats
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "sweep_dir",
        help="Sweep directory name or path"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print summary statistics after generation"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    try:
        # Resolve sweep directory
        sweep_dir = get_sweep_results_dir(args.sweep_dir)

        # Generate summary
        summary_data = generate_summary(sweep_dir, verbose=not args.quiet)

        # Print statistics if requested
        if args.stats:
            print_summary_stats(summary_data)

        sys.exit(0)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error generating summary: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
