#!/usr/bin/env python3
"""
Show status of currently running sweep including in-flight and pending configurations.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import subprocess


def get_latest_sweep_dir(sweep_name: str = None) -> Path:
    """
    Get the latest sweep directory.

    Args:
        sweep_name: Optional sweep name to filter by

    Returns:
        Path to the latest sweep directory
    """
    results_dir = Path("results/sweeps")

    if not results_dir.exists():
        raise FileNotFoundError("No sweeps found in results/sweeps")

    # Get all sweep directories
    sweep_dirs = [d for d in results_dir.iterdir() if d.is_dir()]

    if sweep_name:
        # Filter by sweep name
        sweep_dirs = [d for d in sweep_dirs if d.name.startswith(f"{sweep_name}_")]

    if not sweep_dirs:
        raise FileNotFoundError(f"No sweep directories found{' for ' + sweep_name if sweep_name else ''}")

    # Sort by modification time and return latest
    sweep_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return sweep_dirs[0]


def load_state(sweep_dir: Path) -> Dict[str, Any]:
    """
    Load the state.json file from a sweep directory.

    Args:
        sweep_dir: Path to sweep directory

    Returns:
        State dictionary with pending, running, and completed lists
    """
    state_file = sweep_dir / "state.json"

    if not state_file.exists():
        return {'pending': [], 'running': [], 'completed': []}

    with open(state_file) as f:
        return json.load(f)


def check_namespace_exists(namespace: str) -> bool:
    """
    Check if a namespace exists in Kubernetes.

    Args:
        namespace: Namespace name

    Returns:
        True if namespace exists, False otherwise
    """
    result = subprocess.run(
        ["kubectl", "get", "namespace", namespace],
        capture_output=True,
        text=True
    )
    return result.returncode == 0


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds is None:
        return "N/A"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def show_sweep_status(sweep_dir: Path, verbose: bool = False):
    """
    Show status of a sweep including in-flight and pending configurations.

    Args:
        sweep_dir: Path to sweep directory
        verbose: Show verbose output including completed runs
    """
    print("=" * 80)
    print(f"SWEEP STATUS: {sweep_dir.name}")
    print("=" * 80)

    # Load metadata
    metadata_file = sweep_dir / "metadata.yaml"
    if metadata_file.exists():
        import yaml
        with open(metadata_file) as f:
            metadata = yaml.safe_load(f)

        runtime_config = metadata.get('_runtime_config', {})
        print(f"Sweep Name: {metadata.get('name', 'Unknown')}")
        print(f"Description: {metadata.get('description', 'N/A')}")
        print(f"GPU Budget: {runtime_config.get('gpu_budget', 'Unlimited')}")
        print(f"Max Concurrent: {runtime_config.get('max_concurrent', 1)}")
        print(f"Exclusive Mode: {runtime_config.get('exclusive_mode', False)}")
        print()

    # Load state
    state = load_state(sweep_dir)

    pending = state.get('pending', [])
    running = state.get('running', [])
    completed = state.get('completed', [])

    print(f"Total Configurations: {len(pending) + len(running) + len(completed)}")
    print(f"  Pending: {len(pending)}")
    print(f"  Running: {len(running)}")
    print(f"  Completed: {len(completed)}")
    print()

    # Show running configurations
    if running:
        print("=" * 80)
        print("RUNNING CONFIGURATIONS")
        print("=" * 80)
        print(f"{'Run ID':<8} {'Namespace':<40} {'GPUs':<6} {'Duration':<12} {'Exists':<8}")
        print("-" * 80)

        import time
        current_time = time.time()

        for run in running:
            run_id = run['run_id']
            namespace = run['namespace']
            gpu_claim = run['gpu_claim']
            start_time = run.get('start_time')
            duration = current_time - start_time if start_time else None
            exists = "Yes" if check_namespace_exists(namespace) else "No"

            print(f"{run_id:<8} {namespace:<40} {gpu_claim:<6} {format_duration(duration):<12} {exists:<8}")

        print()

    # Show pending configurations
    if pending:
        print("=" * 80)
        print("PENDING CONFIGURATIONS")
        print("=" * 80)
        print(f"{'Run ID':<8} {'Namespace':<40} {'GPUs':<6}")
        print("-" * 80)

        for run in pending:
            run_id = run['run_id']
            namespace = run['namespace']
            gpu_claim = run['gpu_claim']

            print(f"{run_id:<8} {namespace:<40} {gpu_claim:<6}")

        print()

    # Show completed summary
    if completed:
        print("=" * 80)
        print("COMPLETED SUMMARY")
        print("=" * 80)

        statuses = {}
        for run in completed:
            status = run['status']
            if status not in statuses:
                statuses[status] = 0
            statuses[status] += 1

        for status, count in sorted(statuses.items()):
            print(f"  {status.capitalize()}: {count}")

        if verbose:
            print()
            print("=" * 80)
            print("COMPLETED CONFIGURATIONS (DETAILED)")
            print("=" * 80)
            print(f"{'Run ID':<8} {'Namespace':<40} {'GPUs':<6} {'Status':<12} {'Duration':<12}")
            print("-" * 80)

            for run in completed:
                run_id = run['run_id']
                namespace = run['namespace']
                gpu_claim = run['gpu_claim']
                status = run['status']
                start_time = run.get('start_time')
                end_time = run.get('end_time')
                duration = end_time - start_time if start_time and end_time else None

                print(f"{run_id:<8} {namespace:<40} {gpu_claim:<6} {status:<12} {format_duration(duration):<12}")

        print()

    # Calculate GPU utilization
    if running or pending:
        total_running_gpus = sum(run['gpu_claim'] for run in running)
        total_pending_gpus = sum(run['gpu_claim'] for run in pending)

        metadata_file = sweep_dir / "metadata.yaml"
        if metadata_file.exists():
            import yaml
            with open(metadata_file) as f:
                metadata = yaml.safe_load(f)

            runtime_config = metadata.get('_runtime_config', {})
            gpu_budget = runtime_config.get('gpu_budget')

            if gpu_budget and gpu_budget < 999999:
                print("=" * 80)
                print("GPU UTILIZATION")
                print("=" * 80)
                print(f"Total Budget: {gpu_budget}")
                print(f"In Use (Running): {total_running_gpus}")
                print(f"Available: {gpu_budget - total_running_gpus}")
                print(f"Pending (Waiting): {total_pending_gpus}")
                utilization = (total_running_gpus / gpu_budget) * 100
                print(f"Utilization: {utilization:.1f}%")
                print()

    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Show status of running sweep with in-flight and pending configurations",
        epilog="""
Examples:
  %(prog)s                          # Show latest sweep
  %(prog)s my-sweep                 # Show latest sweep matching name
  %(prog)s --verbose                # Show completed runs in detail
  %(prog)s --sweep-dir results/sweeps/my-sweep_2026-04-15  # Show specific sweep
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "sweep_name",
        nargs="?",
        help="Sweep name to filter by (optional)"
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        help="Direct path to sweep directory"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output including completed runs"
    )
    args = parser.parse_args()

    try:
        if args.sweep_dir:
            sweep_dir = Path(args.sweep_dir)
            if not sweep_dir.exists():
                print(f"Error: Sweep directory not found: {sweep_dir}", file=sys.stderr)
                sys.exit(1)
        else:
            sweep_dir = get_latest_sweep_dir(args.sweep_name)

        show_sweep_status(sweep_dir, verbose=args.verbose)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
