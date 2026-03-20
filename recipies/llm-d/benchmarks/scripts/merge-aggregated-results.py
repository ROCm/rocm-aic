#!/usr/bin/env python3
"""
Simple merge of multiple aggregated_results.json files.
Concatenates runs arrays and recalculates summary statistics.

Usage:
    python merge-aggregated-results.py file1.json file2.json -o merged.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def merge_aggregated_results(files: List[Path]) -> Dict[str, Any]:
    """
    Simple merge: concatenate all runs and recalculate summary.

    Uses the first file's structure as the base template.
    """
    if not files:
        raise ValueError("No files provided")

    # Load first file as base
    with open(files[0]) as f:
        merged = json.load(f)

    # Start with runs from first file
    all_runs = merged["runs"].copy()

    # Append runs from remaining files
    for file_path in files[1:]:
        with open(file_path) as f:
            data = json.load(f)
            all_runs.extend(data["runs"])

    # Update merged result
    merged["runs"] = all_runs

    # Recalculate summary
    successful = sum(1 for r in all_runs if r["status"] == "success")
    failed = sum(1 for r in all_runs if r["status"] == "failed")

    merged["summary"] = {
        "total_runs": len(all_runs),
        "successful_runs": successful,
        "failed_runs": failed
    }

    # Update sweep_name to indicate it's merged
    merged["sweep_name"] = "merged-results"
    merged["sweep_dir"] = "merged"

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple aggregated_results.json files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge two files
  python merge-aggregated-results.py file1.json file2.json -o merged.json

  # Merge all aggregated results in sweep directories
  python merge-aggregated-results.py sweep*/aggregated_results.json --output combined.json

  # Pretty-print output
  python merge-aggregated-results.py file1.json file2.json -o merged.json --pretty
        """
    )

    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Paths to aggregated_results.json files to merge"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output file path for merged results"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output"
    )

    args = parser.parse_args()

    # Validate all files exist
    for file_path in args.files:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return 1

    print(f"Merging {len(args.files)} files...")

    # Perform merge
    try:
        merged = merge_aggregated_results(args.files)
    except Exception as e:
        print(f"Error during merge: {e}")
        return 1

    # Write output
    try:
        with open(args.output, 'w') as f:
            indent = 2 if args.pretty else None
            json.dump(merged, f, indent=indent)
    except Exception as e:
        print(f"Error writing output: {e}")
        return 1

    # Print summary
    print(f"✓ Merged results written to: {args.output}")
    print(f"  Total runs: {merged['summary']['total_runs']}")
    print(f"  Successful: {merged['summary']['successful_runs']}")
    print(f"  Failed: {merged['summary']['failed_runs']}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
