#!/usr/bin/env python3
"""
Aggregate benchmark results from sweep result folders.

This script parses sweep result directories and aggregates metrics from all runs
into a comprehensive JSON or YAML file.

Usage:
    python scripts/aggregate-sweep-results.py results/sweeps/my-sweep-2024-03-15
    python scripts/aggregate-sweep-results.py results/sweeps/my-sweep-2024-03-15 --output aggregated.json
    python scripts/aggregate-sweep-results.py results/sweeps/my-sweep-2024-03-15 --format yaml
    python scripts/aggregate-sweep-results.py results/sweeps/my-sweep-2024-03-15 --pretty
"""

import argparse
import json
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import load generators for parsing
sys.path.insert(0, str(Path(__file__).parent))
from load_generators import get_load_generator


class SweepResultParser:
    """Parse and aggregate sweep result folders."""

    def __init__(self, sweep_dir: Path):
        """
        Initialize parser with sweep directory.

        Args:
            sweep_dir: Path to sweep results directory
        """
        self.sweep_dir = Path(sweep_dir)
        if not self.sweep_dir.exists():
            raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")

        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load sweep metadata.yaml if it exists."""
        metadata_file = self.sweep_dir / "metadata.yaml"
        if metadata_file.exists():
            with open(metadata_file) as f:
                return yaml.safe_load(f)
        return {}

    def parse_sweep(self) -> Dict[str, Any]:
        """
        Parse entire sweep folder.

        Returns:
            Dictionary with:
                - sweep_name: Name of the sweep
                - sweep_dir: Path to sweep directory
                - metadata: Sweep configuration from metadata.yaml
                - runs: List of parsed run results
                - summary: Statistics about the sweep
        """
        result = {
            "sweep_name": self.sweep_dir.name,
            "sweep_dir": str(self.sweep_dir.absolute()),
            "metadata": self.metadata,
            "runs": []
        }

        # Find all run directories
        run_dirs = sorted(self.sweep_dir.glob("run-*"))

        if not run_dirs:
            print(f"Warning: No run directories found in {self.sweep_dir}")

        for run_dir in run_dirs:
            run_data = self._parse_run(run_dir)
            if run_data:
                result["runs"].append(run_data)

        # Compute summary statistics
        successful = sum(1 for r in result["runs"] if r.get("status") == "success")
        result["summary"] = {
            "total_runs": len(result["runs"]),
            "successful_runs": successful,
            "failed_runs": len(result["runs"]) - successful
        }

        return result

    def _parse_run(self, run_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Parse single run directory.

        Args:
            run_dir: Path to run-NNN directory

        Returns:
            Dictionary with run configuration and benchmark results, or None if parsing failed
        """
        try:
            # Load config
            config_file = run_dir / "config.yaml"
            if not config_file.exists():
                print(f"Warning: No config.yaml in {run_dir}")
                return None

            with open(config_file) as f:
                config = yaml.safe_load(f)

            # Extract run ID from directory name
            run_id = int(run_dir.name.split("-")[1])

            # Determine load generator tool
            tool = self.metadata.get("load_generation", {}).get("tool", "unknown")

            # Parse output files based on tool
            benchmark_data = self._parse_benchmark_output(run_dir, tool)

            return {
                "run_id": run_id,
                "config": config,
                "benchmark": benchmark_data,
                "status": "success" if benchmark_data.get("exit_code") == 0 else "failed"
            }

        except Exception as e:
            print(f"Error parsing {run_dir}: {e}")
            return None

    def _parse_benchmark_output(self, run_dir: Path, tool: str) -> Dict[str, Any]:
        """
        Parse benchmark output files using appropriate load generator.

        Args:
            run_dir: Path to run directory
            tool: Load generator tool name

        Returns:
            Dictionary with benchmark results and parsed metrics
        """
        result = {
            "tool": tool,
            "exit_code": 1,
            "parsing_status": "failed"
        }

        try:
            # Create a mock orchestrator (only needed for get_load_generator API)
            class MockOrchestrator:
                pass

            # Get appropriate parser
            generator = get_load_generator(tool, MockOrchestrator())

            if tool == "vllm-bench-serve":
                # Parse clone runs - prefer .json files, fall back to .log files
                runs = []

                # First, try to find JSON files (preferred format)
                json_files = sorted(run_dir.glob("benchmark_output_run*.json"))
                log_files = sorted(run_dir.glob("benchmark_output_run*.log"))

                # Use JSON files if available, otherwise use log files
                output_files = json_files if json_files else log_files

                if not output_files:
                    result["parsing_errors"] = ["No benchmark output files found (tried .json and .log)"]
                    return result

                # Note which format we're using
                using_format = "JSON" if json_files else "log (text)"
                print(f"  Using {using_format} format for {run_dir.name}")

                for output_file in output_files:
                    parsed = generator.parse_metrics(output_file)
                    run_label = output_file.stem.replace("benchmark_output_", "")
                    runs.append({
                        "run_label": run_label,
                        "output_file": str(output_file),
                        **parsed
                    })

                if runs:
                    result["runs"] = runs
                    result["num_runs"] = len(runs)
                    # Consider successful if at least one run was parsed successfully
                    result["exit_code"] = 0 if any(
                        r.get("parsing_status") == "success" for r in runs
                    ) else 1
                    result["parsing_status"] = runs[0].get("parsing_status", "failed")

            elif tool == "multi-turn-benchmark":
                output_file = run_dir / "benchmark_output_run1.txt"
                if not output_file.exists():
                    result["parsing_errors"] = [f"Output file not found: {output_file}"]
                    return result

                parsed = generator.parse_metrics(output_file)
                result.update(parsed)
                result["output_file"] = str(output_file)
                if parsed.get("parsing_status") == "success":
                    result["exit_code"] = 0

            else:
                result["parsing_errors"] = [f"Unknown load generator tool: {tool}"]

        except Exception as e:
            result["parsing_errors"] = [f"Failed to parse: {str(e)}"]

        return result


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark results from sweep folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate results with default output
  python aggregate-sweep-results.py results/sweeps/my-sweep-2024-03-15

  # Specify output file
  python aggregate-sweep-results.py results/sweeps/my-sweep-2024-03-15 --output results.json

  # Output as YAML
  python aggregate-sweep-results.py results/sweeps/my-sweep-2024-03-15 --format yaml

  # Pretty-print JSON
  python aggregate-sweep-results.py results/sweeps/my-sweep-2024-03-15 --pretty
        """
    )
    parser.add_argument(
        "sweep_dir",
        type=Path,
        help="Path to sweep results directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file (default: <sweep_dir>/aggregated_results.json)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "yaml"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--pretty", "-p",
        action="store_true",
        help="Pretty-print JSON output with indentation"
    )

    args = parser.parse_args()

    # Validate sweep directory
    if not args.sweep_dir.exists():
        print(f"Error: Sweep directory not found: {args.sweep_dir}", file=sys.stderr)
        return 1

    # Parse sweep
    print(f"Parsing sweep: {args.sweep_dir}")
    try:
        parser_instance = SweepResultParser(args.sweep_dir)
        results = parser_instance.parse_sweep()
    except Exception as e:
        print(f"Error parsing sweep: {e}", file=sys.stderr)
        return 1

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        suffix = ".json" if args.format == "json" else ".yaml"
        output_file = args.sweep_dir / f"aggregated_results{suffix}"

    # Write results
    print(f"Writing results to: {output_file}")
    try:
        with open(output_file, 'w') as f:
            if args.format == "json":
                indent = 2 if args.pretty else None
                json.dump(results, f, indent=indent)
            else:
                yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        return 1

    # Print summary
    summary = results["summary"]
    print(f"\nSummary:")
    print(f"  Total runs: {summary['total_runs']}")
    print(f"  Successful: {summary['successful_runs']}")
    print(f"  Failed: {summary['failed_runs']}")

    if summary['successful_runs'] > 0:
        print(f"\n✓ Successfully aggregated results from {summary['successful_runs']} runs")
    else:
        print("\n⚠ Warning: No successful runs found", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
