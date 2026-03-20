# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

"""
Schema discovery for benchmark data.

Analyzes aggregated_results.json to discover available fields, metrics,
and configurations for guided plot generation.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class SchemaDiscovery:
    """Automatically discover available fields in benchmark data."""

    def __init__(self, json_path: str | Path):
        """Initialize with path to aggregated_results.json."""
        self.json_path = Path(json_path)
        self.data = self._load_json()

    def _load_json(self) -> dict:
        """Load the JSON file."""
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def analyze(self, verbose: bool = True) -> dict:
        """
        Analyze the benchmark data and return schema.

        Returns:
            Dictionary with:
            - config_fields: Configuration parameters
            - load_params: Load generation parameters
            - metrics: Available metrics with ranges
            - run_info: Statistics about runs
            - suggested_plots: Suggested visualizations
        """
        schema = {
            "data_source": str(self.json_path),
            "sweep_name": self.data.get("sweep_name", "Unknown"),
            "config_fields": {},
            "load_params": {},
            "metrics": {},
            "run_info": {},
            "suggested_plots": [],
            "insights": [],
        }

        runs = self.data.get("runs", [])
        successful_runs = [r for r in runs if r.get("status") == "success"]
        failed_runs = [r for r in runs if r.get("status") != "success"]

        if not successful_runs:
            schema["insights"].append("⚠ No successful runs found")
            return schema

        # Analyze configurations
        config_values = defaultdict(list)
        load_param_values = defaultdict(list)
        metric_values = defaultdict(list)

        for run in successful_runs:
            config = run.get("config", {})

            # Extract config fields
            for key, value in config.items():
                if key.startswith("_"):
                    continue
                if isinstance(value, dict):
                    # Handle nested like vllm_args
                    continue
                config_values[key].append(value)

            # Extract load parameters
            load_params = config.get("_load_params", {})
            for key, value in load_params.items():
                load_param_values[key].append(value)

            # Extract metrics from benchmark runs
            benchmark_runs = run.get("benchmark", {}).get("runs", [])
            for bench_run in benchmark_runs:
                metrics = bench_run.get("metrics", {})
                self._extract_metrics(metrics, metric_values)

        # Build schema for config fields
        for key, values in config_values.items():
            unique_values = list(set(values))
            schema["config_fields"][key] = {
                "type": type(values[0]).__name__ if values else "unknown",
                "unique_count": len(unique_values),
                "values": sorted(unique_values)[:20],  # Limit to 20
            }

        # Build schema for load params
        for key, values in load_param_values.items():
            unique_values = list(set(values))
            schema["load_params"][key] = {
                "type": type(values[0]).__name__ if values else "unknown",
                "unique_count": len(unique_values),
                "values": sorted(unique_values)[:20],
            }

        # Build schema for metrics
        for key, values in metric_values.items():
            numeric_values = [v for v in values if v is not None and isinstance(v, (int, float))]
            if numeric_values:
                schema["metrics"][key] = {
                    "type": "float",
                    "count": len(numeric_values),
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "mean": sum(numeric_values) / len(numeric_values),
                }

        # Run statistics
        schema["run_info"] = {
            "total_runs": len(runs),
            "successful_runs": len(successful_runs),
            "failed_runs": len(failed_runs),
            "runs_per_config": len(successful_runs[0].get("benchmark", {}).get("runs", [])) if successful_runs else 0,
        }

        # Generate insights
        self._generate_insights(schema, successful_runs, failed_runs)

        # Generate plot suggestions
        self._generate_plot_suggestions(schema)

        return schema

    def _extract_metrics(self, metrics: dict, metric_values: dict, prefix: str = "") -> None:
        """Recursively extract metrics from nested structure."""
        for key, value in metrics.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Recurse into nested metrics
                self._extract_metrics(value, metric_values, full_key)
            elif isinstance(value, (int, float)):
                metric_values[full_key].append(value)

    def _generate_insights(self, schema: dict, successful_runs: list, failed_runs: list) -> None:
        """Generate insights about the data."""
        insights = schema["insights"]

        # Check for failures
        if failed_runs:
            # Find common failure patterns
            failed_configs = defaultdict(int)
            for run in failed_runs:
                config = run.get("config", {})
                tp_size = config.get("tensor_parallel_size")
                if tp_size:
                    failed_configs[f"TP={tp_size}"] += 1

            for config, count in failed_configs.items():
                insights.append(f"⚠ All runs with {config} failed ({count} runs)")

        # Check metric ranges
        for metric, info in schema["metrics"].items():
            if info["max"] / info["min"] > 100 if info["min"] > 0 else False:
                insights.append(
                    f"💡 {metric} spans orders of magnitude "
                    f"({info['min']:.1f} - {info['max']:.1f}) - consider log scale"
                )

        # Check variation
        config_variation = sum(
            1 for field in schema["config_fields"].values()
            if field["unique_count"] > 1
        )
        if config_variation == 0:
            insights.append("⚠ No configuration variation - all runs used same settings")

        # Check load param variation
        load_variation = sum(
            1 for field in schema["load_params"].values()
            if field["unique_count"] > 1
        )
        if load_variation > 0:
            insights.append(
                f"✓ {load_variation} load parameter(s) varied - good for scaling analysis"
            )

    def _generate_plot_suggestions(self, schema: dict) -> None:
        """Generate plot suggestions based on data characteristics."""
        suggestions = schema["suggested_plots"]

        # Suggest scaling plots if ISL varies
        isl_field = None
        for field_name in ["random_input_len", "input_len", "sequence_length"]:
            if field_name in schema["load_params"]:
                isl_field = field_name
                break

        if isl_field and schema["load_params"][isl_field]["unique_count"] > 2:
            suggestions.append({
                "title": "Performance Scaling with Input Length",
                "description": f"{isl_field} has {schema['load_params'][isl_field]['unique_count']} values",
                "spec": {
                    "x_axis": isl_field,
                    "y_axis": "ttft.mean_ms",
                    "x_scale": "log",
                },
            })

        # Suggest TP comparison if TP varies
        if "tensor_parallel_size" in schema["config_fields"]:
            tp_info = schema["config_fields"]["tensor_parallel_size"]
            if tp_info["unique_count"] > 1:
                suggestions.append({
                    "title": "Tensor Parallel Size Comparison",
                    "description": f"Compare {tp_info['unique_count']} TP configurations",
                    "spec": {
                        "x_axis": isl_field or "unknown",
                        "y_axis": "ttft.mean_ms",
                        "series_by": ["tensor_parallel_size"],
                    },
                })

        # Suggest throughput analysis
        if "total_token_throughput" in schema["metrics"]:
            suggestions.append({
                "title": "Throughput Analysis",
                "description": "Analyze token processing throughput",
                "spec": {
                    "x_axis": isl_field or "unknown",
                    "y_axis": "total_token_throughput",
                    "x_scale": "log",
                },
            })

        # Suggest percentile comparison if available
        has_percentiles = any(
            "p99" in metric or "p50" in metric or "median" in metric
            for metric in schema["metrics"].keys()
        )
        if has_percentiles:
            suggestions.append({
                "title": "Latency Percentile Analysis",
                "description": "Compare mean, median, and P99 latencies",
                "spec": {
                    "x_axis": isl_field or "unknown",
                    "y_axis": ["ttft.mean_ms", "ttft.median_ms", "ttft.p99_ms"],
                },
            })

    def print_summary(self, schema: dict) -> None:
        """Print a human-readable summary."""
        print("\n" + "=" * 80)
        print(f"📊 BENCHMARK DATA ANALYSIS")
        print("=" * 80)

        print(f"\n📁 Source: {schema['data_source']}")
        print(f"🏷️  Sweep: {schema['sweep_name']}")

        # Configuration fields
        print(f"\n⚙️  CONFIGURATION PARAMETERS:")
        for field, info in schema["config_fields"].items():
            values_str = str(info["values"])
            if len(values_str) > 60:
                values_str = values_str[:57] + "..."
            print(f"  • {field}: {values_str}")
            print(f"    └─ {info['unique_count']} unique value(s)")

        # Load parameters
        if schema["load_params"]:
            print(f"\n📊 LOAD PARAMETERS:")
            for field, info in schema["load_params"].items():
                values_str = str(info["values"])
                if len(values_str) > 60:
                    values_str = values_str[:57] + "..."
                print(f"  • {field}: {values_str}")
                print(f"    └─ {info['unique_count']} unique value(s)")

        # Metrics
        print(f"\n📈 METRICS (ranges across successful runs):")
        for metric, info in sorted(schema["metrics"].items()):
            print(f"  • {metric}:")
            print(f"    └─ Range: {info['min']:.2f} - {info['max']:.2f} "
                  f"(mean: {info['mean']:.2f})")

        # Run statistics
        print(f"\n📋 RUN STATISTICS:")
        info = schema["run_info"]
        print(f"  • Total runs: {info['total_runs']}")
        print(f"  • Successful: {info['successful_runs']}")
        print(f"  • Failed: {info['failed_runs']}")
        print(f"  • Runs per config: {info['runs_per_config']}")

        # Insights
        if schema["insights"]:
            print(f"\n💡 INSIGHTS:")
            for insight in schema["insights"]:
                print(f"  {insight}")

        # Suggestions
        if schema["suggested_plots"]:
            print(f"\n🎨 SUGGESTED PLOTS:")
            for i, suggestion in enumerate(schema["suggested_plots"], 1):
                print(f"  {i}. {suggestion['title']}")
                print(f"     {suggestion['description']}")

        print("\n" + "=" * 80)

    def save(self, schema: dict, output_path: str | Path) -> None:
        """Save schema to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(schema, f, indent=2)

        print(f"💾 Saved schema to {output_path}")


def main():
    """Command-line interface for schema discovery."""
    parser = argparse.ArgumentParser(
        description="Analyze benchmark data and discover available fields",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze and print summary
  python -m vllm.benchmarks.sweep.schema_discovery \\
      results/sweeps/my_sweep/aggregated_results.json

  # Save schema to JSON
  python -m vllm.benchmarks.sweep.schema_discovery \\
      results/sweeps/my_sweep/aggregated_results.json \\
      --output schema.json
        """
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to aggregated_results.json file"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Save schema to JSON file"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Don't print summary (only with --output)"
    )

    args = parser.parse_args()

    # Analyze data
    discovery = SchemaDiscovery(args.input)
    schema = discovery.analyze()

    # Print summary unless quiet
    if not args.quiet:
        discovery.print_summary(schema)

    # Save if requested
    if args.output:
        discovery.save(schema, args.output)


if __name__ == "__main__":
    main()

