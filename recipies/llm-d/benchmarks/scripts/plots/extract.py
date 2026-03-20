# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

"""
Phase 1: Data Extraction from benchmark sweep results.

This module provides functionality to extract and flatten data from nested JSON
structures like aggregated_results.json, using JSONPath-like syntax.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any
import pandas as pd

class JSONPathExtractor:
    """
    Extracts values from nested JSON using a simplified JSONPath syntax.

    Supported syntax:
    - "key" - direct key access
    - "key1.key2.key3" - nested key access
    - "key[0]" - array index access
    - "key[*]" - all array elements (returns list)
    - "key[*].field" - field from all array elements
    """

    def __init__(self, path: str):
        self.path = path
        self.parts = self._parse_path(path)

    def _parse_path(self, path: str) -> list[tuple[str, str | int | None]]:
        """
        Parse a JSONPath string into components.

        Returns list of (key, index) tuples where index can be:
        - None: no indexing
        - int: specific index
        - "*": all elements
        """
        parts = []
        # Split by dots, but handle array notation
        segments = path.split(".")

        for segment in segments:
            # Check for array notation
            match = re.match(r"([^\[]+)(?:\[([^\]]+)\])?", segment)
            if match:
                key = match.group(1)
                index_str = match.group(2)

                if index_str is None:
                    parts.append((key, None))
                elif index_str == "*":
                    parts.append((key, "*"))
                else:
                    try:
                        parts.append((key, int(index_str)))
                    except ValueError:
                        raise ValueError(
                            f"Invalid array index '{index_str}' in path: {path}"
                        )

        return parts

    def extract(self, data: dict | list) -> Any:
        """Extract value from data using the parsed path."""
        return self._extract_recursive(data, self.parts, 0)

    def _extract_recursive(
        self,
        data: Any,
        parts: list[tuple[str, str | int | None]],
        idx: int
    ) -> Any:
        """Recursively extract value following the path."""
        if idx >= len(parts):
            return data

        key, index = parts[idx]

        # Handle dictionary access
        if isinstance(data, dict):
            if key not in data:
                return None
            value = data[key]
        else:
            return None

        # Handle indexing
        if index is None:
            # No indexing, continue
            return self._extract_recursive(value, parts, idx + 1)
        elif index == "*":
            # Extract from all elements
            if not isinstance(value, list):
                return None

            if idx + 1 >= len(parts):
                # This is the last part, return the list
                return value
            else:
                # Continue extraction for each element
                results = []
                for item in value:
                    result = self._extract_recursive(item, parts, idx + 1)
                    if result is not None:
                        results.append(result)
                return results if results else None
        else:
            # Specific index
            if not isinstance(value, list) or index >= len(value):
                return None
            value = value[index]
            return self._extract_recursive(value, parts, idx + 1)


class DataExtractor:
    """Extracts and flattens data from aggregated_results.json"""

    def __init__(self, json_path: str | Path):
        """
        Initialize extractor with path to aggregated_results.json.

        Args:
            json_path: Path to aggregated_results.json file
        """
        self.json_path = Path(json_path)
        self.data = self._load_json()
        self.extracted_records: list[dict[str, Any]] = []

    def _load_json(self) -> dict:
        """Load and parse the JSON file."""
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def extract(
        self,
        field_specs: dict[str, str],
        run_strategy: str = "average",
        filter_failed: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Extract fields from nested JSON according to field_specs.

        Args:
            field_specs: Mapping of output field name to JSONPath
                Example: {"tp_size": "config.tensor_parallel_size"}
            run_strategy: How to handle multiple runs per configuration
                - "average": Average numeric metrics across runs
                - "all": Create separate row for each run
                - "first": Use only first run
                - "last": Use only last run
            filter_failed: If True, skip failed runs

        Returns:
            List of flattened records
        """
        records = []

        # Extract from each run in the sweep
        for run in self.data.get("runs", []):
            # Skip failed runs if requested
            if filter_failed and run.get("status") != "success":
                continue

            # Handle multiple benchmark runs within a single sweep run
            benchmark_runs = run.get("benchmark", {}).get("runs", [])

            if run_strategy == "average" and benchmark_runs:
                record = self._extract_averaged_record(run, field_specs, benchmark_runs)
                if record:
                    records.append(record)

            elif run_strategy == "all":
                for bench_run in benchmark_runs:
                    record = self._extract_single_record(run, field_specs, bench_run)
                    if record:
                        records.append(record)

            elif run_strategy == "first" and benchmark_runs:
                record = self._extract_single_record(run, field_specs, benchmark_runs[0])
                if record:
                    records.append(record)

            elif run_strategy == "last" and benchmark_runs:
                record = self._extract_single_record(run, field_specs, benchmark_runs[-1])
                if record:
                    records.append(record)

        self.extracted_records = records
        return records

    def _extract_single_record(
        self,
        run: dict,
        field_specs: dict[str, str],
        benchmark_run: dict | None = None,
    ) -> dict[str, Any] | None:
        """Extract a single record from a run."""
        record = {}

        for field_name, path_spec in field_specs.items():
            # Create a combined data structure for path resolution
            # This allows paths to reference both run config and benchmark results
            combined_data = {
                **run,
                "benchmark_run": benchmark_run if benchmark_run else {},
            }

            extractor = JSONPathExtractor(path_spec)
            value = extractor.extract(combined_data)

            # If path includes [*], we might get a list
            # For single record extraction, take the first value or average
            if isinstance(value, list):
                if all(isinstance(v, (int, float)) for v in value if v is not None):
                    # Average numeric values
                    numeric_values = [v for v in value if v is not None]
                    value = sum(numeric_values) / len(numeric_values) if numeric_values else None
                elif value:
                    value = value[0]
                else:
                    value = None

            record[field_name] = value

        return record if record else None

    def _extract_averaged_record(
        self,
        run: dict,
        field_specs: dict[str, str],
        benchmark_runs: list[dict],
    ) -> dict[str, Any] | None:
        """Extract a record with metrics averaged across benchmark runs."""
        # First extract from each benchmark run
        run_records = []
        for bench_run in benchmark_runs:
            rec = self._extract_single_record(run, field_specs, bench_run)
            if rec:
                run_records.append(rec)

        if not run_records:
            return None

        # Now average numeric fields across runs
        averaged_record = {}
        for field_name in field_specs.keys():
            values = [rec[field_name] for rec in run_records if rec.get(field_name) is not None]

            if not values:
                averaged_record[field_name] = None
            elif all(isinstance(v, (int, float)) for v in values):
                # Average numeric values
                averaged_record[field_name] = sum(values) / len(values)
            else:
                # For non-numeric, take the first value (they should be the same)
                averaged_record[field_name] = values[0]

        return averaged_record

    def save_extracted_data(
        self,
        output_path: str | Path,
        format: str = "auto",
    ) -> None:
        """
        Save extracted data to file.

        Args:
            output_path: Path to save the data
            format: Output format - "csv", "json", "parquet", or "auto" (infer from extension)
        """
        output_path = Path(output_path)

        if format == "auto":
            format = output_path.suffix.lstrip(".")

        if not self.extracted_records:
            raise ValueError("No data to save. Call extract() first.")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(self.extracted_records, f, indent=2)
        elif format == "csv":
            df = pd.DataFrame(self.extracted_records)
            df.to_csv(output_path, index=False)
        elif format == "parquet":
            df = pd.DataFrame(self.extracted_records)
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Saved {len(self.extracted_records)} records to {output_path}")

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert extracted records to pandas DataFrame."""
        if not self.extracted_records:
            raise ValueError("No data to convert. Call extract() first.")
        return pd.DataFrame(self.extracted_records)


def main():
    """Command-line interface for data extraction."""
    parser = argparse.ArgumentParser(
        description="Extract and flatten data from benchmark sweep results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract basic fields
  python -m vllm.benchmarks.sweep.extract \\
      results/sweeps/my_sweep/aggregated_results.json \\
      --fields isl=config._load_params.random_input_len \\
               tp=config.tensor_parallel_size \\
               ttft=benchmark.runs[*].metrics.ttft.mean_ms \\
      --output extracted_data.csv

  # Extract with specific run strategy
  python -m vllm.benchmarks.sweep.extract \\
      results/sweeps/my_sweep/aggregated_results.json \\
      --fields isl=config._load_params.random_input_len \\
               ttft_run1=benchmark.runs[0].metrics.ttft.mean_ms \\
               ttft_run2=benchmark.runs[1].metrics.ttft.mean_ms \\
      --run-strategy all \\
      --output extracted_data.json
        """
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to aggregated_results.json file"
    )

    parser.add_argument(
        "--fields",
        type=str,
        nargs="+",
        required=True,
        help="Field specifications in format 'name=path'. "
             "Example: isl=config._load_params.random_input_len"
    )

    parser.add_argument(
        "--run-strategy",
        type=str,
        choices=["average", "all", "first", "last"],
        default="average",
        help="How to handle multiple benchmark runs per configuration (default: average)"
    )

    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed runs in output (default: filter them out)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (.json, .csv, or .parquet)"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv", "parquet", "auto"],
        default="auto",
        help="Output format (default: auto-detect from extension)"
    )

    args = parser.parse_args()

    # Parse field specifications
    field_specs = {}
    for field_spec in args.fields:
        if "=" not in field_spec:
            raise ValueError(
                f"Invalid field specification: {field_spec}. "
                "Expected format: 'name=path'"
            )
        name, path = field_spec.split("=", 1)
        field_specs[name] = path

    # Extract data
    extractor = DataExtractor(args.input)
    extractor.extract(
        field_specs=field_specs,
        run_strategy=args.run_strategy,
        filter_failed=not args.include_failed,
    )

    # Save output
    extractor.save_extracted_data(args.output, format=args.format)

    print(f"Extraction complete!")
    print(f"  Total records: {len(extractor.extracted_records)}")
    print(f"  Fields: {', '.join(field_specs.keys())}")


if __name__ == "__main__":
    main()

