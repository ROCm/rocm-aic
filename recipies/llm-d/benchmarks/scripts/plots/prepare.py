# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

"""
Phase 2: Data Preparation for plotting.

This module provides functionality to filter, transform, and aggregate
extracted benchmark data in preparation for plotting.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

# Import the plot filters from plot_filters module
from .plot_filters import PlotFilters, PlotBinners


class PlotDataPrep:
    """Filters, transforms, and prepares data for plotting"""

    def __init__(self, data: list[dict] | str | Path | "pd.DataFrame"):
        """
        Initialize from extracted data, file path, or DataFrame.

        Args:
            data: Can be:
                - list of dictionaries (extracted records)
                - str/Path to CSV or JSON file
                - pandas DataFrame
        """
        if isinstance(data, (str, Path)):
            self.df = self._load_from_file(Path(data))
        elif isinstance(data, list):
            self.df = pd.DataFrame(data)
        else:
            # Assume it's already a DataFrame
            self.df = data

    def _load_from_file(self, path: Path) -> "pd.DataFrame":
        """Load data from file (CSV, JSON, or Parquet)."""
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        elif suffix == ".json":
            with open(path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        elif suffix == ".parquet":
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def filter(
        self,
        filters: list[str] | PlotFilters | str,
    ) -> "PlotDataPrep":
        """
        Apply filters to the data.

        Args:
            filters: Can be:
                - list of filter strings (e.g., ["tp_size==4", "isl>=1000"])
                - PlotFilters object
                - comma-separated string (e.g., "tp_size==4,isl>=1000")

        Returns:
            Self for method chaining

        Examples:
            prep.filter(["tp_size<=4", "isl<128000"])
            prep.filter("tp_size<=4,isl<128000")
        """
        if isinstance(filters, str):
            filters = PlotFilters.parse_str(filters)
        elif isinstance(filters, list):
            filters = PlotFilters.parse_str(",".join(filters))

        self.df = filters.apply(self.df)
        return self

    def bin(
        self,
        binners: list[str] | PlotBinners | str,
    ) -> "PlotDataPrep":
        """
        Apply binning to numeric columns.

        Args:
            binners: Can be:
                - list of binner strings (e.g., ["throughput%1"])
                - PlotBinners object
                - comma-separated string (e.g., "throughput%1,latency%10")

        Returns:
            Self for method chaining

        Examples:
            prep.bin(["throughput%1"])
            prep.bin("throughput%1,latency%10")
        """
        if isinstance(binners, str):
            binners = PlotBinners.parse_str(binners)
        elif isinstance(binners, list):
            binners = PlotBinners.parse_str(",".join(binners))

        self.df = binners.apply(self.df)
        return self

    def aggregate(
        self,
        group_by: list[str],
        agg_funcs: dict[str, str | list[str]],
    ) -> "PlotDataPrep":
        """
        Aggregate data by grouping variables.

        Args:
            group_by: List of columns to group by
            agg_funcs: Dictionary mapping column names to aggregation functions
                Examples: {"ttft": "mean", "throughput": ["mean", "std"]}

        Returns:
            Self for method chaining

        Examples:
            prep.aggregate(
                group_by=["model", "tp_size"],
                agg_funcs={"ttft": "mean", "throughput": ["mean", "std"]}
            )
        """
        if not group_by:
            raise ValueError("group_by must contain at least one column")

        # Validate columns exist
        for col in group_by:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in data")

        for col in agg_funcs.keys():
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in data")

        # Perform aggregation
        grouped = self.df.groupby(group_by, as_index=False)
        self.df = grouped.agg(agg_funcs)

        # Flatten column names if we have multi-level columns
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = [
                f"{col}_{agg}" if agg else col
                for col, agg in self.df.columns
            ]

        return self

    def transform(
        self,
        transformations: dict[str, str | Callable],
    ) -> "PlotDataPrep":
        """
        Apply transformations to create derived columns.

        Args:
            transformations: Dictionary mapping new column names to:
                - Formula string (e.g., "ttft / 1000")
                - Callable function taking row as input

        Returns:
            Self for method chaining

        Examples:
            prep.transform({
                "ttft_sec": "ttft / 1000",
                "tokens_per_ms": lambda row: row['throughput'] / 1000
            })
        """
        for col_name, transform_spec in transformations.items():
            if isinstance(transform_spec, str):
                # It's a formula string - evaluate it
                try:
                    self.df[col_name] = self.df.eval(transform_spec)
                except Exception as e:
                    raise ValueError(
                        f"Error evaluating formula '{transform_spec}' for column '{col_name}': {e}"
                    )
            elif callable(transform_spec):
                # It's a function
                self.df[col_name] = self.df.apply(transform_spec, axis=1)
            else:
                raise ValueError(
                    f"Transform spec for '{col_name}' must be a string formula or callable"
                )

        return self

    def rename(
        self,
        column_mapping: dict[str, str],
    ) -> "PlotDataPrep":
        """
        Rename columns.

        Args:
            column_mapping: Dictionary mapping old names to new names

        Returns:
            Self for method chaining

        Examples:
            prep.rename({"ttft": "time_to_first_token", "tp": "tensor_parallel_size"})
        """
        self.df = self.df.rename(columns=column_mapping)
        return self

    def select(
        self,
        columns: list[str],
    ) -> "PlotDataPrep":
        """
        Select only specific columns.

        Args:
            columns: List of column names to keep

        Returns:
            Self for method chaining

        Examples:
            prep.select(["tp_size", "isl", "ttft", "tpot"])
        """
        missing_cols = set(columns) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")

        self.df = self.df[columns]
        return self

    def drop_nulls(
        self,
        columns: list[str] | None = None,
    ) -> "PlotDataPrep":
        """
        Drop rows with null values.

        Args:
            columns: Specific columns to check for nulls (None = check all)

        Returns:
            Self for method chaining
        """
        if columns:
            self.df = self.df.dropna(subset=columns)
        else:
            self.df = self.df.dropna()
        return self

    def sort(
        self,
        by: list[str] | str,
        ascending: bool = True,
    ) -> "PlotDataPrep":
        """
        Sort the data.

        Args:
            by: Column name(s) to sort by
            ascending: Sort ascending vs descending

        Returns:
            Self for method chaining
        """
        self.df = self.df.sort_values(by=by, ascending=ascending)
        return self

    def save(
        self,
        output_path: str | Path,
        format: str = "auto",
    ) -> None:
        """
        Save prepared data to file.

        Args:
            output_path: Path to save the data
            format: Output format - "csv", "json", "parquet", or "auto"
        """
        output_path = Path(output_path)

        if format == "auto":
            format = output_path.suffix.lstrip(".")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            self.df.to_json(output_path, orient="records", indent=2)
        elif format == "csv":
            self.df.to_csv(output_path, index=False)
        elif format == "parquet":
            self.df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Saved {len(self.df)} records to {output_path}")

    def to_dataframe(self) -> "pd.DataFrame":
        """Return the underlying pandas DataFrame."""
        return self.df

    def summary(self) -> None:
        """Print a summary of the current data."""
        print(f"Data Summary:")
        print(f"  Rows: {len(self.df)}")
        print(f"  Columns: {len(self.df.columns)}")
        print(f"  Column names: {', '.join(self.df.columns)}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        print(f"\nData types:")
        print(self.df.dtypes)


def main():
    """Command-line interface for data preparation."""
    parser = argparse.ArgumentParser(
        description="Filter, transform, and prepare benchmark data for plotting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic filtering
  python -m vllm.benchmarks.sweep.prepare \\
      extracted_data.csv \\
      --filter "tp_size<=4,isl<128000" \\
      --output prepared_data.csv

  # Filter and transform
  python -m vllm.benchmarks.sweep.prepare \\
      extracted_data.csv \\
      --filter "tp_size<=4" \\
      --transform "ttft_sec=ttft/1000" "throughput_k=throughput/1000" \\
      --output prepared_data.csv

  # Aggregate by groups
  python -m vllm.benchmarks.sweep.prepare \\
      extracted_data.csv \\
      --aggregate-by tp_size isl \\
      --aggregate-funcs "ttft=mean" "throughput=mean,std" \\
      --output aggregated_data.csv
        """
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to input data file (.json, .csv, or .parquet)"
    )

    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Comma-separated filter expressions. Example: 'tp_size<=4,isl<128000'"
    )

    parser.add_argument(
        "--bin",
        type=str,
        default="",
        help="Comma-separated bin expressions. Example: 'throughput%%1,latency%%10'"
    )

    parser.add_argument(
        "--transform",
        type=str,
        nargs="*",
        help="Transformations in format 'new_col=formula'. "
             "Example: 'ttft_sec=ttft/1000' 'speed=throughput*2'"
    )

    parser.add_argument(
        "--aggregate-by",
        type=str,
        nargs="*",
        help="Columns to group by for aggregation"
    )

    parser.add_argument(
        "--aggregate-funcs",
        type=str,
        nargs="*",
        help="Aggregation functions in format 'col=func1,func2'. "
             "Example: 'ttft=mean,std' 'throughput=max'"
    )

    parser.add_argument(
        "--rename",
        type=str,
        nargs="*",
        help="Rename columns in format 'old=new'. Example: 'ttft=time_to_first_token'"
    )

    parser.add_argument(
        "--select",
        type=str,
        nargs="*",
        help="Select only these columns"
    )

    parser.add_argument(
        "--drop-nulls",
        action="store_true",
        help="Drop rows with null values"
    )

    parser.add_argument(
        "--sort-by",
        type=str,
        nargs="*",
        help="Columns to sort by"
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

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print data summary after processing"
    )

    args = parser.parse_args()

    # Load data
    prep = PlotDataPrep(args.input)
    print(f"Loaded {len(prep.df)} records from {args.input}")

    # Apply filters
    if args.filter:
        prep.filter(args.filter)
        print(f"After filtering: {len(prep.df)} records")

    # Apply binning
    if args.bin:
        prep.bin(args.bin)
        print(f"Applied binning")

    # Apply transformations
    if args.transform:
        transforms = {}
        for transform_spec in args.transform:
            if "=" not in transform_spec:
                raise ValueError(
                    f"Invalid transform specification: {transform_spec}. "
                    "Expected format: 'new_col=formula'"
                )
            col_name, formula = transform_spec.split("=", 1)
            transforms[col_name] = formula
        prep.transform(transforms)
        print(f"Applied {len(transforms)} transformations")

    # Apply aggregation
    if args.aggregate_by:
        if not args.aggregate_funcs:
            raise ValueError("--aggregate-funcs required when using --aggregate-by")

        agg_funcs = {}
        for agg_spec in args.aggregate_funcs:
            if "=" not in agg_spec:
                raise ValueError(
                    f"Invalid aggregate specification: {agg_spec}. "
                    "Expected format: 'col=func1,func2'"
                )
            col_name, funcs = agg_spec.split("=", 1)
            func_list = funcs.split(",")
            agg_funcs[col_name] = func_list if len(func_list) > 1 else func_list[0]

        prep.aggregate(group_by=args.aggregate_by, agg_funcs=agg_funcs)
        print(f"Aggregated by {args.aggregate_by}")

    # Rename columns
    if args.rename:
        rename_map = {}
        for rename_spec in args.rename:
            if "=" not in rename_spec:
                raise ValueError(
                    f"Invalid rename specification: {rename_spec}. "
                    "Expected format: 'old=new'"
                )
            old_name, new_name = rename_spec.split("=", 1)
            rename_map[old_name] = new_name
        prep.rename(rename_map)
        print(f"Renamed {len(rename_map)} columns")

    # Select columns
    if args.select:
        prep.select(args.select)
        print(f"Selected {len(args.select)} columns")

    # Drop nulls
    if args.drop_nulls:
        before = len(prep.df)
        prep.drop_nulls()
        print(f"Dropped {before - len(prep.df)} rows with null values")

    # Sort
    if args.sort_by:
        prep.sort(args.sort_by)
        print(f"Sorted by {args.sort_by}")

    # Print summary if requested
    if args.summary:
        print("\n" + "=" * 80)
        prep.summary()
        print("=" * 80 + "\n")

    # Save output
    prep.save(args.output, format=args.format)

    print(f"Preparation complete!")
    print(f"  Final records: {len(prep.df)}")
    print(f"  Output saved to: {args.output}")


if __name__ == "__main__":
    main()

