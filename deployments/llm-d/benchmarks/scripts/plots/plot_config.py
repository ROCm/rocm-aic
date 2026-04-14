# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

"""
Configuration file support for the plotting framework.

Allows users to define complete plotting workflows in YAML or JSON files.
"""

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .extract import DataExtractor
from .prepare import PlotDataPrep
from .plot_framework import PlotSpec, PlotGenerator
from .export_plot_json import PlotJSONExporter
from .plot_from_json import JSONPlotter


class PlotConfig:
    """
    Represents a complete plotting configuration from extraction to plot generation.
    """

    def __init__(self, config_dict: dict[str, Any]):
        """
        Initialize from configuration dictionary.

        Args:
            config_dict: Configuration dictionary with sections:
                - extraction: Data extraction configuration
                - preparation: Data preparation configuration
                - plots: List of plot specifications
                - output: Output configuration
        """
        self.config = config_dict
        self._validate()

    def _validate(self) -> None:
        """Validate the configuration."""
        # Check for multi-source vs single-source mode
        is_multi_source = "sources" in self.config
        has_single_extraction = "extraction" in self.config

        # Sources and extraction are mutually exclusive
        if is_multi_source and has_single_extraction:
            raise ValueError(
                "Configuration cannot have both 'sources' and 'extraction' sections.\n"
                "Use 'sources' for multi-source mode OR 'extraction' for single-source mode."
            )

        if is_multi_source:
            # Validate multi-source configuration
            if not isinstance(self.config["sources"], list):
                raise ValueError("'sources' must be a list")
            if len(self.config["sources"]) == 0:
                raise ValueError("'sources' must contain at least one source")

            # Validate each source
            source_names = set()
            for idx, source in enumerate(self.config["sources"]):
                if not isinstance(source, dict):
                    raise ValueError(f"Source at index {idx} must be a dictionary")

                # Check for required 'name' field
                if "name" not in source:
                    raise ValueError(f"Source at index {idx} is missing required field 'name'")

                source_name = source["name"]
                if not source_name or not isinstance(source_name, str):
                    raise ValueError(f"Source at index {idx} has invalid 'name' (must be non-empty string)")

                # Check for duplicate names
                if source_name in source_names:
                    raise ValueError(
                        f"Duplicate source name '{source_name}'. "
                        f"Each source must have a unique name."
                    )
                source_names.add(source_name)

                # Check for required 'extraction' section
                if "extraction" not in source:
                    raise ValueError(f"Source '{source_name}' is missing required 'extraction' section")

                # Validate extraction section
                extraction = source["extraction"]
                if not isinstance(extraction, dict):
                    raise ValueError(f"Source '{source_name}': 'extraction' must be a dictionary")
                if "input" not in extraction:
                    raise ValueError(f"Source '{source_name}': 'extraction' is missing required 'input' field")
                if "fields" not in extraction:
                    raise ValueError(f"Source '{source_name}': 'extraction' is missing required 'fields' field")

                # Preparation is optional for sources
                if "preparation" in source and not isinstance(source["preparation"], dict):
                    raise ValueError(f"Source '{source_name}': 'preparation' must be a dictionary")

            # Validate merge configuration if present
            merge_strategy = self.config.get("merge_strategy", "concat")
            if merge_strategy not in ["concat", "join"]:
                raise ValueError(
                    f"Invalid merge_strategy '{merge_strategy}'. "
                    f"Must be 'concat' or 'join'."
                )

            if merge_strategy == "join":
                # For join strategy, merge_on is required
                if "merge_on" not in self.config:
                    raise ValueError(
                        "merge_strategy 'join' requires 'merge_on' to specify join keys"
                    )
                merge_on = self.config["merge_on"]
                if not isinstance(merge_on, list) or len(merge_on) == 0:
                    raise ValueError("'merge_on' must be a non-empty list of column names")

                # Validate merge_how if specified
                merge_how = self.config.get("merge_how", "inner")
                if merge_how not in ["inner", "outer", "left", "right"]:
                    raise ValueError(
                        f"Invalid merge_how '{merge_how}'. "
                        f"Must be 'inner', 'outer', 'left', or 'right'."
                    )

        else:
            # Single-source mode - require top-level extraction
            if not has_single_extraction:
                raise ValueError("Missing required section: 'extraction'")

        # Check for either old-style (plots + output) or new-style (export + plot)
        has_old_style = "plots" in self.config and "output" in self.config
        has_new_style = "export" in self.config

        if not has_old_style and not has_new_style:
            raise ValueError(
                "Configuration must contain either:\n"
                "  - 'plots' and 'output' sections (old-style plotting), OR\n"
                "  - 'export' section (new-style JSON plotting)"
            )

        # Validate old-style format
        if has_old_style:
            if not isinstance(self.config["plots"], list):
                raise ValueError("'plots' must be a list")
            if len(self.config["plots"]) == 0:
                raise ValueError("'plots' must contain at least one plot specification")

    @classmethod
    def from_file(cls, config_path: str | Path) -> "PlotConfig":
        """
        Load configuration from YAML or JSON file.

        Args:
            config_path: Path to configuration file

        Returns:
            PlotConfig instance
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        suffix = config_path.suffix.lower()

        if suffix in [".yaml", ".yml"]:
            if not HAS_YAML:
                raise ImportError(
                    "PyYAML is required to load YAML configuration files. "
                    "Install it with: pip install pyyaml"
                )
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif suffix == ".json":
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {suffix}")

        return cls(config_dict)

    def execute(self, verbose: bool = True) -> dict[str, Any]:
        """
        Execute the complete plotting workflow.

        Args:
            verbose: Print progress messages

        Returns:
            Dictionary with execution results
        """
        results = {
            "extracted_records": 0,
            "prepared_records": 0,
            "generated_plots": [],
        }

        # Detect single vs multi-source mode
        is_multi_source = "sources" in self.config

        if is_multi_source:
            # Multi-source workflow
            if verbose:
                print("=" * 80)
                print("Phase 1: Multi-Source Data Extraction")
                print("=" * 80)

            prepared_data, extraction_stats = self._execute_multi_source_extraction(verbose)
            results["extracted_records"] = extraction_stats["total_extracted"]

            if verbose:
                print(f"\nExtraction summary:")
                for source_name, count in extraction_stats["per_source"].items():
                    print(f"  {source_name}: {count} records")
                print(f"  Total: {len(prepared_data)} records after merging")

            # Apply merge preparation if configured
            if "merge_preparation" in self.config:
                if verbose:
                    print("\n" + "=" * 80)
                    print("Phase 2: Merge Preparation")
                    print("=" * 80)

                prepared_data = self._apply_merge_preparation(prepared_data, verbose)

            results["prepared_records"] = len(prepared_data)

        else:
            # Single-source workflow (existing logic)
            if verbose:
                print("=" * 80)
                print("Phase 1: Data Extraction")
                print("=" * 80)

            extraction_config = self.config["extraction"]
            extractor = DataExtractor(extraction_config["input"])

            extracted_data = extractor.extract(
                field_specs=extraction_config["fields"],
                run_strategy=extraction_config.get("run_strategy", "average"),
                filter_failed=extraction_config.get("filter_failed", True),
            )

            results["extracted_records"] = len(extracted_data)

            if verbose:
                print(f"Extracted {len(extracted_data)} records")
                print(f"Fields: {', '.join(extraction_config['fields'].keys())}")

            # Phase 2: Preparation (optional)
            if "preparation" in self.config:
                if verbose:
                    print("\n" + "=" * 80)
                    print("Phase 2: Data Preparation")
                    print("=" * 80)

                prep_config = self.config["preparation"]
                prep = PlotDataPrep(extracted_data)

                # Apply filters
                if "filters" in prep_config:
                    prep.filter(prep_config["filters"])
                    if verbose:
                        print(f"After filtering: {len(prep.df)} records")

                # Apply binning
                if "binning" in prep_config:
                    prep.bin(prep_config["binning"])
                    if verbose:
                        print(f"Applied binning")

                # Apply transformations
                if "transformations" in prep_config:
                    prep.transform(prep_config["transformations"])
                    if verbose:
                        print(f"Applied {len(prep_config['transformations'])} transformations")

                # Apply aggregation
                if "aggregate_by" in prep_config:
                    prep.aggregate(
                        group_by=prep_config["aggregate_by"],
                        agg_funcs=prep_config.get("aggregate_funcs", {}),
                    )
                    if verbose:
                        print(f"Aggregated by {prep_config['aggregate_by']}")

                # Rename columns
                if "rename" in prep_config:
                    prep.rename(prep_config["rename"])
                    if verbose:
                        print(f"Renamed {len(prep_config['rename'])} columns")

                # Drop nulls
                if prep_config.get("drop_nulls", False):
                    before = len(prep.df)
                    prep.drop_nulls()
                    if verbose:
                        print(f"Dropped {before - len(prep.df)} rows with null values")

                # Sort
                if "sort_by" in prep_config:
                    prep.sort(prep_config["sort_by"])
                    if verbose:
                        print(f"Sorted by {prep_config['sort_by']}")

                prepared_data = prep.to_dataframe()
            else:
                # No preparation, use extracted data directly
                import pandas as pd
                prepared_data = pd.DataFrame(extracted_data)

            results["prepared_records"] = len(prepared_data)

        # Phase 3: Plot Generation
        # Support both old-style (plots + output) and new-style (export + plot)
        if "export" in self.config:
            # New-style: JSON-based export and plotting
            self._execute_json_workflow(prepared_data, results, verbose)
        else:
            # Old-style: Direct plotting with plot_framework
            self._execute_legacy_workflow(prepared_data, results, verbose)

        if verbose:
            print("\n" + "=" * 80)
            print("Execution Summary")
            print("=" * 80)
            print(f"Extracted records: {results['extracted_records']}")
            print(f"Prepared records: {results['prepared_records']}")
            print(f"Generated plots: {len(results['generated_plots'])}")

        return results

    def _execute_legacy_workflow(
        self,
        prepared_data,
        results: dict[str, Any],
        verbose: bool
    ) -> None:
        """Execute legacy plotting workflow using plot_framework."""
        if verbose:
            print("\n" + "=" * 80)
            print("Phase 3: Plot Generation (Legacy Framework)")
            print("=" * 80)

        output_config = self.config["output"]
        output_dir = Path(output_config["directory"])
        backend = output_config.get("backend", "seaborn")

        generator = PlotGenerator(prepared_data)

        for plot_config in self.config["plots"]:
            plot_name = plot_config.get("name", "plot")

            if verbose:
                print(f"\nGenerating plot: {plot_name}")

            # Build PlotSpec
            spec = PlotSpec(
                x_axis=plot_config["x_axis"],
                y_axis=plot_config["y_axis"],
                series_by=plot_config.get("series_by", []),
                fig_by=plot_config.get("fig_by", []),
                row_by=plot_config.get("row_by", []),
                col_by=plot_config.get("col_by", []),
                title=plot_config.get("title"),
                x_label=plot_config.get("x_label"),
                y_label=plot_config.get("y_label"),
                x_scale=plot_config.get("x_scale", "linear"),
                y_scale=plot_config.get("y_scale", "linear"),
                marker_style=plot_config.get("marker_style"),
                line_style=plot_config.get("line_style"),
                color_palette=plot_config.get("color_palette", "tab10"),
                show_error_bars=plot_config.get("show_error_bars", True),
                show_legend=plot_config.get("show_legend", True),
                legend_location=plot_config.get("legend_location", "best"),
                show_grid=plot_config.get("show_grid", True),
                fig_width=plot_config.get("fig_width", 10.0),
                fig_height=plot_config.get("fig_height", 6.0),
                fig_dpi=output_config.get("dpi", 300),
                show_value_labels=plot_config.get("show_value_labels", False),
                value_label_format=plot_config.get("value_label_format"),
                value_label_fontsize=plot_config.get("value_label_fontsize", 8),
                value_label_offset=tuple(plot_config.get("value_label_offset", [0, 5])),
                output_name=plot_name,
            )

            output_files = generator.plot(spec, output_dir, backend)
            results["generated_plots"].extend(output_files)

            if verbose:
                for f in output_files:
                    print(f"  Created: {f}")

    def _execute_multi_source_extraction(
        self,
        verbose: bool
    ) -> tuple[Any, dict[str, int]]:
        """
        Execute extraction and preparation for multiple sources.

        Returns:
            Tuple of (prepared_dataframe, stats_dict)
        """
        import pandas as pd

        all_dataframes = []
        total_extracted = 0
        stats = {}

        for source_config in self.config["sources"]:
            source_name = source_config["name"]

            if verbose:
                print(f"\n  Source: {source_name}")

            # Extract data
            extraction_config = source_config["extraction"]
            extractor = DataExtractor(extraction_config["input"])

            extracted_data = extractor.extract(
                field_specs=extraction_config["fields"],
                run_strategy=extraction_config.get("run_strategy", "average"),
                filter_failed=extraction_config.get("filter_failed", True),
            )

            if verbose:
                print(f"    Extracted {len(extracted_data)} records")

            total_extracted += len(extracted_data)

            # Convert to DataFrame
            df = pd.DataFrame(extracted_data)

            # Add source_name column automatically (check for conflicts)
            source_col_name = "source_name"
            if source_col_name in df.columns:
                # Conflict: use alternative name
                source_col_name = "_source_name"
                if verbose:
                    print(f"    Warning: Column 'source_name' already exists, using '{source_col_name}' instead")

            df[source_col_name] = source_name

            # Apply per-source preparation if configured
            if "preparation" in source_config:
                prep_config = source_config["preparation"]
                prep = PlotDataPrep(df)

                # Apply filters
                if "filters" in prep_config:
                    prep.filter(prep_config["filters"])
                    if verbose:
                        print(f"    After filtering: {len(prep.df)} records")

                # Apply binning
                if "binning" in prep_config:
                    prep.bin(prep_config["binning"])

                # Apply transformations
                if "transformations" in prep_config:
                    prep.transform(prep_config["transformations"])
                    if verbose:
                        print(f"    Applied {len(prep_config['transformations'])} transformations")

                # Apply aggregation
                if "aggregate_by" in prep_config:
                    prep.aggregate(
                        group_by=prep_config["aggregate_by"],
                        agg_funcs=prep_config.get("aggregate_funcs", {}),
                    )
                    if verbose:
                        print(f"    Aggregated by {prep_config['aggregate_by']}")

                # Rename columns
                if "rename" in prep_config:
                    prep.rename(prep_config["rename"])

                # Drop nulls
                if prep_config.get("drop_nulls", False):
                    before = len(prep.df)
                    prep.drop_nulls()
                    if verbose:
                        print(f"    Dropped {before - len(prep.df)} rows with null values")

                # Sort
                if "sort_by" in prep_config:
                    prep.sort(prep_config["sort_by"])

                df = prep.to_dataframe()

            stats[source_name] = len(df)
            all_dataframes.append(df)

        # Merge or concatenate sources based on strategy
        merge_strategy = self.config.get("merge_strategy", "concat")

        if verbose:
            print(f"\n  Combining {len(all_dataframes)} sources (strategy={merge_strategy})...")

        if merge_strategy == "join":
            # Join strategy - merge on common keys
            merge_on = self.config["merge_on"]
            merge_how = self.config.get("merge_how", "inner")
            source_names = [s["name"] for s in self.config["sources"]]

            combined_df = self._merge_sources(
                all_dataframes,
                source_names,
                merge_on,
                merge_how,
                verbose
            )
        else:
            # Concat strategy (default) - stack vertically
            combined_df = pd.concat(all_dataframes, ignore_index=True)

        if verbose:
            print(f"  Combined dataset: {len(combined_df)} records")
            print(f"  Columns: {list(combined_df.columns)}")

        return combined_df, {"total_extracted": total_extracted, "per_source": stats}

    def _merge_sources(
        self,
        all_dataframes: list,
        source_names: list[str],
        merge_on: list[str],
        merge_how: str,
        verbose: bool
    ) -> Any:
        """
        Merge multiple source DataFrames using join operations.

        This enables cross-source calculations like speedup by merging
        data from different sources on common keys (e.g., isl, tp).

        Args:
            all_dataframes: List of DataFrames from each source
            source_names: List of source names
            merge_on: List of column names to merge on
            merge_how: Merge type ('inner', 'outer', 'left', 'right')
            verbose: Print progress messages

        Returns:
            Merged DataFrame with renamed columns
        """
        import pandas as pd

        if len(all_dataframes) == 0:
            raise ValueError("No dataframes to merge")

        if len(all_dataframes) == 1:
            # Only one source - just return it
            return all_dataframes[0]

        # Validate merge_on columns exist in all DataFrames
        for idx, (df, source_name) in enumerate(zip(all_dataframes, source_names)):
            missing_cols = set(merge_on) - set(df.columns)
            if missing_cols:
                raise ValueError(
                    f"Source '{source_name}': merge_on columns {missing_cols} "
                    f"not found in DataFrame. Available columns: {list(df.columns)}"
                )

        # Rename columns to avoid conflicts (except merge_on columns)
        # Strategy: append _<source_name> suffix to non-key columns
        renamed_dfs = []
        for df, source_name in zip(all_dataframes, source_names):
            rename_map = {}
            for col in df.columns:
                if col not in merge_on and col != "source_name":
                    rename_map[col] = f"{col}_{source_name}"

            renamed_df = df.rename(columns=rename_map)
            renamed_dfs.append(renamed_df)

            if verbose and rename_map:
                print(f"  Renamed columns for '{source_name}': {rename_map}")

        # Perform sequential merges
        result = renamed_dfs[0]
        for idx, (df, source_name) in enumerate(zip(renamed_dfs[1:], source_names[1:]), 1):
            if verbose:
                print(f"  Merging '{source_names[0]}' with '{source_name}' on {merge_on} (how={merge_how})")
                print(f"    Before merge: {len(result)} rows")

            result = pd.merge(
                result,
                df,
                on=merge_on,
                how=merge_how,
                suffixes=(f"_{source_names[0]}", f"_{source_name}")
            )

            if verbose:
                print(f"    After merge: {len(result)} rows")

        return result

    def _apply_merge_preparation(
        self,
        df: Any,
        verbose: bool
    ) -> Any:
        """
        Apply global preparation steps after merging sources.

        Args:
            df: Combined DataFrame from all sources
            verbose: Print progress messages

        Returns:
            Prepared DataFrame
        """
        if "merge_preparation" not in self.config:
            return df

        prep_config = self.config["merge_preparation"]
        prep = PlotDataPrep(df)

        # Apply filters
        if "filters" in prep_config:
            before = len(prep.df)
            prep.filter(prep_config["filters"])
            if verbose:
                print(f"  After filtering: {len(prep.df)} records (removed {before - len(prep.df)})")

        # Apply binning
        if "binning" in prep_config:
            prep.bin(prep_config["binning"])
            if verbose:
                print(f"  Applied binning")

        # Apply transformations
        if "transformations" in prep_config:
            prep.transform(prep_config["transformations"])
            if verbose:
                print(f"  Applied {len(prep_config['transformations'])} transformations")

        # Apply aggregation
        if "aggregate_by" in prep_config:
            prep.aggregate(
                group_by=prep_config["aggregate_by"],
                agg_funcs=prep_config.get("aggregate_funcs", {}),
            )
            if verbose:
                print(f"  Aggregated by {prep_config['aggregate_by']}")

        # Rename columns
        if "rename" in prep_config:
            prep.rename(prep_config["rename"])
            if verbose:
                print(f"  Renamed {len(prep_config['rename'])} columns")

        # Drop nulls
        if prep_config.get("drop_nulls", False):
            before = len(prep.df)
            prep.drop_nulls()
            if verbose:
                print(f"  Dropped {before - len(prep.df)} rows with null values")

        # Sort
        if "sort_by" in prep_config:
            prep.sort(prep_config["sort_by"])
            if verbose:
                print(f"  Sorted by {prep_config['sort_by']}")

        return prep.to_dataframe()

    def _execute_json_workflow(
        self,
        prepared_data,
        results: dict[str, Any],
        verbose: bool
    ) -> None:
        """Execute new JSON-based export and plotting workflow."""
        import pandas as pd

        if verbose:
            print("\n" + "=" * 80)
            print("Phase 3: JSON Export")
            print("=" * 80)

        export_config = self.config["export"]

        # Save prepared data to temporary CSV for exporter
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_csv = f.name
            prepared_data.to_csv(temp_csv, index=False)

        try:
            # Initialize exporter
            exporter = PlotJSONExporter(temp_csv)

            # Parse series configuration
            series_col = export_config.get("series_by")
            if series_col and isinstance(series_col, list):
                if len(series_col) == 1:
                    series_col = series_col[0]

            # Parse marker/color/linestyle configuration
            marker_by = export_config.get("marker_by")
            if marker_by and isinstance(marker_by, list):
                if len(marker_by) == 1:
                    marker_by = marker_by[0]

            color_by = export_config.get("color_by")
            if color_by and isinstance(color_by, list):
                if len(color_by) == 1:
                    color_by = color_by[0]

            linestyle_by = export_config.get("linestyle_by")
            if linestyle_by and isinstance(linestyle_by, list):
                if len(linestyle_by) == 1:
                    linestyle_by = linestyle_by[0]

            # Build export configuration
            export_kwargs = {
                "x_col": export_config["x"],
                "y_col": export_config["y"],
                "series_col": series_col,
                "y_err_col": export_config.get("y_err"),
                "series_labels": export_config.get("series_labels"),
                "series_label_pattern": export_config.get("series_label_pattern"),
                "series_label_format": export_config.get("series_label_format"),
                "series_markers": export_config.get("series_markers"),
                "series_colors": export_config.get("series_colors"),
                "series_linestyles": export_config.get("series_linestyles"),
                "marker_by": marker_by,
                "color_by": color_by,
                "linestyle_by": linestyle_by,
            }

            # Add metadata (only non-None values)
            metadata_config = export_config.get("metadata", {})
            for key in ["title", "x_label", "y_label", "description"]:
                if key in metadata_config and metadata_config[key] is not None:
                    export_kwargs[key] = metadata_config[key]

            # Add axes configuration (only non-None values)
            axes_config = export_config.get("axes", {})
            for key in ["x_scale", "y_scale", "x_lim", "y_lim", "x_ticks", "y_ticks"]:
                if key in axes_config and axes_config[key] is not None:
                    export_kwargs[key] = axes_config[key]

            # Add style configuration (only non-None values)
            style_config = export_config.get("style", {})
            for key in ["grid", "minor_grid", "legend_location", "legend_outside", "markersize", "linewidth", "show_value_labels", "value_label_format", "value_label_fontsize", "value_label_offset"]:
                if key in style_config and style_config[key] is not None:
                    export_kwargs[key] = style_config[key]

            # Add figure configuration (only non-None values)
            figure_config = export_config.get("figure", {})
            for key in ["width", "height", "dpi"]:
                if key in figure_config and figure_config[key] is not None:
                    export_kwargs[key] = figure_config[key]

            # Export to JSON
            plot_json = exporter.export(**export_kwargs)

            # Get output path
            output_path = Path(export_config["output"])
            output_path.parent.mkdir(parents=True, exist_ok=True)

            exporter.save(plot_json, output_path)

            if verbose:
                print(f"✓ Exported plot JSON: {output_path}")
                print(f"  Series: {len(plot_json['data'])}")

            # Phase 4: Plot Generation (if configured)
            if "plot" in self.config:
                if verbose:
                    print("\n" + "=" * 80)
                    print("Phase 4: Plot Generation")
                    print("=" * 80)

                plot_config = self.config["plot"]

                # Get input (default to export output if not specified)
                plot_input = Path(plot_config.get("input", export_config["output"]))
                plot_output = Path(plot_config["output"])

                # Create plotter
                plotter = JSONPlotter(plot_input)

                # Build overrides (only non-None values)
                overrides = {}
                for key in ["title", "x_label", "y_label", "x_scale", "y_scale",
                           "x_lim", "y_lim", "grid", "minor_grid", "legend_location",
                           "legend_outside", "markersize", "linewidth",
                           "width", "height", "dpi"]:
                    if key in plot_config and plot_config[key] is not None:
                        overrides[key] = plot_config[key]

                # Generate plot
                output_file = plotter.plot(plot_output, overrides)
                results["generated_plots"].append(str(output_file))

                if verbose:
                    print(f"✓ Generated plot: {output_file}")

        finally:
            # Clean up temp file
            import os
            if os.path.exists(temp_csv):
                os.unlink(temp_csv)


def main():
    """Command-line interface for config-driven plotting."""
    parser = argparse.ArgumentParser(
        description="Execute plotting workflow from configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Execute a YAML configuration
  python -m vllm.benchmarks.sweep.plot_config plot_config.yaml

  # Execute a JSON configuration
  python -m vllm.benchmarks.sweep.plot_config plot_config.json

  # Quiet mode
  python -m vllm.benchmarks.sweep.plot_config plot_config.yaml --quiet
        """
    )

    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration file (.yaml, .yml, or .json)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    # Load and execute configuration
    config = PlotConfig.from_file(args.config)
    results = config.execute(verbose=not args.quiet)

    if not args.quiet:
        print(f"\n✓ Workflow completed successfully!")


if __name__ == "__main__":
    main()

