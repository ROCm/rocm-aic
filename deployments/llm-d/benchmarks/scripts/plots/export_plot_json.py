#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Export prepared data to self-contained plot JSON format.

This tool converts CSV/DataFrame data into the plot JSON format that includes
both data and plot configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

class PlotJSONExporter:
    """Export data to self-contained plot JSON format."""

    def __init__(self, data_path: str | Path):
        """
        Initialize exporter with data file.

        Args:
            data_path: Path to CSV, JSON, or Parquet file
        """
        self.data_path = Path(data_path)
        self.df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Load data from file."""
        suffix = self.data_path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(self.data_path)
        elif suffix == ".json":
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        elif suffix == ".parquet":
            return pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def export(
        self,
        x_col: str,
        y_col: str,
        series_col: str | list[str] | None = None,
        y_err_col: str | None = None,
        series_labels: dict[str, str] | None = None,
        series_label_pattern: str | None = None,
        series_label_format: str | None = None,
        series_markers: dict[str, str] | None = None,
        series_colors: dict[str, str] | None = None,
        series_linestyles: dict[str, str] | None = None,
        marker_by: str | list[str] | None = None,
        color_by: str | list[str] | None = None,
        linestyle_by: str | list[str] | None = None,
        **config
    ) -> dict:
        """
        Export data to plot JSON format.

        Args:
            x_col: Column for x-axis values
            y_col: Column for y-axis values
            series_col: Column(s) to group by - string or list of strings
            y_err_col: Column for error bars (optional)
            series_labels: Custom names for series values (supports {} placeholder)
            series_label_pattern: Pattern for single column (e.g., "TP={}")
            series_label_format: Format for multiple columns (e.g., "{run_label} - TP={tp}")
            series_markers: Marker styles per series
            series_colors: Colors per series
            series_linestyles: Line styles per series
            marker_by: Column(s) to use for marker lookup (defaults to series_col)
            color_by: Column(s) to use for color lookup (defaults to series_col)
            linestyle_by: Column(s) to use for linestyle lookup (defaults to series_col)
            **config: Additional plot configuration (title, x_scale, etc.)

        Returns:
            Dictionary in plot JSON format
        """
        # Validate columns
        required_cols = [x_col, y_col]
        if series_col:
            if isinstance(series_col, list):
                required_cols.extend(series_col)
            else:
                required_cols.append(series_col)
        if y_err_col:
            required_cols.append(y_err_col)

        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Columns not found in data: {missing}\n"
                f"Available: {list(self.df.columns)}"
            )

        plot_json = {"data": []}

        # Helper function to extract lookup key from series value
        def get_lookup_key(series_value, group_cols, lookup_cols):
            """Extract lookup key based on specified columns."""
            if not lookup_cols:
                # Default: use all group columns
                lookup_cols = group_cols
            else:
                # Normalize to list
                lookup_cols = lookup_cols if isinstance(lookup_cols, list) else [lookup_cols]

            # Build values dict for all group columns
            if isinstance(series_value, tuple):
                values_dict = {col: val for col, val in zip(group_cols, series_value)}
            else:
                values_dict = {group_cols[0]: series_value}

            # Extract values for lookup columns
            lookup_values = []
            for col in lookup_cols:
                if col not in values_dict:
                    raise ValueError(f"Lookup column '{col}' not in series columns {group_cols}")
                val = values_dict[col]
                # Convert float to int if appropriate
                if isinstance(val, float) and val.is_integer():
                    clean_val = str(int(val))
                else:
                    clean_val = str(val)
                lookup_values.append(clean_val)

            # Return single value or joined values
            if len(lookup_values) == 1:
                return lookup_values[0]
            else:
                return "_".join(lookup_values)

        # Generate series data
        if series_col:
            # Handle both single column and list of columns
            group_cols = series_col if isinstance(series_col, list) else [series_col]

            # Multiple series
            for series_value, group_df in self.df.groupby(group_cols):
                # Handle single value or tuple of values
                if isinstance(series_value, tuple):
                    # Multiple grouping columns
                    values_dict = {}
                    orig_values = []
                    for col, val in zip(group_cols, series_value):
                        # Convert float to int if appropriate
                        if isinstance(val, float) and val.is_integer():
                            clean_val = str(int(val))
                        else:
                            clean_val = str(val)
                        values_dict[col] = clean_val
                        orig_values.append(clean_val)

                    # Create default name
                    orig_value = "_".join(orig_values)
                    series_name = " - ".join(f"{col}={val}" for col, val in zip(group_cols, orig_values))

                    # Apply custom formatting
                    if series_label_format:
                        try:
                            series_name = series_label_format.format(**values_dict)
                        except KeyError:
                            pass  # Keep default if format has missing keys
                    elif series_labels and orig_value in series_labels:
                        series_name = series_labels[orig_value]

                else:
                    # Single grouping column
                    if isinstance(series_value, float) and series_value.is_integer():
                        orig_value = str(int(series_value))
                    else:
                        orig_value = str(series_value)

                    series_name = orig_value

                    # Apply custom label if provided
                    if series_labels:
                        if orig_value in series_labels:
                            label_template = series_labels[orig_value]
                            series_name = label_template.replace("{}", orig_value)
                        elif f"{orig_value}.0" in series_labels:
                            label_template = series_labels[f"{orig_value}.0"]
                            series_name = label_template.replace("{}", orig_value)
                    elif series_label_pattern:
                        series_name = series_label_pattern.replace("{}", orig_value)

                # Sort by x for proper line plotting
                group_df = group_df.sort_values(by=x_col)

                series_data = {
                    "name": series_name,
                    "x": group_df[x_col].tolist(),
                    "y": group_df[y_col].tolist(),
                }

                # Add error bars if available
                if y_err_col:
                    series_data["y_err"] = group_df[y_err_col].tolist()

                # Add styling - use appropriate lookup keys
                if series_markers:
                    marker_key = get_lookup_key(series_value, group_cols, marker_by)
                    if marker_key in series_markers:
                        series_data["marker"] = series_markers[marker_key]

                if series_colors:
                    color_key = get_lookup_key(series_value, group_cols, color_by)
                    if color_key in series_colors:
                        series_data["color"] = series_colors[color_key]

                if series_linestyles:
                    linestyle_key = get_lookup_key(series_value, group_cols, linestyle_by)
                    if linestyle_key in series_linestyles:
                        series_data["linestyle"] = series_linestyles[linestyle_key]

                plot_json["data"].append(series_data)
        else:
            # Single series
            sorted_df = self.df.sort_values(by=x_col)

            series_data = {
                "name": config.pop("series_name", y_col),
                "x": sorted_df[x_col].tolist(),
                "y": sorted_df[y_col].tolist(),
            }

            if y_err_col:
                series_data["y_err"] = sorted_df[y_err_col].tolist()

            plot_json["data"].append(series_data)

        # Add metadata
        metadata_keys = ["title", "x_label", "y_label", "description"]
        metadata = {k: v for k, v in config.items() if k in metadata_keys}
        if metadata:
            plot_json["metadata"] = metadata

        # Add axes config
        axes_keys = ["x_scale", "y_scale", "x_lim", "y_lim", "x_ticks", "y_ticks", "x_tick_labels", "y_tick_labels"]
        axes_config = {k: v for k, v in config.items() if k in axes_keys}
        if axes_config:
            plot_json["axes"] = axes_config

        # Add style config
        style_keys = ["grid", "minor_grid", "legend", "legend_location", "legend_outside", "error_bars", "markersize", "linewidth", "show_value_labels", "value_label_format", "value_label_fontsize", "value_label_offset"]
        style_config = {k: v for k, v in config.items() if k in style_keys}
        if style_config:
            plot_json["style"] = style_config

        # Add figure config
        figure_keys = ["width", "height", "dpi"]
        figure_config = {k: v for k, v in config.items() if k in figure_keys}
        if figure_config:
            plot_json["figure"] = figure_config

        return plot_json

    def save(self, plot_json: dict, output_path: str | Path) -> None:
        """Save plot JSON to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(plot_json, f, indent=2)

        print(f"✓ Saved plot JSON: {output_path}")
        print(f"  Series: {len(plot_json['data'])}")
        print(f"  Total data points: {sum(len(s['x']) for s in plot_json['data'])}")


def main():
    """Command-line interface for exporting to plot JSON."""
    parser = argparse.ArgumentParser(
        description="Export prepared data to self-contained plot JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export
  python -m plots.export_plot_json \\
      prepared_data.csv \\
      --x isl \\
      --y ttft \\
      --series-by tp_size \\
      --output plot_data.json

  # With metadata
  python -m plots.export_plot_json \\
      prepared_data.csv \\
      --x isl \\
      --y ttft \\
      --series-by tp_size \\
      --title "TTFT Performance" \\
      --x-label "Input Length" \\
      --y-label "TTFT (ms)" \\
      --x-scale log \\
      --output plot_data.json

  # With custom series names (using {} placeholder)
  python -m plots.export_plot_json \\
      prepared_data.csv \\
      --x isl \\
      --y ttft \\
      --series-by tp_size \\
      --series-labels "1=GPU prefill (tp={})" "2=GPU prefill (tp={})" "4=GPU prefill (tp={})" \\
      --output plot_data.json

  # Or use pattern (simpler when all series have same format)
  python -m plots.export_plot_json \\
      prepared_data.csv \\
      --x isl \\
      --y ttft \\
      --series-by tp_size \\
      --series-label-pattern "TP={}" \\
      --output plot_data.json

  # With custom markers and colors
  python -m plots.export_plot_json \\
      prepared_data.csv \\
      --x isl \\
      --y ttft \\
      --series-by tp_size \\
      --series-label-pattern "GPU prefill (tp={})" \\
      --series-markers "1=o" "2=^" "4=s" \\
      --series-colors "1=#1f77b4" "2=#ff7f0e" "4=#2ca02c" \\
      --output plot_data.json

Then plot it:
  python -m plots.plot_from_json plot_data.json --output plot.png
        """
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to prepared data file (.csv, .json, .parquet)"
    )

    parser.add_argument("--x", dest="x_col", type=str, required=True,
                       help="Column for x-axis")
    parser.add_argument("--y", dest="y_col", type=str, required=True,
                       help="Column for y-axis")
    parser.add_argument("--series-by", type=str, nargs="+",
                       help="Column(s) to group by (creates multiple series). "
                            "Can specify multiple: --series-by run_label tp")
    parser.add_argument("--y-err", type=str,
                       help="Column for error bars")

    # Series customization
    parser.add_argument("--series-labels", type=str, nargs="+",
                       help="Custom series labels (format: 'value=label', supports {} placeholder). "
                            "Example: '1=GPU prefill (tp={})'")
    parser.add_argument("--series-label-pattern", type=str,
                       help="Pattern to apply to all series (e.g., 'TP={}'). "
                            "Simpler than --series-labels when all series use same format.")
    parser.add_argument("--series-label-format", type=str,
                       help="Format for multiple columns (e.g., '{run_label} - TP={tp}'). "
                            "Use column names as placeholders.")
    parser.add_argument("--series-markers", type=str, nargs="+",
                       help="Markers per series (format: 'value=marker')")
    parser.add_argument("--series-colors", type=str, nargs="+",
                       help="Colors per series (format: 'value=color')")
    parser.add_argument("--series-linestyles", type=str, nargs="+",
                       help="Line styles per series (format: 'value=style')")
    parser.add_argument("--marker-by", type=str, nargs="+",
                       help="Column(s) to use for marker lookup (defaults to --series-by). "
                            "Useful when you want all series with same value in this column to share a marker.")
    parser.add_argument("--color-by", type=str, nargs="+",
                       help="Column(s) to use for color lookup (defaults to --series-by)")
    parser.add_argument("--linestyle-by", type=str, nargs="+",
                       help="Column(s) to use for linestyle lookup (defaults to --series-by)")

    # Metadata
    parser.add_argument("--title", type=str, help="Plot title")
    parser.add_argument("--x-label", type=str, help="X-axis label")
    parser.add_argument("--y-label", type=str, help="Y-axis label")
    parser.add_argument("--description", type=str, help="Plot description")

    # Axes config
    parser.add_argument("--x-scale", type=str, choices=["linear", "log", "symlog", "logit"],
                       help="X-axis scale")
    parser.add_argument("--y-scale", type=str, choices=["linear", "log", "symlog", "logit"],
                       help="Y-axis scale")

    # Style
    parser.add_argument("--no-grid", action="store_true", help="Disable grid")
    parser.add_argument("--minor-grid", action="store_true",
                       help="Enable minor grid lines (shows lines at each tick mark, useful for log scales)")
    parser.add_argument("--legend-location", type=str, help="Legend location")
    parser.add_argument("--legend-outside", action="store_true",
                       help="Place legend outside plot area (right side)")

    # Figure
    parser.add_argument("--width", type=float, help="Figure width (inches)")
    parser.add_argument("--height", type=float, help="Figure height (inches)")
    parser.add_argument("--dpi", type=int, help="Figure DPI")

    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file path")

    args = parser.parse_args()

    # Parse mapping arguments
    def parse_mapping(arg_list):
        if not arg_list:
            return None
        mapping = {}
        for item in arg_list:
            if "=" not in item:
                raise ValueError(f"Invalid mapping: {item}. Expected 'key=value'")
            key, value = item.split("=", 1)
            mapping[key] = value
        return mapping

    series_labels = parse_mapping(args.series_labels)
    series_markers = parse_mapping(args.series_markers)
    series_colors = parse_mapping(args.series_colors)
    series_linestyles = parse_mapping(args.series_linestyles)

    # Build config
    config = {}
    if args.title:
        config["title"] = args.title
    if args.x_label:
        config["x_label"] = args.x_label
    if args.y_label:
        config["y_label"] = args.y_label
    if args.description:
        config["description"] = args.description

    if args.x_scale:
        config["x_scale"] = args.x_scale
    if args.y_scale:
        config["y_scale"] = args.y_scale

    if args.no_grid:
        config["grid"] = False
    if args.minor_grid:
        config["minor_grid"] = True
    if args.legend_location:
        config["legend_location"] = args.legend_location
    if args.legend_outside:
        config["legend_outside"] = True

    if args.width:
        config["width"] = args.width
    if args.height:
        config["height"] = args.height
    if args.dpi:
        config["dpi"] = args.dpi

    # Handle series-by as list or single value
    series_by = args.series_by
    if series_by and len(series_by) == 1:
        series_by = series_by[0]  # Single column as string

    # Handle marker-by, color-by, linestyle-by
    marker_by = args.marker_by
    if marker_by and len(marker_by) == 1:
        marker_by = marker_by[0]

    color_by = args.color_by
    if color_by and len(color_by) == 1:
        color_by = color_by[0]

    linestyle_by = args.linestyle_by
    if linestyle_by and len(linestyle_by) == 1:
        linestyle_by = linestyle_by[0]

    # Export
    exporter = PlotJSONExporter(args.input)
    plot_json = exporter.export(
        x_col=args.x_col,
        y_col=args.y_col,
        series_col=series_by,
        y_err_col=args.y_err,
        series_labels=series_labels,
        series_label_pattern=args.series_label_pattern,
        series_label_format=args.series_label_format,
        series_markers=series_markers,
        series_colors=series_colors,
        series_linestyles=series_linestyles,
        marker_by=marker_by,
        color_by=color_by,
        linestyle_by=linestyle_by,
        **config
    )

    exporter.save(plot_json, args.output)

    print(f"\n✓ Export complete!")
    print(f"  Generated plot JSON with {len(plot_json['data'])} series")
    print(f"\nTo plot:")
    print(f"  python -m plots.plot_from_json {args.output} --output plot.png")


if __name__ == "__main__":
    main()
