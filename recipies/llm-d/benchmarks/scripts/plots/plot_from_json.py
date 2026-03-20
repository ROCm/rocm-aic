#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Plot from self-contained JSON format.

This script reads a JSON file containing both data and plot configuration,
then generates the plot. Command-line arguments override JSON settings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

class JSONPlotter:
    """Create plots from self-contained JSON format."""

    def __init__(self, json_path: str | Path):
        """
        Initialize plotter with JSON data file.

        Args:
            json_path: Path to plot data JSON file
        """
        self.json_path = Path(json_path)
        self.config = self._load_json()
        self._validate()

    def _load_json(self) -> dict:
        """Load and parse JSON file."""
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def _validate(self) -> None:
        """Validate JSON structure."""
        if "data" not in self.config:
            raise ValueError("JSON must contain 'data' field with series array")

        if not isinstance(self.config["data"], list):
            raise ValueError("'data' must be an array of series")

        if len(self.config["data"]) == 0:
            raise ValueError("'data' array must contain at least one series")

        for i, series in enumerate(self.config["data"]):
            if "name" not in series:
                raise ValueError(f"Series {i} missing 'name' field")
            if "x" not in series:
                raise ValueError(f"Series {i} ({series['name']}) missing 'x' field")
            if "y" not in series:
                raise ValueError(f"Series {i} ({series['name']}) missing 'y' field")

            if len(series["x"]) != len(series["y"]):
                raise ValueError(
                    f"Series {i} ({series['name']}): x and y arrays must have same length"
                )

    def plot(
        self,
        output_path: str | Path,
        overrides: dict[str, Any] | None = None,
    ) -> Path:
        """
        Generate plot from JSON data.

        Args:
            output_path: Where to save the plot
            overrides: Dictionary of values to override from JSON
                       Keys: title, x_label, y_label, x_scale, y_scale, etc.

        Returns:
            Path to generated plot file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get configuration with overrides
        metadata = self.config.get("metadata", {})
        axes_config = self.config.get("axes", {})
        style_config = self.config.get("style", {})
        figure_config = self.config.get("figure", {})

        # Apply overrides
        overrides = overrides or {}

        title = overrides.get("title", metadata.get("title"))
        x_label = overrides.get("x_label", metadata.get("x_label"))
        y_label = overrides.get("y_label", metadata.get("y_label"))

        x_scale = overrides.get("x_scale", axes_config.get("x_scale", "linear"))
        y_scale = overrides.get("y_scale", axes_config.get("y_scale", "linear"))
        x_lim = overrides.get("x_lim", axes_config.get("x_lim"))
        y_lim = overrides.get("y_lim", axes_config.get("y_lim"))
        x_ticks = overrides.get("x_ticks", axes_config.get("x_ticks"))
        y_ticks = overrides.get("y_ticks", axes_config.get("y_ticks"))
        x_tick_labels = overrides.get("x_tick_labels", axes_config.get("x_tick_labels"))
        y_tick_labels = overrides.get("y_tick_labels", axes_config.get("y_tick_labels"))

        show_grid = overrides.get("grid", style_config.get("grid", True))
        minor_grid = overrides.get("minor_grid", style_config.get("minor_grid", False))
        show_legend = overrides.get("legend", style_config.get("legend", True))
        legend_location = overrides.get("legend_location", style_config.get("legend_location", "upper right"))
        legend_outside = overrides.get("legend_outside", style_config.get("legend_outside", False))
        show_error_bars = overrides.get("error_bars", style_config.get("error_bars", True))
        markersize = overrides.get("markersize", style_config.get("markersize", 8))
        linewidth = overrides.get("linewidth", style_config.get("linewidth", 2))
        show_value_labels = overrides.get("show_value_labels", style_config.get("show_value_labels", False))
        value_label_format = overrides.get("value_label_format", style_config.get("value_label_format"))
        value_label_fontsize = overrides.get("value_label_fontsize", style_config.get("value_label_fontsize", 8))
        value_label_offset = overrides.get("value_label_offset", style_config.get("value_label_offset", (0, 5)))

        fig_width = overrides.get("width", figure_config.get("width", 10))
        fig_height = overrides.get("height", figure_config.get("height", 6))
        fig_dpi = overrides.get("dpi", figure_config.get("dpi", 300))

        # Create figure
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Plot each series
        for series in self.config["data"]:
            x_data = series["x"]
            y_data = series["y"]
            y_err_data = series.get("y_err")
            name = series["name"]

            # Get series-specific styling
            marker = series.get("marker", "o")
            linestyle = series.get("linestyle", "-")
            color = series.get("color")

            # Plot the series
            if show_error_bars and y_err_data:
                ax.errorbar(
                    x_data,
                    y_data,
                    yerr=y_err_data,
                    label=name,
                    marker=marker,
                    linestyle=linestyle,
                    color=color,
                    markersize=markersize,
                    linewidth=linewidth,
                    markeredgewidth=0.5,
                    markeredgecolor='white',
                    capsize=4,
                )
            else:
                ax.plot(
                    x_data,
                    y_data,
                    label=name,
                    marker=marker,
                    linestyle=linestyle,
                    color=color,
                    markersize=markersize,
                    linewidth=linewidth,
                    markeredgewidth=0.5,
                    markeredgecolor='white',
                )

        # Set labels and title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        if x_label:
            ax.set_xlabel(x_label, fontsize=12)
        if y_label:
            ax.set_ylabel(y_label, fontsize=12)

        # Set scales
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)

        # Set limits
        if x_lim:
            ax.set_xlim(x_lim)
        if y_lim:
            ax.set_ylim(y_lim)

        # Set ticks
        if x_ticks:
            ax.set_xticks(x_ticks)
        if y_ticks:
            ax.set_yticks(y_ticks)

        # Set tick labels
        if x_tick_labels:
            ax.set_xticklabels(x_tick_labels)
        if y_tick_labels:
            ax.set_yticklabels(y_tick_labels)

        # Grid
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7, which='major')

            # Add minor grid for log scales (shows lines at each tick mark)
            if minor_grid:
                ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
                # Enable minor ticks for log scales
                if x_scale == 'log':
                    ax.xaxis.set_minor_locator(plt.LogLocator(subs='all'))
                if y_scale == 'log':
                    ax.yaxis.set_minor_locator(plt.LogLocator(subs='all'))

        # Bounding box (all 4 spines)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_edgecolor('black')

        # Legend
        if show_legend and len(self.config["data"]) > 1:
            if legend_outside:
                # Place legend outside plot area on the right
                legend = ax.legend(
                    loc='center left',
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=True,
                    shadow=True,
                    fancybox=False,
                    fontsize=10,
                )
            else:
                # Place legend inside plot area
                legend = ax.legend(
                    loc=legend_location,
                    frameon=True,
                    shadow=True,
                    fancybox=False,
                    fontsize=10,
                )
            legend.get_frame().set_linewidth(1.0)
            legend.get_frame().set_edgecolor('black')

        # Add value labels to data points
        if show_value_labels:
            self._add_value_labels(ax, value_label_format, value_label_fontsize, value_label_offset)

        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"✓ Generated plot: {output_path}")
        return output_path

    def _add_value_labels(
        self,
        ax: "plt.Axes",
        value_label_format: str | None,
        value_label_fontsize: int,
        value_label_offset: tuple[float, float],
    ) -> None:
        """Add value labels to data points."""
        import numpy as np
        import pandas as pd

        # Get the lines plotted on this axis
        lines = ax.get_lines()

        for line in lines:
            # Get the data from the line
            x_data = line.get_xdata()
            y_data = line.get_ydata()

            # Add annotation for each point
            for x_val, y_val in zip(x_data, y_data):
                # Skip if either value is NaN or infinite
                if pd.isna(x_val) or pd.isna(y_val):
                    continue
                if not (np.isfinite(x_val) and np.isfinite(y_val)):
                    continue

                # Format the label
                if value_label_format:
                    try:
                        label_text = value_label_format.format(x=x_val, y=y_val)
                    except (KeyError, ValueError):
                        # Fall back to default
                        label_text = f"{y_val:.1f}"
                else:
                    # Default: show y value with 1 decimal place
                    label_text = f"{y_val:.1f}"

                # Add annotation
                ax.annotate(
                    label_text,
                    xy=(x_val, y_val),
                    xytext=value_label_offset,
                    textcoords='offset points',
                    fontsize=value_label_fontsize,
                    ha='center',
                    va='bottom',
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor='white',
                        edgecolor='gray',
                        alpha=0.7,
                        linewidth=0.5
                    )
                )


def create_json_from_dataframe(
    df,
    x_col: str,
    y_col: str,
    series_col: str | None = None,
    output_path: str | Path | None = None,
    **metadata
) -> dict:
    """
    Create plot JSON from pandas DataFrame.

    Args:
        df: DataFrame with plot data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        series_col: Column to group by (creates multiple series)
        output_path: Where to save JSON (optional)
        **metadata: Additional metadata (title, x_label, y_label, etc.)

    Returns:
        Dictionary in plot JSON format
    """
    plot_data = {"data": []}

    if series_col and series_col in df.columns:
        # Multiple series
        for series_name, group_df in df.groupby(series_col):
            series_data = {
                "name": str(series_name),
                "x": group_df[x_col].tolist(),
                "y": group_df[y_col].tolist(),
            }
            plot_data["data"].append(series_data)
    else:
        # Single series
        series_data = {
            "name": metadata.pop("series_name", y_col),
            "x": df[x_col].tolist(),
            "y": df[y_col].tolist(),
        }
        plot_data["data"].append(series_data)

    # Add metadata
    if metadata:
        plot_data["metadata"] = {}
        for key in ["title", "x_label", "y_label", "description"]:
            if key in metadata:
                plot_data["metadata"][key] = metadata[key]

    # Add axes config
    axes_keys = ["x_scale", "y_scale", "x_lim", "y_lim", "x_ticks", "y_ticks"]
    axes_config = {k: v for k, v in metadata.items() if k in axes_keys}
    if axes_config:
        plot_data["axes"] = axes_config

    # Add style config
    style_keys = ["grid", "minor_grid", "legend", "legend_location", "error_bars", "markersize", "linewidth"]
    style_config = {k: v for k, v in metadata.items() if k in style_keys}
    if style_config:
        plot_data["style"] = style_config

    # Add figure config
    figure_keys = ["width", "height", "dpi"]
    figure_config = {k: v for k, v in metadata.items() if k in figure_keys}
    if figure_config:
        plot_data["figure"] = figure_config

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(plot_data, f, indent=2)
        print(f"✓ Saved plot JSON to {output_path}")

    return plot_data


def main():
    """Command-line interface for plotting from JSON."""
    parser = argparse.ArgumentParser(
        description="Generate plot from self-contained JSON data file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m plots.plot_from_json plot_data.json --output plot.png

  # Override title and scale
  python -m plots.plot_from_json plot_data.json \\
      --output plot.png \\
      --title "Custom Title" \\
      --x-scale log

  # Override multiple settings
  python -m plots.plot_from_json plot_data.json \\
      --output plot.png \\
      --title "Performance Analysis" \\
      --x-label "Sequence Length" \\
      --y-label "Latency (ms)" \\
      --x-scale log \\
      --no-grid \\
      --dpi 150

JSON Format:
  See plot_data_schema.json for complete schema.
  See plot_data_example.json for a working example.
        """
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to plot data JSON file"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output plot file path (.png, .pdf, .svg)"
    )

    # Metadata overrides
    parser.add_argument("--title", type=str, help="Override plot title")
    parser.add_argument("--x-label", type=str, help="Override x-axis label")
    parser.add_argument("--y-label", type=str, help="Override y-axis label")

    # Axis overrides
    parser.add_argument("--x-scale", type=str, choices=["linear", "log", "symlog", "logit"],
                       help="Override x-axis scale")
    parser.add_argument("--y-scale", type=str, choices=["linear", "log", "symlog", "logit"],
                       help="Override y-axis scale")
    parser.add_argument("--x-lim", type=float, nargs=2, metavar=("MIN", "MAX"),
                       help="Override x-axis limits")
    parser.add_argument("--y-lim", type=float, nargs=2, metavar=("MIN", "MAX"),
                       help="Override y-axis limits")

    # Style overrides
    parser.add_argument("--no-grid", action="store_true", help="Disable grid")
    parser.add_argument("--minor-grid", action="store_true",
                       help="Enable minor grid lines (shows lines at each tick mark, useful for log scales)")
    parser.add_argument("--no-legend", action="store_true", help="Disable legend")
    parser.add_argument("--legend-location", type=str, help="Override legend location")
    parser.add_argument("--legend-outside", action="store_true", help="Place legend outside plot area (right side)")
    parser.add_argument("--no-error-bars", action="store_true", help="Disable error bars")
    parser.add_argument("--markersize", type=float, help="Override marker size")
    parser.add_argument("--linewidth", type=float, help="Override line width")
    parser.add_argument("--show-value-labels", action="store_true", help="Show value labels on data points")
    parser.add_argument("--value-label-format", type=str, help="Format string for value labels (e.g., '{y:.1f}' or '{x}, {y:.2f}')")
    parser.add_argument("--value-label-fontsize", type=int, help="Font size for value labels")
    parser.add_argument("--value-label-offset", type=float, nargs=2, metavar=("X", "Y"), help="Label offset in points (x, y)")

    # Figure overrides
    parser.add_argument("--width", type=float, help="Override figure width (inches)")
    parser.add_argument("--height", type=float, help="Override figure height (inches)")
    parser.add_argument("--dpi", type=int, help="Override DPI")

    args = parser.parse_args()

    # Build overrides dict
    overrides = {}

    if args.title:
        overrides["title"] = args.title
    if args.x_label:
        overrides["x_label"] = args.x_label
    if args.y_label:
        overrides["y_label"] = args.y_label

    if args.x_scale:
        overrides["x_scale"] = args.x_scale
    if args.y_scale:
        overrides["y_scale"] = args.y_scale
    if args.x_lim:
        overrides["x_lim"] = args.x_lim
    if args.y_lim:
        overrides["y_lim"] = args.y_lim

    if args.no_grid:
        overrides["grid"] = False
    if args.minor_grid:
        overrides["minor_grid"] = True
    if args.no_legend:
        overrides["legend"] = False
    if args.legend_location:
        overrides["legend_location"] = args.legend_location
    if args.legend_outside:
        overrides["legend_outside"] = True
    if args.no_error_bars:
        overrides["error_bars"] = False
    if args.markersize:
        overrides["markersize"] = args.markersize
    if args.linewidth:
        overrides["linewidth"] = args.linewidth
    if args.show_value_labels:
        overrides["show_value_labels"] = True
    if args.value_label_format:
        overrides["value_label_format"] = args.value_label_format
    if args.value_label_fontsize:
        overrides["value_label_fontsize"] = args.value_label_fontsize
    if args.value_label_offset:
        overrides["value_label_offset"] = tuple(args.value_label_offset)

    if args.width:
        overrides["width"] = args.width
    if args.height:
        overrides["height"] = args.height
    if args.dpi:
        overrides["dpi"] = args.dpi

    # Create plot
    plotter = JSONPlotter(args.input)
    output_file = plotter.plot(args.output, overrides)

    print(f"\n✓ Plot generation complete!")
    print(f"  Input: {args.input}")
    print(f"  Output: {output_file}")

    if overrides:
        print(f"  Overrides applied: {', '.join(overrides.keys())}")


if __name__ == "__main__":
    main()
