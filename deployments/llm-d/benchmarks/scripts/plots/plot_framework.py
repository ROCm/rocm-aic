# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

"""
Phase 3: Plot Generation from prepared data.

This module provides a flexible framework for creating plots from prepared
benchmark data with full control over layout, styling, and appearance.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.markers as mmarkers

import seaborn as sns

from .prepare import PlotDataPrep
from .utils import sanitize_filename


@dataclass
class PlotSpec:
    """Specification for a single plot"""

    # Required: Data specification
    x_axis: str
    y_axis: str

    # Optional: Series/grouping
    series_by: list[str] = field(default_factory=list)

    # Optional: Layout specification
    fig_by: list[str] = field(default_factory=list)
    row_by: list[str] = field(default_factory=list)
    col_by: list[str] = field(default_factory=list)

    # Optional: Labels and titles
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None

    # Optional: Scales
    x_scale: str = "linear"  # "linear", "log", "sqrt"
    y_scale: str = "linear"

    # Optional: Markers & styling
    marker_style: dict[str, str] | None = None
    line_style: dict[str, str] | None = None
    color_palette: str | list[str] = "tab10"

    # Optional: Legend customization
    series_labels: dict[str, str] | None = None  # Map values to custom labels
    series_label_format: str | None = None  # Format template for auto-labels

    # Optional: Plot options
    show_error_bars: bool = True
    show_legend: bool = True
    legend_location: str = "upper right"  # For FacetGrid, 'best' is not supported
    legend_outside: bool = False  # Place legend outside plot area (right side)
    show_grid: bool = True
    fig_width: float = 10.0
    fig_height: float = 6.0
    fig_dpi: int = 300

    # Optional: Value labels on data points
    show_value_labels: bool = False
    value_label_format: str | None = None  # e.g., "{y:.1f}" or "{x}, {y:.2f}"
    value_label_fontsize: int = 8
    value_label_offset: tuple[float, float] = (0, 5)  # (x_offset, y_offset) in points

    # Optional: Output
    output_name: str = "plot"

    def validate(self, df: "pd.DataFrame") -> None:
        """Validate that the spec is compatible with the data."""
        all_columns = (
            [self.x_axis, self.y_axis]
            + self.series_by
            + self.fig_by
            + self.row_by
            + self.col_by
        )

        missing = set(all_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"Columns not found in data: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

    def format_series_label(self, group_values: tuple) -> str:
        """
        Format series label based on customization settings.

        Args:
            group_values: Tuple of (column, value) pairs

        Returns:
            Formatted label string
        """
        if not group_values:
            return ""

        # Convert to dict for easier access
        values_dict = dict(group_values)

        # If custom labels mapping provided, use it
        if self.series_labels:
            # Try to match the full combination
            key = ",".join(f"{k}={v}" for k, v in group_values)
            if key in self.series_labels:
                return self.series_labels[key]

            # Try single value match if only one series_by column
            if len(group_values) == 1:
                value_str = str(group_values[0][1])
                if value_str in self.series_labels:
                    return self.series_labels[value_str]

        # If format template provided, use it
        if self.series_label_format:
            try:
                return self.series_label_format.format(**values_dict)
            except KeyError as e:
                # Fall back to default if template has missing keys
                pass

        # Default: "col1=val1, col2=val2"
        return ", ".join(f"{k}={v}" for k, v in group_values)

    def format_value_label(self, x_val: float, y_val: float) -> str:
        """
        Format value label for a data point.

        Args:
            x_val: X-axis value
            y_val: Y-axis value

        Returns:
            Formatted label string
        """
        if self.value_label_format:
            # Use custom format string
            try:
                return self.value_label_format.format(x=x_val, y=y_val)
            except (KeyError, ValueError) as e:
                # Fall back to default if format string is invalid
                pass

        # Default: show y value with 1 decimal place
        return f"{y_val:.1f}"


class PlotGenerator:
    """Generates plots from prepared data"""

    def __init__(self, data: "pd.DataFrame" | PlotDataPrep | str | Path):
        """
        Initialize plot generator.

        Args:
            data: Can be:
                - pandas DataFrame
                - PlotDataPrep instance
                - Path to CSV/JSON file
        """
        if isinstance(data, (str, Path)):
            prep = PlotDataPrep(data)
            self.df = prep.to_dataframe()
        elif isinstance(data, PlotDataPrep):
            self.df = data.to_dataframe()
        else:
            self.df = data

    def plot(
        self,
        spec: PlotSpec,
        output_dir: str | Path,
        backend: str = "seaborn",
    ) -> list[Path]:
        """
        Generate plots according to specification.

        Args:
            spec: PlotSpec defining what to plot
            output_dir: Directory to save plots
            backend: Plotting backend ("seaborn" or "matplotlib")

        Returns:
            List of paths to generated plot files
        """
        spec.validate(self.df)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if backend == "seaborn":
            return self._plot_seaborn(spec, output_dir)
        elif backend == "matplotlib":
            return self._plot_matplotlib(spec, output_dir)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _plot_seaborn(self, spec: PlotSpec, output_dir: Path) -> list[Path]:
        """Generate plots using seaborn."""
        output_files = []

        # If fig_by is specified, create separate plots for each group
        if spec.fig_by:
            fig_groups = self.df.groupby(spec.fig_by)

            for group_values, group_df in fig_groups:
                # Convert single value to tuple for consistency
                if not isinstance(group_values, tuple):
                    group_values = (group_values,)

                # Create output filename
                fig_name_parts = [spec.output_name]
                for col, val in zip(spec.fig_by, group_values):
                    fig_name_parts.append(f"{col}={val}")
                fig_name = sanitize_filename("-".join(fig_name_parts) + ".png")
                output_path = output_dir / fig_name

                self._create_seaborn_plot(spec, group_df, output_path)
                output_files.append(output_path)
        else:
            # Single plot
            output_path = output_dir / f"{spec.output_name}.png"
            self._create_seaborn_plot(spec, self.df, output_path)
            output_files.append(output_path)

        return output_files

    def _create_seaborn_plot(
        self,
        spec: PlotSpec,
        df: "pd.DataFrame",
        output_path: Path,
    ) -> None:
        """Create a single seaborn plot."""
        # Prepare row/col grouping columns
        if spec.row_by:
            df = df.copy()
            df["_row_group"] = df[spec.row_by].apply(
                lambda row: "\n".join(f"{col}={row[col]}" for col in spec.row_by),
                axis=1
            )
            row_var = "_row_group"
        else:
            row_var = None

        if spec.col_by:
            if not spec.row_by:  # Only copy if not already copied
                df = df.copy()
            df["_col_group"] = df[spec.col_by].apply(
                lambda row: "\n".join(f"{col}={row[col]}" for col in spec.col_by),
                axis=1
            )
            col_var = "_col_group"
        else:
            col_var = None

        # Prepare series grouping with custom labels
        if spec.series_by:
            # Always create a copy if we're customizing labels
            if spec.series_labels or spec.series_label_format:
                if not spec.row_by and not spec.col_by:
                    df = df.copy()

                # Create custom label column
                def make_label(row):
                    group_tuples = tuple((col, row[col]) for col in spec.series_by)
                    return spec.format_series_label(group_tuples)

                df["_custom_series_label"] = df.apply(make_label, axis=1)
                hue = "_custom_series_label"
                style = None
                size = None

            elif len(spec.series_by) <= 3:
                # Use hue, style, size for up to 3 dimensions (no customization)
                hue = spec.series_by[0] if len(spec.series_by) >= 1 else None
                style = spec.series_by[1] if len(spec.series_by) >= 2 else None
                size = spec.series_by[2] if len(spec.series_by) >= 3 else None
            else:
                # Combine into single grouping column
                if not spec.row_by and not spec.col_by:
                    df = df.copy()
                df["_series_group"] = df[spec.series_by].apply(
                    lambda row: "\n".join(f"{col}={row[col]}" for col in spec.series_by),
                    axis=1
                )
                hue = "_series_group"
                style = None
                size = None
        else:
            hue = None
            style = None
            size = None

        # Create the plot
        g = sns.relplot(
            data=df,
            x=spec.x_axis,
            y=spec.y_axis,
            hue=hue,
            style=style,
            size=size,
            row=row_var,
            col=col_var,
            markers=True,
            dashes=False,  # Solid lines
            markersize=8,  # Make markers clearly visible
            errorbar="sd" if spec.show_error_bars else None,
            kind="line",
            height=spec.fig_height,
            aspect=spec.fig_width / spec.fig_height,
            palette=spec.color_palette,
            legend=spec.show_legend,
        )

        # Set titles
        if spec.row_by and spec.col_by:
            g.set_titles("{row_name}\n{col_name}")
        elif spec.row_by:
            g.set_titles("{row_name}")
        elif spec.col_by:
            g.set_titles("{col_name}")
        elif spec.title:
            g.figure.suptitle(spec.title, y=1.02)

        # Set axis labels
        if spec.x_label:
            g.set_axis_labels(spec.x_label, spec.y_label or spec.y_axis)
        elif spec.y_label:
            g.set_axis_labels(spec.x_axis, spec.y_label)

        # Set scales
        if spec.x_scale != "linear":
            g.set(xscale=spec.x_scale)
        if spec.y_scale != "linear":
            g.set(yscale=spec.y_scale)

        # Grid - apply to all axes in the FacetGrid
        if spec.show_grid:
            for ax in g.axes.flat:
                ax.grid(True, alpha=0.3)

        # Apply styling to all axes
        for ax in g.axes.flat:
            # Show bounding box (all 4 spines)
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)

            # Make spines solid and visible
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_edgecolor('black')

        # Adjust legend
        if spec.show_legend:
            # FacetGrid doesn't support loc='best', convert to 'upper right'
            legend_loc = spec.legend_location
            if legend_loc == "best":
                legend_loc = "upper right"

            if spec.legend_outside:
                # Place legend outside plot area on the right
                sns.move_legend(
                    g,
                    "center left",
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=True,
                    shadow=True,
                    fancybox=False
                )
            else:
                # Place legend inside plot area
                sns.move_legend(g, legend_loc, frameon=True, shadow=True, fancybox=False)

            # Add frame to legend
            if g.legend is not None:
                g.legend.get_frame().set_linewidth(1.0)
                g.legend.get_frame().set_edgecolor('black')

        # Add value labels to data points
        if spec.show_value_labels:
            self._add_value_labels_seaborn(g, df, spec)

        # Save
        g.savefig(output_path, dpi=spec.fig_dpi, bbox_inches="tight")
        plt.close(g.figure)

        print(f"Saved plot to {output_path}")

    def _add_value_labels_seaborn(
        self,
        g: "sns.FacetGrid",
        df: "pd.DataFrame",
        spec: PlotSpec,
    ) -> None:
        """Add value labels to data points on a seaborn FacetGrid."""
        import numpy as np

        # Iterate through all axes in the FacetGrid
        for ax in g.axes.flat:
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
                    label_text = spec.format_value_label(x_val, y_val)

                    # Add annotation
                    ax.annotate(
                        label_text,
                        xy=(x_val, y_val),
                        xytext=spec.value_label_offset,
                        textcoords='offset points',
                        fontsize=spec.value_label_fontsize,
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

    def _plot_matplotlib(self, spec: PlotSpec, output_dir: Path) -> list[Path]:
        """Generate plots using matplotlib."""
        output_files = []

        # If fig_by is specified, create separate plots for each group
        if spec.fig_by:
            fig_groups = self.df.groupby(spec.fig_by)

            for group_values, group_df in fig_groups:
                if not isinstance(group_values, tuple):
                    group_values = (group_values,)

                fig_name_parts = [spec.output_name]
                for col, val in zip(spec.fig_by, group_values):
                    fig_name_parts.append(f"{col}={val}")
                fig_name = sanitize_filename("-".join(fig_name_parts) + ".png")
                output_path = output_dir / fig_name

                self._create_matplotlib_plot(spec, group_df, output_path)
                output_files.append(output_path)
        else:
            output_path = output_dir / f"{spec.output_name}.png"
            self._create_matplotlib_plot(spec, self.df, output_path)
            output_files.append(output_path)

        return output_files

    def _create_matplotlib_plot(
        self,
        spec: PlotSpec,
        df: "pd.DataFrame",
        output_path: Path,
    ) -> None:
        """Create a single matplotlib plot."""
        # Determine subplot layout
        if spec.row_by or spec.col_by:
            # Create subplots
            row_vals = df[spec.row_by].drop_duplicates().values if spec.row_by else [None]
            col_vals = df[spec.col_by].drop_duplicates().values if spec.col_by else [None]

            nrows = len(row_vals) if spec.row_by else 1
            ncols = len(col_vals) if spec.col_by else 1

            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(spec.fig_width * ncols, spec.fig_height * nrows),
                squeeze=False,
            )

            for i, row_val in enumerate(row_vals):
                for j, col_val in enumerate(col_vals):
                    ax = axes[i, j]

                    # Filter data for this subplot
                    subplot_df = df.copy()
                    if spec.row_by:
                        subplot_df = subplot_df[subplot_df[spec.row_by[0]] == row_val]
                    if spec.col_by:
                        subplot_df = subplot_df[subplot_df[spec.col_by[0]] == col_val]

                    self._plot_on_axis(spec, subplot_df, ax)

                    # Set subplot title
                    title_parts = []
                    if spec.row_by and row_val is not None:
                        title_parts.append(f"{spec.row_by[0]}={row_val}")
                    if spec.col_by and col_val is not None:
                        title_parts.append(f"{spec.col_by[0]}={col_val}")
                    if title_parts:
                        ax.set_title("\n".join(title_parts))

            # Overall title
            if spec.title:
                fig.suptitle(spec.title, fontsize=16)

        else:
            # Single plot
            fig, ax = plt.subplots(figsize=(spec.fig_width, spec.fig_height))
            self._plot_on_axis(spec, df, ax)

            if spec.title:
                ax.set_title(spec.title)

        plt.tight_layout()
        plt.savefig(output_path, dpi=spec.fig_dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved plot to {output_path}")

    def _plot_on_axis(
        self,
        spec: PlotSpec,
        df: "pd.DataFrame",
        ax: "plt.Axes",
    ) -> None:
        """Plot data on a single axis."""
        # Group by series_by
        if spec.series_by:
            series_groups = df.groupby(spec.series_by)
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

            for idx, (group_vals, group_df) in enumerate(series_groups):
                if not isinstance(group_vals, tuple):
                    group_vals = (group_vals,)

                # Create label using custom formatter
                group_tuples = tuple((col, val) for col, val in zip(spec.series_by, group_vals))
                label = spec.format_series_label(group_tuples)

                # Get marker and line style
                marker = "o"
                linestyle = "-"
                if spec.marker_style and label in spec.marker_style:
                    marker = spec.marker_style[label]
                if spec.line_style and label in spec.line_style:
                    linestyle = spec.line_style[label]

                color = color_cycle[idx % len(color_cycle)]

                # Sort by x-axis for proper line plotting
                group_df = group_df.sort_values(by=spec.x_axis)

                ax.plot(
                    group_df[spec.x_axis],
                    group_df[spec.y_axis],
                    marker=marker,
                    linestyle=linestyle,
                    label=label,
                    color=color,
                    markersize=8,  # Make markers clearly visible
                    markeredgewidth=0.5,
                    markeredgecolor='white',
                )
        else:
            # Single series
            df = df.sort_values(by=spec.x_axis)
            ax.plot(
                df[spec.x_axis],
                df[spec.y_axis],
                marker="o",
                linestyle="-",
                markersize=8,
                markeredgewidth=0.5,
                markeredgecolor='white',
            )

        # Set labels
        ax.set_xlabel(spec.x_label or spec.x_axis)
        ax.set_ylabel(spec.y_label or spec.y_axis)

        # Set scales
        if spec.x_scale != "linear":
            ax.set_xscale(spec.x_scale)
        if spec.y_scale != "linear":
            ax.set_yscale(spec.y_scale)

        # Grid
        if spec.show_grid:
            ax.grid(True, alpha=0.3)

        # Show bounding box (all 4 spines)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        # Make spines solid and visible
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_edgecolor('black')

        # Legend
        if spec.show_legend and spec.series_by:
            if spec.legend_outside:
                # Place legend outside plot area on the right
                legend = ax.legend(
                    loc='center left',
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=True,
                    shadow=True,
                    fancybox=False
                )
            else:
                # Place legend inside plot area
                legend = ax.legend(
                    loc=spec.legend_location,
                    frameon=True,
                    shadow=True,
                    fancybox=False
                )

            # Add frame to legend
            legend.get_frame().set_linewidth(1.0)
            legend.get_frame().set_edgecolor('black')

        # Add value labels to data points
        if spec.show_value_labels:
            self._add_value_labels_matplotlib(ax, spec)

    def _add_value_labels_matplotlib(
        self,
        ax: "plt.Axes",
        spec: PlotSpec,
    ) -> None:
        """Add value labels to data points on a matplotlib axis."""
        import numpy as np

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
                label_text = spec.format_value_label(x_val, y_val)

                # Add annotation
                ax.annotate(
                    label_text,
                    xy=(x_val, y_val),
                    xytext=spec.value_label_offset,
                    textcoords='offset points',
                    fontsize=spec.value_label_fontsize,
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

    def plot_multiple(
        self,
        specs: list[PlotSpec],
        output_dir: str | Path,
        backend: str = "seaborn",
    ) -> dict[str, list[Path]]:
        """
        Generate multiple plots from different specs.

        Args:
            specs: List of PlotSpec objects
            output_dir: Directory to save all plots
            backend: Plotting backend

        Returns:
            Dictionary mapping spec names to list of output files
        """
        results = {}
        for spec in specs:
            output_files = self.plot(spec, output_dir, backend)
            results[spec.output_name] = output_files

        return results


def quick_plot(
    data_path: str | Path,
    x_axis: str,
    y_axis: str,
    output_dir: str | Path,
    series_by: list[str] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    backend: str = "seaborn",
    **kwargs: Any,
) -> list[Path]:
    """
    Quick one-liner to create a plot from prepared data.

    Args:
        data_path: Path to prepared data file
        x_axis: Column for x-axis
        y_axis: Column for y-axis
        output_dir: Where to save plot
        series_by: Columns to create different series
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        x_scale: X-axis scale
        y_scale: Y-axis scale
        backend: Plotting backend
        **kwargs: Additional PlotSpec parameters

    Returns:
        List of generated plot file paths
    """
    spec = PlotSpec(
        x_axis=x_axis,
        y_axis=y_axis,
        series_by=series_by or [],
        title=title,
        x_label=x_label,
        y_label=y_label,
        x_scale=x_scale,
        y_scale=y_scale,
        **kwargs,
    )

    generator = PlotGenerator(data_path)
    return generator.plot(spec, output_dir, backend)


def main():
    """Command-line interface for plot generation."""
    parser = argparse.ArgumentParser(
        description="Generate plots from prepared benchmark data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic line plot
  python -m vllm.benchmarks.sweep.plot_framework \\
      prepared_data.csv \\
      --x isl \\
      --y ttft \\
      --series-by tp_size \\
      --title "TTFT vs Input Length" \\
      --x-scale log \\
      --output plots/

  # Multi-panel plot
  python -m vllm.benchmarks.sweep.plot_framework \\
      prepared_data.csv \\
      --x isl \\
      --y ttft \\
      --series-by model \\
      --col-by tp_size \\
      --title "Performance Matrix" \\
      --output plots/
        """
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to prepared data file (.csv, .json, or .parquet)"
    )

    parser.add_argument("--x", dest="x_axis", type=str, required=True, help="X-axis column")
    parser.add_argument("--y", dest="y_axis", type=str, required=True, help="Y-axis column")

    parser.add_argument("--series-by", type=str, nargs="*", help="Columns for different series/curves")
    parser.add_argument("--fig-by", type=str, nargs="*", help="Columns for separate figures")
    parser.add_argument("--row-by", type=str, nargs="*", help="Columns for subplot rows")
    parser.add_argument("--col-by", type=str, nargs="*", help="Columns for subplot columns")

    parser.add_argument("--title", type=str, help="Plot title")
    parser.add_argument("--x-label", type=str, help="X-axis label")
    parser.add_argument("--y-label", type=str, help="Y-axis label")

    parser.add_argument("--x-scale", type=str, default="linear", help="X-axis scale (linear, log, sqrt)")
    parser.add_argument("--y-scale", type=str, default="linear", help="Y-axis scale (linear, log, sqrt)")

    parser.add_argument("--no-error-bars", action="store_true", help="Disable error bars")
    parser.add_argument("--no-legend", action="store_true", help="Disable legend")
    parser.add_argument("--legend-outside", action="store_true", help="Place legend outside plot area (right side)")
    parser.add_argument("--no-grid", action="store_true", help="Disable grid")

    parser.add_argument("--show-value-labels", action="store_true", help="Show value labels on data points")
    parser.add_argument("--value-label-format", type=str, help="Format string for value labels (e.g., '{y:.1f}' or '{x}, {y:.2f}')")
    parser.add_argument("--value-label-fontsize", type=int, default=8, help="Font size for value labels")
    parser.add_argument("--value-label-offset", type=float, nargs=2, default=[0, 5], metavar=("X", "Y"), help="Label offset in points (x, y)")

    parser.add_argument("--fig-width", type=float, default=10.0, help="Figure width in inches")
    parser.add_argument("--fig-height", type=float, default=6.0, help="Figure height in inches")
    parser.add_argument("--fig-dpi", type=int, default=300, help="Figure DPI")

    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--output-name", type=str, default="plot", help="Output filename prefix")
    parser.add_argument("--backend", type=str, default="seaborn", choices=["seaborn", "matplotlib"])

    args = parser.parse_args()

    # Create plot spec
    spec = PlotSpec(
        x_axis=args.x_axis,
        y_axis=args.y_axis,
        series_by=args.series_by or [],
        fig_by=args.fig_by or [],
        row_by=args.row_by or [],
        col_by=args.col_by or [],
        title=args.title,
        x_label=args.x_label,
        y_label=args.y_label,
        x_scale=args.x_scale,
        y_scale=args.y_scale,
        show_error_bars=not args.no_error_bars,
        show_legend=not args.no_legend,
        legend_outside=args.legend_outside,
        show_grid=not args.no_grid,
        show_value_labels=args.show_value_labels,
        value_label_format=args.value_label_format,
        value_label_fontsize=args.value_label_fontsize,
        value_label_offset=tuple(args.value_label_offset),
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        fig_dpi=args.fig_dpi,
        output_name=args.output_name,
    )

    # Generate plots
    generator = PlotGenerator(args.input)
    output_files = generator.plot(spec, args.output, backend=args.backend)

    print(f"\nPlot generation complete!")
    print(f"  Generated {len(output_files)} plot(s)")
    for f in output_files:
        print(f"    - {f}")


if __name__ == "__main__":
    main()

