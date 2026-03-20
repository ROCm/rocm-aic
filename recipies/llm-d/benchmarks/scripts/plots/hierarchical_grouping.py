#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Helper for hierarchical series grouping.

Supports patterns like "group all TP values for run1, then all TP values for run2"
with automatic styling (linestyle by primary group, color by secondary group).
"""

from __future__ import annotations

from typing import Any


def create_hierarchical_labels(
    df,
    primary_col: str,
    secondary_col: str,
    primary_labels: dict[str, str] | None = None,
    secondary_labels: dict[str, str] | None = None,
    separator: str = " - ",
) -> tuple[Any, dict]:
    """
    Create hierarchical series labels and styling recommendations.

    Args:
        df: DataFrame with data
        primary_col: Primary grouping column (e.g., run_label)
        secondary_col: Secondary grouping column (e.g., tp_size)
        primary_labels: Custom labels for primary values
        secondary_labels: Custom labels for secondary values
        separator: String between primary and secondary labels

    Returns:
        (modified_df, style_recommendations)
    """
    import pandas as pd

    df = df.copy()

    # Get unique values
    primary_values = sorted(df[primary_col].unique())
    secondary_values = sorted(df[secondary_col].unique())

    # Create hierarchical label
    def make_label(row):
        primary_val = row[primary_col]
        secondary_val = row[secondary_col]

        # Apply custom labels
        if primary_labels and str(primary_val) in primary_labels:
            primary_str = primary_labels[str(primary_val)]
        else:
            primary_str = str(primary_val)

        if secondary_labels and str(secondary_val) in secondary_labels:
            secondary_str = secondary_labels[str(secondary_val)]
        else:
            secondary_str = f"{secondary_col}={secondary_val}"

        return f"{primary_str}{separator}{secondary_str}"

    df["_hierarchical_label"] = df.apply(make_label, axis=1)

    # Create styling recommendations
    # Linestyle by primary group, marker/color by secondary group
    linestyles = {"-": "solid", "--": "dashed", "-.": "dashdot", ":": "dotted"}
    markers = ["o", "^", "s", "D", "v", "p", "h", "*"]

    linestyle_map = {}
    for i, pval in enumerate(primary_values):
        linestyle = list(linestyles.keys())[i % len(linestyles)]
        # Map all labels for this primary value to this linestyle
        for sval in secondary_values:
            temp_row = pd.DataFrame({primary_col: [pval], secondary_col: [sval]})
            label = make_label(temp_row.iloc[0])
            linestyle_map[label] = linestyle

    marker_map = {}
    for i, sval in enumerate(secondary_values):
        marker = markers[i % len(markers)]
        # Map all labels for this secondary value to this marker
        for pval in primary_values:
            temp_row = pd.DataFrame({primary_col: [pval], secondary_col: [sval]})
            label = make_label(temp_row.iloc[0])
            marker_map[label] = marker

    recommendations = {
        "series_by": ["_hierarchical_label"],
        "marker_style": marker_map,
        "line_style": linestyle_map,
        "description": (
            f"Hierarchical grouping: {len(primary_values)} {primary_col} groups "
            f"× {len(secondary_values)} {secondary_col} values = "
            f"{len(primary_values) * len(secondary_values)} series"
        )
    }

    return df, recommendations


def create_hierarchical_series_order(
    primary_values: list,
    secondary_values: list,
    primary_labels: dict | None = None,
    secondary_labels: dict | None = None,
    separator: str = " - ",
) -> list[str]:
    """
    Generate ordered list of series names for hierarchical grouping.

    This ensures legend shows series in the right order:
    - All secondary values for first primary value
    - Then all secondary values for second primary value
    - etc.

    Example:
        primary = ["run1", "run2"]
        secondary = [1, 2, 4]
        Result: ["run1 - TP=1", "run1 - TP=2", "run1 - TP=4",
                 "run2 - TP=1", "run2 - TP=2", "run2 - TP=4"]
    """
    series_names = []

    for pval in primary_values:
        pstr = primary_labels.get(str(pval), str(pval)) if primary_labels else str(pval)

        for sval in secondary_values:
            sstr = secondary_labels.get(str(sval), f"{sval}") if secondary_labels else str(sval)
            label = f"{pstr}{separator}{sstr}"
            series_names.append(label)

    return series_names
