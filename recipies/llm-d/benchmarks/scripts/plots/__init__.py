# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark plotting framework - standalone version.

A modular framework for extracting, preparing, and plotting benchmark sweep results.

Main components:
- extract.py: Phase 1 - Extract data from nested JSON
- prepare.py: Phase 2 - Filter, transform, and aggregate data
- plot_framework.py: Phase 3 - Generate plots
- plot_config.py: Execute complete workflows from YAML/JSON configs
- schema_discovery.py: Analyze and discover available fields

Usage:
    See README_PLOTTING.md for complete documentation.

Quick start:
    python -m plots.plot_config my_config.yaml
"""

__version__ = "1.0.0"

from .extract import DataExtractor
from .prepare import PlotDataPrep
from .plot_framework import PlotSpec, PlotGenerator, quick_plot
from .schema_discovery import SchemaDiscovery
from .plot_from_json import JSONPlotter
from .export_plot_json import PlotJSONExporter

__all__ = [
    "DataExtractor",
    "PlotDataPrep",
    "PlotSpec",
    "PlotGenerator",
    "quick_plot",
    "SchemaDiscovery",
    "JSONPlotter",
    "PlotJSONExporter",
]
