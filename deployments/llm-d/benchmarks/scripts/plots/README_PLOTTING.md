# Benchmark Plotting Framework

A modular framework for extracting, preparing, and plotting benchmark sweep results with fine-grained control over data processing and visualization.

## Overview

The framework is organized into **3 independent phases**:

1. **Phase 1: Data Extraction** (`extract.py`) - Extract and flatten nested JSON data
2. **Phase 2: Data Preparation** (`prepare.py`) - Filter, transform, and aggregate data
3. **Phase 3: Plot Generation** (`plot_framework.py`) - Create visualizations

Each phase can be run independently or combined through a configuration file.

## Quick Start

### Using Configuration Files (Recommended)

```bash
# Create a configuration file (see examples/plot_config_example.yaml)
python -m vllm.benchmarks.sweep.plot_config my_config.yaml
```

### Using Individual Phases

```bash
# Phase 1: Extract data
python -m vllm.benchmarks.sweep.extract \
    results/sweeps/my_sweep/aggregated_results.json \
    --fields isl=config._load_params.random_input_len \
             tp=config.tensor_parallel_size \
             ttft=benchmark.runs[*].metrics.ttft.mean_ms \
    --run-strategy average \
    --output extracted_data.csv

# Phase 2: Prepare data
python -m vllm.benchmarks.sweep.prepare \
    extracted_data.csv \
    --filter "tp<=4,isl<128000" \
    --transform "ttft_sec=ttft/1000" \
    --output prepared_data.csv

# Phase 3: Generate plots
python -m vllm.benchmarks.sweep.plot_framework \
    prepared_data.csv \
    --x isl \
    --y ttft \
    --series-by tp \
    --title "TTFT vs Input Length" \
    --x-scale log \
    --output plots/
```

## Phase 1: Data Extraction

Extract and flatten data from nested JSON structures like `aggregated_results.json`.

### JSONPath Syntax

Use dot notation and array indexing to navigate nested structures:

- `config.tensor_parallel_size` - Direct field access
- `config._load_params.random_input_len` - Nested field
- `benchmark.runs[0].metrics.ttft.mean_ms` - Array index (first run)
- `benchmark.runs[*].metrics.ttft.mean_ms` - All runs (averaged by default)

### Run Strategies

When multiple benchmark runs exist per configuration:

- `average` (default) - Average numeric metrics across runs
- `all` - Create separate row for each run
- `first` - Use only first run
- `last` - Use only last run

### Example

```python
from vllm.benchmarks.sweep.extract import DataExtractor

extractor = DataExtractor("results/sweeps/my_sweep/aggregated_results.json")

data = extractor.extract(
    field_specs={
        "tp_size": "config.tensor_parallel_size",
        "isl": "config._load_params.random_input_len",
        "ttft": "benchmark.runs[*].metrics.ttft.mean_ms",
    },
    run_strategy="average",
)

extractor.save_extracted_data("extracted.csv")
```

## Phase 2: Data Preparation

Filter, transform, and aggregate extracted data.

### Operations

- **Filter** - Remove unwanted rows using comparison operators
- **Bin** - Group values into bins to reduce noise
- **Transform** - Create derived columns using formulas
- **Aggregate** - Group and summarize data
- **Rename** - Change column names
- **Select** - Keep only specific columns
- **Drop Nulls** - Remove rows with missing values
- **Sort** - Order the data

### Filter Syntax

```python
# Comparison operators: ==, !=, <, <=, >, >=
prep.filter(["tp_size<=4", "isl>=1000", "isl<128000"])
```

### Transform Syntax

```python
# Use pandas eval syntax or callable functions
prep.transform({
    "ttft_sec": "ttft / 1000",  # Formula string
    "speedup": lambda row: row['baseline'] / row['current'],  # Callable
})
```

### Example

```python
from vllm.benchmarks.sweep.prepare import PlotDataPrep

prep = PlotDataPrep("extracted.csv")

prep = (prep
    .filter(["tp_size<=4", "isl<128000"])
    .transform({"ttft_sec": "ttft/1000"})
    .drop_nulls()
    .sort(["tp_size", "isl"])
)

prep.save("prepared.csv")
```

## Phase 3: Plot Generation

Create publication-quality plots with full control over layout and styling.

### Plot Layout

Control the plot structure using:

- `series_by` - Different lines/curves within a subplot
- `row_by` - Create subplot rows
- `col_by` - Create subplot columns
- `fig_by` - Create separate figure files

### Backends

- `seaborn` (default) - Automatic styling, statistical plots
- `matplotlib` - More control, traditional plotting

### Example

```python
from vllm.benchmarks.sweep.plot_framework import PlotSpec, PlotGenerator

spec = PlotSpec(
    x_axis="isl",
    y_axis="ttft",
    series_by=["tp_size"],
    title="TTFT vs Input Length",
    x_scale="log",
    show_grid=True,
)

generator = PlotGenerator("prepared.csv")
generator.plot(spec, output_dir="plots/")
```

### Quick Plot Helper

```python
from vllm.benchmarks.sweep.plot_framework import quick_plot

quick_plot(
    data_path="prepared.csv",
    x_axis="isl",
    y_axis="ttft",
    series_by=["tp_size"],
    title="TTFT Performance",
    x_scale="log",
    output_dir="plots/",
)
```

## Configuration Files

Define complete workflows in YAML or JSON.

### Structure

```yaml
extraction:
  input: path/to/aggregated_results.json
  fields:
    field_name: json.path.to.value
  run_strategy: average

preparation:  # Optional
  filters:
    - "condition1"
    - "condition2"
  transformations:
    new_col: "formula"

plots:
  - name: plot1
    x_axis: column_x
    y_axis: column_y
    series_by: [column_z]
    title: "Plot Title"

output:
  directory: plots/
  backend: seaborn
  dpi: 300
```

### Examples

See:
- `examples/plot_config_example.yaml` - Basic configuration
- `examples/plot_config_advanced.yaml` - Advanced features

## Use Cases

### Compare Tensor Parallel Sizes

```yaml
plots:
  - name: ttft_by_tp
    x_axis: isl
    y_axis: ttft
    series_by: [tensor_parallel_size]
    x_scale: log
```

### Multi-Panel Comparison

```yaml
plots:
  - name: performance_matrix
    x_axis: isl
    y_axis: ttft
    series_by: [model]
    col_by: [tensor_parallel_size]
    row_by: [gpu_util]
```

### Separate Figures Per Model

```yaml
plots:
  - name: model_performance
    x_axis: isl
    y_axis: throughput
    series_by: [tensor_parallel_size]
    fig_by: [model]
```

## Programmatic Usage

Combine all phases programmatically:

```python
from vllm.benchmarks.sweep.extract import DataExtractor
from vllm.benchmarks.sweep.prepare import PlotDataPrep
from vllm.benchmarks.sweep.plot_framework import PlotSpec, PlotGenerator

# Extract
extractor = DataExtractor("aggregated_results.json")
data = extractor.extract({
    "tp": "config.tensor_parallel_size",
    "isl": "config._load_params.random_input_len",
    "ttft": "benchmark.runs[*].metrics.ttft.mean_ms",
})

# Prepare
prep = PlotDataPrep(data)
prep.filter(["tp<=4"]).transform({"ttft_sec": "ttft/1000"})

# Plot
spec = PlotSpec(x_axis="isl", y_axis="ttft_sec", series_by=["tp"])
generator = PlotGenerator(prep)
generator.plot(spec, "plots/")
```

## Command Reference

### extract.py

```bash
python -m vllm.benchmarks.sweep.extract <input> --fields <specs> --output <file>

Arguments:
  input                 Path to aggregated_results.json
  --fields              Field specs (name=path, ...)
  --run-strategy        average|all|first|last
  --include-failed      Include failed runs
  --output              Output file (.csv, .json, .parquet)
```

### prepare.py

```bash
python -m vllm.benchmarks.sweep.prepare <input> [options] --output <file>

Arguments:
  input                 Path to extracted data
  --filter              Filter expressions
  --bin                 Binning expressions
  --transform           Transformations (name=formula, ...)
  --aggregate-by        Group by columns
  --aggregate-funcs     Aggregation functions
  --rename              Rename columns (old=new, ...)
  --select              Select columns
  --drop-nulls          Drop null rows
  --sort-by             Sort columns
  --output              Output file
```

### plot_framework.py

```bash
python -m vllm.benchmarks.sweep.plot_framework <input> --x <col> --y <col> [options] --output <dir>

Arguments:
  input                 Path to prepared data
  --x                   X-axis column
  --y                   Y-axis column
  --series-by           Series columns
  --fig-by              Figure grouping columns
  --row-by              Row grouping columns
  --col-by              Column grouping columns
  --title               Plot title
  --x-scale             X-axis scale (linear|log|sqrt)
  --y-scale             Y-axis scale
  --backend             seaborn|matplotlib
  --output              Output directory
```

### plot_config.py

```bash
python -m vllm.benchmarks.sweep.plot_config <config.yaml>

Arguments:
  config                Path to YAML/JSON config file
  --quiet               Suppress progress messages
```

## Tips

1. **Use configuration files** for complex workflows - easier to maintain and share
2. **Extract once, plot many times** - save extracted/prepared data and create multiple plots
3. **Start with filters** to focus on relevant data before creating expensive aggregations
4. **Use log scale** for metrics that span multiple orders of magnitude
5. **Validate data** at each phase using the `--summary` flag in prepare.py

## Dependencies

- Required: `pandas`
- Optional: `seaborn`, `matplotlib`, `pyyaml` (for YAML config files)

Install with:
```bash
pip install pandas seaborn matplotlib pyyaml
```
