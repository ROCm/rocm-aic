# AIC Whitepaper

Source for the AMD Infinity Context whitepaper, including benchmark data, plot generation scripts, and document build tooling.

## Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| Python 3.10+ | Plot generation scripts | `sudo apt install python3` |
| pip | Python package manager | bundled with Python |
| [just](https://just.systems) | Task runner | `cargo install just` or `sudo apt install just` |
| pandoc | Markdown → PDF/Word | `sudo apt install pandoc` |
| XeLaTeX | PDF rendering | `sudo apt install texlive-xetex texlive-fonts-recommended` |
| DejaVu fonts | Unicode glyph coverage in PDF | `sudo apt install fonts-dejavu` |

## Setup

**1. Create and activate a virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install Python dependencies:**

```bash
just install
# or directly: pip install -r requirements.txt
```

## Regenerating the plots

Benchmark CSVs are pre-extracted and stored in `results/`. To regenerate both plots from them:

```bash
just plots
```

This produces:
- `kv-throughput-cliff-mi300x.png` — throughput cliff chart (warm run, tok/s vs. concurrent clients)
- `kv-ttft-climb-mi300x.png` — TTFT climb chart (warm run, seconds vs. concurrent clients)

To regenerate a single chart:

```bash
just plot-throughput
just plot-ttft
```

### Re-extracting CSVs from raw JSON

If you have new benchmark JSON files, extract them with:

```bash
python3 json_to_csv.py <path/to/sweep.json> [<path/to/sweep2.json> ...]
```

Then move the generated `.csv` files into `results/` before running `just plots`.

## Building the document

```bash
just pdf     # two-column PDF (requires XeLaTeX)
just docx    # Word document for collaborative editing
just all     # plots + pdf + docx
```

The Word export uses `reference.docx` for styling if one is present in this directory, otherwise pandoc defaults apply. To create a styled template:

```bash
pandoc -o reference.docx --print-default-data-file reference.docx
# open reference.docx, apply your styles, save — then re-run just docx
```

## File overview

```
aic-whitepaper.md      Main whitepaper source (Markdown + pandoc citations)
references.bib         BibTeX bibliography
ieee.csl               IEEE citation style
header.tex             LaTeX preamble fix for tables in two-column layout
results/               Pre-extracted benchmark CSVs (one per series)
json_to_csv.py         Extracts metrics from sweep JSON → CSV
plot_metrics.py        Generates PNG plots from CSVs (configurable)
justfile               Task runner recipes
requirements.txt       Python dependencies
```
