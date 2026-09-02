# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""KV Cache Cliff results → GitHub Pages HTML.

Reads one or more cliff run directories (each containing a results/ subdir with
CSVs and a plots/ subdir with PNGs produced by plot_cliff.py) and generates a
self-contained HTML performance-tracking page.

Each run directory is expected to follow the layout created by run-cliff.sbatch:

    logs/<job-id>/
        results/
            cliff-vram_only-<stamp>.csv
            cliff-kvd_v2-nvme-<stamp>.csv  (optional)
            cliff-kvd_v2-gds-<stamp>.csv   (optional)
        plots/
            nvme/
                cliff-throughput.png
                cliff-latency-p50.png
                cliff-latency-p95.png
            gds/   (same layout, optional)
            vram/  (fallback when no kvd arms ran)

Usage:
    # Single run:
    python benchmarks/cliff_to_html.py \\
        --run-dir logs/12345 \\
        --sha abc1234 --run-date "2026-07-27" \\
        --output-dir cliff-page/

    # Multiple runs merged into one page:
    python benchmarks/cliff_to_html.py \\
        --run-dir logs/12345 logs/12346 \\
        --sha abc1234 --run-date "2026-07-27" \\
        --output-dir cliff-page/

    # Pull run metadata from a JSON manifest (written by spur-cliff-harvest.sh):
    python benchmarks/cliff_to_html.py \\
        --manifest cliff-manifest.json \\
        --output-dir cliff-page/
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _summarise_arm(rows: list[dict]) -> dict[int, dict]:
    """Return {concurrency: {thr_median, p50_median, p95_median}}."""
    by_c: dict[int, list] = {}
    for row in rows:
        try:
            c = int(row["concurrency"])
            thr = float(row["throughput_tok_s_total"])
            p50 = float(row["p50_latency_s"]) * 1000
            p95 = float(row["p95_latency_s"]) * 1000
        except (KeyError, ValueError):
            continue
        by_c.setdefault(c, []).append((thr, p50, p95))
    result: dict[int, dict] = {}
    for c, vals in sorted(by_c.items()):
        result[c] = {
            "thr": statistics.median(v[0] for v in vals),
            "p50": statistics.median(v[1] for v in vals),
            "p95": statistics.median(v[2] for v in vals),
        }
    return result


_ARM_LABELS = {
    "vram_only": "VRAM only (no AIC)",
    "vram_dram": "VRAM + DRAM",
    "kvd_v2":    "AIC (NVMe/NFS via LMCache)",
}

_ARM_COLORS = {
    "vram_only": "#c0392b",
    "vram_dram": "#e67e22",
    "kvd_v2":    "#2980b9",
}


def _peak_thr(summary: dict[int, dict]) -> float:
    return max((v["thr"] for v in summary.values()), default=0.0)


def _cliff_concurrency(summary: dict[int, dict]) -> int | None:
    """Return concurrency where throughput first drops >10% from its peak."""
    peak = 0.0
    for c in sorted(summary):
        y = summary[c]["thr"]
        if y > peak:
            peak = y
        elif peak > 0 and (peak - y) / peak > 0.10:
            return c
    return None


# ---------------------------------------------------------------------------
# PNG → base64 data URI
# ---------------------------------------------------------------------------

def _img_src(path: Path) -> str | None:
    if not path.is_file():
        return None
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def _discover_run(run_dir: Path) -> dict[str, Any]:
    """Scan a single run directory and return its data dict."""
    results_dir = run_dir / "results"
    plots_dir = run_dir / "plots"

    arms: dict[str, dict[int, dict]] = {}
    for csv_path in sorted(results_dir.glob("cliff-*.csv")) if results_dir.is_dir() else []:
        rows = _load_csv(csv_path)
        for row in rows:
            arm = row.get("arm", "").strip()
            if not arm:
                continue
            summary = _summarise_arm(rows)
            if arm not in arms:
                arms[arm] = summary
            break  # one pass per file is enough; arm is constant per CSV

    # charts: try nvme, then gds, then vram subdirs in that order
    charts: dict[str, str | None] = {}
    for sub in ("nvme", "gds", "vram"):
        sub_dir = plots_dir / sub
        if sub_dir.is_dir():
            charts["throughput"] = _img_src(sub_dir / "cliff-throughput.png")
            charts["p50"] = _img_src(sub_dir / "cliff-latency-p50.png")
            charts["p95"] = _img_src(sub_dir / "cliff-latency-p95.png")
            charts["backend"] = sub
            break

    return {"arms": arms, "charts": charts}


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --bg: #0d1117;
  --surface: #161b22;
  --border: #30363d;
  --text: #e6edf3;
  --muted: #8b949e;
  --accent: #2980b9;
  --green: #3fb950;
  --red: #f85149;
  --orange: #e67e22;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 14px; line-height: 1.5; padding: 24px; }
h1 { font-size: 24px; font-weight: 700; margin-bottom: 4px; }
h2 { font-size: 17px; font-weight: 600; margin: 32px 0 12px; color: var(--text); border-bottom: 1px solid var(--border); padding-bottom: 6px; }
.meta { color: var(--muted); font-size: 12px; margin-bottom: 24px; }
.meta a { color: var(--accent); text-decoration: none; }
.meta a:hover { text-decoration: underline; }
.kpi-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px 20px; min-width: 180px; }
.kpi-label { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 6px; }
.kpi-value { font-size: 26px; font-weight: 700; color: var(--text); }
.kpi-sub { font-size: 11px; color: var(--muted); margin-top: 2px; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 20px; margin-bottom: 32px; }
.chart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.chart-card h3 { font-size: 13px; font-weight: 600; color: var(--muted); margin-bottom: 12px; text-transform: uppercase; letter-spacing: .06em; }
.chart-card img { width: 100%; height: auto; border-radius: 4px; }
.chart-card .missing { color: var(--muted); font-size: 12px; font-style: italic; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
thead th { text-align: left; padding: 8px 12px; font-weight: 600; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; border-bottom: 1px solid var(--border); }
tbody tr { border-bottom: 1px solid var(--border); }
tbody tr:hover { background: var(--surface); }
tbody td { padding: 8px 12px; }
.badge-ok { color: var(--green); }
.badge-miss { color: var(--red); }
.badge-na { color: var(--muted); }
.arm-vram_only { color: #e05c4e; }
.arm-vram_dram { color: var(--orange); }
.arm-kvd_v2 { color: var(--accent); }
input#search { width: 100%; max-width: 400px; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; color: var(--text); padding: 8px 12px; font-size: 13px; margin-bottom: 12px; outline: none; }
input#search:focus { border-color: var(--accent); }
"""

_JS = """
document.getElementById('search').addEventListener('input', function() {
  const q = this.value.toLowerCase();
  document.querySelectorAll('#results-table tbody tr').forEach(function(tr) {
    tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
  });
});
"""


def _arm_span(arm: str) -> str:
    label = _ARM_LABELS.get(arm, arm)
    return f'<span class="arm-{arm}">{label}</span>'


def _fmt_thr(v: float) -> str:
    return f"{v:,.0f} tok/s"


def _fmt_lat(v: float) -> str:
    return f"{v:.0f} ms"


def _build_html(
    runs: list[dict[str, Any]],
    page_sha: str,
    page_date: str,
    repo_url: str,
) -> str:

    # Most recent run's data for KPIs + charts
    latest = runs[-1] if runs else {}
    latest_arms = latest.get("arms", {})
    latest_charts = latest.get("charts", {})

    # --- KPI cards ---
    kpi_cards = ""
    for arm in ("vram_only", "vram_dram", "kvd_v2"):
        if arm not in latest_arms:
            continue
        summary = latest_arms[arm]
        peak = _peak_thr(summary)
        cliff = _cliff_concurrency(summary)
        label = _ARM_LABELS.get(arm, arm)
        cliff_str = f"cliff @ c={cliff}" if cliff else "no cliff detected"
        kpi_cards += f"""
        <div class="kpi">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{peak:,.0f}</div>
          <div class="kpi-sub">tok/s peak &nbsp;·&nbsp; {cliff_str}</div>
        </div>"""

    if not kpi_cards:
        kpi_cards = '<div class="kpi"><div class="kpi-label">No data yet</div><div class="kpi-value">—</div></div>'

    # --- Chart cards ---
    def _chart_card(title: str, src_key: str) -> str:
        src = latest_charts.get(src_key)
        inner = f'<img src="{src}" alt="{title}">' if src else '<p class="missing">Chart not available for this run</p>'
        return f'<div class="chart-card"><h3>{title}</h3>{inner}</div>'

    backend = latest_charts.get("backend", "nvme")
    charts_html = f"""
    <div class="charts">
      {_chart_card(f"Throughput vs Concurrency ({backend} backend)", "throughput")}
      {_chart_card(f"p50 Latency vs Concurrency ({backend} backend)", "p50")}
      {_chart_card(f"p95 Latency vs Concurrency ({backend} backend)", "p95")}
    </div>"""

    # --- Results table (one row per arm per concurrency, latest run) ---
    rows_html = ""
    for arm in ("vram_only", "vram_dram", "kvd_v2"):
        if arm not in latest_arms:
            continue
        summary = latest_arms[arm]
        cliff_c = _cliff_concurrency(summary)
        for c in sorted(summary):
            d = summary[c]
            at_cliff = cliff_c is not None and c == cliff_c
            cliff_marker = ' <span class="badge-miss" title="throughput cliff">⚡</span>' if at_cliff else ""
            rows_html += f"""<tr>
              <td>{page_date}</td>
              <td>{_arm_span(arm)}</td>
              <td>{c}</td>
              <td>{_fmt_thr(d['thr'])}{cliff_marker}</td>
              <td>{_fmt_lat(d['p50'])}</td>
              <td>{_fmt_lat(d['p95'])}</td>
            </tr>"""

    if not rows_html:
        rows_html = '<tr><td colspan="6" style="color:var(--muted);text-align:center;padding:24px">No results yet — run the nightly cliff workflow.</td></tr>'

    commit_link = f'<a href="{repo_url}/commit/{page_sha}">{page_sha[:7]}</a>' if repo_url else page_sha[:7]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AIC Cliff Performance — {page_date}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>AIC KV Cache Cliff — Performance Tracking</h1>
  <p class="meta">
    Last updated: {page_date} &nbsp;·&nbsp;
    Commit: {commit_link} &nbsp;·&nbsp;
    <a href="{repo_url}/actions/workflows/aic-amd-nightly-accuracy-cliff.yml">Nightly Cliff workflow</a>
  </p>

  <h2>Latest Run — Key Metrics</h2>
  <div class="kpi-row">{kpi_cards}</div>

  <h2>Charts</h2>
  {charts_html}

  <h2>Raw Results</h2>
  <input id="search" type="search" placeholder="Filter results…" autocomplete="off">
  <table id="results-table">
    <thead>
      <tr>
        <th>Date</th>
        <th>Arm</th>
        <th>Concurrency</th>
        <th>Throughput</th>
        <th>p50 Latency</th>
        <th>p95 Latency</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>

  <script>{_JS}</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--run-dir", nargs="+", metavar="DIR",
        help="One or more run directories (each with results/ and plots/ subdirs)",
    )
    grp.add_argument(
        "--manifest", metavar="JSON",
        help="JSON manifest written by spur-cliff-harvest.sh "
             "(keys: run_dir, sha, run_date, repo_url)",
    )
    parser.add_argument("--sha", default="", help="Git SHA for this run (used in links)")
    parser.add_argument("--run-date", default="", help="ISO date string for this run")
    parser.add_argument(
        "--repo-url", default="https://github.com/ROCm/rocm-aic",
        help="GitHub repo base URL for commit/workflow links",
    )
    parser.add_argument(
        "--output-dir", default="cliff-page",
        help="Directory to write index.html into (default: cliff-page/)",
    )
    args = parser.parse_args()

    if args.manifest:
        with open(args.manifest) as fh:
            manifest = json.load(fh)
        run_dirs = [Path(manifest["run_dir"])]
        sha = manifest.get("sha", "")
        run_date = manifest.get("run_date", "")
        repo_url = manifest.get("repo_url", args.repo_url)
    else:
        run_dirs = [Path(d) for d in args.run_dir]
        sha = args.sha
        run_date = args.run_date
        repo_url = args.repo_url

    runs = []
    for d in run_dirs:
        if not d.is_dir():
            print(f"WARN: {d} is not a directory, skipping", file=sys.stderr)
            continue
        runs.append(_discover_run(d))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = _build_html(runs, sha, run_date, repo_url)
    out_path = out_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Written {out_path} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
