#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Parse wheel-test.out and render an HTML latency table for GitHub Pages.

Usage:
  python3 render-wheel-latency.py <log-file> <sha> <date> <output-html>
"""
import json, re, sys
from pathlib import Path
from datetime import datetime, timezone

log_path = Path(sys.argv[1])
sha      = sys.argv[2]
date_str = sys.argv[3]          # ISO-8601, e.g. 2026-10-30T06:37:00Z
out_html = Path(sys.argv[4])

# ---- Parse results from the log ----------------------------------------
# Lines of interest (emitted by tier_bench.py):
#   [cold] TOTAL=9.872s  n=25  avg=0.395s
#   [cold]   isl=1100: total=1.206s  avg=0.241s  n=5
# Phases: cold / hot / L1-DRAM / L2-disk
PHASES = ["cold", "hot", "L1-DRAM", "L2-disk"]

phase_total: dict[str, float] = {}
phase_isl:   dict[str, dict[int, float]] = {p: {} for p in PHASES}

total_re = re.compile(r'^\s*\[(\w[\w-]*)\] TOTAL=([\d.]+)s')
isl_re   = re.compile(r'^\s*\[(\w[\w-]*)\]\s+isl=\s*(\d+):\s+total=[\d.]+s\s+avg=([\d.]+)s')

text = log_path.read_text(errors="replace")
for line in text.splitlines():
    m = total_re.match(line)
    if m:
        phase, t = m.group(1), float(m.group(2))
        if phase in PHASES:
            phase_total[phase] = t
        continue
    m = isl_re.match(line)
    if m:
        phase, isl, avg = m.group(1), int(m.group(2)), float(m.group(3))
        if phase in PHASES:
            phase_isl[phase][isl] = avg

isls = sorted({isl for d in phase_isl.values() for isl in d})

# ---- Determine speedup over cold ----------------------------------------
def pct(phase_avg: float, cold_avg: float) -> str:
    d = (cold_avg - phase_avg) / cold_avg * 100
    sign = "+" if d < 0 else "−"
    return f"{sign}{abs(d):.0f}%"

# ---- JSON output (for future tooling) -----------------------------------
result = {
    "sha": sha,
    "date": date_str,
    "model": "Qwen/Qwen3-4B",
    "phases": PHASES,
    "isl_tiers": isls,
    "totals": phase_total,
    "avg_per_query": {p: phase_isl[p] for p in PHASES},
}
json_out = out_html.with_suffix(".json")
json_out.write_text(json.dumps(result, indent=2))

# ---- HTML ----------------------------------------------------------------
def cell(v: float) -> str:
    return f"{v:.3f}s"

def speedup_cell(phase: str, isl: int) -> str:
    cold_avg = phase_isl["cold"].get(isl)
    this_avg = phase_isl[phase].get(isl)
    if cold_avg is None or this_avg is None:
        return "<td>—</td>"
    delta_ms = (cold_avg - this_avg) * 1000
    pct_val  = (cold_avg - this_avg) / cold_avg * 100
    cls = "faster" if pct_val > 1 else ("slower" if pct_val < -1 else "neutral")
    sign = "−" if pct_val > 0 else "+"
    return (f'<td class="{cls}">{this_avg:.3f}s '
            f'<span class="delta">({sign}{abs(pct_val):.0f}%, {sign}{abs(delta_ms):.0f} ms)</span></td>')

isl_rows = ""
for isl in isls:
    cold_avg = phase_isl["cold"].get(isl, 0)
    isl_rows += f"<tr><td><strong>{isl:,}</strong></td>"
    isl_rows += f"<td>{cold_avg:.3f}s</td>"
    for phase in PHASES[1:]:
        isl_rows += speedup_cell(phase, isl)
    isl_rows += "</tr>\n"

total_rows = ""
for phase in PHASES:
    t = phase_total.get(phase)
    total_rows += f"<tr><td>{phase}</td><td>{t:.2f}s</td></tr>\n" if t else ""

ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AIC Wheel Latency — {date_str[:10]}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0d1117; color: #e6edf3; padding: 2rem 1.5rem; max-width: 900px; margin: auto; }}
    h1   {{ color: #f0f6fc; font-size: 1.6rem; margin-bottom: 0.25rem; }}
    .meta {{ color: #8b949e; font-size: 0.85rem; margin-bottom: 2rem; }}
    .meta a {{ color: #58a6ff; text-decoration: none; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
    th, td {{ padding: 0.5rem 0.8rem; text-align: right; border: 1px solid #30363d; font-size: 0.9rem; }}
    th {{ background: #161b22; color: #8b949e; text-align: center; }}
    td:first-child {{ text-align: left; }}
    .faster  {{ color: #3fb950; }}
    .slower  {{ color: #f85149; }}
    .neutral {{ color: #e6edf3; }}
    .delta   {{ font-size: 0.78rem; opacity: 0.75; }}
    h2 {{ font-size: 1rem; color: #8b949e; margin: 1.5rem 0 0.4rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    footer {{ color: #484f58; font-size: 0.8rem; margin-top: 2rem; }}
  </style>
</head>
<body>
<h1>AIC Wheel Latency Benchmark</h1>
<p class="meta">
  Model: <strong>Qwen/Qwen3-4B</strong> &nbsp;·&nbsp;
  Hardware: <strong>MI355X (gfx950)</strong> &nbsp;·&nbsp;
  Wheels SHA: <a href="https://github.com/ROCm/rocm-aic/commit/{sha}">{sha[:12]}</a> &nbsp;·&nbsp;
  Date: <strong>{date_str[:10]}</strong> &nbsp;·&nbsp;
  OSL: 64 tokens &nbsp;·&nbsp; Serial (1 request at a time)
</p>

<h2>Latency by ISL tier (avg per request)</h2>
<p style="color:#8b949e;font-size:0.82rem;margin-bottom:0.6rem">
  Speedup vs cold prefill shown in parentheses.
  hot = vLLM VRAM prefix cache &nbsp;|&nbsp;
  L1 = LMCache CPU DRAM &nbsp;|&nbsp;
  L2 = LMCache local disk (fs_native)
</p>
<table>
  <thead>
    <tr>
      <th>ISL (tokens)</th>
      <th>cold</th>
      <th>hot (VRAM)</th>
      <th>L1-DRAM</th>
      <th>L2-disk</th>
    </tr>
  </thead>
  <tbody>
{isl_rows}  </tbody>
</table>

<h2>Total wall time for all questions per phase</h2>
<table>
  <thead><tr><th>Phase</th><th>Total</th></tr></thead>
  <tbody>
{total_rows}  </tbody>
</table>

<footer>Generated {ts} · <a href="https://github.com/ROCm/rocm-aic" style="color:#484f58">ROCm/rocm-aic</a></footer>
</body>
</html>
"""
out_html.write_text(html)
print(f"Written: {out_html} and {json_out}")
