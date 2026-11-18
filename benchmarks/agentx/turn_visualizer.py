#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""AgentX turn visualizer.

Reads the per-request export of an AgentX run (`profile_export.jsonl` in the
run directory, aiperf 0.13.0) and renders an interactive HTML timeline
showing per-turn latencies, agent depth (parent/child subagent fan-out), and
per-conversation flow.

Usage:
    python3 turn_visualizer.py <run dir>/profile_export.jsonl [--output turns.html] [--max-convs 50]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _load_records(jsonl_path: Path) -> list[dict]:
    """Read aiperf's per-request export (profile_export.jsonl).

    aiperf 0.13.0 writes one object per request: `metadata` identifies it
    (conversation, session, turn index, agent depth, phase, timestamps) and
    `metrics` holds `{"value": ..., "unit": ...}` measurements.
    """
    records: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping malformed line {lineno}: {exc}", file=sys.stderr)
                continue
            if not isinstance(obj, dict) or not isinstance(obj.get("metadata"), dict):
                continue
            meta = obj["metadata"]
            metrics = obj.get("metrics") or {}
            # Warmup turns are kept, and drawn in their own colour.
            records.append({
                "phase": meta.get("benchmark_phase", "unknown"),
                "conversation_id": meta.get("conversation_id") or "?",
                "x_correlation_id": meta.get("x_correlation_id") or "?",
                "turn_index": int(meta.get("turn_index") or 0),
                "agent_depth": int(meta.get("agent_depth") or 0),
                "parent_correlation_id": meta.get("parent_correlation_id"),
                "source_kind": meta.get("source_kind"),
                "ttft_ms": _ms(metrics.get("time_to_first_token")),
                "itl_ms": _ms(metrics.get("inter_token_latency")),
                "request_latency_ms": _ms(metrics.get("request_latency")),
                "issued_at_ns": meta.get("credit_issued_ns") or meta.get("request_start_ns"),
                "output_tokens": _val(metrics.get("output_sequence_length")),
                "input_tokens": _val(metrics.get("input_sequence_length")),
                "error": obj.get("error"),
            })
    # The export has no turn count: the last index seen in a session stands in.
    last_turn: dict[str, int] = defaultdict(int)
    for r in records:
        last_turn[r["x_correlation_id"]] = max(last_turn[r["x_correlation_id"]], r["turn_index"])
    for r in records:
        r["last_turn"] = last_turn[r["x_correlation_id"]]
    return records


def _val(v):
    """Unwrap a {"value": x, "unit": y} metric, or return a bare value."""
    if isinstance(v, dict):
        return v.get("value")
    return v


_TO_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3, "sec": 1e3}


def _ms(v) -> float | None:
    """Milliseconds from a metric, by its unit; a bare number is taken as ms."""
    raw = _val(v)
    if not isinstance(raw, (int, float)):
        return None
    unit = v.get("unit", "ms") if isinstance(v, dict) else "ms"
    if unit not in _TO_MS:
        return None
    return round(float(raw) * _TO_MS[unit], 2)


def _build_html(records: list[dict], max_convs: int) -> str:
    # Group by conversation_id then x_correlation_id
    by_conv: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_conv[r["conversation_id"]][r["x_correlation_id"]].append(r)

    # Sort conversations by first issued_at_ns
    def conv_start(conv_id: str) -> float:
        all_r = [r for sess in by_conv[conv_id].values() for r in sess]
        ts = [r["issued_at_ns"] for r in all_r if r.get("issued_at_ns")]
        return min(ts) if ts else 0.0

    conv_ids = sorted(by_conv.keys(), key=conv_start)[:max_convs]

    # Build Plotly traces
    traces: list[dict] = []
    conv_labels: list[str] = []

    for conv_idx, conv_id in enumerate(conv_ids):
        sessions = by_conv[conv_id]
        conv_labels.append(f"conv {conv_idx}: {conv_id[:12]}")

        for corr_id, turns in sessions.items():
            depth = turns[0]["agent_depth"] if turns else 0
            is_child = depth > 0
            parent_corr = turns[0].get("parent_correlation_id")
            label_prefix = f"depth={depth} " if depth > 0 else ""

            x_vals, y_vals, texts, colors, sizes = [], [], [], [], []

            for t in sorted(turns, key=lambda r: r["turn_index"]):
                turn_idx = t["turn_index"]
                latency = t["request_latency_ms"]
                ttft = t["ttft_ms"]
                out_tok = t["output_tokens"]
                in_tok = t["input_tokens"]
                phase = t["phase"]
                err = t["error"]

                x_vals.append(turn_idx)
                y_vals.append(latency if latency is not None else 0)
                color = (
                    "#e74c3c" if err
                    else "#e67e22" if is_child
                    else "#3498db"
                )
                if phase == "warmup":
                    color = "#95a5a6"
                colors.append(color)
                sizes.append(8 if is_child else 12)

                hover = (
                    f"conv={conv_id[:16]}<br>"
                    f"corr={corr_id[:16]}<br>"
                    f"turn={turn_idx} (last seen {t['last_turn']})<br>"
                    f"depth={depth}<br>"
                    f"phase={phase}<br>"
                    f"latency={latency} ms<br>"
                    f"ttft={ttft} ms<br>"
                    f"itl={t['itl_ms']} ms<br>"
                    f"output_tokens={out_tok}<br>"
                    f"input_tokens={in_tok}"
                )
                if err:
                    hover += f"<br><b>ERROR</b>: {str(err)[:80]}"
                if parent_corr:
                    hover += f"<br>parent={parent_corr[:16]}"
                texts.append(hover)

            traces.append({
                "type": "scatter",
                "mode": "lines+markers",
                "name": f"{label_prefix}{corr_id[:12]}",
                "x": x_vals,
                "y": y_vals,
                "text": texts,
                "hoverinfo": "text",
                "marker": {
                    "color": colors,
                    "size": sizes,
                    "symbol": "circle" if not is_child else "diamond",
                },
                "line": {
                    "color": "#e67e22" if is_child else "#3498db",
                    "width": 1,
                    "dash": "dot" if is_child else "solid",
                },
                "legendgroup": conv_id,
                "showlegend": True,
            })

    # Summary stats
    prof_records = [r for r in records if r["phase"] == "profiling" and not r["error"]]
    ttfts = [r["ttft_ms"] for r in prof_records if r["ttft_ms"] is not None]
    latencies = [r["request_latency_ms"] for r in prof_records if r["request_latency_ms"] is not None]
    out_toks = [r["output_tokens"] for r in prof_records if r["output_tokens"] is not None]

    def pct(lst: list, p: float) -> str:
        if not lst:
            return "N/A"
        s = sorted(lst)
        idx = int(len(s) * p / 100)
        return f"{s[min(idx, len(s)-1)]:.1f}"

    summary_html = f"""
    <div class="summary">
      <h2>Run Summary — profiling phase ({len(prof_records)} requests)</h2>
      <table>
        <tr><th>Metric</th><th>P50</th><th>P90</th><th>P99</th></tr>
        <tr><td>TTFT (ms)</td><td>{pct(ttfts,50)}</td><td>{pct(ttfts,90)}</td><td>{pct(ttfts,99)}</td></tr>
        <tr><td>Request Latency (ms)</td><td>{pct(latencies,50)}</td><td>{pct(latencies,90)}</td><td>{pct(latencies,99)}</td></tr>
        <tr><td>Output Tokens</td><td>{pct(out_toks,50)}</td><td>{pct(out_toks,90)}</td><td>{pct(out_toks,99)}</td></tr>
      </table>
      <p>
        Conversations shown: {len(conv_ids)} (of {len(by_conv)} total) &nbsp;|&nbsp;
        Total turns: {len(records)} &nbsp;|&nbsp;
        Errors: {sum(1 for r in records if r['error'])}
      </p>
    </div>
    """

    traces_json = json.dumps(traces)
    layout = json.dumps({
        "title": "AgentX Turn Timeline — Request Latency per Turn",
        "xaxis": {"title": "Turn index"},
        "yaxis": {"title": "Request latency (ms)"},
        "hovermode": "closest",
        "legend": {"orientation": "v", "x": 1.01},
        "plot_bgcolor": "#1e1e2e",
        "paper_bgcolor": "#181825",
        "font": {"color": "#cdd6f4"},
        "xaxis_gridcolor": "#313244",
        "yaxis_gridcolor": "#313244",
    })

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AgentX Turn Visualizer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ margin:0; background:#181825; color:#cdd6f4; font-family:monospace; }}
  #plot {{ width:100%; height:70vh; }}
  .summary {{ padding:1em 2em; }}
  .summary table {{ border-collapse:collapse; margin-top:0.5em; }}
  .summary th,td {{ border:1px solid #313244; padding:0.4em 1em; text-align:right; }}
  .summary th {{ background:#313244; text-align:center; }}
  h1 {{ padding:0.5em 2em 0; }}
  .legend-key {{ display:inline-block; width:1em; height:0.8em; margin:0 0.3em; vertical-align:middle; border-radius:2px; }}
  .legend-row {{ padding:0.2em 2em; font-size:0.85em; }}
</style>
</head>
<body>
<h1>AgentX Turn Visualizer</h1>
{summary_html}
<div class="legend-row">
  <span class="legend-key" style="background:#3498db"></span>Parent (depth=0)
  &nbsp;
  <span class="legend-key" style="background:#e67e22"></span>Child subagent (depth&gt;0)
  &nbsp;
  <span class="legend-key" style="background:#95a5a6"></span>Warmup
  &nbsp;
  <span class="legend-key" style="background:#e74c3c"></span>Error
</div>
<div id="plot"></div>
<script>
  const traces = {traces_json};
  const layout = {layout};
  Plotly.newPlot('plot', traces, layout, {{responsive: true}});
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize AgentX aiperf turn data as HTML")
    parser.add_argument("jsonl", type=Path, help="profile_export.jsonl of an AgentX run")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output HTML file (default: <jsonl>.html)")
    parser.add_argument("--max-convs", type=int, default=60,
                        help="Max conversations to render (default: 60)")
    args = parser.parse_args()

    if not args.jsonl.exists():
        sys.exit(f"ERROR: {args.jsonl} not found")

    print(f"Loading {args.jsonl} ...", file=sys.stderr)
    records = _load_records(args.jsonl)
    print(f"Loaded {len(records)} records.", file=sys.stderr)

    if not records:
        sys.exit("No records found in JSONL file.")

    out = args.output or args.jsonl.with_suffix(".html")
    html = _build_html(records, args.max_convs)
    out.write_text(html, encoding="utf-8")
    print(f"Written: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
