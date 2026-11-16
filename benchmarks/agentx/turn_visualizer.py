#!/usr/bin/env python3
"""AgentX turn visualizer.

Reads an aiperf JSONL results file and renders an interactive HTML timeline
showing per-turn latencies, agent depth (parent/child subagent fan-out), and
per-conversation flow.

Usage:
    python3 turn_visualizer.py <results.jsonl> [--output turns.html] [--max-convs 50]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _load_records(jsonl_path: Path) -> list[dict]:
    records: list[dict] = []
    with open(jsonl_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping malformed line {lineno}: {exc}", file=sys.stderr)
                continue
            if not isinstance(obj, dict):
                continue
            # Accept all phases so warmup turns are visible too, but mark them
            meta = obj.get("metadata", {})
            phase = meta.get("benchmark_phase", "unknown")
            metrics = obj.get("metrics", obj)  # fallback: top-level fields
            error = obj.get("error")
            records.append({
                "phase": phase,
                "conversation_id": metrics.get("conversation_id") or meta.get("conversation_id", "?"),
                "x_correlation_id": metrics.get("x_correlation_id") or meta.get("x_correlation_id", "?"),
                "turn_index": int(metrics.get("turn_index", 0)),
                "num_turns": int(metrics.get("num_turns", 1)),
                "agent_depth": int(metrics.get("agent_depth", 0)),
                "parent_correlation_id": metrics.get("parent_correlation_id"),
                "ttft_ms": _ns_to_ms(_val(metrics.get("time_to_first_token") or metrics.get("ttft_ns"))),
                "itl_ms": _ns_to_ms(_val(metrics.get("inter_token_latency") or metrics.get("itl_ns"))),
                "request_latency_ms": _ms_val(metrics.get("request_latency") or metrics.get("request_latency_ns")),
                "issued_at_ns": _val(metrics.get("issued_at_ns")),
                "output_tokens": _val(metrics.get("output_sequence_length")),
                "input_tokens": _val(metrics.get("input_sequence_length")),
                "error": error,
            })
    return records


def _val(v):
    """Unwrap {"value": x, "unit": y} dicts or return raw value."""
    if isinstance(v, dict):
        return v.get("value")
    return v

def _ms_val(v):
    """Return milliseconds from a metric that may already be in ms or be ns."""
    raw = _val(v)
    if raw is None:
        return None
    # if unit says ms, return directly; otherwise assume ns
    if isinstance(v, dict) and v.get("unit") == "ms":
        return round(float(raw), 2)
    return round(float(raw) / 1_000_000, 2)

def _ns_to_ms(v: int | float | None) -> float | None:
    if v is None:
        return None
    return round(float(v) / 1_000_000, 2)


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
                    f"turn={turn_idx}/{t['num_turns']-1}<br>"
                    f"depth={depth}<br>"
                    f"phase={phase}<br>"
                    f"latency={latency} ms<br>"
                    f"ttft={ttft} ms<br>"
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
    parser.add_argument("jsonl", type=Path, help="aiperf JSONL results file")
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
    out.write_text(html)
    print(f"Written: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
