#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Publish RocJITSU emulated benchmark history and HTML.

Consumes one run's vLLM bench JSON files, merges them into a durable JSONL
history, and renders a static GitHub Pages index.html.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_POINT_RE = re.compile(r"^emu-(?P<model>.+)-isl(?P<isl>\d+)-osl(?P<osl>\d+)-c(?P<conc>\d+)\.json$")


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _parse_result(path: Path) -> dict[str, Any]:
    match = _POINT_RE.match(path.name)
    if not match:
        raise ValueError(f"unexpected result filename: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "name": f"isl{match.group('isl')}/osl{match.group('osl')}/c{match.group('conc')}",
        "model_tag": match.group("model"),
        "input_len": int(match.group("isl")),
        "output_len": int(match.group("osl")),
        "concurrency": int(match.group("conc")),
        "num_prompts": int(payload.get("num_prompts", 0) or 0),
        "mean_ttft_ms": float(payload["mean_ttft_ms"]),
        "mean_tpot_ms": float(payload["mean_tpot_ms"]),
        "output_throughput": float(payload["output_throughput"]),
    }


def _build_record(args: argparse.Namespace) -> dict[str, Any]:
    results = sorted(Path(args.results_dir).glob("emu-*.json"))
    if not results:
        raise FileNotFoundError(f"no emulated result JSONs found under {args.results_dir}")

    points = sorted(
        (_parse_result(path) for path in results),
        key=lambda point: (point["input_len"], point["output_len"], point["concurrency"]),
    )
    model_tag = points[0]["model_tag"]
    run_at = args.run_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return {
        "schema_version": 1,
        "run_at": run_at,
        "run_date": run_at[:10],
        "run_id": args.run_id,
        "sha": args.sha,
        "repo_url": args.repo_url,
        "workflow_path": args.workflow_path,
        "model": args.model,
        "profile_pack": args.profile_pack,
        "model_tag": model_tag,
        "points": points,
    }


def _merge_history(history: list[dict[str, Any]], record: dict[str, Any]) -> list[dict[str, Any]]:
    merged = []
    for row in history:
        if record["run_id"] and row.get("run_id") == record["run_id"]:
            continue
        merged.append(row)
    merged.append(record)
    merged.sort(key=lambda row: (row.get("run_at", ""), row.get("run_id", "")))
    return merged


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _fmt_metric(metric: str, value: float) -> str:
    if metric == "output_throughput":
        return f"{value:,.1f} tok/s"
    return f"{value:,.1f} ms"


def _fmt_delta(metric: str, current: float, previous: float | None) -> str:
    if previous in (None, 0):
        return "—"
    delta = (current - previous) / previous * 100.0
    klass = "flat"
    if delta > 0.05:
        klass = "up" if metric == "output_throughput" else "down"
    elif delta < -0.05:
        klass = "down" if metric == "output_throughput" else "up"
    sign = "+" if delta >= 0 else ""
    return f'<span class="delta {klass}">{sign}{delta:.1f}%</span>'


def _run_url(record: dict[str, Any]) -> str | None:
    repo_url = record.get("repo_url")
    run_id = record.get("run_id")
    if not repo_url or not run_id:
        return None
    return f"{repo_url}/actions/runs/{run_id}"


def _commit_url(record: dict[str, Any]) -> str | None:
    repo_url = record.get("repo_url")
    sha = record.get("sha")
    if not repo_url or not sha:
        return None
    return f"{repo_url}/commit/{sha}"


def _build_html(history: list[dict[str, Any]]) -> str:
    latest = history[-1]
    previous = history[-2] if len(history) > 1 else None
    previous_points = {point["name"]: point for point in previous.get("points", [])} if previous else {}

    latest_cards = []
    for point in latest["points"]:
        prev_point = previous_points.get(point["name"], {})
        latest_cards.append(
            f"""
      <section class="card">
        <h2>{html.escape(point["name"])}</h2>
        <dl>
          <div><dt>TTFT</dt><dd>{_fmt_metric("mean_ttft_ms", point["mean_ttft_ms"])} {_fmt_delta("mean_ttft_ms", point["mean_ttft_ms"], prev_point.get("mean_ttft_ms"))}</dd></div>
          <div><dt>TPOT</dt><dd>{_fmt_metric("mean_tpot_ms", point["mean_tpot_ms"])} {_fmt_delta("mean_tpot_ms", point["mean_tpot_ms"], prev_point.get("mean_tpot_ms"))}</dd></div>
          <div><dt>Throughput</dt><dd>{_fmt_metric("output_throughput", point["output_throughput"])} {_fmt_delta("output_throughput", point["output_throughput"], prev_point.get("output_throughput"))}</dd></div>
        </dl>
      </section>"""
        )

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for record in reversed(history):
        for point in record["points"]:
            grouped[point["name"]].append((record, point))

    history_sections = []
    for name in sorted(grouped, key=lambda label: (
        int(label.split("/")[0][3:]),
        int(label.split("/")[1][3:]),
        int(label.split("/")[2][1:]),
    )):
        rows = []
        for record, point in grouped[name]:
            commit_url = _commit_url(record)
            run_url = _run_url(record)
            sha = html.escape((record.get("sha") or "")[:7] or "—")
            sha_html = f'<a href="{commit_url}">{sha}</a>' if commit_url else sha
            date_html = html.escape(record.get("run_date") or record.get("run_at", "—"))
            run_html = f'<a href="{run_url}">run</a>' if run_url else "—"
            rows.append(
                f"""<tr>
              <td>{date_html}</td>
              <td>{sha_html}</td>
              <td>{run_html}</td>
              <td>{_fmt_metric("mean_ttft_ms", point["mean_ttft_ms"])}</td>
              <td>{_fmt_metric("mean_tpot_ms", point["mean_tpot_ms"])}</td>
              <td>{_fmt_metric("output_throughput", point["output_throughput"])}</td>
            </tr>"""
            )
        history_sections.append(
            f"""
      <section class="history-block">
        <h2>{html.escape(name)}</h2>
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Commit</th>
              <th>Workflow</th>
              <th>TTFT</th>
              <th>TPOT</th>
              <th>Throughput</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </section>"""
        )

    latest_commit_url = _commit_url(latest)
    latest_commit = html.escape((latest.get("sha") or "")[:7] or "—")
    latest_commit_html = f'<a href="{latest_commit_url}">{latest_commit}</a>' if latest_commit_url else latest_commit
    latest_run_url = _run_url(latest)
    latest_run_html = f'<a href="{latest_run_url}">workflow run</a>' if latest_run_url else "workflow run"
    model = html.escape(latest.get("model", ""))
    profile_pack = html.escape(latest.get("profile_pack", ""))
    sweep_points = ", ".join(
        f"<code>{html.escape(point['name'])}</code>"
        for point in latest["points"]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RocJITSU Emulated Performance</title>
  <style>
    :root {{
      --bg: #0d1117;
      --surface: #161b22;
      --border: #30363d;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --good: #3fb950;
      --bad: #f85149;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); padding: 2rem 1.25rem 4rem; }}
    main {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ margin: 0 0 0.5rem; font-size: 2rem; }}
    h2 {{ margin: 0 0 1rem; font-size: 1.1rem; }}
    p, li {{ color: var(--muted); line-height: 1.6; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .lede {{ max-width: 820px; margin-bottom: 1.5rem; }}
    .meta {{ margin-bottom: 2rem; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
    .card, .history-block, .config {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem; }}
    dl {{ margin: 0; display: grid; gap: 0.85rem; }}
    dt {{ color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.25rem; }}
    dd {{ margin: 0; font-size: 1.15rem; font-weight: 600; }}
    .delta {{ margin-left: 0.4rem; font-size: 0.9rem; }}
    .delta.up {{ color: var(--good); }}
    .delta.down {{ color: var(--bad); }}
    .delta.flat {{ color: var(--muted); }}
    .config ul {{ margin: 0.75rem 0 0; padding-left: 1.1rem; }}
    .history {{ display: grid; gap: 1rem; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 0.75rem; border-top: 1px solid var(--border); text-align: left; }}
    th {{ color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; }}
    @media (max-width: 720px) {{
      body {{ padding: 1.25rem 0.75rem 3rem; }}
      th, td {{ padding: 0.6rem 0.45rem; font-size: 0.9rem; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>RocJITSU Emulated Performance</h1>
    <p class="lede">
      Daily CPU-only replays of the current ROCm AIC emulator pack, published to GitHub Pages with durable history on <code>gh-pages</code>.
      This is intended to track the upcoming RocJITSU performance envelope before dedicated hardware lanes are available.
    </p>
    <p class="meta">
      Latest run: {html.escape(latest["run_date"])} · commit {latest_commit_html} · {latest_run_html}
    </p>

    <section class="config">
      <h2>Tracked configuration</h2>
      <ul>
        <li>Model: <code>{model}</code></li>
        <li>Profile pack: <code>{profile_pack}</code></li>
        <li>Sweep points: {sweep_points}</li>
      </ul>
    </section>

    <h2 style="margin-top:2rem">Latest nightly metrics</h2>
    <section class="cards">
      {''.join(latest_cards)}
    </section>

    <h2>History</h2>
    <section class="history">
      {''.join(history_sections)}
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="Directory containing emu-*.json files")
    parser.add_argument("--history", required=True, help="Path to the durable JSONL history file")
    parser.add_argument("--output-dir", required=True, help="Directory to write index.html into")
    parser.add_argument("--sha", default="", help="Git SHA for this run")
    parser.add_argument("--run-id", default="", help="GitHub Actions run id")
    parser.add_argument("--run-at", default="", help="UTC timestamp for this run")
    parser.add_argument("--repo-url", default="https://github.com/ROCm/rocm-aic")
    parser.add_argument("--workflow-path", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile-pack", required=True)
    args = parser.parse_args()

    history_path = Path(args.history)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = _load_history(history_path)
    record = _build_record(args)
    merged = _merge_history(history, record)
    _write_history(history_path, merged)
    (output_dir / "index.html").write_text(_build_html(merged), encoding="utf-8")


if __name__ == "__main__":
    main()
