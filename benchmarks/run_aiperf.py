# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

"""aiperf concurrency sweep for AIC.

Drives an ``aiperf profile`` run for each concurrency level, collects the
per-run ``profile_export_aiperf.json`` artifacts, and writes a single
aggregated CSV that ``plot_aiperf.py`` can chart.

aiperf (https://github.com/SemiAnalysisAI/aiperf) must be installed:
    pip install aiperf

Usage (against a running vLLM endpoint):

    # Synthetic ISL=550 / OSL=200 sweep across default concurrencies:
    python benchmarks/run_aiperf.py \\
        --endpoint http://localhost:8000 \\
        --model openai/gpt-oss-120b \\
        --arm vram_only \\
        --out logs/manual/results/aiperf-vram_only.csv

    # ShareGPT dataset (requires HF download on first run):
    python benchmarks/run_aiperf.py \\
        --endpoint http://localhost:8000 \\
        --model openai/gpt-oss-120b \\
        --arm kvd_v2 \\
        --dataset sharegpt \\
        --tokenizer openai/gpt-oss-120b \\
        --out logs/manual/results/aiperf-kvd_v2.csv

    # Then chart both arms together:
    python benchmarks/plot_aiperf.py \\
        --input logs/manual/results/ \\
        --output-dir logs/manual/plots/aiperf/

CSV columns written:
    arm, concurrency, request_count,
    ttft_ms_avg, ttft_ms_p50, ttft_ms_p95, ttft_ms_p99,
    itl_ms_avg, itl_ms_p50, itl_ms_p95, itl_ms_p99,
    e2e_ms_avg, e2e_ms_p50, e2e_ms_p95, e2e_ms_p99,
    output_tok_tput_per_user, output_tok_tput_total, request_tput
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

_FIELDNAMES = [
    "arm",
    "concurrency",
    "request_count",
    "ttft_ms_avg",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "ttft_ms_p99",
    "itl_ms_avg",
    "itl_ms_p50",
    "itl_ms_p95",
    "itl_ms_p99",
    "e2e_ms_avg",
    "e2e_ms_p50",
    "e2e_ms_p95",
    "e2e_ms_p99",
    "output_tok_tput_per_user",
    "output_tok_tput_total",
    "request_tput",
]

# ---------------------------------------------------------------------------
# Metric extraction helpers
# ---------------------------------------------------------------------------


def _ckpt(msg: str) -> None:
    print(f"[aiperf] {msg}", flush=True)


def _to_ms(value: float | None, unit: str) -> float | str:
    """Convert a metric value to milliseconds. Returns '' when value is None."""
    if value is None:
        return ""
    if unit and unit.lower() == "s":
        return round(value * 1000, 3)
    return round(float(value), 3)  # already in ms


def _extract(data: dict, key: str, stat: str, unit_fallback: str = "ms") -> float | str:
    """Pull data[key][stat] from a JsonMetricResult dict and convert to ms."""
    metric = data.get(key)
    if not isinstance(metric, dict):
        return ""
    value = metric.get(stat)
    unit = metric.get("unit", unit_fallback)
    return _to_ms(value, unit)


def _extract_scalar(data: dict, key: str, stat: str = "avg") -> float | str:
    """Extract a plain scalar metric (no unit conversion needed)."""
    metric = data.get(key)
    if not isinstance(metric, dict):
        return ""
    value = metric.get(stat)
    return "" if value is None else round(float(value), 4)


def _parse_run_json(json_path: Path, arm: str, concurrency: int, request_count: int) -> dict:
    """Read a single-run profile_export_aiperf.json and return a CSV row dict."""
    try:
        data = json.loads(json_path.read_text())
    except Exception as exc:
        _ckpt(f"WARN: could not read {json_path}: {exc}")
        return _empty_row(arm, concurrency, request_count)

    # time_to_first_token and inter_token_latency are in ms in the JSON.
    # request_latency (e2e) is in seconds.
    return {
        "arm": arm,
        "concurrency": concurrency,
        "request_count": request_count,
        # TTFT (ms)
        "ttft_ms_avg": _extract(data, "time_to_first_token", "avg"),
        "ttft_ms_p50": _extract(data, "time_to_first_token", "p50"),
        "ttft_ms_p95": _extract(data, "time_to_first_token", "p95"),
        "ttft_ms_p99": _extract(data, "time_to_first_token", "p99"),
        # ITL (ms)
        "itl_ms_avg": _extract(data, "inter_token_latency", "avg"),
        "itl_ms_p50": _extract(data, "inter_token_latency", "p50"),
        "itl_ms_p95": _extract(data, "inter_token_latency", "p95"),
        "itl_ms_p99": _extract(data, "inter_token_latency", "p99"),
        # E2E request latency (convert s -> ms)
        "e2e_ms_avg": _extract(data, "request_latency", "avg", unit_fallback="s"),
        "e2e_ms_p50": _extract(data, "request_latency", "p50", unit_fallback="s"),
        "e2e_ms_p95": _extract(data, "request_latency", "p95", unit_fallback="s"),
        "e2e_ms_p99": _extract(data, "request_latency", "p99", unit_fallback="s"),
        # Throughput (plain scalars — no unit conversion)
        "output_tok_tput_per_user": _extract_scalar(data, "output_token_throughput_per_user"),
        "output_tok_tput_total": _extract_scalar(data, "output_token_throughput"),
        "request_tput": _extract_scalar(data, "request_throughput"),
    }


def _empty_row(arm: str, concurrency: int, request_count: int) -> dict:
    row: dict = {"arm": arm, "concurrency": concurrency, "request_count": request_count}
    for f in _FIELDNAMES:
        row.setdefault(f, "")
    return row


# ---------------------------------------------------------------------------
# aiperf invocation
# ---------------------------------------------------------------------------


def _run_aiperf(
    *,
    endpoint: str,
    model: str,
    arm: str,
    concurrency: int,
    request_count: int,
    dataset: str,
    isl: int,
    osl: int,
    tokenizer: str | None,
    streaming: bool,
    artifact_dir: Path,
    extra_aiperf_args: list[str],
) -> Path | None:
    """Run aiperf profile for a single concurrency point.

    Returns the path to the generated profile_export_aiperf.json, or None on
    failure.
    """
    cmd = [
        sys.executable, "-m", "aiperf", "profile",
        "--url", endpoint,
        "--model", model,
        "--concurrency", str(concurrency),
        "--request-count", str(request_count),
        "--endpoint-type", "chat",
        "--ui", "none",
        "--output-artifact-dir", str(artifact_dir),
    ]
    if streaming:
        cmd.append("--streaming")
    if dataset == "sharegpt":
        cmd += ["--public-dataset", "sharegpt"]
    else:
        # Synthetic prompts — no HF download needed.
        cmd += ["--isl", str(isl), "--osl", str(osl)]
    if tokenizer:
        cmd += ["--tokenizer", tokenizer]
    else:
        # Let the server report token counts to avoid a tokenizer download.
        cmd.append("--use-server-token-count")
    if extra_aiperf_args:
        cmd += extra_aiperf_args

    _ckpt(
        f"concurrency={concurrency}: aiperf profile -> {artifact_dir}"
    )
    _ckpt(f"  cmd: {' '.join(cmd)}")
    t0 = time.monotonic()
    result = subprocess.run(cmd, check=False)
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        _ckpt(f"  WARN: aiperf exited {result.returncode} after {elapsed:.0f}s (concurrency={concurrency})")
        return None

    json_path = artifact_dir / "profile_export_aiperf.json"
    if not json_path.is_file():
        _ckpt(f"  WARN: profile_export_aiperf.json not found in {artifact_dir}")
        return None

    _ckpt(f"  done in {elapsed:.0f}s -> {json_path}")
    return json_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="aiperf concurrency sweep: collect per-concurrency metrics into a single CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Endpoint
    p.add_argument("--endpoint", default="http://localhost:8000",
                   help="Base URL of the vLLM endpoint (e.g. http://localhost:8000)")
    p.add_argument("--model", required=True,
                   help="Served model name (passed to aiperf --model and vLLM /v1/chat/completions)")
    # Sweep knobs
    p.add_argument("--arm", default="vram_only",
                   help="Arm label written to the CSV (e.g. vram_only, kvd_v2)")
    p.add_argument("--concurrencies", default="1,2,4,8,16,32",
                   help="Comma-separated list of concurrency levels to sweep")
    p.add_argument("--request-count", type=int, default=200,
                   help="Number of requests per concurrency point")
    # Dataset
    p.add_argument("--dataset", choices=["sharegpt", "synthetic"], default="synthetic",
                   help="Dataset: 'sharegpt' downloads from HF; 'synthetic' uses --isl/--osl")
    p.add_argument("--isl", type=int, default=550,
                   help="Synthetic input sequence length (tokens); ignored when --dataset=sharegpt")
    p.add_argument("--osl", type=int, default=200,
                   help="Synthetic output sequence length (tokens); ignored when --dataset=sharegpt")
    # Tokenizer
    p.add_argument("--tokenizer", default="",
                   help="HuggingFace tokenizer ID or path (optional; falls back to --use-server-token-count)")
    p.add_argument("--streaming", action="store_true", default=True,
                   help="Enable streaming to measure TTFT/ITL (default: on)")
    p.add_argument("--no-streaming", dest="streaming", action="store_false")
    # Output
    p.add_argument("--out", required=True,
                   help="Path for the aggregated CSV (parent dir is created as needed)")
    p.add_argument("--artifact-base", default="",
                   help="Base dir for per-concurrency aiperf artifacts. Defaults to <out-parent>/aiperf-artifacts/")
    # Pass-through
    p.add_argument("rest", nargs=argparse.REMAINDER,
                   help="Extra arguments forwarded verbatim to aiperf (e.g. --isl-stddev 50)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    concurrencies = [int(c.strip()) for c in args.concurrencies.split(",") if c.strip()]
    if not concurrencies:
        print("ERROR: --concurrencies is empty", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    artifact_base = Path(args.artifact_base) if args.artifact_base else (out.parent / "aiperf-artifacts")
    artifact_base.mkdir(parents=True, exist_ok=True)

    tokenizer = args.tokenizer.strip() or None
    extra = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest  # strip leading '--' if present

    _ckpt(f"arm={args.arm}  endpoint={args.endpoint}  model={args.model}")
    _ckpt(f"concurrencies={concurrencies}  request_count={args.request_count}  dataset={args.dataset}")
    _ckpt(f"streaming={args.streaming}  tokenizer={tokenizer or '(server count)'}  out={out}")

    rows: list[dict] = []
    fail_count = 0

    for c in concurrencies:
        artifact_dir = artifact_base / f"aiperf-{args.arm}-c{c}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        json_path = _run_aiperf(
            endpoint=args.endpoint,
            model=args.model,
            arm=args.arm,
            concurrency=c,
            request_count=args.request_count,
            dataset=args.dataset,
            isl=args.isl,
            osl=args.osl,
            tokenizer=tokenizer,
            streaming=args.streaming,
            artifact_dir=artifact_dir,
            extra_aiperf_args=extra,
        )

        if json_path is None:
            rows.append(_empty_row(args.arm, c, args.request_count))
            fail_count += 1
        else:
            rows.append(_parse_run_json(json_path, args.arm, c, args.request_count))

    # Write aggregate CSV (overwrite if exists so retries produce a clean file).
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    _ckpt(f"wrote {len(rows)} rows -> {out}  (failures: {fail_count}/{len(concurrencies)})")
    return 1 if fail_count == len(concurrencies) else 0


if __name__ == "__main__":
    sys.exit(main())
