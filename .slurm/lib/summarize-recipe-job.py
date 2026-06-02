#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Summarize vLLM + LMCache recipe Slurm job artifacts under a report directory."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def parse_metadata(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ": " not in line or line.startswith("==="):
            continue
        key, _, val = line.partition(": ")
        if key and not key.startswith(" "):
            out[key.strip()] = val.strip()
    return out


def infer_recipe_name(report_dir: Path, meta: dict[str, str] | None = None) -> str:
    if meta:
        for key in ("RECIPE_NAME", "recipe_name", "IMAGE_NAME"):
            val = meta.get(key, "").strip()
            if val:
                return val
    name = report_dir.name
    for prefix in ("vllm-lmcache-hipfile-", "vllm-lmcache-nixl-", "vllm-radeon-"):
        if name.startswith(prefix):
            return prefix.rstrip("-")
    parent = report_dir.parent.name
    for prefix in ("vllm-lmcache-hipfile-", "vllm-lmcache-nixl-"):
        if parent.startswith(prefix):
            return prefix.rstrip("-")
    return "vllm-lmcache-hipfile"


def _meta_lmcache_io(meta: dict[str, str]) -> str | None:
    for key in ("VLH_LMCACHE_IO", "VLN_LMCACHE_IO", "KURT_LMCACHE_IO"):
        if meta.get(key):
            return meta[key]
    return None


def _meta_benchmark(meta: dict[str, str]) -> str:
    for key in ("VLH_BENCHMARK", "VLN_BENCHMARK", "KURT_BENCHMARK"):
        if meta.get(key):
            return meta[key].lower()
    return "unknown"


def parse_summary_txt(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for token in line.split():
            if "=" not in token:
                continue
            key, _, val = token.partition("=")
            if key:
                out[key.strip()] = val.strip()
    return out


def load_jsonl_record(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _int_or_none(val: Any) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _float_or_none(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _tok_per_s(tokens: int | None, seconds: float | None) -> float | None:
    if tokens is None or seconds is None or seconds <= 0:
        return None
    return tokens / seconds


def find_run_long_parallel_runs(report_dir: Path) -> list[Path]:
    root = report_dir / "run-long-parallel"
    if not root.is_dir():
        return []
    runs = sorted(root.glob("*/worker-*.jsonl"), key=lambda p: p.stat().st_mtime)
    if runs:
        latest = runs[0].parent
        return sorted(latest.glob("worker-*.jsonl"))
    return []


def find_run_long_serial(report_dir: Path) -> Path | None:
    p = report_dir / "run-long.jsonl"
    return p if p.is_file() else None


def _req_id_prefix(req_id: str) -> str:
    """Match vLLM LMCache req_id suffix to OpenAI completion id."""
    if "-" in req_id:
        parts = req_id.rsplit("-", 1)
        if len(parts) == 2 and len(parts[1]) == 8:
            return parts[0]
    return req_id


def parse_lmcache_stores(path: Path) -> dict[str, dict[str, Any]]:
    """Per OpenAI id prefix: aggregated LMCache store stats from server.txt."""
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out

    store_re = re.compile(
        r"\[req_id=([^\]]+)\] Stored (\d+) out of total (\d+) tokens\. "
        r"size: ([0-9.]+) GB, cost ([0-9.]+) ms, throughput: ([0-9.]+) GB/s"
    )
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = store_re.search(line)
        if not m:
            continue
        full_id = m.group(1)
        prefix = _req_id_prefix(full_id)
        rec = out.setdefault(
            prefix,
            {
                "engine_req_ids": [],
                "stored_tokens": 0,
                "store_events": 0,
                "store_cost_ms_sum": 0.0,
                "store_throughput_gbps_max": 0.0,
                "store_size_gb_sum": 0.0,
            },
        )
        if full_id not in rec["engine_req_ids"]:
            rec["engine_req_ids"].append(full_id)
        stored = int(m.group(2))
        rec["stored_tokens"] += stored
        rec["store_events"] += 1
        rec["store_cost_ms_sum"] += float(m.group(5))
        rec["store_throughput_gbps_max"] = max(
            rec["store_throughput_gbps_max"], float(m.group(6))
        )
        rec["store_size_gb_sum"] += float(m.group(4))
    return out


def parse_server_log(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "lmcache_requests": [],
        "lmcache_stores_by_prefix": {},
        "lmcache_zmq_timeouts": 0,
        "external_prefix_cache_hit_rate_last": None,
        "prefix_cache_hit_rate_last": None,
        "peak_prompt_throughput_tok_s": None,
        "peak_generation_throughput_tok_s": None,
    }
    if not path.is_file():
        return out

    peak_prefill = 0.0
    peak_gen = 0.0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(
            r"Reqid: (\S+), Total tokens (\d+), .* LMCache hit tokens: (\d+)",
            line,
        )
        if m:
            rid = m.group(1)
            out["lmcache_requests"].append(
                {
                    "req_id": rid,
                    "req_id_prefix": _req_id_prefix(rid),
                    "total_tokens": int(m.group(2)),
                    "lmcache_hit_tokens": int(m.group(3)),
                }
            )
        if "Timeout occurred for rank" in line and "zmq_transport" in line:
            out["lmcache_zmq_timeouts"] += 1
        m = re.search(r"External prefix cache hit rate: ([0-9.]+)%", line)
        if m:
            out["external_prefix_cache_hit_rate_last"] = float(m.group(1))
        m = re.search(r"Prefix cache hit rate: ([0-9.]+)%", line)
        if m:
            out["prefix_cache_hit_rate_last"] = float(m.group(1))
        m = re.search(
            r"Avg prompt throughput: ([0-9.]+) tokens/s, "
            r"Avg generation throughput: ([0-9.]+) tokens/s",
            line,
        )
        if m:
            peak_prefill = max(peak_prefill, float(m.group(1)))
            peak_gen = max(peak_gen, float(m.group(2)))
    if peak_prefill > 0:
        out["peak_prompt_throughput_tok_s"] = peak_prefill
    if peak_gen > 0:
        out["peak_generation_throughput_tok_s"] = peak_gen

    out["lmcache_stores_by_prefix"] = parse_lmcache_stores(path)
    return out


def parse_nvme_blk_tsv(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path),
        "present": False,
        "io_ops": 0,
        "read_ops": 0,
        "write_ops": 0,
        "read_bytes": 0,
        "write_bytes": 0,
        "total_bytes": 0,
        "sectors": 0,
    }
    if not path.is_file():
        return out
    out["present"] = True
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return out
    if not lines:
        return out

    start = 0
    if lines[0].startswith("ts_ns"):
        start = 1

    for line in lines[start:]:
        if not line.strip() or line.startswith("Attaching"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            rwbs = parts[1].strip()
            nbytes = int(parts[2])
        except ValueError:
            continue
        out["io_ops"] += 1
        out["total_bytes"] += nbytes
        if len(parts) >= 4:
            try:
                out["sectors"] += int(parts[3])
            except ValueError:
                pass
        rw = rwbs.lstrip().upper()
        if rw.startswith("R"):
            out["read_ops"] += 1
            out["read_bytes"] += nbytes
        elif rw.startswith("W"):
            out["write_ops"] += 1
            out["write_bytes"] += nbytes
    return out


def parse_vfs_dir_tsv(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path),
        "present": False,
        "read_ops": 0,
        "write_ops": 0,
        "read_bytes": 0,
        "write_bytes": 0,
        "total_bytes": 0,
        "duration_ns_sum": 0,
        "duration_ns_max": 0,
    }
    if not path.is_file():
        return out
    out["present"] = True
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return out
    if not lines:
        return out

    start = 0
    if lines[0].startswith("ts_begin_ns"):
        start = 1

    for line in lines[start:]:
        if not line.strip() or line.startswith("Attaching"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        try:
            dur = int(parts[2])
            op = parts[3].strip().upper()
            ret_bytes = int(parts[6])
        except ValueError:
            continue
        nbytes = max(ret_bytes, 0)
        out["total_bytes"] += nbytes
        out["duration_ns_sum"] += dur
        out["duration_ns_max"] = max(out["duration_ns_max"], dur)
        if op == "READ":
            out["read_ops"] += 1
            out["read_bytes"] += nbytes
        elif op == "WRITE":
            out["write_ops"] += 1
            out["write_bytes"] += nbytes
    return out


def _load_nvme_smart_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(data, dict) and len(data) == 1:
        inner = next(iter(data.values()))
        if isinstance(inner, dict):
            return inner
    return data if isinstance(data, dict) else {}


def parse_nvme_smart_delta(
    start_path: Path, end_path: Path
) -> dict[str, Any]:
    out: dict[str, Any] = {"present": False}
    if not start_path.is_file() or not end_path.is_file():
        return out
    start = _load_nvme_smart_json(start_path)
    end = _load_nvme_smart_json(end_path)
    if not start or not end:
        return out

    out["present"] = True
    out["start_path"] = str(start_path)
    out["end_path"] = str(end_path)

    for key in (
        "data_units_read",
        "data_units_written",
        "host_read_commands",
        "host_write_commands",
        "power_cycles",
        "power_on_hours",
        "unsafe_shutdowns",
        "media_errors",
        "num_err_log_entries",
    ):
        s = _int_or_none(start.get(key))
        e = _int_or_none(end.get(key))
        if s is not None and e is not None:
            out[f"delta_{key}"] = e - s
            out[f"end_{key}"] = e

    # nvme-cli reports 512-byte units for data_units_* .
    dur = out.get("delta_data_units_written")
    if dur is not None:
        out["delta_bytes_written"] = dur * 512
    dur_r = out.get("delta_data_units_read")
    if dur_r is not None:
        out["delta_bytes_read"] = dur_r * 512
    return out


def parse_lmcache_metrics(report_dir: Path) -> dict[str, float]:
    """Extract LMCache Prometheus counters (worker port file)."""
    metrics: dict[str, float] = {}
    for path in sorted(report_dir.glob("lmcache_internal_api_*_metrics.txt")):
        if "_6990_" not in path.name and "_6991_" not in path.name:
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("lmcache:") or "{" not in line:
                continue
            name = line.split("{", 1)[0].strip()
            try:
                val = float(line.rsplit(" ", 1)[-1])
            except ValueError:
                continue
            if name.endswith("_total") or name.endswith("_created"):
                metrics[name] = val
    return metrics


def _resolve_server_path(report_dir: Path) -> Path:
    server_path = report_dir / "server.txt"
    if not server_path.is_file():
        server_path = report_dir / "logs" / "server.txt"
    if not server_path.is_file():
        alt = sorted(report_dir.glob("**/server.txt"))
        server_path = alt[0] if alt else server_path
    return server_path


def summarize_gutenberg_worker(
    path: Path, lmcache_stores: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    data = load_jsonl_record(path)
    if not data:
        return {"file": str(path), "error": "invalid or empty jsonl"}
    m = re.search(r"worker-(\d+)", path.name)
    worker = int(m.group(1)) if m else None
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    prompt_tokens = _int_or_none(usage.get("prompt_tokens"))
    completion_tokens = _int_or_none(usage.get("completion_tokens"))
    wall = _float_or_none(data.get("client_wall_time_seconds"))
    ttft = _float_or_none(data.get("client_ttft_seconds"))
    decode_s = (wall - ttft) if wall is not None and ttft is not None else None

    completion_id = data.get("id")
    engine: dict[str, Any] | None = None
    if isinstance(completion_id, str):
        engine = lmcache_stores.get(completion_id)
        if engine is None:
            engine = lmcache_stores.get(_req_id_prefix(completion_id))

    row: dict[str, Any] = {
        "file": path.name,
        "worker": data.get("run_long_worker", worker),
        "seed": data.get("run_long_seed"),
        "book": data.get("run_long_book"),
        "completion_id": completion_id,
        "http_status": data.get("http_status"),
        "client_wall_time_seconds": wall,
        "client_ttft_seconds": ttft,
        "client_decode_seconds": decode_s,
        "iteration": data.get("run_long_iteration"),
        "context": data.get("run_long_context"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": _int_or_none(usage.get("total_tokens")),
        "prefill_tok_per_s": _tok_per_s(prompt_tokens, ttft),
        "decode_tok_per_s": _tok_per_s(completion_tokens, decode_s),
        "e2e_tok_per_s": _tok_per_s(
            (prompt_tokens or 0) + (completion_tokens or 0), wall
        ),
    }
    if engine:
        row["lmcache_stored_tokens"] = engine.get("stored_tokens")
        row["lmcache_store_events"] = engine.get("store_events")
        row["lmcache_store_throughput_gbps_max"] = engine.get(
            "store_throughput_gbps_max"
        )
        row["lmcache_store_cost_ms_sum"] = engine.get("store_cost_ms_sum")
    return row


def build_summary(report_dir: Path) -> dict[str, Any]:
    meta = parse_metadata(report_dir / "metadata.txt")
    meta.update(parse_summary_txt(report_dir / "summary.txt"))
    server_path = _resolve_server_path(report_dir)

    bench = _meta_benchmark(meta)
    summary: dict[str, Any] = {
        "recipe": infer_recipe_name(report_dir, meta),
        "job_id": meta.get("SLURM_JOB_ID"),
        "hostname": meta.get("hostname"),
        "rocm_arch": meta.get("ROCM_ARCH"),
        "benchmark": bench,
        "model": meta.get("MODEL") or meta.get("served_model_name"),
        "lmcache_io": _meta_lmcache_io(meta),
        "nvme_base": meta.get("VLH_NVME_BASE"),
        "gutenberg_data_root": meta.get("BOOK_DATA_ROOT")
        or meta.get("VLH_GUTENBERG_DATA_ROOT"),
        "build_rc": _int_or_none(meta.get("BUILD_RC")),
        "run_rc": _int_or_none(meta.get("RUN_RC")),
        "phase_rc": _int_or_none(meta.get("PHASE_RC")),
        "t_build": meta.get("T_BUILD"),
        "t_run": meta.get("T_RUN"),
        "container": meta.get("CONTAINER_NAME"),
        "workers": _int_or_none(meta.get("VLH_RUN_LONG_WORKERS")),
        "iterations_per_worker": _int_or_none(meta.get("VLH_RUN_LONG_ITERATIONS")),
        "nvme_blk_device": meta.get("nvme_blk dev_path") or meta.get("nvme_blk disk_name"),
    }

    server = parse_server_log(server_path)
    summary["server"] = server
    lmcache_stores = server.get("lmcache_stores_by_prefix") or {}

    if bench.startswith("gutenberg"):
        parallel = find_run_long_parallel_runs(report_dir)
        if parallel:
            workers = [
                summarize_gutenberg_worker(p, lmcache_stores) for p in parallel
            ]
            if not summary.get("model"):
                for w in workers:
                    # model lives in jsonl on older runs only via separate field — skip
                    pass
            summary["gutenberg"] = {
                "mode": "parallel",
                "run_dir": str(parallel[0].parent),
                "workers": workers,
            }
            if not summary.get("model"):
                for p in parallel:
                    rec = load_jsonl_record(p)
                    if rec and rec.get("model"):
                        summary["model"] = rec["model"]
                        break
            walls = [
                w["client_wall_time_seconds"]
                for w in workers
                if isinstance(w.get("client_wall_time_seconds"), (int, float))
            ]
            ttfts = [
                w["client_ttft_seconds"]
                for w in workers
                if isinstance(w.get("client_ttft_seconds"), (int, float))
            ]
            if walls:
                summary["gutenberg"]["wall_time_seconds"] = {
                    "min": min(walls),
                    "max": max(walls),
                    "mean": sum(walls) / len(walls),
                }
            if ttfts:
                summary["gutenberg"]["ttft_seconds"] = {
                    "min": min(ttfts),
                    "max": max(ttfts),
                    "mean": sum(ttfts) / len(ttfts),
                }
            statuses = [w.get("http_status") for w in workers]
            summary["gutenberg"]["all_http_ok"] = all(s == 200 for s in statuses)
        else:
            serial = find_run_long_serial(report_dir)
            if serial:
                rec = summarize_gutenberg_worker(serial, lmcache_stores)
                summary["gutenberg"] = {"mode": "serial", "workers": [rec]}

    nvme_tsv = report_dir / "nvme_blk_io.tsv"
    vfs_tsv = report_dir / "vfs_dir_io.tsv"
    summary["bpftrace"] = {
        "nvme_block": parse_nvme_blk_tsv(nvme_tsv),
        "vfs_cgroup": parse_vfs_dir_tsv(vfs_tsv),
        "nvme_bpftrace_log": str(report_dir / "nvme_blk_io.bpftrace.log")
        if (report_dir / "nvme_blk_io.bpftrace.log").is_file()
        else None,
        "vfs_bpftrace_log": str(report_dir / "vfs_dir_io.bpftrace.log")
        if (report_dir / "vfs_dir_io.bpftrace.log").is_file()
        else None,
    }

    smart_start = report_dir / "nvme_smart_log_job_start.json"
    smart_end = report_dir / "nvme_smart_log_job_end.json"
    summary["nvme_smart"] = parse_nvme_smart_delta(smart_start, smart_end)

    summary["lmcache_metrics"] = parse_lmcache_metrics(report_dir)

    if summary.get("gutenberg", {}).get("workers"):
        ok = summary["gutenberg"].get("all_http_ok")
        if ok is False:
            summary["status"] = "benchmark_failed"
        elif summary.get("run_rc") == 0:
            summary["status"] = "ok"
        else:
            summary["status"] = "run_failed"
    elif summary.get("run_rc") == 0:
        summary["status"] = "ok"
    else:
        summary["status"] = "failed"

    return summary


def _fmt_num(v: Any, digits: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _fmt_bytes(n: int | float | None) -> str:
    if n is None:
        return "—"
    n = float(n)
    for unit, div in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{int(n)} B"


def format_summary_md(data: dict[str, Any]) -> str:
    recipe = data.get("recipe") or "vllm-lmcache-hipfile"
    lines: list[str] = [
        f"# {recipe} job results",
        "",
        f"- **Status:** {data.get('status', 'unknown')}",
        f"- **Job ID:** {data.get('job_id', '?')}",
        f"- **Host:** {data.get('hostname', '?')}",
        f"- **Model:** {data.get('model', '?')}",
        f"- **Benchmark:** {data.get('benchmark', '?')}",
        f"- **Build / run / phase RC:** "
        f"{data.get('build_rc')} / {data.get('run_rc')} / {data.get('phase_rc')}",
        f"- **LMCache I/O:** {data.get('lmcache_io', '?')}",
        f"- **Storage (VLH_NVME_BASE):** `{data.get('nvme_base', '?')}`",
        f"- **NVMe block device:** {data.get('nvme_blk_device', '—')}",
        f"- **Run window:** {data.get('t_run', '?')}",
        "",
    ]

    gut = data.get("gutenberg")
    if isinstance(gut, dict) and gut.get("workers"):
        lines.append("## Gutenberg benchmark")
        lines.append("")
        lines.append(
            f"- Mode: **{gut.get('mode')}** ({len(gut['workers'])} worker record(s))"
        )
        wt = gut.get("wall_time_seconds")
        if isinstance(wt, dict):
            lines.append(
                f"- Client wall time (s): min **{wt['min']:.1f}**, "
                f"max **{wt['max']:.1f}**, mean **{wt['mean']:.1f}**"
            )
        tt = gut.get("ttft_seconds")
        if isinstance(tt, dict):
            lines.append(
                f"- Client TTFT (s): min **{tt['min']:.2f}**, "
                f"max **{tt['max']:.2f}**, mean **{tt['mean']:.2f}**"
            )
        lines.append(f"- All HTTP 200: **{gut.get('all_http_ok', '?')}**")
        lines.append("")
        lines.append(
            "| W | Book | HTTP | Wall (s) | TTFT (s) | Prefill tok/s | "
            "Decode tok/s | Prompt | Compl | LMCache store tok |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for w in gut["workers"]:
            if "error" in w:
                lines.append(f"| ? | — | — | — | — | — | — | — | — | {w['error']} |")
                continue
            lines.append(
                f"| {w.get('worker', '?')} | {w.get('book', '?')} | "
                f"{w.get('http_status', '?')} | "
                f"{_fmt_num(w.get('client_wall_time_seconds'), 1)} | "
                f"{_fmt_num(w.get('client_ttft_seconds'), 2)} | "
                f"{_fmt_num(w.get('prefill_tok_per_s'), 0)} | "
                f"{_fmt_num(w.get('decode_tok_per_s'), 1)} | "
                f"{w.get('prompt_tokens', '?')} | "
                f"{w.get('completion_tokens', '?')} | "
                f"{w.get('lmcache_stored_tokens', '—')} |"
            )
        lines.append("")

    srv = data.get("server")
    if isinstance(srv, dict):
        lines.append("## Engine / LMCache (from server.txt)")
        lines.append("")
        reqs = srv.get("lmcache_requests") or []
        if reqs:
            total_tok = sum(r["total_tokens"] for r in reqs)
            total_hits = sum(r["lmcache_hit_tokens"] for r in reqs)
            lines.append(f"- LMCache adapter requests: **{len(reqs)}**")
            lines.append(f"- Total prompt tokens (adapter): **{total_tok}**")
            lines.append(f"- LMCache hit tokens (sum): **{total_hits}**")
        stores = srv.get("lmcache_stores_by_prefix") or {}
        if stores:
            total_stored = sum(s.get("stored_tokens", 0) for s in stores.values())
            lines.append(f"- LMCache store events (requests): **{len(stores)}**")
            lines.append(f"- LMCache stored tokens (sum): **{total_stored}**")
        if srv.get("peak_prompt_throughput_tok_s") is not None:
            lines.append(
                f"- Peak avg prompt throughput: "
                f"**{srv['peak_prompt_throughput_tok_s']:.0f} tok/s**"
            )
        if srv.get("peak_generation_throughput_tok_s") is not None:
            lines.append(
                f"- Peak avg generation throughput: "
                f"**{srv['peak_generation_throughput_tok_s']:.1f} tok/s**"
            )
        if srv.get("external_prefix_cache_hit_rate_last") is not None:
            lines.append(
                f"- External prefix cache hit rate (last): "
                f"**{srv['external_prefix_cache_hit_rate_last']}%**"
            )
        zmq = srv.get("lmcache_zmq_timeouts", 0)
        if zmq:
            lines.append(f"- LMCache ZMQ timeouts (recovered): **{zmq}**")
        lines.append("")

    bpf = data.get("bpftrace")
    if isinstance(bpf, dict):
        nvme = bpf.get("nvme_block") or {}
        vfs = bpf.get("vfs_cgroup") or {}
        if nvme.get("present") or vfs.get("present"):
            lines.append("## Storage I/O (bpftrace)")
            lines.append("")
        if nvme.get("present"):
            lines.append(
                f"- **NVMe block** (`{Path(nvme.get('path', '')).name}`): "
                f"**{nvme.get('io_ops', 0)}** ops, "
                f"read **{_fmt_bytes(nvme.get('read_bytes'))}** "
                f"({nvme.get('read_ops', 0)} ops), "
                f"write **{_fmt_bytes(nvme.get('write_bytes'))}** "
                f"({nvme.get('write_ops', 0)} ops), "
                f"total **{_fmt_bytes(nvme.get('total_bytes'))}**"
            )
        elif bpf.get("nvme_bpftrace_log"):
            lines.append(
                "- **NVMe block:** trace enabled but no TSV "
                f"(see `{Path(bpf['nvme_bpftrace_log']).name}`)"
            )
        if vfs.get("present"):
            dur_ms = (vfs.get("duration_ns_sum") or 0) / 1e6
            lines.append(
                f"- **VFS cgroup** (`{Path(vfs.get('path', '')).name}`): "
                f"read **{_fmt_bytes(vfs.get('read_bytes'))}** "
                f"({vfs.get('read_ops', 0)} ops), "
                f"write **{_fmt_bytes(vfs.get('write_bytes'))}** "
                f"({vfs.get('write_ops', 0)} ops), "
                f"VFS time **{dur_ms:.1f} ms** (sum)"
            )
        if nvme.get("present") or vfs.get("present") or bpf.get("nvme_bpftrace_log"):
            lines.append("")

    smart = data.get("nvme_smart")
    if isinstance(smart, dict) and smart.get("present"):
        lines.append("## NVMe SMART (job delta)")
        lines.append("")
        if smart.get("delta_bytes_written") is not None:
            lines.append(
                f"- Data written (SMART): **{_fmt_bytes(smart['delta_bytes_written'])}** "
                f"({smart.get('delta_data_units_written', '?')} × 512 B units)"
            )
        if smart.get("delta_bytes_read") is not None:
            lines.append(
                f"- Data read (SMART): **{_fmt_bytes(smart['delta_bytes_read'])}**"
            )
        for key in ("delta_host_read_commands", "delta_host_write_commands"):
            if smart.get(key) is not None:
                label = key.replace("delta_", "").replace("_", " ")
                lines.append(f"- {label}: **{smart[key]}**")
        lines.append("")

    metrics = data.get("lmcache_metrics")
    if isinstance(metrics, dict) and metrics:
        lines.append("## LMCache metrics (snapshot)")
        lines.append("")
        for key in sorted(metrics):
            if any(
                x in key
                for x in (
                    "retrieve",
                    "store",
                    "hit_tokens",
                    "miss_tokens",
                    "hit_rate",
                    "stored_tokens",
                    "vllm_hit",
                )
            ):
                lines.append(f"- `{key}`: {metrics[key]}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "report_dir",
        type=Path,
        help="Job report directory (JOB_ROOT/report or .slurm/logs/<recipe>-<id>)",
    )
    p.add_argument(
        "--recipe-name",
        default="",
        help="Override recipe name in summary (default: infer from report path)",
    )
    p.add_argument(
        "--print",
        action="store_true",
        help="Print markdown summary to stdout",
    )
    args = p.parse_args()
    report_dir = args.report_dir.resolve()
    if not report_dir.is_dir():
        print(f"error: not a directory: {report_dir}", file=sys.stderr)
        return 1

    data = build_summary(report_dir)
    if args.recipe_name.strip():
        data["recipe"] = args.recipe_name.strip()
    md = format_summary_md(data)
    json_path = report_dir / "results-summary.json"
    md_path = report_dir / "results-summary.md"
    json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")

    if args.print:
        print(md)
    else:
        print(f"wrote {md_path}", file=sys.stderr)
        print(f"wrote {json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
