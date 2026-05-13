#!/usr/bin/env python3
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

"""Summarize LMCache chunk statistics JSON, hash jsonl, and Prometheus metrics.

Usage::

    python3 analyze_lmcache_chunk_stats.py /path/to/report_or_data_dir

Picks up files named ``lmcache_internal_api_*_chunk_statistics_status.json``,
``lmcache_internal_api_*_metrics.txt``, and ``**/chunk_hashes_*.jsonl`` under
the given directory.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


def _load_status_files(root: Path) -> list[dict]:
    out: list[dict] = []
    for p in sorted(root.rglob("lmcache_internal_api_*_chunk_statistics_status.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as e:
            print(f"(skip {p}: {e})", file=sys.stderr)
    return out


def _parse_prom_metrics(path: Path) -> dict[str, float]:
    """Last scalar value per bare metric name (no labels), for *_total/_sum/_count."""
    scalars: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z0-9_:]+)(?:\{[^}]*\})?\s+([eE0-9.+-]+)$", line)
        if not m:
            continue
        name, val_s = m.group(1), m.group(2)
        try:
            scalars[name] = float(val_s)
        except ValueError:
            continue
    return scalars


def _hist_mean_from_buckets(lines: list[str], metric_prefix: str) -> float | None:
    """Approximate mean from Prometheus histogram buckets (same labels as in file)."""
    buckets: list[tuple[float, float]] = []
    count = None
    sum_v = None
    for line in lines:
        line = line.strip()
        if f"{metric_prefix}_bucket{{" in line and "}" in line:
            mb = re.search(r'le="([^"]+)"}', line)
            mv = re.search(r"\}\s+([eE0-9.+-]+)\s*$", line)
            if mb and mv:
                le = mb.group(1)
                try:
                    upper = float("+Inf" if le == "+Inf" else le)
                except ValueError:
                    continue
                buckets.append((upper, float(mv.group(1))))
        elif line.startswith(f"{metric_prefix}_count{{") or line.startswith(
            f"{metric_prefix}_count "
        ):
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                try:
                    count = float(parts[1])
                except ValueError:
                    pass
        elif line.startswith(f"{metric_prefix}_sum{{") or line.startswith(
            f"{metric_prefix}_sum "
        ):
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                try:
                    sum_v = float(parts[1])
                except ValueError:
                    pass
    if count and sum_v and count > 0:
        return sum_v / count
    return None


def _analyze_jsonl(paths: list[Path]) -> dict:
    lines_n = 0
    lengths: list[int] = []
    all_hashes: list[str] = []
    seq_tuples: list[tuple[str, ...]] = []
    ts_first = None
    ts_last = None
    for jp in paths:
        with jp.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = float(row.get("timestamp", 0.0))
                if ts_first is None:
                    ts_first = ts
                ts_last = ts
                ch = row.get("chunk_hashes") or []
                if not isinstance(ch, list):
                    continue
                hstr = [str(x) for x in ch]
                lines_n += 1
                lengths.append(len(hstr))
                all_hashes.extend(hstr)
                seq_tuples.append(tuple(hstr))
    dur = (ts_last - ts_first) if ts_first is not None and ts_last else 0.0
    uniq = set(all_hashes)
    seq_counter = Counter(seq_tuples)
    distinct_patterns = len(seq_counter)
    redundant_lines = max(0, lines_n - distinct_patterns)
    most_common_seq = seq_counter.most_common(5)
    hash_freq = Counter(all_hashes).most_common(8)
    return {
        "jsonl_files": len(paths),
        "lines": lines_n,
        "duration_s": dur,
        "lengths_min": min(lengths) if lengths else 0,
        "lengths_max": max(lengths) if lengths else 0,
        "lengths_mean": sum(lengths) / len(lengths) if lengths else 0.0,
        "hash_references_total": len(all_hashes),
        "unique_hashes": len(uniq),
        "distinct_lookup_patterns": distinct_patterns,
        "redundant_identical_lookup_lines": redundant_lines,
        "top_repeated_sequences": most_common_seq,
        "top_single_hashes": hash_freq,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    status_list = _load_status_files(root)
    jsonl_paths = sorted(root.rglob("chunk_hashes_*.jsonl"))
    metrics_paths = sorted(root.rglob("lmcache_internal_api_*_metrics.txt"))

    print(f"=== LMCache chunk / KV IO summary ===\nroot: {root}\n")

    if status_list:
        s0 = status_list[0]
        print("--- chunk_statistics/status (first JSON) ---")
        print(f"  enabled: {s0.get('enabled')}")
        print(f"  total_requests: {s0.get('total_requests')}")
        print(f"  total_chunks: {s0.get('total_chunks')}")
        print(f"  unique_chunks: {s0.get('unique_chunks')}")
        print(f"  duplicate_chunks: {s0.get('duplicate_chunks')}")
        print(f"  reuse_rate: {s0.get('reuse_rate')}")
        t = s0.get("timing") or {}
        print("  timing (chunk-stats instrumentation):")
        for k in sorted(t.keys()):
            print(f"    {k}: {t[k]}")
        fh = s0.get("file_hash") or {}
        if fh:
            print("  file_hash log:")
            for k in sorted(fh.keys()):
                print(f"    {k}: {fh[k]}")
        aq = s0.get("async_queue") or {}
        if aq:
            print("  async_queue:")
            for k in sorted(aq.keys()):
                print(f"    {k}: {aq[k]}")
        if len(status_list) > 1:
            same = all(json.dumps(x, sort_keys=True) == json.dumps(s0, sort_keys=True) for x in status_list[1:])
            print(f"  ({len(status_list)} status JSON files; identical: {same})")
        print()

    if metrics_paths:
        mp = metrics_paths[0]
        text = mp.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        scalars = _parse_prom_metrics(mp)

        def pick(*names: str) -> None:
            for n in names:
                if n in scalars:
                    print(f"  {n}: {scalars[n]:g}")
                    return
            print(f"  {' / '.join(names)}: (not found)")

        print(f"--- Prometheus metrics (worker snapshot: {mp.name}) ---")
        pick("lmcache:num_store_requests_total")
        pick("lmcache:num_retrieve_requests_total")
        pick("lmcache:num_stored_tokens_total")
        pick("lmcache:num_hit_tokens_total")
        pick("lmcache:num_vllm_hit_tokens_total")
        pick("lmcache:retrieve_hit_rate")
        pick("lmcache:num_remote_read_bytes_total", "lmcache:num_remote_read_bytes")
        pick("lmcache:num_remote_write_bytes_total", "lmcache:num_remote_write_bytes")
        pick("lmcache:local_storage_usage")
        pick("lmcache:local_cache_usage")

        m_store = _hist_mean_from_buckets(lines, "lmcache:time_to_store")
        m_lk = _hist_mean_from_buckets(lines, "lmcache:time_to_lookup")
        m_spt = _hist_mean_from_buckets(lines, "lmcache:store_process_tokens_time")
        m_ret = _hist_mean_from_buckets(lines, "lmcache:time_to_retrieve")
        print("  histogram means (sum/count where present):")
        print(f"    time_to_store: {m_store}")
        print(f"    time_to_lookup: {m_lk}")
        print(f"    store_process_tokens_time: {m_spt}")
        print(f"    time_to_retrieve: {m_ret}")
        print()

    if jsonl_paths:
        j = _analyze_jsonl(jsonl_paths)
        print("--- chunk_hashes jsonl (per-lookup hash lists) ---")
        print(f"  files: {j['jsonl_files']}")
        print(f"  lines (lookup events): {j['lines']}")
        print(f"  time span: {j['duration_s']:.3f} s")
        if j["duration_s"] > 0 and j["lines"] > 0:
            print(f"  lookups/s (avg over span): {j['lines'] / j['duration_s']:.2f}")
        print(f"  chunk_hashes per line: min={j['lengths_min']} max={j['lengths_max']} mean={j['lengths_mean']:.2f}")
        print(f"  total hash references (sum of list lengths): {j['hash_references_total']}")
        print(f"  unique hash values: {j['unique_hashes']}")
        print(f"  distinct full hash-sequence patterns: {j['distinct_lookup_patterns']}")
        print(
            "  redundant lines (same full sequence as another line): "
            f"{j['redundant_identical_lookup_lines']}"
        )
        if j["top_repeated_sequences"]:
            print("  most common full sequences (count, length, first 3 hashes):")
            for seq, c in j["top_repeated_sequences"]:
                if c <= 1:
                    continue
                head = ", ".join(seq[:3])
                print(f"    count={c} len={len(seq)} head=[{head}, ...]")
        print("  most frequent single chunk hashes (value, count):")
        for h, c in j["top_single_hashes"]:
            print(f"    {h}: {c}")
        print()

    if not status_list and not jsonl_paths and not metrics_paths:
        print("No lmcache_internal_api_*_chunk_statistics_status.json, metrics, or chunk_hashes jsonl found.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
