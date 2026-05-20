#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""ROCm AIC Prometheus textfile exporter for vLLM + LMCache host stats.

Currently exports LMCache KV inventory (per-model file counts and chunk
bytes), filesystem free space on the data mount, and a **Hits per KV file**
histogram from ``chunk_hashes_*.jsonl`` (on-disk ``.data`` files only;
deleted chunks are excluded from the histogram universe).

Use ``--prometheus-textfile`` to write metrics for the node_exporter
textfile collector (same pattern as ``rocm_icms_stack_versions.prom``).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DTYPE_SUFFIXES = ("bfloat16", "half", "float16", "float", "fp16", "fp32")
_DATA_FILE_RE = re.compile(
    r"@([^@]+)@(?:"
    + "|".join(_DTYPE_SUFFIXES)
    + r")\.data$",
    re.IGNORECASE,
)

_METRIC_PREFIX = "rocm_aic"


def _norm_jsonl_hash(raw: str) -> str:
    s = raw.strip().lower()
    return s[2:] if s.startswith("0x") else s


def _aliases_for_jsonl_hash(h: str) -> set[str]:
    """Map a chunk_hashes JSONL entry to filename tag alias(es)."""
    s = h.strip().lower()
    if s.startswith("0x"):
        raw = s[2:]
        v = int(s, 16)
    else:
        raw = _norm_jsonl_hash(s)
        v = int(raw, 16)
    aliases = {raw}
    if v >= 2**63:
        v -= 2**64
    if v < 0:
        aliases.add("-" + format((2**64 + v) & (2**64 - 1), "x"))
    return aliases


def _parse_data_filename(name: str) -> tuple[str, str, str] | None:
    """Return (model_name, hash_tag, dtype) from an LMCache ``.data`` basename."""
    if not name.endswith(".data"):
        return None
    stem = name[: -len(".data")]
    parts = stem.rsplit("@", 4)
    if len(parts) != 5:
        return None
    model, _world, _worker, hash_tag, dtype = parts
    if dtype.lower() not in _DTYPE_SUFFIXES:
        return None
    return model, hash_tag.lower(), dtype.lower()


@dataclass(frozen=True)
class KvDiskInventory:
    """On-disk LMCache chunk files under the KV directory."""

    tag_to_path: dict[str, Path]
    files_by_model: dict[str, int]
    bytes_by_model: dict[str, int]
    chunk_bytes_total: int
    unrecognized_files: int


def _scan_kv_inventory(kv_dir: Path) -> KvDiskInventory:
    tag_to_path: dict[str, Path] = {}
    files_by_model: Counter[str] = Counter()
    bytes_by_model: Counter[str] = Counter()
    chunk_bytes_total = 0
    unrecognized = 0
    if not kv_dir.is_dir():
        return KvDiskInventory(
            tag_to_path=tag_to_path,
            files_by_model=dict(files_by_model),
            bytes_by_model=dict(bytes_by_model),
            chunk_bytes_total=0,
            unrecognized_files=0,
        )
    for path in kv_dir.iterdir():
        if not path.is_file() or path.suffix != ".data":
            continue
        parsed = _parse_data_filename(path.name)
        if parsed is None:
            m = _DATA_FILE_RE.search(path.name)
            if not m:
                unrecognized += 1
                continue
            hash_tag = m.group(1).lower()
            model = "unknown"
        else:
            model, hash_tag, _dtype = parsed
        tag_to_path[hash_tag] = path
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        files_by_model[model] += 1
        bytes_by_model[model] += size
        chunk_bytes_total += size
    return KvDiskInventory(
        tag_to_path=tag_to_path,
        files_by_model=dict(files_by_model),
        bytes_by_model=dict(bytes_by_model),
        chunk_bytes_total=chunk_bytes_total,
        unrecognized_files=unrecognized,
    )


def _scan_disk_kv_files(kv_dir: Path) -> dict[str, Path]:
    """Return hash tag -> path for each ``.data`` file currently on disk."""
    return _scan_kv_inventory(kv_dir).tag_to_path


@dataclass(frozen=True)
class FsUsage:
    total_bytes: int
    used_bytes: int
    free_bytes: int
    path: Path


def _filesystem_usage(path: Path) -> FsUsage:
    resolved = path.resolve()
    usage = shutil.disk_usage(resolved)
    return FsUsage(
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
        path=resolved,
    )


def _load_hit_counts(
    stats_glob: str,
    disk_tags: set[str],
) -> tuple[Counter[str], int, int]:
    """Count stat mentions that resolve to a tag present on disk."""
    hits: Counter[str] = Counter()
    orphan_mentions = 0
    lookup_rows = 0
    for stats_path in sorted(glob.glob(stats_glob)):
        with open(stats_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                lookup_rows += 1
                for h in rec.get("chunk_hashes") or []:
                    if not isinstance(h, str):
                        continue
                    resolved: str | None = None
                    for alias in _aliases_for_jsonl_hash(h):
                        if alias in disk_tags:
                            resolved = alias
                            break
                    if resolved is None:
                        orphan_mentions += 1
                        continue
                    hits[resolved] += 1
    return hits, orphan_mentions, lookup_rows


def kv_block_hit_histogram(
    hit_counts: Counter[str], universe: set[str]
) -> dict[str, int]:
    hist: dict[str, int] = {str(i): 0 for i in range(11)}
    hist[">10"] = 0
    for tag in universe:
        h = hit_counts.get(tag, 0)
        if h > 10:
            hist[">10"] += 1
        else:
            hist[str(h)] += 1
    return hist


@dataclass(frozen=True)
class ChunkHitSummary:
    data_root: Path
    kv_dir: Path
    stats_dir: Path
    disk_file_count: int
    hit_file_count: int
    lookup_rows: int
    stats_files: int
    orphan_stat_mentions: int
    kv_block_hit_histogram: dict[str, int]
    hit_mention_sum: int
    files_by_model: dict[str, int] = field(default_factory=dict)
    bytes_by_model: dict[str, int] = field(default_factory=dict)
    chunk_bytes_total: int = 0
    unrecognized_kv_files: int = 0
    filesystem: FsUsage | None = None


def collect_chunk_hit_summary(
    *,
    data_root: Path,
    kv_subdir: str = "lmcache",
    stats_subdir: str = "lmcache_chunk_stats",
) -> ChunkHitSummary:
    kv_dir = data_root / kv_subdir
    stats_dir = data_root / stats_subdir
    stats_glob = str(stats_dir / "chunk_hashes_*.jsonl")

    inventory = _scan_kv_inventory(kv_dir)
    disk = inventory.tag_to_path
    fs_usage = _filesystem_usage(kv_dir)

    disk_tags = set(disk)
    hits: Counter[str] = Counter()
    orphan_mentions = 0
    lookup_rows = 0
    if disk_tags:
        hits, orphan_mentions, lookup_rows = _load_hit_counts(stats_glob, disk_tags)
    hist = kv_block_hit_histogram(hits, disk_tags) if disk_tags else {
        str(i): 0 for i in range(11)
    } | {">10": 0}
    hit_file_count = sum(1 for t in disk_tags if hits.get(t, 0) > 0)

    return ChunkHitSummary(
        data_root=data_root.resolve(),
        kv_dir=kv_dir.resolve(),
        stats_dir=stats_dir.resolve(),
        disk_file_count=len(disk),
        hit_file_count=hit_file_count,
        lookup_rows=lookup_rows,
        stats_files=len(glob.glob(stats_glob)),
        orphan_stat_mentions=orphan_mentions,
        kv_block_hit_histogram=hist,
        hit_mention_sum=sum(hits.values()),
        files_by_model=inventory.files_by_model,
        bytes_by_model=inventory.bytes_by_model,
        chunk_bytes_total=inventory.chunk_bytes_total,
        unrecognized_kv_files=inventory.unrecognized_files,
        filesystem=fs_usage,
    )


def _prom_label_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "")


def _label_set(extra: dict[str, str] | None) -> str:
    if not extra:
        return ""
    parts = [
        f'{k}="{_prom_label_escape(v)}"'
        for k, v in sorted(extra.items())
        if v
    ]
    return "{" + ",".join(parts) + "}" if parts else ""


def format_prometheus_textfile(
    summary: ChunkHitSummary,
    *,
    extra_labels: dict[str, str] | None = None,
) -> str:
    """Exposition text for node_exporter ``collector.textfile.directory``."""
    labels = _label_set(extra_labels)
    hist = summary.kv_block_hit_histogram
    lines: list[str] = []

    def emit(name: str, help_text: str, typ: str, body: list[str]) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {typ}")
        lines.extend(body)

    emit(
        f"{_METRIC_PREFIX}_kv_disk_files",
        "Number of LMCache .data files on disk (histogram universe).",
        "gauge",
        [f"{_METRIC_PREFIX}_kv_disk_files{labels} {summary.disk_file_count}"],
    )
    emit(
        f"{_METRIC_PREFIX}_kv_files_with_hits",
        "On-disk .data files with at least one chunk_hashes mention.",
        "gauge",
        [f"{_METRIC_PREFIX}_kv_files_with_hits{labels} {summary.hit_file_count}"],
    )
    emit(
        f"{_METRIC_PREFIX}_chunk_stats_lookup_rows",
        "chunk_hashes JSONL rows scanned.",
        "gauge",
        [f"{_METRIC_PREFIX}_chunk_stats_lookup_rows{labels} {summary.lookup_rows}"],
    )
    emit(
        f"{_METRIC_PREFIX}_orphan_stat_mentions",
        "Stat mentions for hashes with no matching on-disk file.",
        "gauge",
        [
            f"{_METRIC_PREFIX}_orphan_stat_mentions{labels} "
            f"{summary.orphan_stat_mentions}"
        ],
    )
    emit(
        f"{_METRIC_PREFIX}_hit_mention_sum",
        "Sum of per-file hit counts (on-disk files only).",
        "gauge",
        [f"{_METRIC_PREFIX}_hit_mention_sum{labels} {summary.hit_mention_sum}"],
    )

    files_by_model_lines = [
        f"{_METRIC_PREFIX}_kv_files"
        f"{_label_set({**(extra_labels or {}), 'model_name': model})} {count}"
        for model, count in sorted(summary.files_by_model.items())
    ]
    emit(
        f"{_METRIC_PREFIX}_kv_files",
        "Number of LMCache .data chunk files on disk.",
        "gauge",
        files_by_model_lines or [f"{_METRIC_PREFIX}_kv_files{labels} 0"],
    )
    chunk_bytes_lines = [
        f"{_METRIC_PREFIX}_kv_chunk_bytes"
        f"{_label_set({**(extra_labels or {}), 'model_name': model})} "
        f"{summary.bytes_by_model.get(model, 0)}"
        for model in sorted(summary.files_by_model)
    ]
    emit(
        f"{_METRIC_PREFIX}_kv_chunk_bytes",
        "Total size in bytes of LMCache .data chunk files.",
        "gauge",
        chunk_bytes_lines or [f"{_METRIC_PREFIX}_kv_chunk_bytes{labels} 0"],
    )

    emit(
        f"{_METRIC_PREFIX}_kv_chunk_bytes_total",
        "Total bytes of all LMCache .data files (all models).",
        "gauge",
        [f"{_METRIC_PREFIX}_kv_chunk_bytes_total{labels} {summary.chunk_bytes_total}"],
    )

    if summary.filesystem is not None:
        fs = summary.filesystem
        fs_lbl = _label_set(
            {
                **(extra_labels or {}),
                "mount_path": str(fs.path),
            }
        )
        emit(
            f"{_METRIC_PREFIX}_data_fs_total_bytes",
            "Total capacity of the filesystem hosting the LMCache data path.",
            "gauge",
            [f"{_METRIC_PREFIX}_data_fs_total_bytes{fs_lbl} {fs.total_bytes}"],
        )
        emit(
            f"{_METRIC_PREFIX}_data_fs_used_bytes",
            "Used bytes on the filesystem hosting the LMCache data path.",
            "gauge",
            [f"{_METRIC_PREFIX}_data_fs_used_bytes{fs_lbl} {fs.used_bytes}"],
        )
        emit(
            f"{_METRIC_PREFIX}_data_fs_free_bytes",
            "Free bytes remaining on the filesystem hosting the LMCache data path.",
            "gauge",
            [f"{_METRIC_PREFIX}_data_fs_free_bytes{fs_lbl} {fs.free_bytes}"],
        )

    by_hit_lines: list[str] = []
    for bucket in [str(i) for i in range(11)] + [">10"]:
        lbl = _label_set({**(extra_labels or {}), "hit_count": bucket})
        by_hit_lines.append(
            f"{_METRIC_PREFIX}_kv_files_by_hit_count{lbl} {hist.get(bucket, 0)}"
        )
    emit(
        f"{_METRIC_PREFIX}_kv_files_by_hit_count",
        "On-disk KV files grouped by how many stat mentions they received.",
        "gauge",
        by_hit_lines,
    )

    # Native histogram buckets (cumulative) for Grafana heatmap / histogram panels.
    bucket_lines: list[str] = []
    cumulative = 0
    for le in range(12):
        cumulative += hist.get(str(le), 0)
        le_lbl = _label_set({**(extra_labels or {}), "le": str(le)})
        bucket_lines.append(
            f"{_METRIC_PREFIX}_kv_file_hits_histogram_bucket{le_lbl} {cumulative}"
        )
    overflow = hist.get(">10", 0)
    cumulative += overflow
    inf_lbl = _label_set({**(extra_labels or {}), "le": "+Inf"})
    bucket_lines.append(
        f"{_METRIC_PREFIX}_kv_file_hits_histogram_bucket{inf_lbl} {cumulative}"
    )
    hist_labels = _label_set(extra_labels)
    bucket_lines.append(
        f"{_METRIC_PREFIX}_kv_file_hits_histogram_sum{hist_labels} "
        f"{summary.hit_mention_sum}"
    )
    bucket_lines.append(
        f"{_METRIC_PREFIX}_kv_file_hits_histogram_count{hist_labels} "
        f"{summary.disk_file_count}"
    )
    emit(
        f"{_METRIC_PREFIX}_kv_file_hits_histogram",
        "Distribution of stat mentions per on-disk KV file.",
        "histogram",
        bucket_lines,
    )

    lines.append(
        f"# {_METRIC_PREFIX} generated_at={int(time.time())} "
        f"data_root={summary.data_root}"
    )
    return "\n".join(lines) + "\n"


def write_prometheus_textfile(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def _format_bytes(n: int) -> str:
    if n >= 1024**4:
        return f"{n / 1024**4:.2f} TiB"
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n} B"


def _print_inventory(summary: ChunkHitSummary) -> None:
    print("\nLMCache KV directory inventory")
    print(f"Path = {summary.kv_dir}")
    if summary.files_by_model:
        print(f"{'Model':<48} {'Files':>10} {'Size':>14}")
        print("-" * 74)
        for model in sorted(summary.files_by_model):
            n = summary.files_by_model[model]
            sz = summary.bytes_by_model.get(model, 0)
            print(f"{model:<48} {n:>10} {_format_bytes(sz):>14}")
        print("-" * 74)
        print(
            f"{'TOTAL':<48} {summary.disk_file_count:>10} "
            f"{_format_bytes(summary.chunk_bytes_total):>14}"
        )
    else:
        print("No .data chunk files found.")
    if summary.unrecognized_kv_files:
        print(f"Unrecognized .data filenames = {summary.unrecognized_kv_files}")
    if summary.filesystem is not None:
        fs = summary.filesystem
        print(f"\nFilesystem {fs.path}")
        print(f"  Total = {_format_bytes(fs.total_bytes)}")
        print(f"  Used  = {_format_bytes(fs.used_bytes)}")
        print(f"  Free  = {_format_bytes(fs.free_bytes)}")


def _print_histogram(summary: ChunkHitSummary) -> None:
    _print_inventory(summary)
    hist = summary.kv_block_hit_histogram
    label_w = 12
    cnt_w = 10
    print("\nHits per on-disk KV file (.data)")
    print(f"Total on-disk files = {summary.disk_file_count}")
    print(f"Files with >= 1 stat mention = {summary.hit_file_count}")
    print(
        f"Stat lookup rows read = {summary.lookup_rows} "
        f"({summary.stats_files} jsonl file(s))"
    )
    if summary.orphan_stat_mentions:
        print(
            f"Stat mentions for deleted/missing files = "
            f"{summary.orphan_stat_mentions} (excluded from histogram)"
        )
    hdr = f"{'Hits':<{label_w}}{'Files':>{cnt_w}}"
    print(hdr)
    print("-" * len(hdr))

    _max_data_lines = 10
    _max_interior = _max_data_lines - 1
    interior: list[tuple[str, int, int]] = []
    for i in range(11):
        k = str(i)
        c = hist.get(k, 0)
        if c <= 0:
            continue
        row_label = "0 hits" if i == 0 else k
        interior.append((row_label, c, i))
    while len(interior) > _max_interior:
        interior.pop()
    for row_label, c, _ in interior:
        print(f"{row_label:<{label_w}}{c:>{cnt_w}d}")
    if interior:
        last_i = interior[-1][2]
        ov_sum = sum(hist.get(str(j), 0) for j in range(last_i + 1, 11)) + hist.get(
            ">10", 0
        )
        tail_label = f">{last_i}"
    else:
        ov_sum = sum(hist.get(str(j), 0) for j in range(11)) + hist.get(">10", 0)
        tail_label = ">10"
    print(f"{tail_label:<{label_w}}{ov_sum:>{cnt_w}d}")


def _default_data_root(recipe_root: Path) -> Path:
    host = os.environ.get("RADEON_HOST_DATA_ROOT", "").strip()
    if host:
        return Path(host)
    makefile_data = os.environ.get("DATA", "").strip()
    if makefile_data:
        return Path(makefile_data)
    return Path("/mnt/lmcache-nvme")


def _default_textfile_path() -> Path | None:
    for key in (
        "ROCM_AIC_EXPORTER_TEXTFILE",
        "RADEON_LMCACHE_CHUNK_HIST_TEXTFILE",
    ):
        v = os.environ.get(key, "").strip()
        if v:
            return Path(v)
    v = os.environ.get("ROCM_ICMS_TEXTFILE_DIR", "").strip()
    if v:
        return Path(v) / "rocm_aic_exporter.prom"
    return Path("/var/lib/prometheus/node-exporter/rocm_aic_exporter.prom")


def main() -> int:
    recipe_root = Path(__file__).resolve().parents[1]
    default_prom = _default_textfile_path()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "LMCache data root (host path). Default: RADEON_HOST_DATA_ROOT, "
            "then DATA, then /mnt/lmcache-nvme."
        ),
    )
    p.add_argument(
        "--kv-subdir",
        default=os.environ.get("RADEON_LMCACHE_KV_SUBDIR", "lmcache"),
        help="KV .data directory under data-root (default: lmcache).",
    )
    p.add_argument(
        "--stats-subdir",
        default=os.environ.get(
            "RADEON_LMCACHE_CHUNK_STATS_SUBDIR", "lmcache_chunk_stats"
        ),
        help="Chunk statistics directory under data-root.",
    )
    p.add_argument(
        "--prometheus-textfile",
        type=Path,
        nargs="?",
        const=default_prom,
        default=None,
        metavar="PATH",
        help=(
            "Write node_exporter textfile metrics (.prom). With no PATH, use "
            "ROCM_AIC_EXPORTER_TEXTFILE, ROCM_ICMS_TEXTFILE_DIR, or "
            "/var/lib/prometheus/node-exporter/rocm_aic_exporter.prom."
        ),
    )
    p.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Prometheus label on every metric (repeatable).",
    )
    p.add_argument(
        "--textfile-only",
        action="store_true",
        help="Only write --prometheus-textfile; no human histogram on stdout.",
    )
    p.add_argument(
        "--top",
        type=int,
        default=0,
        metavar="N",
        help="After the histogram, list top N files by hit count (0=off).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print JSON summary (stdout).",
    )
    args = p.parse_args()

    extra_labels: dict[str, str] = {}
    for item in args.label:
        if "=" not in item:
            print(f"error: --label must be KEY=VALUE, got {item!r}", file=sys.stderr)
            return 1
        key, val = item.split("=", 1)
        extra_labels[key.strip()] = val.strip()

    data_root = args.data_root or _default_data_root(recipe_root)
    if not (data_root / args.kv_subdir).is_dir():
        print(
            f"error: KV directory not found: {data_root / args.kv_subdir}",
            file=sys.stderr,
        )
        return 1

    summary = collect_chunk_hit_summary(
        data_root=data_root,
        kv_subdir=args.kv_subdir,
        stats_subdir=args.stats_subdir,
    )

    if summary.disk_file_count == 0:
        print(
            f"warning: no .data files under {summary.kv_dir}; "
            "hit histogram empty",
            file=sys.stderr,
        )

    if summary.stats_files == 0 and summary.disk_file_count > 0:
        print(
            f"warning: no chunk_hashes_*.jsonl under {summary.stats_dir}; "
            "histogram is all zero-hit files",
            file=sys.stderr,
        )

    prom_path = args.prometheus_textfile
    if prom_path is not None:
        body = format_prometheus_textfile(summary, extra_labels=extra_labels or None)
        write_prometheus_textfile(prom_path, body)
        print(f"wrote {prom_path}", file=sys.stderr)

    if args.json:
        print(
            json.dumps(
                {
                    "data_root": str(summary.data_root),
                    "kv_dir": str(summary.kv_dir),
                    "stats_dir": str(summary.stats_dir),
                    "disk_file_count": summary.disk_file_count,
                    "hit_file_count": summary.hit_file_count,
                    "lookup_rows": summary.lookup_rows,
                    "orphan_stat_mentions": summary.orphan_stat_mentions,
                    "hit_mention_sum": summary.hit_mention_sum,
                    "kv_block_hit_histogram": summary.kv_block_hit_histogram,
                    "files_by_model": summary.files_by_model,
                    "bytes_by_model": summary.bytes_by_model,
                    "chunk_bytes_total": summary.chunk_bytes_total,
                    "filesystem": (
                        {
                            "path": str(summary.filesystem.path),
                            "total_bytes": summary.filesystem.total_bytes,
                            "used_bytes": summary.filesystem.used_bytes,
                            "free_bytes": summary.filesystem.free_bytes,
                        }
                        if summary.filesystem
                        else None
                    ),
                    "prometheus_textfile": str(prom_path) if prom_path else None,
                },
                indent=2,
            )
        )

    if not args.textfile_only:
        print(
            f"rocm-aic-exporter: kv_dir={summary.kv_dir} "
            f"stats_dir={summary.stats_dir}",
            flush=True,
        )
        _print_histogram(summary)
        if args.top > 0:
            disk = _scan_disk_kv_files(summary.kv_dir)
            hits, _, _ = _load_hit_counts(
                str(summary.stats_dir / "chunk_hashes_*.jsonl"),
                set(disk),
            )
            print(f"\nTop {args.top} on-disk files by stat mentions")
            for tag, count in hits.most_common(args.top):
                print(f"  {count:8d}  {disk[tag].name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
