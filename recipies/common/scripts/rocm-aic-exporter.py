#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""ROCm AIC Prometheus textfile exporter for vLLM + LMCache host stats.

Exports LMCache KV inventory (per-model file counts and chunk bytes),
NIXL static pool files (``obj_<slot>_<id>.bin``), filesystem free space on
the data mount, **Hits per KV file** (``.data`` mode) and **chunk hash lookup
frequency** histograms from ``chunk_hashes_*.jsonl`` (always parsed; NIXL
mode uses JSONL-only lookup metrics when no ``.data`` files exist), NFS client
byte totals per mount (via
``nfsiostat`` + ``/proc/self/mountstats``), ROCm/HIP version from
``hipconfig``, optional KFD AIS kernel I/O samples from bpftrace
(``kfd_ais_rw_file`` kprobe; see ``kfd_ais_rw.bt``), and optional hipFile
``ais-stats`` totals from a vLLM container via ``docker exec``.

Use ``--prometheus-textfile`` to write metrics for the node_exporter
textfile collector (same pattern as ``rocm_icms_stack_versions.prom``).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
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
_NIXL_POOL_FILE_RE = re.compile(r"^obj_\d+_[0-9a-fA-F]+\.bin$")

_METRIC_PREFIX = "rocm_aic"
_MOUNTSTATS_DEVICE_RE = re.compile(
    r"^device\s+(.+?)\s+mounted on\s+(.+?)\s+with fstype\s+(\S+)",
)
_MOUNTSTATS_OP_RE = re.compile(r"^\s+([A-Z][A-Z0-9_]+):\s+(.+)$")
_HIP_VERSION_RE = re.compile(
    r"^HIP\s+version:\s*(\S+)",
    re.IGNORECASE | re.MULTILINE,
)
_KFD_AIS_TRACE_LINE_RE = re.compile(
    r"^pid=(\d+) (READ|WRITE) size=(\d+) copied=(\d+) ret=(-?\d+) "
    r"dur_us=(\d+)$"
)
_KFD_AIS_LATENCY_HIST_BUCKETS_US: tuple[int, ...] = (
    1000,
    2500,
    5000,
    7500,
    10000,
    12500,
    15000,
    20000,
)
_AIS_STATS_LEVEL_RE = re.compile(r"^HipFile Stats Level:\s*(\d+)\s*$")
_AIS_STATS_VALUE_RE = re.compile(
    r"^(Total|Average) (Fastpath|Fallback) (Read|Write) "
    r"(Size \(B\)|Errors|Bandwidth \(GiB/s\)|Latency \(us\)):\s*([0-9.]+)\s*$"
)


def _is_nfs_client_fstype(fstype: str) -> bool:
    fs = fstype.lower()
    if fs == "nfsd":
        return False
    return fs.startswith("nfs")


def _norm_jsonl_hash(raw: str) -> str:
    s = raw.strip().lower()
    return s[2:] if s.startswith("0x") else s


def _aliases_for_jsonl_hash(h: str) -> set[str]:
    """Map a ``chunk_hashes`` JSONL entry to on-disk ``chunk_hash_hex`` tag(s).

    FileHashStrategy writes ``hex()`` of the 64-bit value with negatives
    converted to unsigned first. ``CacheEngineKey.to_string()`` uses
    ``f"{chunk_hash:x}"`` (signed Python int). Both must use the same
    ``pre_caching_hash_algorithm`` as storage (see
    ``lmcache-chunk-statistics-hash.patch``).
    """
    s = h.strip().lower()
    if s.startswith("0x"):
        unsigned = int(s, 16)
    else:
        try:
            unsigned = int(s, 16)
        except ValueError:
            unsigned = int(s, 10)
    signed = unsigned if unsigned < 2**64 // 2 else unsigned - 2**64
    tags = {f"{signed:x}"}
    if signed >= 0:
        tags.add(str(signed))
    return tags


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


@dataclass(frozen=True)
class NixlPoolInventory:
    """LMCache NIXL static pool objects under the KV directory."""

    file_count: int
    slots_used: int
    bytes_total: int
    bytes_on_disk: int


def _scan_nixl_pool_inventory(kv_dir: Path) -> NixlPoolInventory:
    """Return NIXL ``obj_*.bin`` slot counts and used bytes.

    Unused pool slots stay at 0 bytes when lazy ``ftruncate`` is enabled;
    ``bytes_total`` sums ``st_size`` only for slots that have been written.
    """
    file_count = 0
    slots_used = 0
    bytes_total = 0
    bytes_on_disk = 0
    if not kv_dir.is_dir():
        return NixlPoolInventory(
            file_count=0,
            slots_used=0,
            bytes_total=0,
            bytes_on_disk=0,
        )
    for path in kv_dir.iterdir():
        if not path.is_file():
            continue
        if not _NIXL_POOL_FILE_RE.match(path.name):
            continue
        file_count += 1
        try:
            st = path.stat()
        except OSError:
            continue
        if st.st_size <= 0:
            continue
        slots_used += 1
        bytes_total += st.st_size
        bytes_on_disk += st.st_blocks * 512
    return NixlPoolInventory(
        file_count=file_count,
        slots_used=slots_used,
        bytes_total=bytes_total,
        bytes_on_disk=bytes_on_disk,
    )


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


def _load_jsonl_hash_mentions(stats_glob: str) -> tuple[Counter[str], int]:
    """Count how often each chunk hash appears in ``chunk_hashes`` JSONL rows."""
    mentions: Counter[str] = Counter()
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
                    mentions[_norm_jsonl_hash(h)] += 1
    return mentions, lookup_rows


def _disk_hit_counts(
    mentions: Counter[str],
    disk_tags: set[str],
) -> tuple[Counter[str], int]:
    """Map JSONL hash mention counts onto on-disk ``.data`` hash tags."""
    hits: Counter[str] = Counter()
    orphan_mentions = 0
    for raw_hash, count in mentions.items():
        resolved: str | None = None
        for alias in _aliases_for_jsonl_hash(raw_hash):
            if alias in disk_tags:
                resolved = alias
                break
        if resolved is None:
            orphan_mentions += count
        else:
            hits[resolved] += count
    return hits, orphan_mentions


def _load_hit_counts(
    stats_glob: str,
    disk_tags: set[str],
) -> tuple[Counter[str], int, int]:
    """Count stat mentions that resolve to a tag present on disk."""
    mentions, lookup_rows = _load_jsonl_hash_mentions(stats_glob)
    if not disk_tags:
        return Counter(), sum(mentions.values()), lookup_rows
    hits, orphan_mentions = _disk_hit_counts(mentions, disk_tags)
    return hits, orphan_mentions, lookup_rows


def _empty_hit_bucket_histogram() -> dict[str, int]:
    return {str(i): 0 for i in range(11)} | {">10": 0}


_CHUNK_LOOKUP_TAIL_BUCKETS = ("11-20", "21-50", "51-100", ">100")
_CHUNK_LOOKUP_HISTOGRAM_LE = (20, 50, 100)


def _chunk_lookup_bucket_labels() -> tuple[str, ...]:
    return tuple(str(i) for i in range(11)) + _CHUNK_LOOKUP_TAIL_BUCKETS


def _empty_chunk_lookup_histogram() -> dict[str, int]:
    return {label: 0 for label in _chunk_lookup_bucket_labels()}


def _chunk_lookup_tail_bucket(count: int) -> str:
    if count <= 10:
        return str(count)
    if count <= 20:
        return "11-20"
    if count <= 50:
        return "21-50"
    if count <= 100:
        return "51-100"
    return ">100"


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


def chunk_hash_lookup_histogram(mentions: Counter[str]) -> dict[str, int]:
    """Histogram of unique chunk hashes by how often each appears in JSONL."""
    hist = _empty_chunk_lookup_histogram()
    for count in mentions.values():
        hist[_chunk_lookup_tail_bucket(count)] += 1
    return hist


@dataclass(frozen=True)
class NfsMountBytes:
    """Cumulative NFS client bytes for one mount (from mountstats)."""

    mount_point: str
    rx_bytes: int
    tx_bytes: int


@dataclass(frozen=True)
class NfsIoStats:
    """NFS observability: ``nfsiostat`` presence and per-mount byte totals."""

    nfsiostat_present: bool
    mounts: tuple[NfsMountBytes, ...]
    mountstats_path: Path
    nfsiostat_error: str | None = None


@dataclass(frozen=True)
class HipconfigStats:
    """ROCm/HIP version from ``hipconfig``."""

    hipconfig_present: bool
    rocm_version: str
    hipconfig_error: str | None = None


def _empty_kfd_ais_latency_histogram() -> dict[str, int]:
    buckets = {str(le): 0 for le in _KFD_AIS_LATENCY_HIST_BUCKETS_US}
    buckets["+Inf"] = 0
    return buckets


@dataclass(frozen=True)
class KfdAisRwSample:
    """One completed ``kfd_ais_rw_file`` transfer from bpftrace output."""

    pid: int
    direction: str
    size_bytes: int
    copied_bytes: int
    ret: int
    duration_us: int


@dataclass(frozen=True)
class KfdAisStats:
    """Aggregated KFD AIS bpftrace sample for one exporter scrape."""

    bpftrace_present: bool
    kprobe_attachable: bool
    sample_seconds: float
    skipped: bool
    bpftrace_error: str | None = None
    operations: dict[tuple[str, str], int] = field(default_factory=dict)
    bytes_by_direction: dict[str, int] = field(default_factory=dict)
    latency_us_histogram: dict[str, int] = field(
        default_factory=_empty_kfd_ais_latency_histogram
    )
    latency_us_sum: int = 0
    latency_us_count: int = 0


@dataclass(frozen=True)
class AisHipfilePathStats:
    """One hipFile path class (fastpath or fallback) and direction from ais-stats."""

    bytes_total: int = 0
    bandwidth_gibps: float = 0.0
    latency_us: float = 0.0
    errors_total: int = 0


@dataclass(frozen=True)
class AisHipfileStats:
    """hipFile ``ais-stats`` totals scraped from a vLLM container."""

    configured: bool
    skipped: bool
    docker_present: bool
    container: str = ""
    collect_error: str | None = None
    stats_level: int = 0
    fastpath_read: AisHipfilePathStats = field(default_factory=AisHipfilePathStats)
    fastpath_write: AisHipfilePathStats = field(default_factory=AisHipfilePathStats)
    fallback_read: AisHipfilePathStats = field(default_factory=AisHipfilePathStats)
    fallback_write: AisHipfilePathStats = field(default_factory=AisHipfilePathStats)


@dataclass(frozen=True)
class ExporterSnapshot:
    chunk: ChunkHitSummary
    nfs: NfsIoStats
    hip: HipconfigStats
    kfd_ais: KfdAisStats
    ais_hipfile: AisHipfileStats
    host_metrics_collected: bool = False


def _empty_nfs_stats(mountstats_path: Path) -> NfsIoStats:
    return NfsIoStats(
        nfsiostat_present=False,
        mounts=(),
        mountstats_path=mountstats_path,
        nfsiostat_error=None,
    )


def _empty_hip_stats() -> HipconfigStats:
    return HipconfigStats(
        hipconfig_present=False,
        rocm_version="",
        hipconfig_error=None,
    )


def _skipped_hip_stats() -> HipconfigStats:
    """PATH presence only; hipconfig not executed (--skip-hipconfig)."""
    return HipconfigStats(
        hipconfig_present=_hipconfig_path() is not None,
        rocm_version="",
        hipconfig_error="skipped",
    )


def _empty_kfd_ais_stats(*, skipped: bool = True) -> KfdAisStats:
    return KfdAisStats(
        bpftrace_present=_bpftrace_path() is not None,
        kprobe_attachable=False,
        sample_seconds=0.0,
        skipped=skipped,
        bpftrace_error=None,
    )


def _skipped_kfd_ais_stats() -> KfdAisStats:
    return KfdAisStats(
        bpftrace_present=_bpftrace_path() is not None,
        kprobe_attachable=False,
        sample_seconds=0.0,
        skipped=True,
        bpftrace_error="skipped",
    )


def _docker_path() -> str | None:
    return shutil.which("docker")


def _empty_ais_hipfile_stats(*, configured: bool = False) -> AisHipfileStats:
    return AisHipfileStats(
        configured=configured,
        skipped=False,
        docker_present=_docker_path() is not None,
    )


def _skipped_ais_hipfile_stats() -> AisHipfileStats:
    return AisHipfileStats(
        configured=False,
        skipped=True,
        docker_present=_docker_path() is not None,
        collect_error="skipped",
    )


@dataclass
class _AisHipfilePathBuilder:
    bytes_total: int = 0
    bandwidth_gibps: float = 0.0
    latency_us: float = 0.0
    errors_total: int = 0


def _ais_path_stats_from_builder(builder: _AisHipfilePathBuilder) -> AisHipfilePathStats:
    return AisHipfilePathStats(
        bytes_total=builder.bytes_total,
        bandwidth_gibps=builder.bandwidth_gibps,
        latency_us=builder.latency_us,
        errors_total=builder.errors_total,
    )


def parse_ais_stats_output(text: str) -> AisHipfileStats:
    """Parse ``ais-stats -i`` stdout into structured hipFile counters."""
    builders: dict[tuple[str, str], _AisHipfilePathBuilder] = {}
    stats_level = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        level_match = _AIS_STATS_LEVEL_RE.match(line)
        if level_match:
            stats_level = int(level_match.group(1))
            continue
        value_match = _AIS_STATS_VALUE_RE.match(line)
        if not value_match:
            continue
        kind, backend, direction, field_name, value_str = value_match.groups()
        key = (backend.lower(), direction.lower())
        builder = builders.setdefault(key, _AisHipfilePathBuilder())
        if kind == "Total" and field_name == "Size (B)":
            builder.bytes_total = int(float(value_str))
        elif kind == "Total" and field_name == "Errors":
            builder.errors_total = int(float(value_str))
        elif kind == "Average" and field_name == "Bandwidth (GiB/s)":
            builder.bandwidth_gibps = float(value_str)
        elif kind == "Average" and field_name == "Latency (us)":
            builder.latency_us = float(value_str)

    def path_stats(backend: str, direction: str) -> AisHipfilePathStats:
        builder = builders.get((backend, direction))
        if builder is None:
            return AisHipfilePathStats()
        return _ais_path_stats_from_builder(builder)

    return AisHipfileStats(
        configured=True,
        skipped=False,
        docker_present=True,
        stats_level=stats_level,
        fastpath_read=path_stats("fastpath", "read"),
        fastpath_write=path_stats("fastpath", "write"),
        fallback_read=path_stats("fallback", "read"),
        fallback_write=path_stats("fallback", "write"),
    )


def collect_ais_hipfile_stats_from_container(
    *,
    container: str,
    docker_bin: str | None = None,
    engine_match: str = "VLLM::EngineCor",
    ais_stats_cmd: str = "ais-stats",
    timeout_seconds: float = 15.0,
) -> AisHipfileStats:
    """Run ``ais-stats -i`` inside ``container`` via ``docker exec``."""
    name = container.strip()
    if not name:
        return _empty_ais_hipfile_stats()

    docker = docker_bin or _docker_path()
    if not docker:
        return AisHipfileStats(
            configured=True,
            skipped=False,
            docker_present=False,
            container=name,
            collect_error="docker not found in PATH",
        )

    inner = (
        f"{ais_stats_cmd} -p $(pgrep -f {shlex.quote(engine_match)} | head -1) -i"
    )
    cmd = [docker, "exec", name, "bash", "-lc", inner]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return AisHipfileStats(
            configured=True,
            skipped=False,
            docker_present=True,
            container=name,
            collect_error=str(exc),
        )

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    parsed = parse_ais_stats_output(combined)
    collect_error: str | None = None
    if proc.returncode != 0:
        err_text = (proc.stderr or proc.stdout or "").strip()
        collect_error = err_text or f"docker exec exited {proc.returncode}"
    elif "HipFile Stats Level" not in combined:
        collect_error = "ais-stats output did not contain HipFile Stats Level"
        return AisHipfileStats(
            configured=True,
            skipped=False,
            docker_present=True,
            container=name,
            collect_error=collect_error,
        )

    return AisHipfileStats(
        configured=True,
        skipped=False,
        docker_present=True,
        container=name,
        collect_error=collect_error,
        stats_level=parsed.stats_level,
        fastpath_read=parsed.fastpath_read,
        fastpath_write=parsed.fastpath_write,
        fallback_read=parsed.fallback_read,
        fallback_write=parsed.fallback_write,
    )


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
    chunk_lookup_histogram: dict[str, int] = field(
        default_factory=_empty_chunk_lookup_histogram
    )
    chunk_hash_mention_sum: int = 0
    unique_chunk_hashes: int = 0
    files_by_model: dict[str, int] = field(default_factory=dict)
    bytes_by_model: dict[str, int] = field(default_factory=dict)
    chunk_bytes_total: int = 0
    unrecognized_kv_files: int = 0
    filesystem: FsUsage | None = None
    nixl_pool: NixlPoolInventory = field(
        default_factory=lambda: NixlPoolInventory(0, 0, 0, 0)
    )


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
    nixl_pool = _scan_nixl_pool_inventory(kv_dir)
    disk = inventory.tag_to_path
    fs_usage = _filesystem_usage(kv_dir)

    disk_tags = set(disk)
    jsonl_mentions, lookup_rows = _load_jsonl_hash_mentions(stats_glob)
    if disk_tags:
        hits, orphan_mentions = _disk_hit_counts(jsonl_mentions, disk_tags)
    else:
        hits = Counter()
        orphan_mentions = sum(jsonl_mentions.values())
    hist = (
        kv_block_hit_histogram(hits, disk_tags)
        if disk_tags
        else _empty_hit_bucket_histogram()
    )
    chunk_lookup_hist = chunk_hash_lookup_histogram(jsonl_mentions)
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
        chunk_lookup_histogram=chunk_lookup_hist,
        chunk_hash_mention_sum=sum(jsonl_mentions.values()),
        unique_chunk_hashes=len(jsonl_mentions),
        files_by_model=inventory.files_by_model,
        bytes_by_model=inventory.bytes_by_model,
        chunk_bytes_total=inventory.chunk_bytes_total,
        unrecognized_kv_files=inventory.unrecognized_files,
        filesystem=fs_usage,
        nixl_pool=nixl_pool,
    )


def _nfsiostat_path() -> str | None:
    return shutil.which("nfsiostat")


def _hipconfig_path() -> str | None:
    return shutil.which("hipconfig")


def _bpftrace_path() -> str | None:
    return shutil.which("bpftrace")


def _timeout_path() -> str | None:
    return shutil.which("timeout")


def _kfd_ais_bpftrace_script_path() -> Path:
    return Path(__file__).resolve().with_name("kfd_ais_rw.bt")


def _default_kfd_ais_sample_seconds() -> float:
    raw = os.environ.get("ROCM_AIC_KFD_AIS_SAMPLE_SECONDS", "").strip()
    if not raw:
        return 10.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 10.0


def _kfd_ais_outcome(*, ret: int, size_bytes: int, copied_bytes: int) -> str:
    if ret != 0:
        return "error"
    if copied_bytes == size_bytes:
        return "success"
    if copied_bytes > 0:
        return "partial"
    # bpftrace often reads copied=0 on kretprobe even when ret=0 and I/O succeeded.
    if size_bytes > 0:
        return "success"
    return "error"


def _kfd_ais_bytes_for_sample(sample: KfdAisRwSample, outcome: str) -> int:
    if outcome != "success":
        return 0
    if sample.copied_bytes > 0:
        return sample.copied_bytes
    return sample.size_bytes


def _kfd_ais_latency_bucket(duration_us: int) -> str:
    for le in _KFD_AIS_LATENCY_HIST_BUCKETS_US:
        if duration_us <= le:
            return str(le)
    return "+Inf"


def parse_kfd_ais_bpftrace_lines(text: str) -> list[KfdAisRwSample]:
    """Parse bpftrace lines emitted by ``kfd_ais_rw.bt``."""
    samples: list[KfdAisRwSample] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _KFD_AIS_TRACE_LINE_RE.match(line)
        if match is None:
            continue
        pid_s, direction_raw, size_s, copied_s, ret_s, dur_s = match.groups()
        samples.append(
            KfdAisRwSample(
                pid=int(pid_s),
                direction=direction_raw.lower(),
                size_bytes=int(size_s),
                copied_bytes=int(copied_s),
                ret=int(ret_s),
                duration_us=int(dur_s),
            )
        )
    return samples


def summarize_kfd_ais_samples(
    samples: list[KfdAisRwSample],
    *,
    sample_seconds: float,
    bpftrace_present: bool,
    kprobe_attachable: bool,
    bpftrace_error: str | None = None,
    skipped: bool = False,
) -> KfdAisStats:
    operations: Counter[tuple[str, str]] = Counter()
    bytes_by_direction: Counter[str] = Counter()
    latency_hist = _empty_kfd_ais_latency_histogram()
    latency_us_sum = 0
    latency_us_count = 0

    for sample in samples:
        outcome = _kfd_ais_outcome(
            ret=sample.ret,
            size_bytes=sample.size_bytes,
            copied_bytes=sample.copied_bytes,
        )
        operations[(sample.direction, outcome)] += 1
        if outcome == "success":
            bytes_by_direction[sample.direction] += _kfd_ais_bytes_for_sample(
                sample, outcome
            )
            latency_us_sum += sample.duration_us
            latency_us_count += 1
            bucket = _kfd_ais_latency_bucket(sample.duration_us)
            latency_hist[bucket] += 1

    return KfdAisStats(
        bpftrace_present=bpftrace_present,
        kprobe_attachable=kprobe_attachable,
        sample_seconds=sample_seconds,
        skipped=skipped,
        bpftrace_error=bpftrace_error,
        operations=dict(operations),
        bytes_by_direction=dict(bytes_by_direction),
        latency_us_histogram=latency_hist,
        latency_us_sum=latency_us_sum,
        latency_us_count=latency_us_count,
    )


def _kfd_ais_kprobe_attachable() -> bool:
    bpf = _bpftrace_path()
    if not bpf:
        return False
    try:
        proc = subprocess.run(
            [bpf, "-l", "kprobe:*kfd_ais*"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "kprobe:kfd_ais_rw_file" in (proc.stdout or "")


def collect_kfd_ais_stats(
    *,
    sample_seconds: float,
    bpftrace_script: Path | None = None,
) -> KfdAisStats:
    """Sample ``kfd_ais_rw_file`` with bpftrace for ``sample_seconds``."""
    bpf = _bpftrace_path()
    script = bpftrace_script or _kfd_ais_bpftrace_script_path()
    kprobe_ok = _kfd_ais_kprobe_attachable() if bpf else False

    if sample_seconds <= 0:
        return summarize_kfd_ais_samples(
            [],
            sample_seconds=0.0,
            bpftrace_present=bpf is not None,
            kprobe_attachable=kprobe_ok,
            skipped=True,
        )

    if not bpf:
        return summarize_kfd_ais_samples(
            [],
            sample_seconds=sample_seconds,
            bpftrace_present=False,
            kprobe_attachable=False,
            bpftrace_error="bpftrace not found in PATH",
        )

    if not script.is_file():
        return summarize_kfd_ais_samples(
            [],
            sample_seconds=sample_seconds,
            bpftrace_present=True,
            kprobe_attachable=kprobe_ok,
            bpftrace_error=f"bpftrace script not found: {script}",
        )

    if not kprobe_ok:
        return summarize_kfd_ais_samples(
            [],
            sample_seconds=sample_seconds,
            bpftrace_present=True,
            kprobe_attachable=False,
            bpftrace_error="kprobe:kfd_ais_rw_file not available",
        )

    timeout_bin = _timeout_path()
    if timeout_bin is None:
        return summarize_kfd_ais_samples(
            [],
            sample_seconds=sample_seconds,
            bpftrace_present=True,
            kprobe_attachable=True,
            bpftrace_error="timeout not found in PATH (required for kfd_ais_rw.bt)",
        )

    sample_secs = max(1, int(sample_seconds))
    run_timeout = sample_secs + 15
    cmd = [
        timeout_bin,
        "--signal=TERM",
        f"{sample_secs}s",
        bpf,
        "-q",
        str(script),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=run_timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return summarize_kfd_ais_samples(
            [],
            sample_seconds=sample_seconds,
            bpftrace_present=True,
            kprobe_attachable=True,
            bpftrace_error=str(exc),
        )

    samples = parse_kfd_ais_bpftrace_lines(proc.stdout or "")
    err: str | None = None
    rc = proc.returncode
    stderr = (proc.stderr or "").strip()
    # timeout(1) exits 124 after SIGTERM; an idle sample window is normal.
    if rc not in (0, 124) and not samples:
        err = stderr or (proc.stdout or "").strip() or f"bpftrace exited {rc}"
    elif "ERROR:" in stderr:
        err = stderr

    return summarize_kfd_ais_samples(
        samples,
        sample_seconds=sample_seconds,
        bpftrace_present=True,
        kprobe_attachable=True,
        bpftrace_error=err,
    )


def _invoke_nfsiostat(*, interval: int = 1, count: int = 1) -> str | None:
    """Run ``nfsiostat`` once; return an error string on failure."""
    bin_path = _nfsiostat_path()
    if not bin_path:
        return "nfsiostat not found in PATH"
    try:
        proc = subprocess.run(
            [bin_path, str(interval), str(count)],
            capture_output=True,
            text=True,
            timeout=max(30, interval * count + 15),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return err or f"nfsiostat exited {proc.returncode}"
    return None


def _parse_mountstats_nfs(path: Path) -> list[NfsMountBytes]:
    """Parse cumulative NFS client bytes per mount from mountstats.

    Per-op lines follow the kernel layout documented in
    Documentation/filesystems/nfs/nfs-stats.txt (bytes_sent, bytes_recv).
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    mounts: list[NfsMountBytes] = []
    current_mount: str | None = None
    current_fstype: str | None = None
    rx = 0
    tx = 0

    def flush() -> None:
        nonlocal current_mount, current_fstype, rx, tx
        if current_mount is None or current_fstype is None:
            return
        if not _is_nfs_client_fstype(current_fstype):
            return
        mounts.append(
            NfsMountBytes(
                mount_point=current_mount,
                rx_bytes=rx,
                tx_bytes=tx,
            )
        )

    for raw in text.splitlines():
        dev = _MOUNTSTATS_DEVICE_RE.match(raw)
        if dev is not None:
            flush()
            current_mount = dev.group(2)
            current_fstype = dev.group(3)
            rx = 0
            tx = 0
            continue
        if current_mount is None:
            continue
        op = _MOUNTSTATS_OP_RE.match(raw)
        if op is None:
            continue
        fields = op.group(2).split()
        if len(fields) < 5:
            continue
        try:
            bytes_sent = int(fields[3])
            bytes_recv = int(fields[4])
        except ValueError:
            continue
        tx += bytes_sent
        rx += bytes_recv

    flush()
    mounts.sort(key=lambda m: m.mount_point)
    return mounts


def collect_nfs_io_stats(
    *,
    mountstats_path: Path | None = None,
    run_nfsiostat: bool = True,
) -> NfsIoStats:
    """Collect NFS byte totals; call ``nfsiostat`` when the binary exists."""
    mpath = mountstats_path or Path("/proc/self/mountstats")
    present = _nfsiostat_path() is not None
    nfs_err: str | None = None
    if present and run_nfsiostat:
        nfs_err = _invoke_nfsiostat()
    mounts = tuple(_parse_mountstats_nfs(mpath)) if mpath.is_file() else ()
    return NfsIoStats(
        nfsiostat_present=present,
        mounts=mounts,
        mountstats_path=mpath,
        nfsiostat_error=nfs_err,
    )


def _parse_hipconfig_version(text: str) -> str:
    m = _HIP_VERSION_RE.search(text)
    if m:
        return m.group(1).strip()
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("rocm version"):
            parts = line.split(":", 1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip()
    return ""


def collect_hipconfig_stats() -> HipconfigStats:
    """Run ``hipconfig`` and parse the HIP/ROCm version string."""
    bin_path = _hipconfig_path()
    if not bin_path:
        return HipconfigStats(
            hipconfig_present=False,
            rocm_version="",
            hipconfig_error="hipconfig not found in PATH",
        )
    try:
        proc = subprocess.run(
            [bin_path],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return HipconfigStats(
            hipconfig_present=True,
            rocm_version="",
            hipconfig_error=str(exc),
        )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    version = _parse_hipconfig_version(combined)
    err: str | None = None
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or (
            f"hipconfig exited {proc.returncode}"
        )
    elif not version:
        err = "hipconfig output did not contain HIP version"
        version = "unknown"
    return HipconfigStats(
        hipconfig_present=True,
        rocm_version=version,
        hipconfig_error=err,
    )


def collect_exporter_snapshot(
    *,
    data_root: Path,
    kv_subdir: str = "lmcache",
    stats_subdir: str = "lmcache_chunk_stats",
    mountstats_path: Path | None = None,
    run_nfsiostat: bool = True,
    collect_hip: bool = True,
    include_host_metrics: bool = False,
    kfd_ais_sample_seconds: float = 0.0,
    collect_kfd_ais: bool = True,
    kfd_ais_bpftrace_script: Path | None = None,
    ais_stats_container: str = "",
    collect_ais_stats: bool = True,
    ais_stats_engine_match: str = "VLLM::EngineCor",
) -> ExporterSnapshot:
    """Collect LMCache stats; NFS/hip/KFD AIS only when ``include_host_metrics``."""
    mpath = mountstats_path or Path("/proc/self/mountstats")

    kfd_ais: KfdAisStats
    if not include_host_metrics:
        kfd_ais = _empty_kfd_ais_stats()
    elif collect_kfd_ais and kfd_ais_sample_seconds > 0:
        # Sample KFD AIS before the slow chunk_hashes scan so traffic is more
        # likely to overlap the bpftrace window when the timer fires.
        kfd_ais = collect_kfd_ais_stats(
            sample_seconds=kfd_ais_sample_seconds,
            bpftrace_script=kfd_ais_bpftrace_script,
        )
    elif not collect_kfd_ais:
        kfd_ais = _skipped_kfd_ais_stats()
    else:
        kfd_ais = summarize_kfd_ais_samples(
            [],
            sample_seconds=0.0,
            bpftrace_present=_bpftrace_path() is not None,
            kprobe_attachable=_kfd_ais_kprobe_attachable(),
            skipped=True,
        )

    ais_hipfile: AisHipfileStats
    if not include_host_metrics:
        ais_hipfile = _empty_ais_hipfile_stats()
    elif not collect_ais_stats:
        ais_hipfile = _skipped_ais_hipfile_stats()
    elif ais_stats_container.strip():
        ais_hipfile = collect_ais_hipfile_stats_from_container(
            container=ais_stats_container,
            engine_match=ais_stats_engine_match,
        )
    else:
        ais_hipfile = _empty_ais_hipfile_stats()

    chunk = collect_chunk_hit_summary(
        data_root=data_root,
        kv_subdir=kv_subdir,
        stats_subdir=stats_subdir,
    )
    if not include_host_metrics:
        return ExporterSnapshot(
            chunk=chunk,
            nfs=_empty_nfs_stats(mpath),
            hip=_empty_hip_stats(),
            kfd_ais=kfd_ais,
            ais_hipfile=ais_hipfile,
            host_metrics_collected=False,
        )
    hip = collect_hipconfig_stats() if collect_hip else _skipped_hip_stats()
    return ExporterSnapshot(
        chunk=chunk,
        nfs=collect_nfs_io_stats(
            mountstats_path=mpath,
            run_nfsiostat=run_nfsiostat,
        ),
        hip=hip,
        kfd_ais=kfd_ais,
        ais_hipfile=ais_hipfile,
        host_metrics_collected=True,
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
    snapshot: ExporterSnapshot,
    *,
    extra_labels: dict[str, str] | None = None,
    include_host_metrics: bool = False,
) -> str:
    """Exposition text for node_exporter ``collector.textfile.directory``.

    NFS and hipconfig series are included only when
    ``include_host_metrics`` is true (set when ``--prometheus-textfile`` runs).
    """
    summary = snapshot.chunk
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

    nixl = summary.nixl_pool
    emit(
        f"{_METRIC_PREFIX}_nixl_pool_present",
        "1 when NIXL obj_*.bin pool files exist under the LMCache KV path.",
        "gauge",
        [
            f"{_METRIC_PREFIX}_nixl_pool_present{labels} "
            f"{1 if nixl.file_count > 0 else 0}"
        ],
    )
    emit(
        f"{_METRIC_PREFIX}_nixl_pool_files",
        "Number of NIXL static pool files (obj_<slot>_<id>.bin).",
        "gauge",
        [f"{_METRIC_PREFIX}_nixl_pool_files{labels} {nixl.file_count}"],
    )
    emit(
        f"{_METRIC_PREFIX}_nixl_pool_slots_used",
        "NIXL pool slots with non-zero size (written at least once).",
        "gauge",
        [f"{_METRIC_PREFIX}_nixl_pool_slots_used{labels} {nixl.slots_used}"],
    )
    emit(
        f"{_METRIC_PREFIX}_nixl_pool_bytes_total",
        "Sum of st_size for used NIXL pool slots (0-byte slots excluded).",
        "gauge",
        [f"{_METRIC_PREFIX}_nixl_pool_bytes_total{labels} {nixl.bytes_total}"],
    )
    emit(
        f"{_METRIC_PREFIX}_nixl_pool_bytes_on_disk",
        "Allocated block bytes for used NIXL pool slots (st_blocks * 512).",
        "gauge",
        [
            f"{_METRIC_PREFIX}_nixl_pool_bytes_on_disk{labels} "
            f"{nixl.bytes_on_disk}"
        ],
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

    lookup_hist = summary.chunk_lookup_histogram
    emit(
        f"{_METRIC_PREFIX}_chunk_hashes_tracked",
        "Distinct chunk hashes seen in chunk_hashes JSONL lookups.",
        "gauge",
        [
            f"{_METRIC_PREFIX}_chunk_hashes_tracked{labels} "
            f"{summary.unique_chunk_hashes}"
        ],
    )
    emit(
        f"{_METRIC_PREFIX}_chunk_hash_mention_sum",
        "Total chunk hash mentions across all chunk_hashes JSONL rows.",
        "gauge",
        [
            f"{_METRIC_PREFIX}_chunk_hash_mention_sum{labels} "
            f"{summary.chunk_hash_mention_sum}"
        ],
    )
    by_lookup_lines: list[str] = []
    for bucket in _chunk_lookup_bucket_labels():
        lbl = _label_set({**(extra_labels or {}), "lookup_count": bucket})
        by_lookup_lines.append(
            f"{_METRIC_PREFIX}_chunk_hashes_by_lookup_count{lbl} "
            f"{lookup_hist.get(bucket, 0)}"
        )
    emit(
        f"{_METRIC_PREFIX}_chunk_hashes_by_lookup_count",
        "Distinct chunk hashes grouped by JSONL lookup mention count.",
        "gauge",
        by_lookup_lines,
    )
    lookup_bucket_lines: list[str] = []
    lookup_cumulative = 0
    for le in range(11):
        lookup_cumulative += lookup_hist.get(str(le), 0)
        le_lbl = _label_set({**(extra_labels or {}), "le": str(le)})
        lookup_bucket_lines.append(
            f"{_METRIC_PREFIX}_chunk_lookup_histogram_bucket{le_lbl} "
            f"{lookup_cumulative}"
        )
    for le_boundary, tail_bucket in zip(
        _CHUNK_LOOKUP_HISTOGRAM_LE, _CHUNK_LOOKUP_TAIL_BUCKETS[:-1], strict=True
    ):
        lookup_cumulative += lookup_hist.get(tail_bucket, 0)
        le_lbl = _label_set({**(extra_labels or {}), "le": str(le_boundary)})
        lookup_bucket_lines.append(
            f"{_METRIC_PREFIX}_chunk_lookup_histogram_bucket{le_lbl} "
            f"{lookup_cumulative}"
        )
    lookup_cumulative += lookup_hist.get(">100", 0)
    lookup_inf_lbl = _label_set({**(extra_labels or {}), "le": "+Inf"})
    lookup_bucket_lines.append(
        f"{_METRIC_PREFIX}_chunk_lookup_histogram_bucket{lookup_inf_lbl} "
        f"{lookup_cumulative}"
    )
    lookup_bucket_lines.append(
        f"{_METRIC_PREFIX}_chunk_lookup_histogram_sum{hist_labels} "
        f"{summary.chunk_hash_mention_sum}"
    )
    lookup_bucket_lines.append(
        f"{_METRIC_PREFIX}_chunk_lookup_histogram_count{hist_labels} "
        f"{summary.unique_chunk_hashes}"
    )
    emit(
        f"{_METRIC_PREFIX}_chunk_lookup_histogram",
        "Distribution of JSONL lookup mentions per distinct chunk hash.",
        "histogram",
        lookup_bucket_lines,
    )

    if include_host_metrics:
        nfs = snapshot.nfs
        emit(
            f"{_METRIC_PREFIX}_nfsiostat_present",
            "1 if nfsiostat is installed on PATH, else 0.",
            "gauge",
            [
                f"{_METRIC_PREFIX}_nfsiostat_present{labels} "
                f"{1 if nfs.nfsiostat_present else 0}"
            ],
        )
        rx_lines = [
            f"{_METRIC_PREFIX}_nfs_mount_rx_bytes_total"
            f"{_label_set({**(extra_labels or {}), 'mount_point': m.mount_point})} "
            f"{m.rx_bytes}"
            for m in nfs.mounts
        ]
        emit(
            f"{_METRIC_PREFIX}_nfs_mount_rx_bytes_total",
            "Cumulative NFS client bytes received (sum of per-op bytes_recv).",
            "counter",
            rx_lines,
        )
        tx_lines = [
            f"{_METRIC_PREFIX}_nfs_mount_tx_bytes_total"
            f"{_label_set({**(extra_labels or {}), 'mount_point': m.mount_point})} "
            f"{m.tx_bytes}"
            for m in nfs.mounts
        ]
        emit(
            f"{_METRIC_PREFIX}_nfs_mount_tx_bytes_total",
            "Cumulative NFS client bytes sent (sum of per-op bytes_sent).",
            "counter",
            tx_lines,
        )

        hip = snapshot.hip
        if hip.hipconfig_error != "skipped":
            emit(
                f"{_METRIC_PREFIX}_hipconfig_present",
                "1 if hipconfig is installed on PATH, else 0.",
                "gauge",
                [
                    f"{_METRIC_PREFIX}_hipconfig_present{labels} "
                    f"{1 if hip.hipconfig_present else 0}"
                ],
            )
            if hip.hipconfig_present and hip.rocm_version:
                ver_lbl = _label_set(
                    {
                        **(extra_labels or {}),
                        "version": hip.rocm_version,
                    }
                )
                emit(
                    f"{_METRIC_PREFIX}_rocm_version_info",
                    "ROCm/HIP stack version from hipconfig (gauge 1).",
                    "gauge",
                    [f"{_METRIC_PREFIX}_rocm_version_info{ver_lbl} 1"],
                )

        kfd = snapshot.kfd_ais
        if kfd.bpftrace_error != "skipped":
            emit(
                f"{_METRIC_PREFIX}_kfd_ais_bpftrace_present",
                "1 if bpftrace is installed on PATH, else 0.",
                "gauge",
                [
                    f"{_METRIC_PREFIX}_kfd_ais_bpftrace_present{labels} "
                    f"{1 if kfd.bpftrace_present else 0}"
                ],
            )
            emit(
                f"{_METRIC_PREFIX}_kfd_ais_kprobe_attachable",
                "1 if kprobe:kfd_ais_rw_file is available, else 0.",
                "gauge",
                [
                    f"{_METRIC_PREFIX}_kfd_ais_kprobe_attachable{labels} "
                    f"{1 if kfd.kprobe_attachable else 0}"
                ],
            )
            emit(
                f"{_METRIC_PREFIX}_kfd_ais_sample_seconds",
                "Duration of the last bpftrace sample window for KFD AIS I/O.",
                "gauge",
                [
                    f"{_METRIC_PREFIX}_kfd_ais_sample_seconds{labels} "
                    f"{kfd.sample_seconds}"
                ],
            )
            emit(
                f"{_METRIC_PREFIX}_kfd_ais_sample_skipped",
                "1 when KFD AIS bpftrace sampling was disabled for this scrape.",
                "gauge",
                [
                    f"{_METRIC_PREFIX}_kfd_ais_sample_skipped{labels} "
                    f"{1 if kfd.skipped else 0}"
                ],
            )

            op_lines: list[str] = []
            for (direction, outcome), count in sorted(kfd.operations.items()):
                op_lbl = _label_set(
                    {
                        **(extra_labels or {}),
                        "direction": direction,
                        "outcome": outcome,
                    }
                )
                op_lines.append(
                    f"{_METRIC_PREFIX}_kfd_ais_rw_operations{op_lbl} {count}"
                )
            for direction in ("read", "write"):
                for outcome in ("success", "partial", "error"):
                    key = (direction, outcome)
                    if key in kfd.operations:
                        continue
                    op_lbl = _label_set(
                        {
                            **(extra_labels or {}),
                            "direction": direction,
                            "outcome": outcome,
                        }
                    )
                    op_lines.append(
                        f"{_METRIC_PREFIX}_kfd_ais_rw_operations{op_lbl} 0"
                    )
            emit(
                f"{_METRIC_PREFIX}_kfd_ais_rw_operations",
                "KFD AIS transfers observed in the last bpftrace sample window.",
                "gauge",
                op_lines,
            )

            byte_lines = [
                f"{_METRIC_PREFIX}_kfd_ais_bytes"
                f"{_label_set({**(extra_labels or {}), 'direction': direction})} "
                f"{kfd.bytes_by_direction.get(direction, 0)}"
                for direction in ("read", "write")
            ]
            emit(
                f"{_METRIC_PREFIX}_kfd_ais_bytes",
                "Successful KFD AIS bytes transferred in the last sample window.",
                "gauge",
                byte_lines,
            )

            lat_hist = kfd.latency_us_histogram
            lat_bucket_lines: list[str] = []
            lat_cumulative = 0
            for le in _KFD_AIS_LATENCY_HIST_BUCKETS_US:
                lat_cumulative += lat_hist.get(str(le), 0)
                le_lbl = _label_set({**(extra_labels or {}), "le": str(le)})
                lat_bucket_lines.append(
                    f"{_METRIC_PREFIX}_kfd_ais_latency_microseconds_bucket"
                    f"{le_lbl} {lat_cumulative}"
                )
            lat_cumulative += lat_hist.get("+Inf", 0)
            lat_inf_lbl = _label_set({**(extra_labels or {}), "le": "+Inf"})
            lat_bucket_lines.append(
                f"{_METRIC_PREFIX}_kfd_ais_latency_microseconds_bucket"
                f"{lat_inf_lbl} {lat_cumulative}"
            )
            lat_bucket_lines.append(
                f"{_METRIC_PREFIX}_kfd_ais_latency_microseconds_sum{hist_labels} "
                f"{kfd.latency_us_sum}"
            )
            lat_bucket_lines.append(
                f"{_METRIC_PREFIX}_kfd_ais_latency_microseconds_count{hist_labels} "
                f"{kfd.latency_us_count}"
            )
            emit(
                f"{_METRIC_PREFIX}_kfd_ais_latency_microseconds",
                "KFD AIS transfer latency for successful operations (microseconds).",
                "histogram",
                lat_bucket_lines,
            )

        ais = snapshot.ais_hipfile
        if ais.collect_error != "skipped":
            ais_lbl = _label_set(
                {
                    **(extra_labels or {}),
                    **({"container": ais.container} if ais.container else {}),
                }
            )
            emit(
                f"{_METRIC_PREFIX}_ais_stats_configured",
                "1 when ROCM_AIC_AIS_STATS_CONTAINER is set for docker exec collection.",
                "gauge",
                [
                    f"{_METRIC_PREFIX}_ais_stats_configured{labels} "
                    f"{1 if ais.configured else 0}"
                ],
            )
            emit(
                f"{_METRIC_PREFIX}_ais_stats_docker_present",
                "1 if docker is installed on PATH for ais-stats collection.",
                "gauge",
                [
                    f"{_METRIC_PREFIX}_ais_stats_docker_present{labels} "
                    f"{1 if ais.docker_present else 0}"
                ],
            )
            emit(
                f"{_METRIC_PREFIX}_ais_stats_collect_ok",
                "1 when the last ais-stats docker exec scrape succeeded.",
                "gauge",
                [
                    f"{_METRIC_PREFIX}_ais_stats_collect_ok{ais_lbl} "
                    f"{1 if ais.configured and ais.collect_error is None else 0}"
                ],
            )
            if ais.configured:
                emit(
                    f"{_METRIC_PREFIX}_ais_stats_level",
                    "HipFile stats level reported by ais-stats inside the container.",
                    "gauge",
                    [f"{_METRIC_PREFIX}_ais_stats_level{ais_lbl} {ais.stats_level}"],
                )
                path_rows = (
                    ("fastpath", "read", ais.fastpath_read),
                    ("fastpath", "write", ais.fastpath_write),
                    ("fallback", "read", ais.fallback_read),
                    ("fallback", "write", ais.fallback_write),
                )
                byte_lines = [
                    f"{_METRIC_PREFIX}_ais_stats_bytes"
                    f"{_label_set({**(extra_labels or {}), **({'container': ais.container} if ais.container else {}), 'backend': backend, 'direction': direction})} "
                    f"{path.bytes_total}"
                    for backend, direction, path in path_rows
                ]
                emit(
                    f"{_METRIC_PREFIX}_ais_stats_bytes",
                    "Cumulative hipFile bytes from ais-stats (since EngineCore start).",
                    "gauge",
                    byte_lines,
                )
                bw_lines = [
                    f"{_METRIC_PREFIX}_ais_stats_bandwidth_gibps"
                    f"{_label_set({**(extra_labels or {}), **({'container': ais.container} if ais.container else {}), 'backend': backend, 'direction': direction})} "
                    f"{path.bandwidth_gibps}"
                    for backend, direction, path in path_rows
                ]
                emit(
                    f"{_METRIC_PREFIX}_ais_stats_bandwidth_gibps",
                    "Average hipFile bandwidth from ais-stats (GiB/s).",
                    "gauge",
                    bw_lines,
                )
                lat_lines = [
                    f"{_METRIC_PREFIX}_ais_stats_latency_microseconds"
                    f"{_label_set({**(extra_labels or {}), **({'container': ais.container} if ais.container else {}), 'backend': backend, 'direction': direction})} "
                    f"{path.latency_us}"
                    for backend, direction, path in path_rows
                ]
                emit(
                    f"{_METRIC_PREFIX}_ais_stats_latency_microseconds",
                    "Average hipFile latency from ais-stats (microseconds).",
                    "gauge",
                    lat_lines,
                )
                err_lines = [
                    f"{_METRIC_PREFIX}_ais_stats_errors_total"
                    f"{_label_set({**(extra_labels or {}), **({'container': ais.container} if ais.container else {}), 'backend': backend, 'direction': direction})} "
                    f"{path.errors_total}"
                    for backend, direction, path in path_rows
                ]
                emit(
                    f"{_METRIC_PREFIX}_ais_stats_errors_total",
                    "Cumulative hipFile errors from ais-stats (since EngineCore start).",
                    "gauge",
                    err_lines,
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
    nixl = summary.nixl_pool
    if nixl.file_count > 0:
        print("\nNIXL static pool (obj_*.bin)")
        print(f"  Files = {nixl.file_count}")
        print(f"  Used slots = {nixl.slots_used}")
        print(f"  Used size (st_size) = {_format_bytes(nixl.bytes_total)}")
        if nixl.bytes_on_disk != nixl.bytes_total:
            print(f"  On disk (blocks) = {_format_bytes(nixl.bytes_on_disk)}")
    if summary.filesystem is not None:
        fs = summary.filesystem
        print(f"\nFilesystem {fs.path}")
        print(f"  Total = {_format_bytes(fs.total_bytes)}")
        print(f"  Used  = {_format_bytes(fs.used_bytes)}")
        print(f"  Free  = {_format_bytes(fs.free_bytes)}")


def _print_bucket_histogram(
    *,
    title: str,
    hist: dict[str, int],
    universe_label: str,
    universe_count: int,
    matched_label: str,
    matched_count: int,
    lookup_rows: int,
    stats_files: int,
    extra_note: str | None = None,
) -> None:
    label_w = 12
    cnt_w = 10
    print(f"\n{title}")
    print(f"{universe_label} = {universe_count}")
    print(f"{matched_label} = {matched_count}")
    print(
        f"Stat lookup rows read = {lookup_rows} "
        f"({stats_files} jsonl file(s))"
    )
    if extra_note:
        print(extra_note)
    hdr = f"{'Mentions':<{label_w}}{'Count':>{cnt_w}}"
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
        row_label = "0 mentions" if i == 0 else k
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


def _print_chunk_lookup_histogram(summary: ChunkHitSummary) -> None:
    hist = summary.chunk_lookup_histogram
    label_w = 12
    cnt_w = 10
    print("\nChunk hash lookup frequency (chunk_hashes JSONL)")
    print(f"Distinct chunk hashes = {summary.unique_chunk_hashes}")
    print(f"Total hash mentions = {summary.chunk_hash_mention_sum}")
    print(
        f"Stat lookup rows read = {summary.lookup_rows} "
        f"({summary.stats_files} jsonl file(s))"
    )
    if summary.disk_file_count == 0 and summary.nixl_pool.file_count > 0:
        print(
            "NIXL mode: lookup references per hash; not mapped to "
            "obj_*.bin slots"
        )
    hdr = f"{'Mentions':<{label_w}}{'Count':>{cnt_w}}"
    print(hdr)
    print("-" * len(hdr))
    rows: list[tuple[str, int]] = []
    for i in range(11):
        c = hist.get(str(i), 0)
        if c <= 0:
            continue
        rows.append(("0 mentions" if i == 0 else str(i), c))
    for bucket in _CHUNK_LOOKUP_TAIL_BUCKETS:
        c = hist.get(bucket, 0)
        if c <= 0:
            continue
        rows.append((bucket, c))
    for row_label, c in rows:
        print(f"{row_label:<{label_w}}{c:>{cnt_w}d}")


def _print_histogram(summary: ChunkHitSummary) -> None:
    _print_inventory(summary)
    if summary.disk_file_count > 0:
        extra = None
        if summary.orphan_stat_mentions:
            extra = (
                f"Stat mentions for deleted/missing files = "
                f"{summary.orphan_stat_mentions} (excluded from histogram)"
            )
        _print_bucket_histogram(
            title="Hits per on-disk KV file (.data)",
            hist=summary.kv_block_hit_histogram,
            universe_label="Total on-disk files",
            universe_count=summary.disk_file_count,
            matched_label="Files with >= 1 stat mention",
            matched_count=summary.hit_file_count,
            lookup_rows=summary.lookup_rows,
            stats_files=summary.stats_files,
            extra_note=extra,
        )
    if summary.lookup_rows > 0 or summary.unique_chunk_hashes > 0:
        _print_chunk_lookup_histogram(summary)
    elif summary.disk_file_count == 0 and summary.nixl_pool.file_count > 0:
        print(
            "\nChunk hash lookup frequency (chunk_hashes JSONL)\n"
            f"No chunk_hashes JSONL rows under {summary.stats_dir}."
        )


def _print_host_observability(snapshot: ExporterSnapshot) -> None:
    nfs = snapshot.nfs
    print("\nNFS client stats (mountstats)")
    print(f"nfsiostat present = {nfs.nfsiostat_present}")
    if nfs.nfsiostat_error:
        print(f"nfsiostat run note = {nfs.nfsiostat_error}")
    print(f"mountstats = {nfs.mountstats_path}")
    if not nfs.mounts:
        print("No NFS mounts in mountstats.")
    else:
        print(f"{'Mount':<32} {'RX':>16} {'TX':>16}")
        print("-" * 66)
        for m in nfs.mounts:
            print(
                f"{m.mount_point:<32} "
                f"{_format_bytes(m.rx_bytes):>16} "
                f"{_format_bytes(m.tx_bytes):>16}"
            )

    hip = snapshot.hip
    print("\nROCm / HIP (hipconfig)")
    if hip.hipconfig_error == "skipped":
        print("hipconfig collection skipped (--skip-hipconfig)")
        print(f"hipconfig on PATH = {hip.hipconfig_present}")
    else:
        print(f"hipconfig present = {hip.hipconfig_present}")
        if hip.hipconfig_present:
            print(f"ROCm/HIP version = {hip.rocm_version or 'unknown'}")
        if hip.hipconfig_error:
            print(f"hipconfig note = {hip.hipconfig_error}")

    kfd = snapshot.kfd_ais
    print("\nKFD AIS (kfd_ais_rw_file bpftrace)")
    if kfd.bpftrace_error == "skipped":
        print("KFD AIS bpftrace collection skipped (--skip-kfd-ais-bpftrace)")
        return
    print(f"bpftrace present = {kfd.bpftrace_present}")
    print(f"kprobe attachable = {kfd.kprobe_attachable}")
    print(f"sample seconds = {kfd.sample_seconds}")
    print(f"sample skipped = {kfd.skipped}")
    if kfd.bpftrace_error:
        print(f"bpftrace note = {kfd.bpftrace_error}")
    if kfd.operations:
        print(f"{'Direction':<8} {'Outcome':<10} {'Count':>8}")
        print("-" * 28)
        for (direction, outcome), count in sorted(kfd.operations.items()):
            print(f"{direction:<8} {outcome:<10} {count:>8d}")
    for direction in ("read", "write"):
        nbytes = kfd.bytes_by_direction.get(direction, 0)
        if nbytes:
            print(f"{direction} bytes (success) = {_format_bytes(nbytes)}")
    if kfd.latency_us_count:
        avg_us = kfd.latency_us_sum / kfd.latency_us_count
        print(
            f"latency us (success) count={kfd.latency_us_count} "
            f"avg={avg_us:.0f}"
        )


def _default_data_root(recipe_root: Path) -> Path:
    for key in ("VLH_HOST_DATA_ROOT", "RADEON_HOST_DATA_ROOT", "DATA"):
        host = os.environ.get(key, "").strip()
        if host:
            return Path(host)
    return Path("/mnt/lmcache-nvme")


def _default_textfile_path() -> Path | None:
    for key in (
        "ROCM_AIC_EXPORTER_TEXTFILE",
        "VLH_LMCACHE_CHUNK_HIST_TEXTFILE",
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
            "LMCache data root (host path). Default: VLH_HOST_DATA_ROOT, "
            "then DATA, then /mnt/lmcache-nvme."
        ),
    )
    p.add_argument(
        "--kv-subdir",
        default=os.environ.get("VLH_LMCACHE_KV_SUBDIR", "lmcache"),
        help="KV .data directory under data-root (default: lmcache).",
    )
    p.add_argument(
        "--stats-subdir",
        default=os.environ.get(
            "VLH_LMCACHE_CHUNK_STATS_SUBDIR", "lmcache_chunk_stats"
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
    p.add_argument(
        "--mountstats-path",
        type=Path,
        default=Path("/proc/self/mountstats"),
        help="NFS mountstats file (default: /proc/self/mountstats).",
    )
    p.add_argument(
        "--skip-nfsiostat-invoke",
        action="store_true",
        help="Do not run nfsiostat; still parse mountstats if readable.",
    )
    p.add_argument(
        "--skip-hipconfig",
        action="store_true",
        help=(
            "Do not run hipconfig; omit hipconfig_present and rocm_version_info "
            "from the textfile (PATH lookup still reported in JSON/CLI)."
        ),
    )
    p.add_argument(
        "--skip-kfd-ais-bpftrace",
        action="store_true",
        help=(
            "Do not run kfd_ais_rw_file bpftrace sampling; omit KFD AIS metrics "
            "from the textfile."
        ),
    )
    p.add_argument(
        "--kfd-ais-sample-seconds",
        type=float,
        default=None,
        metavar="SECS",
        help=(
            "bpftrace sample window for kfd_ais_rw_file (default: "
            "ROCM_AIC_KFD_AIS_SAMPLE_SECONDS or 10 when writing textfile; "
            "0 disables)."
        ),
    )
    p.add_argument(
        "--ais-stats-container",
        default=os.environ.get("ROCM_AIC_AIS_STATS_CONTAINER", ""),
        metavar="NAME",
        help=(
            "Docker container for hipFile ais-stats collection via docker exec "
            "(default: ROCM_AIC_AIS_STATS_CONTAINER; empty disables)."
        ),
    )
    p.add_argument(
        "--skip-ais-stats-container",
        action="store_true",
        help="Do not docker exec ais-stats; omit hipFile ais-stats metrics.",
    )
    p.add_argument(
        "--ais-stats-engine-match",
        default=os.environ.get("ROCM_AIC_AIS_STATS_ENGINE_MATCH", "VLLM::EngineCor"),
        help=(
            "pgrep -f pattern inside the container to locate EngineCore "
            "(default: VLLM::EngineCor)."
        ),
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

    prom_path = args.prometheus_textfile
    if args.skip_kfd_ais_bpftrace:
        kfd_ais_sample_seconds = 0.0
    elif args.kfd_ais_sample_seconds is not None:
        kfd_ais_sample_seconds = max(0.0, args.kfd_ais_sample_seconds)
    elif prom_path is not None:
        kfd_ais_sample_seconds = _default_kfd_ais_sample_seconds()
    else:
        kfd_ais_sample_seconds = 0.0

    snapshot = collect_exporter_snapshot(
        data_root=data_root,
        kv_subdir=args.kv_subdir,
        stats_subdir=args.stats_subdir,
        mountstats_path=args.mountstats_path,
        run_nfsiostat=not args.skip_nfsiostat_invoke,
        collect_hip=not args.skip_hipconfig,
        include_host_metrics=prom_path is not None,
        kfd_ais_sample_seconds=kfd_ais_sample_seconds,
        collect_kfd_ais=not args.skip_kfd_ais_bpftrace,
        ais_stats_container=args.ais_stats_container,
        collect_ais_stats=not args.skip_ais_stats_container,
        ais_stats_engine_match=args.ais_stats_engine_match,
    )
    summary = snapshot.chunk

    if summary.disk_file_count == 0 and summary.nixl_pool.file_count == 0:
        print(
            f"warning: no .data or NIXL obj_*.bin files under {summary.kv_dir}; "
            "storage inventory empty",
            file=sys.stderr,
        )

    if summary.stats_files == 0 and summary.disk_file_count > 0:
        print(
            f"warning: no chunk_hashes_*.jsonl under {summary.stats_dir}; "
            "histogram is all zero-hit files",
            file=sys.stderr,
        )

    if (
        summary.lookup_rows > 0
        and summary.disk_file_count > 0
        and summary.hit_file_count == 0
        and summary.orphan_stat_mentions > 0
    ):
        print(
            "warning: chunk_hashes JSONL did not match any on-disk .data tags "
            "(orphan_stat_mentions="
            f"{summary.orphan_stat_mentions}). Rebuild the vllm-lmcache-hipfile image "
            "with lmcache-chunk-statistics-hash.patch, restart vLLM, and "
            "collect new stats after pre_caching_hash_algorithm matches storage "
            "(e.g. sha256_cbor). Existing JSONL from builtin hashing cannot be "
            "reconciled with stored keys.",
            file=sys.stderr,
        )

    if prom_path is not None and snapshot.host_metrics_collected:
        if not snapshot.nfs.nfsiostat_present:
            print(
                "warning: nfsiostat not found; nfs byte metrics use "
                "mountstats only",
                file=sys.stderr,
            )
        elif snapshot.nfs.nfsiostat_error:
            print(
                f"warning: nfsiostat: {snapshot.nfs.nfsiostat_error}",
                file=sys.stderr,
            )
        if not args.skip_hipconfig:
            if not snapshot.hip.hipconfig_present:
                print("warning: hipconfig not found in PATH", file=sys.stderr)
            elif snapshot.hip.hipconfig_error:
                print(
                    f"warning: hipconfig: {snapshot.hip.hipconfig_error}",
                    file=sys.stderr,
                )
        if (
            not args.skip_kfd_ais_bpftrace
            and kfd_ais_sample_seconds > 0
            and snapshot.host_metrics_collected
        ):
            kfd = snapshot.kfd_ais
            if not kfd.bpftrace_present:
                print("warning: bpftrace not found; KFD AIS metrics empty", file=sys.stderr)
            elif not kfd.kprobe_attachable:
                print(
                    "warning: kprobe:kfd_ais_rw_file not available",
                    file=sys.stderr,
                )
            elif kfd.bpftrace_error and not kfd.operations:
                print(f"warning: kfd ais bpftrace: {kfd.bpftrace_error}", file=sys.stderr)
        if (
            not args.skip_ais_stats_container
            and args.ais_stats_container.strip()
            and snapshot.host_metrics_collected
        ):
            ais = snapshot.ais_hipfile
            if not ais.docker_present:
                print(
                    "warning: docker not found; ais-stats metrics empty",
                    file=sys.stderr,
                )
            elif ais.collect_error:
                print(f"warning: ais-stats: {ais.collect_error}", file=sys.stderr)

    if prom_path is not None:
        body = format_prometheus_textfile(
            snapshot,
            extra_labels=extra_labels or None,
            include_host_metrics=True,
        )
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
                    "chunk_lookup_histogram": summary.chunk_lookup_histogram,
                    "chunk_hash_mention_sum": summary.chunk_hash_mention_sum,
                    "unique_chunk_hashes": summary.unique_chunk_hashes,
                    "files_by_model": summary.files_by_model,
                    "bytes_by_model": summary.bytes_by_model,
                    "chunk_bytes_total": summary.chunk_bytes_total,
                    "nixl_pool": {
                        "file_count": summary.nixl_pool.file_count,
                        "slots_used": summary.nixl_pool.slots_used,
                        "bytes_total": summary.nixl_pool.bytes_total,
                        "bytes_on_disk": summary.nixl_pool.bytes_on_disk,
                    },
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
                    "host_metrics_collected": snapshot.host_metrics_collected,
                    "nfs": (
                        {
                            "nfsiostat_present": snapshot.nfs.nfsiostat_present,
                            "mountstats_path": str(snapshot.nfs.mountstats_path),
                            "nfsiostat_error": snapshot.nfs.nfsiostat_error,
                            "mounts": [
                                {
                                    "mount_point": m.mount_point,
                                    "rx_bytes": m.rx_bytes,
                                    "tx_bytes": m.tx_bytes,
                                }
                                for m in snapshot.nfs.mounts
                            ],
                        }
                        if snapshot.host_metrics_collected
                        else None
                    ),
                    "hipconfig": (
                        {
                            "hipconfig_present": snapshot.hip.hipconfig_present,
                            "rocm_version": snapshot.hip.rocm_version,
                            "hipconfig_error": snapshot.hip.hipconfig_error,
                        }
                        if snapshot.host_metrics_collected
                        else None
                    ),
                    "kfd_ais": (
                        {
                            "bpftrace_present": snapshot.kfd_ais.bpftrace_present,
                            "kprobe_attachable": snapshot.kfd_ais.kprobe_attachable,
                            "sample_seconds": snapshot.kfd_ais.sample_seconds,
                            "skipped": snapshot.kfd_ais.skipped,
                            "bpftrace_error": snapshot.kfd_ais.bpftrace_error,
                            "operations": {
                                f"{direction}:{outcome}": count
                                for (direction, outcome), count in (
                                    snapshot.kfd_ais.operations.items()
                                )
                            },
                            "bytes_by_direction": snapshot.kfd_ais.bytes_by_direction,
                            "latency_us_sum": snapshot.kfd_ais.latency_us_sum,
                            "latency_us_count": snapshot.kfd_ais.latency_us_count,
                        }
                        if snapshot.host_metrics_collected
                        else None
                    ),
                    "ais_hipfile": (
                        {
                            "configured": snapshot.ais_hipfile.configured,
                            "skipped": snapshot.ais_hipfile.skipped,
                            "docker_present": snapshot.ais_hipfile.docker_present,
                            "container": snapshot.ais_hipfile.container,
                            "collect_error": snapshot.ais_hipfile.collect_error,
                            "stats_level": snapshot.ais_hipfile.stats_level,
                            "fastpath_write_bytes": (
                                snapshot.ais_hipfile.fastpath_write.bytes_total
                            ),
                            "fastpath_read_bytes": (
                                snapshot.ais_hipfile.fastpath_read.bytes_total
                            ),
                        }
                        if snapshot.host_metrics_collected
                        else None
                    ),
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
        if snapshot.host_metrics_collected:
            _print_host_observability(snapshot)
        if args.top > 0:
            stats_glob = str(summary.stats_dir / "chunk_hashes_*.jsonl")
            disk = _scan_disk_kv_files(summary.kv_dir)
            if disk:
                hits, _, _ = _load_hit_counts(stats_glob, set(disk))
                print(f"\nTop {args.top} on-disk files by stat mentions")
                for tag, count in hits.most_common(args.top):
                    print(f"  {count:8d}  {disk[tag].name}")
            else:
                mentions, _ = _load_jsonl_hash_mentions(stats_glob)
                if mentions:
                    print(f"\nTop {args.top} chunk hashes by JSONL lookup mentions")
                    for h, count in mentions.most_common(args.top):
                        print(f"  {count:8d}  {h}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
