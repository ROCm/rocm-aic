"""Base workload class for LMCache simulation."""
import hashlib
import json
import math
import statistics
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field


def _kv_block_identity(operation: Dict[str, Any]) -> str:
    """Stable id for KV-block hit histograms.

    Prefer a hash of ``token_ids`` when present (e.g. retrieve-only
    uses random ``key`` per op but repeats chunk content). Otherwise
    use the request ``key`` (store/retrieve req_id).
    """
    toks = operation.get("token_ids")
    if isinstance(toks, list) and len(toks) > 0:
        payload = json.dumps(
            toks,
            separators=(",", ":"),
        ).encode()
        digest = hashlib.blake2b(
            payload,
            digest_size=16,
        ).hexdigest()
        return f"t:{digest}"
    key = operation.get("key")
    if key is not None:
        return f"k:{key}"
    return "?"


def _kv_block_hit_histogram_universe(
    universe: Set[str],
    hit_counts: Dict[str, int],
) -> Dict[str, int]:
    """Bucket each id in *universe* by hit count (0..10, >10)."""
    hist: Dict[str, int] = {
        str(i): 0 for i in range(11)
    }
    hist[">10"] = 0
    for bid in universe:
        h = hit_counts.get(bid, 0)
        if h > 10:
            hist[">10"] += 1
        else:
            hist[str(h)] += 1
    return hist


def _kv_block_hit_histogram(
    stored: Set[str],
    retrieve_seen: Set[str],
    hit_counts: Dict[str, int],
) -> Dict[str, int]:
    """Bucket counts for blocks seen in store/retrieve sim universe."""
    return _kv_block_hit_histogram_universe(
        stored | retrieve_seen,
        hit_counts,
    )


def _enrich_summary_latency_percentiles(
    samples: List[float],
    summary: Dict[str, Any],
) -> None:
    """Add ``latency_std_ms``, ``latency_p99_ms``,
    ``latency_p999_ms`` when ``samples`` is non-empty."""
    if not samples:
        return
    svs = sorted(samples)
    summary["latency_std_ms"] = (
        statistics.stdev(samples)
        if len(samples) > 1
        else 0.0
    )
    summary["latency_p99_ms"] = _percentile_linear(
        svs, 99.0
    )
    summary["latency_p999_ms"] = _percentile_linear(
        svs, 99.9
    )


def _percentile_linear(
    sorted_vals: List[float], p: float
) -> float:
    """Return the p-th percentile (0–100) with linear
    interpolation. ``sorted_vals`` must be sorted."""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = min(f + 1, n - 1)
    return sorted_vals[f] + (
        sorted_vals[c] - sorted_vals[f]
    ) * (k - f)


@dataclass
class OperationTypeMetrics:
    """Per-operation-type metrics."""

    count: int = 0
    successful: int = 0
    failed: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    kv_blocks: int = 0
    total_bytes: int = 0
    hits: int = 0
    misses: int = 0

    def record(
        self,
        success: bool,
        latency_ms: float,
        blocks: int = 0,
        data_bytes: int = 0,
        cache_hit: bool = False,
    ):
        """Record a single operation of this type."""
        self.count += 1
        if success:
            self.successful += 1
            self.total_latency_ms += latency_ms
            self.min_latency_ms = min(
                self.min_latency_ms, latency_ms
            )
            self.max_latency_ms = max(
                self.max_latency_ms, latency_ms
            )
            self.kv_blocks += blocks
            self.total_bytes += data_bytes
            if cache_hit:
                self.hits += 1
            else:
                self.misses += 1
        else:
            self.failed += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get summary for this operation type."""
        avg = (
            self.total_latency_ms / self.successful
            if self.successful > 0
            else 0.0
        )
        total_lookups = self.hits + self.misses
        hit_rate = (
            self.hits / total_lookups
            if total_lookups > 0
            else 0.0
        )
        return {
            "count": self.count,
            "successful": self.successful,
            "failed": self.failed,
            "average_latency_ms": avg,
            "min_latency_ms": self.min_latency_ms
            if self.min_latency_ms != float("inf")
            else 0.0,
            "max_latency_ms": self.max_latency_ms,
            "kv_blocks": self.kv_blocks,
            "total_bytes": self.total_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }


@dataclass
class WorkloadMetrics:
    """Metrics collected during workload execution."""

    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    start_time: float = field(
        default_factory=time.time
    )
    end_time: Optional[float] = None
    store_metrics: OperationTypeMetrics = field(
        default_factory=OperationTypeMetrics
    )
    retrieve_metrics: OperationTypeMetrics = field(
        default_factory=OperationTypeMetrics
    )
    lookup_metrics: OperationTypeMetrics = field(
        default_factory=OperationTypeMetrics
    )
    pass_number: int = 1
    conversations_completed: int = 0
    latency_samples: List[float] = field(
        default_factory=list
    )
    store_latency_samples: List[float] = field(
        default_factory=list
    )
    retrieve_latency_samples: List[float] = field(
        default_factory=list
    )
    lookup_latency_samples: List[float] = field(
        default_factory=list
    )
    stored_kv_block_ids: Set[str] = field(
        default_factory=set
    )
    retrieve_seen_kv_block_ids: Set[str] = field(
        default_factory=set
    )
    kv_block_hit_counts: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    #: Distinct chunk identities from the JSONL sidecar (retrieve-only).
    chunk_index_fingerprints: Optional[frozenset[str]] = None
    #: Resolved path to ``.lmcache_io_chunk_tokens.jsonl`` when set.
    chunk_index_path: Optional[str] = None

    def record_operation(
        self,
        success: bool,
        latency_ms: float,
        cache_hit: bool = False,
        op_type: str = "store",
        kv_blocks: int = 0,
        data_bytes: int = 0,
        block_identity: Optional[str] = None,
    ):
        """Record a single operation."""
        self.total_operations += 1
        if success:
            self.successful_operations += 1
            self.total_latency_ms += latency_ms
            self.min_latency_ms = min(
                self.min_latency_ms, latency_ms
            )
            self.max_latency_ms = max(
                self.max_latency_ms, latency_ms
            )
            self.latency_samples.append(latency_ms)
            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
        else:
            self.failed_operations += 1

        if op_type == "store":
            self.store_metrics.record(
                success, latency_ms, kv_blocks,
                data_bytes=data_bytes,
            )
            if success and block_identity is not None:
                self.stored_kv_block_ids.add(
                    block_identity
                )
            if success:
                self.store_latency_samples.append(
                    latency_ms
                )
        elif op_type == "lookup":
            self.lookup_metrics.record(
                success,
                latency_ms,
                kv_blocks,
                data_bytes=data_bytes,
                cache_hit=cache_hit,
            )
            if success and block_identity is not None:
                self.retrieve_seen_kv_block_ids.add(
                    block_identity
                )
                if cache_hit:
                    self.kv_block_hit_counts[
                        block_identity
                    ] += 1
            if success:
                self.lookup_latency_samples.append(
                    latency_ms
                )
        else:
            self.retrieve_metrics.record(
                success, latency_ms, kv_blocks,
                data_bytes=data_bytes,
                cache_hit=cache_hit,
            )
            if success and block_identity is not None:
                self.retrieve_seen_kv_block_ids.add(
                    block_identity
                )
                if cache_hit:
                    self.kv_block_hit_counts[
                        block_identity
                    ] += 1
            if success:
                self.retrieve_latency_samples.append(
                    latency_ms
                )

    def finalize(self):
        """Finalize metrics collection."""
        self.end_time = time.time()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of metrics."""
        duration = (
            self.end_time - self.start_time
            if self.end_time
            else time.time() - self.start_time
        )
        avg_latency = (
            self.total_latency_ms / self.successful_operations
            if self.successful_operations > 0
            else 0.0
        )
        hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0
            else 0.0
        )
        throughput = (
            self.successful_operations / duration if duration > 0 else 0.0
        )

        store_summary = self.store_metrics.get_summary()
        _enrich_summary_latency_percentiles(
            self.store_latency_samples,
            store_summary,
        )
        retrieve_summary = (
            self.retrieve_metrics.get_summary()
        )
        _enrich_summary_latency_percentiles(
            self.retrieve_latency_samples,
            retrieve_summary,
        )
        lookup_summary = self.lookup_metrics.get_summary()
        _enrich_summary_latency_percentiles(
            self.lookup_latency_samples,
            lookup_summary,
        )

        hc_plain = dict(self.kv_block_hit_counts)
        if self.chunk_index_fingerprints is not None:
            kv_hist = _kv_block_hit_histogram_universe(
                set(self.chunk_index_fingerprints),
                hc_plain,
            )
            chunk_distinct = len(self.chunk_index_fingerprints)
            chunk_never = sum(
                1
                for fp in self.chunk_index_fingerprints
                if hc_plain.get(fp, 0) == 0
            )
        else:
            kv_hist = _kv_block_hit_histogram(
                self.stored_kv_block_ids,
                self.retrieve_seen_kv_block_ids,
                hc_plain,
            )
            chunk_distinct = None
            chunk_never = None

        summary = {
            "duration_seconds": duration,
            "total_operations": (
                self.total_operations
            ),
            "successful_operations": (
                self.successful_operations
            ),
            "failed_operations": (
                self.failed_operations
            ),
            "average_latency_ms": avg_latency,
            "min_latency_ms": (
                self.min_latency_ms
                if self.min_latency_ms
                != float("inf")
                else 0.0
            ),
            "max_latency_ms": self.max_latency_ms,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
            "throughput_ops_per_sec": throughput,
            "store_operations": store_summary,
            "retrieve_operations": retrieve_summary,
            "lookup_operations": lookup_summary,
            "kv_block_hit_histogram": kv_hist,
        }
        if self.chunk_index_fingerprints is not None:
            summary["chunk_index_path"] = (
                self.chunk_index_path
            )
            summary["chunk_index_distinct_chunks"] = (
                chunk_distinct
            )
            summary["chunk_index_never_hit_count"] = (
                chunk_never
            )
        _enrich_summary_latency_percentiles(
            self.latency_samples,
            summary,
        )
        summary["pass_number"] = self.pass_number
        if self.conversations_completed > 0:
            summary["conversations_completed"] = (
                self.conversations_completed
            )
        return summary


class BaseWorkload(ABC):
    """Base class for workload generators."""

    def __init__(self, engine: Any = None):
        """
        Initialize workload.

        Args:
            engine: EngineManager instance for direct
                    engine calls
        """
        self.engine = engine
        self.metrics = WorkloadMetrics()

    @abstractmethod
    def generate_operation(self) -> Dict[str, Any]:
        """
        Generate a single operation.

        Returns:
            Operation dictionary
        """
        pass

    @abstractmethod
    def execute_operation(self, operation: Dict[str, Any]) -> bool:
        """
        Execute a single operation.

        Args:
            operation: Operation dictionary

        Returns:
            True if successful
        """
        pass

    def run(
        self,
        duration: Optional[float] = None,
        num_operations: Optional[int] = None,
        rate: Optional[float] = None,
    ):
        """
        Run workload.

        Args:
            duration: Duration in seconds
            num_operations: Number of operations to perform
            rate: Operations per second
        """
        if duration is None and num_operations is None:
            raise ValueError(
                "Either duration or num_operations must be specified"
            )

        self.metrics.start_time = time.time()
        operation_count = 0
        start_time = time.time()
        next_op_time = start_time

        per_op_log_path = getattr(self, "per_op_log", None)
        legacy_store_log = getattr(
            self, "per_op_store_log", None
        )
        log_path = per_op_log_path or legacy_store_log
        log_all_ops = per_op_log_path is not None
        log_f = None
        if log_path:
            lp = Path(log_path)
            lp.parent.mkdir(parents=True, exist_ok=True)
            pass_num = getattr(
                self.metrics, "pass_number", 1
            )
            mode = "a" if pass_num > 1 else "w"
            log_f = open(lp, mode, encoding="utf-8")

        try:
            while True:
                if duration and (
                    time.time() - start_time
                ) >= duration:
                    break

                if (
                    num_operations
                    and operation_count >= num_operations
                ):
                    break

                if rate:
                    current_time = time.time()
                    if current_time < next_op_time:
                        time.sleep(
                            next_op_time - current_time
                        )
                    next_op_time = max(
                        next_op_time, time.time()
                    ) + (1.0 / rate)

                operation = self.generate_operation()
                op_start = time.time()
                success = self.execute_operation(operation)
                op_latency = (
                    time.time() - op_start
                ) * 1000

                cache_hit = operation.get(
                    "cache_hit", False
                )
                op_type = operation.get(
                    "type", "store"
                )
                kv_blocks = operation.get(
                    "kv_blocks", 0
                )
                data_bytes = operation.get(
                    "data_bytes", 0
                )
                self.metrics.record_operation(
                    success,
                    op_latency,
                    cache_hit,
                    op_type=op_type,
                    kv_blocks=kv_blocks,
                    data_bytes=data_bytes,
                    block_identity=_kv_block_identity(
                        operation
                    ),
                )
                if log_f:
                    if log_all_ops:
                        ts = time.time()
                        ts_iso = datetime.fromtimestamp(
                            ts, tz=timezone.utc
                        ).isoformat()
                        log_f.write(
                            json.dumps(
                                {
                                    "op_index": (
                                        operation_count
                                    ),
                                    "ts_unix": ts,
                                    "ts_iso": ts_iso,
                                    "op_type": op_type,
                                    "success": success,
                                    "cache_hit": cache_hit,
                                    "latency_ms": round(
                                        op_latency, 6
                                    ),
                                    "kv_blocks": kv_blocks,
                                    "data_bytes": data_bytes,
                                }
                            )
                            + "\n"
                        )
                        log_f.flush()
                    elif (
                        op_type == "store"
                        and success
                        and self.metrics.store_latency_samples
                        is not None
                    ):
                        ts = time.time()
                        ts_iso = datetime.fromtimestamp(
                            ts, tz=timezone.utc
                        ).isoformat()
                        log_f.write(
                            json.dumps(
                                {
                                    "op_index": (
                                        operation_count
                                    ),
                                    "ts_unix": ts,
                                    "ts_iso": ts_iso,
                                    "latency_ms": round(
                                        op_latency, 6
                                    ),
                                    "bytes_written": (
                                        data_bytes
                                    ),
                                }
                            )
                            + "\n"
                        )
                        log_f.flush()
                operation_count += 1
        finally:
            if log_f:
                log_f.close()

        convs = getattr(
            self, "conversations_completed", 0
        )
        self.metrics.conversations_completed = convs
        self.metrics.finalize()

    def get_metrics(self) -> Dict[str, Any]:
        """Get workload metrics."""
        return self.metrics.get_summary()
