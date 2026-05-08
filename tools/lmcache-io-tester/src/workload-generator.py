"""Workload generator for LMCache simulation."""
import json
import importlib.util
import sys
from pathlib import Path
from typing import Optional, Dict, Any

def _import_workload_module(name):
    """Import workload module."""
    module_path = Path(__file__).parent / "workloads" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"workloads.{name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module workloads.{name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"workloads.{name}"] = module
    spec.loader.exec_module(module)
    return module

# Import workload modules
_base_module = _import_workload_module("base")
_patterns_module = _import_workload_module("patterns")

BaseWorkload = _base_module.BaseWorkload
RandomWorkload = _patterns_module.RandomWorkload
StoreOnlyWorkload = _patterns_module.StoreOnlyWorkload
RetrieveOnlyWorkload = (
    _patterns_module.RetrieveOnlyWorkload
)
SteadyStateWorkload = _patterns_module.SteadyStateWorkload
ConversationWorkload = (
    _patterns_module.ConversationWorkload
)
CHUNK_TOKENS_INDEX_NAME = (
    _patterns_module.CHUNK_TOKENS_INDEX_NAME
)


def _format_bytes(num_bytes: int) -> str:
    """Format a byte count with appropriate unit."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.2f} KiB"
    elif num_bytes < 1024 ** 3:
        return f"{num_bytes / 1024 ** 2:.2f} MiB"
    elif num_bytes < 1024 ** 4:
        return f"{num_bytes / 1024 ** 3:.2f} GiB"
    return f"{num_bytes / 1024 ** 4:.2f} TiB"


_BAR_WIDTH = 42

# Block elements for ASCII bars (Unicode).
_BLK = "\u2588"  # full block
_DIM = "\u2591"  # light shade


def _ascii_ratio_bar(
    ratio: float, width: int = _BAR_WIDTH
) -> str:
    """Horizontal bar for a fraction in ``[0, 1]``."""
    ratio = max(0.0, min(1.0, float(ratio)))
    filled = int(round(ratio * width))
    filled = min(width, max(0, filled))
    return _BLK * filled + _DIM * (width - filled)


def _hits_per_block(
    hits: int, kv_blocks: int
) -> Optional[float]:
    if kv_blocks <= 0:
        return None
    return hits / kv_blocks


def _fmt_ms_cell(
    op: Dict[str, Any], key: str, width: int
) -> str:
    if op.get("successful", 0) <= 0:
        return f"{'—':>{width}s}"
    if key not in op:
        return f"{'—':>{width}s}"
    return f"{op[key]:>{width}.2f}"


def _print_operation_stats_table(
    store: Dict[str, Any],
    retrieve: Dict[str, Any],
) -> None:
    """Aligned store / retrieve rows (ops, I/O, latency)."""
    col_op = 12
    col_n = 8
    col_ok = 8
    col_fail = 7
    col_kv = 10
    col_bytes = 14
    col_ms = 9

    hdr = (
        f"{'Operation':<{col_op}}"
        f"{'Ops':>{col_n}}"
        f"{'OK':>{col_ok}}"
        f"{'Fail':>{col_fail}}"
        f"{'KV blk':>{col_kv}}"
        f"{'Bytes I/O':>{col_bytes}}"
        f"{'Avg ms':>{col_ms}}"
        f"{'P99 ms':>{col_ms}}"
        f"{'P99.9 ms':>{col_ms + 1}}"
        f"{'Hit%':>8}"
    )
    rule = "-" * len(hdr)
    print(hdr)
    print(rule)

    def row(label: str, op: Dict[str, Any], show_hit: bool) -> None:
        cnt = op.get("count", 0)
        ok = op.get("successful", 0)
        fail = op.get("failed", 0)
        kv = op.get("kv_blocks", 0)
        tb = op.get("total_bytes", 0)
        b_str = _format_bytes(tb) if tb else "0 B"
        if len(b_str) > col_bytes:
            b_str = b_str[: col_bytes - 2] + ".."
        avg = _fmt_ms_cell(op, "average_latency_ms", col_ms)
        p99 = _fmt_ms_cell(op, "latency_p99_ms", col_ms)
        p999 = _fmt_ms_cell(
            op, "latency_p999_ms", col_ms + 1
        )
        if show_hit and cnt > 0 and ok > 0:
            hr = op.get("hit_rate", 0.0)
            hit_cell = f"{hr * 100:>7.1f}%"
        else:
            hit_cell = f"{'—':>8}"
        print(
            f"{label:<{col_op}}"
            f"{cnt:>{col_n}d}"
            f"{ok:>{col_ok}d}"
            f"{fail:>{col_fail}d}"
            f"{kv:>{col_kv}d}"
            f"{b_str:>{col_bytes}s}"
            f"{avg}"
            f"{p99}"
            f"{p999}"
            f"{hit_cell}"
        )

    row("store", store, show_hit=False)
    row("retrieve", retrieve, show_hit=True)


def _print_cache_histograms(
    metrics: Dict[str, Any],
    store: Dict[str, Any],
    retrieve: Dict[str, Any],
) -> None:
    """ASCII bars: workload hit rate, retrieve outcomes,
    hits per KV block."""
    print("\n--- Cache & hits / KV block ---")

    ch = metrics.get("cache_hits", 0)
    cm = metrics.get("cache_misses", 0)
    tot_lm = ch + cm
    overall = metrics.get("cache_hit_rate", 0.0)
    bar_o = _ascii_ratio_bar(overall)
    print(
        "Workload cache hit rate (all ops) "
        f"[{ch} hit / {cm} miss]"
    )
    print(f"  {bar_o}  {overall * 100:5.1f}%")

    rh = retrieve.get("hits", 0)
    rm = retrieve.get("misses", 0)
    r_lookups = rh + rm
    if retrieve.get("count", 0) > 0 and r_lookups > 0:
        rr = retrieve.get("hit_rate", 0.0)
        bar_r = _ascii_ratio_bar(rr)
        print(
            "\nRetrieve prefix-cache (per retrieve op) "
            f"[{rh} hit / {rm} miss]"
        )
        print(f"  {bar_r}  {rr * 100:5.1f}%")

        hpb = _hits_per_block(rh, retrieve.get("kv_blocks", 0))
        if hpb is not None:
            # Bar scales 0–1: ratio is bounded when hits ≤ blocks.
            bar_h = _ascii_ratio_bar(min(1.0, hpb))
            print(
                "\nHits per KV block read (retrieve: "
                "hits ÷ Σ kv_blocks)"
            )
            print(f"  {bar_h}  {hpb:.4f}")
        else:
            print(
                "\nHits per KV block read: n/a "
                "(no KV blocks recorded)"
            )
    else:
        print(
            "\n(no retrieve operations — prefix-cache / "
            "hits-per-block charts omitted)"
        )


class WorkloadGenerator:
    """Generates and executes workloads for LMCache."""

    PATTERNS = {
        "random": RandomWorkload,
        "steady-state": SteadyStateWorkload,
        "conversation": ConversationWorkload,
    }

    def __init__(
        self,
        engine: Any = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Initialize workload generator.

        Args:
            engine: EngineManager instance for direct
                    engine calls
            tokenizer: Optional tokenizer wrapper for
                       text-to-tokens mode
        """
        self.engine = engine
        self.tokenizer = tokenizer

    def create_workload(
        self,
        pattern: str,
        chunk_index_dir: Optional[str] = None,
        chunk_index_file: Optional[str] = None,
        **kwargs,
    ) -> BaseWorkload:
        """
        Create a workload instance.

        Args:
            pattern: Workload pattern name (includes
                     ``store-only`` and ``retrieve-only``)
            chunk_index_dir: Directory containing
                CHUNK_TOKENS_INDEX_NAME (or parent of index)
            chunk_index_file: Explicit path to JSONL index file
            **kwargs: Pattern-specific arguments

        Returns:
            Workload instance
        """
        if pattern == "retrieve-only":
            if chunk_index_file:
                idx = Path(chunk_index_file)
            elif chunk_index_dir:
                idx = (
                    Path(chunk_index_dir)
                    / CHUNK_TOKENS_INDEX_NAME
                )
            else:
                raise ValueError(
                    "pattern retrieve-only requires "
                    "chunk_index_file or chunk_index_dir "
                    f"(default file {CHUNK_TOKENS_INDEX_NAME})"
                )
            if not idx.is_file():
                raise ValueError(
                    f"Chunk token index not found: {idx}"
                )
            return RetrieveOnlyWorkload(
                idx, engine=self.engine
            )

        if pattern == "store-only":
            if chunk_index_file:
                idx_out = Path(chunk_index_file)
            elif chunk_index_dir:
                idx_out = (
                    Path(chunk_index_dir)
                    / CHUNK_TOKENS_INDEX_NAME
                )
            else:
                raise ValueError(
                    "pattern store-only requires "
                    "chunk_index_file or chunk_index_dir "
                    f"(default file {CHUNK_TOKENS_INDEX_NAME})"
                )
            if self.tokenizer is not None:
                kwargs["tokenizer"] = self.tokenizer
            return StoreOnlyWorkload(
                idx_out,
                engine=self.engine,
                key_range=kwargs.get("key_range", 10000),
                value_size=kwargs.get("value_size", 1024),
                tokenizer=kwargs.get("tokenizer"),
                text_input=kwargs.get("text_input"),
            )

        if pattern not in self.PATTERNS:
            raise ValueError(
                f"Unknown pattern: {pattern}. "
                f"Available: "
                f"{list(self.PATTERNS.keys())}"
            )

        workload_class = self.PATTERNS[pattern]
        if self.tokenizer is not None:
            kwargs["tokenizer"] = self.tokenizer
        return workload_class(
            engine=self.engine, **kwargs
        )

    def run_workload(
        self,
        pattern: str,
        duration: Optional[float] = None,
        num_operations: Optional[int] = None,
        rate: Optional[float] = None,
        output_format: str = "json",
        passes: int = 1,
        chunk_index_dir: Optional[str] = None,
        chunk_index_file: Optional[str] = None,
        per_op_store_log: Optional[str] = None,
        **pattern_kwargs,
    ) -> Dict[str, Any]:
        """Run a workload pattern.

        Args:
            pattern: Workload pattern name
            duration: Duration in seconds
            num_operations: Number of operations
            rate: Operations per second
            output_format: Output format (json, text)
            passes: Number of passes over the dataset
            **pattern_kwargs: Pattern-specific args

        Returns:
            Metrics dictionary (aggregate or
            multi-pass)
        """
        passes = max(1, passes)
        workload = self.create_workload(
            pattern,
            chunk_index_dir=chunk_index_dir,
            chunk_index_file=chunk_index_file,
            **pattern_kwargs,
        )
        if per_op_store_log:
            workload.per_op_store_log = (
                per_op_store_log
            )

        if passes == 1:
            workload.run(
                duration=duration,
                num_operations=num_operations,
                rate=rate,
            )
            metrics = workload.get_metrics()
            if output_format == "json":
                print(json.dumps(metrics, indent=2))
            else:
                self._print_metrics(metrics)
            return metrics

        per_pass: list = []
        for p in range(1, passes + 1):
            print(
                f"\n>>> Pass {p}/{passes}",
                file=sys.stderr,
            )
            workload.metrics = (
                _base_module.WorkloadMetrics()
            )
            workload.metrics.pass_number = p
            if hasattr(workload, "reset"):
                workload.reset()
            workload.run(
                duration=duration,
                num_operations=num_operations,
                rate=rate,
            )
            per_pass.append(workload.get_metrics())

        combined = self._combine_passes(per_pass)
        if output_format == "json":
            print(json.dumps(combined, indent=2))
        else:
            self._print_multipass(per_pass, combined)
        return combined

    @staticmethod
    def _combine_passes(
        pass_metrics: list,
    ) -> Dict[str, Any]:
        """Aggregate per-pass metrics into a combined
        summary."""
        total_ops = sum(
            m["total_operations"]
            for m in pass_metrics
        )
        total_ok = sum(
            m["successful_operations"]
            for m in pass_metrics
        )
        total_fail = sum(
            m["failed_operations"]
            for m in pass_metrics
        )
        total_dur = sum(
            m["duration_seconds"]
            for m in pass_metrics
        )
        total_hits = sum(
            m["cache_hits"] for m in pass_metrics
        )
        total_misses = sum(
            m["cache_misses"] for m in pass_metrics
        )
        lookups = total_hits + total_misses
        return {
            "passes": len(pass_metrics),
            "total_duration_seconds": total_dur,
            "total_operations": total_ops,
            "successful_operations": total_ok,
            "failed_operations": total_fail,
            "cache_hits": total_hits,
            "cache_misses": total_misses,
            "cache_hit_rate": (
                total_hits / lookups
                if lookups else 0.0
            ),
            "per_pass": pass_metrics,
        }

    def _print_multipass(
        self,
        per_pass: list,
        combined: Dict[str, Any],
    ):
        """Print per-pass + aggregate metrics."""
        for m in per_pass:
            p = m.get("pass_number", "?")
            print(f"\n{'=' * 60}")
            print(f"Pass {p}")
            print(f"{'=' * 60}")
            self._print_pass_summary(m)

        print(f"\n{'=' * 60}")
        print("Aggregate (all passes)")
        print(f"{'=' * 60}")
        n = combined["passes"]
        dur = combined["total_duration_seconds"]
        print(f"Passes: {n}")
        print(f"Total Duration: {dur:.2f} s")
        print(
            f"Total Operations: "
            f"{combined['total_operations']}"
        )
        hits = combined["cache_hits"]
        misses = combined["cache_misses"]
        rate = combined["cache_hit_rate"]
        print(
            f"Cache Hits: {hits}  "
            f"Misses: {misses}  "
            f"Hit Rate: {rate:.2%}"
        )
        print(f"{'=' * 60}\n")

    def _print_pass_summary(
        self, metrics: Dict[str, Any]
    ):
        """Print a condensed single-pass summary."""
        dur = metrics["duration_seconds"]
        ops = metrics["total_operations"]
        tp = metrics["throughput_ops_per_sec"]
        hr = metrics["cache_hit_rate"]
        convs = metrics.get(
            "conversations_completed", ""
        )
        print(f"Duration: {dur:.2f} s")
        print(f"Operations: {ops}")
        print(f"Throughput: {tp:.2f} ops/sec")
        print(f"Hit Rate: {hr:.2%}")
        if convs:
            print(f"Conversations: {convs}")

    def _print_metrics(self, metrics: Dict[str, Any]):
        """Print metrics in human-readable format."""
        width = 92
        print("\n" + "=" * width)
        print("Workload Metrics")
        print("=" * width)
        dur = metrics["duration_seconds"]
        ops = metrics["total_operations"]
        ok = metrics["successful_operations"]
        fail = metrics["failed_operations"]
        tp = metrics["throughput_ops_per_sec"]
        print(
            f"Duration {dur:.2f} s   operations {ops}   "
            f"ok {ok}   failed {fail}   "
            f"{tp:.2f} ops/s"
        )
        convs = metrics.get(
            "conversations_completed", 0
        )
        if convs:
            print(f"Conversations completed: {convs}")

        print("\n--- Combined latency (all successful ops) ---")
        print(
            f"avg {metrics['average_latency_ms']:.2f} ms   "
            f"min {metrics['min_latency_ms']:.2f} ms   "
            f"max {metrics['max_latency_ms']:.2f} ms"
        )
        if "latency_p99_ms" in metrics:
            print(
                f"std {metrics['latency_std_ms']:.2f} ms   "
                f"P99 {metrics['latency_p99_ms']:.2f} ms   "
                f"P99.9 {metrics['latency_p999_ms']:.2f} ms"
            )

        store = metrics.get("store_operations", {})
        retrieve = metrics.get(
            "retrieve_operations", {}
        )

        print("\n--- Per operation type (I/O + latency) ---")
        _print_operation_stats_table(store, retrieve)
        _print_cache_histograms(metrics, store, retrieve)

        print("=" * width + "\n")
