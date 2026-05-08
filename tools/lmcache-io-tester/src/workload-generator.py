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
SequentialWorkload = _patterns_module.SequentialWorkload
BurstWorkload = _patterns_module.BurstWorkload
SteadyStateWorkload = _patterns_module.SteadyStateWorkload
ConversationWorkload = (
    _patterns_module.ConversationWorkload
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


class WorkloadGenerator:
    """Generates and executes workloads for LMCache."""

    PATTERNS = {
        "random": RandomWorkload,
        "sequential": SequentialWorkload,
        "burst": BurstWorkload,
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
        **kwargs,
    ) -> BaseWorkload:
        """
        Create a workload instance.

        Args:
            pattern: Workload pattern name
            **kwargs: Pattern-specific arguments

        Returns:
            Workload instance
        """
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
            pattern, **pattern_kwargs
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
        print("\n" + "=" * 60)
        print("Workload Metrics")
        print("=" * 60)
        dur = metrics['duration_seconds']
        print(f"Duration: {dur:.2f} seconds")
        ops = metrics['total_operations']
        print(f"Total Operations: {ops}")
        ok = metrics['successful_operations']
        print(f"Successful: {ok}")
        fail = metrics['failed_operations']
        print(f"Failed: {fail}")
        tp = metrics['throughput_ops_per_sec']
        print(f"Throughput: {tp:.2f} ops/sec")
        convs = metrics.get(
            "conversations_completed", 0
        )
        if convs:
            print(
                f"Conversations Completed: {convs}"
            )

        print("\n--- Overall Latency ---")
        print(f"Average: {metrics['average_latency_ms']:.2f} ms")
        print(f"Min: {metrics['min_latency_ms']:.2f} ms")
        print(f"Max: {metrics['max_latency_ms']:.2f} ms")

        # Operation type breakdown
        store = metrics.get("store_operations", {})
        retrieve = metrics.get("retrieve_operations", {})

        print("\n--- Operation Breakdown ---")
        print(f"Store:    {store.get('count', 0):>6} total"
              f"  ({store.get('successful', 0)} ok,"
              f" {store.get('failed', 0)} failed)")
        print(f"Retrieve: {retrieve.get('count', 0):>6} total"
              f"  ({retrieve.get('successful', 0)} ok,"
              f" {retrieve.get('failed', 0)} failed)")

        if store.get("successful", 0) > 0:
            print(f"\n--- Store Latency ---")
            print(f"Average: {store['average_latency_ms']:.2f} ms")
            print(f"Min: {store['min_latency_ms']:.2f} ms")
            print(f"Max: {store['max_latency_ms']:.2f} ms")

        if retrieve.get("successful", 0) > 0:
            print(f"\n--- Retrieve Latency ---")
            print(f"Average: {retrieve['average_latency_ms']:.2f} ms")
            print(f"Min: {retrieve['min_latency_ms']:.2f} ms")
            print(f"Max: {retrieve['max_latency_ms']:.2f} ms")

        # KV block stats
        print("\n--- KV Blocks ---")
        print(
            f"Blocks Written: "
            f"{store.get('kv_blocks', 0)}"
        )
        print(
            f"Blocks Read:    "
            f"{retrieve.get('kv_blocks', 0)}"
        )

        # Storage I/O
        bytes_w = store.get("total_bytes", 0)
        bytes_r = retrieve.get("total_bytes", 0)
        if bytes_w or bytes_r:
            print("\n--- Storage I/O ---")
            print(
                f"Written: {_format_bytes(bytes_w)}"
            )
            print(
                f"Read:    {_format_bytes(bytes_r)}"
            )

        # Cache stats (overall)
        print("\n--- Cache ---")
        print(f"Hits:     {metrics['cache_hits']}")
        print(f"Misses:   {metrics['cache_misses']}")
        print(f"Hit Rate: {metrics['cache_hit_rate']:.2%}")

        # Per-retrieve cache performance
        r_hits = retrieve.get("hits", 0)
        r_misses = retrieve.get("misses", 0)
        r_hit_rate = retrieve.get("hit_rate", 0.0)
        if retrieve.get("count", 0) > 0:
            print(
                "\n--- Retrieve Cache Performance ---"
            )
            print(f"Hits:     {r_hits}")
            print(f"Misses:   {r_misses}")
            print(f"Hit Rate: {r_hit_rate:.2%}")

        print("=" * 60 + "\n")
