"""
Load generator tools for benchmarking sweeps.
"""

from .multi_turn_benchmark import MultiTurnBenchmark
from .vllm_bench_serve import VllmBenchServe

# Registry of available load generators
LOAD_GENERATORS = {
    'multi-turn-benchmark': MultiTurnBenchmark,
    'vllm-bench-serve': VllmBenchServe,
}


def get_load_generator(tool_name: str, orchestrator):
    """
    Get a load generator instance by name.

    Args:
        tool_name: Name of the load generation tool
        orchestrator: SweepOrchestrator instance

    Returns:
        LoadGeneratorBase instance

    Raises:
        ValueError: If tool_name is not recognized
    """
    if tool_name not in LOAD_GENERATORS:
        available = ', '.join(LOAD_GENERATORS.keys())
        raise ValueError(f"Unknown load generation tool: {tool_name}. Available: {available}")

    generator_class = LOAD_GENERATORS[tool_name]
    return generator_class(orchestrator)
