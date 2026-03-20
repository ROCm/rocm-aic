"""
Serving engine adapters for multi-engine deployment support.

This module provides a pluggable abstraction layer for different serving engines
(vLLM, SGLang, etc.), allowing the sweep orchestrator to work with multiple
engines without engine-specific logic in the core code.
"""

from .base import ServingEngineBase
from .vllm import VLLMEngine
from .sglang import SGLangEngine

# Registry of available serving engines
SERVING_ENGINES = {
    'vllm': VLLMEngine,
    'sglang': SGLangEngine,
}


def get_serving_engine(name: str) -> ServingEngineBase:
    """
    Factory function to get a serving engine adapter by name.

    Args:
        name: Engine name ('vllm', 'sglang', etc.)

    Returns:
        ServingEngineBase instance

    Raises:
        ValueError: If engine name is not recognized

    Examples:
        >>> engine = get_serving_engine('vllm')
        >>> engine.name
        'vllm'
        >>> engine.args_key
        'vllm_args'

        >>> engine = get_serving_engine('sglang')
        >>> engine.name
        'sglang'
        >>> engine.args_key
        'sglang_args'
    """
    if name not in SERVING_ENGINES:
        available = ', '.join(SERVING_ENGINES.keys())
        raise ValueError(
            f"Unknown serving engine: '{name}'. "
            f"Available engines: {available}"
        )

    engine_class = SERVING_ENGINES[name]
    return engine_class()


def list_available_engines():
    """Get list of all available engine names."""
    return list(SERVING_ENGINES.keys())


# Export public API
__all__ = [
    'ServingEngineBase',
    'VLLMEngine',
    'SGLangEngine',
    'get_serving_engine',
    'list_available_engines',
    'SERVING_ENGINES',
]
