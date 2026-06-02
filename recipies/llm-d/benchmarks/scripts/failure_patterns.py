"""
Log parsing patterns for detecting model server failures.

Defines regex patterns and categorization for common failure modes
observed in vLLM, SGLang, and other model serving containers.
"""

from health_monitor import FailureCategory, FailurePhase


# Error patterns for log analysis
# Each pattern includes: regex, category, phase, and description
ERROR_PATTERNS = [
    # OOM Errors
    {
        'pattern': r'ERROR.*Failed to load model.*out of memory',
        'category': FailureCategory.OOM.value,
        'phase': FailurePhase.MODEL_LOAD.value,
        'description': 'Model loading failed due to OOM'
    },
    {
        'pattern': r'(HIP|CUDA) out of memory',
        'category': FailureCategory.OOM.value,
        'phase': FailurePhase.MODEL_LOAD.value,
        'description': 'GPU out of memory error'
    },
    {
        'pattern': r'torch\.OutOfMemoryError',
        'category': FailureCategory.OOM.value,
        'phase': FailurePhase.MODEL_LOAD.value,
        'description': 'PyTorch OOM exception'
    },
    {
        'pattern': r'ERROR.*not enough GPU memory',
        'category': FailureCategory.OOM.value,
        'phase': FailurePhase.MODEL_LOAD.value,
        'description': 'Insufficient GPU memory for model'
    },

    # Engine/Core Failures
    {
        'pattern': r'ERROR.*EngineCore failed to start',
        'category': FailureCategory.CRASH.value,
        'phase': FailurePhase.MODEL_LOAD.value,
        'description': 'vLLM EngineCore initialization failed'
    },
    {
        'pattern': r'RuntimeError.*Engine core initialization failed',
        'category': FailureCategory.CRASH.value,
        'phase': FailurePhase.MODEL_LOAD.value,
        'description': 'Engine core failed to initialize'
    },
    {
        'pattern': r'ERROR.*Failed to load model',
        'category': FailureCategory.CRASH.value,
        'phase': FailurePhase.MODEL_LOAD.value,
        'description': 'Model loading failed'
    },

    # Python Exceptions/Tracebacks
    {
        'pattern': r'Traceback \(most recent call last\):.*(?:ERROR|Exception|Error)',
        'category': FailureCategory.CRASH.value,
        'phase': FailurePhase.UNKNOWN.value,
        'description': 'Python exception traceback detected'
    },
    {
        'pattern': r'CRITICAL.*',
        'category': FailureCategory.CRASH.value,
        'phase': FailurePhase.UNKNOWN.value,
        'description': 'Critical error logged'
    },

    # Model/Config Errors
    {
        'pattern': r'ERROR.*model not found',
        'category': FailureCategory.CONFIG_ERROR.value,
        'phase': FailurePhase.MODEL_LOAD.value,
        'description': 'Model not found or unavailable'
    },
    {
        'pattern': r'ERROR.*Failed to download',
        'category': FailureCategory.CONFIG_ERROR.value,
        'phase': FailurePhase.MODEL_LOAD.value,
        'description': 'Model download failed'
    },
    {
        'pattern': r'(ValueError|TypeError|KeyError|AttributeError):',
        'category': FailureCategory.CONFIG_ERROR.value,
        'phase': FailurePhase.MODEL_LOAD.value,
        'description': 'Python configuration error'
    },

    # NCCL/Communication Errors
    {
        'pattern': r'NCCL error',
        'category': FailureCategory.CRASH.value,
        'phase': FailurePhase.MODEL_LOAD.value,
        'description': 'NCCL communication error'
    },

    # Segmentation Faults
    {
        'pattern': r'Segmentation fault|segfault',
        'category': FailureCategory.CRASH.value,
        'phase': FailurePhase.UNKNOWN.value,
        'description': 'Segmentation fault detected'
    },

    # SIGKILL/SIGTERM
    {
        'pattern': r'(Killed|Terminated).*signal',
        'category': FailureCategory.CRASH.value,
        'phase': FailurePhase.UNKNOWN.value,
        'description': 'Process killed by signal'
    },

    # Benchmark-specific errors
    {
        'pattern': r'error: Internal error occurred: unable to upgrade connection: container not found',
        'category': FailureCategory.CONFIG_ERROR.value,
        'phase': FailurePhase.BENCHMARK.value,
        'description': 'Benchmark container not found'
    },

    # Model initialization errors
    {
        'pattern': r'ERROR.*initialize_model',
        'category': FailureCategory.CRASH.value,
        'phase': FailurePhase.MODEL_LOAD.value,
        'description': 'Model initialization failed'
    },

    # vLLM-specific compile errors
    {
        'pattern': r'ERROR.*compilation failed',
        'category': FailureCategory.CRASH.value,
        'phase': FailurePhase.MODEL_LOAD.value,
        'description': 'Model compilation failed'
    },

    # ROCm/HIP specific errors
    {
        'pattern': r'hipError|HIP Error|rocm error',
        'category': FailureCategory.CRASH.value,
        'phase': FailurePhase.MODEL_LOAD.value,
        'description': 'ROCm/HIP runtime error'
    },

    # Timeout errors
    {
        'pattern': r'TimeoutError|timeout exceeded',
        'category': FailureCategory.TIMEOUT.value,
        'phase': FailurePhase.UNKNOWN.value,
        'description': 'Operation timed out'
    },
]


# Whitelist patterns for non-fatal errors
# These patterns match log lines that would otherwise trigger a failure pattern
# but should be ignored because they are benign or expected conditions.
# Each entry has:
#   - pattern: regex that matches against the log line containing the error
#   - description: why this pattern is whitelisted
WHITELIST_PATTERNS = [
    {
        'pattern': r'ValueError.*exceeds model\'s maximum context length',
        'description': 'Input length exceeding context length is a request-level error, not a fatal server error'
    },
    {
        'pattern': r'ValueError.*Input length \(\d+\) exceeds model\'s maximum context length',
        'description': 'Specific format of input length exceeding context length error'
    },
    {
        'pattern': r'Error retrieving safetensors',
        'description': 'Non-fatal safetensors metadata retrieval warning'
    },
    {
        'pattern': r'Could not cache non-existence',
        'description': 'Non-fatal cache miss warning'
    },
    {
        'pattern': r'LMCache ERROR: PrometheusLogger instance already created with different metadata',
        'description': 'LMCache singleton warning during test - expected when multiple engines reuse metrics'
    },
]

def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text."""
    import re
    ansi_pattern = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_pattern.sub('', text)

def is_whitelisted(log_line: str, debug: bool = False) -> bool:
    """Check if a log line matches any whitelist pattern.
    Args:
        log_line: The log line to check (typically the matched line and surrounding context)
        debug: If True, print debug information
    Returns:
        True if the line matches a whitelist pattern and should be ignored
    """
    import re
    clean_line = strip_ansi_codes(log_line)
    if debug:
        print(f"DEBUG is_whitelisted: checking {len(WHITELIST_PATTERNS)} patterns")
        print(f"DEBUG is_whitelisted: log_line repr = {repr(clean_line[:100])}")
    for whitelist_entry in WHITELIST_PATTERNS:
        pattern = whitelist_entry['pattern']
        match = re.search(pattern, clean_line, re.IGNORECASE)
        if debug:
            print(f"DEBUG is_whitelisted: pattern={pattern[:50]}, match={bool(match)}")
        if match:
            return True
    return False


def get_patterns_by_category(category: FailureCategory) -> list:
    """Get all patterns for a specific failure category."""
    return [p for p in ERROR_PATTERNS if p['category'] == category.value]


def get_patterns_by_phase(phase: FailurePhase) -> list:
    """Get all patterns for a specific failure phase."""
    return [p for p in ERROR_PATTERNS if p['phase'] == phase.value]
