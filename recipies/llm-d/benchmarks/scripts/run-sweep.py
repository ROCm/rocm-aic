#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""
Benchmarking sweep orchestrator.
Runs parameterized deployments, load generation, and result collection.

Supports both sequential and parallel execution with GPU budgeting.
"""

import yaml
import subprocess
import time
import json
import sys
import itertools
import os
import threading
import signal
import re
import ast
import operator
import fnmatch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from load_generators import get_load_generator
from serving_engines import get_serving_engine, ServingEngineBase
from health_monitor import (
    CentralizedHealthMonitor,
    DeploymentHealthCheckFailure,
    FailureInfo
)
from namespace_snapshot import NamespaceSnapshot
from sweep_state import RunState, RunStatus, write_state_file
from generate_summary import generate_summary_from_states, write_summary_file


# ============================================================================
# Safe Print Helper
# ============================================================================

def safe_print(*args, **kwargs):
    """Print that handles BrokenPipeError gracefully during cleanup.

    When stdout is broken (e.g., piped to tee which exited), tries stderr as fallback.
    """
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        # Stdout pipe was closed (e.g., tee exited after Ctrl-C)
        # Try stderr as fallback for critical messages
        try:
            # Only use stderr if explicitly not already specified
            if kwargs.get('file') is None:
                print(*args, **kwargs, file=sys.stderr)
        except:
            # Both pipes broken or stderr also specified - give up silently
            pass
    except Exception:
        # Ignore other print errors during cleanup
        pass


# ============================================================================
# Expression Evaluation for Variable Expansion
# ============================================================================

def has_variable_reference(value: Any) -> bool:
    """
    Check if a value contains variable references like {var_name}.

    Args:
        value: The value to check (typically string, int, float, or bool)

    Returns:
        True if the value is a string containing {variable} patterns
    """
    if not isinstance(value, str):
        return False
    return bool(re.search(r'\{[a-zA-Z_][a-zA-Z0-9_]*\}', value))


def extract_variable_names(expression: str) -> List[str]:
    """
    Extract all variable names from an expression string.

    Args:
        expression: String containing {var_name} patterns

    Returns:
        List of variable names (without braces)

    Example:
        extract_variable_names("30 * {max_concurrency}") -> ["max_concurrency"]
        extract_variable_names("{a} + {b}") -> ["a", "b"]
    """
    return re.findall(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}', expression)


def safe_eval_expression(expression: str, variables: Dict[str, Any]) -> Any:
    """
    Safely evaluate a mathematical expression with variable substitution.

    Uses Python's ast module to parse and evaluate only safe operations:
    - Arithmetic: +, -, *, /, //, %, **
    - Comparisons: <, <=, >, >=, ==, !=
    - Numbers: int, float
    - Boolean: and, or, not

    Args:
        expression: String expression with variables to evaluate
        variables: Dict mapping variable names to their values

    Returns:
        Evaluated result (typically int or float)

    Raises:
        ValueError: If expression contains unsafe operations or undefined variables

    Examples:
        safe_eval_expression("30 * {x}", {"x": 4}) -> 120
        safe_eval_expression("{a} + {b}", {"a": 10, "b": 20}) -> 30
    """
    # Replace {var} with variable values
    for var_name, var_value in variables.items():
        # Replace {var_name} with the actual value
        pattern = r'\{' + re.escape(var_name) + r'\}'
        expression = re.sub(pattern, str(var_value), expression)

    # Check if any unreplaced variables remain
    remaining_vars = extract_variable_names(expression)
    if remaining_vars:
        raise ValueError(f"Undefined variables in expression: {remaining_vars}")

    # Parse and validate the expression using AST
    try:
        node = ast.parse(expression, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {expression}") from e

    # Validate that only safe operations are used
    _validate_safe_ast(node)

    # Evaluate the expression
    try:
        result = eval(compile(node, '<string>', 'eval'), {"__builtins__": {}}, {})
        return result
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression '{expression}': {e}") from e


def _validate_safe_ast(node: ast.AST) -> None:
    """
    Validate that an AST only contains safe operations.

    Allowed:
    - BinOp: +, -, *, /, //, %, **
    - UnaryOp: +, -, not
    - Compare: <, <=, >, >=, ==, !=
    - BoolOp: and, or
    - Constant/Num: numbers, strings, booleans

    Raises:
        ValueError: If unsafe operations are detected
    """
    SAFE_NODES = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.BoolOp,
        ast.Constant,  # Python 3.8+
        ast.Num,       # Python 3.7 compatibility
        ast.Str,       # Python 3.7 compatibility
    )

    SAFE_OPERATORS = (
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.UAdd, ast.USub, ast.Not,
        ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
        ast.And, ast.Or,
    )

    for child in ast.walk(node):
        if not isinstance(child, SAFE_NODES + SAFE_OPERATORS):
            raise ValueError(
                f"Unsafe operation in expression: {child.__class__.__name__}. "
                f"Only arithmetic, comparison, and boolean operations are allowed."
            )


def evaluate_expressions_in_combination(combination: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate all expressions in a parameter combination, including nested structures.

    Recursively processes each value in the combination:
    - If it contains {variable} references, evaluates the expression
    - If it's a nested dict or list, recursively processes it
    - Otherwise, keeps the original value

    Args:
        combination: Dictionary of parameter names to values (may contain nested dicts/lists)

    Returns:
        New dictionary with expressions evaluated at all levels

    Examples:
        Simple top-level:
            Input:  {"max_concurrency": 16, "num_prompts": "30 * {max_concurrency}"}
            Output: {"max_concurrency": 16, "num_prompts": 480}

        Nested dict (e.g., lmcache_args):
            Input:  {"tensor_parallel_size": 4,
                     "lmcache_args": {"chunk_size": "256 * {tensor_parallel_size}"}}
            Output: {"tensor_parallel_size": 4,
                     "lmcache_args": {"chunk_size": 1024}}
    """
    # Helper to collect all top-level non-expression values as variables
    # Only top-level params can be referenced (e.g., {tensor_parallel_size})
    def collect_variables(data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect top-level scalar values to use as variables in expressions."""
        variables = {}
        for key, value in data.items():
            # Only collect top-level scalar values (not nested dicts/lists)
            # and only if they don't contain variable references themselves
            if not isinstance(value, (dict, list)) and not has_variable_reference(value):
                variables[key] = value
        return variables

    # Helper to recursively evaluate expressions in nested structures
    def evaluate_recursive(data: Any, variables: Dict[str, Any]) -> Any:
        """Recursively evaluate expressions in data structure."""
        if isinstance(data, dict):
            # Recursively process dictionary
            result = {}
            for key, value in data.items():
                result[key] = evaluate_recursive(value, variables)
            return result
        elif isinstance(data, list):
            # Recursively process list
            return [evaluate_recursive(item, variables) for item in data]
        elif has_variable_reference(data):
            # This is a string expression with variables - evaluate it
            try:
                return safe_eval_expression(data, variables)
            except ValueError as e:
                # Re-raise with context about where the error occurred
                raise ValueError(f"Error evaluating expression '{data}': {e}") from e
        else:
            # Scalar value without variables - return as-is
            return data

    # First pass: collect top-level non-expression values to use as variables
    variables = collect_variables(combination)

    # Second pass: recursively evaluate all expressions
    result = evaluate_recursive(combination, variables)

    return result


# ============================================================================
# End Expression Evaluation
# ============================================================================


# ============================================================================
# Environment Variable Substitution for ${VAR} Syntax
# ============================================================================

def has_env_var_reference(value: Any) -> bool:
    """
    Check if a value contains environment variable references like ${VAR}.

    Args:
        value: The value to check (typically string)

    Returns:
        True if the value contains ${...} patterns
    """
    if not isinstance(value, str):
        return False
    return bool(re.search(r'\$\{[A-Za-z_][A-Za-z0-9_]*(?::-[^}]*)?\}', value))


def substitute_env_vars(content: str, strict: bool = False) -> str:
    """
    Substitute ${VAR} and ${VAR:-default} patterns with environment variables.

    Supports:
    - ${VAR} - Direct substitution (error if missing and strict=True)
    - ${VAR:-default} - Substitution with default value

    Args:
        content: String content with ${VAR} placeholders
        strict: If True, raise error for undefined variables without defaults

    Returns:
        Content with environment variables substituted

    Raises:
        ValueError: If strict=True and a variable without default is undefined

    Examples:
        os.environ['API_KEY'] = 'secret123'
        substitute_env_vars('token: ${API_KEY}') -> 'token: secret123'
        substitute_env_vars('url: ${BASE_URL:-http://localhost}') -> 'url: http://localhost'
    """
    def replace_var(match):
        var_expr = match.group(1)  # Content inside ${}

        # Check for default value syntax: VAR:-default
        if ':-' in var_expr:
            var_name, default_value = var_expr.split(':-', 1)
            return os.environ.get(var_name, default_value)
        else:
            var_name = var_expr
            if var_name in os.environ:
                return os.environ[var_name]
            elif strict:
                raise ValueError(
                    f"Environment variable '${{{var_name}}}' is not defined. "
                    f"Set it in your environment or use '${{VAR:-default}}' syntax."
                )
            else:
                # Return unchanged if not strict
                return match.group(0)

    # Pattern: ${VAR} or ${VAR:-default}
    pattern = r'\$\{([A-Za-z_][A-Za-z0-9_]*(?::-[^}]*)?)\}'
    return re.sub(pattern, replace_var, content)


def substitute_env_vars_in_dict(data: Any, strict: bool = False) -> Any:
    """
    Recursively substitute environment variables in a nested dictionary/list structure.

    Args:
        data: Dictionary, list, or primitive value
        strict: If True, raise error for undefined variables

    Returns:
        Data structure with environment variables substituted
    """
    if isinstance(data, dict):
        return {key: substitute_env_vars_in_dict(value, strict) for key, value in data.items()}
    elif isinstance(data, list):
        return [substitute_env_vars_in_dict(item, strict) for item in data]
    elif isinstance(data, str):
        if has_env_var_reference(data):
            return substitute_env_vars(data, strict)
        return data
    else:
        return data


def load_yaml_config_with_env(config_file: str) -> Dict[str, Any]:
    """
    Load a YAML file after applying ${VAR} host environment substitution.

    Undefined variables are kept unchanged to match the existing sweep config
    behavior.
    """
    with open(config_file) as f:
        raw_content = f.read()

    substituted_content = substitute_env_vars(raw_content, strict=False)
    config = yaml.safe_load(substituted_content) or {}
    if not isinstance(config, dict):
        raise ValueError(f"YAML config must contain a mapping: {config_file}")
    return config


def resolve_runtime_config_paths(runtime_arg: Optional[str] = None) -> List[str]:
    """
    Resolve runtime YAML files in precedence order.

    The checked-in runtime-defaults.yaml is loaded first. A provided runtime
    file, or local runtime.yaml when no file is provided, overrides defaults.
    """
    benchmark_dir = Path(__file__).parent.parent
    runtime_paths: List[str] = []

    defaults_path = benchmark_dir / "runtime-defaults.yaml"
    if defaults_path.exists() and defaults_path.is_file():
        runtime_paths.append(str(defaults_path))

    if runtime_arg:
        runtime_path = Path(runtime_arg).expanduser()
        if runtime_path.exists() and runtime_path.is_file():
            runtime_paths.append(str(runtime_path))
            return runtime_paths
        raise FileNotFoundError(f"Runtime config file not found: {runtime_arg}")

    candidate = benchmark_dir / "runtime.yaml"
    if candidate.exists() and candidate.is_file():
        runtime_paths.append(str(candidate))
    return runtime_paths


def get_nested(config: Dict[str, Any], *keys: str) -> Any:
    """Return a nested runtime config value or None when any key is missing."""
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base without mutating either input."""
    merged = base.copy()
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_runtime_config(
    sweep_config: Dict[str, Any],
    runtime_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge runtime YAML defaults into a sweep config.

    Runtime env_vars are defaults for all runs. The sweep config keeps final
    precedence for any duplicate env var keys because it is closer to the test.
    """
    merged = sweep_config.copy()
    runtime_env_vars: Dict[str, Any] = {}

    for env_section in (
        runtime_config.get("env_vars"),
        get_nested(runtime_config, "sweep", "env_vars"),
    ):
        if env_section is None:
            continue
        if not isinstance(env_section, dict):
            raise ValueError("runtime env_vars must be a mapping")
        runtime_env_vars.update(env_section)

    if runtime_env_vars:
        config_env_vars = merged.get("env_vars", {})
        if config_env_vars is None:
            config_env_vars = {}
        if not isinstance(config_env_vars, dict):
            raise ValueError("sweep env_vars must be a mapping")
        merged["env_vars"] = {**runtime_env_vars, **config_env_vars}

    return merged

# ============================================================================
# End Environment Variable Substitution
# ============================================================================


class GPUBudgetScheduler:
    """Manages GPU budget and schedules configurations based on available resources."""

    def __init__(self, total_budget: int, max_concurrent: int, max_gpus_per_node: int = 8):
        """
        Initialize the GPU budget scheduler.

        Args:
            total_budget: Total number of GPUs available across the cluster
            max_concurrent: Maximum number of configurations to run concurrently (0 = unlimited)
            max_gpus_per_node: Maximum GPUs per node (for exclusive mode)
        """
        self.total_budget = total_budget
        self.max_concurrent = max_concurrent if max_concurrent > 0 else 999999
        self.max_gpus_per_node = max_gpus_per_node
        self.available_budget = total_budget
        self.lock = threading.Lock()
        self.budget_released = threading.Event()
        self.pending_queue: List[RunState] = []
        self.running: Dict[int, RunState] = {}  # run_id -> RunState
        self.completed: List[RunState] = []
        self.shutdown_requested = False

    def add_pending(self, run_state: RunState):
        """Add a configuration to the pending queue."""
        with self.lock:
            self.pending_queue.append(run_state)
            # Sort by GPU claim (smallest first) to maximize throughput
            self.pending_queue.sort(key=lambda x: x.gpu_claim)

    def try_schedule_next(self) -> Optional[RunState]:
        """
        Try to schedule the next configuration that fits in the budget.

        Returns:
            RunState if a configuration was scheduled, None otherwise
        """
        with self.lock:
            if self.shutdown_requested:
                return None

            # Check concurrent limit
            if len(self.running) >= self.max_concurrent:
                return None

            for i, run_state in enumerate(self.pending_queue):
                if run_state.gpu_claim <= self.available_budget:
                    # Schedule this configuration
                    self.pending_queue.pop(i)
                    self.available_budget -= run_state.gpu_claim
                    run_state.status = RunStatus.RUNNING
                    run_state.start_time = time.time()
                    self.running[run_state.run_id] = run_state
                    return run_state

            return None

    def release_resources(self, run_state: RunState):
        """Release GPU resources when a configuration completes."""
        with self.lock:
            if run_state.run_id in self.running:
                del self.running[run_state.run_id]

            run_state.end_time = time.time()
            self.completed.append(run_state)
            self.available_budget += run_state.gpu_claim

        # Signal that budget was released
        self.budget_released.set()

    def request_shutdown(self):
        """Request shutdown of the scheduler."""
        with self.lock:
            self.shutdown_requested = True
            # Cancel all pending runs
            for run_state in self.pending_queue:
                run_state.status = RunStatus.CANCELLED
                self.completed.append(run_state)
            self.pending_queue.clear()

        # Wake up any waiting threads
        self.budget_released.set()

    def get_running_states(self) -> List[RunState]:
        """Get list of currently running configurations."""
        with self.lock:
            return list(self.running.values())

    def get_pending_states(self) -> List[RunState]:
        """Get list of pending configurations."""
        with self.lock:
            return list(self.pending_queue)

    def get_completed_states(self) -> List[RunState]:
        """Get list of completed configurations."""
        with self.lock:
            return list(self.completed)

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        with self.lock:
            return self.shutdown_requested


class SweepOrchestrator:
    # Map deployment types to their justfile directories
    # Note: Only tiered-prefix-cache variants are supported for justfile-based deployment
    DEPLOYMENT_MAP = {
        'inference-scheduling-vllm': {
            'justfile_dir': 'inference-scheduling',
            'deploy_target': 'deploy-vllm',
            'teardown_target': 'teardown-vllm',
            'show_config_target': 'show-config-vllm',
            'use_justfile': True
        },
        'inference-scheduling-sglang': {
            'justfile_dir': 'inference-scheduling',
            'deploy_target': 'deploy-sglang',
            'teardown_target': 'teardown-sglang',
            'show_config_target': 'show-config-sglang',
            'use_justfile': True
        },
        'tiered-prefix-cache-offloading': {
            'justfile_dir': 'tiered-prefix-cache',
            'deploy_target': 'deploy-offloading',
            'teardown_target': 'teardown-offloading',
            'use_justfile': True
        },
        'tiered-prefix-cache-lmcache': {
            'justfile_dir': 'tiered-prefix-cache',
            'deploy_target': 'deploy-lmcache',
            'teardown_target': 'teardown-lmcache',
            'use_justfile': True
        },
        'tiered-prefix-cache-lmcache-ssd': {
            'justfile_dir': 'tiered-prefix-cache',
            'deploy_target': 'deploy-lmcache',
            'teardown_target': 'teardown-lmcache',
            'use_justfile': True
        }
    }

    def __init__(self, config_file: str, gpu_budget: Optional[int] = None,
                 max_concurrent: int = 1, exclusive_mode: bool = False,
                 max_gpus_per_node: int = 8, output_dir: Optional[str] = None,
                 runtime_file: Optional[str] = None):
        """
        Initialize the orchestrator.

        Args:
            config_file: Path to sweep configuration file
            gpu_budget: Total GPU budget (None = unlimited)
            max_concurrent: Maximum concurrent configurations (1 = sequential, 0 = unlimited)
            exclusive_mode: If True, pods request max GPUs per node
            max_gpus_per_node: Maximum GPUs available per node
            output_dir: Custom sweep directory name (optional, overrides auto-generated name)
            runtime_file: Optional host-specific runtime YAML file
        """
        self.config = load_yaml_config_with_env(config_file)
        self.runtime_config_files = resolve_runtime_config_paths(runtime_file)
        self.runtime_config_file = (
            self.runtime_config_files[-1] if self.runtime_config_files else None
        )
        self.runtime_overrides_env = any(
            Path(path).name != "runtime-defaults.yaml"
            for path in self.runtime_config_files
        )
        self.runtime_config: Dict[str, Any] = {}
        for runtime_config_file in self.runtime_config_files:
            runtime_part = load_yaml_config_with_env(runtime_config_file)
            self.runtime_config = deep_merge_dicts(self.runtime_config, runtime_part)
        if self.runtime_config_files:
            self.config = merge_runtime_config(self.config, self.runtime_config)
            print(f"Runtime config: {', '.join(self.runtime_config_files)}")

        # Validate model-specific args schema early
        self._validate_model_specific_args()

        # Validate and extract environment variables configuration
        self._validate_and_parse_env_vars()

        # Extract global env vars
        self.global_env_vars = self.config.get('env_vars', {})

        self.sweep_name = self.config['name']
        self.deployment = self.config['deployment']
        self.timestamp = datetime.now().strftime('%Y-%m-%d')

        # Local/explicit runtime YAML wins over stale host env. With only the
        # checked-in defaults present, host env remains an override.
        runtime_results_dir = (
            get_nested(self.runtime_config, 'sweep', 'results_dir')
            or get_nested(self.runtime_config, 'paths', 'sweep_results_dir')
        )
        if self.runtime_overrides_env:
            base_results_dir = runtime_results_dir or 'results/sweeps'
        else:
            base_results_dir = (
                os.environ.get('SWEEP_RESULTS_DIR')
                or runtime_results_dir
                or 'results/sweeps'
            )
        base_results_path = Path(base_results_dir)

        # Determine sweep directory name
        if output_dir:
            # Validate custom directory name
            if not self._is_valid_directory_name(output_dir):
                raise ValueError(
                    f"Invalid directory name: {output_dir}. "
                    "Use only alphanumeric characters, hyphens, underscores, and periods."
                )
            sweep_dir_name = output_dir
        else:
            # Auto-generate from sweep name and timestamp
            sweep_dir_name = f"{self.sweep_name}_{self.timestamp}"

        # Construct full results directory path
        self.results_dir = base_results_path / sweep_dir_name

        # Check if directory already exists
        if self.results_dir.exists():
            raise FileExistsError(
                f"Sweep directory already exists: {self.results_dir}\n"
                "Use a different --output-dir name or remove the existing directory."
            )

        # Create directory (including parent directories)
        self.results_dir.mkdir(parents=True, exist_ok=False)

        print(f"Results directory: {self.results_dir}")

        self.exclusive_mode = exclusive_mode
        self.max_gpus_per_node = max_gpus_per_node
        self.parallel_mode = max_concurrent != 1

        # Get current user for namespace prefix
        self.user_id = self._get_user_id()

        # Initialize serving engine adapter
        engine_name = self.config.get('serving_engine', 'vllm')  # Default to vllm
        self.serving_engine: ServingEngineBase = get_serving_engine(engine_name)
        print(f"Using serving engine: {self.serving_engine.display_name}")

        # Get deployment configuration
        self.deployment_config = self._get_deployment_config()

        # Initialize GPU budget scheduler (even for sequential mode)
        if gpu_budget is None:
            gpu_budget = 999999  # Effectively unlimited
        self.scheduler = GPUBudgetScheduler(gpu_budget, max_concurrent, max_gpus_per_node)

        # Save sweep metadata with runtime configuration
        metadata = self.config.copy()
        metadata['_runtime_config'] = {
            'gpu_budget': gpu_budget,
            'max_concurrent': max_concurrent,
            'exclusive_mode': exclusive_mode,
            'max_gpus_per_node': max_gpus_per_node,
            'user_id': self.user_id,
            'timestamp': self.timestamp,
            'runtime_config_files': self.runtime_config_files
        }
        with open(self.results_dir / "metadata.yaml", 'w') as f:
            yaml.dump(metadata, f)

        # State file for tracking runs
        self.state_file = self.results_dir / "state.json"

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.shutdown_event = threading.Event()
        self.shutdown_in_progress = False  # Track if we're already shutting down

        # Initialize health monitoring (configurable via config file)
        health_config = self.config.get('health_monitoring', {})
        self.health_monitoring_enabled = health_config.get('enabled', True)  # Default: enabled
        if self.health_monitoring_enabled:
            self.health_monitor = CentralizedHealthMonitor(
                check_interval=health_config.get('check_interval', 15),
                aggressive_timeout=health_config.get('aggressive_timeout', 60),
                api_rate_limit=health_config.get('api_rate_limit', 10)
            )
        else:
            self.health_monitor = None

    def _is_valid_directory_name(self, name: str) -> bool:
        """
        Validate that directory name contains only safe characters.

        Args:
            name: Directory name to validate

        Returns:
            True if valid, False otherwise
        """
        import re
        # Allow alphanumeric, hyphens, underscores, and periods
        # Disallow path separators and other special characters
        pattern = r'^[a-zA-Z0-9._-]+$'
        if not re.match(pattern, name):
            return False
        # Disallow directory names that look like paths
        if '/' in name or '\\' in name:
            return False
        return True

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        # If already shutting down, ignore additional Ctrl-C to let cleanup finish
        if self.shutdown_in_progress:
            safe_print("\n" + "!"*70)
            safe_print("INTERRUPT ALREADY IN PROGRESS")
            safe_print("Please wait for cleanup to complete (deleting namespaces...)")
            safe_print("Interrupting cleanup may leave resources orphaned!")
            safe_print("!"*70)
            return

        # First interrupt - initiate graceful shutdown
        self.shutdown_in_progress = True
        print("\n" + "="*70)
        print("INTERRUPT RECEIVED - Initiating graceful shutdown...")
        print("="*70)
        print("Cancelling pending configurations and cleaning up running ones...")
        print("(Press Ctrl-C again to force exit - may leave resources orphaned)")
        print("="*70)

        self.shutdown_event.set()
        self.scheduler.request_shutdown()

        # Signal health monitor to stop (non-blocking)
        if self.health_monitoring_enabled and self.health_monitor:
            print("Signaling health monitor to stop...")
            self.health_monitor.stop_event.set()  # Just set the flag, don't wait

        # Replace signal handler to warn on subsequent interrupts
        signal.signal(signal.SIGINT, self._cleanup_signal_handler)
        signal.signal(signal.SIGTERM, self._cleanup_signal_handler)

    def _cleanup_signal_handler(self, signum, frame):
        """Handle signals during cleanup - warn but allow force exit."""
        safe_print("\n" + "!"*70)
        safe_print("SECOND INTERRUPT RECEIVED - FORCING EXIT")
        safe_print("WARNING: Cleanup interrupted - namespaces may not be deleted!")
        safe_print("You may need to manually run teardown to clean up resources.")
        safe_print("!"*70)
        sys.exit(130)

    def _validate_model_specific_args(self) -> None:
        """
        Validate model_specific_args schema in the configuration.

        Raises ValueError if the schema is invalid.
        """
        model_specific = self.config.get('parameters', {}).get('model_specific_args', [])

        if not model_specific:
            return  # Optional section, no validation needed

        if not isinstance(model_specific, list):
            raise ValueError(
                "model_specific_args must be a list of override specifications"
            )

        for idx, override in enumerate(model_specific):
            if not isinstance(override, dict):
                raise ValueError(
                    f"model_specific_args[{idx}]: Each override must be a dictionary"
                )

            # Pattern is required
            if 'pattern' not in override:
                raise ValueError(
                    f"model_specific_args[{idx}]: Missing required 'pattern' field"
                )

            if not isinstance(override['pattern'], str):
                raise ValueError(
                    f"model_specific_args[{idx}]: 'pattern' must be a string"
                )

            # Conditions is optional but must be a dict if present
            if 'conditions' in override and not isinstance(override['conditions'], dict):
                raise ValueError(
                    f"model_specific_args[{idx}]: 'conditions' must be a dictionary"
                )

            # At least one engine args section (vllm_args or sglang_args) must be present
            has_engine_args = any(
                key in override for key in ['vllm_args', 'sglang_args']
            )

            if not has_engine_args:
                raise ValueError(
                    f"model_specific_args[{idx}]: Must specify at least one of "
                    "'vllm_args' or 'sglang_args'"
                )

            # Validate engine args are dictionaries
            for args_key in ['vllm_args', 'sglang_args']:
                if args_key in override and not isinstance(override[args_key], dict):
                    raise ValueError(
                        f"model_specific_args[{idx}]: '{args_key}' must be a dictionary"
                    )

    def _validate_and_parse_env_vars(self) -> None:
        """
        Validate environment variables schema in the configuration.

        Validates both:
        - Top-level env_vars (global)
        - Parameter-level env_vars (per-combination)

        Raises ValueError if the schema is invalid.
        """
        # Validate top-level env_vars
        global_env_vars = self.config.get('env_vars', {})

        if global_env_vars:
            if not isinstance(global_env_vars, dict):
                raise ValueError("Top-level 'env_vars' must be a dictionary")

            for key, value in global_env_vars.items():
                if not isinstance(key, str):
                    raise ValueError(f"Environment variable key must be string: {key}")
                # Values can be any type (string, int, bool, etc.)
                # They will be converted to strings when injected

        # Validate parameter-level env_vars
        params = self.config.get('parameters', {})
        for param_name, param_spec in params.items():
            if not isinstance(param_spec, dict):
                continue

            # Check if this parameter has env_vars
            if 'env_vars' in param_spec:
                env_vars = param_spec['env_vars']

                if not isinstance(env_vars, (dict, list)):
                    raise ValueError(
                        f"Parameter '{param_name}': env_vars must be dict or list of dicts"
                    )

                # If it's a list (for combinations/categorical), validate each entry
                if isinstance(env_vars, list):
                    for idx, env_vars_entry in enumerate(env_vars):
                        if not isinstance(env_vars_entry, dict):
                            raise ValueError(
                                f"Parameter '{param_name}': env_vars[{idx}] must be a dict"
                            )

    def _get_user_id(self) -> str:
        """Get current user ID for namespace prefix."""
        # Try to get from environment, fall back to whoami
        user = os.environ.get('USER') or os.environ.get('USERNAME')
        if not user:
            try:
                result = subprocess.run(['whoami'], capture_output=True, text=True, check=True)
                user = result.stdout.strip()
            except:
                user = 'unknown'
        # Sanitize for Kubernetes namespace (lowercase, no special chars)
        return user.lower().replace('_', '-').replace('.', '-')

    def _get_deployment_config(self) -> Dict[str, str]:
        """Get deployment configuration from DEPLOYMENT_MAP.

        Returns empty dict with use_justfile=False for unknown deployment types,
        which will fall back to direct kubectl commands.
        """
        if self.deployment not in self.DEPLOYMENT_MAP:
            print(f"  Note: Deployment type '{self.deployment}' not in justfile map.")
            print(f"  Falling back to direct kubectl commands.")
            print(f"  Supported justfile deployments: {list(self.DEPLOYMENT_MAP.keys())}")
            return {'use_justfile': False}
        return self.DEPLOYMENT_MAP[self.deployment]

    def _get_justfile_path(self) -> Path:
        """Get path to the justfile directory for this deployment."""
        # Get the script directory
        script_dir = Path(__file__).parent.parent
        # Navigate to the deployment's justfile directory
        justfile_dir = script_dir.parent / self.deployment_config['justfile_dir']
        return justfile_dir

    def _generate_namespace(self, run_id: int) -> str:
        """Generate unique namespace for a run."""
        # Format: <user>-<sweep-name>-<timestamp>-<run-id>
        # Keep it under 63 chars for K8s limit
        ns = f"{self.user_id}-{self.sweep_name}-{self.timestamp}-{run_id:03d}"
        # Ensure it's valid (lowercase, alphanumeric + hyphens, starts/ends with alphanumeric)
        ns = ns[:63].rstrip('-')
        return ns

    def _apply_model_specific_args(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply model-specific argument overrides based on pattern matching.

        This allows injecting engine-specific arguments (vllm_args, sglang_args)
        conditionally based on the model name or other parameter values.

        The model_specific_args section in the config should have this structure:

        model_specific_args:
          - pattern: "amd/Qwen3-235B*"
            vllm_args:
              kv_cache_dtype: "fp8"
          - pattern: "moonshotai/Kimi-K2.5"
            conditions:
              tensor_parallel_size: 4
            vllm_args:
              max_num_seqs: 256
              block_size: 64

        Args:
            params: Parameter combination dictionary

        Returns:
            Updated parameter dictionary with model-specific args applied
        """
        model_specific = self.config.get('parameters', {}).get('model_specific_args', [])

        if not model_specific:
            return params

        model_name = params.get('model', '')

        for override in model_specific:
            pattern = override.get('pattern', '')
            conditions = override.get('conditions', {})

            # Check if model matches the pattern (supports wildcards)
            if not fnmatch.fnmatch(model_name, pattern):
                continue

            # Check additional conditions (e.g., tensor_parallel_size, gpu_memory_utilization)
            conditions_met = True
            for cond_key, cond_value in conditions.items():
                if params.get(cond_key) != cond_value:
                    conditions_met = False
                    break

            if not conditions_met:
                continue

            # Apply engine-specific argument overrides
            # Support both vllm_args and sglang_args
            for args_key in ['vllm_args', 'sglang_args']:
                if args_key in override:
                    # Initialize args dict if it doesn't exist
                    if args_key not in params:
                        params[args_key] = {}

                    # Deep merge: update existing args with new ones
                    # New values take precedence (override existing)
                    params[args_key].update(override[args_key])

        return params

    def generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations to test.

        This generates the Cartesian product of:
        - Deployment parameter combinations (vllm_args, lmcache_args, etc.)
        - Load generation benchmark_args combinations (if type: combinations or pairwise)

        Supports the consistent pattern where benchmark_args can have:
        - type: combinations - Cartesian product of mixed sweepable (values) and fixed parameters
        - type: pairwise - Pair-wise zip of sweepable parameters (stops at shortest list)
        - Direct dict (backward compatibility)
        - sweep_args (backward compatibility - deprecated)
        """
        params = self.config['parameters']
        deployment_combinations = []

        # Separate fixed and variable deployment parameters
        fixed = {}
        variable = {}

        for name, spec in params.items():
            # Skip model_specific_args - it's not a regular parameter
            if name == 'model_specific_args':
                continue

            if spec['type'] == 'fixed':
                fixed[name] = spec['value']
            elif spec['type'] == 'categorical':
                variable[name] = spec['values']
            elif spec['type'] in ['combinations', 'pairwise']:
                # Handle all args combinations (vllm_args, lmcache_args, custom args, etc.)
                # Generate combinations using generic method
                combination_mode = 'pairwise' if spec['type'] == 'pairwise' else 'product'
                args_combinations = self._generate_args_combinations(spec['args'], combination_mode)
                variable[name] = args_combinations

        # Generate all combinations of variable deployment parameters
        if variable:
            keys = list(variable.keys())
            values = [variable[k] for k in keys]
            for combo in itertools.product(*values):
                config = fixed.copy()
                config.update(dict(zip(keys, combo)))
                # Evaluate any expressions in this deployment combination
                # This allows top-level parameters (cpu, memory, etc.) to reference
                # other parameters using {var_name} syntax
                config = evaluate_expressions_in_combination(config)
                deployment_combinations.append(config)
        else:
            # Even for single fixed config, evaluate expressions
            # (e.g., cpu: "{tensor_parallel_size} * 16" where tensor_parallel_size is fixed)
            deployment_combinations = [evaluate_expressions_in_combination(fixed)]

        # Check if load_generation has benchmark_args with type: combinations
        load_config = self.config.get('load_generation', {})
        benchmark_args_spec = load_config.get('benchmark_args', {})

        load_combinations = []

        # New pattern: benchmark_args with type: combinations or pairwise
        if isinstance(benchmark_args_spec, dict) and benchmark_args_spec.get('type') in ['combinations', 'pairwise']:
            # Check for explicit combination_mode field first, then fall back to type
            if 'combination_mode' in benchmark_args_spec:
                combination_mode = benchmark_args_spec['combination_mode']
            else:
                combination_mode = 'pairwise' if benchmark_args_spec.get('type') == 'pairwise' else 'product'
            load_combinations = self._generate_args_combinations(benchmark_args_spec['args'], combination_mode)
        # Backward compatibility: sweep_args (deprecated)
        elif 'sweep_args' in load_config:
            load_combinations = self._generate_args_combinations(load_config['sweep_args'])
        # No sweep - single run
        else:
            load_combinations = [{}]

        # Cartesian product: Each deployment config × Each load config
        full_combinations = []
        for deploy_params in deployment_combinations:
            for load_params in load_combinations:
                combo = deploy_params.copy()
                # Store load params separately to merge into benchmark_args later
                combo['_load_params'] = load_params

                # Merge environment variables (global + combination-level)
                combo['_env_vars'] = self._merge_env_vars(combo)

                full_combinations.append(combo)

        # Apply model-specific argument overrides
        full_combinations = [
            self._apply_model_specific_args(combo)
            for combo in full_combinations
        ]

        return full_combinations

    def _generate_args_combinations(self, args_spec: Dict[str, Any], combination_mode: str = 'product') -> List[Dict[str, Any]]:
        """
        Generic expansion of args with 'values' lists into combinations.

        Works for vllm_args, lmcache_args, and any other args that follow
        the pattern of fixed values and sweepable 'values' lists.

        Now supports nested dictionaries with sweepable values (e.g., extra_config.gds_io_threads).

        Supports variable expansion: parameters can reference other parameters
        using {var_name} syntax, e.g., "30 * {max_concurrency}"

        Args:
            args_spec: Dictionary where values can be:
                - Fixed values (any type)
                - Dicts with 'values' key containing a list
                - Nested dicts (for complex config structures with sweepable nested values)
                - Expressions with {var_name} references
            combination_mode: How to combine sweepable parameters:
                - 'product': Cartesian product (default)
                - 'pairwise': Pair-wise zip of values

        Returns:
            List of dictionaries with all combinations and expressions evaluated
        """
        # Helper function to extract sweepable paths from nested dicts
        def extract_sweepable_paths(d: Dict[str, Any], prefix: str = '') -> Dict[str, List[Any]]:
            """Extract all paths that have 'values' lists, including nested ones."""
            sweepable = {}
            for key, value in d.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    if 'values' in value:
                        # This is a sweepable parameter
                        sweepable[path] = value['values']
                    else:
                        # Recurse into nested dict
                        sweepable.update(extract_sweepable_paths(value, path))
            return sweepable

        # Helper function to set a value in a nested dict by path
        def set_nested_value(d: Dict[str, Any], path: str, value: Any) -> None:
            """Set a value in a nested dictionary using dot notation path."""
            keys = path.split('.')
            current = d
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value

        # Helper function to remove sweepable paths from nested dict while preserving structure
        def remove_sweepable_nested(d: Dict[str, Any], prefix: str, sweepable: Dict[str, List[Any]]) -> Dict[str, Any]:
            """Remove sweepable paths from nested dict while preserving structure."""
            import copy
            result = {}
            for key, value in d.items():
                path = f"{prefix}.{key}" if prefix else key
                if path in sweepable:
                    # This path is sweepable, skip it
                    continue
                elif isinstance(value, dict):
                    if 'values' in value:
                        # This is a sweepable value, skip it
                        continue
                    else:
                        # Recurse into nested dict
                        cleaned = remove_sweepable_nested(value, path, sweepable)
                        if cleaned:  # Only add if not empty
                            result[key] = cleaned
                else:
                    # Fixed value
                    result[key] = copy.deepcopy(value)
            return result

        # Extract all sweepable parameters (including nested ones)
        sweepable = extract_sweepable_paths(args_spec)

        # If no sweepable parameters, return single config
        if not sweepable:
            return [args_spec]

        # Create a base config with all non-sweepable values (cleaned)
        import copy
        base_config = {}
        for key, value in args_spec.items():
            if isinstance(value, dict) and 'values' not in value:
                # This is a nested dict, need to clean it recursively
                # but preserve structure, removing only paths that will be swept
                base_config[key] = remove_sweepable_nested(value, key, sweepable)
            elif isinstance(value, dict) and 'values' in value:
                # Skip - this is a top-level sweepable
                pass
            else:
                # Fixed value
                base_config[key] = value

        # Generate combinations of sweepable parameters
        combinations = []
        paths = list(sweepable.keys())
        values_lists = [sweepable[p] for p in paths]

        if combination_mode == 'pairwise':
            # Pair-wise: zip values together (stops at shortest list)
            for values in zip(*values_lists):
                # Start with a deep copy of base config
                config = copy.deepcopy(base_config)
                # Set each swept value
                for path, value in zip(paths, values):
                    set_nested_value(config, path, value)
                # Evaluate any expressions in this combination
                config = evaluate_expressions_in_combination(config)
                combinations.append(config)
        else:
            # Default: Cartesian product
            for values in itertools.product(*values_lists):
                # Start with a deep copy of base config
                config = copy.deepcopy(base_config)
                # Set each swept value
                for path, value in zip(paths, values):
                    set_nested_value(config, path, value)
                # Evaluate any expressions in this combination
                config = evaluate_expressions_in_combination(config)
                combinations.append(config)

        return combinations

    def _merge_env_vars(self, combination: Dict[str, Any]) -> Dict[str, str]:
        """
        Merge global and combination-level environment variables.

        Combination-level env_vars override global env_vars for same keys.

        Args:
            combination: Parameter combination dictionary

        Returns:
            Merged dictionary of environment variables (all values as strings)
        """
        # Start with global env vars
        merged_env_vars = self.global_env_vars.copy()

        # Get parameter configuration
        params = self.config.get('parameters', {})

        # Collect env_vars from parameters in this combination
        for param_name, param_value in combination.items():
            # Skip internal keys and non-parameter fields
            if param_name.startswith('_'):
                continue

            # Get the parameter specification
            param_spec = params.get(param_name)
            if not param_spec or not isinstance(param_spec, dict):
                continue

            # Check if this parameter has env_vars
            if 'env_vars' not in param_spec:
                continue

            param_env_vars = param_spec['env_vars']

            # Handle dict format (for fixed parameters)
            if isinstance(param_env_vars, dict):
                merged_env_vars.update(param_env_vars)

            # Handle list format (for categorical/combinations parameters)
            elif isinstance(param_env_vars, list):
                # Find the index of the current value in the parameter values
                if param_spec['type'] == 'categorical':
                    values = param_spec['values']
                    try:
                        idx = values.index(param_value)
                        if idx < len(param_env_vars):
                            env_vars_for_value = param_env_vars[idx]
                            if isinstance(env_vars_for_value, dict):
                                merged_env_vars.update(env_vars_for_value)
                    except (ValueError, IndexError):
                        # Value not found or index out of range, skip
                        pass

        # Convert all values to strings (required for Kubernetes env)
        return {k: str(v) for k, v in merged_env_vars.items()}

    def build_engine_args(self, args: Dict[str, Any]) -> str:
        """
        Build CLI arguments for the current serving engine.

        Args:
            args: Native engine arguments (e.g., vllm_args or sglang_args)

        Returns:
            Formatted CLI string
        """
        return self.serving_engine.build_server_args(args)

    def generate_timestamp_seed(self) -> int:
        """
        Generate a seed from current timestamp.

        Uses HHMMSS format (hour, minute, second) to create a 6-digit integer.
        This provides reasonable uniqueness while being deterministic within a second.

        Returns:
            Integer seed (e.g., 145934 for 14:59:34)
        """
        return int(time.strftime("%H%M%S"))

    def _convert_args_dict(self, args_dict: Dict[str, Any]) -> List[str]:
        """
        Convert a dictionary to command-line arguments (tool-agnostic).

        Handles:
        - Underscore to hyphen conversion
        - Boolean flags (present if True, omitted if False)
        - Regular key-value pairs
        - Null values (skipped)
        """
        args = []

        for key, value in args_dict.items():
            # Convert underscore to hyphen
            arg_name = key.replace("_", "-")

            # Handle boolean flags
            if isinstance(value, bool):
                if value:
                    args.append(f"--{arg_name}")
                continue

            # Handle null values (skip them)
            if value is None:
                continue

            # Handle regular key-value arguments
            args.extend([f"--{arg_name}", str(value)])

        return args

    def _generate_env_vars_patches(self, env_vars: Dict[str, str]) -> List[str]:
        """
        Generate Kubernetes environment variable patch operations for Kustomization.

        Creates YAML-formatted patch operations to inject environment variables
        into container spec.

        Args:
            env_vars: Dictionary of environment variable names to values

        Returns:
            List of YAML-formatted strings representing patch operations.
            Each item in the list is a complete patch operation as a multi-line string.
            The leading dash will be added by replace_template_variables().

        Example:
            Input:  {'API_KEY': 'secret', 'DEBUG': 'true'}
            Output: ['op: add\npath: /spec/template/spec/containers/0/env/-\nvalue:\n  name: API_KEY\n  value: "secret"',
                     'op: add\npath: /spec/template/spec/containers/0/env/-\nvalue:\n  name: DEBUG\n  value: "true"']
        """
        if not env_vars:
            return []

        patches = []
        for name, value in sorted(env_vars.items()):  # Sort for deterministic output
            # Escape quotes in value and wrap in quotes
            escaped_value = str(value).replace('"', '\\"')

            # Create complete patch operation as a single multi-line string
            patch = f'op: add\npath: /spec/template/spec/containers/0/env/-\nvalue:\n  name: {name}\n  value: "{escaped_value}"'
            patches.append(patch)

        return patches

    def replace_template_variables(self, template_content: str, params: Dict[str, Any]) -> str:
        """
        Replace {{PLACEHOLDER}} variables in template with actual values.

        Handles both simple values and list values (for ENGINE_ARGS_ARRAY).

        Args:
            template_content: Template file content with {{VAR}} placeholders
            params: Dictionary of parameter values

        Returns:
            Rendered content with placeholders replaced
        """
        rendered = template_content

        for key, value in params.items():
            placeholder = f"{{{{{key}}}}}"
            if placeholder in rendered:
                # For array values, use YAML formatting with proper indentation
                if isinstance(value, list):
                    # Split into lines and process line by line
                    lines = rendered.split('\n')
                    new_lines = []

                    for line in lines:
                        if placeholder in line:
                            # Get the indentation (everything before the placeholder)
                            indent = line[:line.index(placeholder)]

                            # Generate YAML list items, each properly indented
                            for item in value:
                                # Handle multi-line items (e.g., env_vars patches)
                                item_str = str(item)
                                if '\n' in item_str:
                                    # Multi-line item: indent all lines properly
                                    item_lines = item_str.split('\n')
                                    # First line gets the list dash
                                    new_lines.append(f'{indent}- {item_lines[0]}')
                                    # Subsequent lines get indented to align with first line content
                                    for item_line in item_lines[1:]:
                                        new_lines.append(f'{indent}  {item_line}')
                                else:
                                    # Single-line item
                                    new_lines.append(f'{indent}- {item}')
                        else:
                            new_lines.append(line)

                    rendered = '\n'.join(new_lines)
                else:
                    rendered = rendered.replace(placeholder, str(value))

        return rendered

    def render_template(self, params: Dict[str, Any], run_dir: Path) -> Path:
        """
        Render deployment template with parameters.

        Supports:
        - Template variable replacement ({{model}}, {{tensor_parallel_size}}, etc.)
        - vllm_args expansion
        - lmcache_args expansion with ConfigMap injection
        - Optional deployment_template override for custom template filenames
        """
        # Get absolute path to template file (relative to benchmarks directory)
        script_dir = Path(__file__).parent.parent

        # Check if user provided a custom deployment_template
        deployment_template = self.config.get('deployment_template')
        if deployment_template:
            # Use the provided template filename directly
            template_file = script_dir / "templates" / deployment_template
        else:
            # Use automatic inference based on deployment name
            template_file = script_dir / "templates" / f"{self.deployment}-kustomization.yaml.tmpl"

        output_dir = run_dir / "manifests-serving"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare template parameters
        template_params = params.copy()

        # Debug: Show what parameters we received (helps diagnose issues)
        print(f"    Template rendering - received params: {list(params.keys())}")
        if 'lmcache_args' in params:
            print(f"    lmcache_args present: {params['lmcache_args']}")

        # Process engine-specific args (vllm_args, sglang_args, etc.)
        engine_args_key = self.serving_engine.args_key  # e.g., 'vllm_args' or 'sglang_args'
        if engine_args_key in params:
            engine_args_array = self.serving_engine.build_server_args_array(params[engine_args_key])
            template_params["ENGINE_ARGS_ARRAY"] = engine_args_array

        # Process environment variables
        if '_env_vars' in params and params['_env_vars']:
            env_vars_patches = self._generate_env_vars_patches(params['_env_vars'])
            template_params["ENV_VARS_PATCH"] = env_vars_patches
        else:
            template_params["ENV_VARS_PATCH"] = []

        # Extract lmcache_args and bench_args for special handling
        lmcache_args = template_params.pop('lmcache_args', None)
        bench_args = template_params.pop('bench_args', None)

        # Debug: Show bench_args if present
        if bench_args:
            print(f"    bench_args present: {bench_args}")

        # Load template file
        with open(template_file) as f:
            content = f.read()

        # Step 1: Always do standard template variable replacement first
        rendered = self.replace_template_variables(content, template_params)

        # Step 2: If we have lmcache_args or bench_args, inject the ConfigMaps
        if lmcache_args or bench_args:
            from lmcache_template_injection import inject_bench_configmap_yaml_parse
            from lmcache_template_injection import inject_lmcache_configmap_into_rendered

            # Inject LMCache config if present
            if lmcache_args:
                rendered = inject_lmcache_configmap_into_rendered(
                    rendered,
                    lmcache_args,
                    use_yaml_parse=True  # Use YAML parsing for precision
                )

            # Inject bench config if present
            if bench_args:
                rendered = inject_bench_configmap_yaml_parse(rendered, bench_args)

        # Write output
        output_file = output_dir / "kustomization.yaml"
        with open(output_file, 'w') as f:
            f.write(rendered)

        return output_dir

    def calculate_gpu_claim(self, params: Dict[str, Any], manifest_dir: Path) -> int:
        """
        Calculate GPU claim for a configuration.

        In exclusive mode, returns replicas * max_gpus_per_node.
        Otherwise, parses the tensor_parallel_size from parameters.

        Args:
            params: Configuration parameters
            manifest_dir: Directory containing rendered manifests

        Returns:
            Number of GPUs claimed by this configuration
        """
        if self.exclusive_mode:
            # In exclusive mode, each replica claims the full node
            # Assume 1 replica for now (can be extended to parse replica count)
            replicas = 1
            return replicas * self.max_gpus_per_node

        # Parse the tensor_parallel_size from parameters
        # This is the standard way GPU count is specified
        tensor_parallel_size = params.get('tensor_parallel_size', 1)

        # For now, assume 1 replica. This could be extended to parse
        # the kustomization.yaml for replica count if needed.
        replicas = 1

        return replicas * tensor_parallel_size

    def apply_exclusive_mode_patches(self, manifest_dir: Path):
        """
        Apply exclusive mode patches to rendered kustomization.

        Modifies GPU resource requests to max_gpus_per_node while preserving
        all vllm arguments and other configuration.

        Args:
            manifest_dir: Directory containing kustomization.yaml
        """
        kustomization_file = manifest_dir / "kustomization.yaml"

        if not kustomization_file.exists():
            return

        # Read the kustomization
        with open(kustomization_file) as f:
            content = f.read()

        # Replace GPU resource limits/requests with max_gpus_per_node
        # This uses string replacement to preserve the YAML structure
        import re

        # Pattern to match the GPU resource in the patches section
        # We look for: amd.com/gpu: "NUMBER"
        pattern = r'(amd\.com/gpu:\s*")[0-9]+"'
        replacement = f'\\1{self.max_gpus_per_node}"'

        modified_content = re.sub(pattern, replacement, content)

        # Write back the modified kustomization
        with open(kustomization_file, 'w') as f:
            f.write(modified_content)

        print(f"    Applied exclusive mode: GPU resources set to {self.max_gpus_per_node}")

    def save_state(self):
        """Save current state to JSON file."""
        write_state_file(
            self.state_file,
            pending=self.scheduler.get_pending_states(),
            running=self.scheduler.get_running_states(),
            completed=self.scheduler.get_completed_states()
        )

    def create_namespace(self, namespace: str):
        """Create a Kubernetes namespace with deployment labels."""
        print(f"  Creating namespace {namespace}...")

        # Create namespace with labels
        result = subprocess.run([
            "kubectl", "create", "namespace", namespace,
            "--dry-run=client", "-o", "yaml"
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to generate namespace YAML: {result.stderr}")

        # Parse YAML and add labels
        import yaml as yaml_lib
        ns_manifest = yaml_lib.safe_load(result.stdout)

        # Add deployment label
        if 'metadata' not in ns_manifest:
            ns_manifest['metadata'] = {}
        if 'labels' not in ns_manifest['metadata']:
            ns_manifest['metadata']['labels'] = {}

        ns_manifest['metadata']['labels']['llm-d.ai/deployment'] = self.DEPLOYMENT_MAP[self.deployment]['justfile_dir']

        # Apply the namespace
        result = subprocess.run([
            "kubectl", "apply", "-f", "-"
        ], input=yaml_lib.dump(ns_manifest), text=True, capture_output=True)

        if result.returncode != 0:
            # Check if it already exists or unchanged
            if "unchanged" not in result.stdout:
                raise RuntimeError(f"Failed to create namespace: {result.stderr}")
            print(f"  Namespace already exists or unchanged, continuing...")

    def _resolve_hf_token(self) -> Optional[str]:
        """Resolve HF token from env, token file, or runtime YAML."""
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            return hf_token

        hf_token_file = (
            os.environ.get('HF_TOKEN_FILE')
            or get_nested(self.runtime_config, 'secrets', 'hf_token_file')
        )
        if hf_token_file:
            token_path = Path(str(hf_token_file)).expanduser()
            if not token_path.exists():
                raise RuntimeError(f"HF_TOKEN_FILE does not exist: {hf_token_file}")
            if not token_path.is_file():
                raise RuntimeError(f"HF_TOKEN_FILE is not a file: {hf_token_file}")
            try:
                return token_path.read_text().strip()
            except OSError as exc:
                raise RuntimeError(
                    f"HF_TOKEN_FILE is not readable: {hf_token_file}"
                ) from exc

        runtime_token = get_nested(self.runtime_config, 'secrets', 'hf_token')
        if runtime_token:
            return str(runtime_token)

        return None

    def inject_hf_token_secret(self, namespace: str):
        """Inject HuggingFace token secret into namespace."""
        hf_token = self._resolve_hf_token()
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN is not set. Set HF_TOKEN, HF_TOKEN_FILE, or "
                "secrets.hf_token_file in runtime YAML before running sweeps."
            )

        print(f"  Injecting HF token secret...")

        # Create secret
        result = subprocess.run([
            "kubectl", "create", "secret", "generic", "llm-d-hf-token",
            f"--from-literal=HF_TOKEN={hf_token}",
            "-n", namespace
        ], capture_output=True, text=True)

        if result.returncode != 0:
            # Check if it already exists
            if "already exists" in result.stderr:
                print(f"  HF secret already exists, continuing...")
            else:
                raise RuntimeError(f"Failed to create HF secret: {result.stderr}")

    def delete_namespace(self, namespace: str):
        """Delete a Kubernetes namespace."""
        safe_print(f"  Deleting namespace {namespace}...")

        result = subprocess.run([
            "kubectl", "delete", "namespace", namespace, "--wait=true"
        ], capture_output=True, text=True)

        if result.returncode != 0:
            safe_print(f"  Warning: Failed to delete namespace: {result.stderr}")

    def deploy(self, manifest_dir: Path, namespace: str):
        """Deploy using justfile targets with parametrized manifests."""
        print(f"  Deploying to namespace {namespace}...")

        if not self.deployment_config.get('use_justfile', False):
            # Fall back to direct kubectl for non-justfile deployments
            result = subprocess.run([
                "kubectl", "apply", "-k", str(manifest_dir), "-n", namespace
            ], capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Deployment failed: {result.stderr}")

            print(f"  Deployment applied successfully (kubectl)")
            return

        # Use justfile-based deployment
        justfile_dir = self._get_justfile_path()
        deploy_target = self.deployment_config['deploy_target']

        # Convert manifest_dir to absolute path for justfile
        manifest_dir_abs = str(manifest_dir.absolute())

        # Run the deploy target from the justfile with MANIFEST_DIR and NAMESPACE
        env = {**os.environ, 'NAMESPACE': namespace, 'MANIFEST_DIR': manifest_dir_abs}
        result = subprocess.run([
            "just",
            deploy_target
        ], capture_output=True, text=True, cwd=str(justfile_dir), env=env)

        if result.returncode != 0:
            print(f"  Deploy stdout: {result.stdout}")
            print(f"  Deploy stderr: {result.stderr}")
            raise RuntimeError(f"Deployment failed: {result.stderr}")

        print(f"  Deployment command completed successfully (justfile)")

    def wait_for_deployment(self, namespace: str, run_id: int):
        """Wait for deployment to be ready with health monitoring.

        Args:
            namespace: Kubernetes namespace
            run_id: Run ID for health monitor tracking

        Raises:
            DeploymentHealthCheckFailure: If health check detects failure
            RuntimeError: If wait command fails or times out
        """
        print(f"  Waiting for deployment to be ready...")

        # Register namespace for health monitoring
        if self.health_monitoring_enabled and self.health_monitor:
            self.health_monitor.register_namespace(namespace, run_id)

        try:
            # Start wait process based on deployment type
            if not self.deployment_config.get('use_justfile', False):
                # Direct kubectl wait for non-justfile deployments
                wait_process = self._start_kubectl_wait_async(namespace)
            else:
                # Justfile-based wait
                wait_process = self._start_justfile_wait_async(namespace)

            # Poll both wait process and health monitor
            while wait_process.poll() is None:
                # Check if health monitor detected failure
                if self.health_monitoring_enabled and self.health_monitor:
                    failure_info = self.health_monitor.check_namespace_health(namespace)
                    if failure_info:
                        # Kill the wait process
                        wait_process.terminate()
                        try:
                            wait_process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            wait_process.kill()

                        # Raise with detailed diagnostics
                        raise DeploymentHealthCheckFailure(
                            f"Pod health check failed: {failure_info.message}",
                            failure_info=failure_info
                        )

                time.sleep(2)  # Poll every 2 seconds

            # Wait process completed - check exit code
            if wait_process.returncode != 0:
                stdout, stderr = wait_process.communicate()
                # Collect diagnostics from health monitor
                diagnostics = ""
                if self.health_monitoring_enabled and self.health_monitor:
                    diag_dict = self.health_monitor.collect_diagnostics(namespace)
                    diagnostics = f"\n\nDiagnostics: {json.dumps(diag_dict, indent=2)}"

                raise RuntimeError(f"Wait for deployment failed: {stderr}{diagnostics}")

            # Success - but do final health check
            if self.health_monitoring_enabled and self.health_monitor:
                failure_info = self.health_monitor.check_namespace_health(namespace)
                if failure_info:
                    raise DeploymentHealthCheckFailure(
                        "Health check failed after deployment reported ready",
                        failure_info=failure_info
                    )

            mode = "justfile" if self.deployment_config.get('use_justfile', False) else "kubectl"
            print(f"  Deployment ready! ({mode})")

        except DeploymentHealthCheckFailure:
            # Re-raise health check failures
            raise
        except Exception as e:
            # For other exceptions, don't unregister yet - keep monitoring
            raise

    def _start_kubectl_wait_async(self, namespace: str) -> subprocess.Popen:
        """Start kubectl wait as async subprocess."""
        cmd = [
            "kubectl", "wait", "--for=condition=available",
            "deployment",
            "-l", "llm-d.ai/role=decode",
            "-n", namespace,
            "--timeout=600s"  # 10 minutes
        ]

        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    def _start_justfile_wait_async(self, namespace: str) -> subprocess.Popen:
        """Start justfile wait as async subprocess."""
        justfile_dir = self._get_justfile_path()

        cmd = ["just", "wait"]
        env = {**os.environ, 'NAMESPACE': namespace}

        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(justfile_dir),
            env=env
        )

    def run_load_generation(self, run_dir: Path, params: Dict[str, Any], namespace: str) -> Dict[str, Any]:
        """
        Run load generation using pluggable load generator modules with health monitoring.

        Args:
            run_dir: Directory for saving results
            params: Deployment parameters
            namespace: Kubernetes namespace

        Returns:
            Dictionary with results from the load generator

        Raises:
            RuntimeError: If pods fail during benchmark execution
        """
        load_config = self.config['load_generation']
        tool = load_config['tool']

        print(f"  Running load generation with {tool}...")

        # Get the appropriate load generator
        generator = get_load_generator(tool, self)

        # Run the load generator
        result = generator.run(load_config, run_dir, params, namespace)

        # Check for failures during benchmark
        if self.health_monitoring_enabled and self.health_monitor:
            failure_info = self.health_monitor.check_namespace_health(namespace)
            if failure_info:
                raise RuntimeError(
                    f"Pods failed during benchmark: {failure_info.message} "
                    f"(category: {failure_info.category}, phase: {failure_info.phase})"
                )

        return result

    def check_model_server_health(self, namespace: str) -> tuple[Optional[str], list[str]]:
        """
        Check model server pods for errors after benchmark completion.

        Returns tuple of (error_message, list of pod names with issues).
        Returns (None, []) if healthy.
        """
        try:
            # Get model server pods
            result = subprocess.run([
                "kubectl", "get", "pods",
                "-n", namespace,
                "-l", "llm-d.ai/role=decode",
                "-o", "json"
            ], capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                return (f"Failed to get pod status: {result.stderr}", [])

            pods_data = json.loads(result.stdout)
            errors = []
            problem_pods = []

            for pod in pods_data.get('items', []):
                pod_name = pod['metadata']['name']
                phase = pod['status'].get('phase', 'Unknown')
                container_statuses = pod.get('status', {}).get('containerStatuses', [])
                pod_has_issue = False

                # Check 1: Pod phase
                if phase in ['Failed', 'Unknown']:
                    errors.append(f"Pod {pod_name} in {phase} state")
                    pod_has_issue = True

                # Check 2: Container statuses
                for container_status in container_statuses:
                    container_name = container_status.get('name', 'unknown')

                    # Check for restarts
                    restart_count = container_status.get('restartCount', 0)
                    if restart_count > 0:
                        errors.append(f"Container '{container_name}' in pod {pod_name} restarted {restart_count} time(s)")
                        pod_has_issue = True

                    # Check for waiting/terminated states
                    state = container_status.get('state', {})

                    if 'waiting' in state:
                        reason = state['waiting'].get('reason', '')
                        message = state['waiting'].get('message', '')
                        if reason in ['CrashLoopBackOff', 'ImagePullBackOff', 'ErrImagePull']:
                            errors.append(f"Container '{container_name}' in pod {pod_name}: {reason} - {message}")
                            pod_has_issue = True

                    if 'terminated' in state:
                        reason = state['terminated'].get('reason', '')
                        message = state['terminated'].get('message', '')
                        exit_code = state['terminated'].get('exitCode', 0)
                        if exit_code != 0:
                            errors.append(f"Container '{container_name}' in pod {pod_name} terminated (exit {exit_code}): {reason} - {message}")
                            pod_has_issue = True

                # Check 3: Recent ERROR lines in logs
                log_result = subprocess.run([
                    "kubectl", "logs", pod_name, "-n", namespace, "--tail=100"
                ], capture_output=True, text=True, timeout=10)

                if log_result.returncode == 0:
                    error_lines = [
                        line.strip() for line in log_result.stdout.split('\n')
                        if 'ERROR' in line
                        and 'Error retrieving safetensors' not in line
                        and 'Could not cache non-existence' not in line
                    ]

                    if error_lines:
                        # Take first few errors
                        sample_errors = error_lines[:3]
                        errors.append(f"ERROR lines found in {pod_name}: {'; '.join(sample_errors)}")
                        pod_has_issue = True

                if pod_has_issue and pod_name not in problem_pods:
                    problem_pods.append(pod_name)

            if errors:
                return ("Model server issues detected:\n  - " + "\n  - ".join(errors), problem_pods)

            return (None, [])

        except Exception as e:
            return (f"Error checking model server health: {e}", [])

    def capture_snapshot(self, run_dir: Path, namespace: str, context: str = "unknown"):
        """
        Capture comprehensive namespace snapshot.

        Args:
            run_dir: Directory for this run's results
            namespace: Kubernetes namespace to snapshot
            context: Context for logging (e.g., "success", "failure", "health_check_failure")
        """
        print(f"  Capturing namespace snapshot (context: {context})...")

        snapshot_dir = run_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Create and execute snapshot
        snapshot = NamespaceSnapshot()
        results = snapshot.capture(
            namespace=namespace,
            output_dir=snapshot_dir,
            label_selector="llm-d.ai/role=decode"
        )

        # Log results
        for step_name, result in results.items():
            if result.status == "success":
                print(f"    ✓ {step_name}: {len(result.files_created)} files")
            elif result.status == "partial":
                print(f"    ⚠ {step_name}: Partial - {result.error_message}")
            else:
                print(f"    ✗ {step_name}: Failed - {result.error_message}")

        return results

    def teardown(self, namespace: str):
        """Teardown deployment using justfile targets."""
        print(f"  Tearing down deployment...")

        if self.deployment_config.get('use_justfile', False):
            # Use justfile-based teardown
            justfile_dir = self._get_justfile_path()
            teardown_target = self.deployment_config['teardown_target']

            # Run the teardown target from the justfile
            result = subprocess.run([
                "just",
                teardown_target
            ], capture_output=True, text=True, cwd=str(justfile_dir),
               env={**os.environ, 'NAMESPACE': namespace})

            if result.returncode != 0:
                print(f"  Warning: Teardown had errors: {result.stderr}")
                # Don't raise exception, teardown is best-effort

        # Delete the namespace (for all deployment types)
        self.delete_namespace(namespace)

    def dry_run(self):
        """Print sweep configuration with GPU budget analysis."""
        combinations = self.generate_parameter_combinations()

        print("=" * 70)
        print(f"DRY RUN: {self.sweep_name}")
        print("=" * 70)
        print(f"Description: {self.config['description']}")
        print(f"Deployment: {self.deployment}")
        print(f"User ID: {self.user_id}")
        print(f"GPU Budget: {self.scheduler.total_budget}")
        print(f"Max Concurrent: {self.scheduler.max_concurrent if self.scheduler.max_concurrent < 999999 else 'unlimited'}")
        print(f"Execution Mode: {'Parallel' if self.parallel_mode else 'Sequential'}")
        print(f"Exclusive Mode: {self.exclusive_mode}")
        if self.exclusive_mode:
            print(f"Max GPUs per Node: {self.max_gpus_per_node}")
        print(f"Total configurations: {len(combinations)}")
        print("=" * 70)
        print()

        for i, params in enumerate(combinations, 1):
            namespace = self._generate_namespace(i)
            print(f"Configuration {i}/{len(combinations)}")
            print("-" * 70)
            print(f"  namespace: {namespace}")

            # Print fixed parameters
            if 'model' in params:
                print(f"  model: {params['model']}")
            if 'tensor_parallel_size' in params:
                print(f"  tensor_parallel_size: {params['tensor_parallel_size']}")

            # Print vllm_args
            if 'vllm_args' in params:
                print(f"  vllm_args:")
                for key, value in params['vllm_args'].items():
                    if key == 'kv_connector':
                        if value is None:
                            print(f"    kv_connector: null  # No caching")
                        else:
                            print(f"    kv_connector:")
                            print(f"      type: {value.get('type', 'unknown')}")
                            if 'cpu_bytes' in value:
                                gb = value['cpu_bytes'] / (1024**3)
                                print(f"      cpu_bytes: {value['cpu_bytes']} ({gb:.0f}GB)")
                            if 'role' in value:
                                print(f"      role: {value['role']}")
                            if 'config_file' in value:
                                print(f"      config_file: {value['config_file']}")
                            if 'raw_json' in value:
                                print(f"      raw_json: {value['raw_json'][:50]}...")
                    else:
                        print(f"    {key}: {value}")

            # Show generated serving engine command
            engine_args_key = self.serving_engine.args_key
            if engine_args_key in params:
                print()
                print(f"  Generated {self.serving_engine.display_name} command:")
                engine_args_str = self.serving_engine.build_server_args(
                    params[engine_args_key]
                )
                print(f"    {self.serving_engine.name} serve {params.get('model', '<model>')} \\")
                print(f"      --tensor-parallel-size {params.get('tensor_parallel_size', 1)} \\")
                print(f"      --port 8000 \\")
                for line in engine_args_str.split('\n'):
                    if line.strip():
                        print(f"      {line.strip()}")

            # Show load generation config
            load_config = self.config.get('load_generation', {})
            if load_config:
                print()
                print("  Load generation:")
                print(f"    tool: {load_config.get('tool', 'unknown')}")
                benchmark_args = load_config.get('benchmark_args', {})
                if benchmark_args:
                    print(f"    benchmark_args:")
                    for key, value in benchmark_args.items():
                        print(f"      {key}: {value}")
            print()

        # Add GPU budget analysis
        total_gpu_claim = 0
        max_gpu_claim = 0
        gpu_distribution = {}

        for i, params in enumerate(combinations, 1):
            # Create a temporary run dir to render template
            temp_run_dir = self.results_dir / f"_dry_run_{i}"
            temp_run_dir.mkdir(exist_ok=True)

            try:
                manifest_dir = self.render_template(params, temp_run_dir)
                gpu_claim = self.calculate_gpu_claim(params, manifest_dir)

                if self.exclusive_mode:
                    # Recalculate for exclusive mode
                    gpu_claim = self.max_gpus_per_node

                total_gpu_claim += gpu_claim
                max_gpu_claim = max(max_gpu_claim, gpu_claim)

                if gpu_claim not in gpu_distribution:
                    gpu_distribution[gpu_claim] = 0
                gpu_distribution[gpu_claim] += 1

                print(f"  gpu_claim: {gpu_claim}")

            finally:
                # Clean up temporary directory
                import shutil
                if temp_run_dir.exists():
                    shutil.rmtree(temp_run_dir)

        print("=" * 70)
        print("GPU BUDGET ANALYSIS")
        print("=" * 70)
        print(f"Total GPU Budget: {self.scheduler.total_budget}")
        print(f"Maximum single config GPU claim: {max_gpu_claim}")
        print(f"Total GPU claims (if all run sequentially): {total_gpu_claim}")
        print(f"\nGPU Distribution:")
        for gpu_count in sorted(gpu_distribution.keys()):
            count = gpu_distribution[gpu_count]
            print(f"  {gpu_count} GPUs: {count} configurations")

        print("\nBUDGET VALIDATION:")
        if max_gpu_claim > self.scheduler.total_budget:
            print(f"  ✗ ERROR: At least one configuration requires {max_gpu_claim} GPUs")
            print(f"           but budget is only {self.scheduler.total_budget} GPUs")
            print(f"           These configurations cannot run!")
            sys.exit(1)
        else:
            print(f"  ✓ All configurations fit within budget")

        if self.parallel_mode:
            print("\nESTIMATED PARALLELISM:")
            avg_claim = total_gpu_claim / len(combinations)
            estimated_parallel = int(self.scheduler.total_budget / avg_claim)
            print(f"  Average GPU claim per config: {avg_claim:.1f}")
            print(f"  Estimated average parallelism: {estimated_parallel} configs")

        print("=" * 70)
        print(f"SUMMARY: {len(combinations)} configurations validated")
        print("=" * 70)

    def execute_run(self, run_state: RunState):
        """
        Execute a single configuration run.

        This is called for each configuration (in a worker thread for parallel mode).
        """
        run_id = run_state.run_id
        namespace = run_state.namespace
        params = run_state.parameters

        prefix = f"[Run {run_id}]" if self.parallel_mode else f"Run {run_id}/{self.total_runs}"

        print(f"\n{prefix} Starting execution (GPUs: {run_state.gpu_claim})")
        print(f"{prefix} Namespace: {namespace}")

        run_dir = self.results_dir / f"run-{run_id:03d}"
        run_dir.mkdir(exist_ok=True)

        # Save configuration with namespace and GPU claim
        config_to_save = params.copy()
        config_to_save['namespace'] = namespace
        config_to_save['gpu_claim'] = run_state.gpu_claim
        config_to_save['exclusive_mode'] = self.exclusive_mode
        with open(run_dir / "config.yaml", 'w') as f:
            yaml.dump(config_to_save, f)

        try:
            # Check for shutdown before each major step
            if self.shutdown_event.is_set():
                raise InterruptedError("Shutdown requested")

            print(f"{prefix} Creating namespace...")
            self.create_namespace(namespace)

            if self.shutdown_event.is_set():
                raise InterruptedError("Shutdown requested")

            print(f"{prefix} Injecting HF token secret...")
            self.inject_hf_token_secret(namespace)

            if self.shutdown_event.is_set():
                raise InterruptedError("Shutdown requested")

            print(f"{prefix} Rendering deployment templates...")
            manifest_dir = self.render_template(params, run_dir)

            # Apply exclusive mode patches if needed
            if self.exclusive_mode:
                self.apply_exclusive_mode_patches(manifest_dir)

            if self.shutdown_event.is_set():
                raise InterruptedError("Shutdown requested")

            print(f"{prefix} Deploying to Kubernetes...")
            self.deploy(manifest_dir, namespace)

            if self.shutdown_event.is_set():
                raise InterruptedError("Shutdown requested")

            print(f"{prefix} Waiting for deployment to be ready...")
            self.wait_for_deployment(namespace, run_id)

            if self.shutdown_event.is_set():
                raise InterruptedError("Shutdown requested")

            print(f"{prefix} Running load generation...")
            # Add run_id to params for unique results directory naming
            params_with_run_id = params.copy()
            params_with_run_id['_run_id'] = run_id
            benchmark_results = self.run_load_generation(run_dir, params_with_run_id, namespace)

            if self.shutdown_event.is_set():
                raise InterruptedError("Shutdown requested")

            self.capture_snapshot(run_dir, namespace, context="success")

            # Check model server health after benchmark completion
            print(f"{prefix} Checking model server health...")
            server_health_error, problem_pods = self.check_model_server_health(namespace)

            # Update run state
            run_state.benchmark_results = benchmark_results

            # Fail if benchmark failed OR if model server had issues
            if benchmark_results['exit_code'] != 0:
                run_state.status = RunStatus.FAILED
                run_state.error = "Benchmark failed"
                print(f"{prefix} ✗ Benchmark failed")
            elif server_health_error:
                run_state.status = RunStatus.FAILED
                run_state.error = f"Model server error: {server_health_error}"
                print(f"{prefix} ✗ Model server errors detected:")
                # Print each error line with indentation
                for line in server_health_error.split('\n'):
                    print(f"{prefix}   {line}")

                # Print pod log paths for easy access
                if problem_pods:
                    print(f"{prefix}   ")
                    print(f"{prefix}   Pod logs saved to:")
                    for pod_name in problem_pods:
                        log_path = run_dir / "snapshots" / f"{pod_name}.log"
                        print(f"{prefix}     {log_path}")
                        # Also print previous log if it exists
                        prev_log_path = run_dir / "snapshots" / f"{pod_name}-previous.log"
                        if prev_log_path.exists():
                            print(f"{prefix}     {prev_log_path}")
            else:
                run_state.status = RunStatus.COMPLETED
                print(f"{prefix} ✓ Completed successfully")

        except InterruptedError:
            run_state.status = RunStatus.CANCELLED
            run_state.error = "Cancelled by user"
            print(f"{prefix} ✗ Cancelled")

        except DeploymentHealthCheckFailure as e:
            # Health check detected a failure
            run_state.status = RunStatus.FAILED
            run_state.error = str(e)
            run_state.failure_info = e.failure_info.to_dict() if e.failure_info else None
            print(f"{prefix} ✗ Health check failed: {e}")

            # Collect snapshot for debugging
            try:
                self.capture_snapshot(run_dir, namespace, context="health_check_failure")
            except Exception as log_error:
                print(f"{prefix} Warning: Failed to capture snapshot: {log_error}")

            # Save detailed diagnostics to run directory
            if e.failure_info:
                diagnostics_file = run_dir / "failure_diagnostics.json"
                with open(diagnostics_file, 'w') as f:
                    json.dump(e.failure_info.to_dict(), f, indent=2)
                print(f"{prefix}   Diagnostics saved to: {diagnostics_file}")

        except Exception as e:
            run_state.status = RunStatus.FAILED
            run_state.error = str(e)
            print(f"{prefix} ✗ Failed: {e}")

            # Collect snapshot for debugging
            try:
                self.capture_snapshot(run_dir, namespace, context="failure")
            except Exception as log_error:
                print(f"{prefix} Warning: Failed to capture snapshot: {log_error}")

        finally:
            # Unregister from health monitoring
            if self.health_monitoring_enabled and self.health_monitor:
                self.health_monitor.unregister_namespace(namespace)

            # Always teardown and release resources
            try:
                safe_print(f"{prefix} Tearing down...")
                self.teardown(namespace)
            except Exception as e:
                safe_print(f"{prefix} Warning: Teardown failed: {e}")

            # Release GPU resources
            self.scheduler.release_resources(run_state)

            # Save state after each completion
            self.save_state()

    def worker_thread(self):
        """Worker thread that continuously tries to schedule and execute runs."""
        while not self.shutdown_event.is_set():
            # Try to schedule next run
            run_state = self.scheduler.try_schedule_next()

            if run_state is not None:
                # Execute the run
                self.execute_run(run_state)
            else:
                # No run could be scheduled, wait for budget to be released
                # or shutdown to be requested
                self.scheduler.budget_released.wait(timeout=1.0)
                self.scheduler.budget_released.clear()

            # Check if all work is done
            with self.scheduler.lock:
                if (len(self.scheduler.pending_queue) == 0 and
                    len(self.scheduler.running) == 0):
                    break

    def run_sweep(self):
        """Execute the full sweep (sequential or parallel based on max_concurrent)."""
        combinations = self.generate_parameter_combinations()
        self.total_runs = len(combinations)

        print("=" * 70)
        mode_str = "Sequential" if not self.parallel_mode else f"Parallel ({self.scheduler.max_concurrent} max concurrent)"
        print(f"Starting {mode_str} Sweep: '{self.sweep_name}'")
        print("=" * 70)
        print(f"User ID: {self.user_id}")
        print(f"Total configurations: {len(combinations)}")
        print(f"GPU Budget: {self.scheduler.total_budget}")
        if self.exclusive_mode:
            print(f"Exclusive Mode: ON (requesting {self.max_gpus_per_node} GPUs per pod)")
        print(f"Results directory: {self.results_dir}")
        print("=" * 70)
        print()

        # Prepare all run states
        for i, params in enumerate(combinations, 1):
            namespace = self._generate_namespace(i)

            # Render template to calculate GPU claim
            run_dir = self.results_dir / f"run-{i:03d}"
            run_dir.mkdir(exist_ok=True)

            manifest_dir = self.render_template(params, run_dir)
            gpu_claim = self.calculate_gpu_claim(params, manifest_dir)

            if self.exclusive_mode:
                gpu_claim = self.max_gpus_per_node

            run_state = RunState(
                run_id=i,
                namespace=namespace,
                parameters=params,
                gpu_claim=gpu_claim,
                status=RunStatus.PENDING
            )

            self.scheduler.add_pending(run_state)

        print(f"Queued {len(combinations)} configurations\n")

        # Save initial state
        self.save_state()

        # Start centralized health monitor
        if self.health_monitoring_enabled and self.health_monitor:
            self.health_monitor.start()
            print("🔍 Centralized health monitoring started")
            print(f"   Check interval: {self.health_monitor.check_interval}s")
            print(f"   Aggressive timeout: {self.health_monitor.aggressive_timeout}s\n")

        try:
            if self.parallel_mode:
                # Parallel execution with worker threads
                num_workers = min(self.scheduler.max_concurrent, len(combinations))
                print(f"Starting {num_workers} worker threads...\n")

                workers = []
                for i in range(num_workers):
                    worker = threading.Thread(target=self.worker_thread, name=f"Worker-{i+1}")
                    worker.start()
                    workers.append(worker)

                # Wait for all workers to complete
                for worker in workers:
                    worker.join()
            else:
                # Sequential execution
                while True:
                    run_state = self.scheduler.try_schedule_next()
                    if run_state is None:
                        break
                    self.execute_run(run_state)

        finally:
            # Stop health monitor when sweep completes
            if self.health_monitoring_enabled and self.health_monitor:
                safe_print("\n🔍 Stopping health monitor...")
                self.health_monitor.stop()

        # Save final state
        self.save_state()

        # Generate summary from completed states
        completed_states = self.scheduler.get_completed_states()
        summary = generate_summary_from_states(completed_states, sweep_dir=self.results_dir)
        write_summary_file(summary, self.results_dir / "summary.json")

        print("\n" + "=" * 70)
        print("SWEEP COMPLETED")
        print("=" * 70)
        self.print_summary(summary)
        print(f"\nResults saved to: {self.results_dir}")

    def print_summary(self, results: List[Dict[str, Any]]):
        """Print sweep summary."""
        successful = sum(1 for r in results if r['status'] in ['success', 'completed'])
        failed = sum(1 for r in results if r['status'] == 'failed')
        cancelled = sum(1 for r in results if r['status'] == 'cancelled')

        print("\n" + "="*60)
        print("SWEEP SUMMARY")
        print("="*60)
        print(f"Total runs: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        if cancelled > 0:
            print(f"Cancelled: {cancelled}")

        if failed > 0:
            print("\nFailed runs:")
            for run in results:
                if run['status'] == 'failed':
                    print(f"  Run {run['run_id']}: {run.get('error', 'Unknown error')}")

        if cancelled > 0:
            print("\nCancelled runs:")
            for run in results:
                if run['status'] == 'cancelled':
                    print(f"  Run {run['run_id']}")


def resolve_config_path(config_arg: str) -> str:
    """
    Resolve the configuration file path from various input formats.

    Supports:
    1. Absolute or relative path to existing file (e.g., /path/to/config.yaml)
    2. Filename with .yaml extension (assumes sweep-configs/ directory)
    3. Basename without extension (assumes sweep-configs/ directory and adds .yaml)

    Args:
        config_arg: Configuration file argument from command line

    Returns:
        Resolved path to configuration file

    Raises:
        FileNotFoundError: If the configuration file cannot be found
    """
    config_path = Path(config_arg)

    # Case 1: File exists at the given path (absolute or relative)
    if config_path.exists() and config_path.is_file():
        return str(config_path)

    # Get script directory to locate sweep-configs
    script_dir = Path(__file__).parent.parent
    sweep_configs_dir = script_dir / "sweep-configs"

    # Case 2: Filename with .yaml extension - check in sweep-configs/
    if config_arg.endswith('.yaml'):
        candidate = sweep_configs_dir / config_arg
        if candidate.exists() and candidate.is_file():
            return str(candidate)

    # Case 3: Basename without extension - add .yaml and check in sweep-configs/
    candidate = sweep_configs_dir / f"{config_arg}.yaml"
    if candidate.exists() and candidate.is_file():
        return str(candidate)

    # If we get here, file was not found
    # Provide helpful error message
    raise FileNotFoundError(
        f"Configuration file not found: {config_arg}\n"
        f"Tried:\n"
        f"  1. Direct path: {config_path.absolute()}\n"
        f"  2. In sweep-configs: {sweep_configs_dir / config_arg}\n"
        f"  3. In sweep-configs with .yaml: {sweep_configs_dir / f'{config_arg}.yaml' if not config_arg.endswith('.yaml') else 'N/A'}"
    )


if __name__ == "__main__":
    import argparse

    # Handle SIGPIPE gracefully (when piped output is closed)
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except AttributeError:
        # SIGPIPE not available on Windows
        pass

    parser = argparse.ArgumentParser(
        description="Run benchmarking sweep with parallel execution and GPU budgeting",
        epilog="""
Examples:
  # Sequential execution (backward compatible):
  %(prog)s my-sweep

  # Parallel execution with GPU budget:
  %(prog)s my-sweep --gpu-budget 16 --max-concurrent 4

  # Exclusive mode (request full node regardless of TP size):
  %(prog)s my-sweep --exclusive-mode --gpu-budget 64

  # Dry-run with budget validation:
  %(prog)s my-sweep --dry-run --gpu-budget 16

  # Sequential with unlimited GPU budget:
  %(prog)s my-sweep --max-concurrent 1
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "config",
        help="Sweep configuration file (basename, filename, or path)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configurations and GPU budget analysis without executing"
    )
    parser.add_argument(
        "--gpu-budget",
        type=int,
        default=None,
        help="Total GPU budget for parallel execution (default: unlimited)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Maximum concurrent configurations (1=sequential, 0=unlimited, default: 1)"
    )
    parser.add_argument(
        "--exclusive-mode",
        action="store_true",
        help="Request max GPUs per node regardless of vllm parallel config"
    )
    parser.add_argument(
        "--max-gpus-per-node",
        type=int,
        default=8,
        help="Maximum GPUs per node (default: 8, used in exclusive mode)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom sweep directory name (default: auto-generated {sweep_name}_{timestamp})"
    )
    parser.add_argument(
        "--runtime-config",
        type=str,
        default=None,
        help=(
            "Optional runtime override YAML file; runtime-defaults.yaml is "
            "always loaded when present"
        )
    )
    args = parser.parse_args()

    try:
        config_file = resolve_config_path(args.config)
        orchestrator = SweepOrchestrator(
            config_file,
            gpu_budget=args.gpu_budget,
            max_concurrent=args.max_concurrent,
            exclusive_mode=args.exclusive_mode,
            max_gpus_per_node=args.max_gpus_per_node,
            output_dir=args.output_dir,
            runtime_file=args.runtime_config
        )

        if args.dry_run:
            orchestrator.dry_run()
        else:
            if args.gpu_budget is None and args.max_concurrent > 1:
                print("Warning: No GPU budget specified for parallel execution")
                print("Use --gpu-budget to limit GPU usage across parallel configs")
                print()

            orchestrator.run_sweep()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        safe_print("\nInterrupted by user")
        sys.exit(130)
