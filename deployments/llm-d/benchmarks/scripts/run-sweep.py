#!/usr/bin/env python3
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
    Evaluate all expressions in a parameter combination.

    Processes each value in the combination:
    - If it contains {variable} references, evaluates the expression
    - Otherwise, keeps the original value

    Args:
        combination: Dictionary of parameter names to values

    Returns:
        New dictionary with expressions evaluated

    Example:
        Input:  {"max_concurrency": 16, "num_prompts": "30 * {max_concurrency}"}
        Output: {"max_concurrency": 16, "num_prompts": 480}
    """
    result = {}

    # First pass: collect non-expression values to use as variables
    variables = {}
    for key, value in combination.items():
        if not has_variable_reference(value):
            variables[key] = value

    # Second pass: evaluate expressions
    for key, value in combination.items():
        if has_variable_reference(value):
            try:
                evaluated = safe_eval_expression(value, variables)
                result[key] = evaluated
            except ValueError as e:
                raise ValueError(f"Error evaluating '{key}': {e}") from e
        else:
            result[key] = value

    return result


# ============================================================================
# End Expression Evaluation
# ============================================================================


class RunStatus(Enum):
    """Status of a configuration run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RunState:
    """State information for a single configuration run."""
    run_id: int
    namespace: str
    parameters: Dict[str, Any]
    gpu_claim: int
    status: RunStatus
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None
    benchmark_results: Optional[Dict[str, Any]] = None
    failure_info: Optional[Dict[str, Any]] = None  # Detailed failure information

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['status'] = self.status.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunState':
        """Create from dictionary loaded from JSON."""
        data['status'] = RunStatus(data['status'])
        return cls(**data)


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
                 max_gpus_per_node: int = 8):
        """
        Initialize the orchestrator.

        Args:
            config_file: Path to sweep configuration file
            gpu_budget: Total GPU budget (None = unlimited)
            max_concurrent: Maximum concurrent configurations (1 = sequential, 0 = unlimited)
            exclusive_mode: If True, pods request max GPUs per node
            max_gpus_per_node: Maximum GPUs available per node
        """
        with open(config_file) as f:
            self.config = yaml.safe_load(f)

        self.sweep_name = self.config['name']
        self.deployment = self.config['deployment']
        self.timestamp = datetime.now().strftime('%Y-%m-%d')
        self.results_dir = Path(f"results/sweeps/{self.sweep_name}_{self.timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)

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
            'timestamp': self.timestamp
        }
        with open(self.results_dir / "metadata.yaml", 'w') as f:
            yaml.dump(metadata, f)

        # State file for tracking runs
        self.state_file = self.results_dir / "state.json"

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.shutdown_event = threading.Event()

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

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print("\n" + "="*70)
        print("INTERRUPT RECEIVED - Initiating graceful shutdown...")
        print("="*70)
        print("Cancelling pending configurations and cleaning up running ones...")
        self.shutdown_event.set()
        self.scheduler.request_shutdown()

        # Stop health monitor
        if self.health_monitoring_enabled and self.health_monitor:
            print("Stopping health monitor...")
            self.health_monitor.stop()

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
                deployment_combinations.append(config)
        else:
            deployment_combinations = [fixed]

        # Check if load_generation has benchmark_args with type: combinations
        load_config = self.config.get('load_generation', {})
        benchmark_args_spec = load_config.get('benchmark_args', {})

        load_combinations = []

        # New pattern: benchmark_args with type: combinations or pairwise
        if isinstance(benchmark_args_spec, dict) and benchmark_args_spec.get('type') in ['combinations', 'pairwise']:
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
                full_combinations.append(combo)

        return full_combinations

    def _generate_args_combinations(self, args_spec: Dict[str, Any], combination_mode: str = 'product') -> List[Dict[str, Any]]:
        """
        Generic expansion of args with 'values' lists into combinations.

        Works for vllm_args, lmcache_args, and any other args that follow
        the pattern of fixed values and sweepable 'values' lists.

        Supports variable expansion: parameters can reference other parameters
        using {var_name} syntax, e.g., "30 * {max_concurrency}"

        Args:
            args_spec: Dictionary where values can be:
                - Fixed values (any type)
                - Dicts with 'values' key containing a list
                - Expressions with {var_name} references
            combination_mode: How to combine sweepable parameters:
                - 'product': Cartesian product (default)
                - 'pairwise': Pair-wise zip of values

        Returns:
            List of dictionaries with all combinations and expressions evaluated
        """
        arg_keys = []
        arg_values = []

        for key, spec in args_spec.items():
            arg_keys.append(key)
            if isinstance(spec, dict) and 'values' in spec:
                # Sweepable parameter
                arg_values.append(spec['values'])
            else:
                # Fixed value
                arg_values.append([spec])

        combinations = []

        if combination_mode == 'pairwise':
            # Pair-wise: zip values together (stops at shortest list)
            for combo in zip(*arg_values):
                combo_dict = dict(zip(arg_keys, combo))
                # Evaluate any expressions in this combination
                combo_dict = evaluate_expressions_in_combination(combo_dict)
                combinations.append(combo_dict)
        else:
            # Default: Cartesian product
            for combo in itertools.product(*arg_values):
                combo_dict = dict(zip(arg_keys, combo))
                # Evaluate any expressions in this combination
                combo_dict = evaluate_expressions_in_combination(combo_dict)
                combinations.append(combo_dict)

        return combinations

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
                                # Add as YAML list item
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

        output_dir = run_dir / "manifests"
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

        # Extract lmcache_args for special handling
        lmcache_args = template_params.pop('lmcache_args', None)

        # Load template file
        with open(template_file) as f:
            content = f.read()

        # Step 1: Always do standard template variable replacement first
        rendered = self.replace_template_variables(content, template_params)

        # Step 2: If we have lmcache_args, inject the LMCache configmap
        if lmcache_args:
            from lmcache_template_injection import inject_lmcache_configmap_into_rendered

            rendered = inject_lmcache_configmap_into_rendered(
                rendered,
                lmcache_args,
                use_yaml_parse=True  # Use YAML parsing for precision
            )

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
        state = {
            'pending': [rs.to_dict() for rs in self.scheduler.get_pending_states()],
            'running': [rs.to_dict() for rs in self.scheduler.get_running_states()],
            'completed': [rs.to_dict() for rs in self.scheduler.get_completed_states()]
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

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

    def inject_hf_token_secret(self, namespace: str):
        """Inject HuggingFace token secret into namespace."""
        # Get HF_TOKEN from environment
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            raise RuntimeError("HF_TOKEN environment variable is not set. Please set it before running sweeps.")

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
        print(f"  Deleting namespace {namespace}...")

        result = subprocess.run([
            "kubectl", "delete", "namespace", namespace, "--wait=true"
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  Warning: Failed to delete namespace: {result.stderr}")

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

    def collect_pod_logs(self, run_dir: Path, namespace: str):
        """Collect vLLM pod logs."""
        print(f"  Collecting pod logs...")

        logs_dir = run_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Get vLLM pods
        result = subprocess.run([
            "kubectl", "get", "pods",
            "-n", namespace,
            "-l", "llm-d.ai/role=decode",
            "-o", "jsonpath={.items[*].metadata.name}"
        ], capture_output=True, text=True, check=True)

        pod_names = result.stdout.strip().split()

        for pod_name in pod_names:
            # Get pod logs
            log_file = logs_dir / f"{pod_name}.log"
            with open(log_file, 'w') as f:
                subprocess.run([
                    "kubectl", "logs", pod_name, "-n", namespace
                ], stdout=f, stderr=subprocess.STDOUT)

            print(f"    Saved logs for {pod_name}")

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

            # Show generated vLLM command
            if 'vllm_args' in params:
                print()
                print("  Generated vLLM command:")
                vllm_args_str = self.build_vllm_args(params['vllm_args'])
                print(f"    vllm serve {params.get('model', '<model>')} \\")
                print(f"      --tensor-parallel-size {params.get('tensor_parallel_size', 1)} \\")
                print(f"      --port 8000 \\")
                for line in vllm_args_str.split('\n'):
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

        prefix = f"[Run {run_id}]" if self.parallel_mode else f"Run {run_id}/{self.scheduler.get_pending_states().__len__() + self.scheduler.get_running_states().__len__() + run_id}"

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
            benchmark_results = self.run_load_generation(run_dir, params, namespace)

            if self.shutdown_event.is_set():
                raise InterruptedError("Shutdown requested")

            print(f"{prefix} Collecting pod logs...")
            self.collect_pod_logs(run_dir, namespace)

            # Update run state
            run_state.benchmark_results = benchmark_results
            if benchmark_results['exit_code'] == 0:
                run_state.status = RunStatus.COMPLETED
                print(f"{prefix} ✓ Completed successfully")
            else:
                run_state.status = RunStatus.FAILED
                run_state.error = "Benchmark failed"
                print(f"{prefix} ✗ Benchmark failed")

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

        finally:
            # Unregister from health monitoring
            if self.health_monitoring_enabled and self.health_monitor:
                self.health_monitor.unregister_namespace(namespace)

            # Always teardown and release resources
            try:
                print(f"{prefix} Tearing down...")
                self.teardown(namespace)
            except Exception as e:
                print(f"{prefix} Warning: Teardown failed: {e}")

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
                print("\n🔍 Stopping health monitor...")
                self.health_monitor.stop()

        # Save final state
        self.save_state()

        # Generate summary from completed states
        completed_states = self.scheduler.get_completed_states()

        summary = []
        for run_state in completed_states:
            run_result = {
                'run_id': run_state.run_id,
                'namespace': run_state.namespace,
                'parameters': run_state.parameters,
                'gpu_claim': run_state.gpu_claim,
                'status': run_state.status.value,
                'start_time': run_state.start_time,
                'end_time': run_state.end_time,
                'duration': run_state.end_time - run_state.start_time if run_state.end_time and run_state.start_time else None,
            }

            if run_state.benchmark_results:
                run_result['benchmark'] = run_state.benchmark_results

            if run_state.error:
                run_result['error'] = run_state.error

            summary.append(run_result)

        # Save summary
        with open(self.results_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

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
    args = parser.parse_args()

    try:
        config_file = resolve_config_path(args.config)
        orchestrator = SweepOrchestrator(
            config_file,
            gpu_budget=args.gpu_budget,
            max_concurrent=args.max_concurrent,
            exclusive_mode=args.exclusive_mode,
            max_gpus_per_node=args.max_gpus_per_node
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
        print("\nInterrupted by user")
        sys.exit(130)
