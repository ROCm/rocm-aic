#!/usr/bin/env python3
"""
Benchmarking sweep orchestrator.
Runs parameterized deployments, load generation, and result collection.
"""

import yaml
import subprocess
import time
import json
import sys
import itertools
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from load_generators import get_load_generator


class SweepOrchestrator:
    # Map deployment types to their justfile directories
    # Note: Only tiered-prefix-cache variants are supported for justfile-based deployment
    DEPLOYMENT_MAP = {
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

    def __init__(self, config_file: str):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)

        self.sweep_name = self.config['name']
        self.deployment = self.config['deployment']
        self.timestamp = datetime.now().strftime('%Y-%m-%d')
        self.results_dir = Path(f"results/sweeps/{self.sweep_name}_{self.timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Get current user for namespace prefix
        self.user_id = self._get_user_id()

        # Get deployment configuration
        self.deployment_config = self._get_deployment_config()

        # Save sweep metadata
        with open(self.results_dir / "metadata.yaml", 'w') as f:
            yaml.dump(self.config, f)

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
        - Load generation benchmark_args combinations (if type: combinations)

        Supports the consistent pattern where benchmark_args can have:
        - type: combinations with mixed sweepable (values) and fixed parameters
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
            elif spec['type'] == 'combinations':
                # Handle all args combinations (vllm_args, lmcache_args, custom args, etc.)
                # Generate combinations using generic method
                args_combinations = self._generate_args_combinations(spec['args'])
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

        # New pattern: benchmark_args with type: combinations
        if isinstance(benchmark_args_spec, dict) and benchmark_args_spec.get('type') == 'combinations':
            load_combinations = self._generate_args_combinations(benchmark_args_spec['args'])
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

    def _generate_args_combinations(self, args_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generic expansion of args with 'values' lists into combinations.

        Works for vllm_args, lmcache_args, and any other args that follow
        the pattern of fixed values and sweepable 'values' lists.

        Args:
            args_spec: Dictionary where values can be:
                - Fixed values (any type)
                - Dicts with 'values' key containing a list

        Returns:
            List of dictionaries with all combinations
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
        for combo in itertools.product(*arg_values):
            combinations.append(dict(zip(arg_keys, combo)))

        return combinations

    def expand_kv_connector(self, connector_config: Dict[str, Any]) -> str:
        """Expand KV connector shortcut to full --kv-transfer-config JSON."""
        if not connector_config:
            return ""

        # Handle raw JSON pass-through
        if "raw_json" in connector_config:
            return f"--kv-transfer-config '{connector_config['raw_json']}'"

        connector_type = connector_config.get("type")
        role = connector_config.get("role", "kv_both")

        if connector_type == "offloading":
            # Offloading connector
            cpu_bytes = connector_config.get("cpu_bytes")
            kv_config = {
                "kv_connector": "OffloadingConnector",
                "kv_role": role,
                "kv_connector_extra_config": {
                    "cpu_bytes_to_use": cpu_bytes
                }
            }
        elif connector_type == "lmcache":
            # LMCache connector
            kv_config = {
                "kv_connector": "LMCacheConnectorV1",
                "kv_role": role
            }
        else:
            raise ValueError(f"Unknown kv_connector type: {connector_type}")

        return f"--kv-transfer-config '{json.dumps(kv_config)}'"

    def build_vllm_args(self, vllm_args: Dict[str, Any]) -> str:
        """Convert vllm_args dict to command-line arguments."""
        args = []

        for key, value in vllm_args.items():
            # Convert underscore to hyphen for vLLM arguments
            arg_name = key.replace("_", "-")

            # Special handling for kv_connector
            if key == "kv_connector":
                if value is not None:
                    args.append(self.expand_kv_connector(value))
                continue

            # Handle boolean flags
            if isinstance(value, bool):
                if value:
                    args.append(f"--{arg_name}")
                continue

            # Handle null values (skip them)
            if value is None:
                continue

            # Handle regular key-value arguments
            args.append(f"--{arg_name} {value}")

        return " \\\n            ".join(args)

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

        # Process vllm_args if present
        if "vllm_args" in params:
            vllm_args_str = self.build_vllm_args(params["vllm_args"])
            template_params["VLLM_ARGS"] = vllm_args_str

        # Extract lmcache_args for special handling
        lmcache_args = template_params.pop('lmcache_args', None)

        # If we have lmcache_args, use template injection
        if lmcache_args:
            from lmcache_template_injection import render_kustomization_with_lmcache

            rendered = render_kustomization_with_lmcache(
                template_file,
                template_params,
                lmcache_args,
                use_yaml_parse=True  # Use YAML parsing for precision
            )
        else:
            # Standard template rendering (no lmcache)
            with open(template_file) as f:
                content = f.read()

            # Replace {{VAR}} with actual values
            rendered = content
            for key, value in template_params.items():
                placeholder = f"{{{{{key}}}}}"
                if placeholder in rendered:
                    rendered = rendered.replace(placeholder, str(value))

        # Write output
        output_file = output_dir / "kustomization.yaml"
        with open(output_file, 'w') as f:
            f.write(rendered)

        return output_dir

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

    def wait_for_deployment(self, namespace: str):
        """Wait for deployment to be ready using justfile wait target."""
        print(f"  Waiting for deployment to be ready...")

        if not self.deployment_config.get('use_justfile', False):
            # Fall back to direct kubectl wait for non-justfile deployments
            max_wait = 600  # 10 minutes
            start_time = time.time()

            while time.time() - start_time < max_wait:
                result = subprocess.run([
                    "kubectl", "wait", "--for=condition=available",
                    "deployment/llm-d-model-server",
                    "-n", namespace,
                    "--timeout=30s"
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    print(f"  Deployment ready! (kubectl wait)")
                    return

                time.sleep(10)

            raise RuntimeError("Deployment did not become ready in time")

        # Use justfile-based wait
        justfile_dir = self._get_justfile_path()

        # Run the wait target from the justfile
        result = subprocess.run([
            "just",
            "wait"
        ], capture_output=True, text=True, cwd=str(justfile_dir),
           env={**os.environ, 'NAMESPACE': namespace})

        if result.returncode != 0:
            print(f"  Wait stdout: {result.stdout}")
            print(f"  Wait stderr: {result.stderr}")
            raise RuntimeError(f"Wait for deployment failed: {result.stderr}")

        print(f"  Deployment ready! (justfile)")

    def run_load_generation(self, run_dir: Path, params: Dict[str, Any], namespace: str) -> Dict[str, Any]:
        """
        Run load generation using pluggable load generator modules.

        Args:
            run_dir: Directory for saving results
            params: Deployment parameters
            namespace: Kubernetes namespace

        Returns:
            Dictionary with results from the load generator
        """
        load_config = self.config['load_generation']
        tool = load_config['tool']

        print(f"  Running load generation with {tool}...")

        # Get the appropriate load generator
        generator = get_load_generator(tool, self)

        # Run the load generator
        result = generator.run(load_config, run_dir, params, namespace)

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
            "-l", "llm-d.ai/inference-serving=true",
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
        """Print sweep configuration without executing."""
        combinations = self.generate_parameter_combinations()

        print("=" * 70)
        print(f"DRY RUN: {self.sweep_name}")
        print("=" * 70)
        print(f"Description: {self.config['description']}")
        print(f"Deployment: {self.deployment}")
        print(f"User ID: {self.user_id}")
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

        print("=" * 70)
        print(f"SUMMARY: {len(combinations)} configurations would be executed")
        print("=" * 70)

    def run_sweep(self):
        """Execute the full sweep."""
        combinations = self.generate_parameter_combinations()

        print(f"Starting sweep '{self.sweep_name}'")
        print(f"User ID: {self.user_id}")
        print(f"Total configurations to test: {len(combinations)}")
        print(f"Results directory: {self.results_dir}")
        print()

        results_summary = []

        for i, params in enumerate(combinations, 1):
            # Generate unique namespace for this run
            namespace = self._generate_namespace(i)

            print(f"Run {i}/{len(combinations)}")
            print(f"  Namespace: {namespace}")
            print(f"  Parameters: {params}")

            run_dir = self.results_dir / f"run-{i:03d}"
            run_dir.mkdir(exist_ok=True)

            # Save configuration with namespace
            config_to_save = params.copy()
            config_to_save['namespace'] = namespace
            with open(run_dir / "config.yaml", 'w') as f:
                yaml.dump(config_to_save, f)

            manifest_dir = None

            try:
                # Create namespace
                self.create_namespace(namespace)

                # Inject HF token secret
                self.inject_hf_token_secret(namespace)

                # Generate manifests (if needed for backwards compatibility)
                manifest_dir = self.render_template(params, run_dir)

                # Deploy using justfile
                self.deploy(manifest_dir, namespace)

                # Wait for deployment to be ready
                self.wait_for_deployment(namespace)

                # Run load generation
                benchmark_results = self.run_load_generation(run_dir, params, namespace)

                print(benchmark_results)
                # Collect pod logs
                self.collect_pod_logs(run_dir, namespace)

                # Success
                run_result = {
                    'run_id': i,
                    'namespace': namespace,
                    'parameters': params,
                    'benchmark': benchmark_results,
                    'status': 'success' if benchmark_results['exit_code'] == 0 else 'failed'
                }

                results_summary.append(run_result)
                if run_result['status'] == 'success':
                    print(f"  ✓ Run {i} completed successfully")
                else:
                    print(f"  ✗ Run {i} failed")

            except Exception as e:
                print(f"  ✗ Run {i} failed: {e}")
                run_result = {
                    'run_id': i,
                    'namespace': namespace,
                    'parameters': params,
                    'status': 'failed',
                    'error': str(e)
                }
                results_summary.append(run_result)

            finally:
                # Always teardown - delete the namespace
                try:
                    self.teardown(namespace)
                except Exception as e:
                    print(f"  Warning: Teardown failed: {e}")

            print()

        # Save summary
        with open(self.results_dir / "summary.json", 'w') as f:
            json.dump(results_summary, f, indent=2)

        print(f"Sweep completed! Results saved to {self.results_dir}")
        self.print_summary(results_summary)

    def print_summary(self, results: List[Dict[str, Any]]):
        """Print sweep summary."""
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - successful

        print("\n" + "="*60)
        print("SWEEP SUMMARY")
        print("="*60)
        print(f"Total runs: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed runs:")
            for run in results:
                if run['status'] == 'failed':
                    print(f"  Run {run['run_id']}: {run.get('error', 'Unknown error')}")


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
        description="Run benchmarking sweep",
        epilog="""
Examples:
  %(prog)s my-sweep                           # Uses sweep-configs/my-sweep.yaml
  %(prog)s my-sweep.yaml                      # Uses sweep-configs/my-sweep.yaml
  %(prog)s sweep-configs/my-sweep.yaml        # Uses relative path
  %(prog)s /absolute/path/to/my-sweep.yaml    # Uses absolute path
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
        help="Show configurations without executing"
    )
    args = parser.parse_args()

    try:
        config_file = resolve_config_path(args.config)
        orchestrator = SweepOrchestrator(config_file)

        if args.dry_run:
            orchestrator.dry_run()
        else:
            orchestrator.run_sweep()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
