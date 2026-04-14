"""
vLLM bench serve load generator.
"""

import subprocess
from pathlib import Path
from typing import Dict, Any, List

from .base import LoadGeneratorBase


class VllmBenchServe(LoadGeneratorBase):
    """vLLM bench serve benchmarking tool."""

    def build_args(self, benchmark_args: Dict[str, Any], model: str,
                   service_info: Dict[str, str], **kwargs) -> List[str]:
        """
        Build command-line arguments for vllm-bench-serve tool.

        Args:
            benchmark_args: Additional benchmark configuration
            model: Model name
            service_info: Dict containing 'base_url' key
            **kwargs: Must contain 'run_dir' and 'seed'

        Returns:
            List of command-line arguments
        """
        run_dir = kwargs['run_dir']
        seed = kwargs['seed']

        args = [
            "--base-url", service_info['base_url'],
            "--model", model,
        ]

        # Add seed if not already in benchmark_args
        if 'seed' not in benchmark_args:
            args.extend(["--seed", str(seed)])

        # Always save results as JSON
        args.append("--save-result")
        args.extend(["--result-dir", str(run_dir.absolute())])

        # Add additional benchmark arguments
        args.extend(self.orchestrator._convert_args_dict(benchmark_args))

        return args

    def run(self, config: Dict[str, Any], run_dir: Path,
            params: Dict[str, Any], namespace: str) -> Dict[str, Any]:
        """Run vLLM bench serve benchmark."""

        # Get configuration
        image = config.get('image', 'vllm/vllm-openai:latest')
        clone_run = config.get('clone_run', False)

        # Get benchmark_args - handle both patterns
        benchmark_args_config = config.get('benchmark_args', {})
        if isinstance(benchmark_args_config, dict) and benchmark_args_config.get('type') == 'combinations':
            # New pattern: type: combinations - extract fixed args (non-sweepable)
            benchmark_args = {}
            for key, value in benchmark_args_config.get('args', {}).items():
                # Skip sweepable parameters (they're in _load_params)
                if not (isinstance(value, dict) and 'values' in value):
                    benchmark_args[key] = value
        else:
            # Old pattern: direct dict or sweep_args pattern
            benchmark_args = benchmark_args_config.copy() if benchmark_args_config else {}

        # Merge load generation sweep parameters (if any)
        if '_load_params' in params:
            benchmark_args.update(params['_load_params'])

        # Get model and construct base URL
        model = params.get('model')
        base_url = f"http://llm-d-inference-gateway-istio.{namespace}.svc.cluster.local:80"

        # Generate or get seed
        if 'seed' in benchmark_args:
            seed = benchmark_args['seed']
        else:
            seed = self.orchestrator.generate_timestamp_seed()
            print(f"  Generated seed: {seed}")

        # Determine number of runs
        num_runs = 2 if clone_run else 1

        results = []
        for run_idx in range(1, num_runs + 1):
            run_label = f"run{run_idx}"
            print(f"  Executing benchmark {run_label}/{num_runs}...")

            result = self._execute_single_run(
                benchmark_args=benchmark_args,
                model=model,
                base_url=base_url,
                run_dir=run_dir,
                run_label=run_label,
                seed=seed,
                image=image,
                namespace=namespace
            )
            results.append(result)

        # Return aggregated results
        return {
            "tool": "vllm-bench-serve",
            "runs": results,
            "clone_run": clone_run,
            "seed": seed,
            "num_runs": num_runs,
            "exit_code": 0 if all(r['exit_code'] == 0 for r in results) else 1
        }

    def _execute_single_run(self, benchmark_args: Dict[str, Any], model: str,
                            base_url: str, run_dir: Path, run_label: str,
                            seed: int, image: str, namespace: str) -> Dict[str, Any]:
        """Execute a single vLLM bench serve benchmark run."""

        # Build benchmark arguments
        benchmark_cmd_args = self.build_args(
            benchmark_args,
            model,
            {'base_url': base_url},
            run_dir=run_dir,
            seed=seed
        )

        # Build script arguments (for run-benchmark.sh)
        script_args = [
            "--image", image,
            "--namespace", namespace,
            "--output-dir", str(run_dir.absolute()),
            "--run-label", run_label
        ]

        # Build full command
        cmd = ["./load-generators/vllm-bench-serve/run-benchmark.sh"]
        cmd.extend(script_args)
        cmd.append("--")  # Separator
        cmd.extend(benchmark_cmd_args)

        print(f"    Running vllm bench serve (image: {image})...")

        # Execute
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"    Warning: Benchmark exited with code {result.returncode}")

        # Save runner output
        output_file = run_dir / f"benchmark_runner_output_{run_label}.txt"
        with open(output_file, "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)

        return {
            "run_label": run_label,
            "output_file": str(run_dir / f"benchmark_output_{run_label}.json"),
            "exit_code": result.returncode,
            "seed": seed
        }
