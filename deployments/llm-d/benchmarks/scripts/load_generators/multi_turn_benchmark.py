"""
Multi-turn benchmark load generator.
"""

import subprocess
from pathlib import Path
from typing import Dict, Any, List

from .base import LoadGeneratorBase


class MultiTurnBenchmark(LoadGeneratorBase):
    """Multi-turn conversation benchmark tool."""

    def build_args(self, benchmark_args: Dict[str, Any], model: str,
                   service_info: Dict[str, str], **kwargs) -> List[str]:
        """
        Build command-line arguments for multi-turn-benchmark tool.

        Args:
            benchmark_args: Additional benchmark configuration
            model: Model name
            service_info: Dict containing 'url' key with service URL
            **kwargs: Unused

        Returns:
            List of command-line arguments
        """
        args = [
            "--model", model,
            "--url", service_info['url']
        ]

        # Add additional benchmark arguments
        args.extend(self.orchestrator._convert_args_dict(benchmark_args))

        return args

    def run(self, config: Dict[str, Any], run_dir: Path,
            params: Dict[str, Any], namespace: str) -> Dict[str, Any]:
        """Run vLLM multi-turn chat benchmark."""

        # Get configuration
        image = config.get('image', 'vllm/vllm-openai:latest')
        workload_file = config.get('workload_file', 'agent_multi_turn.json')

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

        # Resolve workload file path
        workload_path = Path(f"load-generators/multi-turn-benchmark/workloads/{workload_file}")
        if not workload_path.exists():
            raise FileNotFoundError(f"Workload file not found: {workload_path}")

        # Get model and construct service URL
        model = params.get('model')
        service_url = f"http://llm-d-inference-gateway-istio.{namespace}.svc.cluster.local:80"

        # Build script arguments (for run-benchmark.sh)
        script_args = [
            "--image", image,
            "--namespace", namespace,
            "--workload-file", str(workload_path.absolute()),
            "--output-dir", str(run_dir.absolute())
        ]

        # Build benchmark arguments (passed to container after --)
        benchmark_cmd_args = self.build_args(
            benchmark_args,
            model,
            {'url': service_url}
        )

        # Add input-file pointing to mounted path
        benchmark_cmd_args.extend([
            "--input-file", f"/workload/{workload_file}"
        ])

        # Build full command
        cmd = ["./load-generators/multi-turn-benchmark/run-benchmark.sh"]
        cmd.extend(script_args)
        cmd.append("--")  # Separator
        cmd.extend(benchmark_cmd_args)

        print(f"  Running benchmark in pod (image: {image})...")

        # Execute
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  Warning: Benchmark exited with code {result.returncode}")

        # Save runner output
        with open(str(run_dir / "benchmark_runner_output.txt"), "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)

        return {
            "tool": "multi-turn-benchmark",
            "output_file": str(run_dir / "benchmark_output.txt"),
            "exit_code": result.returncode,
            "image": image,
            "workload_file": workload_file
        }
