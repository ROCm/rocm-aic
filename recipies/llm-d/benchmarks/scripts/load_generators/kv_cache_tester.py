"""
KV Cache Tester load generator.

Supports four benchmark variants:
- single_prompt_tester: Simple single-prompt tests
- cache_rate_tester: Test various cache hit rates
- working_set_tester: Test performance across different memory tiers
- trace_replay_tester: Replay real agentic coding traces
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from .base import LoadGeneratorBase


# Valid variant names
VALID_VARIANTS = [
    'single_prompt_tester',
    'cache_rate_tester',
    'working_set_tester',
    'trace_replay_tester',
]


class KvCacheTester(LoadGeneratorBase):
    """KV Cache Tester benchmarking tool with multiple test variants."""

    def build_args(self, benchmark_args: Dict[str, Any], model: str,
                   service_info: Dict[str, str], **kwargs) -> List[str]:
        """
        Build command-line arguments for kv-cache-tester variants.

        Args:
            benchmark_args: Additional benchmark configuration (must include 'variant')
            model: Model name (not used by kv-cache-tester, but part of interface)
            service_info: Dict containing 'api_endpoint' key
            **kwargs: Must contain 'run_dir' and optionally 'variant'

        Returns:
            List of command-line arguments
        """
        run_dir = kwargs.get('run_dir')

        args = [
            "--api-endpoint", service_info['api_endpoint'],
        ]

        # Add output directory if the variant supports it
        variant = benchmark_args.get('variant', kwargs.get('variant'))
        if variant in ['single_prompt_tester', 'cache_rate_tester', 'trace_replay_tester']:
            args.extend(["--output-dir", "/tmp/results"])

        # Add additional benchmark arguments (excluding 'variant' which is handled separately)
        filtered_args = {k: v for k, v in benchmark_args.items() if k != 'variant'}
        args.extend(self.orchestrator._convert_args_dict(filtered_args))

        return args

    def parse_metrics(self, output_file: Path) -> Dict[str, Any]:
        """
        Parse kv-cache-tester output (JSON or text format).

        The kv-cache-tester variants produce JSON or CSV output files with
        performance metrics. This method extracts key metrics from these files.

        Args:
            output_file: Path to benchmark_output_runN.json or benchmark_output_runN.log

        Returns:
            Dictionary with:
                - metrics: Nested dict containing parsed metrics
                - parsing_status: "success", "partial", or "failed"
                - parsing_errors: List of errors (if any)
        """
        result = {
            "metrics": {},
            "parsing_status": "failed",
            "parsing_errors": []
        }

        try:
            # Check if file exists
            if not output_file.exists():
                result["parsing_errors"].append(f"Output file not found: {output_file}")
                return result

            # Try JSON parsing first
            if output_file.suffix == '.json':
                return self._parse_json_output(output_file)

            # Try to find JSON files in the output directory
            output_dir = output_file.parent
            json_files = list(output_dir.glob("*.json"))
            if json_files:
                # Parse the most recent JSON file
                json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                return self._parse_json_output(json_files[0])

            # Fall back to parsing log file
            return self._parse_log_output(output_file)

        except Exception as e:
            result["parsing_errors"].append(f"Unexpected error: {str(e)}")
            return result

    def _parse_json_output(self, output_file: Path) -> Dict[str, Any]:
        """Parse JSON output from kv-cache-tester variants."""
        result = {
            "metrics": {},
            "parsing_status": "failed",
            "parsing_errors": []
        }

        try:
            with open(output_file, 'r') as f:
                data = json.load(f)

            metrics = {}

            # Common metrics across variants
            for key in ["total_requests", "completed_requests", "failed_requests",
                        "request_throughput", "token_throughput"]:
                if key in data:
                    metrics[key] = data[key]

            # TTFT metrics (Time To First Token)
            ttft = {}
            for stat in ["mean", "median", "p99", "p95", "p90", "p75", "p50", "min", "max"]:
                for suffix in ["ttft_ms", "ttft_s", "ttft"]:
                    key = f"{stat}_{suffix}"
                    if key in data:
                        # Normalize to milliseconds
                        value = data[key]
                        if suffix == "ttft_s":
                            value *= 1000
                        ttft[f"{stat}_ms"] = value
                        break
            if ttft:
                metrics["ttft"] = ttft

            # Cache-specific metrics (from cache_rate_tester)
            cache_metrics = {}
            for key in ["cache_hit_rate", "target_cache_rate", "actual_cache_rate",
                        "context_size", "working_set_size"]:
                if key in data:
                    cache_metrics[key] = data[key]
            if cache_metrics:
                metrics["cache"] = cache_metrics

            # Working set metrics (from working_set_tester)
            working_set = {}
            for key in ["min_working_set_size", "max_working_set_size",
                        "current_working_set_size", "memory_tier"]:
                if key in data:
                    working_set[key] = data[key]
            if working_set:
                metrics["working_set"] = working_set

            # Trace replay metrics (from trace_replay_tester)
            trace_metrics = {}
            for key in ["concurrent_users", "max_users", "test_duration",
                        "slo_violations", "max_ttft"]:
                if key in data:
                    trace_metrics[key] = data[key]
            if trace_metrics:
                metrics["trace_replay"] = trace_metrics

            # Handle nested results structure
            if "results" in data and isinstance(data["results"], list):
                # Aggregate metrics from multiple test results
                all_ttfts = []
                for r in data["results"]:
                    if "ttft_ms" in r:
                        all_ttfts.append(r["ttft_ms"])
                    elif "ttft_s" in r:
                        all_ttfts.append(r["ttft_s"] * 1000)
                if all_ttfts:
                    metrics["ttft"] = {
                        "mean_ms": sum(all_ttfts) / len(all_ttfts),
                        "min_ms": min(all_ttfts),
                        "max_ms": max(all_ttfts),
                        "count": len(all_ttfts)
                    }

            result["metrics"] = metrics
            result["parsing_status"] = "success" if metrics else "partial"

        except json.JSONDecodeError as e:
            result["parsing_errors"].append(f"Invalid JSON: {str(e)}")
        except Exception as e:
            result["parsing_errors"].append(f"Unexpected error: {str(e)}")

        return result

    def _parse_log_output(self, output_file: Path) -> Dict[str, Any]:
        """Parse text log output from kv-cache-tester variants."""
        result = {
            "metrics": {},
            "parsing_status": "failed",
            "parsing_errors": []
        }

        try:
            with open(output_file, 'r') as f:
                content = f.read()

            metrics = {}
            found_any = False

            # Parse TTFT metrics
            ttft = {}
            match = re.search(r'(?:Mean|Average)\s+TTFT[:\s]+([\d.]+)\s*(?:ms|s)?', content, re.IGNORECASE)
            if match:
                ttft["mean_ms"] = float(match.group(1))
                found_any = True

            match = re.search(r'P99\s+TTFT[:\s]+([\d.]+)\s*(?:ms|s)?', content, re.IGNORECASE)
            if match:
                ttft["p99_ms"] = float(match.group(1))

            match = re.search(r'Max\s+TTFT[:\s]+([\d.]+)\s*(?:ms|s)?', content, re.IGNORECASE)
            if match:
                ttft["max_ms"] = float(match.group(1))

            if ttft:
                metrics["ttft"] = ttft

            # Parse cache hit rate
            match = re.search(r'Cache\s+(?:hit\s+)?rate[:\s]+([\d.]+)%?', content, re.IGNORECASE)
            if match:
                metrics["cache_hit_rate"] = float(match.group(1))
                found_any = True

            # Parse throughput
            match = re.search(r'Throughput[:\s]+([\d.]+)\s*(?:req/s|requests/s)', content, re.IGNORECASE)
            if match:
                metrics["request_throughput"] = float(match.group(1))
                found_any = True

            # Parse working set size
            match = re.search(r'Working\s+set\s+size[:\s]+([\d.]+)', content, re.IGNORECASE)
            if match:
                metrics["working_set_size"] = float(match.group(1))
                found_any = True

            result["metrics"] = metrics
            result["parsing_status"] = "success" if found_any else "failed"

            if not found_any:
                result["parsing_errors"].append("No recognized metrics found in output")

        except Exception as e:
            result["parsing_errors"].append(f"Unexpected error: {str(e)}")

        return result

    def run(self, config: Dict[str, Any], run_dir: Path,
            params: Dict[str, Any], namespace: str) -> Dict[str, Any]:
        """Run kv-cache-tester benchmark."""

        # Get configuration
        image = config.get('image', 'kv-cache-tester:latest')
        clone_run = config.get('clone_run', False)

        # Get variant - required for kv-cache-tester
        variant = config.get('variant')
        if not variant:
            # Check in benchmark_args
            benchmark_args_config = config.get('benchmark_args', {})
            if isinstance(benchmark_args_config, dict):
                if benchmark_args_config.get('type') in ['combinations', 'pairwise']:
                    variant = benchmark_args_config.get('args', {}).get('variant')
                else:
                    variant = benchmark_args_config.get('variant')

        if not variant:
            raise ValueError("kv-cache-tester requires 'variant' to be specified. "
                           f"Valid variants: {', '.join(VALID_VARIANTS)}")

        if variant not in VALID_VARIANTS:
            raise ValueError(f"Invalid variant '{variant}'. "
                           f"Valid variants: {', '.join(VALID_VARIANTS)}")

        # Get benchmark_args - handle both patterns
        benchmark_args_config = config.get('benchmark_args', {})
        if isinstance(benchmark_args_config, dict) and benchmark_args_config.get('type') in ['combinations', 'pairwise']:
            # New pattern: type: combinations or pairwise - extract fixed args (non-sweepable)
            benchmark_args = {}
            for key, value in benchmark_args_config.get('args', {}).items():
                # Skip sweepable parameters (they're in _load_params)
                if not (isinstance(value, dict) and 'values' in value):
                    benchmark_args[key] = value
        else:
            # Old pattern: direct dict or sweep_args pattern
            benchmark_args = benchmark_args_config.copy() if benchmark_args_config else {}

        # Ensure variant is in benchmark_args
        benchmark_args['variant'] = variant

        # Merge load generation sweep parameters (if any)
        if '_load_params' in params:
            benchmark_args.update(params['_load_params'])

        # Construct API endpoint URL
        api_endpoint = f"http://llm-d-inference-gateway-istio.{namespace}.svc.cluster.local:80"

        # Determine number of runs
        num_runs = 2 if clone_run else 1

        results = []
        for run_idx in range(1, num_runs + 1):
            run_label = f"run{run_idx}"
            print(f"  Executing kv-cache-tester {variant} {run_label}/{num_runs}...")

            result = self._execute_single_run(
                benchmark_args=benchmark_args,
                variant=variant,
                api_endpoint=api_endpoint,
                run_dir=run_dir,
                run_label=run_label,
                image=image,
                namespace=namespace
            )
            results.append(result)

        # Determine overall failure reason from runs
        failed_runs = [r for r in results if r.get('exit_code', 0) != 0]
        failure_reason = None
        failure_details = None

        if failed_runs:
            # Use the first failure's reason
            failure_reason = failed_runs[0].get('failure_reason', self.FAILURE_POD_ERROR)
            failure_details = failed_runs[0].get('failure_details', f"{len(failed_runs)}/{num_runs} runs failed")

        # Return aggregated results
        return {
            "tool": "kv-cache-tester",
            "variant": variant,
            "runs": results,
            "clone_run": clone_run,
            "num_runs": num_runs,
            "exit_code": 0 if all(r['exit_code'] == 0 for r in results) else 1,
            "failure_reason": failure_reason,
            "failure_details": failure_details,
        }

    def _execute_single_run(self, benchmark_args: Dict[str, Any], variant: str,
                            api_endpoint: str, run_dir: Path, run_label: str,
                            image: str, namespace: str) -> Dict[str, Any]:
        """Execute a single kv-cache-tester benchmark run."""

        # Build benchmark arguments
        benchmark_cmd_args = self.build_args(
            benchmark_args,
            model="",  # Not used by kv-cache-tester
            service_info={'api_endpoint': api_endpoint},
            run_dir=run_dir,
            variant=variant
        )

        # Build command using generic runner
        cmd = [
            "./scripts/run-k8s-benchmark.sh",
            "--tool-name", "kv-cache-tester",
            "--manifest-generator", "./load-generators/kv-cache-tester/generate-pod-manifest.sh",
            "--image", image,
            "--namespace", namespace,
            "--output-dir", str(run_dir.absolute()),
            "--run-label", run_label,
            "--completion-timeout", "10800",
            "--",  # Separator for benchmark args
            "--variant", variant,  # Variant is passed first for the manifest generator
        ]
        cmd.extend(benchmark_cmd_args)

        # Execute using common helper
        exec_result = self.execute_benchmark(
            cmd=cmd,
            run_dir=run_dir,
            run_label=run_label,
            image=image,
            tool_name=f"kv-cache-tester {variant} ({run_label})"
        )

        # Parse metrics from output file
        output_json_path = run_dir / f"benchmark_output_{run_label}.json"
        parsed_result = self.parse_metrics(output_json_path)

        # Determine failure reason - execution failure takes precedence
        failure_reason = exec_result.get('failure_reason')
        failure_details = exec_result.get('failure_details')

        if failure_reason is None and parsed_result.get('parsing_status') == 'failed':
            failure_reason = self.FAILURE_PARSING
            failure_details = "; ".join(parsed_result.get('parsing_errors', ['Unknown parsing error']))

        return {
            "run_label": run_label,
            "variant": variant,
            "output_file": str(output_json_path),
            "exit_code": exec_result['exit_code'],
            "failure_reason": failure_reason,
            "failure_details": failure_details,
            **parsed_result  # Add metrics, parsing_status, parsing_errors
        }
