"""
Long document QA load generator.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List

from .base import LoadGeneratorBase


class LongDocQA(LoadGeneratorBase):
    """Long document QA benchmarking tool."""

    def build_args(self, benchmark_args: Dict[str, Any], model: str,
                   service_info: Dict[str, str], **kwargs) -> List[str]:
        """
        Build command-line arguments for long_doc_qa.py script.

        Args:
            benchmark_args: Additional benchmark configuration from sweep YAML
            model: Model name (auto-injected from sweep config)
            service_info: Dict containing 'base_url' key
            **kwargs: Must contain 'run_dir' (for output file paths)

        Returns:
            List of command-line arguments
        """
        run_dir = kwargs['run_dir']

        # Start with fixed args: base-url and model
        args = [
            "--base-url", service_info['base_url'],
            "--model", model,
        ]

        # Always enable JSON output for structured parsing
        args.append("--json-output")

        # Add benchmark_args using orchestrator's conversion utility
        # This handles underscore->hyphen conversion and boolean flags
        args.extend(self.orchestrator._convert_args_dict(benchmark_args))

        return args

    def run(self, config: Dict[str, Any], run_dir: Path,
            params: Dict[str, Any], namespace: str) -> Dict[str, Any]:
        """Run long_doc_qa benchmark."""

        # Get configuration
        image = config.get('image', 'ghcr.io/vcave/vllm:latest')

        # Extract benchmark_args (handle combinations pattern)
        benchmark_args_config = config.get('benchmark_args', {})
        if isinstance(benchmark_args_config, dict) and benchmark_args_config.get('type') in ['combinations', 'pairwise']:
            # Extract fixed args (non-sweepable)
            benchmark_args = {}
            for key, value in benchmark_args_config.get('args', {}).items():
                if not (isinstance(value, dict) and 'values' in value):
                    benchmark_args[key] = value
        else:
            benchmark_args = benchmark_args_config.copy() if benchmark_args_config else {}

        # Merge load generation sweep parameters (if any)
        if '_load_params' in params:
            benchmark_args.update(params['_load_params'])

        # Get model and construct base URL
        model = params.get('model')
        base_url = f"http://llm-d-inference-gateway-istio.{namespace}.svc.cluster.local:80/v1"

        # Determine number of runs
        #TODO clean this up
        num_runs = 1

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
                image=image,
                namespace=namespace
            )
            results.append(result)

        # Return aggregated results
        return {
            "tool": "long_doc_qa",
            "runs": results,
            "num_runs": num_runs,
            "exit_code": 0 if all(r['exit_code'] == 0 for r in results) else 1
        }

    def _execute_single_run(self, benchmark_args: Dict[str, Any], model: str,
                            base_url: str, run_dir: Path, run_label: str,
                            image: str, namespace: str) -> Dict[str, Any]:
        """Execute a single long_doc_qa benchmark run."""

        # Build benchmark arguments
        benchmark_cmd_args = self.build_args(
            benchmark_args,
            model,
            {'base_url': base_url},
            run_dir=run_dir
        )

        # Build command using generic runner
        cmd = [
            "./scripts/run-k8s-benchmark.sh",
            "--tool-name", "long-doc-qa",
            "--manifest-generator", "./load-generators/long_doc_qa/generate-pod-manifest.sh",
            "--image", image,
            "--namespace", namespace,
            "--output-dir", str(run_dir.absolute()),
            "--run-label", run_label,
            "--completion-timeout", "10800",
            "--",  # Separator for benchmark args
        ]
        cmd.extend(benchmark_cmd_args)

        print(f"    Running long_doc_qa (image: {image})...")

        # Execute
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"    Warning: Benchmark exited with code {result.returncode}")

        # Save runner output
        output_file = run_dir / f"benchmark_runner_output_{run_label}.txt"
        with open(output_file, "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)

        # Parse metrics from output files (in run_label subdirectory)
        parsed_result = self.parse_metrics(run_dir / run_label)

        return {
            "run_label": run_label,
            "output_dir": str(run_dir / run_label),
            "exit_code": result.returncode,
            **parsed_result  # Add metrics, parsing_status, parsing_errors
        }

    def parse_metrics(self, output_file: Path) -> Dict[str, Any]:
        """
        Parse long_doc_qa output (CSV and JSON format).

        Extracts performance metrics from CSV files (warmup_round.csv, query_round.csv)
        and JSON summary output generated by long_doc_qa.py.

        Args:
            output_file: Path to directory containing benchmark outputs
                         (e.g., run_dir/run1/)

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
            # output_file is actually a directory path
            output_dir = output_file

            # Parse JSON summary (extracted from logs)
            json_summary = self._parse_json_summary(output_dir)

            # Parse CSV files
            warmup_csv = output_dir / "warmup_round.csv"
            query_csv = output_dir / "query_round.csv"

            warmup_stats = self._parse_csv_file(warmup_csv)
            query_stats = self._parse_csv_file(query_csv)

            # Combine metrics
            metrics = {}
            found_any = False

            # Add JSON summary metrics
            if json_summary:
                metrics["summary"] = json_summary
                found_any = True

            # Add warmup round metrics
            if warmup_stats:
                metrics["warmup_round"] = warmup_stats
                found_any = True

            # Add query round metrics
            if query_stats:
                metrics["query_round"] = query_stats
                found_any = True

            result["metrics"] = metrics

            if found_any:
                if json_summary and warmup_stats and query_stats:
                    result["parsing_status"] = "success"
                else:
                    result["parsing_status"] = "partial"
                    if not json_summary:
                        result["parsing_errors"].append("JSON summary not found")
                    if not warmup_stats:
                        result["parsing_errors"].append("warmup_round.csv not found or empty")
                    if not query_stats:
                        result["parsing_errors"].append("query_round.csv not found or empty")
            else:
                result["parsing_status"] = "failed"
                result["parsing_errors"].append("No recognized metrics found in output")

        except Exception as e:
            result["parsing_errors"].append(f"Unexpected error: {str(e)}")

        return result

    def _parse_json_summary(self, output_dir: Path) -> Dict[str, Any]:
        """
        Parse JSON summary from log file.

        long_doc_qa.py prints a JSON summary to stdout when --json-output is used.
        This is captured in the pod logs.
        """
        log_file = output_dir.parent / f"benchmark_output_{output_dir.name}.log"

        if not log_file.exists():
            return None

        try:
            with open(log_file, 'r') as f:
                content = f.read()

            # Find JSON line (looks for line starting with '{' and containing expected keys)
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('{') and 'query_ttft_per_prompt' in line:
                    return json.loads(line)

        except Exception:
            pass

        return None

    def _parse_csv_file(self, csv_path: Path) -> Dict[str, Any]:
        """
        Parse CSV file (warmup_round.csv or query_round.csv).

        CSV columns (from long_doc_qa.py):
        - prompt_id: int
        - request_start: float (relative to round start)
        - ttft: float (time to first token, seconds)
        - request_end: float (relative to round start)
        - successful: bool
        - is_miss: bool (only in query_round.csv if hit_miss_ratio is set)
        """
        if not csv_path.exists():
            return None

        try:
            import pandas as pd

            df = pd.DataFrame()
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                return None

            if df.empty:
                return None

            # Calculate metrics
            successful_df = df[df['successful'] == True]

            if successful_df.empty:
                return {
                    "total_requests": len(df),
                    "successful_requests": 0,
                    "failed_requests": len(df)
                }

            # Calculate TTFT statistics (in milliseconds)
            ttft_seconds = successful_df['ttft']
            ttft_ms = ttft_seconds * 1000

            # Calculate latency statistics (request_end - request_start)
            latency_seconds = successful_df['request_end'] - successful_df['request_start']
            latency_ms = latency_seconds * 1000

            metrics = {
                "total_requests": len(df),
                "successful_requests": len(successful_df),
                "failed_requests": len(df) - len(successful_df),
                "ttft": {
                    "mean_ms": float(ttft_ms.mean()),
                    "median_ms": float(ttft_ms.median()),
                    "p95_ms": float(ttft_ms.quantile(0.95)),
                    "p99_ms": float(ttft_ms.quantile(0.99)),
                    "min_ms": float(ttft_ms.min()),
                    "max_ms": float(ttft_ms.max()),
                },
                "e2e_latency": {
                    "mean_ms": float(latency_ms.mean()),
                    "median_ms": float(latency_ms.median()),
                    "p95_ms": float(latency_ms.quantile(0.95)),
                    "p99_ms": float(latency_ms.quantile(0.99)),
                    "min_ms": float(latency_ms.min()),
                    "max_ms": float(latency_ms.max()),
                }
            }

            # Add cache hit/miss stats if available (query round only)
            if 'is_miss' in df.columns:
                cache_hits = len(df[df['is_miss'] == False])
                cache_misses = len(df[df['is_miss'] == True])
                metrics["cache_stats"] = {
                    "hits": cache_hits,
                    "misses": cache_misses,
                    "hit_rate": cache_hits / len(df) if len(df) > 0 else 0.0
                }

            return metrics

        except ImportError:
            # pandas not available
            return None
        except Exception:
            return None
