#!/usr/bin/env python3
"""
Parse vLLM pod logs to extract configuration and initialization information.
Generic script that can be used across all deployments.
"""

import re
import json
import sys
import subprocess
from typing import Dict, List, Any, Optional
import argparse


class VLLMLogParser:
    """Parser for vLLM logs to extract configuration information."""

    # Patterns to match in logs
    PATTERNS = {
        'non_default_args': r'non-default args: (.+)',
        'max_model_len': r'Using max model len (\d+)',
        'engine_init': r'Initializing a V1 LLM engine \(v([\d.]+)\) with config: (.+)',
        'attention_backend': r'Using (.+) Attention backend\.',
        'model_loading_time': r'Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds',
        'torch_compile_time': r'torch\.compile takes ([\d.]+) s in total',
        'kv_cache_memory': r'Available KV cache memory: ([\d.]+) GiB',
        'gpu_kv_cache_size': r'GPU KV cache size: ([\d,]+) tokens',
        'max_concurrency': r'Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x',
        'v1_connector': r'Creating v1 connector with name: (\w+) and engine_id: ([\w-]+)',
        'offloading_spec': r'Creating offloading spec with name: (\w+)',
    }

    def __init__(self, deduplicate: bool = False):
        self.deduplicate = deduplicate
        self.results = {
            'pod_name': None,
            'namespace': None,
            'vllm_version': None,
            'non_default_args': {},
            'model_config': {},
            'parallelism': {},
            'performance': {},
            'kv_cache': {},
            'connectors': {},
        }

    def fetch_pod_logs(self, namespace: str, pod_name: str) -> str:
        """Fetch logs from a Kubernetes pod."""
        try:
            result = subprocess.run(
                ['kubectl', 'logs', '-n', namespace, pod_name, '--tail=-1'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error fetching logs: {e}", file=sys.stderr)
            return ""

    def parse_non_default_args(self, match_text: str) -> Dict[str, Any]:
        """Parse non-default args from vLLM (it's a dict-like string)."""
        try:
            # The text is Python dict-like, try to eval it safely
            import ast
            return ast.literal_eval(match_text)
        except:
            # If literal_eval fails (e.g., KVTransferConfig objects),
            # return dict with raw text for string-based extraction
            return {'_raw_text': match_text}

    def extract_parallelism_info(self, args_dict: Dict[str, Any]):
        """Extract parallelism configuration from non-default args."""
        # If parsing failed, extract from raw text
        if '_raw_text' in args_dict:
            raw_text = args_dict['_raw_text']

            tp_match = re.search(r"'tensor_parallel_size':\s*(\d+)", raw_text)
            if tp_match:
                self.results['parallelism']['tensor_parallel_size'] = int(tp_match.group(1))

            pp_match = re.search(r"'pipeline_parallel_size':\s*(\d+)", raw_text)
            if pp_match:
                self.results['parallelism']['pipeline_parallel_size'] = int(pp_match.group(1))

            dp_match = re.search(r"'data_parallel_size':\s*(\d+)", raw_text)
            if dp_match:
                self.results['parallelism']['data_parallel_size'] = int(dp_match.group(1))
        else:
            # Extract from parsed dict
            if 'tensor_parallel_size' in args_dict:
                self.results['parallelism']['tensor_parallel_size'] = args_dict['tensor_parallel_size']

            if 'pipeline_parallel_size' in args_dict:
                self.results['parallelism']['pipeline_parallel_size'] = args_dict['pipeline_parallel_size']

            if 'data_parallel_size' in args_dict:
                self.results['parallelism']['data_parallel_size'] = args_dict['data_parallel_size']

    def extract_kv_connector_info(self, args_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract KV connector information from non-default args."""
        connector_info = {}

        # If parsing failed, use raw text
        if '_raw_text' in args_dict:
            raw_text = args_dict['_raw_text']

            # Extract connector name from raw text
            connector_match = re.search(r"kv_connector='(\w+)'", raw_text)
            if connector_match:
                connector_info['kv_connector'] = connector_match.group(1)

            # Extract CPU bytes from raw text
            cpu_bytes_match = re.search(r"'cpu_bytes_to_use':\s*(\d+)", raw_text)
            if cpu_bytes_match:
                cpu_bytes = int(cpu_bytes_match.group(1))
                connector_info['cpu_bytes_to_use'] = cpu_bytes
                connector_info['cpu_cache_size_gb'] = round(cpu_bytes / (1024**3), 2)

            return connector_info

        # Check if kv_transfer_config exists (parsed successfully)
        kv_config = args_dict.get('kv_transfer_config')
        if kv_config:
            # Extract connector name
            if hasattr(kv_config, 'kv_connector'):
                connector_info['kv_connector'] = kv_config.kv_connector
            elif isinstance(kv_config, dict):
                connector_info['kv_connector'] = kv_config.get('kv_connector')
            else:
                # Try to extract from string representation
                connector_match = re.search(r"kv_connector='(\w+)'", str(kv_config))
                if connector_match:
                    connector_info['kv_connector'] = connector_match.group(1)

            # Extract CPU bytes to use
            cpu_bytes = None
            if hasattr(kv_config, 'kv_connector_extra_config'):
                extra_config = kv_config.kv_connector_extra_config
                cpu_bytes = extra_config.get('cpu_bytes_to_use') if isinstance(extra_config, dict) else None
            elif isinstance(kv_config, dict) and 'kv_connector_extra_config' in kv_config:
                extra_config = kv_config['kv_connector_extra_config']
                cpu_bytes = extra_config.get('cpu_bytes_to_use') if isinstance(extra_config, dict) else None
            else:
                # Extract from string
                cpu_bytes_match = re.search(r"'cpu_bytes_to_use':\s*(\d+)", str(kv_config))
                if cpu_bytes_match:
                    cpu_bytes = int(cpu_bytes_match.group(1))

            if cpu_bytes:
                connector_info['cpu_bytes_to_use'] = cpu_bytes
                connector_info['cpu_cache_size_gb'] = round(cpu_bytes / (1024**3), 2)

        return connector_info

    def classify_connector(self) -> str:
        """
        Classify the connector type based on extracted information.
        Returns a human-readable classification.
        """
        connectors = self.results.get('connectors', {})

        # Get connector name (first key in structured connectors)
        connector_name = None
        connector_details = {}

        for key, value in connectors.items():
            if not key.startswith('_') and isinstance(value, dict):
                connector_name = key
                connector_details = value
                break

        if not connector_name:
            return "No KV Connector (standard vLLM)"

        offloading_spec = connector_details.get('offloading_spec', '')
        cpu_cache_size = connector_details.get('cpu_cache_size_gb')

        # Classify based on connector and spec
        if connector_name == 'OffloadingConnector':
            if offloading_spec == 'CPUOffloadingSpec':
                if cpu_cache_size:
                    return f"CPU Offloading ({cpu_cache_size}GB CPU cache)"
                return "CPU Offloading"
            elif offloading_spec:
                return f"Offloading ({offloading_spec})"
            return "OffloadingConnector (spec unknown)"

        elif connector_name == 'LMCacheConnector':
            if offloading_spec == 'CPUOffloadingSpec':
                if cpu_cache_size:
                    return f"LMCache CPU Offloading ({cpu_cache_size}GB CPU cache)"
                return "LMCache CPU Offloading"
            return "LMCache"

        elif connector_name:
            return f"Custom Connector ({connector_name})"

        return "Unrecognized"

    def parse_logs(self, logs: str):
        """Parse logs and extract configuration information."""
        lines = logs.split('\n')

        for line in lines:
            # Non-default args
            match = re.search(self.PATTERNS['non_default_args'], line)
            if match:
                args_dict = self.parse_non_default_args(match.group(1))
                self.results['non_default_args'] = args_dict

                # Extract parallelism info from args
                self.extract_parallelism_info(args_dict)

                # Extract KV connector info from args
                kv_info = self.extract_kv_connector_info(args_dict)
                if kv_info:
                    self.results['connectors'].update(kv_info)

            # Max model length
            match = re.search(self.PATTERNS['max_model_len'], line)
            if match:
                self.results['model_config']['max_model_len'] = int(match.group(1))

            # Engine initialization
            match = re.search(self.PATTERNS['engine_init'], line)
            if match:
                self.results['vllm_version'] = match.group(1)
                engine_config = match.group(2)

                # Extract parallelism from engine config if not already extracted from args
                if not self.results['parallelism'].get('tensor_parallel_size'):
                    tp_match = re.search(r'tensor_parallel_size=(\d+)', engine_config)
                    if tp_match:
                        self.results['parallelism']['tensor_parallel_size'] = int(tp_match.group(1))

                if not self.results['parallelism'].get('pipeline_parallel_size'):
                    pp_match = re.search(r'pipeline_parallel_size=(\d+)', engine_config)
                    if pp_match:
                        self.results['parallelism']['pipeline_parallel_size'] = int(pp_match.group(1))

                if not self.results['parallelism'].get('data_parallel_size'):
                    dp_match = re.search(r'data_parallel_size=(\d+)', engine_config)
                    if dp_match:
                        self.results['parallelism']['data_parallel_size'] = int(dp_match.group(1))

            # Attention backend
            match = re.search(self.PATTERNS['attention_backend'], line)
            if match:
                self.results['model_config']['attention_backend'] = match.group(1)

            # Model loading time
            match = re.search(self.PATTERNS['model_loading_time'], line)
            if match:
                self.results['performance']['model_loading_memory_gib'] = float(match.group(1))
                self.results['performance']['model_loading_time_seconds'] = float(match.group(2))

            # Torch compile time
            match = re.search(self.PATTERNS['torch_compile_time'], line)
            if match:
                self.results['performance']['torch_compile_time_seconds'] = float(match.group(1))

            # Available KV cache memory
            match = re.search(self.PATTERNS['kv_cache_memory'], line)
            if match:
                self.results['kv_cache']['available_memory_gib'] = float(match.group(1))

            # GPU KV cache size
            match = re.search(self.PATTERNS['gpu_kv_cache_size'], line)
            if match:
                tokens_str = match.group(1).replace(',', '')
                self.results['kv_cache']['gpu_cache_size_tokens'] = int(tokens_str)

            # Maximum concurrency
            match = re.search(self.PATTERNS['max_concurrency'], line)
            if match:
                tokens_str = match.group(1).replace(',', '')
                self.results['kv_cache']['max_tokens_per_request'] = int(tokens_str)
                self.results['kv_cache']['max_concurrency'] = float(match.group(2))

            # V1 connector (extract name only, ignore engine_id)
            match = re.search(self.PATTERNS['v1_connector'], line)
            if match:
                connector_name = match.group(1)
                # Store temporarily to merge later
                self.results['connectors']['_detected_connector'] = connector_name

            # Offloading spec
            match = re.search(self.PATTERNS['offloading_spec'], line)
            if match:
                # Store temporarily to merge later
                self.results['connectors']['_offloading_spec'] = match.group(1)

    def get_results(self) -> Dict[str, Any]:
        """Get parsed results."""
        results = self.results.copy()

        # Restructure connectors to nest under connector type
        results['connectors'] = self._restructure_connectors(results['connectors'])

        return results

    def _restructure_connectors(self, connectors: Dict[str, Any]) -> Dict[str, Any]:
        """Restructure connectors to nest details under connector type."""
        if not connectors:
            return {}

        # Make a copy to avoid modifying original
        conn_copy = connectors.copy()

        # Extract detected connector name
        detected_connector = conn_copy.pop('_detected_connector', None)
        kv_connector = conn_copy.pop('kv_connector', None)
        offloading_spec = conn_copy.pop('_offloading_spec', None)
        cpu_bytes = conn_copy.pop('cpu_bytes_to_use', None)
        cpu_cache_gb = conn_copy.pop('cpu_cache_size_gb', None)

        # Determine the actual connector name
        connector_name = detected_connector or kv_connector

        if not connector_name:
            return {}

        # Restructure: nest details under connector type
        structured = {
            connector_name: {}
        }

        # Add connector-specific details
        if offloading_spec:
            structured[connector_name]['offloading_spec'] = offloading_spec

        if cpu_bytes is not None:
            structured[connector_name]['cpu_bytes_to_use'] = cpu_bytes

        if cpu_cache_gb is not None:
            structured[connector_name]['cpu_cache_size_gb'] = cpu_cache_gb

        return structured

    def set_pod_info(self, namespace: str, pod_name: str):
        """Set pod identification information."""
        self.results['pod_name'] = pod_name
        self.results['namespace'] = namespace


def deduplicate_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deduplicate results across multiple pods.
    Extract common configuration and note per-pod differences.
    """
    if not all_results:
        return {}

    if len(all_results) == 1:
        return all_results[0]

    # Find common configuration
    common = {
        'deployment': 'multiple_pods',
        'pod_count': len(all_results),
        'pod_names': [r['pod_name'] for r in all_results],
    }

    # Check which fields are identical across all pods
    first = all_results[0]

    # vLLM version (should be same)
    if all(r.get('vllm_version') == first.get('vllm_version') for r in all_results):
        common['vllm_version'] = first.get('vllm_version')

    # Non-default args (should be same)
    if all(r.get('non_default_args') == first.get('non_default_args') for r in all_results):
        common['non_default_args'] = first.get('non_default_args')

    # Model config (should be same)
    if all(r.get('model_config') == first.get('model_config') for r in all_results):
        common['model_config'] = first.get('model_config')

    # Parallelism (should be same)
    if all(r.get('parallelism') == first.get('parallelism') for r in all_results):
        common['parallelism'] = first.get('parallelism')

    # Connectors (should be same)
    if all(r.get('connectors') == first.get('connectors') for r in all_results):
        common['connectors'] = first.get('connectors')

    # Performance metrics (aggregate)
    perf_fields = ['model_loading_time_seconds', 'model_loading_memory_gib', 'torch_compile_time_seconds']
    performance_per_pod = []
    for r in all_results:
        if r.get('performance'):
            performance_per_pod.append({
                'pod': r['pod_name'],
                **r['performance']
            })

    if performance_per_pod:
        common['performance'] = {
            'per_pod': performance_per_pod,
            'avg_model_loading_time_seconds': sum(p.get('model_loading_time_seconds', 0) for p in performance_per_pod) / len(performance_per_pod) if performance_per_pod else 0,
            'avg_torch_compile_time_seconds': sum(p.get('torch_compile_time_seconds', 0) for p in performance_per_pod) / len(performance_per_pod) if performance_per_pod else 0,
        }

    # KV cache (aggregate)
    kv_cache_per_pod = []
    for r in all_results:
        if r.get('kv_cache'):
            kv_cache_per_pod.append({
                'pod': r['pod_name'],
                **r['kv_cache']
            })

    if kv_cache_per_pod:
        common['kv_cache'] = {
            'per_pod': kv_cache_per_pod,
            'total_gpu_cache_tokens': sum(p.get('gpu_cache_size_tokens', 0) for p in kv_cache_per_pod),
            'total_available_memory_gib': sum(p.get('available_memory_gib', 0) for p in kv_cache_per_pod),
        }

        # If max concurrency is same across all, show it
        max_concurrencies = [p.get('max_concurrency') for p in kv_cache_per_pod if 'max_concurrency' in p]
        if max_concurrencies and all(mc == max_concurrencies[0] for mc in max_concurrencies):
            common['kv_cache']['max_concurrency_per_pod'] = max_concurrencies[0]

    return common


def main():
    parser = argparse.ArgumentParser(
        description='Parse vLLM pod logs to extract configuration information'
    )
    parser.add_argument(
        '-n', '--namespace',
        required=True,
        help='Kubernetes namespace'
    )
    parser.add_argument(
        '-p', '--pod',
        help='Specific pod name (if not provided, will process all vLLM pods)'
    )
    parser.add_argument(
        '-l', '--label-selector',
        default='llm-d.ai/inference-serving=true',
        help='Label selector for vLLM pods (default: llm-d.ai/inference-serving=true)'
    )
    parser.add_argument(
        '-d', '--deduplicate',
        action='store_true',
        help='Deduplicate common configuration across pods'
    )
    parser.add_argument(
        '--indent',
        type=int,
        default=2,
        help='JSON indentation (default: 2)'
    )

    args = parser.parse_args()

    # Get pods to process
    if args.pod:
        pods = [args.pod]
    else:
        # Get all pods matching label selector
        result = subprocess.run(
            ['kubectl', 'get', 'pods', '-n', args.namespace,
             '-l', args.label_selector,
             '-o', 'jsonpath={.items[*].metadata.name}'],
            capture_output=True,
            text=True,
            check=True
        )
        pods = result.stdout.strip().split()

    if not pods:
        print(json.dumps({'error': 'No pods found'}, indent=args.indent))
        sys.exit(1)

    # Parse logs from each pod
    all_results = []
    for pod in pods:
        parser = VLLMLogParser()
        parser.set_pod_info(args.namespace, pod)

        logs = parser.fetch_pod_logs(args.namespace, pod)
        if logs:
            parser.parse_logs(logs)
            all_results.append(parser.get_results())

    # Output results
    if args.deduplicate and len(all_results) > 1:
        output = deduplicate_results(all_results)
    elif len(all_results) == 1:
        output = all_results[0]
    else:
        output = {
            'pod_count': len(all_results),
            'pods': all_results
        }

    print(json.dumps(output, indent=args.indent))


if __name__ == '__main__':
    main()
