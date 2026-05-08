#!/usr/bin/env python3
"""
LMCache Bench configuration support for benchmark sweeps.

This module provides functions to:
1. Expand bench_args parameter combinations
2. Generate LMCache Bench JSON configuration files
3. Create ConfigMap sections for Kustomization
4. Patch templates with dynamic bench configuration
"""

import json
import itertools
from typing import Dict, List, Any, Optional


def expand_bench_args(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand bench_args combinations into individual configurations.

    Similar to lmcache_args expansion, this handles both fixed parameters and
    sweepable parameters with 'values' lists.

    Args:
        args: Dictionary of bench args where values can be:
            - Fixed values (int, float, str, bool, None)
            - Dicts with 'values' key containing a list of options

    Returns:
        List of dictionaries, each representing one bench configuration

    Example:
        >>> args = {
        ...     'model': 'Qwen/Qwen3-8B',
        ...     'workload': 'long-doc-qa',
        ...     'kv_cache_volume': 10.0,
        ...     'tokens_per_gb_kvcache': 46020,
        ...     'ldqa_query_per_document': {'values': [1, 2, 4]},
        ...     'ldqa_shuffle_policy': {'values': ['tile', 'random']}
        ... }
        >>> configs = expand_bench_args(args)
        >>> len(configs)
        6
        >>> configs[0]['ldqa_query_per_document']
        1
    """
    # Separate fixed args from sweepable args
    fixed = {}
    sweepable = {}

    for key, value in args.items():
        if isinstance(value, dict) and 'values' in value:
            # This is a sweepable parameter
            sweepable[key] = value['values']
        else:
            # Fixed parameter
            fixed[key] = value

    # If no sweepable parameters, return single config
    if not sweepable:
        return [args.copy()]

    # Generate combinations of sweepable parameters
    combinations = []
    keys = sweepable.keys()
    for values in itertools.product(*sweepable.values()):
        config = fixed.copy()
        config.update(dict(zip(keys, values)))
        combinations.append(config)

    return combinations


def generate_bench_config_json(bench_args: Dict[str, Any]) -> str:
    """
    Generate LMCache Bench JSON configuration from bench_args.

    Converts a dictionary of bench parameters into a JSON string suitable
    for embedding in a ConfigMap.

    Args:
        bench_args: Dictionary of bench configuration parameters

    Returns:
        JSON string representation of the configuration

    Example:
        >>> args = {
        ...     'model': 'Qwen/Qwen3-8B',
        ...     'workload': 'long-doc-qa',
        ...     'kv_cache_volume': 10.0,
        ...     'tokens_per_gb_kvcache': 46020,
        ...     'ldqa_document_length': 10000,
        ...     'ldqa_query_per_document': 1,
        ...     'ldqa_shuffle_policy': 'tile',
        ...     'ldqa_num_inflight_requests': 4
        ... }
        >>> json_str = generate_bench_config_json(args)
        >>> 'Qwen/Qwen3-8B' in json_str
        True
    """
    # Use json.dumps with proper formatting
    json_str = json.dumps(
        bench_args,
        indent=2,
        sort_keys=True
    )

    return json_str


def generate_bench_configmap_section(bench_config_json: str) -> str:
    """
    Generate ConfigMap section for bench configuration in Kustomization file.

    Creates a configMapGenerator section that embeds the bench JSON
    configuration as a literal value.

    Args:
        bench_config_json: JSON string of bench configuration

    Returns:
        ConfigMap section as string, ready to be inserted into kustomization.yaml

    Example:
        >>> json_config = '{"model": "Qwen/Qwen3-8B", "workload": "long-doc-qa"}'
        >>> configmap = generate_bench_configmap_section(json_config)
        >>> 'bench-config-map' in configmap
        True
    """
    # Indent the JSON config for proper formatting in the literal
    indented_json = bench_config_json.strip()

    configmap_section = f"""  - name: bench-config-map
    literals:
      - |
        bench_config.json={indented_json}"""

    return configmap_section


def generate_bench_volume_patches() -> str:
    """
    Generate volume and volumeMount patches for bench config.

    Creates JSON patches to:
    1. Add a volume backed by the bench ConfigMap
    2. Mount the volume into the vLLM container

    Returns:
        Multi-line string with volume patches
    """
    volume_patches = """      - op: add
        path: /spec/template/spec/volumes/-
        value:
          name: bench-config-volume
          configMap:
            name: bench-config-map
      - op: add
        path: /spec/template/spec/containers/0/volumeMounts/-
        value:
          name: bench-config-volume
          mountPath: /opt/bench-configs
          readOnly: true"""

    return volume_patches


def validate_bench_args(bench_args: Dict[str, Any]) -> List[str]:
    """
    Validate bench_args configuration.

    Checks for common issues like:
    - Required fields
    - Type mismatches
    - Invalid value ranges
    - Invalid workload types

    Args:
        bench_args: Bench configuration to validate

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check required fields
    required_fields = ['model', 'workload', 'kv_cache_volume', 'tokens_per_gb_kvcache']
    for field in required_fields:
        if field not in bench_args:
            errors.append(f"Missing required field: {field}")

    # Validate workload type
    workload = bench_args.get('workload')
    if workload is not None:
        valid_workloads = ['long-doc-qa', 'short-doc-qa', 'multi-turn']
        if isinstance(workload, dict) and 'values' in workload:
            # Sweepable parameter
            for val in workload['values']:
                if val not in valid_workloads:
                    errors.append(f"Invalid workload value: {val} (must be one of {valid_workloads})")
        elif workload not in valid_workloads:
            errors.append(f"Invalid workload: {workload} (must be one of {valid_workloads})")

    # Validate kv_cache_volume
    kv_volume = bench_args.get('kv_cache_volume')
    if kv_volume is not None:
        if isinstance(kv_volume, dict) and 'values' in kv_volume:
            for val in kv_volume['values']:
                if not isinstance(val, (int, float)) or val <= 0:
                    errors.append(f"Invalid kv_cache_volume value: {val} (must be positive number)")
        elif not isinstance(kv_volume, (int, float)) or kv_volume <= 0:
            errors.append(f"Invalid kv_cache_volume: {kv_volume} (must be positive number)")

    # Validate tokens_per_gb_kvcache
    tokens_per_gb = bench_args.get('tokens_per_gb_kvcache')
    if tokens_per_gb is not None:
        if isinstance(tokens_per_gb, dict) and 'values' in tokens_per_gb:
            for val in tokens_per_gb['values']:
                if not isinstance(val, int) or val <= 0:
                    errors.append(f"Invalid tokens_per_gb_kvcache value: {val} (must be positive integer)")
        elif not isinstance(tokens_per_gb, int) or tokens_per_gb <= 0:
            errors.append(f"Invalid tokens_per_gb_kvcache: {tokens_per_gb} (must be positive integer)")

    # Validate optional integer fields
    int_fields = ['ldqa_document_length', 'ldqa_query_per_document', 'ldqa_num_inflight_requests']
    for field in int_fields:
        value = bench_args.get(field)
        if value is not None:
            if isinstance(value, dict) and 'values' in value:
                for val in value['values']:
                    if not isinstance(val, int) or val <= 0:
                        errors.append(f"Invalid {field} value: {val} (must be positive integer)")
            elif not isinstance(value, int) or value <= 0:
                errors.append(f"Invalid {field}: {value} (must be positive integer)")

    # Validate shuffle policy
    shuffle_policy = bench_args.get('ldqa_shuffle_policy')
    if shuffle_policy is not None:
        valid_policies = ['tile', 'random', 'sequential']
        if isinstance(shuffle_policy, dict) and 'values' in shuffle_policy:
            for val in shuffle_policy['values']:
                if val not in valid_policies:
                    errors.append(f"Invalid ldqa_shuffle_policy value: {val} (must be one of {valid_policies})")
        elif shuffle_policy not in valid_policies:
            errors.append(f"Invalid ldqa_shuffle_policy: {shuffle_policy} (must be one of {valid_policies})")

    return errors


# Example usage demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("LMCache Bench Configuration Support - Examples")
    print("=" * 70)

    # Example 1: Simple fixed configuration
    print("\n1. Simple Fixed Configuration:")
    print("-" * 70)
    simple_args = {
        'model': 'Qwen/Qwen3-8B',
        'workload': 'long-doc-qa',
        'kv_cache_volume': 10.0,
        'tokens_per_gb_kvcache': 46020,
        'ldqa_document_length': 10000,
        'ldqa_query_per_document': 1,
        'ldqa_shuffle_policy': 'tile',
        'ldqa_num_inflight_requests': 4
    }

    configs = expand_bench_args(simple_args)
    print(f"Number of configurations: {len(configs)}")
    print("Configuration:")
    print(json.dumps(configs[0], indent=2))

    json_config = generate_bench_config_json(configs[0])
    print("\nGenerated JSON:")
    print(json_config)

    # Example 2: Parameter sweep
    print("\n2. Parameter Sweep:")
    print("-" * 70)
    sweep_args = {
        'model': 'Qwen/Qwen3-8B',
        'workload': 'long-doc-qa',
        'kv_cache_volume': 10.0,
        'tokens_per_gb_kvcache': 46020,
        'ldqa_document_length': 10000,
        'ldqa_query_per_document': {'values': [1, 2, 4]},
        'ldqa_shuffle_policy': {'values': ['tile', 'random']},
        'ldqa_num_inflight_requests': {'values': [4, 8]}
    }

    configs = expand_bench_args(sweep_args)
    print(f"Number of configurations: {len(configs)}")
    print("\nFirst 3 configurations:")
    for i, config in enumerate(configs[:3], 1):
        print(f"\nConfig {i}:")
        print(f"  ldqa_query_per_document: {config['ldqa_query_per_document']}")
        print(f"  ldqa_shuffle_policy: {config['ldqa_shuffle_policy']}")
        print(f"  ldqa_num_inflight_requests: {config['ldqa_num_inflight_requests']}")

    # Example 3: ConfigMap generation
    print("\n3. ConfigMap Section Generation:")
    print("-" * 70)
    json_config = generate_bench_config_json(configs[0])
    configmap = generate_bench_configmap_section(json_config)
    print(configmap)

    # Example 4: Volume patches
    print("\n4. Volume Patches:")
    print("-" * 70)
    volume_patches = generate_bench_volume_patches()
    print(volume_patches)

    # Example 5: Validation
    print("\n5. Validation:")
    print("-" * 70)
    invalid_args = {
        'model': 'Qwen/Qwen3-8B',
        'workload': 'invalid-workload',
        'kv_cache_volume': -10,
        'tokens_per_gb_kvcache': 'invalid'
    }
    errors = validate_bench_args(invalid_args)
    if errors:
        print("Validation errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid")

    # Test valid configuration
    valid_errors = validate_bench_args(simple_args)
    if not valid_errors:
        print("\nValid configuration passed validation!")
    else:
        print(f"\nUnexpected errors: {valid_errors}")

    print("\n" + "=" * 70)
