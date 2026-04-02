#!/usr/bin/env python3
"""
LMCache configuration support for benchmark sweeps.

This module provides functions to:
1. Expand lmcache_args parameter combinations
2. Generate LMCache YAML configuration files
3. Create ConfigMap sections for Kustomization
4. Patch templates with dynamic LMCache configuration
"""

import yaml
import itertools
from typing import Dict, List, Any, Optional


def expand_lmcache_args(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand lmcache_args combinations into individual configurations.

    Similar to vllm_args expansion, this handles both fixed parameters and
    sweepable parameters with 'values' lists.

    Args:
        args: Dictionary of LMCache args where values can be:
            - Fixed values (int, float, str, bool, None)
            - Dicts with 'values' key containing a list of options
            - Nested dicts (for complex config structures)

    Returns:
        List of dictionaries, each representing one LMCache configuration

    Example:
        >>> args = {
        ...     'chunk_size': {'values': [128, 256]},
        ...     'save_decode_cache': True,
        ...     'max_local_cpu_size': {'values': [100.0, 200.0]}
        ... }
        >>> configs = expand_lmcache_args(args)
        >>> len(configs)
        4
        >>> configs[0]
        {'chunk_size': 128, 'save_decode_cache': True, 'max_local_cpu_size': 100.0}
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


def generate_lmcache_config_yaml(lmcache_args: Dict[str, Any]) -> str:
    """
    Generate LMCache YAML configuration from lmcache_args.

    Converts a dictionary of LMCache parameters into a YAML string suitable
    for embedding in a ConfigMap.

    Args:
        lmcache_args: Dictionary of LMCache configuration parameters

    Returns:
        YAML string representation of the configuration

    Example:
        >>> args = {
        ...     'chunk_size': 256,
        ...     'save_decode_cache': True,
        ...     'local_cpu': True,
        ...     'max_local_cpu_size': 100.0,
        ...     'local_disk': None
        ... }
        >>> yaml_str = generate_lmcache_config_yaml(args)
        >>> print(yaml_str)
        chunk_size: 256
        save_decode_cache: true
        local_cpu: true
        max_local_cpu_size: 100.0
        local_disk: null
    """
    # Use yaml.dump with proper settings for clean output
    yaml_str = yaml.dump(
        lmcache_args,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=1000  # Prevent line wrapping
    )

    return yaml_str


def generate_configmap_section(lmcache_config_yaml: str) -> str:
    """
    Generate ConfigMap section for Kustomization file.

    Creates a configMapGenerator section that embeds the LMCache YAML
    configuration as a literal value.

    Args:
        lmcache_config_yaml: YAML string of LMCache configuration

    Returns:
        ConfigMap section as string, ready to be inserted into kustomization.yaml

    Example:
        >>> yaml_config = "chunk_size: 256\\nsave_decode_cache: true\\n"
        >>> configmap = generate_configmap_section(yaml_config)
        >>> 'configMapGenerator:' in configmap
        True
    """
    # Indent the YAML config for proper formatting in the literal
    indented_yaml = lmcache_config_yaml.strip()

    configmap_section = f"""configMapGenerator:
  - name: config-map
    literals:
      - |
        lmcache_config.yaml={indented_yaml}"""

    return configmap_section


def generate_env_patches() -> str:
    """
    Generate environment variable patches for LMCache.

    Creates JSON patches to add LMCache-related environment variables
    to the vLLM container.

    Returns:
        Multi-line string with environment variable patches
    """
    env_patches = """      - op: add
        path: /spec/template/spec/containers/0/env/-
        value:
          name: LMCACHE_CONFIG_FILE
          value: "/opt/configs/lmcache_config.yaml"
      - op: add
        path: /spec/template/spec/containers/0/env/-
        value:
          name: PYTHONHASHSEED
          value: "123"
      - op: add
        path: /spec/template/spec/containers/0/env/-
        value:
          name: PROMETHEUS_MULTIPROC_DIR
          value: "/tmp/lmcache_prometheus\""""

    return env_patches


def generate_volume_patches() -> str:
    """
    Generate volume and volumeMount patches for LMCache config.

    Creates JSON patches to:
    1. Add a volume backed by the ConfigMap
    2. Mount the volume into the vLLM container

    Returns:
        Multi-line string with volume patches
    """
    volume_patches = """      - op: add
        path: /spec/template/spec/volumes/-
        value:
          name: config-dir-volume
          configMap:
            name: config-map
      - op: add
        path: /spec/template/spec/containers/0/volumeMounts/-
        value:
          name: config-dir-volume
          mountPath: /opt/configs
          readOnly: true"""

    return volume_patches


def render_lmcache_template(
    template_content: str,
    params: Dict[str, Any],
    lmcache_args: Optional[Dict[str, Any]] = None
) -> str:
    """
    Render kustomization template with LMCache configuration.

    Fills in template placeholders including:
    - Regular parameters (model, tensor_parallel_size, etc.)
    - VLLM_ARGS
    - CONFIGMAP_SECTION (if lmcache_args present)
    - ENV_PATCHES (if lmcache_args present)
    - VOLUME_PATCHES (if lmcache_args present)

    Args:
        template_content: Template string with {{PLACEHOLDER}} markers
        params: Dictionary of template parameters
        lmcache_args: Optional LMCache configuration dict

    Returns:
        Rendered template string

    Example:
        >>> template = "model: {{model}}\\n{{CONFIGMAP_SECTION}}"
        >>> params = {'model': 'Qwen/Qwen3-32B'}
        >>> lmcache_args = {'chunk_size': 256}
        >>> rendered = render_lmcache_template(template, params, lmcache_args)
        >>> 'Qwen/Qwen3-32B' in rendered
        True
        >>> 'configMapGenerator' in rendered
        True
    """
    # Start with params copy
    template_params = params.copy()

    # Process lmcache_args if present
    if lmcache_args:
        # Generate LMCache YAML config
        lmcache_yaml = generate_lmcache_config_yaml(lmcache_args)

        # Generate all dynamic sections
        template_params["CONFIGMAP_SECTION"] = generate_configmap_section(lmcache_yaml)
        template_params["ENV_PATCHES"] = generate_env_patches()
        template_params["VOLUME_PATCHES"] = generate_volume_patches()
    else:
        # No lmcache config, use empty sections
        template_params["CONFIGMAP_SECTION"] = ""
        template_params["ENV_PATCHES"] = ""
        template_params["VOLUME_PATCHES"] = ""

    # Replace all placeholders
    rendered = template_content
    for key, value in template_params.items():
        placeholder = f"{{{{{key}}}}}"
        rendered = rendered.replace(placeholder, str(value))

    return rendered


def validate_lmcache_args(lmcache_args: Dict[str, Any]) -> List[str]:
    """
    Validate lmcache_args configuration.

    Checks for common issues like:
    - Required fields
    - Type mismatches
    - Invalid value ranges

    Args:
        lmcache_args: LMCache configuration to validate

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check required fields
    if 'chunk_size' not in lmcache_args:
        errors.append("Missing required field: chunk_size")

    # Validate chunk_size
    chunk_size = lmcache_args.get('chunk_size')
    if chunk_size is not None:
        if isinstance(chunk_size, dict) and 'values' in chunk_size:
            # Sweepable parameter - validate each value
            for val in chunk_size['values']:
                if not isinstance(val, int) or val <= 0:
                    errors.append(f"Invalid chunk_size value: {val} (must be positive integer)")
        elif not isinstance(chunk_size, int) or chunk_size <= 0:
            errors.append(f"Invalid chunk_size: {chunk_size} (must be positive integer)")

    # Validate max_local_cpu_size if present
    max_size = lmcache_args.get('max_local_cpu_size')
    if max_size is not None:
        if isinstance(max_size, dict) and 'values' in max_size:
            for val in max_size['values']:
                if not isinstance(val, (int, float)) or val <= 0:
                    errors.append(f"Invalid max_local_cpu_size value: {val} (must be positive number)")
        elif not isinstance(max_size, (int, float)) or max_size <= 0:
            errors.append(f"Invalid max_local_cpu_size: {max_size} (must be positive number)")

    # Validate booleans
    for bool_field in ['save_decode_cache', 'local_cpu']:
        value = lmcache_args.get(bool_field)
        if value is not None and not isinstance(value, bool):
            errors.append(f"{bool_field} must be boolean, got {type(value).__name__}")

    return errors


# Example usage demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("LMCache Configuration Support - Examples")
    print("=" * 70)

    # Example 1: Simple fixed configuration
    print("\n1. Simple Fixed Configuration:")
    print("-" * 70)
    simple_args = {
        'chunk_size': 256,
        'save_decode_cache': True,
        'local_cpu': True,
        'max_local_cpu_size': 100.0,
        'local_disk': None,
        'remote_url': None
    }

    configs = expand_lmcache_args(simple_args)
    print(f"Number of configurations: {len(configs)}")
    print("Configuration:")
    print(yaml.dump(configs[0], default_flow_style=False))

    yaml_config = generate_lmcache_config_yaml(configs[0])
    print("Generated YAML:")
    print(yaml_config)

    # Example 2: Parameter sweep
    print("\n2. Parameter Sweep:")
    print("-" * 70)
    sweep_args = {
        'chunk_size': {'values': [128, 256, 512]},
        'save_decode_cache': True,
        'local_cpu': True,
        'max_local_cpu_size': {'values': [50.0, 100.0, 150.0]},
        'local_disk': None,
        'remote_url': None
    }

    configs = expand_lmcache_args(sweep_args)
    print(f"Number of configurations: {len(configs)}")
    print("\nFirst 3 configurations:")
    for i, config in enumerate(configs[:3], 1):
        print(f"\nConfig {i}:")
        print(f"  chunk_size: {config['chunk_size']}")
        print(f"  max_local_cpu_size: {config['max_local_cpu_size']}")

    # Example 3: ConfigMap generation
    print("\n3. ConfigMap Section Generation:")
    print("-" * 70)
    yaml_config = generate_lmcache_config_yaml(configs[0])
    configmap = generate_configmap_section(yaml_config)
    print(configmap)

    # Example 4: Validation
    print("\n4. Validation:")
    print("-" * 70)
    invalid_args = {
        'chunk_size': -1,
        'max_local_cpu_size': "invalid"
    }
    errors = validate_lmcache_args(invalid_args)
    if errors:
        print("Validation errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid")

    print("\n" + "=" * 70)
