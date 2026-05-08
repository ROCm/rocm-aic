#!/usr/bin/env python3
"""
LMCache and Bench ConfigMap injection for existing Kustomization templates.

This module provides functions to:
1. Load existing kustomization.yaml.tmpl files
2. Replace template variables ({{model}}, {{ENGINE_ARGS_ARRAY}}, etc.)
3. Inject dynamic ConfigMap with LMCache configuration
4. Inject dynamic ConfigMap with bench configuration
5. Preserve all other template structure and patches
"""

import yaml
import re
from typing import Dict, Any, Optional
from pathlib import Path


def inject_lmcache_configmap(
    template_content: str,
    params: Dict[str, Any],
    lmcache_config_yaml: str
) -> str:
    """
    Inject LMCache ConfigMap into existing kustomization template.

    This function:
    1. Replaces all {{PLACEHOLDER}} variables with values from params
    2. Replaces the hardcoded ConfigMap literals with dynamic lmcache_config_yaml
    3. Preserves all other template structure (resources, patches, etc.)

    Args:
        template_content: Original template file content
        params: Dictionary of template parameters (model, tensor_parallel_size, ENGINE_ARGS_ARRAY, etc.)
        lmcache_config_yaml: Generated LMCache YAML configuration string

    Returns:
        Rendered kustomization.yaml content with injected ConfigMap

    Example:
        >>> template = open('template.yaml.tmpl').read()
        >>> params = {'model': 'Qwen/Qwen3-32B', 'ENGINE_ARGS_ARRAY': '--max-num-seq 1024'}
        >>> lmcache_yaml = 'chunk_size: 256\\nsave_decode_cache: true'
        >>> result = inject_lmcache_configmap(template, params, lmcache_yaml)
        >>> 'chunk_size: 256' in result
        True
    """
    # Step 1: Replace all {{PLACEHOLDER}} variables
    rendered = template_content
    for key, value in params.items():
        placeholder = f"{{{{{key}}}}}"
        rendered = rendered.replace(placeholder, str(value))

    # Step 2: Replace ONLY the lmcache_config.yaml literal content
    #
    # IMPORTANT: We only replace the specific literal that starts with "lmcache_config.yaml="
    # and nothing else in the configMapGenerator section.
    #
    # The template structure is:
    # configMapGenerator:
    #   - name: config-map
    #     literals:
    #       - |
    #         lmcache_config.yaml=chunk_size: 256
    #         save_decode_cache: True
    #         ...
    #       - |                              ← Other literals should NOT be touched
    #         other_config.yaml=...
    #
    # Strategy: Match the literal block that contains "lmcache_config.yaml=" and replace
    # only the content AFTER the "=" sign until the end of that literal block.

    # Pattern explanation:
    # - Find "lmcache_config.yaml="
    # - Capture everything after "=" that's part of the same literal block
    # - A literal block ends when:
    #   a) We hit another "- |" (start of new literal)
    #   b) We hit a line that's not more indented than "literals:" (e.g., "patches:")
    #   c) End of file

    # More precise pattern: match from "lmcache_config.yaml=" to end of its literal block
    # The literal block is the content after the pipe (|) operator that's indented
    #
    # Pattern matches:
    # 1. Start: "lmcache_config.yaml=" at beginning of line (with possible leading whitespace)
    # 2. Content: rest of that line + subsequent lines that are MORE indented than the "lmcache" line
    # 3. Stop: when we hit a line that's LESS or EQUALLY indented (dedent = end of literal block)
    #
    # Example:
    #       - |                           ← literal marker
    #         lmcache_config.yaml=...     ← this line (indent=8)
    #         key: value                  ← continuation (indent=8)
    #       - |                           ← next literal (indent=6) ← STOP HERE

    # Find the indentation level of the lmcache_config.yaml line
    # Then match all subsequent lines with same or greater indentation
    # Stop when we hit a line with less indentation or a new literal marker "- |"

    pattern = r'(\s*)(lmcache_config\.yaml=)[^\n]*(\n\1\s+[^\n]+)*'

    def replacement_func(match):
        # match.group(1) = leading whitespace (indentation)
        # match.group(2) = "lmcache_config.yaml="
        # match.group(3) = continuation lines

        indent = match.group(1)
        # Indent the new config to match the original indentation
        indented_config = lmcache_config_yaml.strip().replace('\n', f'\n{indent}')
        return f'{indent}lmcache_config.yaml={indented_config}'

    rendered = re.sub(pattern, replacement_func, rendered, count=1, flags=re.MULTILINE)

    return rendered


def inject_lmcache_configmap_yaml_parse(
    template_content: str,
    params: Dict[str, Any],
    lmcache_config_yaml: str
) -> str:
    """
    Alternative approach: Parse YAML, modify, and regenerate.

    This approach:
    1. Replaces template variables first
    2. Parses the result as YAML
    3. Modifies ONLY the lmcache_config.yaml literal in configMapGenerator
    4. Preserves all other literals
    5. Dumps back to YAML

    Pros: More robust, handles edge cases, very precise
    Cons: May lose formatting/comments

    Args:
        template_content: Original template file content
        params: Dictionary of template parameters
        lmcache_config_yaml: Generated LMCache YAML configuration string

    Returns:
        Rendered kustomization.yaml content
    """
    # Step 1: Replace template variables
    rendered = template_content
    for key, value in params.items():
        placeholder = f"{{{{{key}}}}}"
        rendered = rendered.replace(placeholder, str(value))

    # Step 2: Parse as YAML
    try:
        kustomization = yaml.safe_load(rendered)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse template as YAML: {e}")

    # Step 3: Modify configMapGenerator - ONLY touch lmcache_config.yaml literal
    if 'configMapGenerator' in kustomization:
        for configmap in kustomization['configMapGenerator']:
            if configmap.get('name') == 'config-map' and 'literals' in configmap:
                # Find and replace ONLY the lmcache_config.yaml literal
                # Preserve all other literals
                new_literals = []
                lmcache_found = False

                for literal in configmap['literals']:
                    if isinstance(literal, str) and literal.startswith('lmcache_config.yaml='):
                        # Replace this one
                        new_literals.append(f'lmcache_config.yaml={lmcache_config_yaml.strip()}')
                        lmcache_found = True
                    else:
                        # Keep all other literals unchanged
                        new_literals.append(literal)

                # If lmcache literal not found, add it
                if not lmcache_found:
                    new_literals.append(f'lmcache_config.yaml={lmcache_config_yaml.strip()}')

                configmap['literals'] = new_literals

    # Step 4: Dump back to YAML
    result = yaml.dump(
        kustomization,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=1000
    )

    return result


def inject_lmcache_configmap_into_rendered(
    rendered_content: str,
    lmcache_args: Dict[str, Any],
    use_yaml_parse: bool = True
) -> str:
    """
    Inject LMCache ConfigMap into already-rendered kustomization content.

    This function is called AFTER template variables have been replaced.
    It only modifies the lmcache_config.yaml configMapGenerator literal.

    Args:
        rendered_content: Already-rendered kustomization.yaml content (no {{placeholders}})
        lmcache_args: LMCache configuration dict
        use_yaml_parse: If True, use YAML parsing; if False, use regex

    Returns:
        Content with LMCache configmap injected
    """
    from lmcache_support import generate_lmcache_config_yaml
    lmcache_config_yaml = generate_lmcache_config_yaml(lmcache_args)

    if use_yaml_parse:
        # Parse, modify, dump back
        try:
            kustomization = yaml.safe_load(rendered_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse rendered content as YAML: {e}")

        # Modify configMapGenerator - ONLY touch lmcache_config.yaml literal
        if 'configMapGenerator' in kustomization:
            for configmap in kustomization['configMapGenerator']:
                if configmap.get('name') == 'config-map' and 'literals' in configmap:
                    new_literals = []
                    lmcache_found = False

                    for literal in configmap['literals']:
                        if isinstance(literal, str) and literal.startswith('lmcache_config.yaml='):
                            new_literals.append(f'lmcache_config.yaml={lmcache_config_yaml.strip()}')
                            lmcache_found = True
                        else:
                            new_literals.append(literal)

                    if not lmcache_found:
                        new_literals.append(f'lmcache_config.yaml={lmcache_config_yaml.strip()}')

                    configmap['literals'] = new_literals

        return yaml.dump(
            kustomization,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=1000
        )
    else:
        # Use regex-based replacement
        pattern = r'(\s*)(lmcache_config\.yaml=)[^\n]*(\n\1\s+[^\n]+)*'

        def replacement_func(match):
            indent = match.group(1)
            prefix = match.group(2)
            # Indent the lmcache config YAML
            indented_config = '\n'.join([indent + line for line in lmcache_config_yaml.strip().split('\n')])
            return f"{indent}{prefix}{indented_config}"

        return re.sub(pattern, replacement_func, rendered_content)


def inject_bench_configmap_yaml_parse(
    rendered_content: str,
    bench_args: Dict[str, Any]
) -> str:
    """
    Inject bench ConfigMap into already-rendered kustomization content.

    This function adds or updates the bench-config-map in the configMapGenerator section.

    Args:
        rendered_content: Already-rendered kustomization.yaml content (no {{placeholders}})
        bench_args: Bench configuration dict

    Returns:
        Content with bench ConfigMap injected
    """
    from lmcache_bench_support import generate_bench_config_json

    bench_config_json = generate_bench_config_json(bench_args)

    # Parse YAML
    try:
        kustomization = yaml.safe_load(rendered_content)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse rendered content as YAML: {e}")

    # Ensure configMapGenerator exists
    if 'configMapGenerator' not in kustomization:
        kustomization['configMapGenerator'] = []

    # Find or create bench-config-map
    bench_configmap_found = False
    for configmap in kustomization['configMapGenerator']:
        if configmap.get('name') == 'bench-config-map':
            # Update existing bench configmap
            configmap['literals'] = [f'bench_config.json={bench_config_json.strip()}']
            bench_configmap_found = True
            break

    # If not found, add new bench configmap
    if not bench_configmap_found:
        kustomization['configMapGenerator'].append({
            'name': 'bench-config-map',
            'literals': [f'bench_config.json={bench_config_json.strip()}']
        })

    # Dump back to YAML
    return yaml.dump(
        kustomization,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=1000
    )


def render_kustomization_with_lmcache_and_bench(
    template_file: Path,
    params: Dict[str, Any],
    lmcache_args: Optional[Dict[str, Any]] = None,
    bench_args: Optional[Dict[str, Any]] = None,
    use_yaml_parse: bool = True
) -> str:
    """
    Main entry point: Load template and render with LMCache and/or bench config.

    Args:
        template_file: Path to kustomization.yaml.tmpl file
        params: Template parameters (model, tensor_parallel_size, ENGINE_ARGS_ARRAY, etc.)
        lmcache_args: Optional LMCache configuration dict
        bench_args: Optional bench configuration dict
        use_yaml_parse: If True, use YAML parsing approach; if False, use regex

    Returns:
        Rendered kustomization.yaml content

    Example:
        >>> template_file = Path('templates/tiered-prefix-cache-lmcache-kustomization.yaml.tmpl')
        >>> params = {'model': 'Qwen/Qwen3-8B', 'ENGINE_ARGS_ARRAY': '--max-num-seq 1024'}
        >>> lmcache_args = {'chunk_size': 256}
        >>> bench_args = {'model': 'Qwen/Qwen3-8B', 'workload': 'long-doc-qa', 'kv_cache_volume': 10.0}
        >>> result = render_kustomization_with_lmcache_and_bench(template_file, params, lmcache_args, bench_args)
    """
    # Read template
    with open(template_file) as f:
        template_content = f.read()

    # Replace template variables first
    rendered = template_content
    for key, value in params.items():
        placeholder = f"{{{{{key}}}}}"
        rendered = rendered.replace(placeholder, str(value))

    # If no config to inject, return rendered template
    if not lmcache_args and not bench_args:
        return rendered

    # Inject LMCache config if present
    if lmcache_args:
        from lmcache_support import generate_lmcache_config_yaml
        lmcache_config_yaml = generate_lmcache_config_yaml(lmcache_args)

        if use_yaml_parse:
            rendered = inject_lmcache_configmap_yaml_parse(
                rendered,
                {},  # params already replaced
                lmcache_config_yaml
            )
        else:
            # Use regex method
            pattern = r'(\s*)(lmcache_config\.yaml=)[^\n]*(\n\1\s+[^\n]+)*'

            def replacement_func(match):
                indent = match.group(1)
                indented_config = lmcache_config_yaml.strip().replace('\n', f'\n{indent}')
                return f'{indent}lmcache_config.yaml={indented_config}'

            rendered = re.sub(pattern, replacement_func, rendered, count=1, flags=re.MULTILINE)

    # Inject bench config if present
    if bench_args:
        rendered = inject_bench_configmap_yaml_parse(rendered, bench_args)

    return rendered


def render_kustomization_with_lmcache(
    template_file: Path,
    params: Dict[str, Any],
    lmcache_args: Optional[Dict[str, Any]] = None,
    use_yaml_parse: bool = True  # Changed default to True - YAML parsing is more reliable
) -> str:
    """
    Main entry point: Load template and render with LMCache config.

    Kept for backward compatibility. Use render_kustomization_with_lmcache_and_bench for new code.

    Args:
        template_file: Path to kustomization.yaml.tmpl file
        params: Template parameters (model, tensor_parallel_size, ENGINE_ARGS_ARRAY, etc.)
        lmcache_args: Optional LMCache configuration dict
        use_yaml_parse: If True, use YAML parsing approach; if False, use regex

    Returns:
        Rendered kustomization.yaml content

    Example:
        >>> from lmcache_support import generate_lmcache_config_yaml
        >>> template_file = Path('templates/tiered-prefix-cache-lmcache-kustomization.yaml.tmpl')
        >>> params = {
        ...     'model': 'Qwen/Qwen3-32B',
        ...     'tensor_parallel_size': 1,
        ...     'ENGINE_ARGS_ARRAY': '--max-num-seq 1024'
        ... }
        >>> lmcache_args = {'chunk_size': 256, 'max_local_cpu_size': 100.0}
        >>> result = render_kustomization_with_lmcache(template_file, params, lmcache_args)
    """
    # Delegate to new combined function
    return render_kustomization_with_lmcache_and_bench(
        template_file,
        params,
        lmcache_args=lmcache_args,
        bench_args=None,
        use_yaml_parse=use_yaml_parse
    )


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("LMCache ConfigMap Injection - Examples")
    print("=" * 70)

    # Example 1: Simple template with regex replacement
    print("\n1. Simple Template Injection (Regex Method):")
    print("-" * 70)

    template = """apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - base-manifest

configMapGenerator:
  - name: config-map
    literals:
      - |
        lmcache_config.yaml=chunk_size: 256
        save_decode_cache: True
        local_cpu: true
        max_local_cpu_size: 100.0

patches:
  - target:
      kind: Deployment
      name: decode
    patch: |-
      - op: replace
        path: /spec/template/spec/containers/0/args/0
        value: |-
            {{model}} \\
            --tensor-parallel-size {{tensor_parallel_size}} \\
            {{ENGINE_ARGS_ARRAY}}
"""

    params = {
        'model': 'Qwen/Qwen3-32B',
        'tensor_parallel_size': 1,
        'ENGINE_ARGS_ARRAY': '--max-num-seq 1024'
    }

    lmcache_yaml = """chunk_size: 512
save_decode_cache: true
local_cpu: true
max_local_cpu_size: 150.0
local_disk: null"""

    result = inject_lmcache_configmap(template, params, lmcache_yaml)

    print("Injected ConfigMap section:")
    # Extract just the configMap section for display
    lines = result.split('\n')
    in_configmap = False
    for line in lines:
        if 'configMapGenerator:' in line:
            in_configmap = True
        if in_configmap:
            print(line)
            if line.strip() and not line.startswith(' ') and 'configMapGenerator' not in line:
                break

    # Verify replacements
    assert 'Qwen/Qwen3-32B' in result
    assert 'chunk_size: 512' in result
    assert 'max_local_cpu_size: 150.0' in result
    print("\n✅ Template injection successful!")

    # Example 2: YAML parsing method
    print("\n2. YAML Parsing Method:")
    print("-" * 70)

    result_yaml = inject_lmcache_configmap_yaml_parse(template, params, lmcache_yaml)

    print("ConfigMap section (YAML parse method):")
    data = yaml.safe_load(result_yaml)
    if 'configMapGenerator' in data:
        print(yaml.dump({'configMapGenerator': data['configMapGenerator']}, default_flow_style=False))

    assert 'chunk_size: 512' in result_yaml
    print("✅ YAML parsing method successful!")

    # Example 3: Using actual template file
    print("\n3. Using Actual Template File:")
    print("-" * 70)

    template_file = Path(__file__).parent.parent / 'templates' / 'tiered-prefix-cache-lmcache-kustomization.yaml.tmpl'

    if template_file.exists():
        print(f"Template file found: {template_file.name}")

        from lmcache_support import generate_lmcache_config_yaml

        lmcache_args = {
            'chunk_size': 512,
            'save_decode_cache': True,
            'local_cpu': True,
            'max_local_cpu_size': 150.0,
            'local_disk': None,
            'remote_url': None
        }

        params_full = {
            'model': 'Qwen/Qwen3-32B',
            'tensor_parallel_size': 2,
            'ENGINE_ARGS_ARRAY': '--max-num-seq 2048 --gpu-memory-utilization 0.9'
        }

        result = render_kustomization_with_lmcache(
            template_file,
            params_full,
            lmcache_args,
            use_yaml_parse=False
        )

        print("\nGenerated kustomization.yaml (first 800 chars):")
        print("-" * 70)
        print(result[:800])
        print("...")
        print("-" * 70)

        # Verify key content
        assert 'Qwen/Qwen3-32B' in result
        assert 'chunk_size: 512' in result
        assert 'tensor-parallel-size 2' in result
        assert '--max-num-seq 2048' in result

        print("\n✅ Actual template rendering successful!")
    else:
        print(f"⚠️  Template file not found: {template_file}")
        print("   Skipping this example")

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
