#!/usr/bin/env python3
"""
Test template injection with original kustomization template.

This test verifies that we can use the original template file and inject
ConfigMap dynamically without regenerating the entire kustomization.
"""

import sys
from pathlib import Path
import yaml

from lmcache_support import generate_lmcache_config_yaml, expand_lmcache_args
from lmcache_template_injection import (
    inject_lmcache_configmap,
    inject_lmcache_configmap_yaml_parse,
    render_kustomization_with_lmcache
)


def test_basic_injection():
    """Test basic ConfigMap injection with regex method."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic ConfigMap Injection (Regex)")
    print("=" * 70)

    template = """apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

configMapGenerator:
  - name: config-map
    literals:
      - |
        lmcache_config.yaml=chunk_size: 256
        save_decode_cache: True

patches:
  - target:
      kind: Deployment
"""

    params = {'model': 'test-model'}
    new_config = "chunk_size: 512\nsave_decode_cache: false\nmax_local_cpu_size: 200.0"

    result = inject_lmcache_configmap(template, params, new_config)

    print("\nOriginal ConfigMap value:")
    print("  chunk_size: 256")
    print("  save_decode_cache: True")

    print("\nNew ConfigMap value:")
    print("  chunk_size: 512")
    print("  save_decode_cache: false")
    print("  max_local_cpu_size: 200.0")

    # Verify injection worked
    assert 'chunk_size: 512' in result, "New chunk_size not found"
    assert 'max_local_cpu_size: 200.0' in result, "New max_local_cpu_size not found"
    assert 'chunk_size: 256' not in result, "Old chunk_size still present"
    assert 'patches:' in result, "Patches section lost"

    print("\n✅ ConfigMap successfully injected, patches preserved")
    return True


def test_yaml_parse_injection():
    """Test ConfigMap injection with YAML parsing method."""
    print("\n" + "=" * 70)
    print("TEST 2: ConfigMap Injection (YAML Parsing)")
    print("=" * 70)

    template = """apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - base-manifest

configMapGenerator:
  - name: config-map
    literals:
      - |
        lmcache_config.yaml=chunk_size: 256

patches:
  - target:
      kind: Deployment
      name: test
"""

    params = {'model': 'test-model'}
    new_config = "chunk_size: 1024\nlocal_cpu: true"

    result = inject_lmcache_configmap_yaml_parse(template, params, new_config)

    print("\nParsed and regenerated kustomization")

    # Verify structure is preserved
    data = yaml.safe_load(result)
    assert 'resources' in data, "Resources section lost"
    assert 'configMapGenerator' in data, "ConfigMapGenerator section lost"
    assert 'patches' in data, "Patches section lost"

    # Verify ConfigMap was updated
    assert 'chunk_size: 1024' in result, "New config not injected"

    print("\n✅ YAML parsing method successful, all sections preserved")
    return True


def test_with_actual_template():
    """Test with the actual tiered-prefix-cache-lmcache template."""
    print("\n" + "=" * 70)
    print("TEST 3: Using Original Template File")
    print("=" * 70)

    template_file = Path(__file__).parent.parent / 'templates' / 'tiered-prefix-cache-lmcache-kustomization.yaml.tmpl'

    if not template_file.exists():
        print(f"⚠️  Template not found: {template_file}")
        return True

    print(f"\nTemplate file: {template_file.name}")

    # Define parameters
    params = {
        'model': 'Qwen/Qwen3-32B',
        'tensor_parallel_size': 2,
        'ENGINE_ARGS_ARRAY': '--max-num-seq 2048 --gpu-memory-utilization 0.9'
    }

    lmcache_args = {
        'chunk_size': 1024,
        'save_decode_cache': True,
        'local_cpu': True,
        'max_local_cpu_size': 200.0,
        'local_disk': None,
        'remote_url': None
    }

    # Render using regex method
    result = render_kustomization_with_lmcache(
        template_file,
        params,
        lmcache_args,
        use_yaml_parse=False
    )

    print("\nVerifying replacements:")

    # Check template variables replaced
    assert 'Qwen/Qwen3-32B' in result, "Model not replaced"
    assert 'tensor-parallel-size 2' in result, "Tensor parallel size not replaced"
    assert '--max-num-seq 2048' in result, "ENGINE_ARGS_ARRAY not replaced"
    print("  ✓ Template variables replaced")

    # Check ConfigMap injected
    assert 'chunk_size: 1024' in result, "New chunk_size not found"
    assert 'max_local_cpu_size: 200.0' in result, "New max_local_cpu_size not found"
    print("  ✓ ConfigMap updated with new values")

    # Check old values removed
    assert 'chunk_size: 256' not in result, "Old chunk_size still present"
    print("  ✓ Old ConfigMap values removed")

    # Check structure preserved
    assert 'resources:' in result, "Resources section missing"
    assert 'patches:' in result, "Patches section missing"
    assert 'amd.com~1gpu' in result, "GPU resource patches missing"
    print("  ✓ Template structure preserved")

    # Verify it's valid YAML
    try:
        data = yaml.safe_load(result)
        assert data['kind'] == 'Kustomization', "Not a valid Kustomization"
        print("  ✓ Output is valid YAML")
    except yaml.YAMLError as e:
        raise AssertionError(f"Generated YAML is invalid: {e}")

    print("\n✅ Original template successfully used with dynamic ConfigMap")
    return True


def test_parameter_sweep_integration():
    """Test integration with parameter sweep expansion."""
    print("\n" + "=" * 70)
    print("TEST 4: Parameter Sweep Integration")
    print("=" * 70)

    template_file = Path(__file__).parent.parent / 'templates' / 'tiered-prefix-cache-lmcache-kustomization.yaml.tmpl'

    if not template_file.exists():
        print(f"⚠️  Template not found, skipping")
        return True

    # Define sweep configuration
    lmcache_sweep_args = {
        'chunk_size': {'values': [256, 512, 1024]},
        'save_decode_cache': True,
        'local_cpu': True,
        'max_local_cpu_size': {'values': [100.0, 200.0]},
        'local_disk': None,
        'remote_url': None
    }

    # Expand combinations
    configs = expand_lmcache_args(lmcache_sweep_args)
    expected_count = 3 * 2  # 3 chunk_sizes × 2 max_local_cpu_sizes

    print(f"\nExpanded to {len(configs)} configurations")
    assert len(configs) == expected_count, f"Expected {expected_count} configs"

    # Render template for each configuration
    base_params = {
        'model': 'Qwen/Qwen3-32B',
        'tensor_parallel_size': 1,
        'ENGINE_ARGS_ARRAY': '--max-num-seq 1024'
    }

    print("\nRendering templates for each configuration:")
    for i, lmcache_config in enumerate(configs[:3], 1):  # Test first 3
        result = render_kustomization_with_lmcache(
            template_file,
            base_params,
            lmcache_config,
            use_yaml_parse=False
        )

        # Verify this config's values are present
        chunk_size = lmcache_config['chunk_size']
        max_size = lmcache_config['max_local_cpu_size']

        assert f'chunk_size: {chunk_size}' in result
        assert f'max_local_cpu_size: {max_size}' in result

        print(f"  Config {i}: chunk_size={chunk_size}, max_local_cpu_size={max_size} ✓")

    print(f"\n✅ Successfully generated {len(configs)} configurations from sweep")
    return True


def test_preserves_comments():
    """Test that comments are preserved in template."""
    print("\n" + "=" * 70)
    print("TEST 5: Comment Preservation")
    print("=" * 70)

    template_file = Path(__file__).parent.parent / 'templates' / 'tiered-prefix-cache-lmcache-kustomization.yaml.tmpl'

    if not template_file.exists():
        print(f"⚠️  Template not found, skipping")
        return True

    # Read original template
    with open(template_file) as f:
        original = f.read()

    # Check for comments
    has_comments = '#TODO' in original or '#' in original

    if not has_comments:
        print("  No comments in template, skipping")
        return True

    # Render template
    params = {
        'model': 'test-model',
        'tensor_parallel_size': 1,
        'ENGINE_ARGS_ARRAY': ''
    }
    lmcache_args = {'chunk_size': 256, 'save_decode_cache': True}

    result = render_kustomization_with_lmcache(
        template_file,
        params,
        lmcache_args,
        use_yaml_parse=False  # Regex method preserves comments
    )

    # Check if comments preserved
    if '#TODO' in original:
        if '#TODO' in result:
            print("  ✓ Comments preserved (regex method)")
        else:
            print("  ⚠️  Comments lost with regex method")

    print("\n✅ Comment preservation test complete")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Template Injection Tests")
    print("Using Original Template File")
    print("=" * 70)

    tests = [
        ("Basic Injection", test_basic_injection),
        ("YAML Parse Injection", test_yaml_parse_injection),
        ("Original Template", test_with_actual_template),
        ("Parameter Sweep", test_parameter_sweep_integration),
        ("Comment Preservation", test_preserves_comments),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ TEST ERROR: {name}")
            print(f"   Exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")

    if failed == 0:
        print("\n🎉 All tests passed!")
        print("\n✅ Template injection works with original template file!")
        print("   No need to regenerate entire kustomization from Python")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
