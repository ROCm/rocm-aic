#!/usr/bin/env python3
"""
Test script for LMCache configuration support.

Run this to verify the lmcache_support module is working correctly.
"""

import sys
from pathlib import Path
import yaml

# Import the lmcache support module
from lmcache_support import (
    expand_lmcache_args,
    generate_lmcache_config_yaml,
    generate_configmap_section,
    generate_env_patches,
    generate_volume_patches,
    render_lmcache_template,
    validate_lmcache_args
)


def test_expand_simple():
    """Test expansion with fixed parameters."""
    print("\n" + "=" * 70)
    print("TEST 1: Expand Simple Configuration (No Sweep)")
    print("=" * 70)

    args = {
        'chunk_size': 256,
        'save_decode_cache': True,
        'local_cpu': True,
        'max_local_cpu_size': 100.0,
        'local_disk': None,
        'remote_url': None
    }

    configs = expand_lmcache_args(args)
    print(f"\nInput args:")
    print(yaml.dump(args, default_flow_style=False))

    print(f"Number of configurations generated: {len(configs)}")
    assert len(configs) == 1, "Should generate exactly 1 config"

    print(f"\nGenerated configuration:")
    print(yaml.dump(configs[0], default_flow_style=False))

    print("✅ Test passed!")
    return True


def test_expand_sweep():
    """Test expansion with parameter sweeps."""
    print("\n" + "=" * 70)
    print("TEST 2: Expand With Parameter Sweep")
    print("=" * 70)

    args = {
        'chunk_size': {'values': [128, 256, 512]},
        'save_decode_cache': True,
        'local_cpu': True,
        'max_local_cpu_size': {'values': [50.0, 100.0]},
        'local_disk': None,
        'remote_url': None
    }

    configs = expand_lmcache_args(args)

    print(f"\nInput args with sweeps:")
    print(f"  chunk_size: {args['chunk_size']['values']}")
    print(f"  max_local_cpu_size: {args['max_local_cpu_size']['values']}")

    expected_count = 3 * 2  # 3 chunk_sizes × 2 max_local_cpu_sizes
    print(f"\nNumber of configurations generated: {len(configs)}")
    print(f"Expected: {expected_count}")
    assert len(configs) == expected_count, f"Should generate {expected_count} configs"

    print(f"\nFirst 3 configurations:")
    for i, config in enumerate(configs[:3], 1):
        print(f"\n  Config {i}:")
        print(f"    chunk_size: {config['chunk_size']}")
        print(f"    max_local_cpu_size: {config['max_local_cpu_size']}")

    print("\n✅ Test passed!")
    return True


def test_yaml_generation():
    """Test YAML config file generation."""
    print("\n" + "=" * 70)
    print("TEST 3: Generate LMCache YAML Configuration")
    print("=" * 70)

    args = {
        'chunk_size': 256,
        'save_decode_cache': True,
        'local_cpu': True,
        'max_local_cpu_size': 100.0,
        'local_disk': None,
        'remote_url': None
    }

    yaml_config = generate_lmcache_config_yaml(args)

    print("\nGenerated lmcache_config.yaml:")
    print("-" * 70)
    print(yaml_config)
    print("-" * 70)

    # Verify it's valid YAML
    parsed = yaml.safe_load(yaml_config)
    assert parsed['chunk_size'] == 256
    assert parsed['save_decode_cache'] is True
    assert parsed['max_local_cpu_size'] == 100.0

    print("\n✅ Test passed!")
    return True


def test_configmap_generation():
    """Test ConfigMap section generation."""
    print("\n" + "=" * 70)
    print("TEST 4: Generate ConfigMap Section")
    print("=" * 70)

    args = {
        'chunk_size': 256,
        'save_decode_cache': True,
        'max_local_cpu_size': 100.0
    }

    yaml_config = generate_lmcache_config_yaml(args)
    configmap = generate_configmap_section(yaml_config)

    print("\nGenerated ConfigMap section:")
    print("-" * 70)
    print(configmap)
    print("-" * 70)

    # Verify structure
    assert 'configMapGenerator:' in configmap
    assert 'name: config-map' in configmap
    assert 'lmcache_config.yaml=' in configmap
    assert 'chunk_size: 256' in configmap

    print("\n✅ Test passed!")
    return True


def test_patches_generation():
    """Test environment and volume patches generation."""
    print("\n" + "=" * 70)
    print("TEST 5: Generate Environment and Volume Patches")
    print("=" * 70)

    env_patches = generate_env_patches()
    print("\nEnvironment patches:")
    print("-" * 70)
    print(env_patches)
    print("-" * 70)

    assert 'LMCACHE_CONFIG_FILE' in env_patches
    assert '/opt/configs/lmcache_config.yaml' in env_patches

    volume_patches = generate_volume_patches()
    print("\nVolume patches:")
    print("-" * 70)
    print(volume_patches)
    print("-" * 70)

    assert 'config-dir-volume' in volume_patches
    assert 'configMap:' in volume_patches
    assert '/opt/configs' in volume_patches

    print("\n✅ Test passed!")
    return True


def test_template_rendering():
    """Test complete template rendering."""
    print("\n" + "=" * 70)
    print("TEST 6: Template Rendering with LMCache Config")
    print("=" * 70)

    template = """apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

model: {{model}}

{{CONFIGMAP_SECTION}}

patches:
  - container_args: {{VLLM_ARGS}}
{{ENV_PATCHES}}
{{VOLUME_PATCHES}}
"""

    params = {
        'model': 'Qwen/Qwen3-32B',
        'VLLM_ARGS': '--max-num-seq 1024'
    }

    lmcache_args = {
        'chunk_size': 256,
        'save_decode_cache': True,
        'max_local_cpu_size': 100.0
    }

    rendered = render_lmcache_template(template, params, lmcache_args)

    print("\nRendered template (first 500 chars):")
    print("-" * 70)
    print(rendered[:500] + "...")
    print("-" * 70)

    # Verify placeholders are replaced
    assert '{{model}}' not in rendered
    assert '{{CONFIGMAP_SECTION}}' not in rendered
    assert 'Qwen/Qwen3-32B' in rendered
    assert 'configMapGenerator:' in rendered
    assert 'chunk_size: 256' in rendered
    assert 'LMCACHE_CONFIG_FILE' in rendered

    print("\n✅ Test passed!")
    return True


def test_validation():
    """Test configuration validation."""
    print("\n" + "=" * 70)
    print("TEST 7: Configuration Validation")
    print("=" * 70)

    # Valid config
    valid_args = {
        'chunk_size': 256,
        'save_decode_cache': True,
        'max_local_cpu_size': 100.0
    }

    print("\nValidating valid configuration...")
    errors = validate_lmcache_args(valid_args)
    print(f"Errors found: {len(errors)}")
    assert len(errors) == 0, "Valid config should have no errors"
    print("✅ Valid config passed!")

    # Invalid config
    invalid_args = {
        'chunk_size': -1,  # Invalid: negative
        'max_local_cpu_size': 'invalid',  # Invalid: string instead of number
        'save_decode_cache': 'yes'  # Invalid: string instead of bool
    }

    print("\nValidating invalid configuration...")
    errors = validate_lmcache_args(invalid_args)
    print(f"Errors found: {len(errors)}")
    if errors:
        for error in errors:
            print(f"  - {error}")

    assert len(errors) > 0, "Invalid config should have errors"
    print("✅ Invalid config correctly rejected!")

    print("\n✅ Test passed!")
    return True


def test_full_example():
    """Test complete workflow from config file to rendered manifests."""
    print("\n" + "=" * 70)
    print("TEST 8: Full Example Workflow")
    print("=" * 70)

    # Load example sweep config
    config_file = Path(__file__).parent.parent / "sweep-configs" / "example-lmcache-sweep.yaml"

    if not config_file.exists():
        print(f"\n⚠️  Skipping - config file not found: {config_file}")
        return True

    print(f"\nLoading sweep config: {config_file.name}")
    with open(config_file) as f:
        sweep_config = yaml.safe_load(f)

    # Extract lmcache_args
    lmcache_spec = sweep_config['parameters'].get('lmcache_args')
    if not lmcache_spec:
        print("⚠️  No lmcache_args in config")
        return True

    print(f"\nLMCache args specification:")
    print(yaml.dump(lmcache_spec, default_flow_style=False))

    # Expand configurations
    lmcache_args = lmcache_spec['args']
    configs = expand_lmcache_args(lmcache_args)

    print(f"\nExpanded to {len(configs)} configurations")

    # Generate YAML for first config
    print(f"\nFirst configuration YAML:")
    print("-" * 70)
    yaml_config = generate_lmcache_config_yaml(configs[0])
    print(yaml_config)
    print("-" * 70)

    # Generate full ConfigMap
    configmap = generate_configmap_section(yaml_config)
    print(f"\nConfigMap section:")
    print("-" * 70)
    print(configmap)
    print("-" * 70)

    print("\n✅ Test passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("LMCache Support Module Tests")
    print("=" * 70)

    tests = [
        ("Simple Expansion", test_expand_simple),
        ("Sweep Expansion", test_expand_sweep),
        ("YAML Generation", test_yaml_generation),
        ("ConfigMap Generation", test_configmap_generation),
        ("Patches Generation", test_patches_generation),
        ("Template Rendering", test_template_rendering),
        ("Validation", test_validation),
        ("Full Example", test_full_example),
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
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
