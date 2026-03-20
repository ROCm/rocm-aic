#!/usr/bin/env python3
"""
Test that ConfigMap injection ONLY modifies lmcache_config.yaml literal
and does not touch any other part of the configMapGenerator or template.
"""

import sys
import yaml
from lmcache_template_injection import (
    inject_lmcache_configmap,
    inject_lmcache_configmap_yaml_parse
)


def test_only_lmcache_literal_replaced():
    """Test that ONLY lmcache_config.yaml literal is replaced."""
    print("\n" + "=" * 70)
    print("TEST 1: Only LMCache Literal Replaced")
    print("=" * 70)

    # Template with MULTIPLE literals in configMapGenerator
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
        max_local_cpu_size: 100.0
      - |
        other_config.yaml=important_setting: do_not_change
        another_setting: keep_this_too
      - simple_key=simple_value

patches:
  - target:
      kind: Deployment
"""

    params = {}
    new_lmcache_config = "chunk_size: 1024\nsave_decode_cache: false"

    # Test with regex method
    print("\nTesting regex method...")
    result = inject_lmcache_configmap(template, params, new_lmcache_config)

    print("\nVerifying changes:")

    # LMCache config should be updated
    assert 'chunk_size: 1024' in result, "LMCache config not updated"
    assert 'save_decode_cache: false' in result, "LMCache config not updated"
    print("  ✓ LMCache config updated")

    # Old LMCache values should be gone
    assert 'chunk_size: 256' not in result, "Old LMCache value still present"
    assert 'save_decode_cache: True' not in result, "Old LMCache value still present"
    print("  ✓ Old LMCache values removed")

    # Other literals should be UNCHANGED
    assert 'other_config.yaml=important_setting: do_not_change' in result, \
        "Other config literal was modified!"
    assert 'another_setting: keep_this_too' in result, \
        "Other config literal was modified!"
    assert 'simple_key=simple_value' in result, \
        "Simple literal was modified!"
    print("  ✓ Other literals preserved")

    # Structure should be preserved
    assert 'resources:' in result
    assert 'patches:' in result
    print("  ✓ Template structure preserved")

    print("\n✅ Regex method: Only LMCache literal modified")

    # Test with YAML parse method
    print("\nTesting YAML parse method...")
    result_yaml = inject_lmcache_configmap_yaml_parse(template, params, new_lmcache_config)

    # Parse to verify structure
    data = yaml.safe_load(result_yaml)
    literals = data['configMapGenerator'][0]['literals']

    print(f"\nNumber of literals: {len(literals)}")
    assert len(literals) == 3, f"Expected 3 literals, got {len(literals)}"
    print("  ✓ Literal count preserved")

    # Find each literal
    lmcache_literal = None
    other_literal = None
    simple_literal = None

    for lit in literals:
        if lit.startswith('lmcache_config.yaml='):
            lmcache_literal = lit
        elif lit.startswith('other_config.yaml='):
            other_literal = lit
        elif lit.startswith('simple_key='):
            simple_literal = lit

    # Verify lmcache was updated
    assert lmcache_literal is not None, "LMCache literal missing"
    assert 'chunk_size: 1024' in lmcache_literal, "LMCache not updated"
    print("  ✓ LMCache literal updated")

    # Verify others unchanged
    assert other_literal is not None, "Other config literal missing"
    assert 'do_not_change' in other_literal, "Other config was modified!"
    assert 'keep_this_too' in other_literal, "Other config was modified!"
    print("  ✓ Other config literal unchanged")

    assert simple_literal == 'simple_key=simple_value', "Simple literal was modified!"
    print("  ✓ Simple literal unchanged")

    print("\n✅ YAML parse method: Only LMCache literal modified")
    return True


def test_multiple_configmaps():
    """Test with multiple configMapGenerators - ensure we only touch the right one."""
    print("\n" + "=" * 70)
    print("TEST 2: Multiple ConfigMaps")
    print("=" * 70)

    template = """apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

configMapGenerator:
  - name: config-map
    literals:
      - |
        lmcache_config.yaml=chunk_size: 256
  - name: other-config-map
    literals:
      - |
        lmcache_config.yaml=do_not_touch_this: true

patches:
  - target:
      kind: Deployment
"""

    params = {}
    new_config = "chunk_size: 512"

    # YAML parse method (more precise for this case)
    result = inject_lmcache_configmap_yaml_parse(template, params, new_config)

    data = yaml.safe_load(result)

    print("\nVerifying only first ConfigMap modified:")

    # First configMap should be updated
    first_literals = data['configMapGenerator'][0]['literals']
    assert 'chunk_size: 512' in first_literals[0], "First ConfigMap not updated"
    print("  ✓ First ConfigMap (config-map) updated")

    # Second configMap should be unchanged
    second_literals = data['configMapGenerator'][1]['literals']
    assert 'do_not_touch_this: true' in second_literals[0], \
        "Second ConfigMap was modified!"
    print("  ✓ Second ConfigMap (other-config-map) unchanged")

    print("\n✅ Only the correct ConfigMap was modified")
    return True


def test_no_false_matches():
    """Test that we don't match similar strings elsewhere."""
    print("\n" + "=" * 70)
    print("TEST 3: No False Matches")
    print("=" * 70)

    template = """apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

# Comment mentioning lmcache_config.yaml should not be touched

configMapGenerator:
  - name: config-map
    literals:
      - |
        lmcache_config.yaml=chunk_size: 256

patches:
  - target:
      kind: Deployment
    patch: |-
      # This patch references lmcache_config.yaml=/opt/configs/lmcache_config.yaml
      - op: add
        path: /spec/template/spec/containers/0/env/-
        value:
          name: LMCACHE_CONFIG_FILE
          value: "/opt/configs/lmcache_config.yaml"
"""

    params = {}
    new_config = "chunk_size: 1024"

    result = inject_lmcache_configmap(template, params, new_config)

    print("\nVerifying no false matches:")

    # Comment should be preserved
    assert '# Comment mentioning lmcache_config.yaml should not be touched' in result, \
        "Comment was modified!"
    print("  ✓ Comment preserved")

    # Patch should be preserved
    assert 'LMCACHE_CONFIG_FILE' in result, "Patch was modified!"
    assert '"/opt/configs/lmcache_config.yaml"' in result, "Patch was modified!"
    print("  ✓ Patch preserved")

    # Only the literal should be updated
    assert 'chunk_size: 1024' in result, "Literal not updated"
    assert 'chunk_size: 256' not in result, "Old literal still present"
    print("  ✓ Only literal updated")

    print("\n✅ No false matches")
    return True


def test_literal_with_pipe_operator():
    """Test that multi-line literals with pipe operator are handled correctly."""
    print("\n" + "=" * 70)
    print("TEST 4: Multi-line Literal with Pipe Operator")
    print("=" * 70)

    template = """apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

configMapGenerator:
  - name: config-map
    literals:
      - |
        lmcache_config.yaml=chunk_size: 256
        save_decode_cache: True
        local_cpu: true
        max_local_cpu_size: 100.0
        local_disk: null
        remote_url: null

patches:
  - target:
      kind: Deployment
"""

    params = {}
    new_config = """chunk_size: 2048
save_decode_cache: false
local_cpu: false
max_local_cpu_size: 500.0
local_disk:
  path: /tmp
  max_size: 100GB
remote_url: redis://cache"""

    result = inject_lmcache_configmap(template, params, new_config)

    print("\nVerifying multi-line replacement:")

    # All new values should be present
    assert 'chunk_size: 2048' in result
    assert 'save_decode_cache: false' in result
    assert 'max_local_cpu_size: 500.0' in result
    assert 'redis://cache' in result
    print("  ✓ All new values present")

    # Old values should be gone
    assert 'chunk_size: 256' not in result
    assert 'max_local_cpu_size: 100.0' not in result
    print("  ✓ Old values removed")

    # Structure preserved
    assert 'patches:' in result
    print("  ✓ Structure preserved")

    print("\n✅ Multi-line literal correctly replaced")
    return True


def test_count_parameter():
    """Test that count=1 ensures only first match is replaced."""
    print("\n" + "=" * 70)
    print("TEST 5: Count Parameter (Only First Match)")
    print("=" * 70)

    # Pathological case: lmcache_config.yaml appears multiple times
    # (shouldn't happen in practice, but let's be safe)
    template = """apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

configMapGenerator:
  - name: config-map
    literals:
      - |
        lmcache_config.yaml=chunk_size: 256
  - name: backup-config
    literals:
      - |
        lmcache_config.yaml=chunk_size: 128

patches: []
"""

    params = {}
    new_config = "chunk_size: 512"

    # Regex method with count=1 should only replace first
    result = inject_lmcache_configmap(template, params, new_config)

    print("\nVerifying only first occurrence replaced:")

    # Count occurrences
    count_512 = result.count('chunk_size: 512')
    count_128 = result.count('chunk_size: 128')

    print(f"  chunk_size: 512 appears {count_512} time(s)")
    print(f"  chunk_size: 128 appears {count_128} time(s)")

    assert count_512 == 1, "First occurrence not replaced correctly"
    assert count_128 == 1, "Second occurrence was modified!"

    print("  ✓ Only first occurrence replaced")

    print("\n✅ Count parameter works correctly")
    return True


def main():
    """Run all precision tests."""
    print("\n" + "=" * 70)
    print("Precise ConfigMap Injection Tests")
    print("Ensuring ONLY lmcache_config.yaml is Modified")
    print("=" * 70)

    tests = [
        ("Only LMCache Literal", test_only_lmcache_literal_replaced),
        ("Multiple ConfigMaps", test_multiple_configmaps),
        ("No False Matches", test_no_false_matches),
        ("Multi-line Literal", test_literal_with_pipe_operator),
        ("Count Parameter", test_count_parameter),
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
        print("\n✅ ConfigMap injection is PRECISE!")
        print("   - Only lmcache_config.yaml literal is modified")
        print("   - All other literals are preserved")
        print("   - No false matches in comments or patches")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
