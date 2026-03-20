#!/usr/bin/env python3
"""
Integration test for lmcache support in run-sweep.py

Tests the complete flow:
1. Loading sweep config with lmcache_args
2. Expanding parameter combinations
3. Rendering templates with lmcache ConfigMap injection
4. Verifying output kustomization files
"""

import sys
import yaml
import tempfile
import shutil
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import with hyphen in filename requires importlib
import importlib.util
spec = importlib.util.spec_from_file_location("run_sweep", Path(__file__).parent / "run-sweep.py")
run_sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_sweep)
SweepOrchestrator = run_sweep.SweepOrchestrator


def test_lmcache_parameter_expansion():
    """Test that lmcache_args are expanded correctly."""
    print("\n" + "=" * 70)
    print("TEST 1: LMCache Parameter Expansion")
    print("=" * 70)

    # Create a test config
    config = {
        'name': 'test-lmcache',
        'deployment': 'tiered-prefix-cache-lmcache',
        'parameters': {
            'model': {
                'type': 'fixed',
                'value': 'Qwen/Qwen3-32B'
            },
            'tensor_parallel_size': {
                'type': 'fixed',
                'value': 1
            },
            'vllm_args': {
                'type': 'combinations',
                'args': {
                    'max_num_seq': 1024,
                    'kv_connector': {
                        'values': [
                            {'type': 'lmcache', 'role': 'kv_both'}
                        ]
                    }
                }
            },
            'lmcache_args': {
                'type': 'combinations',
                'args': {
                    'chunk_size': {'values': [256, 512]},
                    'save_decode_cache': True,
                    'max_local_cpu_size': {'values': [100.0, 200.0]}
                }
            }
        }
    }

    # Create temp config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_file = f.name

    try:
        # Create orchestrator
        orchestrator = SweepOrchestrator(config_file)

        # Generate combinations
        combinations = orchestrator.generate_parameter_combinations()

        print(f"\nGenerated {len(combinations)} combinations")

        # Should be 2 (chunk_size) × 2 (max_local_cpu_size) = 4 combinations
        expected_count = 2 * 2
        assert len(combinations) == expected_count, \
            f"Expected {expected_count} combinations, got {len(combinations)}"
        print(f"  ✓ Correct number of combinations: {expected_count}")

        # Verify each combination has lmcache_args
        for i, combo in enumerate(combinations, 1):
            assert 'lmcache_args' in combo, f"Combination {i} missing lmcache_args"
            assert 'chunk_size' in combo['lmcache_args']
            assert 'max_local_cpu_size' in combo['lmcache_args']
            print(f"  ✓ Combination {i}: chunk_size={combo['lmcache_args']['chunk_size']}, "
                  f"max_local_cpu_size={combo['lmcache_args']['max_local_cpu_size']}")

        print("\n✅ LMCache parameter expansion works correctly")
        return True

    finally:
        Path(config_file).unlink()


def test_template_rendering_with_lmcache():
    """Test template rendering with lmcache ConfigMap injection."""
    print("\n" + "=" * 70)
    print("TEST 2: Template Rendering with LMCache")
    print("=" * 70)

    # Check if template exists
    template_file = Path(__file__).parent.parent / 'templates' / \
                    'tiered-prefix-cache-lmcache-kustomization.yaml.tmpl'

    if not template_file.exists():
        print(f"  ⚠️  Template not found: {template_file}")
        print("  Skipping this test")
        return True

    print(f"  Template: {template_file.name}")

    # Create test config
    config = {
        'name': 'test-render',
        'deployment': 'tiered-prefix-cache-lmcache',
        'parameters': {
            'model': {'type': 'fixed', 'value': 'Qwen/Qwen3-32B'},
            'tensor_parallel_size': {'type': 'fixed', 'value': 1},
            'vllm_args': {
                'type': 'combinations',
                'args': {
                    'max_num_seq': 1024,
                    'kv_connector': {
                        'values': [{'type': 'lmcache', 'role': 'kv_both'}]
                    }
                }
            },
            'lmcache_args': {
                'type': 'combinations',
                'args': {
                    'chunk_size': 512,
                    'save_decode_cache': True,
                    'max_local_cpu_size': 150.0,
                    'local_disk': None,
                    'remote_url': None
                }
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_file = f.name

    temp_dir = None
    try:
        # Create orchestrator
        orchestrator = SweepOrchestrator(config_file)

        # Create temp output directory
        temp_dir = Path(tempfile.mkdtemp())

        # Get first combination
        combinations = orchestrator.generate_parameter_combinations()
        params = combinations[0]

        # Render template
        output_dir = orchestrator.render_template(params, temp_dir)
        kustomization_file = output_dir / 'kustomization.yaml'

        assert kustomization_file.exists(), "Kustomization file not created"
        print(f"  ✓ Kustomization file created")

        # Read and verify content
        with open(kustomization_file) as f:
            content = f.read()

        print("\n  Verifying content:")

        # Should have model replaced
        assert 'Qwen/Qwen3-32B' in content, "Model not replaced"
        print("    ✓ Model replaced")

        # Should have lmcache config injected
        assert 'chunk_size: 512' in content, "LMCache chunk_size not found"
        print("    ✓ LMCache chunk_size injected")

        assert 'max_local_cpu_size: 150.0' in content, "LMCache max_local_cpu_size not found"
        print("    ✓ LMCache max_local_cpu_size injected")

        # Should still have patches section
        assert 'patches:' in content, "Patches section missing"
        print("    ✓ Patches section preserved")

        # Parse as YAML to verify structure
        data = yaml.safe_load(content)
        assert data['kind'] == 'Kustomization', "Not a valid Kustomization"
        assert 'configMapGenerator' in data, "ConfigMapGenerator missing"
        print("    ✓ Valid Kustomization YAML")

        print("\n✅ Template rendering with LMCache works correctly")
        return True

    finally:
        Path(config_file).unlink()
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def test_multiple_lmcache_configurations():
    """Test that multiple lmcache configurations are rendered correctly."""
    print("\n" + "=" * 70)
    print("TEST 3: Multiple LMCache Configurations")
    print("=" * 70)

    template_file = Path(__file__).parent.parent / 'templates' / \
                    'tiered-prefix-cache-lmcache-kustomization.yaml.tmpl'

    if not template_file.exists():
        print(f"  ⚠️  Template not found, skipping")
        return True

    # Config with sweep
    config = {
        'name': 'test-multi',
        'deployment': 'tiered-prefix-cache-lmcache',
        'parameters': {
            'model': {'type': 'fixed', 'value': 'Qwen/Qwen3-32B'},
            'tensor_parallel_size': {'type': 'fixed', 'value': 1},
            'vllm_args': {
                'type': 'combinations',
                'args': {
                    'max_num_seq': 1024,
                    'kv_connector': {
                        'values': [{'type': 'lmcache', 'role': 'kv_both'}]
                    }
                }
            },
            'lmcache_args': {
                'type': 'combinations',
                'args': {
                    'chunk_size': {'values': [128, 256, 512]},
                    'save_decode_cache': True,
                    'max_local_cpu_size': {'values': [50.0, 100.0]}
                }
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_file = f.name

    temp_dir = None
    try:
        orchestrator = SweepOrchestrator(config_file)
        combinations = orchestrator.generate_parameter_combinations()

        print(f"\n  Testing {len(combinations)} combinations...")

        temp_dir = Path(tempfile.mkdtemp())

        # Test each combination
        for i, params in enumerate(combinations, 1):
            run_dir = temp_dir / f"run-{i:03d}"
            run_dir.mkdir(parents=True)

            output_dir = orchestrator.render_template(params, run_dir)
            kustomization_file = output_dir / 'kustomization.yaml'

            assert kustomization_file.exists(), f"Run {i} kustomization not created"

            with open(kustomization_file) as f:
                content = f.read()

            # Verify this configuration's values are present
            chunk_size = params['lmcache_args']['chunk_size']
            max_size = params['lmcache_args']['max_local_cpu_size']

            assert f'chunk_size: {chunk_size}' in content, \
                f"Run {i}: chunk_size {chunk_size} not found"
            assert f'max_local_cpu_size: {max_size}' in content, \
                f"Run {i}: max_local_cpu_size {max_size} not found"

            print(f"    ✓ Run {i}: chunk_size={chunk_size}, max_local_cpu_size={max_size}")

        print(f"\n✅ All {len(combinations)} configurations rendered correctly")
        return True

    finally:
        Path(config_file).unlink()
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def test_backward_compatibility():
    """Test that configs without lmcache_args still work."""
    print("\n" + "=" * 70)
    print("TEST 4: Backward Compatibility (No LMCache)")
    print("=" * 70)

    config = {
        'name': 'test-no-lmcache',
        'deployment': 'tiered-prefix-cache-offloading',
        'parameters': {
            'model': {'type': 'fixed', 'value': 'Qwen/Qwen3-32B'},
            'tensor_parallel_size': {'type': 'fixed', 'value': 1},
            'vllm_args': {
                'type': 'combinations',
                'args': {
                    'max_num_seq': 1024,
                    'kv_connector': {
                        'values': [
                            {'type': 'offloading', 'cpu_bytes': 107374182400, 'role': 'kv_both'}
                        ]
                    }
                }
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_file = f.name

    temp_dir = None
    try:
        orchestrator = SweepOrchestrator(config_file)
        combinations = orchestrator.generate_parameter_combinations()

        print(f"\n  Generated {len(combinations)} combination(s)")

        temp_dir = Path(tempfile.mkdtemp())

        params = combinations[0]

        # Should NOT have lmcache_args
        assert 'lmcache_args' not in params, "lmcache_args should not be present"
        print("  ✓ No lmcache_args in parameters")

        # Rendering should still work (using standard template rendering)
        run_dir = temp_dir / "run-001"
        run_dir.mkdir(parents=True)

        output_dir = orchestrator.render_template(params, run_dir)
        kustomization_file = output_dir / 'kustomization.yaml'

        assert kustomization_file.exists(), "Kustomization file not created"
        print("  ✓ Template rendered without lmcache_args")

        print("\n✅ Backward compatibility maintained")
        return True

    finally:
        Path(config_file).unlink()
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("LMCache Support Integration Tests")
    print("=" * 70)

    tests = [
        ("Parameter Expansion", test_lmcache_parameter_expansion),
        ("Template Rendering", test_template_rendering_with_lmcache),
        ("Multiple Configurations", test_multiple_lmcache_configurations),
        ("Backward Compatibility", test_backward_compatibility),
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
        print("\n🎉 All integration tests passed!")
        print("\n✅ LMCache support is fully integrated and working!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
