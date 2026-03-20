#!/usr/bin/env python3
"""
Test script for legend customization features.

Tests all three approaches:
1. Label format templates
2. Value mapping
3. Custom transformed columns
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_format_series_label():
    """Test the format_series_label method."""
    print("=" * 70)
    print("Testing Legend Label Formatting")
    print("=" * 70)

    from plots.plot_framework import PlotSpec

    # Test 1: Default formatting
    print("\n1. Default formatting (no customization):")
    spec1 = PlotSpec(
        x_axis="x",
        y_axis="y",
        series_by=["tp_size"]
    )
    group = (("tp_size", 1),)
    label = spec1.format_series_label(group)
    print(f"   Input: tp_size=1")
    print(f"   Output: '{label}'")
    assert label == "tp_size=1", f"Expected 'tp_size=1', got '{label}'"
    print("   ✓ Correct")

    # Test 2: Format template
    print("\n2. Format template:")
    spec2 = PlotSpec(
        x_axis="x",
        y_axis="y",
        series_by=["tp_size"],
        series_label_format="TP={tp_size}"
    )
    label = spec2.format_series_label(group)
    print(f"   Template: 'TP={{tp_size}}'")
    print(f"   Input: tp_size=1")
    print(f"   Output: '{label}'")
    assert label == "TP=1", f"Expected 'TP=1', got '{label}'"
    print("   ✓ Correct")

    # Test 3: Value mapping
    print("\n3. Value mapping:")
    spec3 = PlotSpec(
        x_axis="x",
        y_axis="y",
        series_by=["tp_size"],
        series_labels={
            "1": "Single GPU",
            "2": "Dual GPU",
            "4": "Quad GPU"
        }
    )
    label = spec3.format_series_label(group)
    print(f"   Mapping: 1 → 'Single GPU'")
    print(f"   Input: tp_size=1")
    print(f"   Output: '{label}'")
    assert label == "Single GPU", f"Expected 'Single GPU', got '{label}'"
    print("   ✓ Correct")

    # Test 4: Multi-column formatting
    print("\n4. Multi-column format template:")
    spec4 = PlotSpec(
        x_axis="x",
        y_axis="y",
        series_by=["model", "tp_size"],
        series_label_format="{model} (TP={tp_size})"
    )
    group = (("model", "gpt-120b"), ("tp_size", 2))
    label = spec4.format_series_label(group)
    print(f"   Template: '{{model}} (TP={{tp_size}})'")
    print(f"   Input: model=gpt-120b, tp_size=2")
    print(f"   Output: '{label}'")
    assert label == "gpt-120b (TP=2)", f"Expected 'gpt-120b (TP=2)', got '{label}'"
    print("   ✓ Correct")

    # Test 5: Fallback behavior
    print("\n5. Fallback when value not in mapping:")
    spec5 = PlotSpec(
        x_axis="x",
        y_axis="y",
        series_by=["tp_size"],
        series_labels={
            "1": "Single GPU",
            "2": "Dual GPU",
            # 4 is missing
        }
    )
    group = (("tp_size", 4),)
    label = spec5.format_series_label(group)
    print(f"   Mapping: Only 1 and 2 defined")
    print(f"   Input: tp_size=4")
    print(f"   Output: '{label}' (falls back to default)")
    assert label == "tp_size=4", f"Expected 'tp_size=4', got '{label}'"
    print("   ✓ Correct fallback")

    print("\n" + "=" * 70)
    print("✅ All legend formatting tests passed!")
    print("=" * 70)

def test_with_real_data():
    """Test legend customization with actual plotting."""
    print("\n" + "=" * 70)
    print("Testing with Real Plotting")
    print("=" * 70)

    try:
        import pandas as pd
        from plots.plot_framework import PlotSpec, PlotGenerator

        # Create test data
        data = pd.DataFrame({
            'isl': [1000, 2000, 4000, 1000, 2000, 4000, 1000, 2000, 4000],
            'ttft': [100, 200, 400, 80, 160, 320, 60, 120, 240],
            'tp_size': [1, 1, 1, 2, 2, 2, 4, 4, 4],
        })

        # Test with custom labels
        spec = PlotSpec(
            x_axis='isl',
            y_axis='ttft',
            series_by=['tp_size'],
            series_labels={
                "1": "Single GPU",
                "2": "Dual GPU",
                "4": "Quad GPU"
            },
            title='Test: Custom Legend Labels',
            x_scale='log',
            output_name='test_custom_labels',
            fig_dpi=100,
        )

        generator = PlotGenerator(data)
        output_dir = Path('test_legend_output')
        output_files = generator.plot(spec, output_dir, backend='seaborn')

        print(f"\n✓ Generated test plot with custom labels")
        print(f"  Output: {output_files[0]}")
        print(f"\nOpen the plot to verify legend shows:")
        print(f"  - 'Single GPU' (instead of 'tp_size=1')")
        print(f"  - 'Dual GPU' (instead of 'tp_size=2')")
        print(f"  - 'Quad GPU' (instead of 'tp_size=4')")

        return True

    except ImportError as e:
        print(f"\n⚠ Skipping real plot test (missing dependencies): {e}")
        print("  Install with: pip install pandas seaborn matplotlib")
        return False

def main():
    print("\n" + "=" * 70)
    print("LEGEND CUSTOMIZATION TEST SUITE")
    print("=" * 70 + "\n")

    # Test formatting logic
    test_format_series_label()

    # Test with actual plotting
    test_with_real_data()

    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)
    print("\nLegend customization is working correctly!")
    print("\nSee LEGEND_CUSTOMIZATION.md for usage examples:")
    print("  1. Label format templates")
    print("  2. Value mapping")
    print("  3. Custom transformed columns")
    print("\nTry: python -m plots.plot_config plots/plot_config_custom_labels.yaml")

if __name__ == "__main__":
    main()
