#!/usr/bin/env python3
"""
Test script to verify all visual features are working correctly.

This creates a simple test plot to demonstrate:
1. Markers on data points
2. Log/linear scale support
3. Plot bounding box (all 4 sides)
4. Legend bounding box

Run from benchmarks/scripts directory:
    python3 plots/test_visual_features.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_test_plot():
    """Create a test plot with all visual features."""
    try:
        from plots.plot_framework import PlotSpec, PlotGenerator
        import pandas as pd
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install pandas seaborn matplotlib")
        return False

    print("Creating test plot with all visual features...")

    # Create simple test data
    test_data = pd.DataFrame({
        'x': [1, 10, 100, 1000, 10000],
        'y1': [50, 150, 450, 1350, 4050],
        'y2': [40, 100, 300, 900, 2700],
        'y3': [30, 80, 240, 720, 2160],
        'series': ['A', 'A', 'A', 'A', 'A'],
    })

    # Duplicate for different series
    df1 = test_data.copy()
    df1['series'] = 'Series A'
    df1['y'] = df1['y1']

    df2 = test_data.copy()
    df2['series'] = 'Series B'
    df2['y'] = df2['y2']

    df3 = test_data.copy()
    df3['series'] = 'Series C'
    df3['y'] = df3['y3']

    combined_df = pd.concat([df1, df2, df3], ignore_index=True)

    # Create plot spec with all features
    spec = PlotSpec(
        x_axis='x',
        y_axis='y',
        series_by=['series'],
        title='Visual Features Test Plot',
        x_label='X-axis (log scale)',
        y_label='Y-axis (linear scale)',
        x_scale='log',      # ✓ Log scale
        y_scale='linear',   # ✓ Linear scale
        show_legend=True,   # ✓ Legend with bounding box
        show_grid=True,     # ✓ Grid lines
        legend_location='upper left',
        fig_width=10,
        fig_height=6,
        fig_dpi=150,
        output_name='visual_features_test',
    )

    # Generate plot
    output_dir = Path('test_visual_output')
    generator = PlotGenerator(combined_df)
    output_files = generator.plot(spec, output_dir, backend='seaborn')

    print(f"\n✓ Test plot generated successfully!")
    print(f"  Output: {output_files[0]}")
    print(f"\nVisual features to verify:")
    print(f"  ✓ Markers: Look for circular markers at each data point")
    print(f"  ✓ Log scale: X-axis should show 1, 10, 100, 1000, 10000 (powers of 10)")
    print(f"  ✓ Plot box: Should see lines on all 4 sides (top, right, bottom, left)")
    print(f"  ✓ Legend box: Legend should have black frame with shadow")
    print(f"  ✓ Grid: Faint grid lines should be visible")

    return True


def main():
    print("=" * 70)
    print("VISUAL FEATURES TEST")
    print("=" * 70)
    print()

    success = create_test_plot()

    if success:
        print("\n" + "=" * 70)
        print("✅ Test completed!")
        print("=" * 70)
        print("\nOpen the generated plot to verify visual features.")
    else:
        print("\n" + "=" * 70)
        print("⚠ Test skipped due to missing dependencies")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
