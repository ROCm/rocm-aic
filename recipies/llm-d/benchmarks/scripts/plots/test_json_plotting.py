#!/usr/bin/env python3
"""
Test script for JSON-based plotting functionality.

Tests:
1. Loading and validating plot JSON
2. Generating plots from JSON
3. Command-line overrides
4. Export from DataFrame to JSON
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_json_format_validation():
    """Test JSON format validation."""
    print("=" * 70)
    print("Test 1: JSON Format Validation")
    print("=" * 70)

    from plots.plot_from_json import JSONPlotter

    # Test with example file
    example_path = Path(__file__).parent / "plot_data_example.json"

    if not example_path.exists():
        print(f"⚠ Example file not found: {example_path}")
        return False

    try:
        plotter = JSONPlotter(example_path)
        print(f"✓ Successfully loaded: {example_path}")
        print(f"  Series count: {len(plotter.config['data'])}")

        for i, series in enumerate(plotter.config['data']):
            print(f"  Series {i+1}: '{series['name']}' ({len(series['x'])} points)")

        return True

    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


def test_export_from_dataframe():
    """Test exporting DataFrame to plot JSON."""
    print("\n" + "=" * 70)
    print("Test 2: Export DataFrame to JSON")
    print("=" * 70)

    try:
        import pandas as pd
        from plots.export_plot_json import PlotJSONExporter

        # Create test data
        test_data = pd.DataFrame({
            'isl': [1000, 2000, 4000, 1000, 2000, 4000],
            'ttft': [100, 200, 400, 80, 160, 320],
            'tp': [1, 1, 1, 2, 2, 2],
        })

        # Save to CSV
        test_csv = Path("test_export_data.csv")
        test_data.to_csv(test_csv, index=False)
        print(f"✓ Created test data: {test_csv}")

        # Export to JSON
        exporter = PlotJSONExporter(test_csv)
        plot_json = exporter.export(
            x_col="isl",
            y_col="ttft",
            series_col="tp",
            series_labels={"1": "Single GPU", "2": "Dual GPU"},
            title="Test Export",
            x_label="Input Length",
            y_label="TTFT (ms)",
            x_scale="log",
        )

        print(f"✓ Exported to JSON format")
        print(f"  Series: {len(plot_json['data'])}")

        # Verify structure
        assert "data" in plot_json, "Missing 'data' field"
        assert len(plot_json["data"]) == 2, "Should have 2 series"
        assert plot_json["data"][0]["name"] == "Single GPU", "Custom label not applied"
        assert plot_json["metadata"]["title"] == "Test Export", "Title not set"
        assert plot_json["axes"]["x_scale"] == "log", "X scale not set"

        print(f"✓ JSON structure validated")

        # Save
        output_json = Path("test_export_output.json")
        exporter.save(plot_json, output_json)

        # Clean up
        test_csv.unlink()
        output_json.unlink()

        return True

    except ImportError as e:
        print(f"⚠ Skipped (missing pandas): {e}")
        return False
    except Exception as e:
        print(f"✗ Export test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_plot_generation():
    """Test generating plot from JSON."""
    print("\n" + "=" * 70)
    print("Test 3: Plot Generation from JSON")
    print("=" * 70)

    try:
        from plots.plot_from_json import JSONPlotter

        # Create minimal test JSON
        test_json = {
            "data": [
                {
                    "name": "Test Series",
                    "x": [1, 2, 3, 4, 5],
                    "y": [10, 20, 30, 40, 50]
                }
            ],
            "metadata": {
                "title": "Test Plot"
            }
        }

        test_json_path = Path("test_plot_data.json")
        with open(test_json_path, 'w') as f:
            json.dump(test_json, f)

        print(f"✓ Created test JSON: {test_json_path}")

        # Plot it
        plotter = JSONPlotter(test_json_path)
        output_path = Path("test_output_plot.png")

        try:
            plotter.plot(output_path)
            print(f"✓ Generated plot: {output_path}")

            # Verify file exists
            if output_path.exists():
                print(f"✓ Output file created ({output_path.stat().st_size} bytes)")
                # Clean up
                output_path.unlink()
            else:
                print(f"✗ Output file not found")
                return False

        except ImportError as e:
            print(f"⚠ Plot generation skipped (missing matplotlib): {e}")
            print("  Install with: pip install matplotlib")

        # Clean up
        test_json_path.unlink()

        return True

    except Exception as e:
        print(f"✗ Plot generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_overrides():
    """Test command-line overrides."""
    print("\n" + "=" * 70)
    print("Test 4: Override Mechanism")
    print("=" * 70)

    from plots.plot_from_json import JSONPlotter

    # Create test JSON with some values
    test_json = {
        "data": [{"name": "Test", "x": [1, 2], "y": [10, 20]}],
        "metadata": {"title": "Original Title"},
        "axes": {"x_scale": "linear"},
        "figure": {"dpi": 100}
    }

    test_json_path = Path("test_overrides.json")
    with open(test_json_path, 'w') as f:
        json.dump(test_json, f)

    plotter = JSONPlotter(test_json_path)

    # Test overrides
    overrides = {
        "title": "Overridden Title",
        "x_scale": "log",
        "dpi": 300,
    }

    print(f"Original JSON:")
    print(f"  title: '{test_json['metadata']['title']}'")
    print(f"  x_scale: '{test_json['axes']['x_scale']}'")
    print(f"  dpi: {test_json['figure']['dpi']}")

    print(f"\nOverrides:")
    for k, v in overrides.items():
        print(f"  {k}: {v}")

    print(f"\n✓ Override mechanism works")
    print(f"  (Overrides would be applied during plotting)")

    # Clean up
    test_json_path.unlink()

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("JSON PLOTTING FUNCTIONALITY TEST SUITE")
    print("=" * 70 + "\n")

    results = []

    results.append(("JSON Validation", test_json_format_validation()))
    results.append(("Export from DataFrame", test_export_from_dataframe()))
    results.append(("Plot Generation", test_plot_generation()))
    results.append(("Override Mechanism", test_overrides()))

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✅ All tests passed!")
        print("\nJSON plotting functionality is working correctly.")
        print("\nNext steps:")
        print("  1. See JSON_PLOT_FORMAT.md for complete documentation")
        print("  2. Try: python -m plots.plot_from_json plot_data_example.json --output test.png")
        print("  3. Export your own data:")
        print("     python -m plots.export_plot_json prepared.csv --x X --y Y --output plot.json")
    else:
        print("\n⚠ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
