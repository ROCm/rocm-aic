#!/usr/bin/env python3
"""
Test that the standalone plotting framework works without vllm dependencies.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test imports
print("Testing imports...")
try:
    from plots.utils import sanitize_filename, full_groupby
    print("✓ utils imported successfully")
except ImportError as e:
    print(f"✗ Failed to import utils: {e}")
    sys.exit(1)

try:
    from plots.plot_filters import PlotFilters, PlotBinners
    print("✓ plot_filters imported successfully")
except ImportError as e:
    print(f"✗ Failed to import plot_filters: {e}")
    sys.exit(1)

try:
    from plots.extract import DataExtractor
    print("✓ extract imported successfully")
except ImportError as e:
    print(f"✗ Failed to import extract: {e}")
    sys.exit(1)

try:
    from plots.prepare import PlotDataPrep
    print("✓ prepare imported successfully")
except ImportError as e:
    print(f"✗ Failed to import prepare: {e}")
    sys.exit(1)

try:
    from plots.plot_framework import PlotSpec, PlotGenerator
    print("✓ plot_framework imported successfully")
except ImportError as e:
    print(f"✗ Failed to import plot_framework: {e}")
    sys.exit(1)

try:
    from plots.schema_discovery import SchemaDiscovery
    print("✓ schema_discovery imported successfully")
except ImportError as e:
    print(f"✗ Failed to import schema_discovery: {e}")
    sys.exit(1)

try:
    from plots.plot_config import PlotConfig
    print("✓ plot_config imported successfully")
except ImportError as e:
    print(f"✗ Failed to import plot_config: {e}")
    sys.exit(1)

# Test utility functions
print("\nTesting utility functions...")
filename = sanitize_filename("test/path..file")
assert filename == "test_path__file", f"sanitize_filename failed: {filename}"
print("✓ sanitize_filename works")

items = [("a", 1), ("b", 2), ("a", 3)]
groups = full_groupby(items, key=lambda x: x[0])
assert len(groups) == 2, f"full_groupby failed: {len(groups)} groups"
assert groups[0][0] == "a", "full_groupby grouping failed"
assert len(groups[0][1]) == 2, "full_groupby items failed"
print("✓ full_groupby works")

print("\n" + "=" * 60)
print("✅ All standalone tests passed!")
print("=" * 60)
print("\nThe plotting framework is working without vllm dependencies.")
print("\nNext steps:")
print("  1. Test with actual data:")
print("     python -m plots.schema_discovery <path>/aggregated_results.json")
print("  2. See README_PLOTTING.md for full documentation")
print("  3. Try plot_config_example.yaml for complete workflow")
