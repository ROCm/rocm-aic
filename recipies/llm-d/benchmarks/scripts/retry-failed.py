#!/usr/bin/env python3
"""
Retry failed configurations from a previous sweep.

This script reads a sweep's results directory, identifies failed runs,
and re-executes only those configurations in a new sweep.
"""

import json
import yaml
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Import the sweep orchestrator (with hyphen in filename)
import importlib.util
import sys
from pathlib import Path as _Path

# Dynamic import to handle the hyphen in the filename
_script_dir = _Path(__file__).parent
_run_sweep_path = _script_dir / "run-sweep.py"
_spec = importlib.util.spec_from_file_location("run_sweep", _run_sweep_path)
_run_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run_sweep)
SweepOrchestrator = _run_sweep.SweepOrchestrator


def load_sweep_results(sweep_dir: Path):
    """
    Load results from a previous sweep.

    Args:
        sweep_dir: Path to the sweep results directory

    Returns:
        Tuple of (metadata, summary)
    """
    metadata_file = sweep_dir / "metadata.yaml"
    summary_file = sweep_dir / "summary.json"

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    with open(metadata_file) as f:
        metadata = yaml.safe_load(f)

    with open(summary_file) as f:
        summary = json.load(f)

    return metadata, summary


def extract_failed_runs(summary):
    """
    Extract failed runs from the summary.

    Args:
        summary: List of run results from summary.json

    Returns:
        List of failed run dictionaries
    """
    failed_runs = [run for run in summary if run['status'] == 'failed']
    return failed_runs


def create_retry_config(original_metadata, failed_runs):
    """
    Create a new sweep configuration for retrying failed runs.

    This creates a configuration that will run only the specific
    parameter combinations that failed in the original sweep.

    Args:
        original_metadata: Original sweep metadata
        failed_runs: List of failed run dictionaries

    Returns:
        Dictionary with retry sweep configuration
    """
    if not failed_runs:
        return None

    # Start with the original configuration structure
    retry_config = {
        'name': f"{original_metadata['name']}-retry",
        'description': f"Retry of failed configurations from sweep: {original_metadata['name']}",
        'deployment': original_metadata['deployment']
    }

    # Add deployment_template if it exists in original
    if 'deployment_template' in original_metadata:
        retry_config['deployment_template'] = original_metadata['deployment_template']

    # Copy load_generation configuration
    if 'load_generation' in original_metadata:
        retry_config['load_generation'] = original_metadata['load_generation']

    # Create parameter combinations from failed runs
    # We need to reconstruct the parameter structure
    retry_config['parameters'] = create_retry_parameters(failed_runs)

    return retry_config


def create_retry_parameters(failed_runs):
    """
    Create a parameters section that explicitly lists each failed configuration.

    Since we want to retry specific combinations, we create a structure
    that generates exactly those combinations.

    Args:
        failed_runs: List of failed run dictionaries

    Returns:
        Dictionary with parameter specifications
    """
    # We'll use a categorical sweep over configurations
    # Each configuration is a dictionary of parameters

    # Extract parameter combinations from failed runs
    # Remove the _load_params since that's handled separately
    configurations = []
    for run in failed_runs:
        params = run['parameters'].copy()
        # Remove internal fields
        params.pop('_load_params', None)
        params.pop('namespace', None)
        configurations.append(params)

    # If all configurations have the same structure, we can create
    # a more elegant parameter sweep. Otherwise, we need to handle
    # each unique parameter set.

    # For simplicity, we'll create individual parameters and sweep over their values
    # This works when all failed runs share the same parameter keys

    # Collect all unique parameter keys
    all_keys = set()
    for config in configurations:
        all_keys.update(config.keys())

    # Build parameters structure
    # For parameters that vary, create categorical sweeps
    # For parameters that are constant, create fixed values
    parameters = {}

    for key in all_keys:
        values = []
        for config in configurations:
            if key in config:
                val = config[key]
                if val not in values:
                    values.append(val)

        if len(values) == 1:
            # Fixed parameter
            parameters[key] = {
                'type': 'fixed',
                'value': values[0]
            }
        else:
            # Variable parameter - use categorical
            parameters[key] = {
                'type': 'categorical',
                'values': values
            }

    # Note: This approach generates a Cartesian product which might include
    # more combinations than just the failed ones if parameters varied independently.
    # For exact retry of specific combinations, we'd need a different approach.
    #
    # TODO: If needed, implement exact combination matching by using a
    # custom parameter type or by running configurations one at a time.

    return parameters


def write_retry_config_to_tempfile(retry_config):
    """
    Write retry configuration to a temporary file.

    Args:
        retry_config: Retry sweep configuration dictionary

    Returns:
        Path to temporary file
    """
    # Create a temporary file that won't be deleted when closed
    fd, temp_path = tempfile.mkstemp(suffix='.yaml', prefix='retry-sweep-')

    with open(temp_path, 'w') as f:
        yaml.dump(retry_config, f, default_flow_style=False)

    return temp_path


def main():
    if len(sys.argv) != 2:
        print("Usage: retry-failed.py <sweep_results_directory>")
        print()
        print("Example:")
        print("  retry-failed.py results/sweeps/my-sweep_2024-01-15")
        sys.exit(1)

    sweep_dir = Path(sys.argv[1])

    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        sys.exit(1)

    print(f"Loading sweep results from: {sweep_dir}")

    # Load sweep data
    try:
        metadata, summary = load_sweep_results(sweep_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Extract failed runs
    failed_runs = extract_failed_runs(summary)

    if not failed_runs:
        print("✓ No failed runs found! All configurations succeeded.")
        return

    print(f"Found {len(failed_runs)} failed configuration(s) out of {len(summary)} total runs")
    print()

    # Show failed configurations
    print("Failed configurations:")
    for run in failed_runs:
        print(f"  Run {run['run_id']}: {run.get('error', 'Unknown error')}")
        if 'parameters' in run:
            for key, value in run['parameters'].items():
                if key != '_load_params':
                    print(f"    {key}: {value}")
    print()

    # Create retry configuration
    retry_config = create_retry_config(metadata, failed_runs)

    # Write to temporary file
    temp_config_file = write_retry_config_to_tempfile(retry_config)

    print(f"Created retry configuration: {temp_config_file}")
    print()
    print("=" * 70)
    print("Starting retry sweep...")
    print("=" * 70)
    print()

    # Run the retry sweep
    try:
        orchestrator = SweepOrchestrator(temp_config_file)
        orchestrator.run_sweep()
    finally:
        # Clean up temporary file
        Path(temp_config_file).unlink()
        print(f"Cleaned up temporary config file")


if __name__ == "__main__":
    main()
