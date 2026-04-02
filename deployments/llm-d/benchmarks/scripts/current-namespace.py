#!/usr/bin/env python3
"""
Find currently active Kubernetes namespaces for a benchmark sweep.

Usage:
    python3 current-namespace.py <sweep-config-yaml>

Example:
    python3 current-namespace.py sweep-configs/example-cache-sweep.yaml
"""

import sys
import yaml
import subprocess
import os
import json
from pathlib import Path


def get_current_user():
    """Get current username."""
    user = os.environ.get('USER')
    if not user:
        try:
            user = subprocess.check_output(['whoami'], text=True).strip()
        except subprocess.CalledProcessError:
            user = 'unknown'
    return user


def parse_sweep_config(config_file):
    """Extract sweep name from YAML config."""
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config.get('name', 'unknown')


def get_active_namespaces():
    """Get all active Kubernetes namespaces."""
    try:
        result = subprocess.run(
            ['kubectl', 'get', 'ns', '--field-selector', 'status.phase=Active', '-o', 'json'],
            capture_output=True,
            text=True,
            check=True
        )
        ns_data = json.loads(result.stdout)
        return ns_data.get('items', [])
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to query Kubernetes namespaces: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse kubectl output: {e}", file=sys.stderr)
        sys.exit(1)


def filter_sweep_namespaces(namespaces, user, sweep_name):
    """Filter namespaces matching sweep pattern."""
    # Pattern: {user}-{sweep-name}-{timestamp}-{run-id}
    # We'll match on user and sweep-name prefix
    pattern_prefix = f"{user}-{sweep_name}-"

    matching = []
    for ns in namespaces:
        ns_name = ns['metadata']['name']
        if ns_name.startswith(pattern_prefix):
            # Extract run info
            parts = ns_name.split('-')
            # Last part should be run ID (3 digits)
            run_id = parts[-1] if parts else 'unknown'
            # Timestamp should be YYYY-MM-DD format (3rd to last, 2nd to last, last-1)
            timestamp = '-'.join(parts[-4:-1]) if len(parts) >= 4 else 'unknown'

            creation = ns['metadata'].get('creationTimestamp', 'unknown')

            matching.append({
                'namespace': ns_name,
                'run_id': run_id,
                'timestamp': timestamp,
                'created': creation
            })

    return matching


def display_namespaces(namespaces):
    """Display namespaces in a formatted table."""
    if not namespaces:
        print("No active namespaces found")
        return

    print(f"\n{'NAMESPACE':<60} {'RUN ID':<10} {'TIMESTAMP':<15} {'CREATED':<30}")
    print("-" * 115)

    for ns in namespaces:
        print(f"{ns['namespace']:<60} {ns['run_id']:<10} {ns['timestamp']:<15} {ns['created']:<30}")

    print(f"\nTotal: {len(namespaces)} active namespace(s)")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 current-namespace.py <sweep-config-yaml>", file=sys.stderr)
        sys.exit(1)

    config_file = Path(sys.argv[1])
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)

    # Get sweep info
    sweep_name = parse_sweep_config(config_file)
    user = get_current_user()

    # Normalize user (same as run-sweep.py)
    user = user.lower().replace('_', '-').replace('.', '-')

    print(f"Sweep: {sweep_name}")
    print(f"User: {user}")
    print(f"Pattern: {user}-{sweep_name}-*")

    # Query Kubernetes
    all_namespaces = get_active_namespaces()

    # Filter to matching namespaces
    matching = filter_sweep_namespaces(all_namespaces, user, sweep_name)

    # Display results
    display_namespaces(matching)


if __name__ == '__main__':
    main()
