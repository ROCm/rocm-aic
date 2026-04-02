#!/usr/bin/env python3
"""
Extract deployment type from a namespace by querying its Kubernetes labels.

This script reads and prints llm-d.ai/deployment label from the namespace

Usage:
    python3 get-deployment-from-namespace.py <namespace>

Output:
    deployment label
"""

import sys
import json
import subprocess

def get_namespace_labels(namespace):
    """Query Kubernetes for namespace labels."""
    try:
        result = subprocess.run(
            ['kubectl', 'get', 'namespace', namespace, '-o', 'json'],
            capture_output=True,
            text=True,
            check=True
        )
        ns_data = json.loads(result.stdout)
        return ns_data.get('metadata', {}).get('labels', {})
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to query namespace '{namespace}': {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse kubectl output: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 get-deployment-from-namespace.py <namespace>", file=sys.stderr)
        sys.exit(1)

    namespace = sys.argv[1]

    # Get namespace labels
    labels = get_namespace_labels(namespace)

    # Extract deployment from label
    deployment = labels.get('llm-d.ai/deployment')

    if not deployment:
        print(f"Error: Namespace '{namespace}' does not have llm-d.ai/deployment label", file=sys.stderr)
        print(f"Available labels: {list(labels.keys())}", file=sys.stderr)
        sys.exit(1)

    print(deployment)

if __name__ == '__main__':
    main()
