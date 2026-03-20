#!/usr/bin/env python3
"""
Extract pod configuration information from kubectl describe output files.

Parses pod describe files to extract labels, container images, command lines,
environment settings, and resource limits/requests.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def parse_labels(lines: List[str], start_idx: int) -> tuple[Dict[str, str], int]:
    """
    Parse labels from describe output.

    Args:
        lines: All lines from the describe file
        start_idx: Index where "Labels:" line is found

    Returns:
        Tuple of (labels_dict, next_line_index)
    """
    labels = {}
    idx = start_idx + 1

    while idx < len(lines):
        line = lines[idx]

        # Check if we've moved to the next section
        if line and not line.startswith(' ') and ':' in line:
            break

        # Skip empty lines
        if not line.strip():
            idx += 1
            continue

        # Parse label line (format: "  key=value" or "                  key=value")
        stripped = line.strip()
        if '=' in stripped:
            key, value = stripped.split('=', 1)
            labels[key] = value
        elif stripped == '<none>':
            # No labels
            break

        idx += 1

    return labels, idx


def parse_multiline_value(lines: List[str], start_idx: int, indent_level: int = 4) -> tuple[List[str], int]:
    """
    Parse a multiline value (like Command or Args).

    Args:
        lines: All lines from the describe file
        start_idx: Index to start parsing from (line after "Command:" or "Args:")
        indent_level: Expected indentation level for value lines

    Returns:
        Tuple of (list_of_values, next_line_index)
    """
    values = []
    idx = start_idx

    while idx < len(lines):
        line = lines[idx]

        # Check if we've moved to a new field (less indentation or new section)
        if line and not line.startswith(' ' * indent_level):
            break

        # Extract the value
        stripped = line.strip()
        if stripped:
            values.append(stripped)

        idx += 1

    return values, idx


def parse_environment(lines: List[str], start_idx: int) -> tuple[Dict[str, Any], int]:
    """
    Parse environment variables from describe output.

    Args:
        lines: All lines from the describe file
        start_idx: Index where "Environment:" line is found

    Returns:
        Tuple of (env_dict, next_line_index)
    """
    env = {}
    idx = start_idx + 1

    # Expected indentation for environment items (6 spaces in the example)
    expected_indent = 6

    while idx < len(lines):
        line = lines[idx]

        # Check if line is properly indented for an environment entry
        if not line.startswith(' ' * expected_indent):
            # We've reached the end of this section
            break

        # Get the portion after the indent
        content = line[expected_indent:].strip()

        # Skip empty lines
        if not content:
            idx += 1
            continue

        # Check for <none>
        if content == '<none>':
            idx += 1
            break

        # Parse environment variable line
        # Format: "VAR_NAME: value" or "VAR_NAME: <set to the key 'KEY' in secret 'SECRET'>"
        if ':' in content:
            key, value = content.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Check for secret reference
            if '<set to the key' in value:
                env[key] = "type secret_ref"
            else:
                env[key] = value
            idx += 1
        else:
            # Not a valid env line, stop parsing
            break

    return env, idx


def parse_resources(lines: List[str], start_idx: int, section_name: str) -> tuple[Dict[str, str], int]:
    """
    Parse resource limits or requests.

    Args:
        lines: All lines from the describe file
        start_idx: Index where "Limits:" or "Requests:" line is found
        section_name: "Limits" or "Requests" for context

    Returns:
        Tuple of (resources_dict, next_line_index)
    """
    resources = {}
    idx = start_idx + 1

    # Expected indentation for resource items (6 spaces in the example)
    expected_indent = 6

    while idx < len(lines):
        line = lines[idx]

        # Check if line is properly indented for a resource entry
        if not line.startswith(' ' * expected_indent):
            # We've reached the end of this section
            break

        # Get the portion after the indent
        content = line[expected_indent:].strip()

        # Skip empty lines
        if not content:
            idx += 1
            continue

        # Parse resource line (format: "resource: value")
        if ':' in content:
            key, value = content.split(':', 1)
            key = key.strip()
            value = value.strip()
            resources[key] = value
            idx += 1
        else:
            # Not a valid resource line, stop parsing
            break

    return resources, idx


def extract_modelserver_info(lines: List[str]) -> Optional[Dict[str, Any]]:
    """
    Extract information from the modelserver container section.

    Args:
        lines: All lines from the describe file

    Returns:
        Dictionary with modelserver information, or None if not found
    """
    # Find the modelserver container section
    modelserver_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == 'modelserver:':
            modelserver_idx = idx
            break

    if modelserver_idx is None:
        return None

    info = {
        'image': None,
        'command_line': None,
        'environment': {},
        'limits': {},
        'requests': {}
    }

    idx = modelserver_idx + 1
    while idx < len(lines):
        line = lines[idx]

        # Stop if we hit the next container or section
        if line and not line.startswith(' '):
            break

        stripped = line.strip()

        # Extract Image
        if line.startswith('    Image:'):
            info['image'] = stripped.split(':', 1)[1].strip()

        # Extract Command
        elif line.startswith('    Command:'):
            command_values, idx = parse_multiline_value(lines, idx + 1, indent_level=6)

            # Extract Args
            if idx < len(lines) and lines[idx].startswith('    Args:'):
                args_values, idx = parse_multiline_value(lines, idx + 1, indent_level=6)
            else:
                args_values = []

            # Combine command and args
            info['command_line'] = ' '.join(command_values + args_values)
            continue

        # Extract Environment
        elif line.startswith('    Environment:'):
            info['environment'], idx = parse_environment(lines, idx)
            continue

        # Extract Limits
        elif line.startswith('    Limits:'):
            info['limits'], idx = parse_resources(lines, idx, 'Limits')
            continue

        # Extract Requests
        elif line.startswith('    Requests:'):
            info['requests'], idx = parse_resources(lines, idx, 'Requests')
            continue

        idx += 1

    return info


def parse_pod_describe_file(file_path: Path) -> Dict[str, Any]:
    """
    Parse a single pod describe file.

    Args:
        file_path: Path to the describe file

    Returns:
        Dictionary containing extracted information
    """
    with open(file_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')

    result = {
        'file': str(file_path.name),
        'pod_name': None,
        'labels': {},
        'modelserver': None
    }

    # Extract pod name
    for line in lines:
        if line.startswith('Name:'):
            result['pod_name'] = line.split(':', 1)[1].strip()
            break

    # Extract labels
    for idx, line in enumerate(lines):
        if line.startswith('Labels:'):
            result['labels'], _ = parse_labels(lines, idx)
            break

    # Extract modelserver information
    result['modelserver'] = extract_modelserver_info(lines)

    return result


def extract_pod_info_from_run(run_dir: Path) -> List[Dict[str, Any]]:
    """
    Scan a run directory for pod describe files and extract information.

    Args:
        run_dir: Path to the run directory

    Returns:
        List of dictionaries, one per describe file
    """
    # Find all describe files matching the pattern
    snapshots_dir = run_dir / 'snapshots'

    if not snapshots_dir.exists():
        return []

    describe_files = list(snapshots_dir.glob('*-describe.txt'))

    results = []
    for describe_file in sorted(describe_files):
        try:
            pod_info = parse_pod_describe_file(describe_file)
            results.append(pod_info)
        except Exception as e:
            # Log error but continue processing other files
            print(f"Warning: Failed to parse {describe_file}: {e}")
            continue

    return results


def main():
    """CLI entry point for testing the extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract pod information from describe files in a run directory"
    )
    parser.add_argument(
        "run_dir",
        help="Path to the run directory"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output"
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Directory not found: {run_dir}")
        return 1

    results = extract_pod_info_from_run(run_dir)

    if args.pretty:
        print(json.dumps(results, indent=2))
    else:
        print(json.dumps(results))

    print(f"\n# Extracted information from {len(results)} pod(s)", file=__import__('sys').stderr)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
