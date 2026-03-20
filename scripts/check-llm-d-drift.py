#!/usr/bin/env python3
"""
Check for configuration drift between rocm-icms and llm-d upstream.
Compares chart versions, images, and other key configurations.
"""

import re
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys


class DriftChecker:
    """Compare rocm-icms configurations with llm-d guide."""

    def __init__(self, rocm_icms_root: Path, llm_d_submodule: Path):
        self.rocm_icms_root = rocm_icms_root
        self.llm_d = llm_d_submodule
        self.drifts = []

    def check_inference_scheduling_drift(self) -> List[Dict[str, Any]]:
        """Check drift in inference-scheduling deployment."""
        print("Checking inference-scheduling drift...")

        drifts = []

        # Compare helmfiles
        rocm_helmfile = self.rocm_icms_root / "deployments/llm-d/inference-scheduling/helmfile.yaml"
        llm_d_helmfile = self.llm_d / "guides/inference-scheduling/helmfile.yaml.gotmpl"

        rocm_config = self._parse_helmfile(rocm_helmfile)
        llm_d_config = self._parse_helmfile(llm_d_helmfile)

        # Check chart versions
        drifts.extend(self._compare_chart_versions(rocm_config, llm_d_config))

        # Check repository URLs
        drifts.extend(self._compare_repositories(rocm_config, llm_d_config))

        # Compare AMD values files
        drifts.extend(self._compare_amd_values())

        # Check common configuration versions
        drifts.extend(self._compare_common_config())

        return drifts

    def _parse_helmfile(self, filepath: Path) -> Dict[str, Any]:
        """Parse helmfile and extract key configurations."""
        if not filepath.exists():
            return {}

        with open(filepath) as f:
            content = f.read()

        config = {
            'chart_versions': {},
            'repositories': {},
        }

        # Extract chart versions
        # Pattern: version: v1.3.6
        for match in re.finditer(r'version:\s*([v\d.]+)', content):
            version = match.group(1)
            # Try to find chart name nearby
            snippet = content[max(0, match.start() - 200):match.start()]
            chart_match = re.search(r'chart:\s*([\w\-./]+)', snippet)
            if chart_match:
                chart = chart_match.group(1).split('/')[-1]
                config['chart_versions'][chart] = version

        # Extract repositories
        # Pattern: - name: llm-d-modelservice
        #          url: https://...
        repo_pattern = r'- name:\s*(\S+)\s+url:\s*(\S+)'
        for match in re.finditer(repo_pattern, content, re.MULTILINE):
            config['repositories'][match.group(1)] = match.group(2)

        return config

    def _compare_chart_versions(self, rocm: Dict, llm_d: Dict) -> List[Dict[str, Any]]:
        """Compare chart versions between rocm-icms and llm-d."""
        drifts = []

        rocm_versions = rocm.get('chart_versions', {})
        llm_d_versions = llm_d.get('chart_versions', {})

        # Common charts to check
        charts = set(rocm_versions.keys()) | set(llm_d_versions.keys())

        for chart in charts:
            rocm_ver = rocm_versions.get(chart)
            llm_d_ver = llm_d_versions.get(chart)

            if rocm_ver != llm_d_ver:
                drifts.append({
                    'type': 'chart_version',
                    'component': 'inference-scheduling',
                    'item': chart,
                    'rocm_icms': rocm_ver,
                    'llm_d': llm_d_ver,
                    'severity': 'medium' if rocm_ver and llm_d_ver else 'low',
                })

        return drifts

    def _compare_repositories(self, rocm: Dict, llm_d: Dict) -> List[Dict[str, Any]]:
        """Compare Helm repository URLs."""
        drifts = []

        rocm_repos = rocm.get('repositories', {})
        llm_d_repos = llm_d.get('repositories', {})

        repos = set(rocm_repos.keys()) | set(llm_d_repos.keys())

        for repo in repos:
            rocm_url = rocm_repos.get(repo)
            llm_d_url = llm_d_repos.get(repo)

            if rocm_url != llm_d_url:
                drifts.append({
                    'type': 'repository_url',
                    'component': 'inference-scheduling',
                    'item': repo,
                    'rocm_icms': rocm_url,
                    'llm_d': llm_d_url,
                    'severity': 'low',
                })

        return drifts

    def _compare_amd_values(self) -> List[Dict[str, Any]]:
        """Compare AMD values files."""
        drifts = []

        rocm_values_dir = self.rocm_icms_root / "deployments/llm-d/inference-scheduling/values"
        llm_d_values_file = self.llm_d / "guides/inference-scheduling/ms-inference-scheduling/values_amd.yaml"

        # Compare amd-default.yaml with llm-d's values_amd.yaml
        rocm_default = rocm_values_dir / "amd-default.yaml"
        if rocm_default.exists() and llm_d_values_file.exists():
            with open(rocm_default) as f:
                rocm_vals = yaml.safe_load(f)
            with open(llm_d_values_file) as f:
                llm_d_vals = yaml.safe_load(f)

            # Compare image
            rocm_image = self._get_nested(rocm_vals, 'decode.containers.0.image')
            llm_d_image = self._get_nested(llm_d_vals, 'decode.containers.0.image')

            if rocm_image != llm_d_image:
                drifts.append({
                    'type': 'image',
                    'component': 'inference-scheduling',
                    'item': 'vllm_container_image',
                    'rocm_icms': rocm_image,
                    'llm_d': llm_d_image,
                    'severity': 'high',
                    'note': 'Intentional: rocm-icms uses different vLLM image',
                })

            # Compare replicas
            rocm_replicas = self._get_nested(rocm_vals, 'decode.replicas')
            llm_d_replicas = self._get_nested(llm_d_vals, 'decode.replicas')

            if rocm_replicas != llm_d_replicas:
                drifts.append({
                    'type': 'configuration',
                    'component': 'inference-scheduling',
                    'item': 'decode.replicas',
                    'rocm_icms': rocm_replicas,
                    'llm_d': llm_d_replicas,
                    'severity': 'low',
                })

            # Compare tensor parallel size
            rocm_tp = self._get_nested(rocm_vals, 'decode.parallelism.tensor')
            llm_d_tp = self._get_nested(llm_d_vals, 'decode.parallelism.tensor')

            if rocm_tp != llm_d_tp:
                drifts.append({
                    'type': 'configuration',
                    'component': 'inference-scheduling',
                    'item': 'tensor_parallel_size',
                    'rocm_icms': rocm_tp,
                    'llm_d': llm_d_tp,
                    'severity': 'low',
                })

        return drifts

    def _compare_common_config(self) -> List[Dict[str, Any]]:
        """Compare common configuration files."""
        drifts = []

        # Compare llm-d common config with llm-d actual versions
        rocm_config = self.rocm_icms_root / "deployments/llm-d/common/config.yaml"

        if rocm_config.exists():
            with open(rocm_config) as f:
                config = yaml.safe_load(f)

            llm_d_config = config.get('llm_d', {})
            charts_config = config.get('charts', {})

            # Check if configured versions exist in llm-d
            # This would require parsing llm-d files to get actual versions
            # For now, just validate the config file exists and is well-formed
            if 'version' in llm_d_config:
                drifts.append({
                    'type': 'info',
                    'component': 'common',
                    'item': 'llm_d_version',
                    'rocm_icms': llm_d_config['version'],
                    'llm_d': 'check submodule',
                    'severity': 'info',
                })

        return drifts

    def _get_nested(self, d: Dict, path: str) -> Any:
        """Get nested dictionary value using dot notation."""
        keys = path.split('.')
        val = d
        for key in keys:
            if isinstance(val, dict):
                val = val.get(key)
            elif isinstance(val, list):
                try:
                    idx = int(key)
                    val = val[idx] if idx < len(val) else None
                except (ValueError, IndexError):
                    return None
            else:
                return None

            if val is None:
                return None

        return val

    def check_tiered_prefix_cache_drift(self) -> List[Dict[str, Any]]:
        """Check drift in tiered-prefix-cache deployment."""
        print("Checking tiered-prefix-cache drift...")

        drifts = []

        # Compare InferencePool values
        rocm_infpool = self.rocm_icms_root / "deployments/llm-d/tiered-prefix-cache/manifests/inferencepool/values.yaml"
        llm_d_infpool = self.llm_d / "guides/tiered-prefix-cache/cpu/manifests/inferencepool/values.yaml"

        if rocm_infpool.exists() and llm_d_infpool.exists():
            with open(rocm_infpool) as f:
                rocm_vals = yaml.safe_load(f)
            with open(llm_d_infpool) as f:
                llm_d_vals = yaml.safe_load(f)

            # Check lruCapacityPerServer
            rocm_lru = self._extract_lru_capacity(rocm_vals)
            llm_d_lru = self._extract_lru_capacity(llm_d_vals)

            if rocm_lru != llm_d_lru:
                drifts.append({
                    'type': 'configuration',
                    'component': 'tiered-prefix-cache',
                    'item': 'lruCapacityPerServer',
                    'rocm_icms': rocm_lru,
                    'llm_d': llm_d_lru,
                    'severity': 'low',
                })

        # Compare AMD base recipe image
        llm_d_amd_recipe = self.llm_d / "guides/recipes/vllm/amd/kustomization.yaml"

        if llm_d_amd_recipe.exists():
            with open(llm_d_amd_recipe) as f:
                amd_recipe = yaml.safe_load(f)

            if 'images' in amd_recipe:
                for img in amd_recipe['images']:
                    if img.get('name') == 'INFERENCE_SERVER_IMAGE':
                        drifts.append({
                            'type': 'info',
                            'component': 'tiered-prefix-cache',
                            'item': 'amd_base_image',
                            'rocm_icms': f"{img.get('newName')}:{img.get('newTag')}",
                            'llm_d': 'same (modified in submodule)',
                            'severity': 'info',
                            'note': 'rocm-icms modified llm-d AMD base recipe',
                        })

        return drifts

    def _extract_lru_capacity(self, values: Dict[str, Any]) -> Any:
        """Extract lruCapacityPerServer from InferencePool config."""
        # It's nested in pluginsCustomConfig YAML string
        custom_config = self._get_nested(values, 'inferenceExtension.pluginsCustomConfig.custom-plugins.yaml')
        if custom_config and isinstance(custom_config, str):
            match = re.search(r'lruCapacityPerServer:\s*(\d+)', custom_config)
            if match:
                return int(match.group(1))

        return None

    def report_drifts(self, drifts: List[Dict[str, Any]]):
        """Generate drift report."""
        if not drifts:
            print("\n✅ No drift detected - rocm-icms is aligned with llm-d")
            return

        print(f"\n⚠️  Found {len(drifts)} differences between rocm-icms and llm-d")
        print("="*80)

        # Group by severity
        by_severity = {
            'high': [d for d in drifts if d['severity'] == 'high'],
            'medium': [d for d in drifts if d['severity'] == 'medium'],
            'low': [d for d in drifts if d['severity'] == 'low'],
            'info': [d for d in drifts if d['severity'] == 'info'],
        }

        for severity in ['high', 'medium', 'low', 'info']:
            items = by_severity[severity]
            if not items:
                continue

            severity_emoji = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢',
                'info': 'ℹ️',
            }

            print(f"\n{severity_emoji[severity]} {severity.upper()} ({len(items)})")
            print("-"*80)

            for drift in items:
                print(f"\nComponent: {drift['component']}")
                print(f"Type:      {drift['type']}")
                print(f"Item:      {drift['item']}")
                print(f"rocm-icms: {drift['rocm_icms']}")
                print(f"llm-d:     {drift['llm_d']}")

                if 'note' in drift:
                    print(f"Note:      {drift['note']}")

        print("\n" + "="*80)
        print("\nRecommendations:")
        print("- HIGH: Review immediately, may cause compatibility issues")
        print("- MEDIUM: Consider updating to stay aligned")
        print("- LOW: Minor differences, monitor for future updates")
        print("- INFO: Informational only, likely intentional")

    def generate_json_report(self, drifts: List[Dict[str, Any]], output_file: Path):
        """Generate JSON report for programmatic consumption."""
        report = {
            'total_drifts': len(drifts),
            'by_severity': {
                'high': len([d for d in drifts if d['severity'] == 'high']),
                'medium': len([d for d in drifts if d['severity'] == 'medium']),
                'low': len([d for d in drifts if d['severity'] == 'low']),
                'info': len([d for d in drifts if d['severity'] == 'info']),
            },
            'drifts': drifts,
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nJSON report saved to: {output_file}")

    def check_all(self) -> List[Dict[str, Any]]:
        """Check all deployments for drift."""
        all_drifts = []

        all_drifts.extend(self.check_inference_scheduling_drift())
        all_drifts.extend(self.check_tiered_prefix_cache_drift())

        return all_drifts


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Check for configuration drift between rocm-icms and llm-d'
    )
    parser.add_argument(
        '--rocm-icms-root',
        type=Path,
        default=Path(__file__).parent.parent,
        help='Path to rocm-icms root directory'
    )
    parser.add_argument(
        '--llm-d-submodule',
        type=Path,
        help='Path to llm-d submodule (auto-detected if not provided)'
    )
    parser.add_argument(
        '--json',
        type=Path,
        help='Output JSON report to file'
    )
    parser.add_argument(
        '--deployment',
        choices=['inference-scheduling', 'tiered-prefix-cache', 'all'],
        default='all',
        help='Which deployment to check'
    )

    args = parser.parse_args()

    # Auto-detect llm-d submodule
    if not args.llm_d_submodule:
        args.llm_d_submodule = args.rocm_icms_root / "submodules/llm-d"

    if not args.llm_d_submodule.exists():
        print(f"Error: llm-d submodule not found at {args.llm_d_submodule}")
        print("Run 'just setup-submodules' from rocm-icms root")
        sys.exit(1)

    # Create checker
    checker = DriftChecker(args.rocm_icms_root, args.llm_d_submodule)

    # Check for drift
    if args.deployment == 'inference-scheduling':
        drifts = checker.check_inference_scheduling_drift()
    elif args.deployment == 'tiered-prefix-cache':
        drifts = checker.check_tiered_prefix_cache_drift()
    else:
        drifts = checker.check_all()

    # Report results
    checker.report_drifts(drifts)

    # Generate JSON report if requested
    if args.json:
        checker.generate_json_report(drifts, args.json)

    # Exit with non-zero if high severity drifts found
    high_severity = [d for d in drifts if d['severity'] == 'high']
    if high_severity:
        sys.exit(1)


if __name__ == '__main__':
    main()
