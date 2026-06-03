#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Lint Grafana dashboard JSON under grafana/.

Checks structure expected by Grafana v2 dashboard schema, git normalization
rules, and project conventions (datasource variable shape, PromQL selectors).

Usage:
  lint-dashboards.py [--check] [FILE ...]

Default: all grafana/*.json dashboard candidates.
Exit 0 when clean; exit 1 when --check and any file would change or has errors.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent


def _load_wire_module():
    path = _SCRIPT_DIR / "dashboard_v2_wire.py"
    spec = importlib.util.spec_from_file_location("dashboard_v2_wire", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_wire = _load_wire_module()
variable_spec = _wire.variable_spec
is_wrapped_variable = _wire.is_wrapped_variable
is_v2beta1_transformation = _wire.is_v2beta1_transformation


def _load_normalize_module():
    path = _SCRIPT_DIR / "normalize-dashboard.py"
    spec = importlib.util.spec_from_file_location("normalize_dashboard", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_norm = _load_normalize_module()
detect_format = _norm.detect_format
is_normalized = _norm.is_normalized
volatile_metadata_errors = _norm.volatile_metadata_errors
default_dashboard_paths = _norm.default_dashboard_paths

VALID_VARIABLE_HIDE = frozenset(
    {"dontHide", "hideLabel", "hideVariable", "inControlsMenu"}
)
VALID_VARIABLE_REFRESH = frozenset(
    {"never", "onDashboardLoad", "onTimeRangeChanged", "onVariableChange"}
)
EMPTY_VARIABLE_CURRENT = {"text": "", "value": ""}

# Legacy import name; prefer ${datasource} in panel queries.
LEGACY_DATASOURCE_VAR = "DS_PROMETHEUS"


def repo_root() -> Path:
    return _SCRIPT_DIR.parents[1]


def dashboard_candidates() -> list[Path]:
    root = repo_root() / "grafana"
    return sorted(p for p in root.glob("*.json") if p.is_file())


def iter_objects(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from iter_objects(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_objects(item)


def variable_names(dash: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for var in dash.get("spec", {}).get("variables", []):
        name = variable_spec(var).get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def panel_count_v2(dash: dict[str, Any]) -> int:
    return sum(
        1
        for elem in dash.get("spec", {}).get("elements", {}).values()
        if elem.get("kind") == "Panel"
    )


def lint_variable(var: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []
    if is_wrapped_variable(var):
        wrapper = next(k for k in var if k.endswith("VariableKind"))
        errors.append(
            f"{path}: variable {wrapper!r} uses union-key wrapper; "
            f'use flat {{"kind": "...", "spec": {{...}}}} instead'
        )
        return errors

    kind = var.get("kind")
    spec = var.get("spec")
    if not isinstance(spec, dict):
        return [f"{path}: variable missing spec"]

    name = spec.get("name", "")
    prefix = f"{path}: variable {name!r}"

    hide = spec.get("hide")
    if hide is not None and hide not in VALID_VARIABLE_HIDE:
        errors.append(f"{prefix}: invalid hide {hide!r}")

    refresh = spec.get("refresh")
    if refresh is not None and refresh not in VALID_VARIABLE_REFRESH:
        errors.append(f"{prefix}: invalid refresh {refresh!r}")

    current = spec.get("current")
    if current is None:
        errors.append(f"{prefix}: missing current (use {{\"text\": \"\", \"value\": \"\"}})")
    elif not isinstance(current, dict):
        errors.append(f"{prefix}: current must be an object")
    elif "text" not in current or "value" not in current:
        errors.append(
            f"{prefix}: current must include text and value keys "
            f"(got {sorted(current.keys())})"
        )

    if kind == "DatasourceVariable":
        if name == LEGACY_DATASOURCE_VAR:
            errors.append(
                f"{prefix}: rename to 'datasource' and use ${{datasource}} in queries"
            )
        if not spec.get("pluginId"):
            errors.append(f"{prefix}: DatasourceVariable requires pluginId")

    if kind == "QueryVariable":
        q = spec.get("query")
        if not isinstance(q, dict) or q.get("group") != "prometheus":
            errors.append(f"{prefix}: QueryVariable requires prometheus query")
        else:
            ds = q.get("datasource", {})
            ds_name = ds.get("name") if isinstance(ds, dict) else None
            if ds_name == f"${{{LEGACY_DATASOURCE_VAR}}}":
                errors.append(f"{prefix}: use datasource ${'{datasource}'} not DS_PROMETHEUS")

    return errors


def lint_prom_expr(expr: str, path: Path, *, ctx: str) -> list[str]:
    errors: list[str] = []
    if "${DS_PROMETHEUS}" in expr or '"${DS_PROMETHEUS}"' in expr:
        errors.append(f"{path}: {ctx} uses legacy ${{DS_PROMETHEUS}}; use ${{datasource}}")
    vllm_amd = path.name.startswith("vllm-")
    if vllm_amd and "vllm:" in expr and "job=~" not in expr and 'job="' not in expr:
        errors.append(
            f"{path}: {ctx} vLLM query missing job filter "
            f'(expected job=~"vllm-exporter|vllm")'
        )
    if vllm_amd and re.search(r'server_name="\$instance"', expr):
        errors.append(
            f"{path}: {ctx} uses server_name=\"$instance\"; "
            f"vLLM metrics should use instance=~\"$instance\""
        )
    return errors


def lint_dashboard(path: Path, dash: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    errors.extend(f"{path}: {msg}" for msg in volatile_metadata_errors(dash))

    fmt = detect_format(dash)
    if fmt == "unknown":
        return errors + [f"{path}: unrecognized dashboard format"]

    if fmt == "v2":
        if dash.get("apiVersion") != "dashboard.grafana.app/v2":
            errors.append(f"{path}: apiVersion must be dashboard.grafana.app/v2")
        if dash.get("kind") != "Dashboard":
            errors.append(f"{path}: kind must be Dashboard")
        if panel_count_v2(dash) < 1:
            errors.append(f"{path}: expected at least one Panel in spec.elements")

        names = variable_names(dash)
        if "datasource" not in names and LEGACY_DATASOURCE_VAR not in names:
            errors.append(f"{path}: missing datasource variable")

        for var in dash.get("spec", {}).get("variables", []):
            errors.extend(lint_variable(var, path))

        for elem in dash.get("spec", {}).get("elements", {}).values():
            if elem.get("kind") != "Panel":
                continue
            transforms = (
                elem.get("spec", {})
                .get("data", {})
                .get("spec", {})
                .get("transformations", [])
            )
            if not isinstance(transforms, list):
                continue
            for i, t in enumerate(transforms):
                if not isinstance(t, dict) or not t:
                    continue
                if t.get("kind") == "Transformation":
                    errors.append(
                        f"{path}: transformation[{i}] uses Transformation wrapper; "
                        f'use {{"kind": "<id>", "spec": {{"id": "<id>", "options": ...}}}}'
                    )
                elif "id" in t and "options" in t and "spec" not in t:
                    errors.append(
                        f"{path}: transformation[{i}] uses top-level id/options; "
                        f'use {{"kind": "<id>", "spec": {{"id": "<id>", "options": ...}}}}'
                    )
                elif not is_v2beta1_transformation(t):
                    errors.append(
                        f"{path}: transformation[{i}] must be v2beta1 "
                        f"TransformationKind (kind + spec.id + spec.options)"
                    )

        for obj in iter_objects(dash.get("spec", {})):
            if not isinstance(obj, dict):
                continue
            ds = obj.get("datasource")
            if isinstance(ds, dict):
                ds_name = ds.get("name")
                if ds_name == f"${{{LEGACY_DATASOURCE_VAR}}}":
                    errors.append(
                        f"{path}: panel/query uses legacy ${{DS_PROMETHEUS}} datasource"
                    )
                if isinstance(ds_name, str) and ds_name.startswith("${") and ds_name.endswith("}"):
                    ref = ds_name[2:-1]
                    if ref not in names and ref != "datasource":
                        errors.append(
                            f"{path}: datasource reference ${{{ref}}} "
                            f"has no matching variable"
                        )
            spec = obj.get("spec")
            if isinstance(spec, dict) and isinstance(spec.get("expr"), str):
                errors.extend(
                    lint_prom_expr(spec["expr"], path, ctx="PromQL expr")
                )

    if not is_normalized(dash):
        errors.append(f"{path}: not normalized (run make grafana-normalize)")

    return errors


def process_file(path: Path, *, check: bool) -> int:
    if not path.is_file():
        print(f"lint-dashboards: not found: {path}", file=sys.stderr)
        return 1

    try:
        dash = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"lint-dashboards: invalid JSON in {path}: {exc}", file=sys.stderr)
        return 1

    errors = lint_dashboard(path, dash)
    if errors:
        for msg in errors:
            print(f"lint-dashboards: {msg}", file=sys.stderr)
        if check:
            print("lint-dashboards: fix errors above and re-run", file=sys.stderr)
        return 1

    print(f"lint-dashboards: OK {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Lint Grafana dashboard JSON.")
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Dashboard JSON file(s) (default: grafana/*.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 when lint fails (for CI)",
    )
    args = parser.parse_args()

    paths = [p.resolve() for p in (args.files or dashboard_candidates())]
    if not paths:
        print("lint-dashboards: no dashboard candidates found", file=sys.stderr)
        return 1

    rc = 0
    for path in paths:
        if process_file(path, check=args.check) != 0:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
