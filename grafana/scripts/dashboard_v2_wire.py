# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Grafana dashboard v2 schema helpers for variables and transformations."""

from __future__ import annotations

from typing import Any


def is_wrapped_variable(var: dict[str, Any]) -> bool:
    """True when a variable uses mistaken *VariableKind union-key wrappers."""
    return any(k.endswith("VariableKind") for k in var)


def variable_inner(var: dict[str, Any]) -> dict[str, Any]:
    """Return {kind, spec} for a dashboard variable in any on-disk format."""
    if is_wrapped_variable(var):
        for key, inner in var.items():
            if key.endswith("VariableKind") and isinstance(inner, dict):
                return inner
        return {}
    return var


def variable_spec(var: dict[str, Any]) -> dict[str, Any]:
    inner = variable_inner(var)
    spec = inner.get("spec")
    return spec if isinstance(spec, dict) else {}


def variable_dict(kind: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Grafana v2 variable entry: {kind, spec} at the array element root."""
    return {"kind": kind, "spec": spec}


def fix_variables_wire_format(variables: list[dict[str, Any]]) -> None:
    """Normalize variables to flat {kind, spec} (unwrap mistaken union wrappers)."""
    for i, var in enumerate(variables):
        inner = variable_inner(var)
        kind = inner.get("kind")
        spec = inner.get("spec")
        if isinstance(kind, str) and isinstance(spec, dict):
            variables[i] = variable_dict(kind, spec)


def is_v2beta1_transformation(t: dict[str, Any]) -> bool:
    spec = t.get("spec")
    return (
        isinstance(t.get("kind"), str)
        and t.get("kind") != "Transformation"
        and isinstance(spec, dict)
        and isinstance(spec.get("id"), str)
    )


def transformation_dict(trans_id: str, options: dict[str, Any]) -> dict[str, Any]:
    """Grafana v2beta1 TransformationKind: kind is the transformer id."""
    return {
        "kind": trans_id,
        "spec": {
            "id": trans_id,
            "options": options,
        },
    }


def fix_transformations_wire_format(dash: dict[str, Any]) -> None:
    """Normalize panel transformations to v2beta1 TransformationKind."""
    elements = dash.get("spec", {}).get("elements", {})
    if not isinstance(elements, dict):
        return
    for elem in elements.values():
        if elem.get("kind") != "Panel":
            continue
        qspec = elem.get("spec", {}).get("data", {}).get("spec", {})
        if not isinstance(qspec, dict):
            continue
        transforms = qspec.get("transformations")
        if not isinstance(transforms, list):
            continue
        fixed: list[dict[str, Any]] = []
        for t in transforms:
            if is_v2beta1_transformation(t):
                fixed.append(t)
            elif t.get("kind") == "Transformation" and isinstance(t.get("group"), str):
                fixed.append(
                    transformation_dict(t["group"], t.get("spec", {}).get("options") or {})
                )
            elif isinstance(t.get("id"), str):
                fixed.append(transformation_dict(t["id"], t.get("options") or {}))
            else:
                fixed.append(t)
        qspec["transformations"] = fixed


def fix_dashboard_v2_wire_format(dash: dict[str, Any]) -> None:
    """Apply v2 schema fixes for variables and transformations."""
    spec = dash.get("spec")
    if not isinstance(spec, dict):
        return
    variables = spec.get("variables")
    if isinstance(variables, list):
        fix_variables_wire_format(variables)
    fix_transformations_wire_format(dash)
