#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Normalize Grafana dashboard JSON for version control.

Strips server-owned metadata (resourceVersion, Grafana user IDs, internal
labels) from v2 exports and removes classic ``id`` fields. Keeps a stable
``metadata.uid`` (or top-level ``uid``) so imports update the same dashboard.

Optional git provenance is written under ``rocm-aic.git.*`` annotations (v2)
or a ``__inputs``-free custom block is avoided — only v2 annotations for now.

Usage:
  normalize-dashboard.py [--check] [FILE ...]
  normalize-dashboard.py --ensure-uid [FILE ...]

Default FILE: grafana/rocm-aic-dashboard.json relative to repo root.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DASH = "grafana/rocm-aic-dashboard.json"

# Grafana UI / API metadata not useful in git (v2).
GRAFANA_APP_ANNOTATION_PREFIX = "grafana.app/"

# Written on each normalize (provenance for humans; Grafana may ignore).
ROCM_GIT_ANNOTATION_KEYS = (
    "rocm-aic.git.revision",
    "rocm-aic.git.author",
    "rocm-aic.git.normalizedAt",
)


def repo_root() -> Path:
    script = Path(__file__).resolve()
    # grafana/scripts/normalize-dashboard.py -> repo root
    return script.parents[2]


def default_dashboard_path() -> Path:
    return repo_root() / DEFAULT_DASH


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root(),
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_author() -> str:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%ae"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root(),
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_provenance_annotations() -> dict[str, str]:
    ann: dict[str, str] = {
        "rocm-aic.git.normalizedAt": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }
    rev = git_revision()
    if rev:
        ann["rocm-aic.git.revision"] = rev
    author = git_author()
    if author:
        ann["rocm-aic.git.author"] = author
    return ann


def detect_format(dash: dict[str, Any]) -> str:
    if dash.get("apiVersion") == "dashboard.grafana.app/v2" and dash.get("kind") == "Dashboard":
        return "v2"
    if "panels" in dash or dash.get("uid"):
        return "classic"
    return "unknown"


def strip_grafana_app_annotations(annotations: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in annotations.items()
        if not k.startswith(GRAFANA_APP_ANNOTATION_PREFIX)
        and not k.startswith("rocm-aic.git.")
    }


def normalize_v2(dash: dict[str, Any], *, add_git_provenance: bool) -> dict[str, Any]:
    out = deepcopy(dash)
    meta = out.setdefault("metadata", {})
    uid = meta.get("uid") or str(uuid.uuid4())
    name = meta.get("name") or "rocm-aic-dashboard"
    namespace = meta.get("namespace") or "default"

    new_meta: dict[str, Any] = {
        "name": name,
        "namespace": namespace,
        "uid": uid,
    }

    if add_git_provenance:
        new_meta["annotations"] = git_provenance_annotations()
    else:
        existing = strip_grafana_app_annotations(meta.get("annotations") or {})
        if existing:
            new_meta["annotations"] = existing

    out["metadata"] = new_meta
    return out


def normalize_classic(dash: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(dash)
    out.pop("id", None)
    return out


def normalize(
    dash: dict[str, Any],
    *,
    add_git_provenance: bool = True,
) -> dict[str, Any]:
    fmt = detect_format(dash)
    if fmt == "v2":
        return normalize_v2(dash, add_git_provenance=add_git_provenance)
    if fmt == "classic":
        return normalize_classic(dash)
    raise ValueError(
        f"Unrecognized dashboard format (expected v2 or classic, got keys: "
        f"{sorted(dash.keys())[:12]}...)"
    )


def serialize(dash: dict[str, Any]) -> str:
    return json.dumps(dash, indent=4, ensure_ascii=False) + "\n"


def volatile_metadata_errors(dash: dict[str, Any]) -> list[str]:
    """Return human-readable errors if server-owned fields are still present."""
    errors: list[str] = []
    fmt = detect_format(dash)
    if fmt == "v2":
        meta = dash.get("metadata") or {}
        for key in ("resourceVersion", "generation", "creationTimestamp", "labels"):
            if key in meta:
                errors.append(f"metadata.{key} must be removed (Grafana-owned)")
        allowed = {"name", "namespace", "uid", "annotations"}
        extra = set(meta.keys()) - allowed
        if extra:
            errors.append(f"unexpected metadata keys: {sorted(extra)}")
        if not meta.get("uid"):
            errors.append("metadata.uid is required (stable dashboard identity)")
        for key in meta.get("annotations") or {}:
            if key.startswith(GRAFANA_APP_ANNOTATION_PREFIX):
                errors.append(f"remove Grafana UI annotation {key!r}")
            elif not key.startswith("rocm-aic.git."):
                errors.append(f"unknown metadata annotation {key!r}")
    elif fmt == "classic":
        if "id" in dash:
            errors.append('top-level "id" must be removed (Grafana-owned)')
        if not dash.get("uid"):
            errors.append('top-level "uid" is required')
    else:
        errors.append("unrecognized dashboard format")
    return errors


def is_normalized(dash: dict[str, Any]) -> bool:
    """True when volatile fields are gone and core content matches normalize()."""
    if volatile_metadata_errors(dash):
        return False
    try:
        canonical = normalize(dash, add_git_provenance=False)
    except ValueError:
        return False
    fmt = detect_format(dash)
    if fmt == "v2":
        if dash.get("spec") != canonical.get("spec"):
            return False
        for key in ("name", "namespace", "uid"):
            if dash.get("metadata", {}).get(key) != canonical.get("metadata", {}).get(key):
                return False
        return True
    if fmt == "classic":
        return dash == canonical
    return False


def ensure_uid(dash: dict[str, Any]) -> bool:
    """Assign uid if missing. Returns True if uid was added."""
    fmt = detect_format(dash)
    if fmt == "v2":
        meta = dash.setdefault("metadata", {})
        if meta.get("uid"):
            return False
        meta["uid"] = str(uuid.uuid4())
        return True
    if fmt == "classic":
        if dash.get("uid"):
            return False
        dash["uid"] = str(uuid.uuid4())
        return True
    raise ValueError("Unrecognized dashboard format")


def process_file(
    path: Path,
    *,
    check: bool,
    ensure_uid_only: bool,
    no_git_provenance: bool,
) -> int:
    if not path.is_file():
        print(f"normalize-dashboard: not found: {path}", file=sys.stderr)
        return 1

    try:
        raw = path.read_text(encoding="utf-8")
        dash = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"normalize-dashboard: invalid JSON in {path}: {exc}", file=sys.stderr)
        return 1

    if ensure_uid_only:
        if ensure_uid(dash):
            path.write_text(serialize(dash), encoding="utf-8")
            print(f"normalize-dashboard: assigned new uid in {path}")
        else:
            print(f"normalize-dashboard: uid already set in {path}")
        return 0

    try:
        normalized = normalize(dash, add_git_provenance=not no_git_provenance)
    except ValueError as exc:
        print(f"normalize-dashboard: {path}: {exc}", file=sys.stderr)
        return 1

    if check:
        errors = volatile_metadata_errors(dash)
        if errors or not is_normalized(dash):
            for msg in errors:
                print(f"normalize-dashboard: {path}: {msg}", file=sys.stderr)
            if not errors and not is_normalized(dash):
                print(
                    f"normalize-dashboard: {path}: content drift "
                    f"(re-run normalize)",
                    file=sys.stderr,
                )
            print(
                "normalize-dashboard: run: make grafana-normalize",
                file=sys.stderr,
            )
            return 1
        print(f"normalize-dashboard: OK {path}")
        return 0

    rendered = serialize(normalized)

    if rendered != raw:
        path.write_text(rendered, encoding="utf-8")
        print(f"normalize-dashboard: normalized {path}")
    else:
        print(f"normalize-dashboard: unchanged {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize Grafana dashboard JSON for git.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help=f"Dashboard JSON file(s) (default: {DEFAULT_DASH})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if file(s) would change (for CI)",
    )
    parser.add_argument(
        "--ensure-uid",
        action="store_true",
        help="Only assign uid when missing; do not strip metadata",
    )
    parser.add_argument(
        "--no-git-provenance",
        action="store_true",
        help="Do not write rocm-aic.git.* annotations (v2 only)",
    )
    args = parser.parse_args()

    paths = args.files or [default_dashboard_path()]
    rc = 0
    for path in paths:
        path = path.resolve()
        if process_file(
            path,
            check=args.check,
            ensure_uid_only=args.ensure_uid,
            no_git_provenance=args.no_git_provenance,
        ) != 0:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
