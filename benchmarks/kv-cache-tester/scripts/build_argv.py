#!/usr/bin/env python3
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Build kv-cache-tester CLI argv from a YAML profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    print("ERROR: PyYAML required (pip install pyyaml)", file=sys.stderr)
    raise SystemExit(1) from exc


def _flag_name(key: str) -> str:
    return key.replace("_", "-")


def build_argv(
    profile: dict[str, Any],
    *,
    api_endpoint: str,
    upstream_root: Path,
    output_dir: Path,
) -> list[str]:
    script = profile.get("script")
    if not isinstance(script, str) or not script.strip():
        raise ValueError("profile must define script")

    args_cfg = profile.get("args")
    if not isinstance(args_cfg, dict):
        args_cfg = {}

    argv = [
        str(upstream_root / script.strip()),
        "--api-endpoint",
        api_endpoint,
        "--output-dir",
        str(output_dir),
    ]

    for key, value in args_cfg.items():
        flag = f"--{_flag_name(str(key))}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv.extend([flag, str(value)])

    return argv


def load_profile(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected mapping at top level")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--api-endpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="emit JSON array instead of shell-escaped argv",
    )
    args = parser.parse_args()

    profile = load_profile(args.config)
    argv = build_argv(
        profile,
        api_endpoint=args.api_endpoint,
        upstream_root=args.upstream_root,
        output_dir=args.output_dir,
    )

    if args.print_json:
        print(json.dumps(argv))
    else:
        for part in argv:
            print(part)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
