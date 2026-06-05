#!/usr/bin/env python3
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Run a kv-cache-tester profile against a live vLLM endpoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from build_argv import build_argv, load_profile


def _upstream_python(upstream_root: Path) -> str:
    venv_py = upstream_root / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--api-endpoint", required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    args = parser.parse_args()

    profile = load_profile(args.config)
    output_rel = profile.get("output_dir", "logs/run-latest")
    if not isinstance(output_rel, str):
        output_rel = "logs/run-latest"
    output_dir = (args.bench_root / output_rel).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    argv = build_argv(
        profile,
        api_endpoint=args.api_endpoint,
        upstream_root=args.upstream_root,
        output_dir=output_dir,
    )

    python = _upstream_python(args.upstream_root)
    cmd = [python, *argv]
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=args.upstream_root)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
