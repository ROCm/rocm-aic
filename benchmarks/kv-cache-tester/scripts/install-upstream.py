#!/usr/bin/env python3
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Clone upstream kv-cache-tester (with traces submodule)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    print("ERROR: PyYAML required", file=sys.stderr)
    raise SystemExit(1) from exc


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()

    profile = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        print("ERROR: invalid config", file=sys.stderr)
        return 1

    upstream = profile.get("upstream")
    if not isinstance(upstream, dict):
        print("ERROR: config missing upstream mapping", file=sys.stderr)
        return 1

    repo = str(upstream.get("repo", "")).strip()
    ref = str(upstream.get("ref", "master")).strip()
    if not repo:
        print("ERROR: upstream.repo required", file=sys.stderr)
        return 1

    dest = args.dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if (dest / ".git").is_dir():
        _run(["git", "-C", str(dest), "fetch", "origin", ref])
        _run(["git", "-C", str(dest), "checkout", ref])
        _run(["git", "-C", str(dest), "pull", "--ff-only", "origin", ref])
        _run(["git", "-C", str(dest), "submodule", "update", "--init", "--recursive"])
    else:
        _run(
            [
                "git",
                "clone",
                "--recursive",
                "--branch",
                ref,
                repo,
                str(dest),
            ]
        )

    traces = dest / "traces"
    if not traces.is_dir():
        print(f"ERROR: missing traces/ under {dest} (recursive clone failed?)", file=sys.stderr)
        return 1

    print(f"OK: kv-cache-tester at {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
