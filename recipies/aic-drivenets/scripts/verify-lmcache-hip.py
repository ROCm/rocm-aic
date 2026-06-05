#!/usr/bin/env python3
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Verify LMCache HIP c_ops was built (not Python fallback on ROCm)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _lmcache_git_sha() -> str | None:
    git_dir = Path("/app/LMCache")
    if not (git_dir / ".git").is_dir():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(git_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _site_package_roots() -> list[Path]:
    roots: list[Path] = [Path("/app/LMCache")]
    try:
        import site

        for entry in site.getsitepackages():
            roots.append(Path(entry) / "lmcache")
    except ImportError:
        pass
    return roots


def _find_c_ops_so() -> Path | None:
    for root in _site_package_roots():
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("c_ops*.so")):
            if path.is_file():
                return path
    return None


def _check_rocm_torch() -> str:
    import torch

    ver = torch.__version__
    if "rocm" not in ver and not torch.version.hip:
        raise SystemExit(f"ERROR: expected ROCm torch, got {ver}")
    hip = Path(torch.__file__).resolve().parent / "lib" / "libtorch_hip.so"
    if not hip.is_file():
        raise SystemExit(f"ERROR: missing {hip}")
    return ver


def main() -> int:
    try:
        torch_ver = _check_rocm_torch()
    except ImportError as e:
        print(f"FAIL: cannot import torch ({e})", file=sys.stderr)
        return 1

    so_path = _find_c_ops_so()
    if so_path is None:
        print(
            "FAIL: no lmcache c_ops*.so found under /app/LMCache or site-packages.\n"
            "Rebuild with BUILD_WITH_HIP=1 and pip install -e . --no-deps.",
            file=sys.stderr,
        )
        return 1

    try:
        import lmcache.c_ops as c_ops  # noqa: F401
    except ImportError:
        pass

    sha = _lmcache_git_sha()
    print(f"OK: ROCm torch={torch_ver}")
    print(f"    LMCache HIP c_ops at {so_path}")
    if sha:
        print(f"    LMCache git HEAD: {sha}")

    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "show", "lmcache"],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in out.stdout.splitlines():
            if line.startswith(("Name:", "Version:", "Location:")):
                print(f"    {line}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
