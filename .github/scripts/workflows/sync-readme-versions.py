#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Keep README component metadata aligned with the source-built stack."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[3]
# docker/docker-bake.hcl owns every pin; aic-bake-env.sh prints what it resolves.
BAKE_FILE = REPO_ROOT / "docker" / "docker-bake.hcl"
BAKE_ENV = REPO_ROOT / "docker" / "scripts" / "aic-bake-env.sh"
README = REPO_ROOT / "README.md"
REQUIRED_ARGS = (
    "ROCM_VERSION",
    "ROCM_BASE_IMAGE",
    "PYTORCH_REF",
    "PYTORCH_SERIES",
    "VLLM_REF",
    "AITER_REF",
    "LMCACHE_REF",
    "NIXL_REF",
    "HSA_SNOOP_REF",
)


class SyncError(Exception):
    """Raised when source metadata cannot be rendered unambiguously."""

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)
        self.path = path


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def report_error(path: Path, message: str) -> None:
    print(f"ERROR: {relative(path)}: {message}", file=sys.stderr)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        annotation = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error file={relative(path)}::{annotation}", file=sys.stderr)


def bake_values() -> dict[str, str]:
    # The README documents the repo's defaults, not this shell's overrides.
    env = {key: value for key, value in os.environ.items() if key not in REQUIRED_ARGS}
    try:
        result = subprocess.run(
            ["bash", str(BAKE_ENV), "--json"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
    except OSError as error:
        raise SyncError(BAKE_ENV, f"cannot run aic-bake-env.sh: {error}") from error
    if result.returncode != 0:
        raise SyncError(BAKE_FILE, f"aic-bake-env.sh --json failed: {result.stderr.strip()}")
    resolved = json.loads(result.stdout)
    values: dict[str, str] = {}
    for name in REQUIRED_ARGS:
        if not resolved.get(name):
            raise SyncError(BAKE_FILE, f"bake resolved no value for {name}")
        values[name] = resolved[name]
    return values


def ensure_table_safe(label: str, value: str, path: Path = BAKE_FILE) -> None:
    if any(character in value for character in ("`", "|", "\r", "\n", "$")):
        raise SyncError(path, f"{label} contains unsupported README characters: {value!r}")


def shield_value(value: str) -> str:
    # Static badge messages escape their delimiter and underscore before URL encoding.
    escaped = value.replace("-", "--").replace("_", "__")
    return quote(escaped, safe=".")


def patch_files(directory: Path, *, required: bool) -> list[Path]:
    if not directory.is_dir():
        raise SyncError(directory, "patch directory is missing")
    patches = sorted(
        path
        for path in directory.iterdir()
        if not path.name.startswith(".") and path.name.endswith(".patch") and path.is_file()
    )
    if required and not patches:
        raise SyncError(directory, "expected at least one .patch file, found 0")
    return patches


def replace_one(
    text: str,
    pattern: str,
    replacement: str,
    label: str,
    changed: list[str],
) -> str:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if len(matches) != 1:
        raise SyncError(README, f"expected exactly one {label}, found {len(matches)}")
    old = matches[0].group(0)
    if old != replacement:
        changed.append(label)
    return text[: matches[0].start()] + replacement + text[matches[0].end() :]


def render_readme(readme_text: str) -> tuple[str, list[str]]:
    values = bake_values()
    rocm = values["ROCM_VERSION"]
    if not re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?(?:[-+][0-9A-Za-z.]+)?", rocm):
        raise SyncError(BAKE_FILE, f"ROCM_VERSION is not a supported release version: {rocm!r}")

    rocm_base = values["ROCM_BASE_IMAGE"]
    expected_rocm_base = f"rocm/dev-ubuntu-26.04:{rocm}-full"
    if rocm_base != expected_rocm_base:
        raise SyncError(
            BAKE_FILE,
            "ROCM_BASE_IMAGE must match the README Ubuntu 26.04/Python 3.14 contract: "
            f"expected {expected_rocm_base!r}, found {rocm_base!r}",
        )

    for name, value in values.items():
        if name == "ROCM_BASE_IMAGE":
            continue
        ensure_table_safe(name, value)
    ensure_table_safe("ROCM_BASE_IMAGE", rocm_base)

    vllm_patches = patch_files(REPO_ROOT / "patches" / "vllm", required=False)
    lmcache_patches = patch_files(REPO_ROOT / "patches" / "lmcache", required=True)
    nixl_patches = patch_files(REPO_ROOT / "patches" / "nixl", required=True)
    for patch in nixl_patches:
        ensure_table_safe("NIXL patch filename", patch.name, patch)
    nixl_patch_summary = " + ".join(f"`{patch.name}`" for patch in nixl_patches)
    rocm_short = ".".join(rocm.split(".")[:2])
    # PYTORCH_REF is the immutable commit built; PYTORCH_SERIES is the readable
    # series.  Links point at the commit; the shield shows the series only for
    # brevity, while the table row also carries the short SHA for traceability.
    pytorch_ref = values["PYTORCH_REF"]
    pytorch_series = values["PYTORCH_SERIES"]
    pytorch = f"{pytorch_series} ({pytorch_ref[:7]})"  # table row: "2.15 (abecaa8)"
    vllm = values["VLLM_REF"]
    aiter = values["AITER_REF"]
    lmcache = values["LMCACHE_REF"]
    nixl = values["NIXL_REF"]
    hsa = values["HSA_SNOOP_REF"]

    changed: list[str] = []
    updates = (
        (
            "ROCm badge",
            r"^\[!\[ROCm\]\(https://img\.shields\.io/badge/ROCm-[^\s/)]+-green\.svg\)\]\(https://rocm\.docs\.amd\.com\)$",
            f"[![ROCm](https://img.shields.io/badge/ROCm-{shield_value(rocm_short)}-green.svg)](https://rocm.docs.amd.com)",
        ),
        (
            "PyTorch badge",
            r"^\[!\[PyTorch\]\(https://img\.shields\.io/badge/PyTorch-[^\s/)]+-blue\.svg\)\]\(https://github\.com/ROCm/pytorch/tree/[^\s)]+\)$",
            f"[![PyTorch](https://img.shields.io/badge/PyTorch-{shield_value(pytorch_series)}-blue.svg)](https://github.com/ROCm/pytorch/tree/{pytorch_ref})",
        ),
        (
            "AITER badge",
            r"^\[!\[AITER\]\(https://img\.shields\.io/badge/AITER-[^\s/)]+-blue\.svg\)\]\(https://github\.com/ROCm/aiter/tree/[^\s)]+\)$",
            f"[![AITER](https://img.shields.io/badge/AITER-{shield_value(aiter)}-blue.svg)](https://github.com/ROCm/aiter/tree/{aiter})",
        ),
        (
            "vLLM badge",
            r"^\[!\[vLLM\]\(https://img\.shields\.io/badge/vLLM-[^\s/)]+-blue\.svg\)\]\(https://github\.com/vllm-project/vllm\)$",
            f"[![vLLM](https://img.shields.io/badge/vLLM-{shield_value(vllm)}-blue.svg)](https://github.com/vllm-project/vllm)",
        ),
        (
            "LMCache badge",
            r"^\[!\[LMCache\]\(https://img\.shields\.io/badge/LMCache-[^\s/)]+-blue\.svg\)\]\(https://github\.com/LMCache/LMCache\)$",
            f"[![LMCache](https://img.shields.io/badge/LMCache-{shield_value(lmcache)}-blue.svg)](https://github.com/LMCache/LMCache)",
        ),
        (
            "NIXL badge",
            r"^\[!\[NIXL\]\(https://img\.shields\.io/badge/NIXL-[^\s/)]+-blue\.svg\)\]\(https://github\.com/ai-dynamo/nixl\)$",
            f"[![NIXL](https://img.shields.io/badge/NIXL-{shield_value(nixl)}-blue.svg)](https://github.com/ai-dynamo/nixl)",
        ),
        (
            "HSA-Snoop badge",
            r"^\[!\[HSA-Snoop\]\(https://img\.shields\.io/badge/HSA--Snoop-[^\s/)]+-blue\.svg\)\]\(https://github\.com/sbates130272/hsa-snoop\)$",
            f"[![HSA-Snoop](https://img.shields.io/badge/HSA--Snoop-{shield_value(hsa)}-blue.svg)](https://github.com/sbates130272/hsa-snoop)",
        ),
        (
            "base image row",
            r"^\| Base OS \| `[^`]+` \| Ubuntu 26\.04, ROCm [^ |]+, Python 3\.14 \|$",
            f"| Base OS | `{rocm_base}` | Ubuntu 26.04, ROCm {rocm_short}, Python 3.14 |",
        ),
        (
            "PyTorch row",
            r"^\| PyTorch \| `ROCm/pytorch` \(source build\) \| `[^`]+` \|$",
            f"| PyTorch | `ROCm/pytorch` (source build) | `{pytorch}` |",
        ),
        (
            "vLLM row",
            r"^\| vLLM \| `github\.com/vllm-project/vllm` \(source build\) \| `[^`]+` \+ [0-9]+ AMD patches \|$",
            f"| vLLM | `github.com/vllm-project/vllm` (source build) | `{vllm}` + {len(vllm_patches)} AMD patches |",
        ),
        (
            "AITER row",
            r"^\| AITER \| `ROCm/aiter` \(source build\) \| `[^`]+` \|$",
            f"| AITER | `ROCm/aiter` (source build) | `{aiter}` |",
        ),
        (
            "LMCache row",
            r"^\| LMCache \| `LMCache/LMCache` \(upstream\) \| `[^`]+` \+ [0-9]+ AMD patches \|$",
            f"| LMCache | `LMCache/LMCache` (upstream) | `{lmcache}` + {len(lmcache_patches)} AMD patches |",
        ),
        (
            "NIXL row",
            r"^\| NIXL \| `ai-dynamo/nixl` \(upstream\) \| `[^`]+`(?: \+ `[^`]+`)+ \|$",
            f"| NIXL | `ai-dynamo/nixl` (upstream) | `{nixl}` + {nixl_patch_summary} |",
        ),
        (
            "HSA Snoop row",
            r"^\| HSA Snoop \| `sbates130272/hsa-snoop` \(source build\) \| `[^`]+` \|$",
            f"| HSA Snoop | `sbates130272/hsa-snoop` (source build) | `{hsa}` |",
        ),
    )

    rendered = readme_text
    for label, pattern, replacement in updates:
        rendered = replace_one(rendered, pattern, replacement, label, changed)
    return rendered, changed


def check_or_write(write: bool) -> int:
    readme_text = README.read_text(encoding="utf-8")
    rendered, changed = render_readme(readme_text)

    if rendered == readme_text:
        print("README component metadata is in sync.")
        return 0

    if write:
        README.write_text(rendered, encoding="utf-8")
        print(f"Updated README.md fields: {', '.join(changed)}")
        return 0

    report_error(README, f"component metadata is out of sync: {', '.join(changed)}")
    sys.stdout.writelines(
        difflib.unified_diff(
            readme_text.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile="README.md (committed)",
            tofile="README.md (expected)",
        )
    )
    print(
        "Run `python3 .github/scripts/workflows/sync-readme-versions.py --write`, "
        "then commit README.md.",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="fail if README.md is stale")
    mode.add_argument("--write", action="store_true", help="update README.md in place")
    args = parser.parse_args()

    try:
        return check_or_write(write=args.write)
    except (OSError, UnicodeError) as error:
        path = Path(error.filename) if getattr(error, "filename", None) else REPO_ROOT
        report_error(path, str(error))
    except SyncError as error:
        report_error(error.path, str(error))
        if error.path == README:
            print(
                "Restore the named README badge or Stack Overview row, then run "
                "`python3 .github/scripts/workflows/sync-readme-versions.py --write`.",
                file=sys.stderr,
            )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
