#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Keep README component metadata aligned with the source-built stack."""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
README = REPO_ROOT / "README.md"
REQUIRED_ARGS = (
    "ROCM_VERSION",
    "ROCM_BASE_IMAGE",
    "VLLM_REF",
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


def docker_arg_defaults(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in REQUIRED_ARGS:
        pattern = re.compile(
            rf"^[ \t]*(?i:ARG)[ \t]+{re.escape(name)}=(.*)$",
            re.MULTILINE,
        )
        defaults = [match.group(1).strip() for match in pattern.finditer(text)]
        if len(defaults) != 1 or not defaults[0]:
            nonempty = sum(bool(value) for value in defaults)
            raise SyncError(
                DOCKERFILE,
                f"expected exactly one non-empty ARG {name}= default; "
                f"found {len(defaults)} default(s), {nonempty} non-empty",
            )
        if defaults[0].endswith(("\\", "`")):
            raise SyncError(
                DOCKERFILE,
                f"ARG {name}= must use a single-line default; continuations are unsupported",
            )
        values[name] = defaults[0]
    return values


def ensure_table_safe(label: str, value: str, path: Path = DOCKERFILE) -> None:
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


def render_readme(dockerfile_text: str, readme_text: str) -> tuple[str, list[str]]:
    values = docker_arg_defaults(dockerfile_text)
    rocm = values["ROCM_VERSION"]
    if not re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?(?:[-+][0-9A-Za-z.]+)?", rocm):
        raise SyncError(DOCKERFILE, f"ROCM_VERSION is not a supported release version: {rocm!r}")

    rocm_base = values["ROCM_BASE_IMAGE"].replace("${ROCM_VERSION}", rocm)
    rocm_base = re.sub(r"\$ROCM_VERSION(?![A-Za-z0-9_])", rocm, rocm_base)
    if "$" in rocm_base:
        raise SyncError(DOCKERFILE, f"ROCM_BASE_IMAGE contains an unresolved variable: {rocm_base!r}")
    expected_rocm_base = f"rocm/dev-ubuntu-24.04:{rocm}-full"
    if rocm_base != expected_rocm_base:
        raise SyncError(
            DOCKERFILE,
            "ROCM_BASE_IMAGE must match the README Ubuntu 24.04/Python 3.12 contract: "
            f"expected {expected_rocm_base!r}, found {rocm_base!r}",
        )

    for name, value in values.items():
        if name == "ROCM_BASE_IMAGE":
            continue
        ensure_table_safe(name, value)
    ensure_table_safe("resolved ROCM_BASE_IMAGE", rocm_base)

    vllm_patches = patch_files(REPO_ROOT / "patches" / "vllm", required=False)
    lmcache_patches = patch_files(REPO_ROOT / "patches" / "lmcache", required=True)
    nixl_patches = patch_files(REPO_ROOT / "patches" / "nixl", required=True)
    for patch in nixl_patches:
        ensure_table_safe("NIXL patch filename", patch.name, patch)
    nixl_patch_summary = " + ".join(f"`{patch.name}`" for patch in nixl_patches)
    rocm_short = ".".join(rocm.split(".")[:2])
    vllm = values["VLLM_REF"]
    lmcache = values["LMCACHE_REF"]
    nixl = values["NIXL_REF"]
    hsa = values["HSA_SNOOP_REF"]

    changed: list[str] = []
    updates = (
        (
            "ROCm badge",
            r"^\[!\[ROCm\]\(https://img\.shields\.io/badge/ROCm-[^\s/)]+-green\.svg\)\]\(https://rocm\.docs\.amd\.com\)$",
            f"[![ROCm](https://img.shields.io/badge/ROCm-{shield_value(rocm)}-green.svg)](https://rocm.docs.amd.com)",
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
            "hsa-snoop badge",
            r"^\[!\[hsa-snoop\]\(https://img\.shields\.io/badge/hsa--snoop-[^\s/)]+-blue\.svg\)\]\(https://github\.com/sbates130272/hsa-snoop\)$",
            f"[![hsa-snoop](https://img.shields.io/badge/hsa--snoop-{shield_value(hsa)}-blue.svg)](https://github.com/sbates130272/hsa-snoop)",
        ),
        (
            "base image row",
            r"^\| Base OS \| `[^`]+` \| Ubuntu 24\.04, ROCm [^ |]+, Python 3\.12 \|$",
            f"| Base OS | `{rocm_base}` | Ubuntu 24.04, ROCm {rocm_short}, Python 3.12 |",
        ),
        (
            "vLLM row",
            r"^\| vLLM \| `github\.com/vllm-project/vllm` \(source build\) \| `[^`]+` \+ [0-9]+ AMD patches \|$",
            f"| vLLM | `github.com/vllm-project/vllm` (source build) | `{vllm}` + {len(vllm_patches)} AMD patches |",
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
            "hsa-snoop row",
            r"^\| hsa-snoop \| `sbates130272/hsa-snoop` \(source build\) \| `[^`]+` \|$",
            f"| hsa-snoop | `sbates130272/hsa-snoop` (source build) | `{hsa}` |",
        ),
        (
            "hipFile row",
            r"^\| hipFile \| ROCm [^ |]+ base image \| GA in ROCm 7\.14 \N{EM DASH} no separate source build \|$",
            f"| hipFile | ROCm {rocm_short} base image | GA in ROCm 7.14 \N{EM DASH} no separate source build |",
        ),
    )

    rendered = readme_text
    for label, pattern, replacement in updates:
        rendered = replace_one(rendered, pattern, replacement, label, changed)
    return rendered, changed


def check_or_write(write: bool) -> int:
    dockerfile_text = DOCKERFILE.read_text(encoding="utf-8")
    readme_text = README.read_text(encoding="utf-8")
    rendered, changed = render_readme(dockerfile_text, readme_text)

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
