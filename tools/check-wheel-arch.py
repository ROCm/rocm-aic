#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""Report which GPU architectures a built wheel actually carries code for.

The AIC wheel set is arch-specific but the filenames are not: every arch group
publishes a byte-identical NAME (``vllm-0.29.0+rocm-cp312-cp312-linux_x86_64``)
holding different device code.  Installing the wrong one imports cleanly and
then fails at the first kernel launch -- the exact trap docs/PIP_WHEELS.md
warns about.

This scans the ELF payloads inside a wheel for ``gfx`` target strings so CI can
assert the arch BEFORE spending an hour booting a VM to find out.

Usage:
    python3 -I tools/check-wheel-arch.py <wheel> [<wheel> ...] [--expect gfx1250]

Exit status is non-zero when --expect is given and any wheel carrying device
code lacks that target.  Pure-Python wheels (no ELF payload) are reported as
``none`` and never fail the check.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile

# Match the clang OFFLOAD BUNDLE ENTRY ID, not a bare "gfxNNNN".
#
# A bare gfx match is worthless here and was measured to be: the CDNA-only wheel
# set reports gfx1250 too, because every ROCm .so links LLVM target tables that
# name every architecture the compiler knows about.  The bundle entry ID is
# emitted only where real device code is embedded:
#
#     __CLANG_OFFLOAD_BUNDLE__ ... hipv4-amdgcn-amd-amdhsa--gfx942
#
# so this distinguishes "has kernels for gfx1250" from "mentions gfx1250".
_GFX = re.compile(rb"amdgcn-amd-amdhsa--(?:gfx[0-9]{3,4}[a-z]?)")
_GFX_NAME = re.compile(rb"gfx[0-9]{3,4}[a-z]?")

# Only these members can hold device code; scanning the whole archive on a
# 389 MB torch wheel is needlessly slow.
_BINARY_SUFFIXES = (".so", ".hsaco", ".a", ".bc", ".co")


def _is_binary_member(name: str) -> bool:
    if name.endswith(_BINARY_SUFFIXES):
        return True
    # Versioned shared objects: libfoo.so.1.2
    return ".so." in name


def wheel_targets(path: str) -> set[str]:
    """Return the set of gfx targets referenced by ELF payloads in `path`."""
    found: set[str] = set()
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not _is_binary_member(name):
                continue
            try:
                blob = zf.read(name)
            except (KeyError, zipfile.BadZipFile, OSError) as exc:
                print(f"  warning: could not read {name}: {exc}", file=sys.stderr)
                continue
            for m in _GFX.findall(blob):
                name = _GFX_NAME.search(m)
                if name:
                    found.add(name.group().decode())
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wheels", nargs="+", help="wheel files to inspect")
    ap.add_argument(
        "--expect",
        metavar="GFX",
        help="require this target in every wheel that carries device code",
    )
    args = ap.parse_args(argv)

    failed = False
    # Count wheels that positively carry the expected target.  Needed because
    # "no device code found" is NOT proof of a pure-Python wheel: torch stores
    # its device images in COMPRESSED offload bundles (CCOB), which this scanner
    # cannot read, so torch always reports `none` even when it is full of
    # kernels.  Treating `none` as a pass is therefore the only safe per-wheel
    # rule -- but it means an empty, truncated or entirely wrong-arch set would
    # sail through silently.  Requiring at least one POSITIVE match closes that.
    matched_expected = 0
    for path in args.wheels:
        try:
            targets = wheel_targets(path)
        except (zipfile.BadZipFile, OSError) as exc:
            print(f"{path}: ERROR: not a readable wheel ({exc})", file=sys.stderr)
            failed = True
            continue

        name = path.rsplit("/", 1)[-1]
        if not targets:
            print(f"  {name:<70} none (pure-Python or no device code)")
            continue

        print(f"  {name:<70} {','.join(sorted(targets))}")
        if args.expect:
            if args.expect in targets:
                matched_expected += 1
            else:
                print(
                    f"ERROR: {name} carries device code but not for "
                    f"{args.expect}; found {sorted(targets)}",
                    file=sys.stderr,
                )
                failed = True

    if args.expect and matched_expected == 0:
        print(
            f"ERROR: no wheel in this set positively carries {args.expect} "
            "device code.  Every wheel reported 'none', which means the set is "
            "empty, truncated, or pure-Python only -- not that it matches.",
            file=sys.stderr,
        )
        failed = True

    if failed:
        print(
            "check-wheel-arch: FAILED -- this wheel set does not match the "
            "requested architecture",
            file=sys.stderr,
        )
        return 1
    print(f"check-wheel-arch: OK{f' ({args.expect})' if args.expect else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
