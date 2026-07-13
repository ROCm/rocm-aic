#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Inject ROCm/HIP toolchain support into andyluo7/nixl amd-support (post use_rocm).

Upstream f72aad2+ uses -Dwheel_variant=rocm for Python wheel naming only. AIS/hipFile
plugins still need hipcc, HIP link flags, and a found-but-empty cuda_dep shim.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROCM_OPTIONS = """option('rocm_path', type: 'string', value: '', description: 'Path to ROCm install root. Non-empty enables HIP toolchain (pair with -Dwheel_variant=rocm).')
"""

CUDA_BLOCK_OLD = """cuda_inc_path = get_option('cudapath_inc')
cuda_lib_path = get_option('cudapath_lib')
cuda_stub_path = get_option('cudapath_stub')

if cuda_lib_path == ''"""

CUDA_BLOCK_NEW = """rocm_path = get_option('rocm_path')
use_rocm = rocm_path != ''

cuda_inc_path = get_option('cudapath_inc')
cuda_lib_path = get_option('cudapath_lib')
cuda_stub_path = get_option('cudapath_stub')

if use_rocm
    hipcc_prog = find_program(rocm_path + '/bin/hipcc', required: false)
    if not hipcc_prog.found()
        error('rocm_path set but hipcc not found at ' + rocm_path + '/bin/hipcc. ' +
              'Install ROCm or pass -Drocm_path=<prefix>.')
    endif

    rocm_inc = rocm_path + '/include'
    rocm_lib = rocm_path + '/lib'
    add_project_arguments('-D__HIP_PLATFORM_AMD__', language: 'cpp')
    add_project_link_arguments(['-L' + rocm_lib, '-lamdhip64', '-lhiprtc'], language: 'cpp')

    # Empty-but-found cuda_dep lets HIP plugins list it defensively (UCX, AIS).
    cuda_dep = declare_dependency()
    summary({'GPU vendor': 'AMD ROCm', 'ROCm path': rocm_path}, section: 'Build')
elif cuda_lib_path == ''"""

CUDA_FOUND_OLD = "if cuda_dep.found()\n    add_languages('CUDA')"
CUDA_FOUND_NEW = "if cuda_dep.found() and not use_rocm\n    add_languages('CUDA')"

CUDA_ELSE_OLD = """else
    warning('CUDA not found. UCX backend will be built without CUDA support, and some plugins will be disabled.')
    doca_gpunetio_dep = disabler()
    cuda_wheel_dir = 'nixl_cu12'
endif

# Allow callers (e.g. ROCm CI/Dockerfile) to override the wheel variant suffix"""

CUDA_ELSE_NEW = """elif use_rocm
    doca_gpunetio_dep = disabler()
else
    warning('CUDA not found. UCX backend will be built without CUDA support, and some plugins will be disabled.')
    doca_gpunetio_dep = disabler()
    cuda_wheel_dir = 'nixl_cu12'
endif

# Allow callers (e.g. ROCm CI/Dockerfile) to override the wheel variant suffix"""


def main() -> int:
    root = Path(os.environ.get("NIXL_SRC", "/tmp/nixl"))
    meson = root / "meson.build"
    opts = root / "meson_options.txt"

    if not meson.is_file():
        print(f"ERROR: missing {meson}", file=sys.stderr)
        return 1

    text = meson.read_text(encoding="utf-8")
    if "rocm_path = get_option('rocm_path')" not in text:
        if CUDA_BLOCK_OLD not in text:
            print("ERROR: meson.build CUDA block anchor not found", file=sys.stderr)
            return 1
        text = text.replace(CUDA_BLOCK_OLD, CUDA_BLOCK_NEW, 1)
        if CUDA_FOUND_OLD not in text:
            print("ERROR: meson.build cuda_dep.found anchor not found", file=sys.stderr)
            return 1
        text = text.replace(CUDA_FOUND_OLD, CUDA_FOUND_NEW, 1)
        if CUDA_ELSE_OLD not in text:
            print("ERROR: meson.build CUDA else anchor not found", file=sys.stderr)
            return 1
        text = text.replace(CUDA_ELSE_OLD, CUDA_ELSE_NEW, 1)
        meson.write_text(text, encoding="utf-8")
        print("Updated meson.build for rocm_path / HIP toolchain")
    else:
        print("meson.build already has rocm_path; skipping")

    otext = opts.read_text(encoding="utf-8")
    if "option('rocm_path'" not in otext:
        needle = (
            "option('wheel_variant', type: 'string', value: '', "
            "description: 'Override wheel variant suffix"
        )
        idx = otext.find(needle)
        if idx < 0:
            print("ERROR: meson_options wheel_variant anchor not found", file=sys.stderr)
            return 1
        line_end = otext.find("\n", idx)
        if line_end < 0:
            print("ERROR: malformed meson_options.txt", file=sys.stderr)
            return 1
        opts.write_text(
            otext[: line_end + 1] + ROCM_OPTIONS + otext[line_end + 1 :],
            encoding="utf-8",
        )
        print("Updated meson_options.txt with rocm_path")
    else:
        print("meson_options.txt already has rocm_path; skipping")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
