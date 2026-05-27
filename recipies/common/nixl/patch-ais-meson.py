#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Inject AIS/AIS_MT into andyluo7/nixl amd-support meson files."""
from __future__ import annotations

import os
import sys
from pathlib import Path

AIS_PLUGINS_BLOCK = """
disable_ais_backend = get_option('disable_ais_backend')
if enabled_plugins.get('AIS')
    if (disable_ais_backend or not use_rocm or not cuda_dep.found()) and is_explicit_enable
        if disable_ais_backend
            error('AIS plugin requested but AIS backend is disabled')
        elif not use_rocm
            error('AIS plugin requested but ROCm is not enabled (use_rocm empty)')
        else
            error('AIS plugin requested but ROCm dependency not found')
        endif
    elif not disable_ais_backend and use_rocm and cuda_dep.found()
        subdir('ais')
    endif
endif

if enabled_plugins.get('AIS_MT')
    if (disable_ais_backend or not use_rocm or not cuda_dep.found() or not taskflow_proj.found()) and is_explicit_enable
        if disable_ais_backend
            error('AIS_MT plugin requested but AIS backend is disabled')
        elif not use_rocm
            error('AIS_MT plugin requested but ROCm is not enabled (use_rocm empty)')
        elif not cuda_dep.found()
            error('AIS_MT plugin requested but ROCm dependency not found')
        else
            error('AIS_MT plugin requested but Taskflow dependency not found')
        endif
    elif not disable_ais_backend and use_rocm and cuda_dep.found() and taskflow_proj.found()
        subdir('ais_mt')
    endif
endif
"""

MESON_OPTIONS_LINES = """option('ais_path', type: 'string', value: '', description: 'Path to hipFile (AIS) install')
option('disable_ais_backend', type: 'boolean', value: false, description: 'disable AIS (hipFile) backend')
"""


def main() -> int:
    root = Path(os.environ.get("NIXL_SRC", "/tmp/nixl"))
    meson = root / "meson.build"
    opts = root / "meson_options.txt"
    plugins = root / "src/plugins/meson.build"

    if not meson.is_file():
        print(f"ERROR: missing {meson}", file=sys.stderr)
        return 1

    text = meson.read_text(encoding="utf-8")
    old = "all_plugins = ['UCX', 'LIBFABRIC', 'POSIX', 'OBJ', 'GDS', 'GDS_MT', 'MOONCAKE'"
    new = "all_plugins = ['UCX', 'LIBFABRIC', 'POSIX', 'OBJ', 'GDS', 'GDS_MT', 'AIS', 'AIS_MT', 'MOONCAKE'"
    if "'AIS'" not in text:
        if old not in text:
            print("ERROR: meson.build all_plugins line not found", file=sys.stderr)
            return 1
        meson.write_text(text.replace(old, new, 1), encoding="utf-8")
        print("Updated meson.build all_plugins")

    otext = opts.read_text(encoding="utf-8")
    if "disable_ais_backend" not in otext:
        needle = "option('static_plugins', type: 'string', value: '', description: 'Plugins to be built-in, comma-separated')\n"
        if needle not in otext:
            print("ERROR: meson_options anchor not found", file=sys.stderr)
            return 1
        opts.write_text(
            otext.replace(needle, MESON_OPTIONS_LINES + needle, 1),
            encoding="utf-8",
        )
        print("Updated meson_options.txt")

    ptext = plugins.read_text(encoding="utf-8")
    if "subdir('ais')" not in ptext:
        anchor = "endif\n\ncc = meson.get_compiler('cpp')"
        if anchor not in ptext:
            print("ERROR: plugins/meson.build anchor not found", file=sys.stderr)
            return 1
        plugins.write_text(
            ptext.replace(anchor, "endif\n" + AIS_PLUGINS_BLOCK + "\ncc = meson.get_compiler('cpp')", 1),
            encoding="utf-8",
        )
        print("Updated src/plugins/meson.build")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
