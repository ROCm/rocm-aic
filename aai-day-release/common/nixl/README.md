<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# NIXL build (ROCm, ai-dynamo upstream + AIS_MT patch)

Clone + build scripts for the NIXL step of the aai-day-release image.

## Source

NIXL is built from upstream `ai-dynamo/nixl` at the pinned `NIXL_SHA`. The
Dockerfile applies `patches/nixl/nixl-rocm-ais-mt.patch` to the fresh checkout
before invoking `build-nixl.sh`; that patch adds ROCm/HIP detection and the
`AIS_MT` (hipFile / AMD Infinity Storage) plugin tree, turning the checkout into
a native-HIP tree. `build-nixl.sh` then builds it via its `_NATIVE_HIP` path.

There is no plain single-threaded `AIS` plugin — the multi-threaded `AIS_MT`
backend is the only hipFile backend, and it is mandatory.

## Meson flags (recipe image)

```bash
meson setup build \
  -Dwheel_variant=rocm \
  -Drocm_ais_path=/opt/rocm \
  -Ducx_path=/opt/rocnixl-ucx \
  -Ddisable_gds_backend=true \
  --prefix=/opt/nixl
ninja -C build install
```

## Validation gates

1. `python3 -c "import nixl; print(nixl.__file__)"` with `NIXL_PLUGIN_DIR` set
   and `PYTHONPATH` including `${NIXL_INSTALL_PREFIX}/lib/python3/dist-packages`
   (build script installs a local `nixl` shim over meson's `nixl_rocm` package).
2. `libplugin_AIS_MT.so` present under the build tree (build fails otherwise).

## Local build (host or container)

```bash
export NIXL_GIT_URL=https://github.com/ai-dynamo/nixl.git
export NIXL_SHA=644facf0eb3de14ec63c1d2831238f63cd03c0e0
./clone-nixl.sh
git -C /tmp/nixl apply /path/to/patches/nixl/nixl-rocm-ais-mt.patch
NIXL_SRC=/tmp/nixl AIS_PATH=/opt/rocm ./build-nixl.sh
```
