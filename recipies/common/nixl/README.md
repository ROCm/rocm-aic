<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# NIXL build (ROCm @ pinned SHA + AIS overlay)

Shared clone/build scripts for [vllm-lmcache-nixl](../vllm-lmcache-nixl) and
[rocm-inference-stack](../rocm-inference-stack).

## Source

| Variable | Default |
|----------|---------|
| `NIXL_GIT_URL` | `https://github.com/andyluo7/nixl.git` |
| `NIXL_REF` | `amd-support` (branch; optional if `NIXL_SHA` set) |
| `NIXL_SHA` | pinned in `defaults.mk` (`f72aad2…`, amd-support tip) |

AIS and AIS_MT plugins are copied from `overlay/` at build time
(`apply-ais-overlay.sh` + `patch-ais-meson.py`).

## Meson flags (recipe image)

```bash
meson setup build \
  -Dwheel_variant=rocm \
  -Drocm_path=/opt/rocm \
  -Ducx_path=/opt/rocnixl-ucx \
  -Ddisable_gds_backend=true \
  -Dais_path=/opt/rocm \
  --prefix=/opt/nixl
ninja -C build install
```

## Validation gates

1. `python3 -c "import nixl; nixl.nixl_agent('t')"` with `NIXL_PLUGIN_DIR` set
   and `PYTHONPATH` including `${NIXL_INSTALL_PREFIX}/lib/python3/dist-packages`
   (build script installs a local `nixl` shim over meson’s `nixl_rocm` package)
2. `libplugin_AIS.so` and `libplugin_AIS_MT.so` under the build tree
3. Slurm: `sbatch .slurm/test-nixl-ais-mt.sbatch` on a MARKHAM+NVME node
4. End-to-end: `./run-slurm-nixl.sh` with `VLN_LMCACHE_IO=nixl-posix`

## Local build (host or container)

```bash
export NIXL_GIT_URL=https://github.com/andyluo7/nixl.git
export NIXL_SHA=f72aad2cf4da0dff5d710dfcaa8666defa114d78
./clone-nixl.sh
NIXL_SRC=/tmp/nixl ./build-nixl.sh
```
