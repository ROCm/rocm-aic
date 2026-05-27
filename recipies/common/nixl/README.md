<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# NIXL build (ROCm amd-support + AIS overlay)

Shared clone/build scripts for [vllm-lmcache-nixl](../vllm-lmcache-nixl) and
[rocm-inference-stack](../rocm-inference-stack).

## Source

| Variable | Default |
|----------|---------|
| `NIXL_GIT_URL` | `https://github.com/andyluo7/nixl.git` |
| `NIXL_REF` | `amd-support` |
| `NIXL_AMD_SUPPORT_SHA` | pinned in `defaults.mk` |

AIS and AIS_MT plugins are copied from `overlay/` at build time
(`apply-ais-overlay.sh` + `patch-ais-meson.py`).

## Meson flags (recipe image)

```bash
meson setup build \
  -Duse_rocm=/opt/rocm \
  -Ducx_path=/opt/rocnixl-ucx \
  -Ddisable_gds_backend=true \
  -Dais_path=/opt/rocm \
  --prefix=/opt/nixl
ninja -C build install
```

## Validation gates

1. `python3 -c "import nixl; nixl.nixl_agent('t')"` with `NIXL_PLUGIN_DIR` set
2. `libplugin_AIS.so` and `libplugin_AIS_MT.so` under the build tree
3. Slurm: `sbatch .slurm/test-nixl-ais-mt.sbatch` on a MARKHAM+NVME node
4. End-to-end: `./run-slurm-nixl.sh` with `VLN_LMCACHE_IO=nixl-posix`

## Local build (host or container)

```bash
export NIXL_GIT_URL=https://github.com/andyluo7/nixl.git
export NIXL_REF=amd-support
./clone-nixl.sh
NIXL_SRC=/tmp/nixl ./build-nixl.sh
```
