<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# vllm-lmcache-nixl

ROCm **vLLM** + **LMCache** + **NIXL** (andyluo7/nixl `amd-support` + AIS/hipfile
overlay). Uses LMCache **`enable_nixl_storage`** with **POSIX** (MVP) or **AIS**
backends on AMD Instinct GPUs.

Part of [rocm-aic](../../README.md). Gutenberg benchmarks live in
[benchmarks/llm-prefill-benchmark](../../benchmarks/llm-prefill-benchmark).

## Quick start

```bash
export ROCM_ARCH=gfx942
make -C recipies/vllm-lmcache-nixl build
export HF_TOKEN=...
make -C recipies/vllm-lmcache-nixl run VLN_LMCACHE_IO=nixl-posix
```

## Storage modes (`VLN_LMCACHE_IO`)

| Value | LMCache / NIXL backend |
|-------|-------------------------|
| `nixl-posix` (default) | NIXL POSIX on local NVMe |
| `ais` | NIXL AIS + hipFile |
| `ais_mt` | NIXL AIS_MT + hipFile |

## Build arguments

| ARG | Default |
|-----|---------|
| `NIXL_GIT_URL` | `https://github.com/andyluo7/nixl.git` |
| `NIXL_REF` | `amd-support` |
| `HIPFILE_SHA` | same as hipfile recipe |
| `LMCACHE_SHA` | pinned in Dockerfile |

NIXL is built via [recipies/common/nixl](../common/nixl) with the AIS plugin
overlay applied at image build time.

## Slurm

From repo root:

```bash
./run-slurm-nixl.sh
```

## Compare to vllm-lmcache-hipfile

| Recipe | KV disk path |
|--------|----------------|
| vllm-lmcache-hipfile | LMCache GdsBackend → hipFile directly |
| vllm-lmcache-nixl | LMCache NixlStorageBackend → NIXL POSIX/AIS |

Both use the same [llm-prefill-benchmark](../../benchmarks/llm-prefill-benchmark)
Gutenberg workload for TTFT measurement.

## NIXL pool size (`VLN_NIXL_POOL_SIZE`)

The NIXL backend pre-allocates a fixed pool of ``obj_*.bin`` slots under
``$DATA/lmcache/``. Default **`nixl_pool_size: 4096`** in the LMCache YAML
(~36 GiB for Qwen2.5-3B at ~9 MiB/slot). Override at run time:

```bash
make run VLN_NIXL_POOL_SIZE=8192   # ~72 GiB pool (8192 × ~9 MiB)
```

Restart after changing pool size (``clear_gds_dir_before_start`` recreates the
pool). First start with a large pool can take a minute while slots allocate.

**File descriptor limit:** NIXL opens one FD per pool slot during init. Docker
defaults to ~65536; a pool of **131072** fails with ``Too many open files``
unless ``make run`` passes a higher limit (``VLN_DOCKER_NOFILE``, default
**1048576**). Rule of thumb: ``VLN_NIXL_POOL_SIZE + 5000 < VLN_DOCKER_NOFILE``.
