<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# aic-drivenets

ROCm **vLLM** + **LMCache** local CPU / NVMe tiers on **gfx950 dGPU**, using
base image **`rocm/vllm:rocm7.13.0_gfx950-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1`**.
Work from **`recipies/aic-drivenets/`**.

Part of [rocm-aic](../../README.md). Unlike [vllm-lmcache-hipfile](../vllm-lmcache-hipfile)
(GdsBackend + hipFile) and [vllm-lmcache-nixl](../vllm-lmcache-nixl) (NIXL
POSIX/AIS), this recipe uses LMCache's **`LMCACHE_LOCAL_*`** environment API
for HBM L1 + CPU DRAM L2 + optional NVMe L3.

## Contents

- [Where things live](#where-things-live)
- [Quick start](#quick-start)
- [LMCache tiers](#lmcache-tiers-ade_lmcache_tier)
- [Benchmark / stimulus](#benchmark--stimulus)
- [Compare to other recipes](#compare-to-other-recipes)

## Where things live

| What you need | File |
| --- | --- |
| **`make build` / `make run` / `make verify`**, mounts, **`ADE_*`** | **`Makefile`** |
| **`rocm/vllm`** gfx950 base + HIP LMCache | **`Dockerfile`** |
| Defaults (prefix cache, LMCache tiers, vLLM serve) | **`configs/aic-drivenets.yaml`** |
| Server entry point | **`scripts/vllm-server`** |
| HIP backend check | **`scripts/verify-lmcache-hip.py`** |

### Base image

Default **`rocm/vllm:rocm7.13.0_gfx950-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1`**.
Override with **`make build VLLM_BASE_TAG=…`**.

**Hardware:** gfx950 dGPU (Strix Halo class). For MI300X / ATOM workloads use
[vllm-atom-andy](../vllm-atom-andy).

## Quick start

```bash
export ROCM_ARCH=gfx950
make -C recipies/aic-drivenets build
make -C recipies/aic-drivenets verify
export HF_TOKEN=your_hf_token_here
make -C recipies/aic-drivenets run
```

The **Makefile** bind-mounts **`configs/`** and **`scripts/`** so YAML and
Python update without **`docker build`**. LMCache NVMe state uses host **`DATA`**
(default **`/data`** → container **`/data`**). Server logs tee to host **`LOG`**
(default **`recipies/aic-drivenets/logs`**, file **`server.txt`**).

Port **`800{GPU}`** matches the first index in **`ROCR_VISIBLE_DEVICES`**.

### DriveNets run flags

**`make run`** passes **`--device=/dev/kfd`**, **`--device=/dev/dri`**,
**`--network=host`**, **`--ipc=host`**, **`--group-add 44`**, **`--group-add 993`**,
**`--cap-add=SYS_PTRACE`**, **`--cap-add=SYS_NICE`**, **`--shm-size=64g`**, and
**`-v /data:/data`**.

Mandatory image env: **`PYTHONHASHSEED=0`**, **`VLLM_FLOAT32_MATMUL_PRECISION=high`**.

## LMCache tiers (`ADE_LMCACHE_TIER`)

| Tier | Env | Behavior |
| --- | --- | --- |
| **`hbm`** | (LMCache env off) | vLLM prefix cache in HBM only |
| **`cpu`** | **`LMCACHE_LOCAL_CPU=true`**, 64 GB L2 default | HBM L1 + CPU DRAM L2 |
| **`cpu-nvme`** | above + disk L3 | adds **`LMCACHE_LOCAL_DISK`**, **`USE_GDS=false`**, **`NUMA_MODE=auto`** |

Default tier: **`cpu-nvme`** (1500 GB disk cap).

Examples:

```bash
make run ADE_LMCACHE_TIER=cpu
make run ADE_LMCACHE_TIER=cpu-nvme DATA=/data
```

## Benchmark / stimulus

Load testing uses **[kv-cache-tester](../../benchmarks/kv-cache-tester/)** (same
harness as the [LMCache MI300X blog][atom-blog]). After the server is up:

```bash
make -C benchmarks/kv-cache-tester install data check-server
make -C benchmarks/kv-cache-tester run BASE_URL=http://127.0.0.1:8000
```

See [benchmarks/kv-cache-tester/README.md](../../benchmarks/kv-cache-tester/README.md).

## Compare to other recipes

| Recipe | LMCache backend | Base image |
| --- | --- | --- |
| vllm-lmcache-hipfile | GdsBackend → hipFile | vllm/vllm-openai-rocm:v0.19.0 |
| vllm-lmcache-nixl | NixlStorageBackend → NIXL | vllm/vllm-openai-rocm:v0.19.0 |
| vllm-atom-andy | **`LMCACHE_LOCAL_*`** + ATOM | rocm/atom-dev |
| **aic-drivenets** | **`LMCACHE_LOCAL_*`** | **rocm/vllm** (gfx950) |

<!-- References -->
[atom-blog]: https://andyluo7.github.io/llm/amd/mi300x/vllm/lmcache/performance/2026/05/22/atom-lmcache-kv-cache-offload-mi300x/
