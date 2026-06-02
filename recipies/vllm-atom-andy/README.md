<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# vllm-atom-andy

ROCm **vLLM** + **ATOM** (AiTer Optimized Model plugin) + **LMCache**
local CPU / NVMe tiers, following [Andy Luo's ATOM + LMCache MI300X blog][atom-blog].
Base image **`rocm/atom-dev:vllm-v0.19.0-nightly_20260522`**, pinned by digest in
**`Dockerfile`** (not **`vllm-latest`**). Work from **`recipies/vllm-atom-andy/`**.

Part of [rocm-aic](../../README.md). Unlike [vllm-lmcache-hipfile](../vllm-lmcache-hipfile)
(GdsBackend + hipFile) and [vllm-lmcache-nixl](../vllm-lmcache-nixl) (NIXL
POSIX/AIS), this recipe uses LMCache's **`LMCACHE_LOCAL_*`** environment API
for HBM L1 + CPU DRAM L2 + optional NVMe L3.

## Contents

- [Where things live](#where-things-live)
- [Quick start](#quick-start)
- [LMCache tiers](#lmcache-tiers-vaa_lmcache_tier)
- [Critical pitfalls](#critical-pitfalls)
- [Blog reproduction (2× MI300X, MiniMax-M2.5)](#blog-reproduction-2-mi300x-minimax-m25)
- [kv-cache-tester / trace replay](#kv-cache-tester--trace-replay)
- [Compare to other recipes](#compare-to-other-recipes)

## Where things live

| What you need | File |
| --- | --- |
| **`make build` / `make run` / `make verify`**, mounts, **`VAA_*`** | **`Makefile`** |
| **`rocm/atom-dev@sha256:…`** (20260522 nightly) + HIP LMCache + **`aiofile`** | **`Dockerfile`** |
| Defaults (FP8 KV, CUDA graphs, ATOM env, tiers) | **`configs/vllm-atom-andy.yaml`** |
| Server entrypoint | **`scripts/vllm-server`** |
| HIP backend check (blog Step 3) | **`scripts/verify-lmcache-hip.py`** |

### Base image

Default **`rocm/atom-dev:vllm-v0.19.0-nightly_20260522`**, referenced by digest
in **`Dockerfile`** and **`Makefile`** **`ATOM_DEV_DIGEST`**. Override with
**`make build ATOM_DEV_DIGEST=sha256:…`** after verifying a newer dated nightly on
Docker Hub.

**Hardware:** ATOM targets **AMD Instinct MI300X (gfx942)**. Radeon GPUs (e.g.
gfx1201) are not supported by the **`rocm/atom-dev`** stack; use
[vllm-lmcache-hipfile](../vllm-lmcache-hipfile) on Radeon instead.

## Quick start

```bash
export ROCM_ARCH=gfx942   # MI300X; required for LMCache HIP build
make -C recipies/vllm-atom-andy build
make -C recipies/vllm-atom-andy verify
export HF_TOKEN=your_hf_token_here
make -C recipies/vllm-atom-andy run
```

The **Makefile** bind-mounts **`configs/`** and **`scripts/`** so YAML and
Python update without **`docker build`**. LMCache NVMe state uses host **`DATA`**
(default **`/mnt/lmcache-nvme`** → container **`/data`**). Server logs tee to
host **`LOG`** (default **`recipies/vllm-atom-andy/logs`**, file **`server.txt`**).

Port **`800{GPU}`** matches the first index in **`ROCR_VISIBLE_DEVICES`** (e.g.
**`8000`** for **`GPU=0`**). Override **`CONTAINER_NAME`**, **`DATA`**, **`LOG`**
via **`make run`** variables (see **`make help`**).

Prepare the host **`DATA`** path before **`cpu-nvme`**. Prefer stable NVMe
identity under **`/dev/disk/by-id/`** when formatting or mounting test SSDs;
do not rely on volatile **`/dev/nvme*n*`** namespace paths in docs or scripts.

## LMCache tiers (`VAA_LMCACHE_TIER`)

| Tier | Env | Behavior |
| --- | --- | --- |
| **`hbm`** | (default off LMCache env) | ATOM FP8 prefix cache in HBM only |
| **`cpu`** | **`LMCACHE_LOCAL_CPU=true`**, 64 GB L2 default | HBM L1 + CPU DRAM L2 |
| **`cpu-nvme`** | above + **`LMCACHE_LOCAL_DISK=/data/lmcache`** | adds NVMe L3 spill |

Examples:

```bash
make run VAA_LMCACHE_TIER=hbm
make run VAA_LMCACHE_TIER=cpu
make run VAA_LMCACHE_TIER=cpu-nvme DATA=/mnt/nvme/lmcache
```

### When to use which tier (from blog)

| Scenario | Recommended |
| --- | --- |
| Low concurrency, short context (&lt;32K) | **`hbm`** |
| Moderate concurrency, mixed context | **`cpu`** |
| High concurrency, long context (100K+), SLO on p95 | **`cpu-nvme`** |
| Decode-bound (short input, long output) | **`hbm`** — cache won't help decode |

## Critical pitfalls

1. **CUDA graphs** — ATOM defaults to eager mode. This recipe always passes
   **`--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'`**. Do not
   set **`VAA_ENFORCE_EAGER`** (the server rejects it). Without graphs, expect
   3–5× throughput loss.
2. **LMCache HIP build** — PyPI **`pip install lmcache`** on ROCm silently falls
   back to Python **`non_cuda_equivalents`**. Run **`make verify`** after build;
   **`c_ops`** must be a **`.so`**, not **`.py`**.
3. **`PYTHONHASHSEED=0`** — required for LMCache cache-key consistency across TP
   workers (set in the image and server).
4. **`aiofile`** — required for the NVMe disk tier; installed in the Dockerfile.
   Without it, disk I/O may block or fail.

## Blog reproduction (2× MI300X, MiniMax-M2.5)

Defaults are **TP=1**, **`openai/gpt-oss-120b`**, GMU **0.90**. To match the blog
stress arm on **2× MI300X** with **MiniMax-M2.5**:

```bash
make -C recipies/vllm-atom-andy run \
  GPU=0,1 \
  VAA_TENSOR_PARALLEL_SIZE=2 \
  VLLM_MODEL=/path/to/MiniMax-M2.5 \
  VAA_MODEL_PROFILE=minimax \
  VAA_GPU_MEMORY_UTILIZATION=0.78 \
  VAA_MAX_MODEL_LEN=100000 \
  VAA_LMCACHE_TIER=cpu-nvme \
  DATA=/mnt/nvme/lmcache
```

Mount model weights with **`EXTRA_DOCKER_RUN_FLAGS`**, e.g.
**`-v /mnt/nvme/models:/work/models`**, and set **`VLLM_MODEL=/work/models/MiniMax-M2.5`**
if serving from that path.

## kv-cache-tester / trace replay

The blog uses **`trace_replay_tester.py`** from
[callanjfox/kv-cache-tester][kv-cache-tester] (739 Claude Code traces). This repo
ships an Ansible role at
[ansible/roles/kv_cache_tester](../../ansible/roles/kv_cache_tester).

Stress run parameters from the blog (after the server is up on port 8000):

```bash
ansible-playbook ansible/playbooks/kv-cache-tester.yml \
  -e kv_cache_tester_api_endpoint=http://127.0.0.1:8000 \
  -e kv_cache_tester_script=trace_replay_tester.py \
  -e 'kv_cache_tester_extra_args=["--trace-directory","traces","--start-users","4","--max-users","32","--max-ttft","60.0","--test-duration","1200","--max-context","100000","--warm-prefix-pct","0.5","--timing-strategy","think-only","--recycle","--seed","42"]'
```

Traces must exist under the kv-cache-tester checkout (**`traces/`**). Clone with
**`git clone --recursive`** per upstream instructions.

## Compare to other recipes

| Recipe | LMCache backend | Base image |
| --- | --- | --- |
| vllm-lmcache-hipfile | GdsBackend → hipFile | vllm/vllm-openai-rocm:v0.19.0 |
| vllm-lmcache-nixl | NixlStorageBackend → NIXL POSIX/AIS | vllm/vllm-openai-rocm:v0.19.0 |
| **vllm-atom-andy** | **`LMCACHE_LOCAL_CPU` / `LMCACHE_LOCAL_DISK`** | **rocm/atom-dev** (digest-pinned) |

## Future work

Slurm sbatch integration, GitHub Actions workflows, and Grafana panels are not
included in v1; use [vllm-lmcache-hipfile](../vllm-lmcache-hipfile) for cluster
automation patterns.

<!-- References -->
[atom-blog]: https://andyluo7.github.io/llm/amd/mi300x/vllm/lmcache/performance/2026/05/22/atom-lmcache-kv-cache-offload-mi300x/
[kv-cache-tester]: https://github.com/callanjfox/kv-cache-tester
