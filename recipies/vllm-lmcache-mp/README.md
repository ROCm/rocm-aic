<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# vllm-lmcache-mp

ROCm **vLLM** + standalone **LMCache server** (MP mode) as two independent
containers, connected over ZMQ on a dedicated Docker bridge network.

Each container can be started, stopped, updated, and restarted independently.
The lmcache server survives vLLM restarts, preserving warm KV state.

| Container | Role                                                        |
|-----------|-------------------------------------------------------------|
| lmcache   | Standalone cache server; L1 GPU/CPU memory + L2 storage     |
| vllm      | Inference engine; connects to lmcache over ZMQ              |

Storage tiers (all owned by the lmcache container):

| Tier          | Backend           | Transport                                        |
|---------------|-------------------|--------------------------------------------------|
| L1 (default)  | GPU / CPU memory  | in-process (lmcache managed)                     |
| L1 (opt-in)   | hipFile GDS slab  | hipFile P2PDMA to dedicated NVMe (GDS_SLAB_DATA) |
| L2a (NVMe)    | nixl_store AIS_MT | hipFile P2PDMA to local NVMe                     |
| L2b (NFS)     | nixl_store POSIX  | NFS-over-RDMA (host kernel NFS client)           |

Part of [rocm-aic](../../README.md).

## Architecture

```text
  ┌──────────────────────────────────────┐
  │  vllm container                      │
  │  vllm serve                          │
  │  --kv-transfer-config                │
  │    LMCacheMPConnector                │
  │    host=lmcache port=6555            │
  └──────────────┬───────────────────────┘
                 │ ZMQ (lmcache-net bridge)
  ┌──────────────┴───────────────────────┐
  │  lmcache container                   │
  │  lmcache server --host 0.0.0.0       │
  │                                      │
  │  L1: GPU/CPU memory (--l1-size-gb)   │
  │  L2a: nixl_store AIS_MT             │
  │       → /data/nvme  (local NVMe)    │
  │  L2b: nixl_store POSIX              │
  │       → /data/nfs   (NFS-over-RDMA) │
  └──────────────────────────────────────┘
```

## Prerequisites

### 1. NFS-over-RDMA mount

Mount the NFS export on the host before `make up`. The container bind-mounts
`NFS_DATA` as `/data/nfs`.

```bash
sudo modprobe rdma_rxe                        # software RoCE if needed
sudo mount -t nfs -o rdma,port=20049 \
    nfs-server:/exports/lmcache /mnt/lmcache-nfs
export NFS_DATA=/mnt/lmcache-nfs
```

### 2. Local NVMe mount

```bash
sudo mkdir -p /mnt/lmcache-nvme && sudo chown $USER /mnt/lmcache-nvme
export NVME_DATA=/mnt/lmcache-nvme
```

## Quick start

```bash
export ROCM_ARCH=gfx942
export HF_TOKEN=hf_...
export NVME_DATA=/mnt/lmcache-nvme
export NFS_DATA=/mnt/lmcache-nfs

make -C recipies/vllm-lmcache-mp build
make -C recipies/vllm-lmcache-mp up-batch

# Test
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"openai/gpt-oss-120b","prompt":"Hello","max_tokens":16}'

# Check NVMe L2 usage
find $NVME_DATA/lmcache -name 'obj_*.bin' -size +0 | wc -l

# Check NFS L2 usage
ls $NFS_DATA/lmcache/

make -C recipies/vllm-lmcache-mp down
```

## Independent lifecycle management

```bash
# Restart vLLM only — lmcache stays up, warm KV preserved
make -C recipies/vllm-lmcache-mp restart-vllm

# Restart lmcache only — vLLM reconnects automatically
make -C recipies/vllm-lmcache-mp restart-lmcache

# Follow lmcache logs while vllm is separate
make -C recipies/vllm-lmcache-mp logs-lmcache
make -C recipies/vllm-lmcache-mp logs-vllm

# Shell into either container independently
make -C recipies/vllm-lmcache-mp shell-lmcache
make -C recipies/vllm-lmcache-mp shell-vllm
```

## Tuning knobs

| Var | Default | Notes |
| --- | ------- | ----- |
| GPU | 0 | ROCR_VISIBLE_DEVICES for vllm |
| GDS_SLAB_DATA | (unset) | Host mount point for hipFile GDS L1 slab NVMe; enables `--gds-l1-backend hipfile` when set |
| NVME_DATA | /mnt/lmcache-nvme | Host path for L2a NVMe (AIS_MT) |
| NFS_DATA | /mnt/lmcache-nfs | Host path for L2b NFS-over-RDMA |
| LMCACHE_PORT | 6555 | ZMQ port between containers |
| LMCACHE_L1_SIZE_GB | 20 | L1 GPU/CPU memory cap (GiB) |
| LMCACHE_NVME_POOL | 4096 | NIXL pool slots for NVMe adapter |
| LMCACHE_NVME_SLOT_SIZE | 268435456 | Bytes per NVMe pool slot (AIS_MT backend_params.file_size) |
| LMCACHE_NFS_POOL | 1024 | NIXL pool slots for NFS adapter |
| VLLM_MODEL | openai/gpt-oss-120b | Model to serve |
| TENSOR_PARALLEL_SIZE | 1 | vLLM tensor parallelism |
| VLM_GPU_MEMORY_UTILIZATION | 0.90 | vLLM GPU memory fraction |
| VLM_MAX_MODEL_LEN | 32768 | vLLM max context length |

Example with tuned L1 and reduced NVMe pool for a 16 GiB GPU:

```bash
LMCACHE_L1_SIZE_GB=8 \
LMCACHE_NVME_POOL=2048 \
VLM_GPU_MEMORY_UTILIZATION=0.85 \
make -C recipies/vllm-lmcache-mp up-batch
```

## hipFile AIS stats

The lmcache container has `ais-stats` on PATH:

```bash
docker exec -it vllm-lmcache-mp-lmcache bash -lc \
  'ais-stats -p $(pgrep -f lmcache | head -1) -i'
```

## Build arguments

| ARG | Default |
| --- | ------- |
| `ROCM_ARCH` | (required) |
| `LMCACHE_GIT_URL` | `https://github.com/amd-ivaganev/LMCache.git` |
| `LMCACHE_GIT_REF` | `hipfile-for-mp` |
| `LMCACHE_SHA` | (empty = branch head) |
| `HIPFILE_SHA` | pinned in Dockerfile |
| `NIXL_GIT_URL` | `https://github.com/sbates130272/nixl.git` (feat/amd-ais-mt) |
| `NIXL_SHA` | `9d14642` (feat/amd-ais-mt HEAD; pinned in Dockerfile + Makefile) |

## Compare to other recipes

| Recipe | LMCache mode | Storage |
| ------ | ------------ | ------- |
| `vllm-lmcache-hipfile` | in-process | hipFile AIS NVMe |
| `vllm-lmcache-nixl` | in-process | NIXL AIS_MT NVMe or POSIX |
| **`vllm-lmcache-mp`** | **separate container (MP/ZMQ)** | **L1 GPU/CPU + AIS_MT NVMe + POSIX NFS-over-RDMA** |
