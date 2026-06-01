#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Submit vllm-lmcache-hipfile on Slurm (Docker build + run + Gutenberg run-long-parallel.sh).
#
#   export HF_TOKEN=...                    # or HF_TOKEN_FILE=~/.hf_token
#   export VLH_NVME_BASE=/mnt/nvme       # or leave unset to auto-mount local NVMe
#   export VLH_NVME_MKFS=1               # format blank nvme before mount
#   export VLH_GUTENBERG_DATA_ROOT=/scratch/$USER/vllm-lmcache-hipfile/gutenberg
#   bash .slurm/run-vllm-lmcache-hipfile.sh
#
# Optional GPU / resource selection (cluster-specific Slurm features):
#   export VLH_SLURM_CONSTRAINT=MARKHAM   # any Markham ROCm node (default)
#   export VLH_SLURM_CONSTRAINT='MARKHAM&GFX942'   # MI300X only
#   export VLH_SLURM_EXCLUDE=node1,node2          # sbatch --exclude
#   export VLH_SLURM_NODELIST=mlse-alola-b39-ws2   # sbatch --nodelist (pin one host)
#   export VLH_SLURM_MEM=128G             # large-memory nodes (default 64G)
#
# Generate Gutenberg fixtures once on that shared path:
#   make -C recipies/vllm-lmcache-hipfile data-all BOOK_DATA_ROOT=$VLH_GUTENBERG_DATA_ROOT
#
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p .slurm/logs

_exports="ALL"
if [[ -n "${VLH_NVME_BASE:-}" ]]; then
    _exports="${_exports},VLH_NVME_BASE=${VLH_NVME_BASE}"
fi
for _flag in \
    VLH_NVME_BLK_BPFTRACE \
    VLH_NVME_SMART_LOG \
    VLH_LMCACHE_ENABLE_KV_EVENTS \
    VLH_VFS_BPFTRACE \
    VLH_BENCHMARK \
    VLH_LMCACHE_IO \
    VLH_LMCACHE_GDS_BUFFER_SIZE \
    HF_TOKEN \
    HF_TOKEN_FILE \
    VLH_GUTENBERG_DATA_ROOT \
    VLH_SHARED_ROOT \
    VLH_HF_HOME \
    VLLM_MODEL \
    VLH_VLLM_READY_TIMEOUT \
    VLH_RUN_LONG_ITERATIONS \
    VLH_RUN_LONG_WORKERS \
    VLH_RUN_LONG_PARALLEL \
    VLH_RUN_LONG_BASE_SEED \
    VLH_RUN_LONG_MAX_TOKENS \
    VLH_RUN_LONG_STAGGER_SEC \
    BASE_SEED \
    BOOK_SLUG \
    BOOK_SLUGS \
    BOOK_SLUG_FILE \
    VLH_SKIP_BUILD \
    VLH_NVME_AUTO_USE \
    VLH_NVME_SCRATCH_FALLBACK \
    VLH_NVME_SCRATCH_ROOT \
    VLH_NVME_MIN_AVAIL_GB \
    VLH_NVME_USE_SHARED_DATA_DOCKER \
    VLH_NVME_AUTO_DEVICE \
    VLH_NVME_MKFS \
    VLH_NVME_MOUNT \
    VLH_NVME_DEVICE \
    ROCM_ARCH; do
    if [[ -n "${!_flag:-}" ]]; then
        _exports="${_exports},${_flag}=${!_flag}"
    fi
done

# Defaults when not set in the environment
: "${VLH_SHARED_ROOT:=/scratch/${USER}/vllm-lmcache-hipfile}"
: "${VLH_HF_HOME:=${VLH_SHARED_ROOT}/hf}"
_exports="${_exports},VLH_SHARED_ROOT=${VLH_SHARED_ROOT}"
_exports="${_exports},VLH_HF_HOME=${VLH_HF_HOME}"
: "${VLH_NVME_BLK_BPFTRACE:=1}"
: "${VLH_NVME_SMART_LOG:=1}"
: "${VLH_LMCACHE_ENABLE_KV_EVENTS:=1}"
_exports="${_exports},VLH_NVME_BLK_BPFTRACE=${VLH_NVME_BLK_BPFTRACE}"
_exports="${_exports},VLH_NVME_SMART_LOG=${VLH_NVME_SMART_LOG}"
_exports="${_exports},VLH_LMCACHE_ENABLE_KV_EVENTS=${VLH_LMCACHE_ENABLE_KV_EVENTS}"
: "${VLH_BENCHMARK:=gutenberg}"
_exports="${_exports},VLH_BENCHMARK=${VLH_BENCHMARK}"
# hipFile pool (MiB); 1024 default is too small for parallel long-context get_blocking
: "${VLH_LMCACHE_GDS_BUFFER_SIZE:=2048}"
_exports="${_exports},VLH_LMCACHE_GDS_BUFFER_SIZE=${VLH_LMCACHE_GDS_BUFFER_SIZE}"
if [[ -n "${VLH_GUTENBERG_DATA_ROOT:-}" ]]; then
    _exports="${_exports},VLH_GUTENBERG_DATA_ROOT=${VLH_GUTENBERG_DATA_ROOT}"
fi

_sbopts=()
if [[ -n "${VLH_SLURM_PARTITION:-}" ]]; then
    _sbopts+=(--partition="${VLH_SLURM_PARTITION}")
fi
if [[ -n "${VLH_SLURM_CONSTRAINT:-}" ]]; then
    _sbopts+=(--constraint="${VLH_SLURM_CONSTRAINT}")
fi
if [[ -n "${VLH_SLURM_EXCLUDE:-}" ]]; then
    _sbopts+=(--exclude="${VLH_SLURM_EXCLUDE}")
fi
if [[ -n "${VLH_SLURM_NODELIST:-}" ]]; then
    _sbopts+=(--nodelist="${VLH_SLURM_NODELIST}")
fi
if [[ -n "${VLH_SLURM_MEM:-}" ]]; then
    _sbopts+=(--mem="${VLH_SLURM_MEM}")
fi
if [[ -n "${VLH_SLURM_CPUS:-}" ]]; then
    _sbopts+=(--cpus-per-task="${VLH_SLURM_CPUS}")
fi
if [[ -n "${VLH_SLURM_TIME:-}" ]]; then
    _sbopts+=(--time="${VLH_SLURM_TIME}")
fi
if [[ -n "${VLH_SLURM_GRES:-}" ]]; then
    _sbopts+=(--gres="${VLH_SLURM_GRES}")
fi

out="$(sbatch "${_sbopts[@]}" --export="${_exports}" .slurm/vllm-lmcache-hipfile.sbatch)"
echo "${out}"
jid="${out##* }"
echo "Log:  tail -f .slurm/logs/vllm-lmcache-hipfile-${jid}.log"
echo "Artifacts: .slurm/logs/vllm-lmcache-hipfile-${jid}/"
echo "Done:  sacct -j ${jid} --format=JobID,State,ExitCode,Elapsed,End"
