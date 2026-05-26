#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Submit vllm-radeon on Slurm (Docker build + run + Gutenberg run-long-parallel.sh).
#
#   export HF_TOKEN=...                    # or HF_TOKEN_FILE=~/.hf_token
#   export RADEON_NVME_BASE=/mnt/nvme       # or leave unset to auto-mount local NVMe
#   export RADEON_NVME_MKFS=1               # format blank nvme before mount
#   export RADEON_GUTENBERG_DATA_ROOT=/scratch/$USER/vllm-radeon/gutenberg
#   bash .slurm/run-vllm-radeon.sh
#
# Optional GPU / resource selection (cluster-specific Slurm features):
#   export RADEON_SLURM_CONSTRAINT=MARKHAM   # any Markham ROCm node (default)
#   export RADEON_SLURM_CONSTRAINT='MARKHAM&GFX942'   # MI300X only
#   export RADEON_SLURM_MEM=128G             # large-memory nodes (default 64G)
#
# Generate Gutenberg fixtures once on that shared path:
#   make -C recipies/vllm-radeon data-all BOOK_DATA_ROOT=$RADEON_GUTENBERG_DATA_ROOT
#
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p .slurm/logs

_exports="ALL"
if [[ -n "${RADEON_NVME_BASE:-}" ]]; then
    _exports="${_exports},RADEON_NVME_BASE=${RADEON_NVME_BASE}"
fi
for _flag in \
    RADEON_NVME_BLK_BPFTRACE \
    RADEON_NVME_SMART_LOG \
    RADEON_LMCACHE_ENABLE_KV_EVENTS \
    RADEON_VFS_BPFTRACE \
    RADEON_BENCHMARK \
    RADEON_LMCACHE_IO \
    RADEON_LMCACHE_GDS_BUFFER_SIZE \
    HF_TOKEN \
    HF_TOKEN_FILE \
    RADEON_GUTENBERG_DATA_ROOT \
    RADEON_SHARED_ROOT \
    RADEON_HF_HOME \
    VLLM_MODEL \
    RADEON_VLLM_READY_TIMEOUT \
    RADEON_RUN_LONG_ITERATIONS \
    RADEON_RUN_LONG_WORKERS \
    RADEON_RUN_LONG_PARALLEL \
    RADEON_RUN_LONG_BASE_SEED \
    RADEON_RUN_LONG_STAGGER_SEC \
    BASE_SEED \
    BOOK_SLUG \
    BOOK_SLUGS \
    BOOK_SLUG_FILE \
    RADEON_SKIP_BUILD \
    RADEON_NVME_AUTO_USE \
    RADEON_NVME_SCRATCH_FALLBACK \
    RADEON_NVME_SCRATCH_ROOT \
    RADEON_NVME_MIN_AVAIL_GB \
    RADEON_NVME_USE_SHARED_DATA_DOCKER \
    RADEON_NVME_AUTO_DEVICE \
    RADEON_NVME_MKFS \
    RADEON_NVME_MOUNT \
    RADEON_NVME_DEVICE \
    ROCM_ARCH; do
    if [[ -n "${!_flag:-}" ]]; then
        _exports="${_exports},${_flag}=${!_flag}"
    fi
done

# Defaults when not set in the environment
: "${RADEON_SHARED_ROOT:=/scratch/${USER}/vllm-radeon}"
: "${RADEON_HF_HOME:=${RADEON_SHARED_ROOT}/hf}"
_exports="${_exports},RADEON_SHARED_ROOT=${RADEON_SHARED_ROOT}"
_exports="${_exports},RADEON_HF_HOME=${RADEON_HF_HOME}"
: "${RADEON_NVME_BLK_BPFTRACE:=1}"
: "${RADEON_NVME_SMART_LOG:=1}"
: "${RADEON_LMCACHE_ENABLE_KV_EVENTS:=1}"
_exports="${_exports},RADEON_NVME_BLK_BPFTRACE=${RADEON_NVME_BLK_BPFTRACE}"
_exports="${_exports},RADEON_NVME_SMART_LOG=${RADEON_NVME_SMART_LOG}"
_exports="${_exports},RADEON_LMCACHE_ENABLE_KV_EVENTS=${RADEON_LMCACHE_ENABLE_KV_EVENTS}"
: "${RADEON_BENCHMARK:=gutenberg}"
_exports="${_exports},RADEON_BENCHMARK=${RADEON_BENCHMARK}"
# hipFile pool (MiB); 1024 default is too small for parallel long-context get_blocking
: "${RADEON_LMCACHE_GDS_BUFFER_SIZE:=2048}"
_exports="${_exports},RADEON_LMCACHE_GDS_BUFFER_SIZE=${RADEON_LMCACHE_GDS_BUFFER_SIZE}"
if [[ -n "${RADEON_GUTENBERG_DATA_ROOT:-}" ]]; then
    _exports="${_exports},RADEON_GUTENBERG_DATA_ROOT=${RADEON_GUTENBERG_DATA_ROOT}"
fi

_sbopts=()
if [[ -n "${RADEON_SLURM_PARTITION:-}" ]]; then
    _sbopts+=(--partition="${RADEON_SLURM_PARTITION}")
fi
if [[ -n "${RADEON_SLURM_CONSTRAINT:-}" ]]; then
    _sbopts+=(--constraint="${RADEON_SLURM_CONSTRAINT}")
fi
if [[ -n "${RADEON_SLURM_MEM:-}" ]]; then
    _sbopts+=(--mem="${RADEON_SLURM_MEM}")
fi
if [[ -n "${RADEON_SLURM_CPUS:-}" ]]; then
    _sbopts+=(--cpus-per-task="${RADEON_SLURM_CPUS}")
fi
if [[ -n "${RADEON_SLURM_TIME:-}" ]]; then
    _sbopts+=(--time="${RADEON_SLURM_TIME}")
fi
if [[ -n "${RADEON_SLURM_GRES:-}" ]]; then
    _sbopts+=(--gres="${RADEON_SLURM_GRES}")
fi

out="$(sbatch "${_sbopts[@]}" --export="${_exports}" .slurm/vllm-radeon.sbatch)"
echo "${out}"
jid="${out##* }"
echo "Log:  tail -f .slurm/logs/vllm-radeon-${jid}.log"
echo "Artifacts: .slurm/logs/vllm-radeon-${jid}/"
echo "Done:  sacct -j ${jid} --format=JobID,State,ExitCode,Elapsed,End"
