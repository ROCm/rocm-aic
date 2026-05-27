#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Submit vllm-lmcache-nixl on Slurm (Docker build + run + Gutenberg benchmark).
#
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p .slurm/logs

_exports="ALL"
if [[ -n "${VLN_NVME_BASE:-}" ]]; then
    _exports="${_exports},VLN_NVME_BASE=${VLN_NVME_BASE}"
fi
for _flag in \
    VLN_NVME_BLK_BPFTRACE \
    VLN_NVME_SMART_LOG \
    VLN_LMCACHE_ENABLE_KV_EVENTS \
    VLN_VFS_BPFTRACE \
    VLN_BENCHMARK \
    VLN_LMCACHE_IO \
    VLN_NIXL_BUFFER_SIZE \
    HF_TOKEN \
    HF_TOKEN_FILE \
    VLN_GUTENBERG_DATA_ROOT \
    VLN_SHARED_ROOT \
    VLN_HF_HOME \
    VLLM_MODEL \
    VLN_VLLM_READY_TIMEOUT \
    VLN_RUN_LONG_ITERATIONS \
    VLN_RUN_LONG_WORKERS \
    VLN_RUN_LONG_PARALLEL \
    VLN_RUN_LONG_BASE_SEED \
    VLN_RUN_LONG_STAGGER_SEC \
    BASE_SEED \
    BOOK_SLUG \
    BOOK_SLUGS \
    BOOK_SLUG_FILE \
    VLN_SKIP_BUILD \
    VLN_NVME_AUTO_USE \
    VLN_NVME_SCRATCH_FALLBACK \
    VLN_NVME_SCRATCH_ROOT \
    VLN_NVME_MIN_AVAIL_GB \
    VLN_NVME_USE_SHARED_DATA_DOCKER \
    VLN_NVME_AUTO_DEVICE \
    VLN_NVME_MKFS \
    VLN_NVME_MOUNT \
    VLN_NVME_DEVICE \
    ROCM_ARCH \
    NIXL_GIT_URL \
    NIXL_REF; do
    if [[ -n "${!_flag:-}" ]]; then
        _exports="${_exports},${_flag}=${!_flag}"
    fi
done

: "${VLN_SHARED_ROOT:=/scratch/${USER}/vllm-lmcache-nixl}"
: "${VLN_HF_HOME:=${VLN_SHARED_ROOT}/hf}"
_exports="${_exports},VLN_SHARED_ROOT=${VLN_SHARED_ROOT}"
_exports="${_exports},VLN_HF_HOME=${VLN_HF_HOME}"
: "${VLN_NVME_BLK_BPFTRACE:=1}"
: "${VLN_NVME_SMART_LOG:=1}"
: "${VLN_LMCACHE_ENABLE_KV_EVENTS:=1}"
_exports="${_exports},VLN_NVME_BLK_BPFTRACE=${VLN_NVME_BLK_BPFTRACE}"
_exports="${_exports},VLN_NVME_SMART_LOG=${VLN_NVME_SMART_LOG}"
_exports="${_exports},VLN_LMCACHE_ENABLE_KV_EVENTS=${VLN_LMCACHE_ENABLE_KV_EVENTS}"
: "${VLN_BENCHMARK:=gutenberg}"
_exports="${_exports},VLN_BENCHMARK=${VLN_BENCHMARK}"
: "${VLN_LMCACHE_IO:=nixl-posix}"
_exports="${_exports},VLN_LMCACHE_IO=${VLN_LMCACHE_IO}"
if [[ -n "${VLN_GUTENBERG_DATA_ROOT:-}" ]]; then
    _exports="${_exports},VLN_GUTENBERG_DATA_ROOT=${VLN_GUTENBERG_DATA_ROOT}"
fi

_sbopts=()
if [[ -n "${VLN_SLURM_PARTITION:-}" ]]; then
    _sbopts+=(--partition="${VLN_SLURM_PARTITION}")
fi
if [[ -n "${VLN_SLURM_CONSTRAINT:-}" ]]; then
    _sbopts+=(--constraint="${VLN_SLURM_CONSTRAINT}")
fi
if [[ -n "${VLN_SLURM_EXCLUDE:-}" ]]; then
    _sbopts+=(--exclude="${VLN_SLURM_EXCLUDE}")
fi
if [[ -n "${VLN_SLURM_NODELIST:-}" ]]; then
    _sbopts+=(--nodelist="${VLN_SLURM_NODELIST}")
fi
if [[ -n "${VLN_SLURM_MEM:-}" ]]; then
    _sbopts+=(--mem="${VLN_SLURM_MEM}")
fi
if [[ -n "${VLN_SLURM_CPUS:-}" ]]; then
    _sbopts+=(--cpus-per-task="${VLN_SLURM_CPUS}")
fi
if [[ -n "${VLN_SLURM_TIME:-}" ]]; then
    _sbopts+=(--time="${VLN_SLURM_TIME}")
fi
if [[ -n "${VLN_SLURM_GRES:-}" ]]; then
    _sbopts+=(--gres="${VLN_SLURM_GRES}")
fi

out="$(sbatch "${_sbopts[@]}" --export="${_exports}" .slurm/vllm-lmcache-nixl.sbatch)"
echo "${out}"
jid="${out##* }"
echo "Log:  tail -f .slurm/logs/vllm-lmcache-nixl-${jid}.log"
echo "Artifacts: .slurm/logs/vllm-lmcache-nixl-${jid}/"
echo "Done:  sacct -j ${jid} --format=JobID,State,ExitCode,Elapsed,End"
