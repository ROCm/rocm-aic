#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Submit vllm-lmcache-hipfile on Slurm with sensible defaults (override any variable before
# running). From the repository root:
#
#   ./run-slurm.sh
#
# One-time Gutenberg library on shared storage:
#   make -C recipies/vllm-lmcache-hipfile data-all \
#     BOOK_DATA_ROOT="${VLH_GUTENBERG_DATA_ROOT:-/scratch/$USER/vllm-lmcache-hipfile/gutenberg}"
#
# Gutenberg benchmarks live under benchmarks/llm-prefill-benchmark/ (engine-
# agnostic). Slurm sets LLM_PREFILL_BENCH_ROOT automatically; override only
# when using a non-default checkout path.
#
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_runtime_loader="${PWD}/recipies/common/scripts/load-recipe-runtime.sh"
if [[ -x "${_runtime_loader}" ]]; then
    if ! _runtime_exports="$("${_runtime_loader}" vllm-lmcache-hipfile \
        "${PWD}/recipies/vllm-lmcache-hipfile")"; then
        exit 1
    fi
    if [[ -n "${_runtime_exports}" ]]; then
        source /dev/stdin <<<"${_runtime_exports}"
    fi
fi

# --- Hugging Face auth (override or export HF_TOKEN instead) ---
if [[ -z "${HF_TOKEN:-}" && -z "${HF_TOKEN_FILE:-}" ]]; then
    if [[ -r "${HOME}/.batesste-hugging-face-read-march-2026.token" ]]; then
        export HF_TOKEN_FILE="${HOME}/.batesste-hugging-face-read-march-2026.token"
    elif [[ -r "${HOME}/.hf_token" ]]; then
        export HF_TOKEN_FILE="${HOME}/.hf_token"
    fi
fi

# --- Shared scratch tree (Gutenberg + golden HF Hub cache; never per-job lmcache-*/hf) ---
: "${VLH_SHARED_ROOT:=/scratch/${USER}/vllm-lmcache-hipfile}"
export VLH_SHARED_ROOT
: "${VLH_HF_HOME:=/scratch/${USER}/vllm-lmcache-hipfile/hf}"
export VLH_HF_HOME
: "${VLH_GUTENBERG_DATA_ROOT:=${VLH_SHARED_ROOT}/gutenberg}"
export VLH_GUTENBERG_DATA_ROOT

# --- LMCache DATA only: unset VLH_NVME_BASE → discover on the compute node ---
# Order: blank nvme*n* → mounted NVMe (/mnt, /local, …) → /scratch/.../lmcache-<jobid>
: "${VLH_NVME_AUTO_USE:=1}"
: "${VLH_NVME_SCRATCH_FALLBACK:=1}"
export VLH_NVME_AUTO_USE VLH_NVME_SCRATCH_FALLBACK
# Blank nvme*n* is formatted when found (only if VLH_NVME_BASE is unset):
if [[ -z "${VLH_NVME_BASE:-}" ]]; then
    : "${VLH_NVME_MKFS:=1}"
    export VLH_NVME_MKFS
fi
# export VLH_NVME_MKFS=0          # never mkfs; use mounted/scratch/tmp only

# --- LMCache disk backend ---
: "${VLH_LMCACHE_IO:=posix}"
export VLH_LMCACHE_IO
if [[ "${VLH_LMCACHE_IO}" == hipfile ]]; then
    : "${VLH_LMCACHE_GDS_BUFFER_SIZE:=2048}"
    export VLH_LMCACHE_GDS_BUFFER_SIZE
fi

# --- Model (server + benchmarks): VLLM_MODEL only; unset → gpt-oss-120b in yaml ---
# export VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct

# --- Benchmark: parallel Gutenberg via run-long-parallel.sh ---
: "${VLH_BENCHMARK:=gutenberg}"
: "${VLH_RUN_LONG_PARALLEL:=1}"
: "${VLH_RUN_LONG_WORKERS:=4}"
: "${VLH_RUN_LONG_ITERATIONS:=1}"
: "${VLH_RUN_LONG_BASE_SEED:=42}"
export VLH_BENCHMARK VLH_RUN_LONG_PARALLEL
export VLH_RUN_LONG_WORKERS VLH_RUN_LONG_ITERATIONS VLH_RUN_LONG_BASE_SEED

# export VLH_NVME_BLK_BPFTRACE=0   # bpftrace often needs root on compute nodes

# --- Optional Slurm (uncomment to narrow nodes / raise memory) ---
# export VLH_SLURM_CONSTRAINT=MARKHAM
: "${VLH_SLURM_CONSTRAINT:=MARKHAM&NVME}"
: "${VLH_SLURM_EXCLUDE:=ctr-cx65-mi300x-30}"
export VLH_SLURM_CONSTRAINT VLH_SLURM_EXCLUDE
# Pin to a workstation where you built the image (local docker only):
# export VLH_SLURM_NODELIST='mlse-alola-b39-ws2'
# export VLH_SKIP_BUILD=1
# export ROCM_ARCH=gfx1100w
# export VLH_SLURM_MEM=128G
# export VLH_SLURM_CPUS=16

exec "${PWD}/.slurm/run-vllm-lmcache-hipfile.sh" "$@"
