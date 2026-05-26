#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Submit vllm-radeon on Slurm with sensible defaults (override any variable before
# running). From the repository root:
#
#   ./run-slurm.sh
#
# One-time Gutenberg library on shared storage:
#   make -C recipies/vllm-radeon data-all \
#     BOOK_DATA_ROOT="${RADEON_GUTENBERG_DATA_ROOT:-/scratch/$USER/vllm-radeon/gutenberg}"
#
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Hugging Face auth (override or export HF_TOKEN instead) ---
if [[ -z "${HF_TOKEN:-}" && -z "${HF_TOKEN_FILE:-}" ]]; then
    if [[ -r "${HOME}/.batesste-hugging-face-read-march-2026.token" ]]; then
        export HF_TOKEN_FILE="${HOME}/.batesste-hugging-face-read-march-2026.token"
    elif [[ -r "${HOME}/.hf_token" ]]; then
        export HF_TOKEN_FILE="${HOME}/.hf_token"
    fi
fi

# --- Shared scratch tree (Gutenberg + golden HF Hub cache; never per-job lmcache-*/hf) ---
: "${RADEON_SHARED_ROOT:=/scratch/${USER}/vllm-radeon}"
export RADEON_SHARED_ROOT
: "${RADEON_HF_HOME:=/scratch/${USER}/vllm-radeon/hf}"
export RADEON_HF_HOME
: "${RADEON_GUTENBERG_DATA_ROOT:=${RADEON_SHARED_ROOT}/gutenberg}"
export RADEON_GUTENBERG_DATA_ROOT

# --- LMCache DATA only: unset RADEON_NVME_BASE → discover on the compute node ---
# Order: blank nvme*n* → mounted NVMe (/mnt, /local, …) → /scratch/.../lmcache-<jobid>
: "${RADEON_NVME_AUTO_USE:=1}"
: "${RADEON_NVME_SCRATCH_FALLBACK:=1}"
export RADEON_NVME_AUTO_USE RADEON_NVME_SCRATCH_FALLBACK
# Blank nvme*n* is formatted when found (only if RADEON_NVME_BASE is unset):
if [[ -z "${RADEON_NVME_BASE:-}" ]]; then
    : "${RADEON_NVME_MKFS:=1}"
    export RADEON_NVME_MKFS
fi
# export RADEON_NVME_MKFS=0          # never mkfs; use mounted/scratch/tmp only

# --- LMCache disk backend ---
: "${RADEON_LMCACHE_IO:=posix}"
export RADEON_LMCACHE_IO
if [[ "${RADEON_LMCACHE_IO}" == hipfile ]]; then
    : "${RADEON_LMCACHE_GDS_BUFFER_SIZE:=2048}"
    export RADEON_LMCACHE_GDS_BUFFER_SIZE
fi

# --- Model (server + benchmarks): VLLM_MODEL only; unset → gpt-oss-120b in yaml ---
# export VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct

# --- Benchmark: parallel Gutenberg via run-long-parallel.sh ---
: "${RADEON_BENCHMARK:=gutenberg}"
: "${RADEON_RUN_LONG_PARALLEL:=1}"
: "${RADEON_RUN_LONG_WORKERS:=4}"
: "${RADEON_RUN_LONG_ITERATIONS:=1}"
: "${RADEON_RUN_LONG_BASE_SEED:=42}"
export RADEON_BENCHMARK RADEON_RUN_LONG_PARALLEL
export RADEON_RUN_LONG_WORKERS RADEON_RUN_LONG_ITERATIONS RADEON_RUN_LONG_BASE_SEED

# export RADEON_NVME_BLK_BPFTRACE=0   # bpftrace often needs root on compute nodes

# --- Optional Slurm (uncomment to narrow nodes / raise memory) ---
# export RADEON_SLURM_CONSTRAINT=MARKHAM
export RADEON_SLURM_CONSTRAINT='MARKHAM&NVME'
# export RADEON_SLURM_MEM=128G
# export RADEON_SLURM_CPUS=16

exec "${PWD}/.slurm/run-vllm-radeon.sh" "$@"
