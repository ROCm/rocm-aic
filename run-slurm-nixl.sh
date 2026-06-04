#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Submit vllm-lmcache-nixl on Slurm with sensible defaults. From repo root:
#
#   ./run-slurm-nixl.sh
#
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_runtime_loader="${PWD}/recipies/common/scripts/load-recipe-runtime.sh"
if [[ -x "${_runtime_loader}" ]]; then
    if ! _runtime_exports="$("${_runtime_loader}" vllm-lmcache-nixl \
        "${PWD}/recipies/vllm-lmcache-nixl")"; then
        exit 1
    fi
    if [[ -n "${_runtime_exports}" ]]; then
        source /dev/stdin <<<"${_runtime_exports}"
    fi
fi

if [[ -z "${HF_TOKEN:-}" && -z "${HF_TOKEN_FILE:-}" ]]; then
    if [[ -r "${HOME}/.batesste-hugging-face-read-march-2026.token" ]]; then
        export HF_TOKEN_FILE="${HOME}/.batesste-hugging-face-read-march-2026.token"
    elif [[ -r "${HOME}/.hf_token" ]]; then
        export HF_TOKEN_FILE="${HOME}/.hf_token"
    fi
fi

: "${VLN_SHARED_ROOT:=/scratch/${USER}/vllm-lmcache-nixl}"
export VLN_SHARED_ROOT
: "${VLN_HF_HOME:=/scratch/${USER}/vllm-lmcache-nixl/hf}"
export VLN_HF_HOME
: "${VLN_GUTENBERG_DATA_ROOT:=${VLN_SHARED_ROOT}/gutenberg}"
export VLN_GUTENBERG_DATA_ROOT

: "${VLN_NVME_AUTO_USE:=1}"
: "${VLN_NVME_SCRATCH_FALLBACK:=1}"
export VLN_NVME_AUTO_USE VLN_NVME_SCRATCH_FALLBACK
if [[ -z "${VLN_NVME_BASE:-}" ]]; then
    : "${VLN_NVME_MKFS:=1}"
    export VLN_NVME_MKFS
fi

: "${VLN_LMCACHE_IO:=nixl-posix}"
export VLN_LMCACHE_IO

: "${VLN_BENCHMARK:=gutenberg}"
: "${VLN_RUN_LONG_PARALLEL:=1}"
: "${VLN_RUN_LONG_WORKERS:=4}"
: "${VLN_RUN_LONG_ITERATIONS:=1}"
: "${VLN_RUN_LONG_BASE_SEED:=42}"
export VLN_BENCHMARK VLN_RUN_LONG_PARALLEL
export VLN_RUN_LONG_WORKERS VLN_RUN_LONG_ITERATIONS VLN_RUN_LONG_BASE_SEED

: "${VLN_SLURM_CONSTRAINT:=MARKHAM&NVME}"
export VLN_SLURM_CONSTRAINT

exec "${PWD}/.slurm/run-vllm-lmcache-nixl.sh" "$@"
