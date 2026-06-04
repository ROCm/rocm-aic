#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Start vLLM with LMCache.  All tunables are read from env vars
# so the same script works for every cache backend.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_LOADER="${APP_DIR}/../lib/load-runtime.sh"
if [[ -x "${RUNTIME_LOADER}" ]]; then
    if ! runtime_exports="$("${RUNTIME_LOADER}" ttft-lmcache "${APP_DIR}")"; then
        exit 1
    fi
    if [[ -n "${runtime_exports}" ]]; then
        source /dev/stdin <<<"${runtime_exports}"
    fi
fi

MODEL="${MODEL:-Qwen/Qwen3-8B}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SEED="${SEED:-42}"

LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-${SCRIPT_DIR}/../configs/lmcache-disk.yaml}"
export LMCACHE_CONFIG_FILE

export LMCACHE_LOG_LEVEL="${LMCACHE_LOG_LEVEL:-WARNING}"
export PYTHONHASHSEED="${SEED}"

echo "=== serve.sh ==="
echo "  MODEL               = ${MODEL}"
echo "  GPU_MEMORY_UTIL      = ${GPU_MEMORY_UTILIZATION}"
echo "  TENSOR_PARALLEL_SIZE = ${TENSOR_PARALLEL_SIZE}"
echo "  LMCACHE_CONFIG_FILE  = ${LMCACHE_CONFIG_FILE}"
echo "  PYTHONHASHSEED       = ${PYTHONHASHSEED}"
echo "================"

vllm serve "${MODEL}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
