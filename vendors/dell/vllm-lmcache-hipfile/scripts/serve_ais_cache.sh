#!/bin/bash

# Restrict to 32% of MI325X's memory to increase VRAM pressure
#
#  Available KV cache memory: 10.79 GiB
#  GPU KV cache size: 157,200 tokens
#

SCRIPT_DIR=$(realpath "$(dirname "$0")")

export LMCACHE_LOG_LEVEL=WARNING
export PYTHONHASHSEED=42
export LMCACHE_CONFIG_FILE="${SCRIPT_DIR}/configs/lmcache-ais.yaml"

# Set default values for GPU memory utilization and tensor parallel size
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.32}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}

vllm serve openai/gpt-oss-120b --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
     --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
     --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
