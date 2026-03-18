#!/bin/bash

# Restrict to 32% of MI325X's memory to increase VRAM pressure
#
#  Available KV cache memory: 10.79 GiB
#  GPU KV cache size: 157,200 tokens
#

SCRIPT_DIR=$(realpath "$(dirname "$0")")

export LMCACHE_LOG_LEVEL=WARNING
export PYTHONHASHSEED=42
export LMCACHE_CONFIG_FILE="${SCRIPT_DIR}/configs/lmcache-cpu.yaml"

vllm serve openai/gpt-oss-120b --gpu-memory-utilization 0.32 \
     --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
