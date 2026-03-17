#!/bin/bash

# Restrict to 32% of MI325X's memory to increase VRAM pressure
#
#  Available KV cache memory: 10.79 GiB
#  GPU KV cache size: 157,200 tokens
#

export LMCACHE_LOG_LEVEL=ERROR
export PYTHONHASHSEED=42
export LMCACHE_CONFIG_FILE=configs/lmcache-cpu.yaml

vllm serve openai/gpt-oss-120b --gpu-memory-utilization 0.32 \
     --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
