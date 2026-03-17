#!/bin/bash

# Restrict to 30% of MI325X's memory to increase VRAM pressure
#
#  Available KV cache memory: 10.79 GiB
#  GPU KV cache size: 157,200 tokens
#

vllm serve openai/gpt-oss-120b --gpu-memory-utilization 0.3
