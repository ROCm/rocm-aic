#!/bin/bash

# Uses vLLM's default short multi-turn example config

SCRIPT_DIR=$(realpath "$(dirname "$0")")

python vllm/benchmarks/multi_turn/benchmark_serving_multi_turn.py \
    --model openai/gpt-oss-120b \
    --input-file "${SCRIPT_DIR}/configs/generate_multi_turn_short.json" \
    --num-clients 2 \
    --max-active-conversations 6
