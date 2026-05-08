#!/bin/bash

LOG_FILE="${LOG_FILE:-logs/run-simple-test.log}"

rm -rf /tmp/lmcache
mkdir -p /tmp/lmcache

python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path /tmp/lmcache \
    --hf-model-name gpt2 \
    --local-only \
    --auto-kv-shape \
    --tokenizer-mode text-to-tokens \
    --text-input "data/sample-text.txt" \
    --pattern random \
    --duration 10 > "$LOG_FILE" 2>&1