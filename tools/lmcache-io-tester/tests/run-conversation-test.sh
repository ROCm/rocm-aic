#!/bin/bash

LOG_FILE="${LOG_FILE:-logs/run-conversation-test.log}"

export HF_HUB_OFFLINE=1

#rm -rf /tmp/lmcache
mkdir -p /tmp/lmcache

python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path /tmp/lmcache \
    --hf-model-name gpt2 \
    --local-only \
    --auto-kv-shape \
    --tokenizer-mode text-to-tokens \
    --conversation-file \
        data/sample-conversations.json \
    --pattern conversation \
    --duration 10 > "$LOG_FILE" 2>&1
