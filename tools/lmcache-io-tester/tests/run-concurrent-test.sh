#!/bin/bash
# Test concurrent conversations with multi-pass and
# persistent cache using the sample dataset.

LOG_FILE="${LOG_FILE:-logs/run-concurrent-test.log}"

export HF_HUB_OFFLINE=1

mkdir -p /tmp/lmcache
mkdir -p logs

echo "=== Concurrent + Multi-Pass Test ==="

python -m src.lmcache-sim run \
    --local-cpu \
    --max-local-cpu-size 1.0 \
    --storage-type filesystem \
    --storage-path /tmp/lmcache \
    --hf-model-name gpt2 \
    --local-only \
    --auto-kv-shape \
    --tokenizer-mode text-to-tokens \
    --conversation-file \
        data/vicuna-50000.json \
    --pattern conversation \
    --concurrency 32 \
    --passes 5 \
    --persist-cache \
    --duration 60 \
    > "$LOG_FILE" 2>&1

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "PASSED (exit $STATUS)"
    echo "Log: $LOG_FILE"
else
    echo "FAILED (exit $STATUS)"
    tail -20 "$LOG_FILE"
fi
exit $STATUS
