#!/bin/bash

LOG_FILE="${LOG_FILE:-logs/run-download-test.log}"

NUM_CONVERSATIONS="${NUM_CONVERSATIONS:-50000}"

echo "=== Download Conversations Test ==="

python -m src.lmcache-sim download \
    --dataset sharegpt \
    --output "data/vicuna-${NUM_CONVERSATIONS}.json" \
    --max-conversations "${NUM_CONVERSATIONS}" > "$LOG_FILE" 2>&1

python -m src.lmcache-sim download \
    --reprocess "data/vicuna-${NUM_CONVERSATIONS}.json" > "$LOG_FILE" 2>&1
