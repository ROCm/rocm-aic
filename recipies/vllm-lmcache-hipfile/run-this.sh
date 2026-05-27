#!/bin/bash
# Must match **`make run`**: default **`vllm-lmcache-hipfile-gpu0`** (**`IMAGE_NAME-gpu${GPU}`**).
IMAGE_NAME="${IMAGE_NAME:-vllm-lmcache-hipfile}"
GPU="${GPU:-0}"
CONTAINER_NAME="${CONTAINER_NAME:-${IMAGE_NAME}-gpu${GPU}}"

docker exec -it "${CONTAINER_NAME}" python3 \
  /app/LMCache/benchmarks/long_doc_qa/long_doc_qa.py \
  --port 8000 \
  --model openai/gpt-oss-120b \
  --num-documents 40 \
  --document-length 24000 \
  --output-len 128 \
  --repeat-count 4 \
  --repeat-mode tile \
  --hit-miss-ratio 1:2 \
  --max-inflight-requests 4 \
  --sleep-time-after-warmup 10 \
  --visualize \
  --completions \
  --json-output \
  --trim-fraction 0.1
