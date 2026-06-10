#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Quick cliff smoke against a running vllm-optimus-kvd container.
# Mirrors `make cliff` but standalone for ad-hoc inspection.

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-vllm-optimus-kvd}"
GPU="${GPU:-0}"
CONTAINER_NAME="${CONTAINER_NAME:-${IMAGE_NAME}-gpu${GPU}}"
PORT="${PORT:-8000}"
VLLM_MODEL="${VLLM_MODEL:-openai/gpt-oss-120b}"

docker exec -it "${CONTAINER_NAME}" \
    python3 /app/cliff/run_cliff.py \
        --endpoint "http://127.0.0.1:${PORT}" \
        --model "${VLLM_MODEL}" \
        --concurrencies "16,32,48,64" \
        --isl 60000 --shared-prefix-tokens 60000 \
        --iters 3 --max-tokens 64
