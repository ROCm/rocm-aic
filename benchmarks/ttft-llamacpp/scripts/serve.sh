#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Start llama-server with configurable parameters.
# All tunables are read from env vars.
#
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B-GGUF}"
CTX_SIZE="${CTX_SIZE:-16384}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
SLOT_SAVE_PATH="${SLOT_SAVE_PATH:-/slots}"
HOST="${LLAMA_HOST:-0.0.0.0}"
PORT="${LLAMA_PORT:-8080}"

# pin to a single GPU if not already restricted
if [[ -z "${HIP_VISIBLE_DEVICES:-}" ]] && [[ -z "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES=0
fi

echo "=== serve.sh ==="
echo "  MODEL          = ${MODEL}"
echo "  CTX_SIZE       = ${CTX_SIZE}"
echo "  N_GPU_LAYERS   = ${N_GPU_LAYERS}"
echo "  SLOT_SAVE_PATH = ${SLOT_SAVE_PATH}"
echo "  HOST:PORT      = ${HOST}:${PORT}"
echo "================"

mkdir -p "${SLOT_SAVE_PATH}"

llama-server \
    --model "${MODEL}" \
    --ctx-size "${CTX_SIZE}" \
    --n-gpu-layers "${N_GPU_LAYERS}" \
    --slot-save-path "${SLOT_SAVE_PATH}" \
    --host "${HOST}" \
    --port "${PORT}"
