#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Launch the llama.cpp TTFT benchmark container with ROCm GPU
# passthrough.
#
# Optional env vars:
#   IMAGE_TAG    -- image name (default: $USER-ttft-llamacpp)
#   SLOT_MOUNT   -- host path for slot save files
#                   (default: /tmp/llamacpp-slots)
#   MODEL_DIR    -- host path containing GGUF model files
#                   (mounted at /models inside container)
#   SLOT_TMPFS   -- if "1", mount /slots as tmpfs for RAM-speed
#                   slot storage (default: 0)
#
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-$(whoami)-ttft-llamacpp}"
SLOT_MOUNT="${SLOT_MOUNT:-/tmp/llamacpp-slots}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
SLOT_TMPFS="${SLOT_TMPFS:-0}"

mkdir -p "${SLOT_MOUNT}"

DOCKER_ARGS=(
    -it --rm
    --device /dev/kfd
    --device /dev/dri
    --security-opt apparmor=unconfined
    --security-opt seccomp=unconfined
    --network host --ipc host
    --shm-size=4G
    --ulimit memlock=-1
)

if [[ "${SLOT_TMPFS}" == "1" ]]; then
    DOCKER_ARGS+=(--tmpfs "/slots:rw,size=4g")
    echo "Slot storage: tmpfs (RAM-backed, 4G)"
else
    DOCKER_ARGS+=(-v "${SLOT_MOUNT}:/slots")
    echo "Slot storage: ${SLOT_MOUNT} -> /slots"
fi

if [[ -d "${MODEL_DIR}" ]]; then
    DOCKER_ARGS+=(-v "${MODEL_DIR}:/models:ro")
    echo "Model dir: ${MODEL_DIR} -> /models"
fi

echo "Launching container: ${IMAGE_TAG}"

exec docker run "${DOCKER_ARGS[@]}" "${IMAGE_TAG}" "$@"
