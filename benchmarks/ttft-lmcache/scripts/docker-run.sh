#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Launch the TTFT benchmark container with ROCm GPU passthrough.
#
# Optional env vars:
#   IMAGE_TAG       -- image name       (default: $USER-ttft-lmcache)
#   CACHE_MOUNT     -- host path for KV cache storage
#                      (default: /tmp/lmcache-bench)
#   DATA_MOUNT      -- host path for AIS/NVMe storage
#                      (default: /data)
#   HF_TOKEN        -- HuggingFace token (passed as HF_TOKEN env)
#
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-$(whoami)-ttft-lmcache}"
CACHE_MOUNT="${CACHE_MOUNT:-/tmp/lmcache-bench}"
DATA_MOUNT="${DATA_MOUNT:-/data}"

mkdir -p "${CACHE_MOUNT}"

DOCKER_ARGS=(
    -it --rm
    --device /dev/kfd
    --device /dev/dri
    --security-opt apparmor=unconfined
    --security-opt seccomp=unconfined
    --network host --ipc host
    --shm-size=10G
    --ulimit memlock=-1
    --ulimit stack=67108864
    --cap-add IPC_LOCK
    --cap-add SYS_PTRACE
    -v "${CACHE_MOUNT}:/cache"
    -e "HF_TOKEN=${HF_TOKEN:-}"
)

if [[ -d /dev/infiniband ]]; then
    DOCKER_ARGS+=(--device /dev/infiniband)
fi

if [[ -d "${DATA_MOUNT}" ]]; then
    DOCKER_ARGS+=(-v "${DATA_MOUNT}:/data")
fi

echo "Launching container: ${IMAGE_TAG}"
echo "  cache mount: ${CACHE_MOUNT} -> /cache"
echo "  data mount:  ${DATA_MOUNT} -> /data"

exec docker run "${DOCKER_ARGS[@]}" "${IMAGE_TAG}" "$@"
