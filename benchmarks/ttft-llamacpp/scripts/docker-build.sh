#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Build the llama.cpp TTFT benchmark Docker image.
#
# Optional env vars:
#   IMAGE_TAG              -- image name (default: $USER-ttft-llamacpp)
#   APPLY_CACHE_DISK_PATCH -- 1 to apply the cache-disk patch (default: 0)
#   GPU_ARCHS              -- semicolon-separated GPU arch list
#                             (default: auto-detected at build time)
#   LLAMACPP_TAG           -- llama.cpp git tag to build (default: b8799)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_CTX="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-$(whoami)-ttft-llamacpp}"
APPLY_CACHE_DISK_PATCH="${APPLY_CACHE_DISK_PATCH:-0}"

EXTRA_ARGS=()
EXTRA_ARGS+=(--build-arg "APPLY_CACHE_DISK_PATCH=${APPLY_CACHE_DISK_PATCH}")

if [[ -n "${GPU_ARCHS:-}" ]]; then
    EXTRA_ARGS+=(--build-arg "GPU_ARCHS=${GPU_ARCHS}")
fi

if [[ -n "${LLAMACPP_TAG:-}" ]]; then
    EXTRA_ARGS+=(--build-arg "LLAMACPP_TAG=${LLAMACPP_TAG}")
fi

echo "Building image: ${IMAGE_TAG}"
echo "  context:              ${BUILD_CTX}"
echo "  APPLY_CACHE_DISK_PATCH: ${APPLY_CACHE_DISK_PATCH}"
echo "  GPU_ARCHS:            ${GPU_ARCHS:-auto}"

docker buildx build \
    -f "${BUILD_CTX}/Dockerfile" \
    -t "${IMAGE_TAG}" \
    "${EXTRA_ARGS[@]}" \
    "${BUILD_CTX}"
