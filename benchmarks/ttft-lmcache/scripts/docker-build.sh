#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Build the TTFT benchmark Docker image.
#
# Optional env vars:
#   IMAGE_TAG       -- image name  (default: $USER-ttft-lmcache)
#   WITH_HIPFILE    -- 1 to include hipFile/AIS support (default: 1)
#   GPU_ARCHS       -- semicolon-separated GPU arch list
#                      (default: auto-detected at build time)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_CTX="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-$(whoami)-ttft-lmcache}"
WITH_HIPFILE="${WITH_HIPFILE:-1}"

EXTRA_ARGS=()
EXTRA_ARGS+=(--build-arg "WITH_HIPFILE=${WITH_HIPFILE}")

if [[ -n "${GPU_ARCHS:-}" ]]; then
    EXTRA_ARGS+=(--build-arg "GPU_ARCHS=${GPU_ARCHS}")
fi

echo "Building image: ${IMAGE_TAG}"
echo "  context:      ${BUILD_CTX}"
echo "  WITH_HIPFILE: ${WITH_HIPFILE}"
echo "  GPU_ARCHS:    ${GPU_ARCHS:-auto}"

docker buildx build \
    -f "${BUILD_CTX}/Dockerfile" \
    -t "${IMAGE_TAG}" \
    "${EXTRA_ARGS[@]}" \
    "${BUILD_CTX}"
