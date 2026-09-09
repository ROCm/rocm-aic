#!/usr/bin/env bash
set -euo pipefail

# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# This script runs directly from a workflow checkout; it is not installed on a runner.
if [[ $# -ne 6 ]]; then
    echo "usage: $0 <Dockerfile> <tag> <source-sha> <gpu-arch> <image-name> <repository>" >&2
    exit 2
fi

DOCKERFILE="$1"
TAG="$2"
SOURCE_SHA="$3"
GPU_ARCH="$4"
IMAGE_NAME="$5"
REPOSITORY="$6"

[[ -r "${DOCKERFILE}" ]] || {
    echo "release notes: cannot read ${DOCKERFILE}" >&2
    exit 1
}
for input in TAG SOURCE_SHA GPU_ARCH IMAGE_NAME REPOSITORY; do
    [[ -n "${!input}" ]] || {
        echo "release notes: ${input} must not be empty" >&2
        exit 1
    }
done

_arg() {
    local name="$1" value
    value="$(awk -v prefix="ARG ${name}=" '
        index($0, prefix) == 1 {
            print substr($0, length(prefix) + 1)
            exit
        }
    ' "${DOCKERFILE}")"
    [[ -n "${value}" ]] || {
        echo "release notes: ${name} is not set in ${DOCKERFILE}" >&2
        return 1
    }
    printf '%s\n' "${value}"
}

lmcache_url="$(_arg LMCACHE_GIT_URL)"
lmcache_ref="$(_arg LMCACHE_REF)"
mooncake_url="$(_arg MOONCAKE_GIT_URL)"
mooncake_ref="$(_arg MOONCAKE_REF)"
nixl_url="$(_arg NIXL_GIT_URL)"
nixl_ref="$(_arg NIXL_REF)"
rocm_version="$(_arg ROCM_VERSION)"
rocm_base_image="$(_arg ROCM_BASE_IMAGE)"
vllm_url="$(_arg VLLM_GIT_URL)"
vllm_ref="$(_arg VLLM_REF)"
rocm_base_image="${rocm_base_image//\$\{ROCM_VERSION\}/${rocm_version}}"

case "${rocm_base_image}" in
    *\$\{*)
        echo "release notes: unresolved variable in ROCM_BASE_IMAGE=${rocm_base_image}" >&2
        exit 1
        ;;
esac

{
    echo "Stable release of the **AMD Infinity Context (AIC)** patched stack."
    echo ""
    echo "- **Tag:** \`${TAG}\`"
    echo "- **Source SHA:** \`${SOURCE_SHA}\`"
    echo "- **GPU arch set:** \`${GPU_ARCH}\`"
    echo "- **Base:** ${rocm_base_image} (ROCm ${rocm_version}, Python 3.12, x86_64)"
    echo "- **vLLM:** ${vllm_url} @ \`${vllm_ref}\` (source build)"
    echo "- **LMCache:** ${lmcache_url} @ \`${lmcache_ref}\` + AIC patches"
    echo "- **Mooncake:** ${mooncake_url} @ \`${mooncake_ref}\` (ROCm source build)"
    echo "- **NIXL:** ${nixl_url} @ \`${nixl_ref}\` + nixl-rocm-ais-mt patch"
    echo ""
    echo "### Install wheels"
    echo '```bash'
    echo "python3 -m pip install \\"
    echo "  \"torch==2.13.0+rocm7.2\" \\"
    echo "  \"torchvision==0.28.0+rocm7.2\" \\"
    echo "  --index-url https://download.pytorch.org/whl/rocm7.2"
    echo ""
    echo "python3 -m pip install \\"
    echo "  https://github.com/${REPOSITORY}/releases/download/${TAG}/<vllm-wheel> \\"
    echo "  https://github.com/${REPOSITORY}/releases/download/${TAG}/<mooncake_transfer_engine_rocm-wheel> \\"
    echo "  https://github.com/${REPOSITORY}/releases/download/${TAG}/<lmcache-wheel> \\"
    echo "  https://github.com/${REPOSITORY}/releases/download/${TAG}/<nixl_rocm-wheel>"
    echo '```'
    echo ""
    echo "### Docker"
    echo '```bash'
    echo "docker pull ${IMAGE_NAME}:${TAG}"
    echo '```'
    echo ""
    echo "> These wheels are **not** manylinux: ROCm ${rocm_version} + Python 3.12 + x86_64 only."
    echo "> Install the pinned ROCm Torch pair first; LMCache's open Torch dependency can"
    echo "> otherwise select PyPI's CUDA build in a clean environment."
    echo "> Install the LMCache and Mooncake ROCm wheels together; the LMCache Mooncake"
    echo "> extension resolves libmooncake_store from the companion wheel."
    echo "> The \`nixl_rocm\` wheel needs the ROCm runtime (libamdhip64) and hipFile"
    echo "> present on the host; see README.md."
}
