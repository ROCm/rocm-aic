#!/usr/bin/env bash
set -euo pipefail

# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

readonly PYTORCH_REPO_DEFAULT='https://github.com/ROCm/pytorch.git'

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

# Derive the sibling Dockerfile paths for the split-Dockerfile layout.
# When DOCKERFILE is docker/lmcache/Dockerfile, also probe docker/base/Dockerfile
# for ROCM_VERSION / ROCM_BASE_IMAGE (which live there after the split).
_DOCKERFILE_DIR="$(dirname "${DOCKERFILE}")"
_DOCKERFILES=("${DOCKERFILE}")
if [[ -r "${_DOCKERFILE_DIR}/../base/Dockerfile" ]]; then
    _DOCKERFILES+=("${_DOCKERFILE_DIR}/../base/Dockerfile")
fi
if [[ -r "${_DOCKERFILE_DIR}/../vllm/Dockerfile" ]]; then
    _DOCKERFILES+=("${_DOCKERFILE_DIR}/../vllm/Dockerfile")
fi

_arg() {
    local name="$1" value df
    for df in "${_DOCKERFILES[@]}"; do
        value="$(awk -v prefix="ARG ${name}=" '
            index($0, prefix) == 1 {
                print substr($0, length(prefix) + 1)
                exit
            }
        ' "${df}")"
        [[ -n "${value}" ]] && { printf '%s\n' "${value}"; return 0; }
    done
    echo "release notes: ${name} is not set in any Dockerfile (searched: ${_DOCKERFILES[*]})" >&2
    return 1
}

_optional_arg() {
    local name="$1" default="$2" value
    value="$(awk -v prefix="ARG ${name}=" '
        index($0, prefix) == 1 {
            print substr($0, length(prefix) + 1)
            exit
        }
    ' "${DOCKERFILE}")"
    printf '%s\n' "${value:-$default}"
}

lmcache_url="$(_arg LMCACHE_GIT_URL)"
lmcache_ref="$(_arg LMCACHE_REF)"
nixl_url="$(_arg NIXL_GIT_URL)"
nixl_ref="$(_arg NIXL_REF)"
vllm_url="$(_arg VLLM_GIT_URL)"
vllm_ref="$(_arg VLLM_REF)"
torch_repo="$(_optional_arg PYTORCH_REPO "${PYTORCH_REPO_DEFAULT}")"
torch_ref="$(_arg PYTORCH_BRANCH)"
torchvision_url="$(_arg TORCHVISION_REPO)"
torchvision_ref="$(_arg TORCHVISION_BRANCH)"
aiter_url="$(_arg AITER_GIT_URL)"
aiter_ref="$(_arg AITER_REF)"
flash_attn_url="$(_arg FLASH_ATTN_GIT_URL)"
flash_attn_ref="$(_arg FLASH_ATTN_REF)"
hsa_snoop_url="$(_arg HSA_SNOOP_GIT_URL)"
hsa_snoop_ref="$(_arg HSA_SNOOP_REF)"
rocm_version="$(_arg ROCM_VERSION)"
rocm_base_image="$(_arg ROCM_BASE_IMAGE)"
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
    echo "- **PyTorch:** ${torch_repo} @ \`${torch_ref}\` (source build)"
    echo "- **TorchVision:** ${torchvision_url} @ \`${torchvision_ref}\` (source build)"
    echo "- **vLLM:** ${vllm_url} @ \`${vllm_ref}\` (source build)"
    echo "- **AITER:** ${aiter_url} @ \`${aiter_ref}\` (official ROCm wheel)"
    echo "- **FlashAttention:** ${flash_attn_url} @ \`${flash_attn_ref}\` (source build, CDNA-only)"
    echo "- **LMCache:** ${lmcache_url} @ \`${lmcache_ref}\` + AIC patches"
    echo "- **NIXL:** ${nixl_url} @ \`${nixl_ref}\` + nixl-rocm-ais-mt patch"
    echo "- **hsa-snoop:** ${hsa_snoop_url} @ \`${hsa_snoop_ref}\` (source build)"
    echo "- **hipFile:** from ${rocm_base_image} (GA in ROCm 7.14)"
    echo ""
    echo "### Install wheels"
    echo '```bash'
    echo "pip install \\"
    echo "  https://github.com/${REPOSITORY}/releases/download/${TAG}/<vllm-wheel> \\"
    echo "  https://github.com/${REPOSITORY}/releases/download/${TAG}/<lmcache-wheel> \\"
    echo "  https://github.com/${REPOSITORY}/releases/download/${TAG}/<nixl_rocm-wheel>"
    echo '```'
    echo ""
    echo "### Source"
    echo '```bash'
    echo "curl -LO https://github.com/${REPOSITORY}/releases/download/${TAG}/aic-release-<stamp>.tar.gz"
    echo "curl -LO https://github.com/${REPOSITORY}/releases/download/${TAG}/SHA256SUMS"
    echo "sha256sum -c SHA256SUMS"
    echo '```'
    echo ""
    echo "### Docker"
    echo '```bash'
    echo "docker pull ${IMAGE_NAME}:${TAG}"
    echo '```'
    echo ""
    echo "> These wheels are **not** manylinux: ROCm ${rocm_version} + Python 3.12 + x86_64 only."
    echo "> The \`nixl_rocm\` wheel needs the ROCm runtime (libamdhip64) and hipFile"
    echo "> present on the host; see README.md."
}
