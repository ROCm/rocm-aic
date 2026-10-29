#!/usr/bin/env bash
# -E inherit ERR trap in functions and subshells,
# -e exit the script upon commands failing,
# -u to error if the script ever references a variable that is unset,
# -o pipefail to error if any stage of a pipeline fails.
set -Eeuo pipefail

# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# errexit exits without saying where; report it.  Only the line number and
# function call stack are printed, never the command text, which can carry
# secret values.
_on_err() {
    # set -E carries this trap into $(...), where errexit is off.
    [[ $- == *e* ]] || return 0
    # || : so a failed write cannot replace the exit status.
    printf '%s: command failed (exit %s, pipeline status %s) at line %s%s\n' \
        "${0##*/}" "$1" "$2" "$3" "${4:+ in ${4// / <- }}" >&2 || :
}
trap '_on_err "$?" "${PIPESTATUS[*]}" "$LINENO" "${FUNCNAME[*]-}"' ERR

# Release notes for a pip-wheel release (the rolling nightly or a v* release),
# printed to stdout.  Version pins come from `docker buildx bake --print wheels`
# -- docker/docker-bake.hcl is authoritative, the Dockerfile ARG defaults are
# only standalone fallbacks -- and the install command lists the wheels actually
# being published.
# This script runs directly from a workflow checkout; it is not installed on a runner.
if [[ $# -lt 6 || $# -gt 7 ]]; then
    echo "usage: $0 <nightly|release> <tag> <source-sha> <gpu-arch> <wheel-dir> <repository> [<image-name>]" >&2
    exit 2
fi

KIND="$1"
TAG="$2"
SOURCE_SHA="$3"
GPU_ARCH="$4"
WHEEL_DIR="$5"
REPOSITORY="$6"
IMAGE_NAME="${7:-}"

case "${KIND}" in
    nightly | release) ;;
    *)
        echo "release notes: kind must be nightly or release, not '${KIND}'" >&2
        exit 2
        ;;
esac
for input in TAG SOURCE_SHA GPU_ARCH REPOSITORY; do
    [[ -n "${!input}" ]] || {
        echo "release notes: ${input} must not be empty" >&2
        exit 1
    }
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BAKE_FILE="${REPO_ROOT}/docker/docker-bake.hcl"
BASE_DOCKERFILE="${REPO_ROOT}/docker/base/Dockerfile"

shopt -s nullglob
wheels=("${WHEEL_DIR}"/*.whl)
[[ ${#wheels[@]} -gt 0 ]] || {
    echo "release notes: no wheels in ${WHEEL_DIR}" >&2
    exit 1
}

# Resolve the same configuration the wheels were built with.
if [[ -z "${AIC_VERSION:-}" && -r "${REPO_ROOT}/VERSION" ]]; then
    AIC_VERSION="$(<"${REPO_ROOT}/VERSION")"
    export AIC_VERSION
fi
bake_err="$(mktemp)"
trap 'rm -f "${bake_err}"' EXIT
bake_json="$(ROCM_ARCH="${GPU_ARCH}" docker buildx bake -f "${BAKE_FILE}" --print wheels 2>"${bake_err}")" || {
    echo "release notes: docker buildx bake --print wheels failed:" >&2
    cat "${bake_err}" >&2
    exit 1
}

# Every build arg across the wheel targets, as KEY=VALUE lines.  A key two
# targets disagree on would make the notes ambiguous, so it is an error.
pins="$(python3 -c '
import json, sys
merged = {}
for target in json.load(sys.stdin)["target"].values():
    for key, value in target.get("args", {}).items():
        if key == "BLAS_CACHE_BUST":
            continue
        if merged.setdefault(key, value) != value:
            sys.exit(f"release notes: {key} differs between wheel targets ({merged[key]!r} vs {value!r})")
for key, value in sorted(merged.items()):
    print(f"{key}={value}")
' <<<"${bake_json}")"
declare -A pin=()
while IFS='=' read -r key value; do
    pin["${key}"]="${value}"
done <<<"${pins}"

for key in ROCM_VERSION PYTORCH_GIT_URL PYTORCH_REF PYTORCH_SERIES VLLM_GIT_URL VLLM_REF \
    AITER_GIT_URL AITER_REF FLASH_ATTN_GIT_URL FLASH_ATTN_REF LMCACHE_GIT_URL LMCACHE_REF \
    NIXL_GIT_URL NIXL_REF; do
    [[ -n "${pin[${key}]:-}" ]] || {
        echo "release notes: bake resolved no ${key} for the wheel targets" >&2
        exit 1
    }
done
rocm_version="${pin[ROCM_VERSION]}"

# The build image is not a bake variable; it is base's ROCM_BASE_IMAGE default.
rocm_base_image="$(awk -v prefix="ARG ROCM_BASE_IMAGE=" '
    index($0, prefix) == 1 { print substr($0, length(prefix) + 1); exit }
' "${BASE_DOCKERFILE}")"
rocm_base_image="${rocm_base_image//\$\{ROCM_VERSION\}/${rocm_version}}"
case "${rocm_base_image}" in
    '' | *\$\{*)
        echo "release notes: cannot resolve ROCM_BASE_IMAGE from ${BASE_DOCKERFILE} ('${rocm_base_image}')" >&2
        exit 1
        ;;
esac

has_flash_attn=0
for w in "${wheels[@]}"; do
    if [[ "${w##*/}" == flash_attn-* ]]; then has_flash_attn=1; fi
done

{
    if [[ "${KIND}" == nightly ]]; then
        echo "Nightly ROCm pip wheels for the **AMD Infinity Context (AIC)** patched stack."
    else
        echo "Stable release of the **AMD Infinity Context (AIC)** patched stack."
    fi
    echo ""
    echo "- **Tag:** \`${TAG}\`"
    echo "- **Source SHA:** \`${SOURCE_SHA}\`"
    if [[ -n "${GITHUB_RUN_ID:-}" && -n "${GITHUB_SERVER_URL:-}" ]]; then
        echo "- **Built by:** ${GITHUB_SERVER_URL}/${REPOSITORY}/actions/runs/${GITHUB_RUN_ID}"
    fi
    echo "- **GPU arch set:** \`${GPU_ARCH}\`"
    echo "- **Build base:** ${rocm_base_image} (ROCm ${rocm_version}, Python 3.12, x86_64)"
    echo "- **PyTorch (built against, not included):** ${pin[PYTORCH_GIT_URL]} @ \`${pin[PYTORCH_REF]}\` (${pin[PYTORCH_SERIES]} series, source build)"
    echo "- **vLLM:** ${pin[VLLM_GIT_URL]} @ \`${pin[VLLM_REF]}\` + AIC patches (source build)"
    echo "- **AITER:** ${pin[AITER_GIT_URL]} @ \`${pin[AITER_REF]}\`"
    if [[ "${has_flash_attn}" == 1 ]]; then
        echo "- **FlashAttention:** ${pin[FLASH_ATTN_GIT_URL]} @ \`${pin[FLASH_ATTN_REF]}\`"
    fi
    echo "- **LMCache:** ${pin[LMCACHE_GIT_URL]} @ \`${pin[LMCACHE_REF]}\` + AIC patches"
    echo "- **NIXL:** ${pin[NIXL_GIT_URL]} @ \`${pin[NIXL_REF]}\` + nixl-rocm-ais-mt patch"
    echo ""
    echo "### Install wheels"
    echo '```bash'
    echo "pip install \\"
    last=$((${#wheels[@]} - 1))
    for i in "${!wheels[@]}"; do
        url="https://github.com/${REPOSITORY}/releases/download/${TAG}/${wheels[${i}]##*/}"
        if [[ "${i}" -lt "${last}" ]]; then
            echo "  ${url} \\"
        else
            echo "  ${url}"
        fi
    done
    echo '```'
    echo ""
    if [[ -n "${IMAGE_NAME}" ]]; then
        echo "### Docker"
        echo '```bash'
        echo "docker pull ${IMAGE_NAME}:${TAG}"
        echo '```'
        echo ""
    fi
    echo "> These wheels are **not** manylinux: ROCm ${rocm_version} + Python 3.12 + x86_64 only,"
    echo "> with GPU code for the arch set above.  Their C++ extensions link against the"
    echo "> PyTorch build named above, which is not published here."
    echo "> The \`nixl_rocm\` wheel needs the ROCm runtime (libamdhip64) and hipFile on the host."
    echo "> See [docs/PIP_WHEELS.md](https://github.com/${REPOSITORY}/blob/main/docs/PIP_WHEELS.md)."
}
