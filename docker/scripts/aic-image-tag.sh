#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Derive the AIC image tag from framework version build args, falling back to
# the versions pinned in the Dockerfile.
#
# Emits just the tag component (no image name) in the following format:
#   0.1.0-rocm7.14.0-vllm0.26.0-lmcache0.5.2-nixl1.3.2-hsasnoop1.0.0
# Where 0.1.0 represents the AIC version.
#
# hipFile has no component here: since ROCm 7.14 it ships in the base image, so
# it is pinned by ROCM_VERSION rather than independently.
#
# Usage:  aic-image-tag.sh [path/to/Dockerfile]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKERFILE="${1:-${SCRIPT_DIR}/../Dockerfile}"
VERSION_FILE="${REPO_ROOT}/VERSION"
[[ -r "${DOCKERFILE}" ]] || {
  echo "aic-image-tag: cannot read ${DOCKERFILE}" >&2
  exit 1
}
[[ -r "${VERSION_FILE}" ]] || {
  echo "aic-image-tag: cannot read ${VERSION_FILE}" >&2
  exit 1
}

aic="$(<"${VERSION_FILE}")"

# A same-named environment variable represents a user-provided build-arg
# override. Honour even an explicitly empty override so validation below fails
# instead of silently producing a tag for the Dockerfile default.
_arg() {
  if [[ -v "$1" ]]; then
    printf '%s\n' "${!1}"
  else
    grep -E "^ARG $1=" "${DOCKERFILE}" | head -1 | cut -d= -f2-
  fi
}

rocm="$(_arg ROCM_VERSION)"

# Refs are git tags like v0.5.1 so we drop the leading v.
# vLLM is built from source, so VLLM_REF is the pin (no prebuilt-wheel variant).
vllm="$(_arg VLLM_REF | sed 's/^v//')"
lmcache="$(_arg LMCACHE_REF | sed 's/^v//')"
nixl="$(_arg NIXL_REF | sed 's/^v//')"
hsasnoop="$(_arg HSA_SNOOP_REF | sed 's/^v//')"

for _v in aic rocm vllm lmcache nixl hsasnoop; do
  [[ -n "${!_v}" ]] || {
    echo "aic-image-tag: could not resolve ${_v}" >&2
    exit 1
  }
done

printf '%s-rocm%s-vllm%s-lmcache%s-nixl%s-hsasnoop%s\n' \
  "${aic}" "${rocm}" "${vllm}" "${lmcache}" "${nixl}" "${hsasnoop}"
