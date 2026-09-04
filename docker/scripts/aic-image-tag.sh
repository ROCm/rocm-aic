#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Derive the AIC image tag from framework version build args, falling back to
# the versions pinned in the Dockerfile.
#
# Emits just the tag component (no image name) in the following format:
#   0.1.0-rocm7.14.0-vllm0.27.1-lmcache0.5.4-mooncake7197358-nixl1.3.2-hsasnoop1.0.0
# Where 0.1.0 represents the AIC version.  Mooncake is pinned by commit, so it
# contributes the first 7 characters of MOONCAKE_REF.
#
# Optional segments: VLLM_ROCM_VARIANT appended to the vllm component,
# and a hipfile{SHA7} segment when HIPFILE_SHA is set.
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
# VLLM_VERSION env overrides VLLM_REF for pre-built wheel variants.
if [[ -v VLLM_VERSION ]]; then
  vllm="${VLLM_VERSION#v}"
else
  vllm="$(_arg VLLM_REF | sed 's/^v//')"
fi
vllm_variant="${VLLM_ROCM_VARIANT:-}"
lmcache="$(_arg LMCACHE_REF | sed 's/^v//')"
mooncake="$(_arg MOONCAKE_REF)"
nixl="$(_arg NIXL_REF | sed 's/^v//')"
hipfile_sha="${HIPFILE_SHA:-}"
hsasnoop="$(_arg HSA_SNOOP_REF | sed 's/^v//')"

for _v in aic rocm vllm lmcache mooncake nixl hsasnoop; do
  [[ -n "${!_v}" ]] || {
    echo "aic-image-tag: could not resolve ${_v}" >&2
    exit 1
  }
done

tag="${aic}-rocm${rocm}-vllm${vllm}"
[[ -n "${vllm_variant}" ]] && tag="${tag}-${vllm_variant}"
tag="${tag}-lmcache${lmcache}-mooncake${mooncake:0:7}-nixl${nixl}"
[[ -n "${hipfile_sha}" ]] && tag="${tag}-hipfile${hipfile_sha:0:7}"
tag="${tag}-hsasnoop${hsasnoop}"
printf '%s\n' "${tag}"
