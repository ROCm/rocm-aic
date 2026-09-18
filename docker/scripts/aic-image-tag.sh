#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Derive the AIC image tag from framework version build args, falling back to
# the versions pinned in the split Dockerfiles.
#
# Emits just the tag component (no image name) in the following format:
#   CDNA (gfx9xx):  0.1.0-rocm7.14.1-pytorch2.13-vllm0.29.0-aiter0.1.22.post1-fa2.8.3.post1-lmcache0.5.5-nixl1.4.1-hsasnoop1.1.1
#   RDNA (gfx12xx): 0.1.0-rocm7.14.1-pytorch2.13-vllm0.29.0-aiter0.1.22.post1-lmcache0.5.5-nixl1.4.1-hsasnoop1.1.1
# FlashAttention is omitted for non-CDNA arches (gfx10xx/gfx11xx/gfx12xx) because
# the CK backend does not support RDNA Wave32 GPUs.
# Where 0.1.0 represents the AIC version.
#
# Usage:  aic-image-tag.sh
#         aic-image-tag.sh [path/to/base/Dockerfile] [path/to/vllm/Dockerfile] [path/to/lmcache/Dockerfile]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_DOCKERFILE="${1:-${SCRIPT_DIR}/../base/Dockerfile}"
VLLM_DOCKERFILE="${2:-${SCRIPT_DIR}/../vllm/Dockerfile}"
LMCACHE_DOCKERFILE="${3:-${SCRIPT_DIR}/../lmcache/Dockerfile}"
VERSION_FILE="${REPO_ROOT}/VERSION"

for f in "${BASE_DOCKERFILE}" "${VLLM_DOCKERFILE}" "${LMCACHE_DOCKERFILE}" "${VERSION_FILE}"; do
  [[ -r "${f}" ]] || { echo "aic-image-tag: cannot read ${f}" >&2; exit 1; }
done

aic="$(<"${VERSION_FILE}")"

# A same-named environment variable represents a user-provided build-arg
# override.  Honour even an explicitly empty override so validation below fails
# instead of silently producing a tag for the Dockerfile default.
# Probe the correct Dockerfile for each ARG.
_arg() {
  local name="$1" dockerfile="$2"
  if [[ -v "${name}" ]]; then
    printf '%s\n' "${!name}"
  else
    grep -E "^ARG ${name}=" "${dockerfile}" | head -1 | cut -d= -f2-
  fi
}

rocm="$(_arg ROCM_VERSION "${BASE_DOCKERFILE}")"

# PYTORCH_BRANCH is a branch name (e.g. release/2.13) or a commit hash.
pytorch_raw="$(_arg PYTORCH_BRANCH "${BASE_DOCKERFILE}")"
if [[ "${pytorch_raw}" =~ release/([0-9]+\.[0-9]+) ]]; then
  pytorch="${BASH_REMATCH[1]}"
else
  pytorch="${pytorch_raw:0:7}"
fi

# Refs are git tags like v0.29.0 so we drop the leading v.
vllm="$(_arg VLLM_REF "${VLLM_DOCKERFILE}" | sed 's/^v//')"
lmcache="$(_arg LMCACHE_REF "${LMCACHE_DOCKERFILE}" | sed 's/^v//')"
nixl="$(_arg NIXL_REF "${LMCACHE_DOCKERFILE}" | sed 's/^v//')"
aiter="$(_arg AITER_REF "${LMCACHE_DOCKERFILE}" | sed 's/^v//')"
flash_attn="$(_arg FLASH_ATTN_REF "${LMCACHE_DOCKERFILE}" | sed 's/^v//')"
hsasnoop="$(_arg HSA_SNOOP_REF "${LMCACHE_DOCKERFILE}" | sed 's/^v//')"

# FlashAttention is CDNA-only (gfx9xx). Detect from ROCM_ARCH env (set by make/caller)
# or fall back to including fa in the tag when arch is unknown (conservative default).
_cdna_only() {
  local a
  for a in $(echo "$1" | tr ';' '\n'); do
    [[ "$a" == gfx9* ]] || return 1
  done
  return 0
}
if [[ -n "${ROCM_ARCH:-}" ]] && ! _cdna_only "${ROCM_ARCH}"; then
  include_fa=0
else
  include_fa=1
fi

for _v in aic rocm pytorch vllm aiter lmcache nixl hsasnoop; do
  [[ -n "${!_v}" ]] || {
    echo "aic-image-tag: could not resolve ${_v}" >&2
    exit 1
  }
done
[[ "${include_fa}" -eq 1 ]] && [[ -z "${flash_attn}" ]] && {
  echo "aic-image-tag: could not resolve flash_attn" >&2
  exit 1
}

tag="${aic}-rocm${rocm}-pytorch${pytorch}-vllm${vllm}-aiter${aiter}"
[[ "${include_fa}" -eq 1 ]] && tag="${tag}-fa${flash_attn}"
tag="${tag}-lmcache${lmcache}-nixl${nixl}-hsasnoop${hsasnoop}"
printf '%s\n' "${tag}"
