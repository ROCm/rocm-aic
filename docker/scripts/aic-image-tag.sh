#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Derive the AIC image tag from framework version build args, falling back to
# the versions pinned in the Dockerfile.
#
# Emits just the tag component (no image name) in the following format:
#   CDNA (gfx9xx):  0.1.0-rocm7.14.1-pytorch2.13-vllm0.28.0-aiter0.1.19-fa0e60e394-lmcache0.5.4-nixl1.4.1-hsasnoop1.1.0
#   RDNA (gfx12xx): 0.1.0-rocm7.14.1-pytorch2.13-vllm0.28.0-aiter0.1.19-lmcache0.5.4-nixl1.4.1-hsasnoop1.1.0
# FlashAttention is omitted for non-CDNA arches (gfx10xx/gfx11xx/gfx12xx) because
# the CK backend does not support RDNA Wave32 GPUs.
# Where 0.1.0 represents the AIC version.
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

# PYTORCH_BRANCH is a branch name (e.g. release/2.13) or a commit hash.
# Extract just the version number for the tag (release/2.13 -> 2.13; a 7-char
# hash is used verbatim when no version is found).
pytorch_raw="$(_arg PYTORCH_BRANCH)"
if [[ "${pytorch_raw}" =~ release/([0-9]+\.[0-9]+) ]]; then
  pytorch="${BASH_REMATCH[1]}"
else
  pytorch="${pytorch_raw:0:7}"
fi

# Refs are git tags like v0.28.0 so we drop the leading v.
vllm="$(_arg VLLM_REF | sed 's/^v//')"
lmcache="$(_arg LMCACHE_REF | sed 's/^v//')"
nixl="$(_arg NIXL_REF | sed 's/^v//')"
aiter="$(_arg AITER_REF | sed 's/^v//')"
flash_attn="$(_arg FLASH_ATTN_REF)"
hsasnoop="$(_arg HSA_SNOOP_REF | sed 's/^v//')"

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
