#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Emit the AIC image tag (no image name), e.g.
#   CDNA (gfx9xx):  0.1.0-rocm7.14.1-pytorch2.13-vllm0.29.0-aiter0.1.22.post1-fa2.8.3.post1-lmcache0.5.5-nixl1.4.1-hsasnoop1.2.0-gfx942
#   RDNA (gfx12xx): 0.1.0-rocm7.14.1-pytorch2.13-vllm0.29.0-aiter0.1.22.post1-lmcache0.5.5-nixl1.4.1-hsasnoop1.2.0-gfx1250
#
# The tag is now derived in docker/docker-bake.hcl (the single source of truth
# for version pins and tag format).  This script is a thin wrapper that asks bake
# to resolve the `base` target and strips the image name off its tag, so every
# existing caller keeps working unchanged.
#
# Overrides still work exactly as before: a same-named environment variable
# (ROCM_ARCH, ROCM_VERSION, PYTORCH_SERIES, VLLM_REF, AITER_REF, FLASH_ATTN_REF,
# LMCACHE_REF, NIXL_REF, HSA_SNOOP_REF, ...) overrides the HCL default because
# bake reads variables from the environment.
#
# Usage:  aic-image-tag.sh
#         aic-image-tag.sh [...]   # trailing args accepted for back-compat, ignored
set -Eeuo pipefail
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BAKE_FILE="${REPO_ROOT}/docker/docker-bake.hcl"

[[ -r "${BAKE_FILE}" ]] || { echo "aic-image-tag: cannot read ${BAKE_FILE}" >&2; exit 1; }

# VERSION owns AIC_VERSION; forward it so a bumped VERSION is reflected without
# editing the HCL default.  An explicit AIC_VERSION in the environment still wins.
if [[ -z "${AIC_VERSION:-}" && -r "${REPO_ROOT}/VERSION" ]]; then
  AIC_VERSION="$(<"${REPO_ROOT}/VERSION")"
  export AIC_VERSION
fi

tag="$(
  docker buildx bake -f "${BAKE_FILE}" base --print 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["target"]["base"]["tags"][0].split(":", 1)[1])'
)"

[[ -n "${tag}" ]] || { echo "aic-image-tag: bake --print produced no tag" >&2; exit 1; }
printf '%s\n' "${tag}"
