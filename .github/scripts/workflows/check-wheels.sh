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

# The published AIC wheel set: the single list the nightly and release workflows
# check a `docker buildx bake wheels` export against.  Each pattern must match
# exactly one wheel, and no other wheel may be present, so the release carries
# exactly this set.  Edit this list when the `wheels` stages change.
#
#   vllm-*         docker/vllm/Dockerfile    build_vllm (VLLM_VERSION_OVERRIDE)
#   amd_aiter-*    docker/vllm/Dockerfile    build_vllm
#   flash_attn-*   docker/vllm/Dockerfile    build_vllm, arch lists with a gfx9*
#   lmcache-*      docker/lmcache/Dockerfile build_lmcache_wheel
#   nixl*rocm*     docker/base/Dockerfile    build_nixl (build-nixl.sh's own glob)
EXPECTED_WHEELS=(
    'vllm-*.whl'
    'amd_aiter-*.whl'
    'lmcache-*.whl'
    'nixl*rocm*.whl'
)
# FlashAttention is built when at least one arch is gfx9* (docker/vllm/Dockerfile;
# the same rule as has_gfx9() in docker/docker-bake.hcl, which also treats an
# empty list as CDNA).
GFX9_WHEELS=(
    'flash_attn-*.whl'
)
# GitHub rejects release assets of 2 GiB or more; fail here, not after upload.
MAX_ASSET_BYTES=$((2 * 1024 * 1024 * 1024 - 1))

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <wheel-dir> <rocm-arch>" >&2
    echo "  Checks <wheel-dir> holds exactly the wheel set for the ';'/','/space-separated <rocm-arch> list." >&2
    exit 2
fi
WHEEL_DIR="$1"
ROCM_ARCH="$2"

[[ -d "${WHEEL_DIR}" ]] || { echo "check-wheels: ${WHEEL_DIR} is not a directory" >&2; exit 1; }

read -r -a _archs <<<"${ROCM_ARCH//[;,]/ }"
# An empty list counts as CDNA, like has_gfx9() in docker/docker-bake.hcl.
has_gfx9=0
if [[ ${#_archs[@]} -eq 0 ]]; then
    has_gfx9=1
fi
for _a in "${_archs[@]}"; do
    if [[ "${_a}" == gfx9* ]]; then
        has_gfx9=1
    fi
done

expected=("${EXPECTED_WHEELS[@]}")
if [[ "${has_gfx9}" == 1 ]]; then
    expected+=("${GFX9_WHEELS[@]}")
fi

shopt -s nullglob
present=("${WHEEL_DIR}"/*.whl)
echo "${WHEEL_DIR}: ${#present[@]} wheel(s) for ROCM_ARCH='${ROCM_ARCH}' (FlashAttention: $([[ ${has_gfx9} == 1 ]] && echo yes || echo no))"

failed=0
declare -A matched=()
for pattern in "${expected[@]}"; do
    # Deliberately unquoted: the pattern is a glob.
    # shellcheck disable=SC2206
    hits=("${WHEEL_DIR}"/${pattern})
    if [[ ${#hits[@]} -ne 1 ]]; then
        echo "ERROR: expected exactly one ${pattern}, found ${#hits[@]}:${hits[*]+ ${hits[*]##*/}}" >&2
        failed=1
    fi
    for h in "${hits[@]}"; do matched["${h}"]=1; done
done

for w in "${present[@]}"; do
    size="$(stat -c %s "${w}")"
    printf '  %-72s %12s bytes\n' "${w##*/}" "${size}"
    if [[ -z "${matched[${w}]+x}" ]]; then
        echo "ERROR: unexpected wheel ${w##*/} (not in this script's expected set for this arch list)" >&2
        failed=1
    fi
    if [[ "${size}" -gt "${MAX_ASSET_BYTES}" ]]; then
        echo "ERROR: ${w##*/} is ${size} bytes; GitHub release assets must be under 2 GiB" >&2
        failed=1
    fi
done

if [[ "${failed}" -ne 0 ]]; then
    echo "check-wheels: ${WHEEL_DIR} failed the wheel-set check (expected exactly: ${expected[*]})" >&2
    exit 1
fi
echo "check-wheels: OK (${expected[*]})"
