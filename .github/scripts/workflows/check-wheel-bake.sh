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

# PR-time guard for the wheel publishing workflows, which otherwise only run
# nightly or on a v* tag.  Without building anything, it checks that
#   1. `docker buildx bake --print wheels` resolves for every arch list those
#      workflows build, with a non-empty ROCM_VERSION, the ';'-joined arch
#      list, aic-base wired in as a build context, and a local wheel export;
#   2. check-wheels.sh accepts each group's expected wheel set and rejects a
#      missing or extra wheel.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BAKE_FILE="${REPO_ROOT}/docker/docker-bake.hcl"
NIGHTLY="${REPO_ROOT}/.github/workflows/rocm-aic-nightly-wheels-publication.yml"
RELEASE="${REPO_ROOT}/.github/workflows/rocm-aic-release-workflow.yml"
CHECK_WHEELS="${REPO_ROOT}/.github/scripts/workflows/check-wheels.sh"

# The arch lists exactly as the workflows set them (`  NAME: '<list>'`).
workflow_env() {
    local file="$1" name="$2" value
    value="$(sed -n "s/^  ${name}: '\(.*\)'\$/\1/p" "${file}")"
    [[ -n "${value}" ]] || { echo "check-wheel-bake: no ${name} in ${file}" >&2; return 1; }
    printf '%s\n' "${value}"
}
cdna="$(workflow_env "${NIGHTLY}" CDNA_ARCH)"
rdna="$(workflow_env "${NIGHTLY}" RDNA_ARCH)"
gfx1250="$(workflow_env "${NIGHTLY}" GFX1250_ARCH)"
release="$(workflow_env "${RELEASE}" ROCM_ARCH)"

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

# The nightly workflow writes its arch lists ','-separated (they travel through
# ssh to the SPUR head node, where a ';' would terminate the remote command),
# so this also covers bake's rocm_arch_list() re-joining them with ';'.
for arch in "${cdna}" "${rdna}" "${gfx1250}" "${release}"; do
    echo "--- bake --print wheels, ROCM_ARCH='${arch}'"
    ROCM_ARCH="${arch}" docker buildx bake -f "${BAKE_FILE}" --print wheels \
        >"${tmp}/bake.json" 2>"${tmp}/bake.err" || {
        cat "${tmp}/bake.err" >&2
        exit 1
    }
    python3 - "${tmp}/bake.json" "${arch}" <<'PY'
import json, re, sys

with open(sys.argv[1]) as f:
    d = json.load(f)
arch = ";".join(re.findall(r"[^;, ]+", sys.argv[2]))
errors = []
wanted = ["wheels-base", "wheels-lmcache", "wheels-vllm"]
group = sorted(d["group"]["wheels"]["targets"])
if group != wanted:
    errors.append(f"wheels group is {group}, expected {wanted}")
for name in wanted:
    t = d["target"].get(name)
    if t is None:
        errors.append(f"{name}: missing")
        continue
    args = t.get("args", {})
    outs = t.get("output", [])
    base_ctx = t.get("contexts", {}).get("base")
    if t.get("target") != "wheels":
        errors.append(f"{name}: builds stage {t.get('target')!r}, not wheels")
    if not args.get("ROCM_VERSION"):
        errors.append(f"{name}: ROCM_VERSION is empty")
    if args.get("ROCM_ARCH") != arch:
        errors.append(f"{name}: ROCM_ARCH is {args.get('ROCM_ARCH')!r}, expected {arch!r}")
    if len(outs) != 1 or outs[0].get("type") != "local" or not outs[0].get("dest"):
        errors.append(f"{name}: output is {outs}, expected one type=local export")
    if name != "wheels-base" and base_ctx != "target:base":
        errors.append(f"{name}: base context is {base_ctx!r}, expected target:base")
    print(f"  {name}: ROCM_VERSION={args.get('ROCM_VERSION')} ROCM_ARCH={args.get('ROCM_ARCH')} -> {outs}")
for e in errors:
    print(f"ERROR: {e}", file=sys.stderr)
sys.exit(1 if errors else 0)
PY
done

echo "--- check-wheels.sh against fabricated wheel sets"
fake_set() {
    local dir="$1" n
    shift
    mkdir -p "${dir}"
    for n in "$@"; do : >"${dir}/${n}"; done
}
common=(
    vllm-0.29.0+rocm-cp38-abi3-linux_x86_64.whl
    amd_aiter-0.1.22.post1-cp312-cp312-linux_x86_64.whl
    lmcache-0.5.6-cp312-cp312-linux_x86_64.whl
    nixl_rocm-1.4.1-cp312-cp312-linux_x86_64.whl
)
fa=flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl
fake_set "${tmp}/cdna" "${common[@]}" "${fa}"
fake_set "${tmp}/rdna" "${common[@]}"
bash "${CHECK_WHEELS}" "${tmp}/cdna" "${cdna}"
bash "${CHECK_WHEELS}" "${tmp}/rdna" "${rdna}"
# gfx1250 is not gfx9*, so it takes the no-FlashAttention set like RDNA.
bash "${CHECK_WHEELS}" "${tmp}/rdna" "${gfx1250}"
bash "${CHECK_WHEELS}" "${tmp}/rdna" "${release}"
for bad in "rdna:${cdna}" "cdna:${rdna}" "cdna:${gfx1250}"; do
    if bash "${CHECK_WHEELS}" "${tmp}/${bad%%:*}" "${bad#*:}" >/dev/null 2>&1; then
        echo "ERROR: check-wheels.sh accepted the ${bad%%:*} set for ROCM_ARCH='${bad#*:}'" >&2
        exit 1
    fi
done
echo "check-wheel-bake: OK"
