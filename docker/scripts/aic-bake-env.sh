#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Print the build inputs docker/docker-bake.hcl resolves, for everything that
# is not bake itself (make, Slurm scripts, CI, the README sync).  The HCL is the
# only place a build default lives; this runs `docker buildx bake --print` once
# and reports what bake would use, so no caller keeps its own copy.
#
# A same-named environment variable overrides an HCL default exactly as it does
# for bake (including an empty value), because bake itself does the resolving.
# AIC_VERSION is exported from the repo-root VERSION file when unset or empty.
#
# Usage:
#   aic-bake-env.sh --env
#       Shell-evaluable KEY='value' lines, sorted:  eval "$(aic-bake-env.sh --env)"
#   aic-bake-env.sh --get NAME [NAME...]
#       The value of each NAME, one per line.  Unknown NAME: error, exit 1,
#       nothing on stdout.
#   aic-bake-env.sh --json
#       The same map as one JSON object.
#   aic-bake-env.sh --make
#       `KEY ?= value` lines for a Makefile to -include ('#' and '$' escaped).
#       `?=` so a value make already holds is never replaced.
#   aic-bake-env.sh --build-args <target>
#       One `--build-arg=KEY=VALUE` word per line for exactly that bake target's
#       resolved args (empty ones and base's BLAS_CACHE_BUST included, as bake
#       passes them), for raw `docker buildx build` callers:
#           mapfile -t args < <(aic-bake-env.sh --build-args vllm)
#           docker buildx build "${args[@]}" ...
#   aic-bake-env.sh --variables
#       The name of every bake variable, one per line, sorted: what a caller
#       must carry into another shell (a Slurm job) for bake to resolve the same
#       values there.  From `docker buildx bake --list=variables`; where docker
#       buildx is missing or its bake has no --list (a Slurm login node) the
#       names are read off the HCL's `variable "NAME"` lines, and
#       tests/test_aic_bake_env.py keeps the two equal.  Names only: values
#       always come from bake.
#
# The map behind --env/--get/--json/--make holds:
#   - every build arg of the image targets (base vllm lmcache hsa-snoop emulate
#     kvbench nvme-exporter rdma-exporter agentx) merged by name.
#     BLAS_CACHE_BUST is left out (a timestamp), VLLM_TARGET_DEVICE is the vllm
#     target's (emulate overrides it to "empty"), and ROCM_ARCH is bake's
#     ';'-joined form.  Any other arg two targets disagree on is an error.
#   - IMAGE_TAG: the version tag shared by the stack images, without a name.
#   - the first tag of each target as a full image ref:
#       AIC_BASE_IMAGE_REF       base
#       VLLM_IMAGE_REF           vllm
#       LMCACHE_IMAGE_REF        lmcache
#       HSA_SNOOP_IMAGE_REF      hsa-snoop
#       VLLM_EMULATE_IMAGE_REF   emulate
#       KVBENCH_IMAGE_REF        kvbench
#       NVME_EXPORTER_IMAGE_REF  nvme-exporter
#       RDMA_EXPORTER_IMAGE_REF  rdma-exporter
#       AGENTX_IMAGE_REF         agentx
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

usage() {
  echo "usage: aic-bake-env.sh --env | --json | --make | --variables | --get NAME [NAME...] | --build-args <target>" >&2
  exit 2
}

mode="${1:-}"
[[ $# -gt 0 ]] && shift
case "${mode}" in
  --env | --json | --make | --variables) [[ $# -eq 0 ]] || usage ;;
  --get) [[ $# -ge 1 ]] || usage ;;
  --build-args) [[ $# -eq 1 && -n "$1" ]] || usage ;;
  *) usage ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BAKE_FILE="${REPO_ROOT}/docker/docker-bake.hcl"

# `bake --print` stats the secret file, and no mode reports it: a TLS_CERT that
# is absent on this host must not fail the query.
unset TLS_CERT

[[ -r "${BAKE_FILE}" ]] || { echo "aic-bake-env: cannot read ${BAKE_FILE}" >&2; exit 1; }

if [[ -z "${AIC_VERSION:-}" && -r "${REPO_ROOT}/VERSION" ]]; then
  AIC_VERSION="$(<"${REPO_ROOT}/VERSION")"
  export AIC_VERSION
fi

if [[ "${mode}" == --variables ]]; then
  hcl_names() {
    sed -n 's/^variable "\([A-Za-z_][A-Za-z0-9_]*\)".*/\1/p' "${BAKE_FILE}" | LC_ALL=C sort
  }
  if ! docker buildx version >/dev/null 2>&1; then
    hcl_names
    exit 0
  fi
  list_err="$(mktemp)"
  trap 'rm -f "${list_err}"' EXIT
  # A buildx whose bake predates --list (a login node) still gets the names.
  if ! list_json="$(docker buildx bake -f "${BAKE_FILE}" --list=type=variables,format=json 2>"${list_err}")"; then
    echo "aic-bake-env: warning: docker buildx bake --list=variables failed; reading the names off the HCL:" >&2
    cat "${list_err}" >&2
    hcl_names
    exit 0
  fi
  python3 -c 'import json, sys; print("\n".join(sorted(v["name"] for v in json.load(sys.stdin))))' \
    <<<"${list_json}" || exit $?
  exit 0
fi

IMAGE_TARGETS=(base vllm lmcache hsa-snoop emulate kvbench nvme-exporter rdma-exporter agentx)
if [[ "${mode}" == --build-args ]]; then
  targets=("$1")
else
  targets=("${IMAGE_TARGETS[@]}")
fi

# Keep bake's stderr off the JSON, but show it on failure: it is the only
# clue when buildx is missing or the HCL does not parse.
bake_err="$(mktemp)"
trap 'rm -f "${bake_err}"' EXIT
if ! bake_json="$(docker buildx bake -f "${BAKE_FILE}" --print "${targets[@]}" 2>"${bake_err}")"; then
  echo "aic-bake-env: docker buildx bake --print ${targets[*]} failed:" >&2
  cat "${bake_err}" >&2
  exit 1
fi

read -r -d '' render <<'PY' || :
import json
import sys

mode, names = sys.argv[1], sys.argv[2:]
targets = json.load(sys.stdin)["target"]


def die(message):
    sys.exit(f"aic-bake-env: {message}")


if mode == "--build-args":
    if names[0] not in targets:
        die(f"bake resolved no target {names[0]!r}")
    for key, value in sorted(targets[names[0]].get("args", {}).items()):
        if "\n" in value:
            die(f"{key} contains a newline")
        print(f"--build-arg={key}={value}")
    sys.exit(0)

IMAGE_REFS = {
    "AIC_BASE_IMAGE_REF": "base",
    "VLLM_IMAGE_REF": "vllm",
    "LMCACHE_IMAGE_REF": "lmcache",
    "HSA_SNOOP_IMAGE_REF": "hsa-snoop",
    "VLLM_EMULATE_IMAGE_REF": "emulate",
    "KVBENCH_IMAGE_REF": "kvbench",
    "NVME_EXPORTER_IMAGE_REF": "nvme-exporter",
    "RDMA_EXPORTER_IMAGE_REF": "rdma-exporter",
    "AGENTX_IMAGE_REF": "agentx",
}

values = {}
for name in IMAGE_REFS.values():
    if name not in targets:
        die(f"bake resolved no target {name!r}")
    for key, value in targets[name].get("args", {}).items():
        if key == "BLAS_CACHE_BUST" or (key == "VLLM_TARGET_DEVICE" and name != "vllm"):
            continue
        if values.setdefault(key, value) != value:
            die(f"{key} differs between bake targets ({values[key]!r} vs {value!r})")
for ref, name in IMAGE_REFS.items():
    tags = targets[name].get("tags") or []
    if not tags or not tags[0]:
        die(f"bake target {name!r} has no tag")
    values[ref] = tags[0]
# The tag is everything after the last ':' -- a registry prefix may carry a port.
values["IMAGE_TAG"] = values["AIC_BASE_IMAGE_REF"].rpartition(":")[2]

if mode == "--get":
    unknown = [name for name in names if name not in values]
    if unknown:
        die(f"unknown name(s): {' '.join(unknown)}")
    for name in names:
        print(values[name])
elif mode == "--json":
    json.dump(values, sys.stdout, indent=2, sort_keys=True)
    print()
elif mode == "--env":
    for key, value in sorted(values.items()):
        print("{}='{}'".format(key, value.replace("'", "'\\''")))
elif mode == "--make":
    for key, value in sorted(values.items()):
        if "\n" in value:
            die(f"{key} contains a newline; make cannot hold it")
        print("{} ?= {}".format(key, value.replace("$", "$$").replace("#", "\\#")))
PY

# The renderer reports its own errors; || exit keeps the ERR trap quiet.
python3 -c "${render}" "${mode}" "$@" <<<"${bake_json}" || exit $?
