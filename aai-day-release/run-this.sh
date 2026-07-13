#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Convenience driver for the aai-day-release build + distribute flow.
#
#   ./run-this.sh --build       # build the image on a CPU-only alola node + save tarball
#   ./run-this.sh --push        # push the built image to a registry (needs AAI_PUSH_REF)
#   ./run-this.sh --run-test    # smoke-test the image on a GPU+NVMe node
#   ./run-this.sh --cliff       # sbatch the full run_cliff.py sweep on a GPU+NVMe node
#   ./run-this.sh --build --push --run-test   # build, push, then test (runs in order)
#
# Node selection is via constraints in run-build-distribute.sh:
#   build : MARKHAM&CPUONLY        (override with AAI_BUILD_CONSTRAINT / AAI_BUILD_NODE)
#   test  : MARKHAM&GFX942&NVME    (override with AAI_TEST_CONSTRAINT  / AAI_TEST_NODE)
#   cliff : MARKHAM&GFX942&NVME    (override with AAI_CLIFF_NODE, or edit the sbatch)
# Common overrides: AAI_ROCM_ARCH=gfx942, AAI_BUILD_NODE=ctr2-alola-compile-11
#   --push needs AAI_PUSH_REF, e.g. registry-sc-harbor.amd.com/<proj>/aai-day:latest
#   (run `docker login <registry>` once first)
#   --cliff submits .slurm/run-cliff.sbatch; pin a node with AAI_CLIFF_NODE=<node>
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST="${HERE}/.slurm/run-build-distribute.sh"

# Shared BuildKit build cache (file-based): every good layer is written to a dir
# on shared /scratch as it builds, so a failed build resumes from the last good
# layer on ANY node instead of restarting from scratch.  /scratch (BeeGFS) is
# mounted on every build node, so no registry, auth, or TLS is involved -- the
# script namespaces the cache per-arch under this dir.  Override in the
# environment, or set AAI_CACHE_DIR= (empty) to fall back to the no-cache build.
# (To use a registry cache instead, export AAI_CACHE_REF; it takes precedence.)
export AAI_CACHE_DIR="${AAI_CACHE_DIR-/scratch/${USER}/images/buildcache}"

do_build=0
do_push=0
do_test=0
do_cliff=0
for arg in "$@"; do
    case "${arg}" in
        --build)    do_build=1 ;;
        --push)     do_push=1 ;;
        --run-test) do_test=1 ;;
        --cliff)    do_cliff=1 ;;
        -h|--help)  sed -n '7,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)          echo "unknown arg: ${arg} (use --build, --push, --run-test and/or --cliff)" >&2; exit 1 ;;
    esac
done

if (( do_build )); then
    # picks any idle CPU-only Markham build node, builds, and saves the tarball
    "${DIST}" build
fi

if (( do_push )); then
    # tags the built image as AAI_PUSH_REF and pushes it to the registry
    "${DIST}" push
fi

if (( do_test )); then
    # loads the image on a GPU+NVMe node if needed, then runs the smoke test
    "${DIST}" test
fi

if (( do_cliff )); then
    # submit the full cliff sweep (vram_only + kvd_v2 nvme + kvd_v2 gds) as a
    # batch job on a Markham GPU+NVMe node.  Loads the image from the shared
    # tarball on the node if absent.  Pin a node with AAI_CLIFF_NODE=<node>.
    command -v sbatch >/dev/null 2>&1 || { echo "sbatch not found" >&2; exit 1; }
    sbatch_args=()
    [[ -n "${AAI_CLIFF_NODE:-}" ]] && sbatch_args+=(--nodelist="${AAI_CLIFF_NODE}")
    ( cd "${HERE}" && sbatch "${sbatch_args[@]}" .slurm/run-cliff.sbatch )
fi
