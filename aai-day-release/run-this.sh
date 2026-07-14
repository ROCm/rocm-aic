#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Convenience driver for the aai-day-release build + distribute flow.
#
#   ./run-this.sh --build       # build the image (+ nvme/rdma exporter images) on a CPU-only alola node + save tarballs
#   ./run-this.sh --push        # push the built image to a registry (needs AAI_PUSH_REF)
#   ./run-this.sh --run-smoke-test  # smoke-test the image on a GPU+NVMe node
#   ./run-this.sh --cliff       # sbatch the full run_cliff.py sweep on a GPU+NVMe node
#   ./run-this.sh --cliff-short # sbatch a 1-point cliff (concur=1, 1 iter) to smoke-test the flow
#   ./run-this.sh --build --push --run-smoke-test   # build, push, then smoke-test (in order)
#
# Node selection is via constraints in run-build-distribute.sh:
#   build : MARKHAM&CPUONLY        (override with AAI_BUILD_CONSTRAINT / AAI_BUILD_NODE)
#   test  : MARKHAM&GFX942&NVME    (override with AAI_TEST_CONSTRAINT  / AAI_TEST_NODE)
#   cliff : MARKHAM&GFX942&NVME    (override with AAI_CLIFF_NODE, or edit the sbatch)
# Common overrides: AAI_ROCM_ARCH=gfx942, AAI_BUILD_NODE=ctr2-alola-compile-11
#   --push needs AAI_PUSH_REF, e.g. registry-sc-harbor.amd.com/<proj>/aai-day:latest
#   (run `docker login <registry>` once first)
#   --cliff submits .slurm/run-cliff.sbatch; pin a node with AAI_CLIFF_NODE=<node>
#   --cliff-short is --cliff with AAI_BENCH_CONCUR=1 AAI_BENCH_ITERS=1 (fast setup check);
#     it runs all three arms unless narrowed with --cliff-arm
#   --cliff-arm=<list> runs only the named arms (vram,nvme,gds; default all), e.g.
#     --cliff-arm=nvme for just the AIS_MT NVMe arm; implies --cliff, composes with
#     --cliff-short (fastest single-arm check)
#   --cliff auto-starts a Prometheus metrics sidecar (AAI_MONITORING=0 to skip);
#     set AAI_METRICS_DIR=<nfs-dir> for the TSDB, AAI_EXPORTERS=0 to use host exporters
#   --build also builds the nvme/rdma exporter images (AAI_BUILD_EXPORTERS=0 to skip);
#     --cliff then auto-loads them on the node and points the sidecar at them when
#     present (else it falls back to host exporters / node-exporter collectors)
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
do_cliff_short=0
for arg in "$@"; do
    case "${arg}" in
        --build)          do_build=1 ;;
        --push)           do_push=1 ;;
        --run-smoke-test) do_test=1 ;;
        --cliff)          do_cliff=1 ;;
        --cliff-short)    do_cliff=1; do_cliff_short=1 ;;
        --cliff-arm=*)    do_cliff=1; export AAI_CLIFF_ARMS="${arg#*=}" ;;
        -h|--help)  awk 'NR>=7{ if(/^#/){sub(/^# ?/,"");print} else exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)          echo "unknown arg: ${arg} (use --build, --push, --run-smoke-test, --cliff, --cliff-short, --cliff-arm=<list>)" >&2; exit 1 ;;
    esac
done

if (( do_build )); then
    # picks any idle CPU-only Markham build node, builds, and saves the tarball
    "${DIST}" build
    # Also build + save the fabric exporter images (nvme_exporter / rdma_exporter)
    # so --cliff can auto-load them on bare nodes.  Skip with AAI_BUILD_EXPORTERS=0.
    if [[ "${AAI_BUILD_EXPORTERS:-1}" == "1" ]]; then
        "${DIST}" build-exporters
    fi
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
    if (( do_cliff_short )); then
        # Fast setup check: a single concurrency point, one timed iteration.  Still
        # runs all three arms (vram_only, kvd_v2 nvme, kvd_v2 gds) so the AIS_MT and
        # GDS paths are exercised.  Exported so the sbatch job inherits them; user
        # overrides of either variable are respected.
        export AAI_BENCH_CONCUR="${AAI_BENCH_CONCUR:-1}"
        export AAI_BENCH_ITERS="${AAI_BENCH_ITERS:-1}"
        sbatch_args+=(--job-name=aai-day-cliff-short)
        echo "cliff-short: single point (AAI_BENCH_CONCUR=${AAI_BENCH_CONCUR} AAI_BENCH_ITERS=${AAI_BENCH_ITERS})"
    fi
    # The job creates logs/<job-id>/ itself and redirects its output there
    # (Slurm's own --output is /dev/null), so nothing to pre-create here.
    jobid="$(cd "${HERE}" && sbatch --parsable "${sbatch_args[@]}" .slurm/run-cliff.sbatch)"
    echo "submitted cliff job ${jobid}"
    echo "log: ${HERE}/logs/${jobid}/cliff.out"
fi
