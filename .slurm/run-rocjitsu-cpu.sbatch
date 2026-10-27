#!/usr/bin/env bash
#SBATCH --job-name=aic-rocjitsu
#SBATCH --partition=defq
#SBATCH --nodelist=ctr-smc-s26-005
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#
# Run `make test-rocjitsu-local` on a CPU-only node.  rocjitsu emulates the
# gfx1250 GPU in software behind a vfio-user socket, so no AMD GPU is needed --
# only /dev/kvm, which the QEMU container receives via --device and uses as
# uid 0 even where the host user has no access to it.
#
# RJ_FORCE_BUILD defaults to 0, so an arch-tagged image already on the node is
# reused and step [2/6] is skipped.  A cold build is ~50 min on this node.
set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:?}"
cd "${REPO}" || exit 1

export AIC_CACHE_DIR="${AIC_CACHE_DIR:-/tmp/aic-buildx-cache-gfx1250}"
export RJ_WORK_DIR="${RJ_WORK_DIR:-/tmp/aic-rocjitsu-test}"
mkdir -p "${AIC_CACHE_DIR}" "${RJ_WORK_DIR}"

echo "=== node=$(hostname -s) cpus=$(nproc) job=${SLURM_JOB_ID:-none} ==="
echo "=== repo=${REPO}"
echo "=== cache=${AIC_CACHE_DIR}  work=${RJ_WORK_DIR}"

# The guest disk is extracted only when vm-info.json is absent, so a changed
# RJ_QCOW2_IMAGE pin would otherwise keep using the previously extracted disk.
echo "=== dropping ${RJ_WORK_DIR}/vm so the current qcow2 pin is extracted ==="
rm -rf "${RJ_WORK_DIR}/vm"

# Leftover containers from an earlier run hold the vfio socket and the SSH port.
docker rm -f aic-rj-qemu aic-rj-rocjitsu >/dev/null 2>&1 || true

echo "=== gfx1250 images currently on this node ==="
docker images --format '  {{.Repository}}:{{.Tag}}' \
    | grep -E 'aic-(vllm|lmcache|nvme-exporter|rdma-exporter|hsa-snoop)' | sort || echo "  (none)"

echo
echo "########## make test-rocjitsu-local ##########"
# RJ_MAKE_ARGS lets the submitter steer the run without editing this file --
# chiefly to characterise the emulator rather than exercise vLLM:
#   sbatch --export=ALL,RJ_MAKE_ARGS="RJ_PROBE=1 RJ_PROBE_ONLY=1" ...
echo "=== extra make args: ${RJ_MAKE_ARGS:-(none)}"
# shellcheck disable=SC2086 # deliberate word splitting: these are make VAR=VAL
make test-rocjitsu-local \
    ROCM_ARCH=gfx1250 \
    BUILD_JOBS="$(nproc)" \
    RJ_READY_S=600 \
    ${RJ_MAKE_ARGS:-}
_rc=$?

echo
echo "=== test-rocjitsu-local exited ${_rc} ==="

docker rm -f aic-rj-qemu aic-rj-rocjitsu >/dev/null 2>&1 || true
exit "${_rc}"
