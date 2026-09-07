#!/usr/bin/env bash
set -euo pipefail

# Installed on the self-hosted runner; SSHes to the SPUR head node (AIC_SPUR_HOST), uses the clone and
# tarball left by spur-dist-build.sh, and runs a cliff benchmark.
#
# Usage: spur-cliff.sh <full-sha> <target>
#   target: cliff-short  -- 1-point sweep, quick PR gate
#           cliff-submit -- full 3-arm sweep, nightly/post-merge
#
# Always cleans up the clone and tarball dir on exit.

SHA="${1:?usage: $0 <full-sha> <cliff-short|cliff-submit>}"
TARGET="${2:?usage: $0 <full-sha> <cliff-short|cliff-submit>}"
SHORT="${SHA:0:7}"
AIC_IMAGE_NAME="rocm-aic-ci-${SHORT}"
AIC_SPUR_HOST="${AIC_SPUR_HOST:?AIC_SPUR_HOST must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_HOST="${AIC_SPUR_HOST//[$'\t\r\n ']}"
AIC_SHARED_NFS="${AIC_SHARED_NFS:?AIC_SHARED_NFS must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER:?AIC_SPUR_CONTROLLER must be set (e.g. via GitHub repo variable)}"
AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-}"
REPO="https://github.com/ROCm/rocm-aic.git"

case "${TARGET}" in
    cliff-short|cliff-submit) ;;
    *) echo "ERROR: target must be cliff-short or cliff-submit" >&2; exit 1 ;;
esac

ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "${AIC_SPUR_HOST}" env \
    SHA="${SHA}" \
    REPO="${REPO}" \
    TARGET="${TARGET}" \
    AIC_IMAGE_NAME="${AIC_IMAGE_NAME}" \
    AIC_SHARED_NFS="${AIC_SHARED_NFS}" \
    AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT}" \
    AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER}" \
    SPUR_CONTROLLER_ADDR="${AIC_SPUR_CONTROLLER}" \
    bash << 'REMOTE'
set -euo pipefail
export PATH="/usr/local/bin:${PATH}"

SHORT="${SHA:0:7}"
WORKDIR="$HOME/Projects/rocm-aic.${SHORT}"
CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-$HOME/Projects/rocm-aic-ci}"
TARBALL_DIR="${CI_STORAGE_ROOT}/images/aic-ci-${SHORT}"

cleanup() {
    echo "=== Cleaning up ==="
    rm -rf "${WORKDIR}" "${TARBALL_DIR}"
}
trap cleanup EXIT

# Re-clone if WORKDIR is missing or checked out at the wrong SHA.
ACTUAL_SHA="$(git -C "${WORKDIR}" rev-parse HEAD 2>/dev/null || true)"
if [[ ! -d "${WORKDIR}" || "${ACTUAL_SHA}" != "${SHA}" ]]; then
    echo "=== (Re-)cloning ${REPO} at ${SHA} ==="
    rm -rf "${WORKDIR}"
    git clone --filter=blob:none --no-single-branch "${REPO}" "${WORKDIR}"
    git -C "${WORKDIR}" checkout "${SHA}"
fi

if [[ ! -d "${TARBALL_DIR}" ]]; then
    echo "ERROR: ${TARBALL_DIR} not found — did dist-build run first?" >&2
    exit 1
fi

echo "=== Running ${TARGET} (AIC_IMAGE_NAME=${AIC_IMAGE_NAME}) ==="
cd "${WORKDIR}"

JOB_ID=$(AIC_SPUR_CLUSTER=1 \
    AIC_IMAGE_NAME="${AIC_IMAGE_NAME}" \
    AIC_IMAGE_DIR="${TARBALL_DIR}" \
    make "${TARGET}" 2>&1 \
    | grep -oE '(submitted (cliff-short|aic-cliff) job |Submitted batch job )[0-9]+' \
    | grep -oE '[0-9]+$' | tail -1)

if [[ -z "${JOB_ID}" ]]; then
    echo "ERROR: could not determine Slurm job ID from make ${TARGET} output" >&2
    exit 1
fi

echo "=== Cliff job ${JOB_ID} submitted — polling for completion ==="
LOG="logs/${JOB_ID}/cliff.out"

while squeue -j "${JOB_ID}" -h 2>/dev/null |
    awk -v id="${JOB_ID}" '$1 == id { found = 1 } END { exit found ? 0 : 1 }'; do
    sleep 30
done

STATE=$(sacct -j "${JOB_ID}" --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
echo "=== Job ${JOB_ID} finished with state: ${STATE} ==="

if [[ -f "${LOG}" ]]; then
    echo "=== Cliff output (${LOG}) ==="
    cat "${LOG}"
fi

[[ "${STATE}" == "COMPLETED" ]] || { echo "ERROR: job ${JOB_ID} ended in state ${STATE}" >&2; exit 1; }
echo "=== ${TARGET} complete ==="
REMOTE

echo "Cliff run passed for ${SHORT}"
