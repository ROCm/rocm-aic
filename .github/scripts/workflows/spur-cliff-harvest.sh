#!/usr/bin/env bash
set -euo pipefail

# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Runs from a workflow checkout on the self-hosted runner; SSHes to the SPUR head node (AIC_SPUR_HOST),
# submits the full cliff-submit job via `make cliff-submit`, waits for it to
# finish, then:
#   1. Copies CSVs + plots to a stable NFS staging dir (survives workdir cleanup)
#   2. Calls benchmarks/cliff_to_html.py to generate a self-contained HTML page
#   3. Writes a JSON manifest alongside the HTML
#
# The generated cliff-page/ directory is copied back to the runner so the
# workflow can upload it as an artifact and deploy to gh-pages.
#
# Requires (all already in use by sibling nightly workflows):
#   secrets.AIC_SPUR_HOST        — SSH target for the SPUR head node
#   secrets.AIC_SHARED_NFS       — shared NFS path for image tarballs + scratch
#   secrets.AIC_SPUR_CONTROLLER  — SPUR controller address
#
# Usage:
#   bash .github/scripts/workflows/spur-cliff-harvest.sh <full-sha> <run-date-ISO>
#   # run-date-ISO defaults to $(date +%Y-%m-%d) if omitted

SHA="${1:?usage: $0 <full-sha> [run-date]}"
RUN_DATE="${2:-$(date +%Y-%m-%d)}"
SHORT="${SHA:0:7}"
AIC_IMAGE_NAME="rocm-aic-ci-${SHORT}"
AIC_SPUR_HOST="${AIC_SPUR_HOST:?AIC_SPUR_HOST must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_HOST="${AIC_SPUR_HOST//[$'\t\r\n ']}"
AIC_SHARED_NFS="${AIC_SHARED_NFS:?AIC_SHARED_NFS must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER:?AIC_SPUR_CONTROLLER must be set (e.g. via GitHub repo variable)}"
AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-}"
REPO="https://github.com/ROCm/rocm-aic.git"

ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "${AIC_SPUR_HOST}" env \
    SHA="${SHA}" \
    REPO="${REPO}" \
    RUN_DATE="${RUN_DATE}" \
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
CLIFF_STAGING_DIR="${AIC_SHARED_NFS}/${USER}/cliff-results-${SHORT}"

cleanup() {
    echo "=== Cleaning up WORKDIR and TARBALL_DIR ==="
    rm -rf "${WORKDIR}" "${TARBALL_DIR}" 2>/dev/null || \
        rm -rf "${WORKDIR}" "${TARBALL_DIR}" 2>/dev/null || true
    # CLIFF_STAGING_DIR is on NFS and intentionally NOT cleaned here;
    # it is removed by the runner after scp completes.
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Clone the repo at this SHA
# ---------------------------------------------------------------------------
ACTUAL_SHA="$(git -C "${WORKDIR}" rev-parse HEAD 2>/dev/null || true)"
if [[ ! -d "${WORKDIR}" || "${ACTUAL_SHA}" != "${SHA}" ]]; then
    echo "=== (Re-)cloning ${REPO} at ${SHA} ==="
    rm -rf "${WORKDIR}"
    git clone --filter=blob:none --no-single-branch "${REPO}" "${WORKDIR}"
fi
cd "${WORKDIR}"
git checkout --quiet "${SHA}"

if [[ ! -d "${TARBALL_DIR}" ]]; then
    echo "ERROR: ${TARBALL_DIR} not found — did dist-build run first?" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Submit the cliff job and poll for completion
# ---------------------------------------------------------------------------
echo "=== Submitting cliff-submit job (AIC_IMAGE_NAME=${AIC_IMAGE_NAME}) ==="
JOB_ID=$(SPUR_CONTROLLER_ADDR="${AIC_SPUR_CONTROLLER}" \
    AIC_SPUR_CLUSTER=1 \
    AIC_IMAGE_NAME="${AIC_IMAGE_NAME}" \
    AIC_IMAGE_DIR="${TARBALL_DIR}" \
    make cliff-submit 2>&1 \
  | grep -oE '(submitted (cliff-short|aic-cliff) job |Submitted batch job )[0-9]+' \
  | grep -oE '[0-9]+$' | tail -1)

if [[ -z "${JOB_ID}" ]]; then
    echo "ERROR: could not determine Slurm job ID from make cliff-submit output" >&2
    exit 1
fi

echo "=== Cliff job ${JOB_ID} submitted — polling for completion ==="

while squeue -j "${JOB_ID}" -h 2>/dev/null |
    awk -v id="${JOB_ID}" '$1 == id { found = 1 } END { exit found ? 0 : 1 }'; do
    sleep 30
done

STATE=$(sacct -j "${JOB_ID}" --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
echo "=== Job ${JOB_ID} finished with state: ${STATE} ==="

LOG="logs/${JOB_ID}/cliff.out"
if [[ -f "${LOG}" ]]; then
    echo "=== Cliff output (${LOG}) ==="
    cat "${LOG}"
fi

[[ "${STATE}" == "COMPLETED" ]] || { echo "ERROR: job ${JOB_ID} ended in state ${STATE}" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Copy results to NFS staging BEFORE the EXIT trap deletes WORKDIR
# ---------------------------------------------------------------------------
RUN_DIR="${WORKDIR}/logs/${JOB_ID}"

if [[ ! -d "${RUN_DIR}/results" ]]; then
    echo "ERROR: expected results dir not found at ${RUN_DIR}/results" >&2
    echo "Contents of ${WORKDIR}/logs/:" >&2
    ls "${WORKDIR}/logs/" 2>/dev/null || true
    exit 1
fi

mkdir -p "${CLIFF_STAGING_DIR}"
echo "=== Copying results to ${CLIFF_STAGING_DIR} ==="
cp -r "${RUN_DIR}/results" "${CLIFF_STAGING_DIR}/"
[[ -d "${RUN_DIR}/plots" ]] && cp -r "${RUN_DIR}/plots" "${CLIFF_STAGING_DIR}/"

# ---------------------------------------------------------------------------
# Generate the HTML performance page on the head node
# ---------------------------------------------------------------------------
echo "=== Generating cliff HTML page ==="
python3 "${WORKDIR}/benchmarks/cliff_to_html.py" \
    --run-dir "${CLIFF_STAGING_DIR}" \
    --sha     "${SHA}" \
    --run-date "${RUN_DATE}" \
    --output-dir "${CLIFF_STAGING_DIR}/cliff-page"

# Write a JSON manifest so future runs can append history
cat > "${CLIFF_STAGING_DIR}/cliff-page/manifest.json" << JSON
{
  "sha": "${SHA}",
  "run_date": "${RUN_DATE}",
  "job_id": "${JOB_ID}",
  "repo_url": "https://github.com/ROCm/rocm-aic"
}
JSON

echo "=== Cliff page written to ${CLIFF_STAGING_DIR}/cliff-page/ ==="
wc -l "${CLIFF_STAGING_DIR}/cliff-page/index.html"
REMOTE

# ---------------------------------------------------------------------------
# Copy the HTML page back to the runner working directory
# ---------------------------------------------------------------------------
SPUR_USER="$(ssh -o ServerAliveInterval=30 "${AIC_SPUR_HOST}" id -un)"
REMOTE_CLIFF_STAGING_DIR="${AIC_SHARED_NFS}/${SPUR_USER}/cliff-results-${SHORT}"

echo "=== Copying cliff page back to runner ==="
rm -rf cliff-page
mkdir -p cliff-page
scp -r -o ServerAliveInterval=30 \
    "${AIC_SPUR_HOST}:${REMOTE_CLIFF_STAGING_DIR}/cliff-page/." \
    cliff-page/

# Clean up the NFS staging dir now that we have a local copy
ssh -o ServerAliveInterval=30 "${AIC_SPUR_HOST}" \
    "rm -rf '${REMOTE_CLIFF_STAGING_DIR}'" || true

echo "=== Cliff harvest complete for ${SHORT} — page at cliff-page/index.html ==="
