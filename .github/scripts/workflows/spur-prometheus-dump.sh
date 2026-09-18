#!/usr/bin/env bash
set -euo pipefail

# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Runs from a workflow checkout on the self-hosted runner; SSHes to the SPUR
# head node (AIC_SPUR_HOST), clones the repo at the given SHA, then runs
# `make prometheus-dump` which submits a GPU sbatch job, waits for it to
# finish, and writes a Markdown metrics reference to shared NFS.
#
# The generated prometheus-dump.md is copied back to the runner working
# directory so the workflow can upload it as an artifact and deploy to
# gh-pages at https://rocm.github.io/rocm-aic/prometheus/.
#
# Requires:
#   secrets.AIC_SPUR_HOST        — SSH target for the SPUR head node
#   secrets.AIC_SHARED_NFS       — shared NFS path for image tarballs + scratch
#   secrets.AIC_SPUR_CONTROLLER  — SPUR controller address
#
# Usage:
#   bash .github/scripts/workflows/spur-prometheus-dump.sh <full-sha>

SHA="${1:?usage: $0 <full-sha>}"
SHORT="${SHA:0:7}"
AIC_IMAGE_NAME="rocm-aic-ci-${SHORT}"
AIC_SPUR_HOST="${AIC_SPUR_HOST:?AIC_SPUR_HOST must be set}"
AIC_SPUR_HOST="${AIC_SPUR_HOST//[$'\t\r\n ']}"
AIC_SHARED_NFS="${AIC_SHARED_NFS:?AIC_SHARED_NFS must be set}"
AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER:?AIC_SPUR_CONTROLLER must be set}"
AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-}"
REPO="https://github.com/ROCm/rocm-aic.git"

ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "${AIC_SPUR_HOST}" env \
    SHA="${SHA}" \
    REPO="${REPO}" \
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
PROM_DUMP_OUT="${AIC_SHARED_NFS}/rocm-aic/prometheus-dump-${SHORT}.md"

cleanup() {
    echo "=== Cleaning up WORKDIR ==="
    rm -rf "${WORKDIR}" 2>/dev/null || true
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
git checkout "${SHA}"

# ---------------------------------------------------------------------------
# Run make prometheus-dump — submits an sbatch GPU job on SPUR and waits
# ---------------------------------------------------------------------------
echo "=== Running make prometheus-dump ==="
make prometheus-dump \
    AIC_SPUR_CLUSTER=1 \
    AIC_SHARED_NFS="${AIC_SHARED_NFS}" \
    SPUR_CONTROLLER_ADDR="${SPUR_CONTROLLER_ADDR}" \
    AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER}" \
    AIC_IMAGE_DIR="${CI_STORAGE_ROOT}/images/aic-ci-${SHORT}" \
    PROM_DUMP_OUT="${PROM_DUMP_OUT}"

echo "=== prometheus-dump complete: ${PROM_DUMP_OUT} ==="
REMOTE

# ---------------------------------------------------------------------------
# Copy Markdown report back to runner working directory
# ---------------------------------------------------------------------------
AIC_REMOTE_PROM_DUMP="${AIC_SHARED_NFS}/rocm-aic/prometheus-dump-${SHORT}.md"

echo "=== Copying prometheus-dump.md to runner ==="
mkdir -p prometheus-page
scp -o ServerAliveInterval=30 \
    "${AIC_SPUR_HOST}:${AIC_REMOTE_PROM_DUMP}" \
    prometheus-page/prometheus-dump.md
ssh -o ServerAliveInterval=30 "${AIC_SPUR_HOST}" rm -f "${AIC_REMOTE_PROM_DUMP}"

echo "=== prometheus-dump.md ready in prometheus-page/ ==="
wc -l prometheus-page/prometheus-dump.md
