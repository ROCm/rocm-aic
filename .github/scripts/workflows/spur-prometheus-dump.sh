#!/usr/bin/env bash
# -E inherit ERR trap in functions and subshells,
# -e exit the script upon commands failing,
# -u to error if the script ever references a variable that is unset,
# -o pipefail to error if any stage of a pipeline fails.
set -Eeuo pipefail

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
#
# Usage:
#   bash .github/scripts/workflows/spur-prometheus-dump.sh <full-sha>

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

SHA="${1:?usage: $0 <full-sha>}"
SHORT="${SHA:0:7}"
AIC_IMAGE_NAME="rocm-aic-ci-${SHORT}"
AIC_SPUR_HOST="${AIC_SPUR_HOST:?AIC_SPUR_HOST must be set}"
AIC_SPUR_HOST="${AIC_SPUR_HOST//[$'\t\r\n ']}"
AIC_SHARED_NFS="${AIC_SHARED_NFS:?AIC_SHARED_NFS must be set}"
AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-}"
REPO="https://github.com/ROCm/rocm-aic.git"

ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "${AIC_SPUR_HOST}" env \
    SHA="${SHA}" \
    REPO="${REPO}" \
    AIC_IMAGE_NAME="${AIC_IMAGE_NAME}" \
    AIC_SHARED_NFS="${AIC_SHARED_NFS}" \
    AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT}" \
    bash << 'REMOTE'
# -E inherit ERR trap in functions and subshells,
# -e exit the script upon commands failing,
# -u to error if the script ever references a variable that is unset,
# -o pipefail to error if any stage of a pipeline fails.
set -Eeuo pipefail
# errexit exits without saying where; report it.  Only the line number and
# function call stack are printed, never the command text, which can carry
# secret values.
_on_err() {
    # set -E carries this trap into $(...), where errexit is off.
    [[ $- == *e* ]] || return 0
    # || : so a failed write cannot replace the exit status.
    printf '%s: command failed (exit %s, pipeline status %s) at line %s%s\n' \
        'spur-prometheus-dump.sh (remote)' "$1" "$2" "$3" "${4:+ in ${4// / <- }}" >&2 || :
}
trap '_on_err "$?" "${PIPESTATUS[*]}" "$LINENO" "${FUNCNAME[*]-}"' ERR
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
