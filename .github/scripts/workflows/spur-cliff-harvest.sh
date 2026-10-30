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
#
# Usage:
#   bash .github/scripts/workflows/spur-cliff-harvest.sh <full-sha> <run-date-ISO>
#   # run-date-ISO defaults to $(date +%Y-%m-%d) if omitted

# errexit exits without saying where; report it.  Only the line number and
# function call stack are printed, never the command text, which can carry
# secret values.
_on_err() {
    # set -E carries this trap into $(...), where errexit is off.
    [[ $- == *e* ]] || return 0
    # The failing command was a child that reports its own failures.
    [[ -z ${_aic_err_child_reports-} ]] || return 0
    # || : so a failed write cannot replace the exit status.
    printf '%s: command failed (exit %s, pipeline status %s) at line %s%s\n' \
        "${0##*/}" "$1" "$2" "$3" "${4:+ in ${4// / <- }}" >&2 || :
}
trap '_on_err "$?" "${PIPESTATUS[*]}" "$LINENO" "${FUNCNAME[*]-}"' ERR

SHA="${1:?usage: $0 <full-sha> [run-date]}"
RUN_DATE="${2:-$(date +%Y-%m-%d)}"
SHORT="${SHA:0:7}"
AIC_SPUR_HOST="${AIC_SPUR_HOST:?AIC_SPUR_HOST must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_HOST="${AIC_SPUR_HOST//[$'\t\r\n ']}"
AIC_SHARED_NFS="${AIC_SHARED_NFS:?AIC_SHARED_NFS must be set (e.g. via GitHub repo variable)}"
AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-}"
REPO="https://github.com/ROCm/rocm-aic.git"

# This script runs from the workflow checkout rather than the deployed
# /usr/local/lib/aic-ci copy, so take the session helpers from the same checkout.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.github/scripts/runners/spur-ci-common.sh
source "${SCRIPT_DIR}/../runners/spur-ci-common.sh"
aic_ci_session_init "${SHORT}" "cliff-harvest"

aic_ci_ssh_bash \
    SHA="${SHA}" \
    REPO="${REPO}" \
    RUN_DATE="${RUN_DATE}" \
    AIC_SHARED_NFS="${AIC_SHARED_NFS}" \
    AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT}" \
    AIC_BUILD_EXCLUDE_NODES="${AIC_BUILD_EXCLUDE_NODES:-}" << 'REMOTE'
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
    # The failing command was a child that reports its own failures.
    [[ -z ${_aic_err_child_reports-} ]] || return 0
    # || : so a failed write cannot replace the exit status.
    printf '%s: command failed (exit %s, pipeline status %s) at line %s%s\n' \
        'spur-cliff-harvest.sh (remote)' "$1" "$2" "$3" "${4:+ in ${4// / <- }}" >&2 || :
}
trap '_on_err "$?" "${PIPESTATUS[*]}" "$LINENO" "${FUNCNAME[*]-}"' ERR
export PATH="/usr/local/bin:${PATH}"

SHORT="${SHA:0:7}"
WORKDIR="$HOME/Projects/rocm-aic.${SHORT}.${AIC_CI_RUN_KEY}"
CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-$HOME/Projects/rocm-aic-ci}"
TARBALL_DIR="${CI_STORAGE_ROOT}/images/aic-ci-${SHORT}"
CLIFF_STAGING_DIR="${AIC_SHARED_NFS}/${USER}/cliff-results-${SHORT}"
CONTROL_PREFIX="${CI_STORAGE_ROOT}/control/${SHORT}.${AIC_CI_RUN_KEY}.${AIC_CI_STAGE}"
PID_FILE="${CONTROL_PREFIX}.pid"
JOB_FILE="${CONTROL_PREFIX}.job"
CANCEL_FILE="${CONTROL_PREFIX}.cancel"
SUBMIT_LOG=""

mkdir -p "${CI_STORAGE_ROOT}/control"
printf '%s\n' "${BASHPID}" > "${PID_FILE}"
if [[ -e "${CANCEL_FILE}" ]]; then
    echo "CI session was cancelled before remote startup completed" >&2
    rm -f "${PID_FILE}" "${JOB_FILE}" "${CANCEL_FILE}" 2>/dev/null || true
    exit 143
fi
# A cliff holds a GPU for hours: record the Slurm job id where the wrapper's
# cancel path can find it.
export AIC_CI_ACTIVE_JOB_FILE="${JOB_FILE}"

_best_effort_remove() {
    rm -rf "$@" || echo "WARNING: cleanup could not fully remove: $*" >&2
}

# Keep the SPUR job logs that cleanup would otherwise delete with the worktree, so
# a failed run can still be debugged.
_archive_job_logs() {
    local archive="${CI_STORAGE_ROOT}/joblogs/${SHORT}.${AIC_CI_RUN_KEY}.${AIC_CI_STAGE}"
    local -a srcs=()
    shopt -s nullglob
    srcs=("${WORKDIR}"/spur-*.out)
    shopt -u nullglob
    [[ -d "${WORKDIR}/logs" ]] || (( ${#srcs[@]} > 0 )) || return 0
    if ! mkdir -p "${archive}"; then
        echo "WARNING: cannot create ${archive}; SPUR job logs will be lost" >&2
        return 0
    fi
    # logs/<job-id>/ also holds the Prometheus TSDB, results and plots, which are
    # large and not diagnostics, and this is near-full shared NFS: keep only the
    # job's *.out/*.exit, the container-*.log files and the vllm/ and lmcache/
    # log dirs, under the same <job-id>/ layout.
    if [[ -d "${WORKDIR}/logs" ]]; then
        (cd "${WORKDIR}/logs" && find . -type f -regextype posix-extended \
            -regex './[^/]+/([^/]+\.(out|exit)|container-[^/]+\.log|(vllm|lmcache)/.+)' -print0 |
            xargs -0 -r cp --parents -a -t "${archive}") ||
            echo "WARNING: some SPUR job logs could not be archived to ${archive}" >&2
    fi
    (( ${#srcs[@]} == 0 )) || cp -a "${srcs[@]}" "${archive}/" ||
        echo "WARNING: some SPUR job logs could not be archived to ${archive}" >&2
    echo "=== SPUR job logs kept at $(hostname):${archive} ==="
    # Bounded retention -- this is shared, quota'd NFS.  A failure here silently
    # leaks quota until it breaks unrelated jobs, so report it; do not let it
    # fail the cleanup path, which is already handling an earlier failure.
    local prune_err="" prune_rc=0
    prune_err=$(find "${CI_STORAGE_ROOT}/joblogs" -mindepth 1 -maxdepth 1 -type d \
        -mtime +14 -exec rm -rf {} + 2>&1 >/dev/null) || prune_rc=$?
    if (( prune_rc != 0 )) || [[ -n "${prune_err}" ]]; then
        echo "WARNING: could not expire SPUR job logs older than 14d under" \
            "${CI_STORAGE_ROOT}/joblogs (rc=${prune_rc}): ${prune_err:-no error output}" >&2
    fi
}

cleanup() {
    local rc=$?
    trap - EXIT
    (( rc == 0 )) || _archive_job_logs
    echo "=== Cleaning up WORKDIR and TARBALL_DIR ==="
    _best_effort_remove "${WORKDIR}" "${TARBALL_DIR}"
    [[ -z "${SUBMIT_LOG}" ]] || _best_effort_remove "${SUBMIT_LOG}"
    # CLIFF_STAGING_DIR is on NFS and intentionally NOT cleaned here;
    # it is removed by the runner after scp completes.
    if (( rc == 0 )); then
        rm -f "${PID_FILE}" "${JOB_FILE}" "${CANCEL_FILE}" 2>/dev/null || true
    fi
    exit "${rc}"
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
# Run the cliff job
# ---------------------------------------------------------------------------
# `make cliff-submit` blocks until the Slurm job ends and fails with it:
# _sbatch_run takes the job's exit file as the verdict (SPUR's accounting is not
# trustworthy -- job 150665 died on `docker load` and still accounted as
# COMPLETED) and falls back to a terminal sacct state, never to a bare queue
# check.  Tee the output so it streams into the CI log as the job runs and the job
# id, which names the results dir, can be read back afterwards.
echo "=== Running cliff-submit ==="
SUBMIT_LOG="$(mktemp)"
# make reports its own failure.  A function-scoped flag (_err_delegate) cannot
# silence this pipeline: the failure is reported from this shell after the
# pipeline returns, when the function's local is gone.  Set it here instead; a
# failing pipeline exits at once, so it is never left set for a later command.
_aic_err_child_reports=1
AIC_SPUR_CLUSTER=1 \
    AIC_IMAGE_DIR="${TARBALL_DIR}" \
    make cliff-submit 2>&1 | tee "${SUBMIT_LOG}"
unset _aic_err_child_reports

JOB_ID="$(grep -oE 'submitted [a-z0-9-]+ as job [0-9]+' "${SUBMIT_LOG}" \
    | grep -oE '[0-9]+$' | tail -1 || true)"
rm -f "${SUBMIT_LOG}"
if [[ -z "${JOB_ID}" ]]; then
    echo "ERROR: could not determine Slurm job ID from make cliff-submit output" >&2
    exit 1
fi
echo "=== Cliff job ${JOB_ID} finished ==="

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
# The remote stage finished; a failure in the local steps below has no remote
# session left to cancel.
aic_ci_remote_done

# ---------------------------------------------------------------------------
# Copy the HTML page back to the runner working directory
# ---------------------------------------------------------------------------
SPUR_USER="$(ssh -o ServerAliveInterval=30 "${AIC_SPUR_HOST}" id -un)"
REMOTE_CLIFF_STAGING_DIR="${AIC_SHARED_NFS}/${SPUR_USER}/cliff-results-${SHORT}"

# Remove the NFS staging dir.  Capture ssh's stderr and report it with the exit
# status: a leaked directory on quota'd NFS breaks unrelated jobs later.  Never
# fails, so it can run on the failure path without hiding the error being handled.
_remove_remote_staging() {
    local err="" rc=0
    err="$(ssh -o ServerAliveInterval=30 "${AIC_SPUR_HOST}" \
        "rm -rf '${REMOTE_CLIFF_STAGING_DIR}'" 2>&1)" || rc=$?
    (( rc == 0 )) ||
        echo "WARNING: could not remove staging dir ${REMOTE_CLIFF_STAGING_DIR}" \
            "on the SPUR head node (rc=${rc}): ${err:-no error output}" >&2
}

echo "=== Copying cliff page back to runner ==="
rm -rf cliff-page
scp_rc=0
aic_ci_scp_from "${REMOTE_CLIFF_STAGING_DIR}/cliff-page" cliff-page || scp_rc=$?

# Clean up the NFS staging dir whether or not the copy worked: the remote stage
# is done, so nothing else will remove it.
_remove_remote_staging
if (( scp_rc != 0 )); then
    echo "ERROR: could not copy the cliff page back from ${REMOTE_CLIFF_STAGING_DIR} (exit ${scp_rc})" >&2
    exit "${scp_rc}"
fi

echo "=== Cliff harvest complete for ${SHORT} — page at cliff-page/index.html ==="
