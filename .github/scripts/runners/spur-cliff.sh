#!/usr/bin/env bash
# -E inherit ERR trap in functions and subshells,
# -e exit the script upon commands failing,
# -u to error if the script ever references a variable that is unset,
# -o pipefail to error if any stage of a pipeline fails.
set -Eeuo pipefail

# Installed on the self-hosted runner; SSHes to the SPUR head node (AIC_SPUR_HOST), uses the clone and
# tarball left by spur-dist-build.sh, and runs a cliff benchmark.
#
# Usage: spur-cliff.sh <full-sha> <target>
#   target: cliff-short  -- 1-point sweep, quick PR gate
#           cliff-submit -- full 3-arm sweep, nightly/post-merge
#
# `make <target>` blocks until the Slurm job ends and fails with it (see cmd_cliff
# in run-build-distribute.sh), so this script only has to run it; the job's own
# exit-file-then-sacct gating lives in _sbatch_run.
#
# Cleanup ownership: the cliff is terminal in the on-demand flow, so on success it
# removes the clone and tarball dir.  KEEP_ARTIFACTS=1 keeps the tarballs for a
# later stage; a failure always removes them and keeps the job logs under
# joblogs/ instead.

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

# Commit `make install-ci-scripts` deployed; if it predates your change, redeploy.
echo "Installed CI scripts: $(cat "$(dirname -- "${BASH_SOURCE[0]}")/VERSION" || echo unknown)"

SHA="${1:?usage: $0 <full-sha> <cliff-short|cliff-submit>}"
TARGET="${2:?usage: $0 <full-sha> <cliff-short|cliff-submit>}"
SHORT="${SHA:0:7}"
AIC_SPUR_HOST="${AIC_SPUR_HOST:?AIC_SPUR_HOST must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_HOST="${AIC_SPUR_HOST//[$'\t\r\n ']}"
AIC_SHARED_NFS="${AIC_SHARED_NFS:?AIC_SHARED_NFS must be set (e.g. via GitHub repo variable)}"
AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-}"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-0}"
REPO="https://github.com/ROCm/rocm-aic.git"

case "${TARGET}" in
    cliff-short|cliff-submit) ;;
    *) echo "ERROR: target must be cliff-short or cliff-submit" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.github/scripts/runners/spur-ci-common.sh
source "${SCRIPT_DIR}/spur-ci-common.sh"
aic_ci_session_init "${SHORT}" "cliff"

aic_ci_ssh_bash \
    SHA="${SHA}" \
    REPO="${REPO}" \
    TARGET="${TARGET}" \
    AIC_SHARED_NFS="${AIC_SHARED_NFS}" \
    AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT}" \
    KEEP_ARTIFACTS="${KEEP_ARTIFACTS}" \
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
        'spur-cliff.sh (remote)' "$1" "$2" "$3" "${4:+ in ${4// / <- }}" >&2 || :
}
trap '_on_err "$?" "${PIPESTATUS[*]}" "$LINENO" "${FUNCNAME[*]-}"' ERR
# Run a child that reports its own failures; only the child reports.
_err_delegate() { local _aic_err_child_reports=1; "$@"; }

SHORT="${SHA:0:7}"
WORKDIR="$HOME/Projects/rocm-aic.${SHORT}.${AIC_CI_RUN_KEY}"
CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-$HOME/Projects/rocm-aic-ci}"
TARBALL_DIR="${CI_STORAGE_ROOT}/images/aic-ci-${SHORT}"
CONTROL_PREFIX="${CI_STORAGE_ROOT}/control/${SHORT}.${AIC_CI_RUN_KEY}.${AIC_CI_STAGE}"
PID_FILE="${CONTROL_PREFIX}.pid"
JOB_FILE="${CONTROL_PREFIX}.job"
CANCEL_FILE="${CONTROL_PREFIX}.cancel"

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

_cleanup() {
    local rc=$?
    trap - EXIT
    (( rc == 0 )) || _archive_job_logs
    echo "=== Cleaning up run-attempt worktree ==="
    _best_effort_remove "${WORKDIR}"
    if (( rc != 0 )) || [[ "${KEEP_ARTIFACTS}" != "1" ]]; then
        echo "=== Removing staged image ==="
        _best_effort_remove "${TARBALL_DIR}"
    fi
    if (( rc == 0 )); then
        rm -f "${PID_FILE}" "${JOB_FILE}" "${CANCEL_FILE}" 2>/dev/null || true
    fi
    exit "${rc}"
}
trap _cleanup EXIT

# Re-clone if WORKDIR is missing or checked out at the wrong SHA (e.g. stale
# leftover from a prior failed run at a different commit with the same prefix).
ACTUAL_SHA="$(git -C "${WORKDIR}" rev-parse HEAD 2>/dev/null || true)"
if [[ ! -d "${WORKDIR}" || "${ACTUAL_SHA}" != "${SHA}" ]]; then
    echo "=== (Re-)cloning ${REPO} at ${SHA} ==="
    rm -rf "${WORKDIR}"
    git clone --filter=blob:none --no-single-branch "${REPO}" "${WORKDIR}"
    git -C "${WORKDIR}" checkout "${SHA}"
fi

if [[ ! -d "${TARBALL_DIR}" ]]; then
    echo "ERROR: ${TARBALL_DIR} not found — no dist-build staged images for this SHA," \
        "or an earlier run already removed them (a finished cliff always does)." \
        "For the on-demand workflow use \"Re-run all jobs\", not \"Re-run failed jobs\"." >&2
    exit 1
fi

echo "=== Running ${TARGET} ==="
# AIC_BUILD_EXCLUDE_NODES, when set, is already in this shell's environment and so
# reaches make and _sbatch_run unchanged.
AIC_SPUR_CLUSTER=1 \
    AIC_IMAGE_DIR="${TARBALL_DIR}" \
    _err_delegate make -C "${WORKDIR}" "${TARGET}"

echo "=== ${TARGET} complete ==="
REMOTE

echo "Cliff run passed for ${SHORT}"
