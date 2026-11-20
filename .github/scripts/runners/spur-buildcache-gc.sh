#!/usr/bin/env bash
# -E inherit ERR trap in functions and subshells,
# -e exit the script upon commands failing,
# -u to error if the script ever references a variable that is unset,
# -o pipefail to error if any stage of a pipeline fails.
set -Eeuo pipefail

# Installed on the self-hosted runner; SSHes to the SPUR head node (AIC_SPUR_HOST), clones the repo at
# the given SHA, and runs `make cache-gc` there: it deletes from the build
# registry storage the cache and images no job has used lately.  The storage is
# the one `make dist-build` resolves (AIC_CACHE_DIR's SPUR default), so neither
# names a path.

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

SHA="${1:?usage: $0 <full-sha> [dry-run]}"
case "${2:-}" in
    "") DRY_RUN=0 ;;
    dry-run) DRY_RUN=1 ;;
    *) echo "ERROR: second argument must be dry-run or absent" >&2; exit 2 ;;
esac
SHORT="${SHA:0:7}"
REPO="https://github.com/ROCm/rocm-aic.git"
AIC_SPUR_HOST="${AIC_SPUR_HOST:?AIC_SPUR_HOST must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_HOST="${AIC_SPUR_HOST//[$'\t\r\n ']}"
AIC_SHARED_NFS="${AIC_SHARED_NFS:?AIC_SHARED_NFS must be set (e.g. via GitHub repo variable)}"
AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.github/scripts/runners/spur-ci-common.sh
source "${SCRIPT_DIR}/spur-ci-common.sh"
aic_ci_session_init "${SHORT}" "buildcache-gc"

aic_ci_ssh_bash \
    SHA="${SHA}" \
    REPO="${REPO}" \
    DRY_RUN="${DRY_RUN}" \
    AIC_SHARED_NFS="${AIC_SHARED_NFS}" \
    AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT}" \
    AIC_BUILD_EXCLUDE_NODES="${AIC_BUILD_EXCLUDE_NODES:-}" << 'REMOTE'
# -E inherit ERR trap in functions and subshells,
# -e exit the script upon commands failing,
# -u to error if the script ever references a variable that is unset,
# -o pipefail to error if any stage of a pipeline fails.
set -Eeuo pipefail
_on_err() {
    [[ $- == *e* ]] || return 0
    [[ -z ${_aic_err_child_reports-} ]] || return 0
    printf '%s: command failed (exit %s, pipeline status %s) at line %s%s\n' \
        'spur-buildcache-gc.sh (remote)' "$1" "$2" "$3" "${4:+ in ${4// / <- }}" >&2 || :
}
trap '_on_err "$?" "${PIPESTATUS[*]}" "$LINENO" "${FUNCNAME[*]-}"' ERR
# Run a child that reports its own failures; only the child reports.
_err_delegate() { local _aic_err_child_reports=1; "$@"; }

SHORT="${SHA:0:7}"
WORKDIR="$HOME/Projects/rocm-aic.${SHORT}.${AIC_CI_RUN_KEY}.gc"
CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-$HOME/Projects/rocm-aic-ci}"
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
export AIC_CI_ACTIVE_JOB_FILE="${JOB_FILE}"

_cleanup() {
    local rc=$?
    trap - EXIT
    # make cache-gc tails the job log into this output, so nothing is lost.
    echo "=== Cleaning up run-attempt worktree ==="
    rm -rf "${WORKDIR}" || echo "WARNING: cleanup could not fully remove ${WORKDIR}" >&2
    if (( rc == 0 )); then
        rm -f "${PID_FILE}" "${JOB_FILE}" "${CANCEL_FILE}" 2>/dev/null || true
    fi
    exit "${rc}"
}
trap _cleanup EXIT

echo "=== Cloning ${REPO} at ${SHA} into ${WORKDIR} ==="
rm -rf "${WORKDIR}"
git clone --filter=blob:none --no-single-branch "${REPO}" "${WORKDIR}"
cd "${WORKDIR}"
git checkout "${SHA}"

echo "=== Running cache-gc (dry run: ${DRY_RUN}) ==="
AIC_SPUR_CLUSTER=1 \
    AIC_CACHE_GC_DRY_RUN="${DRY_RUN}" \
    _err_delegate make cache-gc
REMOTE

echo "Build cache GC finished for ${SHORT}"
