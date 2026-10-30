#!/usr/bin/env bash
# -E inherit ERR trap in functions and subshells,
# -e exit the script upon commands failing,
# -u to error if the script ever references a variable that is unset,
# -o pipefail to error if any stage of a pipeline fails.
set -Eeuo pipefail

# Installed on the self-hosted runner; SSHes to the SPUR head node
# (AIC_SPUR_HOST), clones the repo at the current SHA, runs `make dist-wheels`
# for one GPU arch group, and copies the resulting .whl files back into
# AIC_WHEEL_OUT on the runner for the workflow to upload and publish.
#
# This exists because the nightly wheels cannot be built on a GitHub-hosted
# runner: the wheels come out of the same PyTorch + vLLM source builds the
# images do, which overrun the 6h job limit every time.  The SPUR path already
# completes the whole stack in well under two hours.
#
# Usage: spur-wheels-build.sh <full-sha> <arch-csv> <group>
#   arch-csv  comma-separated gfx list, e.g. gfx90a,gfx942,gfx950.  Commas, not
#             semicolons: ssh joins its argv into a remote shell command, where
#             a ';' would terminate it.  Converted back on the remote side.
#   group     short label for the arch set (cdna / rdna / gfx1250).  Each group
#             gets its own artifact and its own rolling release: the wheel
#             FILENAMES are identical across groups but the GPU code is not.

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

SHA="${1:?usage: $0 <full-sha> <arch-csv> <group>}"
ARCH_CSV="${2:?usage: $0 <full-sha> <arch-csv> <group>}"
GROUP="${3:?usage: $0 <full-sha> <arch-csv> <group>}"

# Both land in remote paths and a remote env assignment, so reject anything
# that is not a plain gfx list / label before it gets near a shell.
[[ "${ARCH_CSV}" =~ ^gfx[0-9a-z]+(,gfx[0-9a-z]+)*$ ]] \
    || { echo "ERROR: arch must be a comma-separated gfx list, got: ${ARCH_CSV}" >&2; exit 2; }
[[ "${GROUP}" =~ ^[A-Za-z0-9_-]+$ ]] \
    || { echo "ERROR: group must be alphanumeric, got: ${GROUP}" >&2; exit 2; }

SHORT="${SHA:0:7}"
REPO="https://github.com/ROCm/rocm-aic.git"
AIC_SPUR_HOST="${AIC_SPUR_HOST:?AIC_SPUR_HOST must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_HOST="${AIC_SPUR_HOST//[$'\t\r\n ']}"
AIC_SHARED_NFS="${AIC_SHARED_NFS:?AIC_SHARED_NFS must be set (e.g. via GitHub repo variable)}"
# Required here (unlike spur-dist-build.sh, which only passes it through): the
# local side builds a remote path from it to scp from and then `rm -rf`, so it
# must be a known absolute path rather than a remote-expanded fallback.
AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:?AIC_CI_STORAGE_ROOT must be set}"
[[ "${AIC_CI_STORAGE_ROOT}" == /* ]] \
    || { echo "ERROR: AIC_CI_STORAGE_ROOT must be absolute, got: ${AIC_CI_STORAGE_ROOT}" >&2; exit 2; }
# Normalise once, here, and send the normalised value remote: both sides derive
# the same wheel path from it and a stray trailing slash would desynchronise them.
AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT%/}"
# Where the wheels end up on THIS runner; the workflow uploads this directory.
AIC_WHEEL_OUT="${AIC_WHEEL_OUT:-${GITHUB_WORKSPACE:-$PWD}/wheels-out}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.github/scripts/runners/spur-ci-common.sh
source "${SCRIPT_DIR}/spur-ci-common.sh"
aic_ci_session_init "${SHORT}" "wheels-${GROUP}"

# Must match WHEEL_DIR in the remote stage: that side writes it, this side
# scp's it back and then removes it.
REMOTE_WHEEL_DIR="${AIC_CI_STORAGE_ROOT}/wheels/${SHORT}.${AIC_CI_RUN_KEY}.${GROUP}"

aic_ci_ssh_bash \
    SHA="${SHA}" \
    REPO="${REPO}" \
    ARCH_CSV="${ARCH_CSV}" \
    GROUP="${GROUP}" \
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
        'spur-wheels-build.sh (remote)' "$1" "$2" "$3" "${4:+ in ${4// / <- }}" >&2 || :
}
trap '_on_err "$?" "${PIPESTATUS[*]}" "$LINENO" "${FUNCNAME[*]-}"' ERR
# Run a child that reports its own failures; only the child reports.
_err_delegate() { local _aic_err_child_reports=1; "$@"; }

SHORT="${SHA:0:7}"
# The arch list travels as CSV to survive ssh's argv-to-command-string join;
# the Dockerfiles and PYTORCH_ROCM_ARCH want it ';'-separated.
ROCM_ARCH_LIST="${ARCH_CSV//,/;}"
WORKDIR="$HOME/Projects/rocm-aic.${SHORT}.${AIC_CI_RUN_KEY}.${GROUP}"
CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-$HOME/Projects/rocm-aic-ci}"
WHEEL_DIR="${CI_STORAGE_ROOT}/wheels/${SHORT}.${AIC_CI_RUN_KEY}.${GROUP}"
# Build cache lives on shared NFS under the running user, never in $HOME (small
# and quota'd on SPUR).  Shared with the nightly dist-build on purpose: the
# wheels come out of the same layers, so a warm image cache makes this cheap.
CACHE_DIR="${AIC_SHARED_NFS%/}/${USER:-$(id -un)}/buildcache"
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

_best_effort_remove() {
    rm -rf "$@" || echo "WARNING: cleanup could not fully remove: $*" >&2
}

_cleanup() {
    local rc=$?
    trap - EXIT
    echo "=== Cleaning up run-attempt worktree ==="
    _best_effort_remove "${WORKDIR}"
    # The wheel dir is the artifact; the local side scp's it and removes it.
    # Only a failed run drops it here, where nothing will come to collect it.
    if (( rc != 0 )); then
        echo "=== Wheel build failed — removing staged wheels ==="
        _best_effort_remove "${WHEEL_DIR}"
    else
        rm -f "${PID_FILE}" "${JOB_FILE}" "${CANCEL_FILE}" 2>/dev/null || true
    fi
    # Bounded retention: shared, quota'd NFS.  A wheel set is ~1 GB, and an
    # interrupted run leaves one behind with no local side left to collect it.
    find "${CI_STORAGE_ROOT}/wheels" -mindepth 1 -maxdepth 1 -type d -mtime +3 \
        -exec rm -rf {} + 2>/dev/null || true
    exit "${rc}"
}
trap _cleanup EXIT

echo "=== Cloning ${REPO} at ${SHA} into ${WORKDIR} ==="
rm -rf "${WORKDIR}"
git clone --filter=blob:none --no-single-branch "${REPO}" "${WORKDIR}"
cd "${WORKDIR}"
git checkout "${SHA}"

echo "=== Running dist-wheels (arch=${ROCM_ARCH_LIST}, group=${GROUP}) ==="
AIC_SPUR_CLUSTER=1 \
    AIC_ROCM_ARCH="${ROCM_ARCH_LIST}" \
    AIC_WHEEL_DIR="${WHEEL_DIR}" \
    AIC_CACHE_DIR="${CACHE_DIR}" \
    ${AIC_BUILD_EXCLUDE_NODES:+AIC_BUILD_EXCLUDE_NODES="${AIC_BUILD_EXCLUDE_NODES}"} \
    _err_delegate make dist-wheels

echo "=== dist-wheels complete — wheels in ${WHEEL_DIR} ==="
ls -lh "${WHEEL_DIR}"
REMOTE
# The remote stage finished; a failed copy below has no remote session to cancel.
aic_ci_remote_done

echo "=== Copying wheels back to ${AIC_WHEEL_OUT} ==="
rm -rf "${AIC_WHEEL_OUT}"
aic_ci_scp_from "${REMOTE_WHEEL_DIR}" "${AIC_WHEEL_OUT}"

# The remote copy is ~1 GB on quota'd NFS and has served its purpose.  Best
# effort: a failure here must not fail a build whose wheels are already local.
ssh -o ConnectTimeout=15 "${AIC_SPUR_HOST}" \
    rm -rf "${REMOTE_WHEEL_DIR}" 2>/dev/null \
    || echo "WARNING: could not remove ${REMOTE_WHEEL_DIR} on ${AIC_SPUR_HOST}" >&2

shopt -s nullglob
wheels=("${AIC_WHEEL_OUT}"/*.whl)
(( ${#wheels[@]} > 0 )) || { echo "ERROR: no wheels copied back to ${AIC_WHEEL_OUT}" >&2; exit 1; }
echo "Wheels for ${GROUP} (${ARCH_CSV}) at ${SHORT}:"
for w in "${wheels[@]}"; do
    printf '  - %s (%s)\n' "$(basename "$w")" "$(du -h "$w" | cut -f1)"
done
