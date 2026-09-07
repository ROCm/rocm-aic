#!/usr/bin/env bash
set -euo pipefail

# Installed on the self-hosted runner; SSHes to the SPUR head node (AIC_SPUR_HOST), clones the repo at
# the current SHA, and runs the requested dist-build target with a CI-scoped
# image name and tarball path.  The tarball is left in place for
# spur-smoke-test.sh; the run-attempt-scoped clone is always removed.

SHA="${1:?usage: $0 <full-sha> [dist-build|dist-build-fast]}"
AIC_DIST_BUILD_TARGET="${2:-dist-build}"
case "${AIC_DIST_BUILD_TARGET}" in
    dist-build | dist-build-fast) ;;
    *)
        echo "ERROR: unsupported build target: ${AIC_DIST_BUILD_TARGET}" >&2
        exit 2
        ;;
esac
SHORT="${SHA:0:7}"
REPO="https://github.com/ROCm/rocm-aic.git"
AIC_IMAGE_NAME="rocm-aic-ci-${SHORT}"
AIC_SPUR_HOST="${AIC_SPUR_HOST:?AIC_SPUR_HOST must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_HOST="${AIC_SPUR_HOST//[$'\t\r\n ']}"
AIC_SHARED_NFS="${AIC_SHARED_NFS:?AIC_SHARED_NFS must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER:?AIC_SPUR_CONTROLLER must be set (e.g. via GitHub repo variable)}"
AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.github/scripts/runners/spur-ci-common.sh
source "${SCRIPT_DIR}/spur-ci-common.sh"
aic_ci_session_init "${SHORT}" "dist-build"

aic_ci_ssh_bash \
    SHA="${SHA}" \
    REPO="${REPO}" \
    AIC_IMAGE_NAME="${AIC_IMAGE_NAME}" \
    AIC_DIST_BUILD_TARGET="${AIC_DIST_BUILD_TARGET}" \
    AIC_SHARED_NFS="${AIC_SHARED_NFS}" \
    AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT}" \
    AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER}" \
    SPUR_CONTROLLER_ADDR="${AIC_SPUR_CONTROLLER}" << 'REMOTE'
set -euo pipefail

SHORT="${SHA:0:7}"
WORKDIR="$HOME/Projects/rocm-aic.${SHORT}.${AIC_CI_RUN_KEY}"
CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-$HOME/Projects/rocm-aic-ci}"
TARBALL_DIR="${CI_STORAGE_ROOT}/images/aic-ci-${SHORT}"
# Build cache lives on shared NFS under the running user, never in $HOME (small
# and quota'd on SPUR) and never in a path shared across users (not writable by
# all of them).  Matches the AIC_SPUR_CLUSTER=1 default in the Makefile.
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

# Save stderr of run-build-distribute.sh that would otherwise be deleted upon
# cleanup for CI debugging.
_archive_job_logs() {
    local archive="${CI_STORAGE_ROOT}/joblogs/${SHORT}.${AIC_CI_RUN_KEY}.${AIC_CI_STAGE}"
    local -a srcs=()
    shopt -s nullglob
    srcs=("${WORKDIR}"/logs/* "${WORKDIR}"/spur-*.out)
    shopt -u nullglob
    (( ${#srcs[@]} > 0 )) || return 0
    if ! mkdir -p "${archive}"; then
        echo "WARNING: cannot create ${archive}; SPUR job logs will be lost" >&2
        return 0
    fi
    cp -a "${srcs[@]}" "${archive}/" ||
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
    if (( rc != 0 )); then
        echo "=== Build failed — removing staged image ==="
        _best_effort_remove "${TARBALL_DIR}"
    fi
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

mkdir -p "${TARBALL_DIR}"

echo "=== Running ${AIC_DIST_BUILD_TARGET} (AIC_SPUR_CLUSTER=1, AIC_IMAGE_NAME=${AIC_IMAGE_NAME}) ==="
AIC_SPUR_CLUSTER=1 \
    AIC_IMAGE_NAME="${AIC_IMAGE_NAME}" \
    AIC_IMAGE_DIR="${TARBALL_DIR}" \
    AIC_CACHE_DIR="${CACHE_DIR}" \
    make "${AIC_DIST_BUILD_TARGET}"

echo "=== ${AIC_DIST_BUILD_TARGET} complete — tarball in ${TARBALL_DIR} ==="
REMOTE

echo "Build succeeded for ${SHORT}"
