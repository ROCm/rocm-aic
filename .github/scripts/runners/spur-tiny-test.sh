#!/usr/bin/env bash
set -euo pipefail

# Installed on the self-hosted runner; SSHes to the SPUR head node (AIC_SPUR_HOST) and runs tiny-test
# against the tarball produced by spur-dist-build.sh for the same SHA (the stage
# after spur-smoke-test.sh).  tiny-test brings up the compose MP stack
# (standalone lmcache server + vLLM LMCacheMPConnector) with a tiny model and
# asserts one non-empty chat completion.
#
# Tarball cleanup ownership depends on whether a cliff stage follows:
#   * PR flow (dist-build -> smoke-test -> tiny-test): tiny-test is terminal, so
#     it owns the final cleanup (removes the clone + tarball on exit).
#   * Nightly (dist-build -> smoke -> tiny -> accuracy -> cliff): the stages
#     after this one need the artifacts, so the nightly tiny-test step sets
#     KEEP_ARTIFACTS=1 and this script only cleans up on failure
#     (spur-cliff-harvest.sh does the final cleanup).
# The tiny model uses the cluster-wide HF cache so it is downloaded once and
# reused across CI workflows and SPUR accounts.

SHA="${1:?usage: $0 <full-sha> [tiny-test|tiny-test-fast]}"
AIC_TINY_TEST_TARGET="${2:-tiny-test}"
case "${AIC_TINY_TEST_TARGET}" in
    tiny-test | tiny-test-fast) ;;
    *)
        echo "ERROR: unsupported tiny-test target: ${AIC_TINY_TEST_TARGET}" >&2
        exit 2
        ;;
esac
SHORT="${SHA:0:7}"
AIC_IMAGE_NAME="rocm-aic-ci-${SHORT}"
AIC_SPUR_HOST="${AIC_SPUR_HOST:?AIC_SPUR_HOST must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_HOST="${AIC_SPUR_HOST//[$'\t\r\n ']}"
AIC_SHARED_NFS="${AIC_SHARED_NFS:?AIC_SHARED_NFS must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER:?AIC_SPUR_CONTROLLER must be set (e.g. via GitHub repo variable)}"
AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-}"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-0}"
REPO="https://github.com/ROCm/rocm-aic.git"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.github/scripts/runners/spur-ci-common.sh
source "${SCRIPT_DIR}/spur-ci-common.sh"
aic_ci_session_init "${SHORT}" "tiny-test"

aic_ci_ssh_bash \
    SHA="${SHA}" \
    REPO="${REPO}" \
    AIC_IMAGE_NAME="${AIC_IMAGE_NAME}" \
    AIC_TINY_TEST_TARGET="${AIC_TINY_TEST_TARGET}" \
    AIC_SHARED_NFS="${AIC_SHARED_NFS}" \
    AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT}" \
    KEEP_ARTIFACTS="${KEEP_ARTIFACTS}" \
    AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER}" \
    SPUR_CONTROLLER_ADDR="${AIC_SPUR_CONTROLLER}" \
    HF_TOKEN="${HF_TOKEN:-}" << 'REMOTE'
set -euo pipefail

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
export AIC_CI_ACTIVE_JOB_FILE="${JOB_FILE}"

_best_effort_remove() {
    rm -rf "$@" || echo "WARNING: cleanup could not fully remove: $*" >&2
}
_cleanup() {
    local rc=$?
    trap - EXIT
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

echo "=== Running ${AIC_TINY_TEST_TARGET} (AIC_IMAGE_NAME=${AIC_IMAGE_NAME}) ==="
AIC_SPUR_CLUSTER=1 \
    AIC_IMAGE_NAME="${AIC_IMAGE_NAME}" \
    AIC_IMAGE_DIR="${TARBALL_DIR}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    make -C "${WORKDIR}" "${AIC_TINY_TEST_TARGET}"

echo "=== ${AIC_TINY_TEST_TARGET} complete ==="
REMOTE

echo "Tiny test passed for ${SHORT}"
