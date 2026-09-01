#!/usr/bin/env bash
set -euo pipefail

# Runs on the self-hosted runner; SSHes to the SPUR head node (AIC_SPUR_HOST) and
# runs accuracy-test against the tarball produced by spur-dist-build.sh for the
# same SHA (the stage after spur-tiny-test.sh in the nightly chain, or triggered
# on demand via /run-ci-accuracy or a manual workflow_dispatch).
#
# The two accepted targets are the same gate: accuracy-test-fast is
# accuracy-test with AIC_ROCM_ARCH pinned to AIC_FAST_ARCH, so it matches the
# tarball a dist-build-fast produced.  Neither is cheaper than the other.
#
# accuracy-test is the KV-integrity gate: it scores gsm8k against a VRAM-only arm
# and a tiered (LMCache + NIXL POSIX NVMe) arm in the same job and asserts that
# routing KV through DRAM/NVMe did not change the answers.  There are no
# committed golden files -- the oracle is the same-run difference between the two
# arms, plus an absolute floor from tests/accuracy/expected.json.  See
# tests/accuracy/README.md for what each assertion catches.
#
# Cleanup ownership depends on whether another stage follows:
#   * On-demand flow: accuracy-test is terminal, so it owns the final cleanup
#     (removes the clone + tarball on exit).
#   * Nightly chain: the cliff job needs the artifacts and runs next whatever
#     this gate concludes, so the nightly step sets KEEP_ARTIFACTS=1 and this
#     script leaves them alone even on failure (spur-cliff-harvest.sh does the
#     final cleanup).
#
# Log harvesting is independent of that ownership split and happens on both
# paths: the per-job scores and per-arm container logs live under the clone,
# which one side or the other always deletes, so they are archived off it and
# scp'd back here for the workflow to upload.  A red gate is exactly when they
# matter, so neither the archive nor the fetch is conditional on success.
# The model uses the cluster-wide HF cache so it is downloaded once and reused
# across CI workflows and SPUR accounts.

SHA="${1:?usage: $0 <full-sha> [accuracy-test|accuracy-test-fast]}"
AIC_ACCURACY_TEST_TARGET="${2:-accuracy-test}"
case "${AIC_ACCURACY_TEST_TARGET}" in
    accuracy-test | accuracy-test-fast) ;;
    *)
        echo "ERROR: unsupported accuracy-test target: ${AIC_ACCURACY_TEST_TARGET}" >&2
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

# Where the remote archives its per-job logs, and where they land here.  When
# AIC_CI_STORAGE_ROOT is unset this stays relative, which both the remote's
# mkdir (its shell starts in $HOME) and scp below resolve against the remote
# home — so the two halves agree without the runner knowing the remote $HOME.
ACCURACY_LOG_ARCHIVE_DIR="${AIC_CI_STORAGE_ROOT:+${AIC_CI_STORAGE_ROOT}/}accuracy-logs"
AIC_ACCURACY_LOG_DEST="${AIC_ACCURACY_LOG_DEST:-${RUNNER_TEMP:-/tmp}/accuracy-logs}"

rc=0
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "${AIC_SPUR_HOST}" env \
    ACCURACY_LOG_ARCHIVE_DIR="${ACCURACY_LOG_ARCHIVE_DIR}" \
    SHA="${SHA}" \
    REPO="${REPO}" \
    AIC_IMAGE_NAME="${AIC_IMAGE_NAME}" \
    AIC_ACCURACY_TEST_TARGET="${AIC_ACCURACY_TEST_TARGET}" \
    AIC_SHARED_NFS="${AIC_SHARED_NFS}" \
    AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT}" \
    KEEP_ARTIFACTS="${KEEP_ARTIFACTS}" \
    AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER}" \
    SPUR_CONTROLLER_ADDR="${AIC_SPUR_CONTROLLER}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    bash << 'REMOTE' || rc=$?
set -euo pipefail

SHORT="${SHA:0:7}"
WORKDIR="$HOME/Projects/rocm-aic.${SHORT}"
CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-$HOME/Projects/rocm-aic-ci}"
TARBALL_DIR="${CI_STORAGE_ROOT}/images/aic-ci-${SHORT}"

_cleanup() {
    echo "=== Cleaning up ==="
    # Keep logs/: _harvest_logs has already archived them outside WORKDIR, but a
    # failed archive should not also destroy the originals.  Mirrors
    # spur-smoke-test.sh, which preserves logs/ for the same reason.
    find "${WORKDIR}" -mindepth 1 -maxdepth 1 -not -name logs -exec rm -rf {} +
    rm -rf "${TARBALL_DIR}"
}

# The per-job accuracy logs (scores, per-arm container logs) live under
# WORKDIR/logs/$SLURM_JOB_ID and were previously destroyed by the cleanup above,
# leaving a failed gate with nothing but the Actions stdout tail.  Archive them
# outside WORKDIR before anything deletes it; the runner scp's this back and
# uploads it.  Runs on both paths — the nightly's downstream cliff harvest
# removes WORKDIR too.
_harvest_logs() {
    echo "=== Archiving accuracy logs ==="
    if [[ ! -d "${WORKDIR}/logs" ]]; then
        echo "WARNING: no ${WORKDIR}/logs to archive" >&2
        return 0
    fi
    mkdir -p "${ACCURACY_LOG_ARCHIVE_DIR}" || {
        echo "WARNING: could not create ${ACCURACY_LOG_ARCHIVE_DIR}" >&2; return 0; }
    if tar -czf "${ACCURACY_LOG_ARCHIVE_DIR}/accuracy-${SHORT}.tar.gz" -C "${WORKDIR}" logs; then
        echo "=== Archived to ${ACCURACY_LOG_ARCHIVE_DIR}/accuracy-${SHORT}.tar.gz ==="
    else
        echo "WARNING: failed to archive accuracy logs from ${WORKDIR}/logs" >&2
    fi
    # Prune archives from earlier runs; without this the dir grows unbounded.
    find "${ACCURACY_LOG_ARCHIVE_DIR}" -maxdepth 1 -name 'accuracy-*.tar.gz' -mtime +7 \
        -delete || echo "WARNING: could not prune old accuracy log archives" >&2
    return 0
}

if [[ "${KEEP_ARTIFACTS}" == "1" ]]; then
    # The downstream stage runs regardless of how this one ends and reuses the
    # artifacts, so removing them here — even on failure — would break it.
    echo "=== KEEP_ARTIFACTS=1: downstream stage owns cleanup ==="
    trap _harvest_logs EXIT
else
    # Terminal stage: always clean up, but harvest first.
    trap '_harvest_logs; _cleanup' EXIT
fi

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
    echo "ERROR: ${TARBALL_DIR} not found — did dist-build run first?" >&2
    exit 1
fi

echo "=== Running ${AIC_ACCURACY_TEST_TARGET} (AIC_IMAGE_NAME=${AIC_IMAGE_NAME}) ==="
AIC_SPUR_CLUSTER=1 \
    AIC_IMAGE_NAME="${AIC_IMAGE_NAME}" \
    AIC_IMAGE_DIR="${TARBALL_DIR}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    make -C "${WORKDIR}" "${AIC_ACCURACY_TEST_TARGET}"

echo "=== ${AIC_ACCURACY_TEST_TARGET} complete ==="
REMOTE

# Fetch the archive whatever the gate concluded — a red gate is precisely when
# these logs are the only post-mortem.  `set -e` would have skipped this on
# failure, hence the explicit rc capture above.
mkdir -p "${AIC_ACCURACY_LOG_DEST}"
if scp -q "${AIC_SPUR_HOST}:${ACCURACY_LOG_ARCHIVE_DIR}/accuracy-${SHORT}.tar.gz" \
        "${AIC_ACCURACY_LOG_DEST}/"; then
    echo "Accuracy logs retrieved to ${AIC_ACCURACY_LOG_DEST}/accuracy-${SHORT}.tar.gz"
else
    echo "WARNING: could not retrieve accuracy logs for ${SHORT}" >&2
fi

if [[ "${rc}" -ne 0 ]]; then
    echo "Accuracy test FAILED for ${SHORT} (exit ${rc})" >&2
    exit "${rc}"
fi

echo "Accuracy test passed for ${SHORT}"
