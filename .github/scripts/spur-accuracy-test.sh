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

ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "${AIC_SPUR_HOST}" env \
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
    bash << 'REMOTE'
set -euo pipefail

SHORT="${SHA:0:7}"
WORKDIR="$HOME/Projects/rocm-aic.${SHORT}"
CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-$HOME/Projects/rocm-aic-ci}"
TARBALL_DIR="${CI_STORAGE_ROOT}/images/aic-ci-${SHORT}"

_cleanup() {
    echo "=== Cleaning up ==="
    rm -rf "${WORKDIR}" "${TARBALL_DIR}"
}
if [[ "${KEEP_ARTIFACTS}" == "1" ]]; then
    # The downstream stage runs regardless of how this one ends and reuses the
    # artifacts, so removing them here — even on failure — would break it.
    echo "=== KEEP_ARTIFACTS=1: downstream stage owns cleanup ==="
else
    # Terminal stage: always clean up.
    trap _cleanup EXIT
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

echo "Accuracy test passed for ${SHORT}"
