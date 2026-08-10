#!/usr/bin/env bash
set -euo pipefail

# Runs on the self-hosted runner; SSHes to the SPUR head node (AIC_SPUR_HOST) and
# runs accuracy-test against the tarball produced by spur-dist-build.sh for the same
# SHA (the stage after spur-tiny-test.sh in the nightly chain, or triggered on-demand
# via /run-accuracy).
#
# accuracy-test is a three-phase correctness gate:
#   Phase A  Responses must match .github/accuracy/reference.json exactly
#            (temperature=0, constrained VRAM forces KV eviction to NVMe/DRAM)
#   Phase B  LMCache POSIX pool files on NVMe are md5sum'd and compared against
#            .github/accuracy/nvme-checksums.md5 to verify KV block integrity
#   Phase C  vLLM is restarted (GPU KV cache flushed), prompts re-issued, and
#            responses must still match reference.json (KV blocks retrieved from NVMe)
#
# Bootstrap mode (AIC_ACCURACY_BOOTSTRAP=1): runs the same stack but writes
# reference.json + nvme-checksums.md5 instead of asserting against them.
#
# Cleanup: this script does NOT own the clone/tarball cleanup when KEEP_ARTIFACTS=1
# (nightly chain: cliff follows and owns final cleanup).  On failure it always cleans up.

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
AIC_ACCURACY_BOOTSTRAP="${AIC_ACCURACY_BOOTSTRAP:-0}"
REPO="https://github.com/ROCm/rocm-aic.git"

ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "${AIC_SPUR_HOST}" env \
    SHA="${SHA}" \
    REPO="${REPO}" \
    AIC_IMAGE_NAME="${AIC_IMAGE_NAME}" \
    AIC_ACCURACY_TEST_TARGET="${AIC_ACCURACY_TEST_TARGET}" \
    AIC_SHARED_NFS="${AIC_SHARED_NFS}" \
    AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT}" \
    KEEP_ARTIFACTS="${KEEP_ARTIFACTS}" \
    AIC_ACCURACY_BOOTSTRAP="${AIC_ACCURACY_BOOTSTRAP}" \
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
    # A downstream stage follows and reuses the artifacts; only clean up on failure.
    cleanup_on_fail() { echo "=== Accuracy test failed — cleaning up ==="; _cleanup; }
    trap cleanup_on_fail ERR
else
    # Terminal stage: always clean up.
    trap _cleanup EXIT
fi

# Re-clone if WORKDIR is missing or checked out at the wrong SHA.
ACTUAL_SHA="$(git -C "${WORKDIR}" rev-parse HEAD 2>/dev/null || true)"
if [[ ! -d "${WORKDIR}" || "${ACTUAL_SHA}" != "${SHA}" ]]; then
    echo "=== (Re-)cloning ${REPO} at ${SHA} ==="
    rm -rf "${WORKDIR}"
    git clone --filter=blob:none --no-single-branch "${REPO}" "${WORKDIR}"
    git -C "${WORKDIR}" checkout "${SHA}"
fi

echo "=== Running ${AIC_ACCURACY_TEST_TARGET} (AIC_IMAGE_NAME=${AIC_IMAGE_NAME} bootstrap=${AIC_ACCURACY_BOOTSTRAP}) ==="
AIC_SPUR_CLUSTER=1 \
    AIC_IMAGE_NAME="${AIC_IMAGE_NAME}" \
    AIC_IMAGE_DIR="${TARBALL_DIR}" \
    AIC_ACCURACY_BOOTSTRAP="${AIC_ACCURACY_BOOTSTRAP}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    make -C "${WORKDIR}" "${AIC_ACCURACY_TEST_TARGET}"

echo "=== ${AIC_ACCURACY_TEST_TARGET} complete ==="
REMOTE

echo "Accuracy test passed for ${SHORT}"
