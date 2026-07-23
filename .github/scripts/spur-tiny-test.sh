#!/usr/bin/env bash
set -euo pipefail

# Runs on the self-hosted runner; SSHes to amd-aic-spur and runs tiny-test
# against the tarball produced by spur-dist-build.sh for the same SHA (the stage
# after spur-smoke-test.sh).  tiny-test brings up the compose MP stack
# (standalone lmcache server + vLLM LMCacheMPConnector) with a tiny model and
# asserts one non-empty chat completion.
#
# Cleanup ownership depends on whether a cliff stage follows:
#   * PR flow (dist-build -> smoke-test -> tiny-test): tiny-test is terminal, so
#     it owns the final cleanup (removes the clone + tarball on exit).
#   * Nightly (dist-build -> smoke -> tiny -> cliff): cliff runs next and needs
#     the artifacts, so the nightly tiny-test step sets KEEP_ARTIFACTS=1 and this
#     script only cleans up on failure (spur-cliff.sh does the final cleanup).
# The tiny model is cached in a persistent shared HF dir (outside the tarball
# dir) so it is downloaded once and reused across runs.

SHA="${1:?usage: $0 <full-sha>}"
SHORT="${SHA:0:7}"
AIC_IMAGE="rocm-aic-ci-${SHORT}:latest"
TARBALL_DIR="/shared_nfs/${USER}/images/aic-ci-${SHORT}"
TINY_HF_HOME="/shared_nfs/${USER}/tiny-hf"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-0}"

ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=4 amd-aic-spur env \
    SHA="${SHA}" \
    AIC_IMAGE="${AIC_IMAGE}" \
    TARBALL_DIR="${TARBALL_DIR}" \
    TINY_HF_HOME="${TINY_HF_HOME}" \
    KEEP_ARTIFACTS="${KEEP_ARTIFACTS}" \
    bash << 'REMOTE'
set -euo pipefail

SHORT="${SHA:0:7}"
WORKDIR="$HOME/Projects/rocm-aic.${SHORT}"

_cleanup() {
    echo "=== Cleaning up ==="
    rm -rf "${WORKDIR}" "${TARBALL_DIR}"
}
if [[ "${KEEP_ARTIFACTS}" == "1" ]]; then
    # A cliff stage follows and reuses the artifacts; only clean up on failure.
    cleanup_on_fail() { echo "=== Tiny test failed — cleaning up ==="; _cleanup; }
    trap cleanup_on_fail ERR
else
    # Terminal stage: always clean up.
    trap _cleanup EXIT
fi

echo "=== Running tiny-test (AIC_IMAGE=${AIC_IMAGE}) ==="
AIC_SPUR_CLUSTER=1 \
    AIC_IMAGE="${AIC_IMAGE}" \
    AIC_IMAGE_DIR="${TARBALL_DIR}" \
    AIC_TINY_HF_HOME="${TINY_HF_HOME}" \
    make -C "${WORKDIR}" tiny-test

echo "=== tiny-test complete ==="
REMOTE

echo "Tiny test passed for ${SHORT}"
