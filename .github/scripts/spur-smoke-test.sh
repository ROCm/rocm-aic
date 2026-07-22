#!/usr/bin/env bash
set -euo pipefail

# Runs on the self-hosted runner; SSHes to the SPUR head node (AIC_SPUR_HOST) and runs smoke-test
# against the tarball produced by spur-dist-build.sh for the same SHA.
# The clone and tarball are left in place for spur-tiny-test.sh (the next stage)
# to use; spur-tiny-test.sh owns the final cleanup.  On failure, cleans up
# immediately so no stale state is left behind.

SHA="${1:?usage: $0 <full-sha>}"
SHORT="${SHA:0:7}"
AIC_IMAGE="rocm-aic-ci-${SHORT}:latest"
AIC_SPUR_HOST="${AIC_SPUR_HOST:?AIC_SPUR_HOST must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_HOST="${AIC_SPUR_HOST//[$'\t\r\n ']}"
AIC_SHARED_NFS="${AIC_SHARED_NFS:?AIC_SHARED_NFS must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER:?AIC_SPUR_CONTROLLER must be set (e.g. via GitHub repo variable)}"
TARBALL_DIR="${AIC_SHARED_NFS}/${USER}/images/aic-ci-${SHORT}"

ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "${AIC_SPUR_HOST}" env \
    SHA="${SHA}" \
    AIC_IMAGE="${AIC_IMAGE}" \
    TARBALL_DIR="${TARBALL_DIR}" \
    AIC_SHARED_NFS="${AIC_SHARED_NFS}" \
    AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER}" \
    SPUR_CONTROLLER_ADDR="${AIC_SPUR_CONTROLLER}" \
    bash << 'REMOTE'
set -euo pipefail

SHORT="${SHA:0:7}"
WORKDIR="$HOME/Projects/rocm-aic.${SHORT}"

cleanup_on_fail() {
    echo "=== Smoke test failed — cleaning up ==="
    rm -rf "${WORKDIR}" "${TARBALL_DIR}"
}
trap cleanup_on_fail ERR

echo "=== Running smoke-test (AIC_IMAGE=${AIC_IMAGE}) ==="
AIC_SPUR_CLUSTER=1 \
    AIC_IMAGE="${AIC_IMAGE}" \
    AIC_IMAGE_DIR="${TARBALL_DIR}" \
    make -C "${WORKDIR}" smoke-test

echo "=== smoke-test complete ==="
REMOTE

echo "Smoke test passed for ${SHORT}"
