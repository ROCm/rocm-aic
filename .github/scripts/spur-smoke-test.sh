#!/usr/bin/env bash
set -euo pipefail

# Runs on the self-hosted runner; SSHes to amd-aic-spur and runs smoke-test
# against the tarball produced by spur-dist-build.sh for the same SHA.
# Always cleans up the clone and tarball dir on exit regardless of pass/fail.

SHA="${1:?usage: $0 <full-sha>}"
SHORT="${SHA:0:7}"
AIC_IMAGE="rocm-aic-ci-${SHORT}:latest"
TARBALL_DIR="/tmp/aic-ci-${SHORT}"

ssh amd-aic-spur env \
    SHA="${SHA}" \
    AIC_IMAGE="${AIC_IMAGE}" \
    TARBALL_DIR="${TARBALL_DIR}" \
    bash << 'REMOTE'
set -euo pipefail

SHORT="${SHA:0:7}"
WORKDIR="$HOME/Projects/rocm-aic.${SHORT}"

cleanup() {
    echo "=== Cleaning up ==="
    rm -rf "${WORKDIR}" "${TARBALL_DIR}"
}
trap cleanup EXIT

echo "=== Running smoke-test (AIC_IMAGE=${AIC_IMAGE}) ==="
AIC_SPUR_CLUSTER=1 \
    AIC_IMAGE="${AIC_IMAGE}" \
    AIC_IMAGE_DIR="${TARBALL_DIR}" \
    make -C "${WORKDIR}" smoke-test

echo "=== smoke-test complete ==="
REMOTE

echo "Smoke test passed for ${SHORT}"
