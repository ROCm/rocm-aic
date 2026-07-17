#!/usr/bin/env bash
set -euo pipefail

# Runs on the self-hosted runner; SSHes to amd-aic-spur and runs smoke-test
# against the tarball produced by spur-dist-build.sh for the same SHA.
# Always cleans up the clone and tarball dir on exit regardless of pass/fail.

SHA="${1:?usage: $0 <full-sha>}"
SHORT="${SHA:0:7}"
IMAGE_NAME="rocm-aic-ci-${SHORT}"
WORKDIR="\$HOME/Projects/rocm-aic.${SHORT}"
TARBALL_DIR="/tmp/aic-ci-${SHORT}"

ssh amd-aic-spur bash << REMOTE
set -euo pipefail

cleanup() {
    echo "=== Cleaning up ==="
    rm -rf "${WORKDIR}" "${TARBALL_DIR}"
}
trap cleanup EXIT

echo "=== Running smoke-test (image=${IMAGE_NAME}) ==="
AIC_SPUR_CLUSTER=1 \
    IMAGE_NAME="${IMAGE_NAME}" \
    AIC_IMAGE_DIR="${TARBALL_DIR}" \
    make -C "${WORKDIR}" smoke-test

echo "=== smoke-test complete ==="
REMOTE

echo "Smoke test passed for ${SHORT}"
