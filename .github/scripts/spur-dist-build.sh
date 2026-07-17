#!/usr/bin/env bash
set -euo pipefail

# Runs on the self-hosted runner; SSHes to amd-aic-spur, clones the repo at
# the current SHA, and runs dist-build with a CI-scoped image name and tarball
# path. The clone and tarball are left in place for spur-smoke-test.sh to use;
# spur-smoke-test.sh owns the final cleanup.
#
# On failure, cleans up immediately so no stale state is left behind.

SHA="${1:?usage: $0 <full-sha>}"
SHORT="${SHA:0:7}"
REPO="git@github.com:ROCm/rocm-icms.git"
AIC_IMAGE="rocm-aic-ci-${SHORT}:latest"
TARBALL_DIR="/tmp/aic-ci-${SHORT}"

ssh amd-aic-spur env \
    SHA="${SHA}" \
    REPO="${REPO}" \
    AIC_IMAGE="${AIC_IMAGE}" \
    TARBALL_DIR="${TARBALL_DIR}" \
    bash << 'REMOTE'
set -euo pipefail

SHORT="${SHA:0:7}"
WORKDIR="$HOME/Projects/rocm-aic.${SHORT}"

cleanup_on_fail() {
    echo "=== Build failed — cleaning up ==="
    rm -rf "${WORKDIR}" "${TARBALL_DIR}"
}
trap cleanup_on_fail ERR

echo "=== Cloning ${REPO} at ${SHA} into ${WORKDIR} ==="
git clone --filter=blob:none "${REPO}" "${WORKDIR}"
cd "${WORKDIR}"
git checkout "${SHA}"

mkdir -p "${TARBALL_DIR}"

echo "=== Running dist-build (AIC_SPUR_CLUSTER=1, AIC_IMAGE=${AIC_IMAGE}) ==="
AIC_SPUR_CLUSTER=1 \
    AIC_IMAGE="${AIC_IMAGE}" \
    AIC_IMAGE_DIR="${TARBALL_DIR}" \
    make dist-build

echo "=== dist-build complete — tarball in ${TARBALL_DIR} ==="
REMOTE

echo "Build succeeded for ${SHORT}"
