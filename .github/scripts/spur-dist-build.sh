#!/usr/bin/env bash
set -euo pipefail

# Runs on the self-hosted runner; SSHes to the SPUR head node (AIC_SPUR_HOST), clones the repo at
# the current SHA, and runs the requested dist-build target with a CI-scoped
# image name and tarball path. The clone and tarball are left in place for
# spur-smoke-test.sh to use; spur-smoke-test.sh owns the final cleanup.
#
# On failure, cleans up immediately so no stale state is left behind.

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
AIC_PIP_WHEELS_DIR="${AIC_PIP_WHEELS_DIR:-${AIC_SHARED_NFS}/rocm-aic/pip-wheels}"

ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "${AIC_SPUR_HOST}" env \
    SHA="${SHA}" \
    REPO="${REPO}" \
    AIC_IMAGE_NAME="${AIC_IMAGE_NAME}" \
    AIC_DIST_BUILD_TARGET="${AIC_DIST_BUILD_TARGET}" \
    AIC_SHARED_NFS="${AIC_SHARED_NFS}" \
    AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT}" \
    AIC_PIP_WHEELS_DIR="${AIC_PIP_WHEELS_DIR}" \
    AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER}" \
    SPUR_CONTROLLER_ADDR="${AIC_SPUR_CONTROLLER}" \
    bash << 'REMOTE'
set -euo pipefail

SHORT="${SHA:0:7}"
WORKDIR="$HOME/Projects/rocm-aic.${SHORT}"
CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-$HOME/Projects/rocm-aic-ci}"
TARBALL_DIR="${CI_STORAGE_ROOT}/images/aic-ci-${SHORT}"
CACHE_DIR="${CI_STORAGE_ROOT}/buildcache"

cleanup_on_fail() {
    echo "=== Build failed — cleaning up ==="
    rm -rf "${WORKDIR}" "${TARBALL_DIR}"
}
trap cleanup_on_fail ERR

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
    AIC_PIP_WHEELS_DIR="${AIC_PIP_WHEELS_DIR}" \
    make "${AIC_DIST_BUILD_TARGET}"

echo "=== ${AIC_DIST_BUILD_TARGET} complete — tarball in ${TARBALL_DIR} ==="
REMOTE

echo "Build succeeded for ${SHORT}"
