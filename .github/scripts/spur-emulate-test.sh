#!/usr/bin/env bash
set -euo pipefail

# Runs on the self-hosted runner; SSHes to the SPUR head node (AIC_SPUR_HOST),
# builds the CPU-only emulation image and serve-tests it.
#
# Unlike spur-smoke-test.sh / spur-tiny-test.sh / spur-cliff.sh, this stage needs
# NO GPU at either end: the image is built with --target emulate and
# VLLM_TARGET_DEVICE=empty (no GPU kernels compiled at all), and the test runs
# the emulator on a CPU-only node with no /dev/kfd mapped into the container.
# So it can run while every GPU on the cluster is busy, and it is the only CI
# stage that exercises the serving path without competing for hardware.
#
# It is self-contained -- it builds its own image rather than reusing the GPU
# tarball from spur-dist-build.sh -- so it does not have to be chained behind
# the dist-build/smoke/tiny/cliff sequence.
#
# What the test asserts is in .slurm/run-build-distribute.sh (cmd_emulate_test):
# no GPU device in the container, a completion with completion_tokens > 0, the
# LLM-Emu hook active, steps drawn from the profile pack, and no model weights
# loaded.  Those assertions matter because the dangerous failure mode is a
# silent fall-through to real execution rather than a crash.

SHA="${1:?usage: $0 <full-sha>}"
SHORT="${SHA:0:7}"
AIC_EMULATE_IMAGE="rocm-aic-ci-emu-${SHORT}:latest"
AIC_SPUR_HOST="${AIC_SPUR_HOST:?AIC_SPUR_HOST must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_HOST="${AIC_SPUR_HOST//[$'\t\r\n ']}"
AIC_SHARED_NFS="${AIC_SHARED_NFS:?AIC_SHARED_NFS must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER:?AIC_SPUR_CONTROLLER must be set (e.g. via GitHub repo variable)}"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-0}"
REPO="https://github.com/ROCm/rocm-aic.git"

ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "${AIC_SPUR_HOST}" env \
    SHA="${SHA}" \
    REPO="${REPO}" \
    AIC_EMULATE_IMAGE="${AIC_EMULATE_IMAGE}" \
    AIC_SHARED_NFS="${AIC_SHARED_NFS}" \
    KEEP_ARTIFACTS="${KEEP_ARTIFACTS}" \
    AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER}" \
    SPUR_CONTROLLER_ADDR="${AIC_SPUR_CONTROLLER}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    bash << 'REMOTE'
set -euo pipefail

SHORT="${SHA:0:7}"
WORKDIR="$HOME/Projects/rocm-aic.emu.${SHORT}"
# $USER here is the head-node user — define paths here, not on the runner.
TARBALL_DIR="${AIC_SHARED_NFS}/${USER}/images/aic-ci-emu-${SHORT}"
# The emulator downloads only the model's config + tokenizer (weights are never
# loaded), so it can share the small persistent HF cache the tiny test uses.
EMU_HF_HOME="${AIC_SHARED_NFS}/${USER}/tiny-hf"

_cleanup() {
    echo "=== Cleaning up ==="
    rm -rf "${WORKDIR}" "${TARBALL_DIR}"
}
if [[ "${KEEP_ARTIFACTS}" == "1" ]]; then
    cleanup_on_fail() { echo "=== Emulate test failed — cleaning up ==="; _cleanup; }
    trap cleanup_on_fail ERR
else
    trap _cleanup EXIT
fi

ACTUAL_SHA="$(git -C "${WORKDIR}" rev-parse HEAD 2>/dev/null || true)"
if [[ ! -d "${WORKDIR}" || "${ACTUAL_SHA}" != "${SHA}" ]]; then
    echo "=== (Re-)cloning ${REPO} at ${SHA} ==="
    rm -rf "${WORKDIR}"
    git clone --filter=blob:none --no-single-branch "${REPO}" "${WORKDIR}"
    git -C "${WORKDIR}" checkout "${SHA}"
fi

echo "=== Building the emulation image (no GPU kernels) ==="
AIC_SPUR_CLUSTER=1 \
    AIC_EMULATE_IMAGE="${AIC_EMULATE_IMAGE}" \
    AIC_IMAGE_DIR="${TARBALL_DIR}" \
    make -C "${WORKDIR}" dist-build-emulate

echo "=== Serve-testing it on a CPU-only node ==="
AIC_SPUR_CLUSTER=1 \
    AIC_EMULATE_IMAGE="${AIC_EMULATE_IMAGE}" \
    AIC_IMAGE_DIR="${TARBALL_DIR}" \
    AIC_TINY_HF_HOME="${EMU_HF_HOME}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    make -C "${WORKDIR}" emulate-test

echo "=== emulate-test complete ==="
REMOTE

echo "Emulate test passed for ${SHORT}"
