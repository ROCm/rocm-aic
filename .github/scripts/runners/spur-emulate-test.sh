#!/usr/bin/env bash
# -E inherit ERR trap in functions and subshells,
# -e exit the script upon commands failing,
# -u to error if the script ever references a variable that is unset,
# -o pipefail to error if any stage of a pipeline fails.
set -Eeuo pipefail

# Used by the emulate-smoke job in aic-amd-dist-build-fast.yml, which runs on
# a self-hosted runner and SSHes to the SPUR head node (AIC_SPUR_HOST) to build
# and serve-test the emulation image on a CPU-only SPUR node.
#
# NOTE: the standalone nightly emulate test (aic-nightly-emulate-test.yml) no
# longer uses this script — it runs entirely on a GitHub-hosted runner using the
# GHA BuildKit layer cache, with no SPUR dependency.  This script remains for the
# fast-dist-build emulate-smoke job, which must reuse the SPUR-built image
# tarball from the preceding dist-build step on shared NFS.
#
# What the test asserts is in .slurm/run-build-distribute.sh (cmd_emulate_test):
# no GPU device in the container, a completion with completion_tokens > 0, the
# LLM-Emu hook active, steps drawn from the profile pack, and no model weights
# loaded.  Those assertions matter because the dangerous failure mode is a
# silent fall-through to real execution rather than a crash.

# errexit exits without saying where; report it.  Only the line number and
# function call stack are printed, never the command text, which can carry
# secret values.
_on_err() {
    # set -E carries this trap into $(...), where errexit is off.
    [[ $- == *e* ]] || return 0
    # The failing command was a child that reports its own failures.
    [[ -z ${_aic_err_child_reports-} ]] || return 0
    # || : so a failed write cannot replace the exit status.
    printf '%s: command failed (exit %s, pipeline status %s) at line %s%s\n' \
        "${0##*/}" "$1" "$2" "$3" "${4:+ in ${4// / <- }}" >&2 || :
}
trap '_on_err "$?" "${PIPESTATUS[*]}" "$LINENO" "${FUNCNAME[*]-}"' ERR
# Run a child that reports its own failures; only the child reports.
_err_delegate() { local _aic_err_child_reports=1; "$@"; }

# Commit `make install-ci-scripts` deployed; if it predates your change, redeploy.
echo "Installed CI scripts: $(cat "$(dirname -- "${BASH_SOURCE[0]}")/VERSION" || echo unknown)"

SHA="${1:?usage: $0 <full-sha>}"
SHORT="${SHA:0:7}"
AIC_EMULATE_IMAGE="rocm-aic-ci-emu-${SHORT}:latest"
AIC_SPUR_HOST="${AIC_SPUR_HOST:?AIC_SPUR_HOST must be set (e.g. via GitHub repo variable)}"
AIC_SPUR_HOST="${AIC_SPUR_HOST//[$'\t\r\n ']}"
AIC_SHARED_NFS="${AIC_SHARED_NFS:?AIC_SHARED_NFS must be set (e.g. via GitHub repo variable)}"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-0}"
REPO="https://github.com/ROCm/rocm-aic.git"

_err_delegate ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "${AIC_SPUR_HOST}" env \
    SHA="${SHA}" \
    REPO="${REPO}" \
    AIC_EMULATE_IMAGE="${AIC_EMULATE_IMAGE}" \
    AIC_SHARED_NFS="${AIC_SHARED_NFS}" \
    KEEP_ARTIFACTS="${KEEP_ARTIFACTS}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    bash << 'REMOTE'
# -E inherit ERR trap in functions and subshells,
# -e exit the script upon commands failing,
# -u to error if the script ever references a variable that is unset,
# -o pipefail to error if any stage of a pipeline fails.
set -Eeuo pipefail
# errexit exits without saying where; report it.  Only the line number and
# function call stack are printed, never the command text, which can carry
# secret values.
_on_err() {
    # set -E carries this trap into $(...), where errexit is off.
    [[ $- == *e* ]] || return 0
    # The failing command was a child that reports its own failures.
    [[ -z ${_aic_err_child_reports-} ]] || return 0
    # || : so a failed write cannot replace the exit status.
    printf '%s: command failed (exit %s, pipeline status %s) at line %s%s\n' \
        'spur-emulate-test.sh (remote)' "$1" "$2" "$3" "${4:+ in ${4// / <- }}" >&2 || :
}
trap '_on_err "$?" "${PIPESTATUS[*]}" "$LINENO" "${FUNCNAME[*]-}"' ERR
# Run a child that reports its own failures; only the child reports.
_err_delegate() { local _aic_err_child_reports=1; "$@"; }

SHORT="${SHA:0:7}"
WORKDIR="$HOME/Projects/rocm-aic.emu.${SHORT}"
# $USER here is the head-node user — define paths here, not on the runner.
TARBALL_DIR="${AIC_SHARED_NFS}/${USER}/images/aic-ci-emu-${SHORT}"

_cleanup() {
    echo "=== Cleaning up ==="
    rm -rf "${WORKDIR}" "${TARBALL_DIR}"
}
if [[ "${KEEP_ARTIFACTS}" == "1" ]]; then
    cleanup_on_fail() { echo "=== Emulate test failed — cleaning up ==="; _cleanup; }
    trap '_on_err "$?" "${PIPESTATUS[*]}" "$LINENO" "${FUNCNAME[*]-}"; cleanup_on_fail' ERR
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
    _err_delegate make -C "${WORKDIR}" dist-build-emulate

echo "=== Serve-testing it on a CPU-only node ==="
AIC_SPUR_CLUSTER=1 \
    AIC_EMULATE_IMAGE="${AIC_EMULATE_IMAGE}" \
    AIC_IMAGE_DIR="${TARBALL_DIR}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    _err_delegate make -C "${WORKDIR}" emulate-test

echo "=== emulate-test complete ==="
REMOTE

echo "Emulate test passed for ${SHORT}"
