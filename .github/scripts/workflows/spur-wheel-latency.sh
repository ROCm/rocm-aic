#!/usr/bin/env bash
# -E inherit ERR trap in functions and subshells,
# -e exit the script upon commands failing,
# -u to error if the script ever references a variable that is unset,
# -o pipefail to error if any stage of a pipeline fails.
set -Eeuo pipefail

# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Runs from a workflow checkout on the self-hosted runner, which is NOT a SPUR
# node: it has no /shared_nfs and no sbatch.  SSHes to the SPUR head node
# (AIC_SPUR_HOST), clones the repo at the given SHA, builds the AIC images at
# that SHA, downloads the named wheels from the release to shared NFS (compute
# nodes have no GitHub access), submits .slurm/test-nightly-wheels.sbatch and
# waits for it, then copies the job log back to OUT_DIR for the render step.
#
# Usage:
#   bash .github/scripts/workflows/spur-wheel-latency.sh \
#       <full-sha> <release-tag> <wheel-name-csv> <out-dir>
#
# Env: AIC_SPUR_HOST, AIC_SHARED_NFS, AIC_CI_STORAGE_ROOT (required);
#      VLLM_MODEL, HF_HOME, AIC_BUILD_EXCLUDE_NODES (optional);
#      AIC_IMAGE_DIR (optional): test in the images already in this dir
#      instead of building them.

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

usage="usage: $0 <full-sha> <release-tag> <wheel-name-csv> <out-dir>"
SHA="${1:?${usage}}"
TAG="${2:?${usage}}"
WHEELS_CSV="${3:?${usage}}"
OUT_DIR="${4:?${usage}}"

# All of these land in remote paths, a URL and a remote env assignment.
[[ "${SHA}" =~ ^[0-9a-f]{40}$ ]] \
    || { echo "ERROR: sha must be a full commit SHA, got: ${SHA}" >&2; exit 2; }
[[ "${TAG}" =~ ^[A-Za-z0-9._-]+$ ]] \
    || { echo "ERROR: release tag must be alphanumeric, got: ${TAG}" >&2; exit 2; }
[[ "${WHEELS_CSV}" =~ ^[A-Za-z0-9._+-]+\.whl(,[A-Za-z0-9._+-]+\.whl)*$ ]] \
    || { echo "ERROR: wheel list must be comma-separated .whl names, got: ${WHEELS_CSV}" >&2; exit 2; }

SHORT="${SHA:0:7}"
REPO="https://github.com/ROCm/rocm-aic.git"
AIC_SPUR_HOST="${AIC_SPUR_HOST:?AIC_SPUR_HOST must be set}"
AIC_SPUR_HOST="${AIC_SPUR_HOST//[$'\t\r\n ']}"
AIC_SHARED_NFS="${AIC_SHARED_NFS:?AIC_SHARED_NFS must be set}"
AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:?AIC_CI_STORAGE_ROOT must be set}"
[[ "${AIC_CI_STORAGE_ROOT}" == /* ]] \
    || { echo "ERROR: AIC_CI_STORAGE_ROOT must be absolute, got: ${AIC_CI_STORAGE_ROOT}" >&2; exit 2; }
AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT%/}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The checkout's copy, not the installed one: this script is not installed, so
# it must not depend on what /usr/local/lib/aic-ci happens to hold.
# shellcheck source=.github/scripts/runners/spur-ci-common.sh
source "${SCRIPT_DIR}/../runners/spur-ci-common.sh"
aic_ci_session_init "${SHORT}" wheel-latency

# Must match RUN_DIR in the remote stage: that side writes the log there, this
# side scp's it back and then removes it.
REMOTE_RUN_DIR="${AIC_CI_STORAGE_ROOT}/wheel-latency/${SHORT}.${AIC_CI_RUN_KEY}"

remote_rc=0
aic_ci_ssh_bash \
    SHA="${SHA}" \
    REPO="${REPO}" \
    TAG="${TAG}" \
    WHEELS_CSV="${WHEELS_CSV}" \
    AIC_SHARED_NFS="${AIC_SHARED_NFS}" \
    AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT}" \
    AIC_IMAGE_DIR="${AIC_IMAGE_DIR:-}" \
    VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-4B}" \
    HF_HOME="${HF_HOME:-${AIC_SHARED_NFS%/}/huggingface}" \
    AIC_SPUR_PARTITION="${AIC_SPUR_PARTITION:-amd-spur-v2}" \
    AIC_BUILD_EXCLUDE_NODES="${AIC_BUILD_EXCLUDE_NODES:-}" << 'REMOTE' || remote_rc=$?
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
    # || : so a failed write cannot replace the exit status.
    printf '%s: command failed (exit %s, pipeline status %s) at line %s%s\n' \
        'spur-wheel-latency.sh (remote)' "$1" "$2" "$3" "${4:+ in ${4// / <- }}" >&2 || :
}
trap '_on_err "$?" "${PIPESTATUS[*]}" "$LINENO" "${FUNCNAME[*]-}"' ERR

SHORT="${SHA:0:7}"
WORKDIR="$HOME/Projects/rocm-aic.${SHORT}.${AIC_CI_RUN_KEY}.${AIC_CI_STAGE}"
RUN_DIR="${AIC_CI_STORAGE_ROOT}/wheel-latency/${SHORT}.${AIC_CI_RUN_KEY}"
WHEEL_DIR="${RUN_DIR}/wheels"
# The images the wheels are tested in.  Nothing else keeps a current set: CI
# builds land in a per-SHA dir the nightly chain deletes, and the top-level
# images/ holds whatever a manual build last left there -- an image older than
# the scripts that drive it (no curl, no venv) fails the test for that alone.
# So build this SHA's, unless the caller named a dir of prebuilt ones.
BUILT_IMAGE_DIR=""
if [[ -z "${AIC_IMAGE_DIR}" ]]; then
    BUILT_IMAGE_DIR="${AIC_CI_STORAGE_ROOT}/images/aic-wheel-latency-${SHORT}.${AIC_CI_RUN_KEY}"
    AIC_IMAGE_DIR="${BUILT_IMAGE_DIR}"
fi
CONTROL_PREFIX="${AIC_CI_STORAGE_ROOT}/control/${SHORT}.${AIC_CI_RUN_KEY}.${AIC_CI_STAGE}"
PID_FILE="${CONTROL_PREFIX}.pid"
JOB_FILE="${CONTROL_PREFIX}.job"
CANCEL_FILE="${CONTROL_PREFIX}.cancel"

mkdir -p "${AIC_CI_STORAGE_ROOT}/control"
printf '%s\n' "${BASHPID}" > "${PID_FILE}"
if [[ -e "${CANCEL_FILE}" ]]; then
    echo "CI session was cancelled before remote startup completed" >&2
    rm -f "${PID_FILE}" "${JOB_FILE}" "${CANCEL_FILE}" || :
    exit 143
fi
# The build records its SPUR job here, so a cancelled run can scancel it.
export AIC_CI_ACTIVE_JOB_FILE="${JOB_FILE}"

_cleanup() {
    local rc=$?
    trap - EXIT
    echo "=== Cleaning up run-attempt worktree, wheels and built images ==="
    # rm -rf ignores the "" an unbuilt BUILT_IMAGE_DIR expands to.
    rm -rf "${WORKDIR}" "${WHEEL_DIR}" "${BUILT_IMAGE_DIR}" \
        || echo "WARNING: cleanup could not fully remove ${WORKDIR} ${WHEEL_DIR} ${BUILT_IMAGE_DIR}" >&2
    (( rc != 0 )) || rm -f "${PID_FILE}" "${JOB_FILE}" "${CANCEL_FILE}" || :
    # Bounded retention for logs nobody came back for (an interrupted run).
    local prune_err="" prune_rc=0
    prune_err=$(find "${AIC_CI_STORAGE_ROOT}/wheel-latency" -mindepth 1 -maxdepth 1 \
        -type d -mtime +14 -exec rm -rf {} + 2>&1 >/dev/null) || prune_rc=$?
    if (( prune_rc != 0 )) || [[ -n "${prune_err}" ]]; then
        echo "WARNING: could not expire wheel-latency runs older than 14d" \
            "(rc=${prune_rc}): ${prune_err:-no error output}" >&2
    fi
    exit "${rc}"
}
trap _cleanup EXIT

echo "=== Cloning ${REPO} at ${SHA} into ${WORKDIR} ==="
rm -rf "${WORKDIR}"
git clone --filter=blob:none --no-single-branch "${REPO}" "${WORKDIR}"
git -C "${WORKDIR}" checkout "${SHA}"

if [[ -n "${BUILT_IMAGE_DIR}" ]]; then
    echo "=== Building the gfx950 AIC images at ${SHORT} into ${BUILT_IMAGE_DIR} ==="
    mkdir -p "${BUILT_IMAGE_DIR}"
    # The test runs on gfx950 only, and dist-build compiles every vLLM arch:
    # the fast build is the gfx950 one.  The build cache is spur-dist-build.sh's,
    # so CI's earlier layers are reused.  The build's default allocation is
    # four hours; capped at two, it and the one-hour test fit the job timeout.
    (cd "${WORKDIR}" && AIC_SPUR_CLUSTER=1 \
        AIC_BUILD_TIME=02:00:00 \
        AIC_IMAGE_DIR="${BUILT_IMAGE_DIR}" \
        AIC_CACHE_DIR="${AIC_SHARED_NFS%/}/${USER:-$(id -un)}/buildcache" \
        make dist-build-fast)
else
    echo "=== Using the prebuilt images in ${AIC_IMAGE_DIR} ==="
fi

echo "=== Downloading ${TAG} wheels to ${WHEEL_DIR} ==="
rm -rf "${WHEEL_DIR}"
mkdir -p "${WHEEL_DIR}"
IFS=, read -r -a wheels <<<"${WHEELS_CSV}"
for w in "${wheels[@]}"; do
    # Local version labels put a '+' in the name; the asset URL escapes it.
    curl -fsSL --retry 3 -o "${WHEEL_DIR}/${w}" \
        "https://github.com/ROCm/rocm-aic/releases/download/${TAG}/${w//+/%2B}"
    printf '  %s (%s)\n' "${w}" "$(du -h "${WHEEL_DIR}/${w}" | cut -f1)"
done

echo "=== Submitting the wheel latency benchmark ==="
exclude=()
[[ -z "${AIC_BUILD_EXCLUDE_NODES}" ]] || exclude=(--exclude="${AIC_BUILD_EXCLUDE_NODES}")
submit_out="$(cd "${WORKDIR}" && \
    WHEEL_DIR="${WHEEL_DIR}" NIGHTLY_TAG="${TAG}" AIC_IMAGE_DIR="${AIC_IMAGE_DIR}" \
    sbatch --chdir="${WORKDIR}" --partition="${AIC_SPUR_PARTITION}" \
        "${exclude[@]}" --export=ALL .slurm/test-nightly-wheels.sbatch 2>&1)" \
    || { echo "ERROR: sbatch failed: ${submit_out}" >&2; exit 1; }
job_id="$(grep -oE '[0-9]+$' <<<"${submit_out}" | tail -1)"
[[ -n "${job_id}" ]] || { echo "ERROR: no job id in sbatch output: ${submit_out}" >&2; exit 1; }
printf '%s\n' "${job_id}" > "${JOB_FILE}"
echo "Submitted SPUR job ${job_id}"

# SPUR's squeue may ignore -j and return the whole queue, so select the row by
# id.  A failed squeue RPC is not "the job left the queue": retry it.  SPUR can
# also leave a finished job RUNNING until its walltime, so the job's exit file
# ends the wait too.
log_dir="${WORKDIR}/logs/${job_id}"
queue_failures=0
while :; do
    [[ -f "${log_dir}/wheel-test.exit" ]] && break
    if out="$(squeue -h -o '%i %T' 2>&1)"; then
        queue_failures=0
        state="$(awk -v id="${job_id}" '$1 == id { print $2 }' <<<"${out}")"
        [[ -n "${state}" ]] || break
        echo "  $(date -u +%H:%M:%S) job ${job_id}: ${state}"
    else
        (( ++queue_failures <= 10 )) \
            || { echo "ERROR: squeue failed 10 times in a row: ${out}" >&2; exit 1; }
        echo "  squeue failed (${queue_failures}/10): ${out}" >&2
    fi
    sleep 30
done

# The exit file is written by the job's EXIT trap; NFS can lag its close.
for _ in $(seq 1 30); do
    [[ -f "${log_dir}/wheel-test.exit" ]] && break
    sleep 2
done
mkdir -p "${RUN_DIR}"
shopt -s nullglob
logs=("${log_dir}"/*.out "${log_dir}"/*.log)
shopt -u nullglob
(( ${#logs[@]} == 0 )) || cp "${logs[@]}" "${RUN_DIR}/"
if [[ ! -f "${log_dir}/wheel-test.exit" ]]; then
    echo "ERROR: job ${job_id} left no exit file in ${log_dir}; it died before finishing" >&2
    [[ -f "${log_dir}/wheel-test.out" ]] && tail -50 "${log_dir}/wheel-test.out" >&2
    exit 1
fi
job_rc="$(tr -d '[:space:]' < "${log_dir}/wheel-test.exit")"
echo "=== job ${job_id} exited ${job_rc} ==="
tail -40 "${log_dir}/wheel-test.out"
[[ "${job_rc}" == 0 ]] || exit 1
REMOTE

# Copied back on failure too: the log is what the artifact upload is for.
echo "=== Copying the benchmark log to ${OUT_DIR} ==="
rm -rf "${OUT_DIR}"
copy_rc=0
aic_ci_scp_from "${REMOTE_RUN_DIR}" "${OUT_DIR}" || copy_rc=$?
if (( copy_rc == 0 )); then
    ssh -o ConnectTimeout=15 "${AIC_SPUR_HOST}" rm -rf "${REMOTE_RUN_DIR}" \
        || echo "WARNING: could not remove ${REMOTE_RUN_DIR} on ${AIC_SPUR_HOST}" >&2
else
    echo "WARNING: could not copy ${REMOTE_RUN_DIR} back (scp exit ${copy_rc})" >&2
fi
(( remote_rc == 0 )) || exit "${remote_rc}"
[[ -s "${OUT_DIR}/wheel-test.out" ]] \
    || { echo "ERROR: no wheel-test.out copied back to ${OUT_DIR}" >&2; exit 1; }
