#!/usr/bin/env bash

# Shared lifecycle helpers for installed CI wrappers that run work on the SPUR head node.
# Call aic_ci_session_init before starting SSH, then use aic_ci_ssh_bash for the
# remote heredoc.  If the local wrapper is interrupted, its EXIT trap reconnects
# and cancels only the Slurm job/process group recorded for this run and stage.

aic_ci_session_init() {
    local short_sha="${1:?short SHA is required}"
    local stage="${2:?CI stage is required}"
    local run_id="${GITHUB_RUN_ID:-manual-${BASHPID}}"
    local run_attempt="${GITHUB_RUN_ATTEMPT:-1}"

    AIC_CI_RUN_KEY="${run_id}-${run_attempt}"
    AIC_CI_STAGE="${stage}"
    case "${AIC_CI_RUN_KEY}.${AIC_CI_STAGE}.${short_sha}" in
        *[!A-Za-z0-9._-]*)
            echo "ERROR: unsafe CI session identifier" >&2
            exit 2
            ;;
    esac

    AIC_CI_SHORT_SHA="${short_sha}"
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap 'exit 129' HUP
    trap _aic_ci_session_exit EXIT
}

aic_ci_ssh_bash() {
    ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "${AIC_SPUR_HOST}" env \
        PATH="/usr/local/bin:\${PATH}" \
        AIC_CI_RUN_KEY="${AIC_CI_RUN_KEY}" \
        AIC_CI_STAGE="${AIC_CI_STAGE}" \
        "$@" \
        setsid --fork --wait bash
}

_aic_ci_session_exit() {
    local rc=$?
    trap - EXIT INT TERM HUP

    if (( rc != 0 )); then
        echo "=== CI wrapper interrupted/failed; cancelling its remote session ===" >&2
        ssh -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=2 \
            "${AIC_SPUR_HOST}" env \
            AIC_CI_RUN_KEY="${AIC_CI_RUN_KEY}" \
            AIC_CI_STAGE="${AIC_CI_STAGE}" \
            AIC_CI_SHORT_SHA="${AIC_CI_SHORT_SHA}" \
            AIC_CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT}" \
            AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER}" \
            bash <<'REMOTE_CANCEL' || true
set -u
export PATH="/usr/local/bin:${PATH}"

CI_STORAGE_ROOT="${AIC_CI_STORAGE_ROOT:-$HOME/Projects/rocm-aic-ci}"
CONTROL_PREFIX="${CI_STORAGE_ROOT}/control/${AIC_CI_SHORT_SHA}.${AIC_CI_RUN_KEY}.${AIC_CI_STAGE}"
JOB_FILE="${CONTROL_PREFIX}.job"
PID_FILE="${CONTROL_PREFIX}.pid"
CANCEL_FILE="${CONTROL_PREFIX}.cancel"
remote_session_found=0

# Close the small race where cancellation arrives before the remote shell has
# written its PID.  A shell that starts later sees this sentinel and exits
# before cloning or submitting work.
mkdir -p "${CI_STORAGE_ROOT}/control"
: > "${CANCEL_FILE}"

if [[ -s "${JOB_FILE}" ]]; then
    read -r job_id < "${JOB_FILE}" || job_id=""
    if [[ "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "Cancelling Slurm job ${job_id}"
        scancel --controller="${AIC_SPUR_CONTROLLER}" "${job_id}" || true
    fi
fi

if [[ -s "${PID_FILE}" ]]; then
    read -r remote_pid < "${PID_FILE}" || remote_pid=""
    if [[ "${remote_pid}" =~ ^[0-9]+$ ]]; then
        remote_session_found=1
        if kill -0 -- "-${remote_pid}" 2>/dev/null; then
            echo "Terminating remote process group ${remote_pid}"
            kill -TERM -- "-${remote_pid}" 2>/dev/null || true
            for _ in 1 2 3 4 5 6 7 8 9 10; do
                kill -0 -- "-${remote_pid}" 2>/dev/null || break
                sleep 0.2
            done
            kill -KILL -- "-${remote_pid}" 2>/dev/null || true
        fi
    fi
fi

rm -f "${JOB_FILE}" "${PID_FILE}" 2>/dev/null || true
if (( remote_session_found == 1 )); then
    rm -f "${CANCEL_FILE}" 2>/dev/null || true
fi
REMOTE_CANCEL
    fi

    exit "${rc}"
}
