#!/usr/bin/env bash
set -euo pipefail

# Prune expired CI image directories and abandoned SPUR checkouts while always
# protecting the SHA associated with the current workflow run.

SHA="${1:?usage: $0 <full-sha> <retention-days>}"
RETENTION_DAYS="${2:?usage: $0 <full-sha> <retention-days>}"

if [[ ! "${SHA}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "ERROR: expected a full 40-character Git SHA, got: ${SHA}" >&2
    exit 2
fi
if [[ ! "${RETENTION_DAYS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: retention days must be a positive integer, got: ${RETENTION_DAYS}" >&2
    exit 2
fi

AIC_SPUR_HOST="${AIC_SPUR_HOST:?AIC_SPUR_HOST must be set}"
AIC_SPUR_HOST="${AIC_SPUR_HOST//[$'\t\r\n ']}"
AIC_SHARED_NFS="${AIC_SHARED_NFS:?AIC_SHARED_NFS must be set}"
PROTECTED_SHORT="${SHA:0:7}"
RETENTION_MINUTES="$((RETENTION_DAYS * 24 * 60))"

ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "${AIC_SPUR_HOST}" env \
    AIC_SHARED_NFS="${AIC_SHARED_NFS}" \
    PROTECTED_SHORT="${PROTECTED_SHORT}" \
    RETENTION_MINUTES="${RETENTION_MINUTES}" \
    bash << 'REMOTE'
set -euo pipefail

images_root="${AIC_SHARED_NFS%/}/rocm-aic/images"
projects_root="${HOME}/Projects"

prune_image_dirs() {
    [[ -d "${images_root}" ]] || return 0

    while IFS= read -r -d '' candidate; do
        name="${candidate##*/}"
        if [[ ! "${name}" =~ ^aic-ci-([0-9a-fA-F]{7,40})$ ]]; then
            echo "Skipping unexpected artifact directory: ${candidate}" >&2
            continue
        fi
        if [[ "${BASH_REMATCH[1]:0:7}" == "${PROTECTED_SHORT}" ]]; then
            echo "Keeping current artifact directory: ${candidate}"
            continue
        fi
        echo "Pruning expired artifact directory: ${candidate}"
        rm -rf -- "${candidate}"
    done < <(
        find -P "${images_root}" -mindepth 1 -maxdepth 1 -type d \
            -name 'aic-ci-*' -mmin "+${RETENTION_MINUTES}" -print0
    )
}

prune_project_dirs() {
    [[ -d "${projects_root}" ]] || return 0

    while IFS= read -r -d '' candidate; do
        name="${candidate##*/}"
        short=""
        if [[ "${name}" =~ ^rocm-aic\.([0-9a-fA-F]{7})$ ]]; then
            short="${BASH_REMATCH[1]}"
        elif [[ "${name}" =~ ^rocm-aic-coordinator-smoke\.([0-9a-fA-F]{7})\.[a-zA-Z0-9_.-]+$ ]]; then
            short="${BASH_REMATCH[1]}"
        else
            echo "Skipping unexpected project directory: ${candidate}" >&2
            continue
        fi
        if [[ "${short}" == "${PROTECTED_SHORT}" ]]; then
            echo "Keeping current project directory: ${candidate}"
            continue
        fi
        echo "Pruning expired project directory: ${candidate}"
        rm -rf -- "${candidate}"
    done < <(
        find -P "${projects_root}" -mindepth 1 -maxdepth 1 -type d \
            \( -name 'rocm-aic.[0-9a-fA-F]*' \
               -o -name 'rocm-aic-coordinator-smoke.[0-9a-fA-F]*' \) \
            -mmin "+${RETENTION_MINUTES}" -print0
    )
}

prune_image_dirs
prune_project_dirs
REMOTE
