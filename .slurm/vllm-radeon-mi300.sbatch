#!/bin/bash
#SBATCH --job-name=vllm-radeon
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --constraint="MARKHAM&(GFX90A|GFX942)"
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=.slurm/logs/vllm-radeon-%j.log
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Build and run recipies/vllm-radeon on an MI300X Slurm node (Docker).
#
# Submit from repo root:
#   bash .slurm/run-vllm-radeon.sh
#   sbatch --export=ALL,RADEON_NVME_BASE=/mnt/nvme .slurm/vllm-radeon-mi300.sbatch
#
# Required: HF_TOKEN or HF_TOKEN_FILE
# Recommended: RADEON_NVME_BASE=/mnt/nvme (LMCache + HF cache; avoid filling /tmp)
#
# Optional NVMe mkfs on exclusive nodes:
#   RADEON_NVME_MKFS=1 RADEON_NVME_AUTO_DEVICE=1 RADEON_NVME_MOUNT=/mnt/radeon-nvme
#
# Benchmark: RADEON_BENCHMARK=long_doc_qa|test_aic|none (default long_doc_qa)
#
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RECIPE_DIR="${REPO_DIR}/recipies/vllm-radeon"
SLURM_DIR="${REPO_DIR}/.slurm"
SLURM_LIB="${SLURM_DIR}/lib"
JOB_ROOT="/tmp/vllm-radeon-${SLURM_JOB_ID:-local$$}"
REPORT_DIR="${JOB_ROOT}/report"
SLURM_LOG_DIR="${SLURM_DIR}/logs/vllm-radeon-${SLURM_JOB_ID:-local$$}"

IMAGE_NAME="${IMAGE_NAME:-vllm-radeon}"
GPU="${GPU:-0}"
VLLM_PORT="800${GPU}"
CONTAINER_NAME="${CONTAINER_NAME:-${IMAGE_NAME}-gpu${GPU}-${SLURM_JOB_ID:-local$$}}"

RADEON_NVME_BASE="${RADEON_NVME_BASE:-${JOB_ROOT}}"
DATA_HOST="${RADEON_NVME_BASE}/data"
HF_HOST="${RADEON_NVME_BASE}/hf"
CONTAINER_LOG_HOST="${REPORT_DIR}/logs"
RADEON_LMCACHE_IO="${RADEON_LMCACHE_IO:-hipfile}"
RADEON_BENCHMARK="${RADEON_BENCHMARK:-long_doc_qa}"

BUILD_RC=1
RUN_RC=1
PHASE_RC=1

# shellcheck source=lib/vllm-radeon-common.sh
source "${SLURM_LIB}/vllm-radeon-common.sh"

_radeon_env_aliases

cleanup() {
    _radeon_job_cleanup
}
trap cleanup EXIT

cd "${REPO_DIR}"
mkdir -p "${JOB_ROOT}" "${REPORT_DIR}" "${CONTAINER_LOG_HOST}" "${DATA_HOST}" \
    "${HF_HOST}/hub" "${HF_HOST}/datasets" "${HF_HOST}/vllm" \
    "${HF_HOST}/vllm_config" "${HF_HOST}/torch" "${HF_HOST}/torch_inductor" \
    "${SLURM_LOG_DIR}" "${SLURM_DIR}/logs"

_radeon_log "REPO_DIR=${REPO_DIR} RECIPE_DIR=${RECIPE_DIR} JOB_ROOT=${JOB_ROOT}"

if [[ "${RADEON_NVME_BASE}" == /tmp/* || "${RADEON_NVME_BASE}" == "${JOB_ROOT}" ]]; then
    _radeon_log "WARN: RADEON_NVME_BASE=${RADEON_NVME_BASE} — set RADEON_NVME_BASE=/mnt/nvme on real runs"
fi
_radeon_log "Using RADEON_NVME_BASE=${RADEON_NVME_BASE}"

_radeon_detect_rocm_arch
_radeon_resolve_hf_token
_radeon_metadata_init

_radeon_docker_cleanup

if ! _radeon_nvme_setup; then
    PHASE_RC=1
    exit 1
fi
DATA_HOST="${RADEON_NVME_BASE}/data"
HF_HOST="${RADEON_NVME_BASE}/hf"
mkdir -p "${DATA_HOST}" "${HF_HOST}/hub" "${HF_HOST}/datasets" \
    "${HF_HOST}/vllm" "${HF_HOST}/vllm_config" "${HF_HOST}/torch" \
    "${HF_HOST}/torch_inductor"

_radeon_nvme_smart_log "start"
_radeon_bpftrace_nvme_start

if ! _radeon_build_image; then
    PHASE_RC=1
    exit 1
fi

T_RUN_START="$(date -Iseconds)"
export HF_TOKEN
if ! make -C "${RECIPE_DIR}" run-batch \
    IMAGE_NAME="${IMAGE_NAME}" \
    GPU="${GPU}" \
    CONTAINER_NAME="${CONTAINER_NAME}" \
    DATA="${DATA_HOST}" \
    LOG="${CONTAINER_LOG_HOST}" \
    HF_HOME="${HF_HOST}" \
    HF_TOKEN="${HF_TOKEN}" \
    ROCM_ARCH="${ROCM_ARCH}" \
    RADEON_LMCACHE_IO="${RADEON_LMCACHE_IO}" \
    RADEON_LMCACHE_LOG_LEVEL="${RADEON_LMCACHE_LOG_LEVEL:-}" \
    RADEON_LMCACHE_GDS_BUFFER_SIZE="${RADEON_LMCACHE_GDS_BUFFER_SIZE:-}" \
    VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}" \
    TZ="${TZ:-America/Edmonton}" \
    EXTRA_DOCKER_RUN_FLAGS="${EXTRA_DOCKER_RUN_FLAGS:-} -e RADEON_LMCACHE_ENABLE_KV_EVENTS=${RADEON_LMCACHE_ENABLE_KV_EVENTS:-1} -e RADEON_LMCACHE_ENABLE_CHUNK_STATISTICS=${RADEON_LMCACHE_ENABLE_CHUNK_STATISTICS:-1}"; then
    RUN_RC=1
    PHASE_RC=1
    exit 1
fi

sleep 2
_radeon_bpftrace_vfs_start

if ! _radeon_wait_vllm; then
    RUN_RC=1
    PHASE_RC=1
    docker logs "${CONTAINER_NAME}" 2>&1 | tail -200 || true
    exit 1
fi

case "${RADEON_BENCHMARK,,}" in
    none | skip)
        _radeon_log "RADEON_BENCHMARK=none; skipping benchmark"
        RUN_RC=0
        ;;
    long_doc_qa | long-doc-qa | "")
        if _radeon_run_long_doc_qa; then
            RUN_RC=0
        else
            RUN_RC=1
        fi
        ;;
    test_aic | test-aic)
        if _radeon_run_test_aic; then
            RUN_RC=0
        else
            RUN_RC=1
        fi
        ;;
    *)
        _radeon_log "ERROR: unknown RADEON_BENCHMARK=${RADEON_BENCHMARK}"
        RUN_RC=1
        PHASE_RC=1
        exit 1
        ;;
esac

_radeon_collect_lmcache_api

T_RUN_END="$(date -Iseconds)"
_radeon_metadata_append "RUN_RC" "${RUN_RC}"
_radeon_metadata_append "T_RUN" "${T_RUN_START} .. ${T_RUN_END}"

if [[ "${RUN_RC}" -eq 0 ]]; then
    PHASE_RC=0
else
    PHASE_RC=1
fi

_radeon_write_summary
_radeon_write_report
_radeon_collect_artifacts

{
    echo ""
    echo "=== lsblk (nvme lines) ==="
    lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,PKNAME,MODEL 2>/dev/null | grep -i nvme || true
    echo ""
    echo "=== rocm-smi ==="
    rocm-smi 2>/dev/null || true
} >> "${METADATA_FILE}"

if [[ -f "${CONTAINER_LOG_HOST}/server.txt" ]]; then
    {
        echo ""
        echo "=== server.txt (tail 200 lines) ==="
        tail -200 "${CONTAINER_LOG_HOST}/server.txt"
    } >> "${METADATA_FILE}"
fi

_radeon_log "Done PHASE_RC=${PHASE_RC} RUN_RC=${RUN_RC} BUILD_RC=${BUILD_RC}"
_radeon_log "Report: ${SLURM_LOG_DIR}"

exit "${PHASE_RC}"
