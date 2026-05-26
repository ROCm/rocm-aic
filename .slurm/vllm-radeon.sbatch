#!/bin/bash
#SBATCH --job-name=vllm-radeon
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=.slurm/logs/vllm-radeon-%j.log
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Build and run recipies/vllm-radeon on a ROCm GPU node (Docker).
#
# Submit from repo root:
#   bash .slurm/run-vllm-radeon.sh
#   sbatch --export=ALL,RADEON_NVME_BASE=/mnt/nvme .slurm/vllm-radeon.sbatch
#
# GPU node selection (optional, via run-vllm-radeon.sh or sbatch CLI):
#   RADEON_SLURM_CONSTRAINT=MARKHAM              # any Markham ROCm GPU
#   RADEON_SLURM_CONSTRAINT='MARKHAM&GFX942'       # MI300X only
#   RADEON_SLURM_CONSTRAINT='MARKHAM&(GFX90A|GFX942)'  # MI210 + MI300
#   ROCM_ARCH=gfx1201  # force image build target (else auto-detected on node)
#
# Required: HF_TOKEN or HF_TOKEN_FILE
# Storage (when RADEON_NVME_BASE is unset): auto on the compute node —
#   1) mount blank nvme*n*  2) use an existing NVMe mount (/mnt, /local, …)
#   3) /scratch/$USER/vllm-radeon/lmcache-<jobid>  4) /tmp job dir
#   RADEON_NVME_MKFS=1 by default when RADEON_NVME_BASE is unset (blank nvme*n* only)
#   RADEON_NVME_MKFS=0 to skip mkfs; override RADEON_NVME_BASE=/path to use existing storage
#
# Benchmark: RADEON_BENCHMARK=gutenberg|gutenberg_serial|none|long_doc_qa|test_aic
# Gutenberg (default): run-long-parallel.sh — RADEON_RUN_LONG_WORKERS (default 4)
# Gutenberg fixtures (generate once): RADEON_GUTENBERG_DATA_ROOT or RADEON_SHARED_ROOT/gutenberg
# HF Hub weights: golden cache RADEON_HF_HOME (default /scratch/$USER/vllm-radeon/hf)
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

# RADEON_NVME_BASE set after optional auto-detect (below)
DATA_HOST=""
HF_HOST=""
CONTAINER_LOG_HOST="${REPORT_DIR}/logs"
RADEON_LMCACHE_IO="${RADEON_LMCACHE_IO:-hipfile}"
RADEON_BENCHMARK="${RADEON_BENCHMARK:-gutenberg}"

BUILD_RC=1
RUN_RC=1
PHASE_RC=1

# shellcheck source=lib/vllm-radeon-common.sh
source "${SLURM_LIB}/vllm-radeon-common.sh"

_radeon_env_aliases
_radeon_resolve_model_env

cleanup() {
    _radeon_job_cleanup
}
trap cleanup EXIT

cd "${REPO_DIR}"
mkdir -p "${JOB_ROOT}" "${REPORT_DIR}" "${CONTAINER_LOG_HOST}" \
    "${SLURM_LOG_DIR}" "${SLURM_DIR}/logs"

_radeon_log "REPO_DIR=${REPO_DIR} RECIPE_DIR=${RECIPE_DIR} JOB_ROOT=${JOB_ROOT}"

_radeon_detect_rocm_arch
_radeon_resolve_hf_token
_radeon_metadata_init

_radeon_docker_cleanup

# Format blank nvme*n* when auto-discovering storage (not when RADEON_NVME_BASE is preset).
if [[ -z "${RADEON_NVME_BASE:-}" ]]; then
    : "${RADEON_NVME_MKFS:=1}"
    export RADEON_NVME_MKFS
fi

if [[ -n "${RADEON_NVME_BASE:-}" ]]; then
    _radeon_nvme_setup || true
else
    _radeon_nvme_auto_use || true
    if [[ -z "${RADEON_NVME_BASE:-}" ]]; then
        _radeon_nvme_scratch_fallback || true
    fi
fi
RADEON_NVME_BASE="${RADEON_NVME_BASE:-${JOB_ROOT}}"
export RADEON_NVME_BASE
_radeon_metadata_append "RADEON_NVME_BASE" "${RADEON_NVME_BASE}"
DATA_HOST="${RADEON_NVME_BASE}/data"
_radeon_resolve_hf_home
mkdir -p "${CONTAINER_LOG_HOST}" "${DATA_HOST}" \
    "${HF_HOST}/hub" "${HF_HOST}/datasets" "${HF_HOST}/vllm" \
    "${HF_HOST}/vllm_config" "${HF_HOST}/torch" "${HF_HOST}/torch_inductor"

if [[ "${RADEON_NVME_BASE}" == "${JOB_ROOT}" ]]; then
    _radeon_log "WARN: using JOB_ROOT for LMCache data (see NVMe inventory above if logged)"
else
    _radeon_log "Using RADEON_NVME_BASE=${RADEON_NVME_BASE} (LMCache data)"
fi
_radeon_log "Using RADEON_HF_HOME=${HF_HOST} (Hub / vLLM model cache)"
_radeon_metadata_append "VLLM_MODEL" "$(_radeon_served_model)"
_radeon_hf_fix_ownership

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
    VLLM_MODEL="${VLLM_MODEL:-}" \
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
    gutenberg | run_long | run-long | run_long_parallel | run-long-parallel)
        if _radeon_run_gutenberg; then
            RUN_RC=0
        else
            RUN_RC=1
        fi
        ;;
    gutenberg_serial | run_long_serial)
        if _radeon_run_gutenberg_serial; then
            RUN_RC=0
        else
            RUN_RC=1
        fi
        ;;
    long_doc_qa | long-doc-qa)
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
_radeon_metadata_append "PHASE_RC" "${PHASE_RC}"
_radeon_metadata_append "T_RUN" "${T_RUN_START} .. ${T_RUN_END}"

if [[ "${RUN_RC}" -eq 0 ]]; then
    PHASE_RC=0
else
    PHASE_RC=1
fi

_radeon_write_summary
_radeon_summarize_job
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
