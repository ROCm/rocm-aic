#!/bin/bash
#SBATCH --job-name=vllm-lmcache-hipfile
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=.slurm/logs/vllm-lmcache-hipfile-%j.log
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Build and run recipies/vllm-lmcache-hipfile on a ROCm GPU node (Docker).
#
# Submit from repo root:
#   bash .slurm/run-vllm-lmcache-hipfile.sh
#   sbatch --export=ALL,VLH_NVME_BASE=/mnt/nvme .slurm/vllm-lmcache-hipfile.sbatch
#
# GPU node selection (optional, via run-vllm-lmcache-hipfile.sh or sbatch CLI):
#   VLH_SLURM_CONSTRAINT=MARKHAM              # any Markham ROCm GPU
#   VLH_SLURM_CONSTRAINT='MARKHAM&GFX942'       # MI300X only
#   VLH_SLURM_CONSTRAINT='MARKHAM&(GFX90A|GFX942)'  # MI210 + MI300
#   ROCM_ARCH=gfx1201  # force image build target (else auto-detected on node)
#
# Required: HF_TOKEN or HF_TOKEN_FILE
# Storage (when VLH_NVME_BASE is unset): auto on the compute node —
#   1) mount blank nvme*n*  2) use an existing NVMe mount (/mnt, /local, …)
#   3) /scratch/$USER/vllm-lmcache-hipfile/lmcache-<jobid>  4) /tmp job dir
#   VLH_NVME_MKFS=1 by default when VLH_NVME_BASE is unset (blank nvme*n* only)
#   VLH_NVME_MKFS=0 to skip mkfs; override VLH_NVME_BASE=/path to use existing storage
#
# Benchmark: VLH_BENCHMARK=gutenberg|gutenberg_serial|none|long_doc_qa|test_aic
# Gutenberg (default): run-long-parallel.sh — VLH_RUN_LONG_WORKERS (default 4)
# Gutenberg fixtures (generate once): VLH_GUTENBERG_DATA_ROOT or VLH_SHARED_ROOT/gutenberg
# HF Hub weights: golden cache VLH_HF_HOME (default /scratch/$USER/vllm-lmcache-hipfile/hf)
#
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RECIPE_DIR="${REPO_DIR}/recipies/vllm-lmcache-hipfile"
SLURM_DIR="${REPO_DIR}/.slurm"
SLURM_LIB="${SLURM_DIR}/lib"
JOB_ROOT="/tmp/vllm-lmcache-hipfile-${SLURM_JOB_ID:-local$$}"
REPORT_DIR="${JOB_ROOT}/report"
SLURM_LOG_DIR="${SLURM_DIR}/logs/vllm-lmcache-hipfile-${SLURM_JOB_ID:-local$$}"

IMAGE_NAME="${IMAGE_NAME:-vllm-lmcache-hipfile}"
GPU="${GPU:-0}"
VLLM_PORT="800${GPU}"
CONTAINER_NAME="${CONTAINER_NAME:-${IMAGE_NAME}-gpu${GPU}-${SLURM_JOB_ID:-local$$}}"

# VLH_NVME_BASE set after optional auto-detect (below)
DATA_HOST=""
HF_HOST=""
CONTAINER_LOG_HOST="${REPORT_DIR}/logs"
VLH_LMCACHE_IO="${VLH_LMCACHE_IO:-hipfile}"
VLH_BENCHMARK="${VLH_BENCHMARK:-gutenberg}"

BUILD_RC=1
RUN_RC=1
PHASE_RC=1

# shellcheck source=lib/vllm-lmcache-hipfile-common.sh
source "${SLURM_LIB}/vllm-lmcache-hipfile-common.sh"

_vlh_env_aliases
_vlh_resolve_model_env

cleanup() {
    _vlh_job_cleanup
}
trap cleanup EXIT

cd "${REPO_DIR}"
mkdir -p "${JOB_ROOT}" "${REPORT_DIR}" "${CONTAINER_LOG_HOST}" \
    "${SLURM_LOG_DIR}" "${SLURM_DIR}/logs"

_vlh_log "REPO_DIR=${REPO_DIR} RECIPE_DIR=${RECIPE_DIR} JOB_ROOT=${JOB_ROOT}"

_vlh_detect_rocm_arch
_vlh_resolve_hf_token
_vlh_metadata_init

_vlh_docker_cleanup

# Format blank nvme*n* when auto-discovering storage (not when VLH_NVME_BASE is preset).
if [[ -z "${VLH_NVME_BASE:-}" ]]; then
    : "${VLH_NVME_MKFS:=1}"
    export VLH_NVME_MKFS
fi

if [[ -n "${VLH_NVME_BASE:-}" ]]; then
    _vlh_nvme_setup || true
else
    _vlh_nvme_auto_use || true
    if [[ -z "${VLH_NVME_BASE:-}" ]]; then
        _vlh_nvme_scratch_fallback || true
    fi
fi
VLH_NVME_BASE="${VLH_NVME_BASE:-${JOB_ROOT}}"
export VLH_NVME_BASE
_vlh_metadata_append "VLH_NVME_BASE" "${VLH_NVME_BASE}"
DATA_HOST="${VLH_NVME_BASE}/data"
_vlh_resolve_hf_home
mkdir -p "${CONTAINER_LOG_HOST}" "${DATA_HOST}" \
    "${HF_HOST}/hub" "${HF_HOST}/datasets" "${HF_HOST}/vllm" \
    "${HF_HOST}/vllm_config" "${HF_HOST}/torch" "${HF_HOST}/torch_inductor"

if [[ "${VLH_NVME_BASE}" == "${JOB_ROOT}" ]]; then
    _vlh_log "WARN: using JOB_ROOT for LMCache data (see NVMe inventory above if logged)"
else
    _vlh_log "Using VLH_NVME_BASE=${VLH_NVME_BASE} (LMCache data)"
fi
_vlh_log "Using VLH_HF_HOME=${HF_HOST} (Hub / vLLM model cache)"
_vlh_metadata_append "VLLM_MODEL" "$(_vlh_served_model)"
_vlh_hf_fix_ownership

_vlh_nvme_smart_log "start"
_vlh_bpftrace_nvme_start

if ! _vlh_build_image; then
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
    VLH_LMCACHE_IO="${VLH_LMCACHE_IO}" \
    VLH_LMCACHE_LOG_LEVEL="${VLH_LMCACHE_LOG_LEVEL:-}" \
    VLH_LMCACHE_GDS_BUFFER_SIZE="${VLH_LMCACHE_GDS_BUFFER_SIZE:-}" \
    VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}" \
    TZ="${TZ:-America/Edmonton}" \
    EXTRA_DOCKER_RUN_FLAGS="${EXTRA_DOCKER_RUN_FLAGS:-} -e VLH_LMCACHE_ENABLE_KV_EVENTS=${VLH_LMCACHE_ENABLE_KV_EVENTS:-1} -e VLH_LMCACHE_ENABLE_CHUNK_STATISTICS=${VLH_LMCACHE_ENABLE_CHUNK_STATISTICS:-1}"; then
    RUN_RC=1
    PHASE_RC=1
    exit 1
fi

sleep 2
_vlh_bpftrace_vfs_start

if ! _vlh_wait_vllm; then
    RUN_RC=1
    PHASE_RC=1
    docker logs "${CONTAINER_NAME}" 2>&1 | tail -200 || true
    exit 1
fi

case "${VLH_BENCHMARK,,}" in
    none | skip)
        _vlh_log "VLH_BENCHMARK=none; skipping benchmark"
        RUN_RC=0
        ;;
    gutenberg | run_long | run-long | run_long_parallel | run-long-parallel)
        if _vlh_run_gutenberg; then
            RUN_RC=0
        else
            RUN_RC=1
        fi
        ;;
    gutenberg_serial | run_long_serial)
        if _vlh_run_gutenberg_serial; then
            RUN_RC=0
        else
            RUN_RC=1
        fi
        ;;
    long_doc_qa | long-doc-qa)
        if _vlh_run_long_doc_qa; then
            RUN_RC=0
        else
            RUN_RC=1
        fi
        ;;
    test_aic | test-aic)
        if _vlh_run_test_aic; then
            RUN_RC=0
        else
            RUN_RC=1
        fi
        ;;
    *)
        _vlh_log "ERROR: unknown VLH_BENCHMARK=${VLH_BENCHMARK}"
        RUN_RC=1
        PHASE_RC=1
        exit 1
        ;;
esac

_vlh_collect_lmcache_api

T_RUN_END="$(date -Iseconds)"
_vlh_metadata_append "RUN_RC" "${RUN_RC}"
_vlh_metadata_append "PHASE_RC" "${PHASE_RC}"
_vlh_metadata_append "T_RUN" "${T_RUN_START} .. ${T_RUN_END}"

if [[ "${RUN_RC}" -eq 0 ]]; then
    PHASE_RC=0
else
    PHASE_RC=1
fi

_vlh_write_summary
_vlh_summarize_job
_vlh_write_report
_vlh_collect_artifacts

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

_vlh_log "Done PHASE_RC=${PHASE_RC} RUN_RC=${RUN_RC} BUILD_RC=${BUILD_RC}"
_vlh_log "Report: ${SLURM_LOG_DIR}"

exit "${PHASE_RC}"
