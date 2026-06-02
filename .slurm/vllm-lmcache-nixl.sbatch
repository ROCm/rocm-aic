#!/bin/bash
#SBATCH --job-name=vllm-lmcache-nixl
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=.slurm/logs/vllm-lmcache-nixl-%j.log
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Build and run recipies/vllm-lmcache-nixl on a ROCm GPU node (Docker).
#
# Submit from repo root:
#   bash .slurm/run-vllm-lmcache-nixl.sh
#   sbatch --export=ALL,VLN_NVME_BASE=/mnt/nvme .slurm/vllm-lmcache-nixl.sbatch
#
# GPU node selection (optional, via run-vllm-lmcache-nixl.sh or sbatch CLI):
#   VLN_SLURM_CONSTRAINT=MARKHAM              # any Markham ROCm GPU
#   VLN_SLURM_CONSTRAINT='MARKHAM&GFX942'       # MI300X only
#   VLN_SLURM_CONSTRAINT='MARKHAM&(GFX90A|GFX942)'  # MI210 + MI300
#   ROCM_ARCH=gfx1201  # force image build target (else auto-detected on node)
#
# Required: HF_TOKEN or HF_TOKEN_FILE
# Storage (when VLN_NVME_BASE is unset): auto on the compute node —
#   1) mount blank nvme*n*  2) use an existing NVMe mount (/mnt, /local, …)
#   3) /scratch/$USER/vllm-lmcache-nixl/lmcache-<jobid>  4) /tmp job dir
#   VLN_NVME_MKFS=1 by default when VLN_NVME_BASE is unset (blank nvme*n* only)
#   VLN_NVME_MKFS=0 to skip mkfs; override VLN_NVME_BASE=/path to use existing storage
#
# Benchmark: VLN_BENCHMARK=gutenberg|gutenberg_serial|none|long_doc_qa|test_aic
# Gutenberg (default): run-long-parallel.sh — VLN_RUN_LONG_WORKERS (default 4)
# Gutenberg fixtures (generate once): VLN_GUTENBERG_DATA_ROOT or VLN_SHARED_ROOT/gutenberg
# HF Hub weights: golden cache VLN_HF_HOME (default /scratch/$USER/vllm-lmcache-nixl/hf)
#
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RECIPE_DIR="${REPO_DIR}/recipies/vllm-lmcache-nixl"
SLURM_DIR="${REPO_DIR}/.slurm"
SLURM_LIB="${SLURM_DIR}/lib"
JOB_ROOT="/tmp/vllm-lmcache-nixl-${SLURM_JOB_ID:-local$$}"
REPORT_DIR="${JOB_ROOT}/report"
SLURM_LOG_DIR="${SLURM_DIR}/logs/vllm-lmcache-nixl-${SLURM_JOB_ID:-local$$}"

IMAGE_NAME="${IMAGE_NAME:-vllm-lmcache-nixl}"
GPU="${GPU:-0}"
VLLM_PORT="800${GPU}"
CONTAINER_NAME="${CONTAINER_NAME:-${IMAGE_NAME}-gpu${GPU}-${SLURM_JOB_ID:-local$$}}"

# VLN_NVME_BASE set after optional auto-detect (below)
DATA_HOST=""
HF_HOST=""
CONTAINER_LOG_HOST="${REPORT_DIR}/logs"
CONTAINER_LOG_SUBDIR="vllm-lmcache-nixl"
export CONTAINER_LOG_SUBDIR
VLN_LMCACHE_IO="${VLN_LMCACHE_IO:-nixl-posix}"
VLN_BENCHMARK="${VLN_BENCHMARK:-gutenberg}"

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

# Format blank nvme*n* when auto-discovering storage (not when VLN_NVME_BASE is preset).
if [[ -z "${VLN_NVME_BASE:-}" ]]; then
    : "${VLN_NVME_MKFS:=1}"
    export VLN_NVME_MKFS
fi

if [[ -n "${VLN_NVME_BASE:-}" ]]; then
    _vlh_nvme_setup || true
else
    _vlh_nvme_auto_use || true
    if [[ -z "${VLN_NVME_BASE:-}" ]]; then
        _vlh_nvme_scratch_fallback || true
    fi
fi
VLN_NVME_BASE="${VLN_NVME_BASE:-${JOB_ROOT}}"
export VLN_NVME_BASE
_vlh_metadata_append "VLN_NVME_BASE" "${VLN_NVME_BASE}"
DATA_HOST="${VLN_NVME_BASE}/data"
_vlh_resolve_hf_home
mkdir -p "${CONTAINER_LOG_HOST}" "${DATA_HOST}" \
    "${HF_HOST}/hub" "${HF_HOST}/datasets" "${HF_HOST}/vllm" \
    "${HF_HOST}/vllm_config" "${HF_HOST}/torch" "${HF_HOST}/torch_inductor"

if [[ "${VLN_NVME_BASE}" == "${JOB_ROOT}" ]]; then
    _vlh_log "WARN: using JOB_ROOT for LMCache data (see NVMe inventory above if logged)"
else
    _vlh_log "Using VLN_NVME_BASE=${VLN_NVME_BASE} (LMCache data)"
fi
_vlh_log "Using VLN_HF_HOME=${HF_HOST} (Hub / vLLM model cache)"
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
    VLN_LMCACHE_IO="${VLN_LMCACHE_IO}" \
    VLN_LMCACHE_LOG_LEVEL="${VLN_LMCACHE_LOG_LEVEL:-}" \
    VLN_NIXL_BUFFER_SIZE="${VLN_NIXL_BUFFER_SIZE:-}" \
    VLN_NIXL_BUFFER_DEVICE="${VLN_NIXL_BUFFER_DEVICE:-}" \
    VLN_NIXL_POOL_SIZE="${VLN_NIXL_POOL_SIZE:-}" \
    VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}" \
    TZ="${TZ:-America/Edmonton}" \
    EXTRA_DOCKER_RUN_FLAGS="${EXTRA_DOCKER_RUN_FLAGS:-} -e VLN_LMCACHE_ENABLE_KV_EVENTS=${VLN_LMCACHE_ENABLE_KV_EVENTS:-1} -e VLN_LMCACHE_ENABLE_CHUNK_STATISTICS=${VLN_LMCACHE_ENABLE_CHUNK_STATISTICS:-1}"; then
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

case "${VLN_BENCHMARK,,}" in
    none | skip)
        _vlh_log "VLN_BENCHMARK=none; skipping benchmark"
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
        _vlh_log "ERROR: unknown VLN_BENCHMARK=${VLN_BENCHMARK}"
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
