#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Slurm: MI300-class GPU (1), NVMe-capable nodes, vllm-from-kurt build/run/report.
#
# Edit partition/constraints for your site. Slurm does not expand env in #SBATCH;
# override at submit time: sbatch --partition=X --constraint=Y this.sbatch
#
# RUNTIME=docker (default): docker build in recipies/vllm-from-kurt, docker run.
# RUNTIME=pyxis: set CONTAINER_IMAGE to docker://registry/.../image:tag or a .sqsh
# path; no docker build in the job. Uses srun --container-image and
# --container-mounts (NVIDIA Pyxis + Enroot style; flag names may vary by site).
# vllm-kurt image ENTRYPOINT is /bin/bash: docker run uses --entrypoint /bin/bash … -lc;
# Pyxis uses --no-container-entrypoint so /bin/bash -lc is not prefixed twice.
#
#SBATCH --job-name=vllm-kurt-mi300
#SBATCH --gres=gpu:1
#SBATCH --constraint=MARKHAM&GFX942&NVME
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=.slurm/logs/vllm-from-kurt-mi300-%j.log

set -euo pipefail

# ── configuration (env overrides) ─────────────────────────────
RUNTIME="${RUNTIME:-docker}"
DOCKER="${DOCKER:-docker}"
IMAGE_NAME="${IMAGE_NAME:-vllm-kurt}"
REPO_DIR="${SLURM_SUBMIT_DIR}"
RECIPE_DIR="${REPO_DIR}/recipies/vllm-from-kurt"
LOG_DIR="${REPO_DIR}/.slurm/logs"
CONTAINER_HF_HOME="${CONTAINER_HF_HOME:-/hf}"
CONTAINER_DATA_DIR="${CONTAINER_DATA_DIR:-/data}"
BACKEND="${BACKEND:-native}"
ROCM_ARCH="${ROCM_ARCH:-}"
MIN_FREE_KB="${MIN_FREE_KB:-10485760}"
READY_TIMEOUT="${READY_TIMEOUT:-7200}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HOME}/.batesste-hugging-face-read-march-2026.token}"
GPU_IDX="${GPU_IDX:-0}"
BENCH_EXTRA=(--repeat-count 3 --completions --trim-fraction 0.1)
if [[ -n "${BENCH_EXTRA_ARGS:-}" ]]; then
  read -r -a _bench_extra <<< "${BENCH_EXTRA_ARGS}"
  BENCH_EXTRA+=("${_bench_extra[@]}")
fi

mkdir -p "${LOG_DIR}"
PHASE_RC=0
BUILD_RC=""
RUN_RC=""
T_BUILD_START=""
T_BUILD_END=""
T_RUN_START=""
T_RUN_END=""
IMAGE_ID_LINE=""

log() { echo "[$(date -Iseconds)] $*"; }

require_hf_token() {
  if [[ -z "${HF_TOKEN:-}" ]] && [[ -r "${HF_TOKEN_FILE}" ]]; then
    HF_TOKEN="$(tr -d '\r\n' < "${HF_TOKEN_FILE}")"
  fi
  if [[ -z "${HF_TOKEN:-}" ]]; then
    log "ERROR: set HF_TOKEN or a readable HF_TOKEN_FILE (default: ${HF_TOKEN_FILE})"
    exit 1
  fi
  export HF_TOKEN
}

detect_rocm_arch() {
  if [[ -n "${ROCM_ARCH}" ]]; then
    echo "${ROCM_ARCH}"
    return
  fi
  local a
  a="$(rocm_agent_enumerator 2>/dev/null | grep -E '^gfx' | grep -v '^gfx000' | head -1 || true)"
  if [[ -n "${a}" ]]; then
    echo "${a}"
    return
  fi
  echo "gfx942"
}

# Optional: KURT_NVME_MKFS=1 KURT_NVME_DEVICE=/dev/nvme0n1 KURT_NVME_MOUNT=/mnt/kurt-nvme (DANGEROUS on shared systems)
maybe_mkfs_mount_nvme() {
  if [[ "${KURT_NVME_MKFS:-0}" != "1" ]]; then
    return 0
  fi
  if [[ -z "${KURT_NVME_DEVICE:-}" ]] || [[ -z "${KURT_NVME_MOUNT:-}" ]]; then
    log "ERROR: KURT_NVME_MKFS=1 requires KURT_NVME_DEVICE and KURT_NVME_MOUNT"
    exit 1
  fi
  mkdir -p "${KURT_NVME_MOUNT}"
  if ! mountpoint -q "${KURT_NVME_MOUNT}"; then
    if [[ ! -b "${KURT_NVME_DEVICE}" ]]; then
      log "ERROR: KURT_NVME_DEVICE is not a block device: ${KURT_NVME_DEVICE}"
      exit 1
    fi
    if ! blkid "${KURT_NVME_DEVICE}" &>/dev/null; then
      log "WARN: formatting ${KURT_NVME_DEVICE} as xfs (KURT_NVME_MKFS=1)"
      mkfs.xfs -f "${KURT_NVME_DEVICE}"
    fi
    mount "${KURT_NVME_DEVICE}" "${KURT_NVME_MOUNT}"
  fi
  export KURT_NVME_BASE="${KURT_NVME_MOUNT}"
}

free_kb() {
  local path="$1"
  df -Pk "${path}" 2>/dev/null | awk 'NR==2 {print $4}'
}

pick_nvme_base() {
  maybe_mkfs_mount_nvme
  if [[ -n "${KURT_NVME_BASE:-}" ]]; then
    if [[ ! -d "${KURT_NVME_BASE}" ]] || [[ ! -w "${KURT_NVME_BASE}" ]]; then
      log "ERROR: KURT_NVME_BASE not a writable directory: ${KURT_NVME_BASE}"
      exit 1
    fi
    echo "${KURT_NVME_BASE}"
    return
  fi
  local c free
  for c in /mnt/nvme /local /scratch "${SLURM_TMPDIR:-}" "${TMPDIR:-/tmp}" /tmp; do
    [[ -z "${c}" ]] && continue
    [[ -d "${c}" ]] || continue
    [[ -w "${c}" ]] || continue
    free="$(free_kb "${c}" || echo 0)"
    if [[ "${free}" =~ ^[0-9]+$ ]] && [[ "${free}" -ge "${MIN_FREE_KB}" ]]; then
      echo "${c}"
      return
    fi
  done
  log "ERROR: no writable path with >= ${MIN_FREE_KB} KiB free; set KURT_NVME_BASE"
  exit 1
}

prepare_hf_cache_host() {
  local hf="$1"
  mkdir -p \
    "${hf}/hub" \
    "${hf}/datasets" \
    "${hf}/vllm" \
    "${hf}/vllm_config" \
    "${hf}/torch" \
    "${hf}/torch_inductor"
}

write_job_metadata() {
  local out="$1"
  {
    echo "=== job metadata ==="
    echo "hostname: $(hostname)"
    echo "date: $(date -Iseconds)"
    echo "RUNTIME: ${RUNTIME}"
    echo "ROCM_ARCH: ${ROCM_ARCH}"
    echo "IMAGE_NAME: ${IMAGE_NAME}"
    echo "BACKEND: ${BACKEND}"
    echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
    echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-}"
    echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-}"
    echo "JOB_ROOT: ${JOB_ROOT}"
    echo "IMAGE_ID: ${IMAGE_ID_LINE}"
    echo "BUILD_RC: ${BUILD_RC:-}"
    echo "RUN_RC: ${RUN_RC:-}"
    echo "T_BUILD: ${T_BUILD_START:-} .. ${T_BUILD_END:-}"
    echo "T_RUN: ${T_RUN_START:-} .. ${T_RUN_END:-}"
    echo ""
    echo "=== rocm-smi ==="
    if command -v rocm-smi &>/dev/null; then
      rocm-smi --showproductname --showmeminfo vram --showuse 2>&1 || true
    else
      echo "(rocm-smi not in PATH)"
    fi
    echo ""
    echo "=== rocminfo (head) ==="
    if command -v rocminfo &>/dev/null; then
      rocminfo 2>&1 | head -80 || true
    else
      echo "(rocminfo not in PATH)"
    fi
  } >"${out}"
}

append_server_log() {
  local srv="${JOB_ROOT}/data/server.txt"
  local out="$1"
  {
    echo ""
    echo "=== server.txt (full) ==="
    if [[ -f "${srv}" ]]; then
      cat "${srv}"
    else
      echo "(missing ${srv})"
    fi
  } >>"${out}"
}

# Copy LMCache observability artifacts from the data mount into REPORT_DIR so
# report.md and Slurm log bundles reference one tree.
sync_lmcache_report_files() {
  local f
  for f in "${DATA_HOST}"/lmcache_internal_api_*.*; do
    [[ -e "${f}" ]] || continue
    cp -a "${f}" "${REPORT_DIR}/" 2>/dev/null || true
  done
  if [[ -d "${DATA_HOST}/lmcache_chunk_stats" ]]; then
    rm -rf "${REPORT_DIR}/lmcache_chunk_stats" 2>/dev/null || true
    cp -a "${DATA_HOST}/lmcache_chunk_stats" "${REPORT_DIR}/" 2>/dev/null || true
  fi
}

write_report_md() {
  local out="$1"
  local json="${REPORT_DIR}/long_doc_qa.json"
  {
    echo "# vllm-from-kurt job report"
    echo ""
    echo "- Job ID: ${SLURM_JOB_ID:-n/a}"
    echo "- RUNTIME: ${RUNTIME}"
    echo "- BACKEND: ${BACKEND}"
    echo "- ROCM_ARCH: ${ROCM_ARCH}"
    echo "- JOB_ROOT: ${JOB_ROOT}"
    echo "- long_doc_qa JSON: \`${json}\`"
    echo ""
    echo "## LMCache observability"
    echo ""
    if ls "${REPORT_DIR}"/lmcache_internal_api_*.* &>/dev/null; then
      echo "Internal API snapshots (**/chunk_statistics/status** JSON and **/metrics** text):"
      ls -1 "${REPORT_DIR}"/lmcache_internal_api_*.* | while IFS= read -r p; do
        echo "- \`${p}\`"
      done
    else
      echo "(No **lmcache_internal_api_*** files; see **\`${DATA_HOST}/\`** if the run used a different layout.)"
    fi
    if [[ -d "${REPORT_DIR}/lmcache_chunk_stats" ]]; then
      echo ""
      echo "- Chunk hash directory (**file_hash**): \`${REPORT_DIR}/lmcache_chunk_stats/\`"
    fi
    echo ""
    echo "Per-call **store**/**retrieve** detail beyond chunk statistics uses **KURT_LMCACHE_ENABLE_KV_EVENTS** (default **1**) and **KURT_LMCACHE_LOG_LEVEL** (default **INFO** → **LMCACHE_LOG_LEVEL** in **server.txt**). Set **KURT_LMCACHE_ENABLE_KV_EVENTS=0** to disable KV events."
    echo ""
    echo "## Summary text"
    echo ""
    echo '```'
    cat "${REPORT_DIR}/summary.txt"
    echo '```'
    echo ""
    if command -v jq &>/dev/null && [[ -f "${json}" ]]; then
      echo "## long_doc_qa (jq .)"
      echo ""
      echo '```json'
      jq . "${json}" 2>/dev/null || cat "${json}"
      echo '```'
    fi
  } >"${out}"
}

copy_to_submit_logs() {
  local stamp="${LOG_DIR}/vllm-from-kurt-mi300-${SLURM_JOB_ID:-manual}"
  mkdir -p "${stamp}"
  cp -a "${REPORT_DIR}/summary.txt" "${stamp}/" 2>/dev/null || true
  cp -a "${REPORT_DIR}/metadata.txt" "${stamp}/" 2>/dev/null || true
  cp -a "${REPORT_DIR}/report.md" "${stamp}/" 2>/dev/null || true
  cp -a "${REPORT_DIR}/long_doc_qa.json" "${stamp}/" 2>/dev/null || true
  if ls "${REPORT_DIR}"/lmcache_internal_api_*.* &>/dev/null; then
    cp -a "${REPORT_DIR}"/lmcache_internal_api_*.* "${stamp}/" 2>/dev/null || true
  fi
  if [[ -d "${REPORT_DIR}/lmcache_chunk_stats" ]]; then
    cp -a "${REPORT_DIR}/lmcache_chunk_stats" "${stamp}/" 2>/dev/null || true
  fi
  log "Copied report artifacts to ${stamp}/"
}

# ── start ───────────────────────────────────────────────────────
log "RUNTIME=${RUNTIME} RECIPE_DIR=${RECIPE_DIR}"
require_hf_token

if [[ ! -f "${RECIPE_DIR}/Dockerfile" ]]; then
  log "ERROR: missing ${RECIPE_DIR}/Dockerfile (submit from repo root?)"
  exit 1
fi

ROCM_ARCH="$(detect_rocm_arch)"
log "Using ROCM_ARCH=${ROCM_ARCH}"

KURT_NVME_BASE="$(pick_nvme_base)"
log "Using KURT_NVME_BASE=${KURT_NVME_BASE}"
JOB_ROOT="${KURT_NVME_BASE}/vllm-kurt-${SLURM_JOB_ID:-0}"
DATA_HOST="${JOB_ROOT}/data"
HF_HOST="${JOB_ROOT}/hf"
REPORT_DIR="${JOB_ROOT}/report"
mkdir -p "${DATA_HOST}" "${REPORT_DIR}"
prepare_hf_cache_host "${HF_HOST}"

export ROCR_VISIBLE_DEVICES="${GPU_IDX}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# long_doc_qa (LMCache) default prompts + --output-len must fit under vLLM
# --max-model-len: the server rejects when prompt_tokens + max_new_tokens exceed
# the context cap. Job 302650 hit: ~16285 input + 100 output => need >= 16385
# (16384 failed by one token). Extra slack covers template overhead and longer
# rounds; raise further if you pass larger --document-length / --num-documents
# via BENCH_EXTRA_ARGS (README suggests e.g. 131072 for MI300 stress).
KURT_MAX_MODEL_LEN="${KURT_MAX_MODEL_LEN:-32768}"
KURT_MAX_NUM_BATCHED_TOKENS="${KURT_MAX_NUM_BATCHED_TOKENS:-8192}"

BOOTCFG="/boot/config-$(uname -r)"
BOOT_MOUNT=()
if [[ -r "${BOOTCFG}" ]]; then
  BOOT_MOUNT=( -v "${BOOTCFG}:${BOOTCFG}:ro" )
fi

# Command run inside the container (host-expanded args, safely quoted for bash -lc).
BENCH_CMD=(
  python3 /app/run_long_doc_qa.py
  --backend "${BACKEND}"
  --ready-timeout "${READY_TIMEOUT}"
  --json-output
)
BENCH_CMD+=("${BENCH_EXTRA[@]}")
BENCH_CMD_QUOTED="$(printf '%q ' "${BENCH_CMD[@]}")"

export CONTAINER_HF_HOME CONTAINER_DATA_DIR
export KURT_CONTAINER_DATA_DIR="${CONTAINER_DATA_DIR}"

export_for_container() {
  export HF_HOME="${CONTAINER_HF_HOME}"
  export HUGGINGFACE_HUB_CACHE="${CONTAINER_HF_HOME}/hub"
  export HF_HUB_CACHE="${CONTAINER_HF_HOME}/hub"
  export HF_DATASETS_CACHE="${CONTAINER_HF_HOME}/datasets"
  export VLLM_CACHE_ROOT="${CONTAINER_HF_HOME}/vllm"
  export VLLM_CONFIG_ROOT="${CONTAINER_HF_HOME}/vllm_config"
  export TORCH_HOME="${CONTAINER_HF_HOME}/torch"
  export TORCHINDUCTOR_CACHE_DIR="${CONTAINER_HF_HOME}/torch_inductor"
  export KURT_CONTAINER_DATA_DIR="${CONTAINER_DATA_DIR}"
  export ROCR_VISIBLE_DEVICES="${GPU_IDX}"
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
  export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
  export KURT_LMCACHE_ENABLE_CHUNK_STATISTICS="${KURT_LMCACHE_ENABLE_CHUNK_STATISTICS:-1}"
  export KURT_LMCACHE_CHUNK_STATISTICS_STRATEGY="${KURT_LMCACHE_CHUNK_STATISTICS_STRATEGY:-file_hash}"
  export KURT_LMCACHE_COLLECT_INTERNAL_API="${KURT_LMCACHE_COLLECT_INTERNAL_API:-1}"
  export KURT_LMCACHE_INTERNAL_API_MAX_OFFSET="${KURT_LMCACHE_INTERNAL_API_MAX_OFFSET:-3}"
  export KURT_LMCACHE_ENABLE_KV_EVENTS="${KURT_LMCACHE_ENABLE_KV_EVENTS:-1}"
  export KURT_LMCACHE_LOG_LEVEL="${KURT_LMCACHE_LOG_LEVEL:-INFO}"
  export KURT_LMCACHE_INTERNAL_API_VERBOSE="${KURT_LMCACHE_INTERNAL_API_VERBOSE:-0}"
  [[ -n "${VLLM_MODEL:-}" ]] && export VLLM_MODEL
  [[ -n "${KURT_MAX_MODEL_LEN:-}" ]] && export KURT_MAX_MODEL_LEN
  [[ -n "${KURT_MAX_NUM_BATCHED_TOKENS:-}" ]] && export KURT_MAX_NUM_BATCHED_TOKENS
  [[ -n "${KURT_ENFORCE_EAGER:-}" ]] && export KURT_ENFORCE_EAGER
  [[ -n "${KURT_GPU_MEMORY_UTILIZATION:-}" ]] && export KURT_GPU_MEMORY_UTILIZATION
}

run_docker() {
  if ! command -v "${DOCKER}" &>/dev/null; then
    log "ERROR: ${DOCKER} not found (set DOCKER=podman or install Docker)"
    exit 1
  fi
  T_BUILD_START="$(date -Iseconds)"
  (
    cd "${RECIPE_DIR}"
    "${DOCKER}" build -t "${IMAGE_NAME}" --build-arg "ROCM_ARCH=${ROCM_ARCH}" .
  )
  BUILD_RC=$?
  T_BUILD_END="$(date -Iseconds)"
  if [[ "${BUILD_RC}" -ne 0 ]]; then
    log "ERROR: docker build failed rc=${BUILD_RC}"
    return "${BUILD_RC}"
  fi
  if IMAGE_ID_LINE="$("${DOCKER}" inspect --format='{{.Id}}' "${IMAGE_NAME}" 2>/dev/null)"; then
    true
  else
    IMAGE_ID_LINE="(inspect failed)"
  fi

  T_RUN_START="$(date -Iseconds)"
  set +e
  "${DOCKER}" run --rm \
    --name="gpu${GPU_IDX}" \
    --hostname="gpu${GPU_IDX}" \
    --ipc=host \
    --network=host \
    --cap-add=CAP_SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --env HF_HOME="${CONTAINER_HF_HOME}" \
    --env HUGGINGFACE_HUB_CACHE="${CONTAINER_HF_HOME}/hub" \
    --env HF_HUB_CACHE="${CONTAINER_HF_HOME}/hub" \
    --env HF_DATASETS_CACHE="${CONTAINER_HF_HOME}/datasets" \
    --env VLLM_CACHE_ROOT="${CONTAINER_HF_HOME}/vllm" \
    --env VLLM_CONFIG_ROOT="${CONTAINER_HF_HOME}/vllm_config" \
    --env TORCH_HOME="${CONTAINER_HF_HOME}/torch" \
    --env TORCHINDUCTOR_CACHE_DIR="${CONTAINER_HF_HOME}/torch_inductor" \
    --env KURT_CONTAINER_DATA_DIR="${CONTAINER_DATA_DIR}" \
    --env HF_TOKEN="${HF_TOKEN}" \
    --env ROCR_VISIBLE_DEVICES="${GPU_IDX}" \
    --env PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
    --env VLLM_MODEL="${VLLM_MODEL:-}" \
    --env KURT_MAX_MODEL_LEN="${KURT_MAX_MODEL_LEN:-}" \
    --env KURT_MAX_NUM_BATCHED_TOKENS="${KURT_MAX_NUM_BATCHED_TOKENS:-}" \
    --env KURT_ENFORCE_EAGER="${KURT_ENFORCE_EAGER:-}" \
    --env KURT_GPU_MEMORY_UTILIZATION="${KURT_GPU_MEMORY_UTILIZATION:-}" \
    --env PYTHONHASHSEED=0 \
    --env KURT_LMCACHE_ENABLE_CHUNK_STATISTICS="${KURT_LMCACHE_ENABLE_CHUNK_STATISTICS:-1}" \
    --env KURT_LMCACHE_CHUNK_STATISTICS_STRATEGY="${KURT_LMCACHE_CHUNK_STATISTICS_STRATEGY:-file_hash}" \
    --env KURT_LMCACHE_COLLECT_INTERNAL_API="${KURT_LMCACHE_COLLECT_INTERNAL_API:-1}" \
    --env KURT_LMCACHE_INTERNAL_API_MAX_OFFSET="${KURT_LMCACHE_INTERNAL_API_MAX_OFFSET:-3}" \
    --env KURT_LMCACHE_ENABLE_KV_EVENTS="${KURT_LMCACHE_ENABLE_KV_EVENTS:-1}" \
    --env KURT_LMCACHE_LOG_LEVEL="${KURT_LMCACHE_LOG_LEVEL:-INFO}" \
    --env KURT_LMCACHE_INTERNAL_API_VERBOSE="${KURT_LMCACHE_INTERNAL_API_VERBOSE:-0}" \
    -v "${HF_HOST}:${CONTAINER_HF_HOME}" \
    -v "${DATA_HOST}:${CONTAINER_DATA_DIR}" \
    "${BOOT_MOUNT[@]}" \
    --entrypoint /bin/bash \
    "${IMAGE_NAME}" \
    -lc "${BENCH_CMD_QUOTED}" \
    | tee "${REPORT_DIR}/long_doc_qa.json"
  RUN_RC=${PIPESTATUS[0]}
  set -e
  T_RUN_END="$(date -Iseconds)"
  return "${RUN_RC}"
}

run_pyxis() {
  local img="${CONTAINER_IMAGE:-}"
  if [[ -z "${img}" ]]; then
    log "ERROR: RUNTIME=pyxis requires CONTAINER_IMAGE (docker://registry/.../vllm-kurt:tag or /path/image.sqsh)"
    exit 1
  fi
  BUILD_RC="skipped"
  T_BUILD_START="$(date -Iseconds)"
  T_BUILD_END="${T_BUILD_START}"
  IMAGE_ID_LINE="${img}"

  local mounts="${HF_HOST}:${CONTAINER_HF_HOME}:rw,${DATA_HOST}:${CONTAINER_DATA_DIR}:rw"
  if [[ -r "${BOOTCFG}" ]]; then
    mounts="${mounts},${BOOTCFG}:${BOOTCFG}:ro"
  fi

  export_for_container
  T_RUN_START="$(date -Iseconds)"
  set +e
  srun --ntasks=1 --export=ALL --container-image="${img}" \
    --container-mounts="${mounts}" \
    --container-writable \
    --no-container-entrypoint \
    /bin/bash -lc "${BENCH_CMD_QUOTED}" \
    | tee "${REPORT_DIR}/long_doc_qa.json"
  RUN_RC=${PIPESTATUS[0]}
  set -e
  T_RUN_END="$(date -Iseconds)"
  return "${RUN_RC}"
}

case "${RUNTIME}" in
  docker) run_docker ;;
  pyxis) run_pyxis ;;
  *)
    log "ERROR: RUNTIME must be docker or pyxis (got ${RUNTIME})"
    exit 1
    ;;
esac
PHASE_RC=$?
RUN_RC="${RUN_RC:-${PHASE_RC}}"

SUMMARY="${REPORT_DIR}/summary.txt"
{
  echo "RUNTIME=${RUNTIME} PHASE_RC=${PHASE_RC} RUN_RC=${RUN_RC} BUILD_RC=${BUILD_RC:-}"
  echo "JOB_ROOT=${JOB_ROOT}"
} >"${SUMMARY}"

META="${REPORT_DIR}/metadata.txt"
sync_lmcache_report_files
write_job_metadata "${META}"
append_server_log "${META}"
write_report_md "${REPORT_DIR}/report.md"
copy_to_submit_logs

if [[ "${PHASE_RC}" -ne 0 ]]; then
  log "ERROR: workload failed rc=${PHASE_RC}"
  exit "${PHASE_RC}"
fi
log "Done. Reports under ${REPORT_DIR} and ${LOG_DIR}/vllm-from-kurt-mi300-${SLURM_JOB_ID}/"
exit 0
