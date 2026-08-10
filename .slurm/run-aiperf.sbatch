#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
#SBATCH --job-name=aic-aiperf
#SBATCH --partition=defq
# Same node class as the cliff sweep (gfx942 + local NVMe), but aiperf
# exercises the serving path (TTFT, ITL, throughput) rather than KV-cache
# eviction, so any GPU node with the AIC image works.  Override via
#   make aiperf-submit AIC_CLIFF_GFX=gfx950
# or the full AIC_CLIFF_CONSTRAINT=... knob in the Makefile.
#SBATCH --constraint=GFX942
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#
# Run the aiperf generative-AI benchmark (benchmarks/run_aiperf.py) end-to-end
# against two stack configurations on ONE GPU node:
#
#   1. vram_only   plain vLLM, VRAM prefix cache only (baseline)
#   2. kvd_v2      LMCache MP: DRAM L1 + NIXL AIS_MT -> local NVMe L2
#
# Each arm uses the SAME docker/docker-compose.yml as the cliff sweep; the
# benchmark client (aiperf + run_aiperf.py) runs on the HOST rather than
# inside a container, since aiperf only needs HTTP access to localhost:8000.
#
# aiperf is installed into the repo .venv (or the system Python if the .venv
# is absent) at job startup; no separate Docker image layer is required.
#
# Submit from the repo root:
#   make aiperf-submit
#   sbatch .slurm/run-aiperf.sbatch
#   # pin a node:
#   sbatch --nodelist=<node> .slurm/run-aiperf.sbatch
#   # override the sweep:
#   AIPERF_CONCUR=1,4,16,64 VLLM_MODEL=openai/gpt-oss-120b \
#       sbatch .slurm/run-aiperf.sbatch
#
set -uo pipefail

# --- Repo path (same logic as run-cliff.sbatch) ------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# Per-job log dir with full output redirect
LOGDIR_REL="logs/${SLURM_JOB_ID:-manual}"
LOGDIR="${REPO}/${LOGDIR_REL}"
if mkdir -p "${LOGDIR}" 2>/dev/null; then
    exec >>"${LOGDIR}/aiperf.out" 2>&1
fi

# --- Config ------------------------------------------------------------------
AIC_IMAGE="${AIC_IMAGE:-rocm-aic:latest}"
AIC_IMAGE_DIR="${AIC_IMAGE_DIR:-/scratch/${USER}/images}"
VLLM_MODEL="${VLLM_MODEL:-}"
AIC_MODEL_BIG="${AIC_MODEL_BIG:-openai/gpt-oss-120b}"
AIC_MODEL_SMALL="${AIC_MODEL_SMALL:-Qwen/Qwen2.5-3B-Instruct}"
HF_HOME="${HF_HOME:-/scratch/${USER}/vllm-lmcache-hipfile/hf}"

# aiperf sweep knobs
AIPERF_CONCUR="${AIPERF_CONCUR:-1,2,4,8,16,32}"
AIPERF_REQUEST_COUNT="${AIPERF_REQUEST_COUNT:-200}"
AIPERF_DATASET="${AIPERF_DATASET:-synthetic}"  # 'synthetic' or 'sharegpt'
AIPERF_ISL="${AIPERF_ISL:-550}"               # synthetic input tokens
AIPERF_OSL="${AIPERF_OSL:-200}"               # synthetic output tokens
AIPERF_TOKENIZER="${AIPERF_TOKENIZER:-}"      # HF tokenizer ID (empty = server count)
AIPERF_ARMS="${AIPERF_ARMS:-vram,nvme}"       # comma list of vram,nvme

# vLLM / LMCache knobs (same defaults as run-cliff.sbatch)
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLM_GPU_MEMORY_UTILIZATION="${VLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLM_MAX_MODEL_LEN="${VLM_MAX_MODEL_LEN:-32768}"
VLM_MAX_NUM_BATCHED_TOKENS="${VLM_MAX_NUM_BATCHED_TOKENS:-4096}"
LMCACHE_PORT=6555
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-20}"
LMCACHE_NVME_POOL="${LMCACHE_NVME_POOL:-4096}"
LMCACHE_NVME_SLOT_SIZE="${LMCACHE_NVME_SLOT_SIZE:-268435456}"
AIC_NIXL_BACKEND="${AIC_NIXL_BACKEND:-AIS_MT}"
AIC_ATTENTION_BACKEND="${AIC_ATTENTION_BACKEND:-TRITON_ATTN}"
AIC_KV_CACHE_DTYPE="${AIC_KV_CACHE_DTYPE:-fp8}"
AIC_ROCM_USE_AITER="${AIC_ROCM_USE_AITER:-1}"
AIC_VLLM_READY_TIMEOUT="${AIC_VLLM_READY_TIMEOUT:-1200}"
VLLM_ENDPOINT_PORT=8000

# NVMe storage (auto-detect from run-cliff.sbatch defaults)
AIC_NVME_AUTO="${AIC_NVME_AUTO:-1}"
AIC_NVME_MOUNT="${AIC_NVME_MOUNT:-/mnt/aic-lmcache}"
AIC_NVME_LABEL="${AIC_NVME_LABEL:-aic-lmcache}"
_AIC_LOCAL_ROOT_PINNED="${AIC_LOCAL_ROOT:+1}"
_AIC_NVME_DATA_PINNED="${NVME_DATA:+1}"

# Monitoring
AIC_MONITORING="${AIC_MONITORING:-1}"
AIC_METRICS_DIR="${AIC_METRICS_DIR:-${REPO}/logs/${SLURM_JOB_ID:-manual}/prometheus}"
AIC_EXPORTERS="${AIC_EXPORTERS:-1}"
AIC_HSA_SNOOP_PID_MODE="${AIC_HSA_SNOOP_PID_MODE:-host}"
MON_COMPOSE="${REPO}/monitoring/docker-compose.monitoring.yml"
MON_DIR="${REPO}/monitoring"

STAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS="${LOGDIR}/results"
PLOTS="${LOGDIR}/plots/aiperf"

log() { printf '[aiperf-job] %s\n' "$*" >&2; }
die() { printf '[aiperf-job] ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Reuse storage helpers from run-cliff.sbatch (same logic, copied inline to
# keep this script self-contained without an explicit source dependency).
# ---------------------------------------------------------------------------
_root_disk() {
    local src; src="$(findmnt -no SOURCE / 2>/dev/null)"
    [[ -n "${src}" ]] || return 0
    local pk; pk="$(lsblk -no PKNAME "${src}" 2>/dev/null | head -1)"
    printf '%s\n' "${pk:-$(basename "${src}")}"
}

_detect_or_provision_nvme() {
    command -v lsblk   >/dev/null 2>&1 || { log "  nvme-auto: lsblk missing; using /tmp"; return 0; }
    command -v findmnt >/dev/null 2>&1 || { log "  nvme-auto: findmnt missing; using /tmp"; return 0; }
    local rootdev; rootdev="$(_root_disk)"
    log "  nvme-auto: root disk = ${rootdev:-unknown}"
    if mountpoint -q "${AIC_NVME_MOUNT}" 2>/dev/null; then
        log "  nvme-auto: reusing already-mounted ${AIC_NVME_MOUNT}"
        printf '%s\n' "${AIC_NVME_MOUNT}"; return 0
    fi
    local name fstype mnt cand="" szb
    while read -r name fstype mnt; do
        [[ "${name}" =~ ^nvme[0-9]+n[0-9]+$ ]] || continue
        [[ "${name}" == "${rootdev}" ]] && continue
        [[ -n "${fstype}" ]] && continue
        [[ -n "${mnt}" ]] && continue
        [[ "$(lsblk -rno NAME "/dev/${name}" 2>/dev/null | wc -l)" -eq 1 ]] || continue
        grep -q "^/dev/${name}[[:space:]]" /proc/swaps 2>/dev/null && continue
        szb="$(lsblk -brndo SIZE "/dev/${name}" 2>/dev/null | head -1)"
        [[ -n "${szb}" && "${szb}" -ge 100000000000 ]] || continue
        cand="/dev/${name}"; break
    done < <(lsblk -rndo NAME,FSTYPE,MOUNTPOINT 2>/dev/null)
    if [[ -n "${cand}" ]]; then
        if sudo -n mkfs.ext4 -q -F -L "${AIC_NVME_LABEL}" "${cand}" >/dev/null 2>&1 \
           && sudo -n mkdir -p "${AIC_NVME_MOUNT}" >/dev/null 2>&1 \
           && sudo -n mount -o noatime "${cand}" "${AIC_NVME_MOUNT}" >/dev/null 2>&1; then
            sudo -n chown "$(id -u):$(id -g)" "${AIC_NVME_MOUNT}" >/dev/null 2>&1 \
                || sudo -n chmod 1777 "${AIC_NVME_MOUNT}" >/dev/null 2>&1 || true
            printf '%s\n' "${AIC_NVME_MOUNT}"; return 0
        fi
    fi
    log "  nvme-auto: no dedicated NVMe found; falling back to /tmp"
}

provision_storage() {
    if [[ -n "${_AIC_LOCAL_ROOT_PINNED}" || -n "${_AIC_NVME_DATA_PINNED}" ]]; then
        log "storage: honoring caller-pinned paths"
    elif [[ "${AIC_NVME_AUTO}" == "1" ]]; then
        local base; base="$(_detect_or_provision_nvme)"
        [[ -n "${base}" ]] && AIC_LOCAL_ROOT="${base}/aic-aiperf.${SLURM_JOB_ID:-manual}"
    fi
    AIC_LOCAL_ROOT="${AIC_LOCAL_ROOT:-/tmp/aic-aiperf.${SLURM_JOB_ID:-manual}}"
    NVME_DATA="${NVME_DATA:-${AIC_LOCAL_ROOT}/nvme}"
    NFS_DATA="${NFS_DATA:-${AIC_LOCAL_ROOT}/nfs}"
    GDS_SLAB_DATA="${GDS_SLAB_DATA:-${AIC_LOCAL_ROOT}/slab}"
}

detect_gpu_arch() {
    local a
    a="$(rocm_agent_enumerator 2>/dev/null | grep -m1 -E '^gfx[0-9a-f]+')" \
        && [[ -n "${a}" ]] && { printf '%s' "${a}"; return; }
    a="$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+')" \
        && [[ -n "${a}" ]] && { printf '%s' "${a}"; return; }
}

select_default_model() {
    [[ -n "${VLLM_MODEL}" ]] && { log "model       : ${VLLM_MODEL} (pinned)"; return; }
    local arch; arch="$(detect_gpu_arch)"
    case "${arch}" in
        gfx942|gfx950)
            VLLM_MODEL="${AIC_MODEL_BIG}"
            log "model       : auto (${arch}, big-VRAM CDNA) -> ${VLLM_MODEL}" ;;
        *)
            VLLM_MODEL="${AIC_MODEL_SMALL}"
            log "model       : auto (${arch:-unknown}) -> ${VLLM_MODEL}" ;;
    esac
}

# ---------------------------------------------------------------------------
# aiperf install / Python selection
# ---------------------------------------------------------------------------
AIPERF_PYTHON=""
ensure_aiperf() {
    # Prefer the repo .venv if it exists (has bench deps).
    local python="${REPO}/.venv/bin/python3"
    [[ -x "${python}" ]] || python="$(command -v python3 || true)"
    [[ -n "${python}" ]] || die "python3 not found on $(hostname)"
    if "${python}" -c "import aiperf" 2>/dev/null; then
        log "aiperf: already available in ${python}"
    else
        log "aiperf: not found; installing into $(dirname "${python}")..."
        "${python}" -m pip install -q "aiperf>=0.12" \
            || die "aiperf pip install failed (network access required for first install)"
        log "aiperf: installed"
    fi
    AIPERF_PYTHON="${python}"
}

# ---------------------------------------------------------------------------
# Compose wrappers (same as run-cliff.sbatch)
# ---------------------------------------------------------------------------
COMPOSE_FILE="${REPO}/docker/docker-compose.yml"
compose() { docker compose -f "${COMPOSE_FILE}" "$@"; }

wait_ready() {  # $1=port $2=timeout_s
    local port="$1" timeout="${2:-${AIC_VLLM_READY_TIMEOUT}}" waited=0
    log "waiting for vLLM on :${port} (up to ${timeout}s)"
    while (( waited < timeout )); do
        if curl -fsS "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            log "vLLM :${port} is ready (${waited}s)"; return 0
        fi
        sleep 10; waited=$((waited + 10))
    done
    return 1
}

rm_containers() { docker rm -f "$@" >/dev/null 2>&1 || true; }

stop_stack() {
    for svc in vllm lmcache; do
        timeout 30 compose logs --no-color --no-log-prefix "${svc}" \
            > "${LOGDIR}/container-aic-${svc}.log" 2>&1 || true
    done
    pkill -9 -f 'vllm.entrypoints.openai' 2>/dev/null || true
    pkill -9 -f 'EngineCore'              2>/dev/null || true
    pkill -9 -f 'lmcache server'          2>/dev/null || true
    sleep 2
    timeout 60 compose --profile cache down --remove-orphans --timeout 5 >/dev/null 2>&1 || true
    for c in aic-vllm-gpu0 aic-lmcache; do timeout 30 docker rm -f "${c}" >/dev/null 2>&1 || true; done
}

# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

# Run the aiperf concurrency sweep against a live endpoint.
# $1=arm  $2=out_csv
run_aiperf_sweep() {
    local arm="$1" out="$2"
    log "running aiperf sweep: arm=${arm} -> ${out}"
    local cmd=(
        "${AIPERF_PYTHON}" "${REPO}/benchmarks/run_aiperf.py"
        --endpoint "http://localhost:${VLLM_ENDPOINT_PORT}"
        --model "${VLLM_MODEL}"
        --arm "${arm}"
        --concurrencies "${AIPERF_CONCUR}"
        --request-count "${AIPERF_REQUEST_COUNT}"
        --dataset "${AIPERF_DATASET}"
        --isl "${AIPERF_ISL}"
        --osl "${AIPERF_OSL}"
        --out "${out}"
        --artifact-base "${RESULTS}/aiperf-artifacts"
    )
    [[ -n "${AIPERF_TOKENIZER}" ]] && cmd+=(--tokenizer "${AIPERF_TOKENIZER}")
    "${cmd[@]}"
}

# Plot all CSVs collected so far.
run_plot() {
    local -a csvs=()
    local csv
    for csv in "${RESULTS}"/aiperf-*.csv; do
        [[ -s "${csv}" ]] && csvs+=("${RESULTS}/$(basename "${csv}")")
    done
    (( ${#csvs[@]} )) || { log "plot: no aiperf CSVs, skipping"; return; }
    mkdir -p "${PLOTS}"
    "${AIPERF_PYTHON}" "${REPO}/benchmarks/plot_aiperf.py" \
        --input "${csvs[@]}" \
        --output-dir "${PLOTS}" \
        && log "aiperf plots -> ${PLOTS}/" || log "plot: plot_aiperf.py failed"
}

# =============================================================================
# Arm A — vram_only (plain vLLM, no LMCache connector)
# =============================================================================
VRAM_CSV="${RESULTS}/aiperf-vram_only-${STAMP}.csv"
run_arm_vram() {
    rm_containers aic-vllm-gpu0 aic-lmcache
    log "=== Arm A (vram_only): plain vLLM on :${VLLM_ENDPOINT_PORT} ==="
    export KV_TRANSFER_ARG=""
    export GDS_MODE=""
    export VLLM_PID_MODE=""
    if ! compose up -d vllm; then
        log "Arm A: compose up vllm failed"; stop_stack; return 1
    fi
    if ! wait_ready "${VLLM_ENDPOINT_PORT}"; then
        log "Arm A: server never became ready; last logs:"
        compose logs --tail 40 --no-color vllm 2>&1 | sed 's/^/    /'
        stop_stack; return 1
    fi
    run_aiperf_sweep vram_only "${VRAM_CSV}"
    local rc=$?
    stop_stack
    return $rc
}

# =============================================================================
# Arm B — kvd_v2/nvme (LMCache MP: DRAM L1 + NIXL NVMe L2)
# =============================================================================
KVD_CSV="${RESULTS}/aiperf-kvd_v2-${STAMP}.csv"
run_arm_kvd() {
    rm_containers aic-vllm-gpu0 aic-lmcache
    log "=== Arm B (kvd_v2, nvme) — LMCache MP connector on :${VLLM_ENDPOINT_PORT} ==="
    local _hb="${LMCACHE_HEARTBEAT_INTERVAL:-900}"
    export KV_TRANSFER_ARG="--kv-transfer-config '{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://localhost\",\"lmcache.mp.port\":${LMCACHE_PORT},\"lmcache.mp.heartbeat_interval\":${_hb}}}'"
    export GDS_MODE=""
    export AIC_L2_BACKEND="${AIC_L2_BACKEND:-nixl}"
    export LMCACHE_L1_SIZE_GB
    export VLLM_PID_MODE=service:lmcache
    unset LMCACHE_CONFIG_FILE LMCACHE_CONFIG_FILE_HOST
    if ! compose --profile cache up -d; then
        log "kvd/nvme: compose up failed"; stop_stack; return 1
    fi
    if ! wait_ready "${VLLM_ENDPOINT_PORT}"; then
        log "kvd/nvme: server never became ready; last logs:"
        compose logs --tail 40 --no-color lmcache 2>&1 | sed 's/^/  [lmcache] /'
        compose logs --tail 60 --no-color vllm    2>&1 | sed 's/^/  [vllm]    /'
        stop_stack; return 1
    fi
    run_aiperf_sweep kvd_v2 "${KVD_CSV}"
    local rc=$?
    stop_stack
    return $rc
}

# =============================================================================
# Main
# =============================================================================
log "host        : $(hostname)  job=${SLURM_JOB_ID:-manual}"
log "image       : ${AIC_IMAGE}"
select_default_model
log "HF_HOME     : ${HF_HOME}"
provision_storage
log "NVMe        : ${NVME_DATA}"
log "results     : ${RESULTS}"
command -v docker >/dev/null 2>&1 || die "docker not found on $(hostname)"
[[ -f "${REPO}/benchmarks/run_aiperf.py" ]] || \
    die "benchmarks/run_aiperf.py not found under REPO=${REPO}"
[[ -d "${HF_HOME}/hub" ]] || die "HF cache not found at ${HF_HOME}/hub"

mkdir -p "${RESULTS}" "${PLOTS}" "${LOGDIR}/lmcache" "${LOGDIR}/vllm"

# Install aiperf on the host (runs the benchmark client, not inside the container).
ensure_aiperf

# Load the main image.
_img_base="$(printf '%s' "${AIC_IMAGE}" | tr '/:' '--')"
# Source the load_image_if_needed helper from run-cliff.sbatch via monitoring-lib path
# reuse: define an inline minimal version here to stay self-contained.
_load_image() {
    local image="$1" glob="$2"
    local tb; tb="$(ls -t ${glob} 2>/dev/null | head -1 || true)"
    local have; have="$(docker images -q "${image}" 2>/dev/null)"
    local tarm; tarm="$(stat -c %Y "${tb}" 2>/dev/null || echo 0)"
    local marker="/var/tmp/aic-loaded-$(id -u)-$(printf '%s' "${image}" | tr '/:' '__').mtime"
    local loadedm; loadedm="$(cat "${marker}" 2>/dev/null || echo 0)"
    if [[ "${AIC_FORCE_LOAD:-0}" == "1" || -z "${have}" || "${tarm}" -gt "${loadedm}" ]]; then
        [[ -z "${tb}" ]] && { [[ -n "${have}" ]] && return 0; die "image ${image} absent and no tarball"; }
        log "loading ${image} from ${tb}"
        case "${tb}" in *.zst) zstd -dc "${tb}" | docker load ;; *.gz) gzip -dc "${tb}" | docker load ;; *) docker load -i "${tb}" ;; esac \
            || die "docker load failed"
        echo "${tarm}" > "${marker}" 2>/dev/null || true
    else
        log "image up to date on $(hostname): ${image}"
    fi
}
_load_image "${AIC_IMAGE}" "${AIC_IMAGE_DIR}/${_img_base}-*.tar.*"

# Export compose variables (same as run-cliff.sbatch).
export IMAGE_NAME="${AIC_IMAGE}"
export ROCM_ARCH="${ROCM_ARCH:-$(detect_gpu_arch)}"
export GPU=0
export HF_HOME NVME_DATA NFS_DATA GDS_SLAB_DATA
export LOG="${LOGDIR}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_MODEL TENSOR_PARALLEL_SIZE
export VLM_GPU_MEMORY_UTILIZATION VLM_MAX_MODEL_LEN VLM_MAX_NUM_BATCHED_TOKENS
export VLM_ATTENTION_BACKEND="${AIC_ATTENTION_BACKEND}"
export VLM_KV_CACHE_DTYPE="${AIC_KV_CACHE_DTYPE}"
export AIC_ROCM_USE_AITER
export LMCACHE_PORT LMCACHE_NVME_POOL LMCACHE_NVME_SLOT_SIZE
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
VLLM_EXTRA_ARGS=""
export VLLM_EXTRA_ARGS
AIC_GDS_SLAB_GB="${LMCACHE_L1_SIZE_GB}"

# Source monitoring helpers (same library as run-cliff.sbatch).
source "${REPO}/monitoring/monitoring-lib.sh"
ensure_compose || die "docker compose unavailable and could not be installed"
trap stop_monitoring EXIT

# Pre-pull Prometheus image.
_prom_img="${AIC_PROM_IMAGE:-prom/prometheus:v2.55.1}"
docker pull "${_prom_img}" >/dev/null 2>&1 && log "  ${_prom_img}: ready" \
    || log "  WARN: pull failed for ${_prom_img}"

start_monitoring

# Arm selector
_arm_on() { case ",${AIPERF_ARMS}," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
for _a in ${AIPERF_ARMS//,/ }; do
    case "${_a}" in vram|nvme) ;; *) die "unknown arm '${_a}' in AIPERF_ARMS (valid: vram,nvme)"; esac
done
log "arms        : ${AIPERF_ARMS}"
log "concur      : ${AIPERF_CONCUR}  requests/point=${AIPERF_REQUEST_COUNT}"
log "dataset     : ${AIPERF_DATASET}  ISL=${AIPERF_ISL} OSL=${AIPERF_OSL}"
log "tokenizer   : ${AIPERF_TOKENIZER:-(server token count)}"

declare -A STATUS
if _arm_on vram; then run_arm_vram && STATUS[vram_only]=ok || STATUS[vram_only]=FAIL; else STATUS[vram_only]=skip; fi
if _arm_on nvme; then run_arm_kvd  && STATUS[kvd_nvme]=ok  || STATUS[kvd_nvme]=FAIL;  else STATUS[kvd_nvme]=skip;  fi

run_plot

# Clean NVMe scratch
rm -rf "${AIC_LOCAL_ROOT}" 2>/dev/null || true

log "===================== SUMMARY ====================="
for k in vram_only kvd_nvme; do log "  ${k}: ${STATUS[$k]:-?}"; done
log "  CSVs  : ${RESULTS}"
log "  plots : ${PLOTS}/"
[[ "${AIC_MONITORING}" == "1" ]] && log "  metrics: ${AIC_METRICS_DIR} (Prometheus TSDB)"
log "==================================================="

rc=0
for k in vram_only kvd_nvme; do [[ "${STATUS[$k]:-skip}" == FAIL ]] && rc=1; done
exit "${rc}"
