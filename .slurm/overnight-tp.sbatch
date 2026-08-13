#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Overnight exploration job: bring up the compose MP stack with a given
# tensor-parallel size / L2 backend, measure serving throughput, tear down.
#
# Deliberately standalone rather than a new run-build-distribute.sh subcommand:
# this is exploratory, and editing that script while another invocation of it is
# running corrupts the running interpreter (bash reads scripts by byte offset).
#
# Driven by .slurm/overnight-run.sh.  Every knob arrives through the environment
# (sbatch --export), and one result line is printed as RESULT: key=value ... so
# the driver can parse it without reading the whole log.
#
# Env:
#   AIC_DAY_DIR   repo root (required)
#   AIC_IMAGE     image ref (required)
#   TARBALL       image tarball to load if absent (required)
#   RUN_NAME      label for this configuration (required)
#   TP            --tensor-parallel-size            (default: 1)
#   DEVS          ROCR_VISIBLE_DEVICES              (default: 0)
#   MODEL         served model                      (default: Qwen/Qwen2.5-3B-Instruct)
#   L2            AIC_L2_BACKEND                    (default: none)
#   L1_GB         LMCACHE_L1_SIZE_GB                (default: 8)
#   GPU_UTIL      --gpu-memory-utilization          (default: 0.85)
#   MAX_LEN       --max-model-len                   (default: 8192)
#   CONC          concurrent probe streams          (default: 16)
#   PROBE_S       probe duration in seconds         (default: 60)
#   NVME_BASE     parent dir for the L2 pool        (default: auto-detect)
set -uo pipefail

: "${AIC_DAY_DIR:?}" "${AIC_IMAGE:?}" "${TARBALL:?}" "${RUN_NAME:?}"
TP="${TP:-1}"; DEVS="${DEVS:-0}"; MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
L2="${L2:-none}"; L1_GB="${L1_GB:-8}"; GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_LEN="${MAX_LEN:-8192}"; CONC="${CONC:-16}"; PROBE_S="${PROBE_S:-60}"

_logdir="${AIC_DAY_DIR}/logs/${SLURM_JOB_ID:-manual}"
mkdir -p "${_logdir}"
say() { printf '[overnight:%s] %s\n' "${RUN_NAME}" "$*"; }

say "host=$(hostname) tp=${TP} devs=${DEVS} model=${MODEL} l2=${L2} l1=${L1_GB}G util=${GPU_UTIL} connector=${CONNECTOR:-1}"

# --- image ------------------------------------------------------------------
if [ -z "$(docker images -q "${AIC_IMAGE}")" ]; then
    say "loading ${AIC_IMAGE}"
    zstd -dc "${TARBALL}" | docker load >/dev/null || { say "FAIL: image load"; exit 1; }
fi

# --- storage ----------------------------------------------------------------
_nvme="${NVME_BASE:-}"
if [ -z "${_nvme}" ]; then
    for d in /mnt/nixl-nvme-0 /mnt/nixl-nvme-* /mnt/lmcache-nvme; do
        [ -d "$d" ] && [ -w "$d" ] && { _nvme="$d"; break; }
    done
fi
[ -n "${_nvme}" ] || { say "FAIL: no writable NVMe mount"; exit 1; }
POOL="${_nvme}/aic-overnight-${SLURM_JOB_ID:-manual}"
NFS="${_nvme}/aic-overnight-nfs-${SLURM_JOB_ID:-manual}"
mkdir -p "${POOL}" "${NFS}"

# --- stack env --------------------------------------------------------------
cd "${AIC_DAY_DIR}"
# shellcheck source=/dev/null
source "${AIC_DAY_DIR}/monitoring/monitoring-lib.sh"
ensure_compose || { say "FAIL: no docker compose"; exit 1; }

export IMAGE_REF="${AIC_IMAGE}" IMAGE_NAME="${AIC_IMAGE%:*}"
export GPU=0                        # names the container; visibility is separate
export AIC_VISIBLE_DEVICES="${DEVS}" # the TP fix: comma list, container name unaffected
export TENSOR_PARALLEL_SIZE="${TP}"
export VLLM_MODEL="${MODEL}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # models are pre-cached; no token needed
export LOG="${_logdir}"
export NVME_DATA="${POOL}" NFS_DATA="${NFS}"
export GDS_SLAB_DATA=""
export AIC_METRICS_DIR="${_logdir}/prometheus"
export LMCACHE_L1_SIZE_GB="${L1_GB}"
export AIC_L2_BACKEND="${L2}"
export LMCACHE_NIXL_POSIX_SLOT_SIZE="${LMCACHE_NIXL_POSIX_SLOT_SIZE:-67108864}"
export VLM_GPU_MEMORY_UTILIZATION="${GPU_UTIL}"
export VLM_MAX_MODEL_LEN="${MAX_LEN}"
export VLM_MAX_NUM_BATCHED_TOKENS="${MAX_LEN}"
# CONNECTOR=0 isolates vLLM from LMCache entirely: no MP connector, no shared
# IPC/PID namespaces, and lmcache is not started at all (its depends_on is
# required:false).  This is the control for "does TP>1 work on this node at all",
# separating a vLLM/ROCm problem from an LMCacheMPConnector one -- the connector
# needs lmcache to share ONE device with vLLM for cross-container HIP IPC, which
# cannot hold when TP spreads KV tensors across N devices.
CONNECTOR="${CONNECTOR:-1}"
if [ "${CONNECTOR}" = "1" ]; then
    export VLLM_IPC_MODE=service:lmcache VLLM_PID_MODE=service:lmcache
    export KV_TRANSFER_ARG="--kv-transfer-config '{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://aic-lmcache\",\"lmcache.mp.port\":6555}}'"
    PROFILE=(--profile cache)
else
    export VLLM_IPC_MODE=shareable VLLM_PID_MODE=""
    export KV_TRANSFER_ARG=""
    PROFILE=()
fi

compose() { docker compose -f "${AIC_DAY_DIR}/docker/docker-compose.yml" "$@"; }
API="http://aic-vllm-gpu0:8000"

cleanup() {
    for svc in vllm lmcache; do
        # `timeout` execs a binary and cannot run the compose() shell function --
        # it would find /usr/bin/compose (mailcap) instead of docker compose.
        timeout 30 docker compose -f "${AIC_DAY_DIR}/docker/docker-compose.yml" \
            logs --no-color --no-log-prefix "$svc" \
            > "${_logdir}/${RUN_NAME}-${svc}.log" 2>&1 || true
    done
    pkill -9 -f 'vllm.entrypoints.openai' 2>/dev/null || true
    pkill -9 -f 'EngineCore'              2>/dev/null || true
    pkill -9 -f 'lmcache server'          2>/dev/null || true
    sleep 2
    timeout 90 compose --profile cache --profile monitoring down --remove-orphans --timeout 5 >/dev/null 2>&1 || true
    for c in aic-vllm-gpu0 aic-lmcache aic-client aic-prometheus aic-grafana; do
        timeout 30 docker rm -f "$c" >/dev/null 2>&1 || true
    done
    rm -rf "${POOL}" "${NFS}" 2>/dev/null || true
}
trap cleanup EXIT

# --- bring up ---------------------------------------------------------------
_t0=$(date +%s)
if ! compose "${PROFILE[@]}" up -d vllm client >/dev/null 2>&1; then
    say "FAIL: compose up"
    compose logs --tail 40 --no-color vllm 2>&1 | sed 's/^/    /'
    echo "RESULT: name=${RUN_NAME} tp=${TP} status=compose_up_failed"
    exit 1
fi

ready=0
for _ in $(seq 1 "${READY_TRIES:-120}"); do   # 120x5s = 10 min; a stack not up by then is not coming up
    docker exec aic-client curl -fsS "${API}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
    sleep 5
done
_ttr=$(( $(date +%s) - _t0 ))
if [ "${ready}" != "1" ]; then
    say "FAIL: vLLM not ready after ${_ttr}s"
    compose logs --tail 60 --no-color vllm 2>&1 | sed 's/^/    /' | tail -40
    echo "RESULT: name=${RUN_NAME} tp=${TP} status=not_ready ttr_s=${_ttr}"
    exit 1
fi
say "ready in ${_ttr}s"

# --- throughput probe -------------------------------------------------------
# Counter delta over a fixed window is far more robust than timing individual
# requests: it needs no client-side concurrency accounting and ignores warm-up.
gen_tokens() {
    docker exec aic-client curl -fsS "${API}/metrics" 2>/dev/null |
        awk '/^vllm:generation_tokens_total/ {s+=$2} END {printf "%d", s+0}'
}
PROMPT='Write a detailed technical explanation of GPU memory hierarchies.'
fire() {
    local end=$(( $(date +%s) + ${1:-$PROBE_S} ))
    while [ "$(date +%s)" -lt "${end}" ]; do
        docker exec aic-client curl -fsS "${API}/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT} (${RANDOM})\"}],\"max_tokens\":256,\"temperature\":0.7}" \
            >/dev/null 2>&1 || true
    done
}
say "warming up (20s, 1 stream)"; fire 20
say "probing ${PROBE_S}s at concurrency ${CONC}"
_b0=$(gen_tokens); _s0=$(date +%s)
for _ in $(seq 1 "${CONC}"); do fire "${PROBE_S}" & done
wait
_b1=$(gen_tokens); _s1=$(date +%s)
_dt=$(( _s1 - _s0 )); [ "${_dt}" -lt 1 ] && _dt=1
_tok=$(( _b1 - _b0 ))
_tps=$(awk -v t="${_tok}" -v d="${_dt}" 'BEGIN{printf "%.1f", t/d}')

say "generated ${_tok} tokens in ${_dt}s -> ${_tps} tok/s"
echo "RESULT: name=${RUN_NAME} tp=${TP} devs=${DEVS} connector=${CONNECTOR} model=${MODEL} l2=${L2} l1_gb=${L1_GB} conc=${CONC} ttr_s=${_ttr} tokens=${_tok} secs=${_dt} tok_s=${_tps} status=ok"
exit 0
