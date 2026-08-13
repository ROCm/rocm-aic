#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Does TP>1 raise the CONCURRENCY / CONTEXT ceiling?
#
# The first throughput probe could not answer this.  It used a ~10-token prompt
# with max_tokens=256, so ~266 tokens of KV per request; at c=64 that is ~17k
# tokens against a KV cache of 9.1M.  The cache was never the binding constraint,
# so splitting it across GPUs could not possibly help.  On a 3B model on a 192GB
# MI300X, vLLM reports 554x concurrency at 16k context on ONE GPU -- TP only
# matters when the weights leave little room for KV.
#
# Measures the ceiling two ways:
#   1. Static  -- vLLM's own "GPU KV cache size" / "Maximum concurrency" (INFO log)
#                 and cache_config_info from /metrics.
#   2. Dynamic -- long prompts at rising concurrency, watching
#                 vllm:num_requests_waiting.  Requests queue once KV is full, so
#                 the concurrency where waiting first goes positive IS the ceiling.
#
# Env:
#   AIC_DAY_DIR AIC_IMAGE TARBALL RUN_NAME   (required)
#   TP DEVS MODEL GPU_UTIL MAX_LEN L1_GB L2  (stack config)
#   ISL_WORDS    prompt length in words      (default: 6000 ~= 8k tokens)
#   OUT_TOKENS   max_tokens per request      (default: 128)
#   CONC_LIST    concurrencies to sweep      (default: "8 32 64 128")
#   READY_TRIES  x5s readiness budget        (default: 120 = 10 min)
set -uo pipefail

: "${AIC_DAY_DIR:?}" "${AIC_IMAGE:?}" "${TARBALL:?}" "${RUN_NAME:?}"
TP="${TP:-1}"; DEVS="${DEVS:-0}"; MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
GPU_UTIL="${GPU_UTIL:-0.85}"; MAX_LEN="${MAX_LEN:-16384}"
ISL_WORDS="${ISL_WORDS:-6000}"; OUT_TOKENS="${OUT_TOKENS:-128}"
CONC_LIST="${CONC_LIST:-8 32 64 128}"; READY_TRIES="${READY_TRIES:-120}"

_logdir="${AIC_DAY_DIR}/logs/${SLURM_JOB_ID:-manual}"; mkdir -p "${_logdir}"
say() { printf '[capacity:%s] %s\n' "${RUN_NAME}" "$*"; }
say "host=$(hostname) tp=${TP} devs=${DEVS} model=${MODEL} isl_words=${ISL_WORDS} max_len=${MAX_LEN}"

[ -n "$(docker images -q "${AIC_IMAGE}")" ] || zstd -dc "${TARBALL}" | docker load >/dev/null

NV=""
for d in /mnt/nixl-nvme-0 /mnt/nixl-nvme-* /mnt/lmcache-nvme; do
    [ -d "$d" ] && [ -w "$d" ] && { NV="$d"; break; }
done
[ -n "${NV}" ] || { say "FAIL: no writable NVMe mount"; exit 1; }
POOL="${NV}/aic-cap-${SLURM_JOB_ID:-manual}"; NFSD="${NV}/aic-cap-nfs-${SLURM_JOB_ID:-manual}"
mkdir -p "${POOL}" "${NFSD}"

cd "${AIC_DAY_DIR}"
# shellcheck source=/dev/null
source "${AIC_DAY_DIR}/monitoring/monitoring-lib.sh"; ensure_compose || exit 1

export IMAGE_REF="${AIC_IMAGE}" IMAGE_NAME="${AIC_IMAGE%:*}" GPU=0
export AIC_VISIBLE_DEVICES="${DEVS}" TENSOR_PARALLEL_SIZE="${TP}"
export VLLM_MODEL="${MODEL}" HF_HOME="${HF_HOME:-/projects/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LOG="${_logdir}"
export NVME_DATA="${POOL}" NFS_DATA="${NFSD}" GDS_SLAB_DATA=""
export AIC_METRICS_DIR="${_logdir}/prometheus"
export LMCACHE_L1_SIZE_GB="${L1_GB:-8}" AIC_L2_BACKEND="${L2:-none}"
export VLM_GPU_MEMORY_UTILIZATION="${GPU_UTIL}"
export VLM_MAX_MODEL_LEN="${MAX_LEN}" VLM_MAX_NUM_BATCHED_TOKENS="${MAX_LEN}"
export VLLM_IPC_MODE=service:lmcache VLLM_PID_MODE=service:lmcache
export VLLM_LOGGING_LEVEL=INFO   # WARNING hides vLLM's own KV-capacity line
export KV_TRANSFER_ARG="--kv-transfer-config '{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://aic-lmcache\",\"lmcache.mp.port\":6555}}'"

COMPOSE_FILE="${AIC_DAY_DIR}/docker/docker-compose.yml"
compose() { docker compose -f "${COMPOSE_FILE}" "$@"; }
API="http://aic-vllm-gpu0:8000"

cleanup() {
    for svc in vllm lmcache; do
        # NOT `timeout ... compose`: timeout execs a binary and cannot run a shell
        # function, so it would find /usr/bin/compose (mailcap) instead.
        timeout 60 docker compose -f "${COMPOSE_FILE}" logs --no-color --no-log-prefix "$svc" \
            > "${_logdir}/${RUN_NAME}-${svc}.log" 2>&1 || true
    done
    pkill -9 -f 'vllm.entrypoints.openai' 2>/dev/null || true
    pkill -9 -f 'EngineCore' 2>/dev/null || true
    pkill -9 -f 'lmcache server' 2>/dev/null || true
    sleep 2
    timeout 120 compose --profile cache down --remove-orphans --timeout 5 >/dev/null 2>&1 || true
    for c in aic-vllm-gpu0 aic-lmcache aic-client aic-prometheus aic-grafana; do
        timeout 30 docker rm -f "$c" >/dev/null 2>&1 || true
    done
    rm -rf "${POOL}" "${NFSD}" 2>/dev/null || true
}
trap cleanup EXIT

_t0=$(date +%s)
compose --profile cache up -d >/dev/null 2>&1 || {
    echo "RESULT: name=${RUN_NAME} tp=${TP} status=up_failed"; exit 1; }

ready=0
for _ in $(seq 1 "${READY_TRIES}"); do
    docker exec aic-client curl -fsS "${API}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
    sleep 5
done
TTR=$(( $(date +%s) - _t0 ))
if [ "${ready}" != "1" ]; then
    say "FAIL: not ready after ${TTR}s"
    compose logs --tail 40 --no-color vllm 2>&1 | sed 's/^/    /' | tail -25
    echo "RESULT: name=${RUN_NAME} tp=${TP} devs=${DEVS} status=not_ready ttr_s=${TTR}"
    exit 1
fi
say "ready in ${TTR}s"

m() { docker exec aic-client curl -fsS "${API}/metrics" 2>/dev/null; }
gen()   { m | awk '/^vllm:generation_tokens_total/ {s+=$2} END {printf "%d", s+0}'; }
wait_q(){ m | awk '/^vllm:num_requests_waiting/ {s+=$2} END {printf "%d", s+0}'; }
run_q() { m | awk '/^vllm:num_requests_running/ {s+=$2} END {printf "%d", s+0}'; }

# vLLM states its own capacity in the INFO log; that is the authoritative number.
KVLINE=$(compose logs --no-color vllm 2>/dev/null | grep -m1 'GPU KV cache size' | tr -s ' ')
MAXCONC=$(compose logs --no-color vllm 2>/dev/null | grep -m1 'Maximum concurrency' | tr -s ' ')
say "${KVLINE:-<no KV cache line>}"
say "${MAXCONC:-<no max-concurrency line>}"
KVTOK=$(printf '%s' "${KVLINE}" | grep -oE '[0-9,]+ tokens' | tr -d ', tokens')
CONCX=$(printf '%s' "${MAXCONC}" | grep -oE '[0-9.]+x' | tr -d x)

# The random prefix MUST come first.  vLLM's automatic prefix caching matches from
# position 0, so an identical leading prompt with a random SUFFIX is deduped into a
# single shared prefix -- 246 concurrent 8k requests then consumed almost no KV and
# never queued, making the cache look infinite.  Randomising the head breaks sharing.
PROMPT=$(printf 'token %.0s' $(seq 1 "${ISL_WORDS}"))

# Drive the whole fan-out INSIDE the client container in ONE docker exec.  The
# previous probe spawned one `docker exec` per request from the job shell; at
# c>=32 that made the CLIENT the bottleneck -- throughput went flat at ~1300 tok/s
# while the server reported an empty queue, the signature of a starved server
# rather than a saturated one.
run_load() {  # $1=concurrency $2=duration_s
    docker exec -i aic-client bash -s <<INNER
p='${PROMPT}'
for i in \$(seq 1 $1); do
  ( e=\$(( \$(date +%s) + $2 ))
    while [ "\$(date +%s)" -lt "\$e" ]; do
      curl -fsS '${API}/v1/chat/completions' -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"\$RANDOM\$RANDOM\$RANDOM \${p}\"}],\"max_tokens\":${OUT_TOKENS},\"temperature\":0.7}" \
        >/dev/null 2>&1 || true
    done ) &
done
wait
INNER
}

RESULTS=""
for C in ${CONC_LIST}; do
    say "sweep c=${C}"
    b0=$(gen); t0=$(date +%s); maxq=0; maxr=0
    run_load "${C}" 45 >/dev/null 2>&1 &
    lp=$!
    while kill -0 "${lp}" 2>/dev/null; do
        q=$(wait_q); r=$(run_q)
        [ "${q:-0}" -gt "${maxq}" ] && maxq="${q}"
        [ "${r:-0}" -gt "${maxr}" ] && maxr="${r}"
        sleep 5
    done
    wait "${lp}" 2>/dev/null
    b1=$(gen); t1=$(date +%s); dt=$(( t1 - t0 )); [ "${dt}" -lt 1 ] && dt=1
    tps=$(awk -v t="$(( b1 - b0 ))" -v d="${dt}" 'BEGIN{printf "%.1f", t/d}')
    say "c=${C}: ${tps} tok/s  peak_running=${maxr}  peak_queued=${maxq}"
    RESULTS="${RESULTS} c${C}=${tps}/r${maxr}/q${maxq}"
done

echo "RESULT: name=${RUN_NAME} tp=${TP} devs=${DEVS} model=${MODEL} ttr_s=${TTR} kv_tokens=${KVTOK:-?} max_conc=${CONCX:-?} isl_words=${ISL_WORDS} sweep=${RESULTS# } status=ok"
