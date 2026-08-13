#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# One serving-benchmark run: bring the compose stack up in a given configuration,
# measure it with vLLM's own `vllm bench serve`, tear down, print one RESULT line.
#
# Why `vllm bench serve` rather than a hand-rolled curl loop.  The previous probe
# (.slurm/capacity-probe.sbatch) reported a single throughput number derived from
# a Prometheus counter delta over a fixed wall-clock window.  That has no warm-up,
# no per-request latency, no notion of a completed request, and it counted the
# ramp-up.  `vllm bench serve` issues a fixed number of requests, waits for all of
# them, and reports TTFT / TPOT / ITL / E2E percentiles alongside throughput -- so
# "tokens/s/user" is measured rather than divided out, and prefill and decode can
# finally be told apart.
#
# The benchmark client runs INSIDE the vllm container (docker exec) because it is
# the only container in the stack that has vLLM installed -- the `client` service
# is a plain python:3.12 image.  The client is async and the node has 384 CPUs, so
# co-locating it does not perturb the measurement at these concurrencies (verified
# against a c=64 run from a separate container: within 1%).
#
# Env:
#   AIC_DAY_DIR   repo root                                (required)
#   AIC_IMAGE     image ref                                (required)
#   RUN_NAME      label for this configuration             (required)
#   TP            --tensor-parallel-size                   (default: 1)
#   DEVS          ROCR_VISIBLE_DEVICES                     (default: 0)
#   MODEL         served model                             (default: Qwen2.5-3B)
#   ISL / OSL     random-input-len / random-output-len     (default: 2048 / 512)
#   CONC_LIST     --max-concurrency values to sweep        (default: "8 32 64")
#   NUM_MULT      requests per concurrency = NUM_MULT x c  (default: 8, min 64)
#   GPU_UTIL      --gpu-memory-utilization                 (default: 0.85)
#   MAX_LEN       --max-model-len                          (default: ISL+OSL+1024)
#   BATCHED_TOK   --max-num-batched-tokens                 (default: 4096)
#   L1_GB / L2    LMCache L1 size / L2 backend             (default: 8 / none)
#   CONNECTOR     1 = LMCacheMPConnector, 0 = plain vLLM   (default: 1)
#   PREFIX_CACHE  1 = leave --enable-prefix-caching on     (default: 1)
#   EAGER         1 = add --enforce-eager (no HIP graphs)  (default: 0)
#   MONITORING    1 = also start prometheus/grafana        (default: 0)
#   READY_TRIES   x5s readiness budget                     (default: 240 = 20 min)
set -uo pipefail

: "${AIC_DAY_DIR:?}" "${AIC_IMAGE:?}" "${RUN_NAME:?}"
TP="${TP:-1}"; DEVS="${DEVS:-0}"; MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
ISL="${ISL:-2048}"; OSL="${OSL:-512}"
CONC_LIST="${CONC_LIST:-8 32 64}"; NUM_MULT="${NUM_MULT:-8}"
GPU_UTIL="${GPU_UTIL:-0.85}"; MAX_LEN="${MAX_LEN:-$(( ISL + OSL + 1024 ))}"
BATCHED_TOK="${BATCHED_TOK:-4096}"
L1_GB="${L1_GB:-8}"; L2="${L2:-none}"; CONNECTOR="${CONNECTOR:-1}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"; EAGER="${EAGER:-0}"
MONITORING="${MONITORING:-0}"; READY_TRIES="${READY_TRIES:-240}"

_logdir="${AIC_DAY_DIR}/logs/bench/${RUN_NAME}"; mkdir -p "${_logdir}"
say() { printf '[bench:%s] %s\n' "${RUN_NAME}" "$*"; }
say "host=$(hostname) tp=${TP} devs=${DEVS} model=${MODEL} isl=${ISL} osl=${OSL} conn=${CONNECTOR}"

[ -n "$(docker images -q "${AIC_IMAGE}")" ] || { say "FAIL: image ${AIC_IMAGE} not loaded"; exit 1; }

# NVMe scratch for the L2 arms.  Every arm gets its own pool dir so a previous
# run's cache files can never be counted as this run's hits.
NV=""
for d in /mnt/nixl-nvme-0 /mnt/nixl-nvme-* /mnt/lmcache-nvme; do
    [ -d "$d" ] && [ -w "$d" ] && { NV="$d"; break; }
done
[ -n "${NV}" ] || { say "FAIL: no writable NVMe mount"; exit 1; }
_stamp="${SLURM_JOB_ID:-$$}-${RUN_NAME}"
POOL="${NV}/aic-bench-${_stamp}"; NFSD="${NV}/aic-bench-nfs-${_stamp}"
mkdir -p "${POOL}" "${NFSD}"

cd "${AIC_DAY_DIR}" || exit 1
# shellcheck source=/dev/null
source "${AIC_DAY_DIR}/monitoring/monitoring-lib.sh"; ensure_compose || exit 1

export IMAGE_REF="${AIC_IMAGE}" IMAGE_NAME="${AIC_IMAGE%:*}" GPU=0
export AIC_VISIBLE_DEVICES="${DEVS}" TENSOR_PARALLEL_SIZE="${TP}"
export VLLM_MODEL="${MODEL}" HF_HOME="${HF_HOME:-/projects/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LOG="${_logdir}"
export NVME_DATA="${POOL}" NFS_DATA="${NFSD}" GDS_SLAB_DATA=""
export AIC_METRICS_DIR="${_logdir}/prometheus"
export LMCACHE_L1_SIZE_GB="${L1_GB}" AIC_L2_BACKEND="${L2}"
export VLM_GPU_MEMORY_UTILIZATION="${GPU_UTIL}"
export VLM_MAX_MODEL_LEN="${MAX_LEN}" VLM_MAX_NUM_BATCHED_TOKENS="${BATCHED_TOK}"
export VLLM_LOGGING_LEVEL=INFO   # WARNING hides vLLM's own KV-capacity line
PROM_UID="$(id -u)"; PROM_GID="$(id -g)"; export PROM_UID PROM_GID
mkdir -p "${_logdir}/vllm" "${AIC_METRICS_DIR}"

_extra=""
[ "${PREFIX_CACHE}" = "0" ] && _extra="${_extra} --no-enable-prefix-caching"
[ "${EAGER}" = "1" ]        && _extra="${_extra} --enforce-eager"
export VLLM_EXTRA_ARGS="${_extra}"

if [ "${CONNECTOR}" = "1" ]; then
    export VLLM_IPC_MODE=service:lmcache VLLM_PID_MODE=service:lmcache
    export KV_TRANSFER_ARG="--kv-transfer-config '{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://aic-lmcache\",\"lmcache.mp.port\":6555}}'"
    PROFILE=(--profile cache)
else
    export VLLM_IPC_MODE=shareable VLLM_PID_MODE=""
    export KV_TRANSFER_ARG=""
    PROFILE=()
fi
[ "${MONITORING}" = "1" ] && PROFILE+=(--profile monitoring)

COMPOSE_FILE="${AIC_DAY_DIR}/docker/docker-compose.yml"
compose() { docker compose -f "${COMPOSE_FILE}" "$@"; }
VLLM_CTR="aic-vllm-gpu0"

cleanup() {
    for svc in vllm lmcache lmcache-coordinator; do
        # `timeout` execs a binary and cannot run the compose() shell function --
        # it would resolve /usr/bin/compose (mailcap) instead of docker compose.
        timeout 60 docker compose -f "${COMPOSE_FILE}" logs --no-color --no-log-prefix "$svc" \
            > "${_logdir}/${svc}.log" 2>&1 || true
    done
    pkill -9 -f 'vllm.entrypoints.openai' 2>/dev/null || true
    pkill -9 -f 'EngineCore' 2>/dev/null || true
    pkill -9 -f 'lmcache server' 2>/dev/null || true
    sleep 2
    timeout 180 compose --profile cache --profile monitoring down --remove-orphans --timeout 5 >/dev/null 2>&1 || true
    for c in "${VLLM_CTR}" aic-lmcache aic-lmcache-coordinator aic-client aic-prometheus aic-grafana; do
        timeout 30 docker rm -f "$c" >/dev/null 2>&1 || true
    done
    rm -rf "${POOL}" "${NFSD}" 2>/dev/null || true
}
trap cleanup EXIT

_t0=$(date +%s)
if ! compose "${PROFILE[@]}" up -d > "${_logdir}/compose-up.log" 2>&1; then
    # compose's own stderr is the only place a name/port/mount conflict is
    # reported -- the vllm container may never have been created at all.
    say "FAIL: compose up"
    sed 's/^/    /' "${_logdir}/compose-up.log" | tail -25
    compose logs --tail 40 --no-color vllm 2>&1 | sed 's/^/    /' | tail -25
    echo "RESULT: name=${RUN_NAME} status=up_failed"; exit 1
fi

ready=0
for _ in $(seq 1 "${READY_TRIES}"); do
    docker exec "${VLLM_CTR}" curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { ready=1; break; }
    docker inspect -f '{{.State.Running}}' "${VLLM_CTR}" 2>/dev/null | grep -q true || break
    sleep 5
done
TTR=$(( $(date +%s) - _t0 ))
if [ "${ready}" != "1" ]; then
    say "FAIL: not ready after ${TTR}s"
    compose logs --tail 60 --no-color vllm 2>&1 | sed 's/^/    /' | tail -40
    echo "RESULT: name=${RUN_NAME} tp=${TP} devs=${DEVS} status=not_ready ttr_s=${TTR}"
    exit 1
fi
say "ready in ${TTR}s"

# vLLM states its own KV capacity in the INFO log; that is the authoritative number.
KVTOK=$(compose logs --no-color vllm 2>/dev/null | grep -m1 'GPU KV cache size' |
        grep -oE '[0-9,]+ tokens' | tr -d ', tokens')
CONCX=$(compose logs --no-color vllm 2>/dev/null | grep -m1 'Maximum concurrency' |
        grep -oE '[0-9.]+x' | tr -d x)
say "kv_tokens=${KVTOK:-?} max_conc=${CONCX:-?}x"

# --ignore-eos keeps every request exactly OSL tokens long.  Without it the model
# decides when to stop, output length varies per request, and a throughput number
# stops being comparable across configurations.
# The seed varies per (arm, concurrency).  With one fixed seed the warm-up and
# every sweep point draw the SAME prompt list, so later points hit KV that earlier
# points left in the prefix cache -- which quietly inflates every prefix-cache-on
# arm and makes the prefix-cache on/off comparison meaningless.
_seed_for() { printf '%s%s' "${RUN_NAME}" "$1" | cksum | cut -d' ' -f1; }

bench() {  # $1=concurrency $2=num-prompts $3=result-filename ; echoes the json path
    docker exec "${VLLM_CTR}" vllm bench serve \
        --backend openai-chat --endpoint /v1/chat/completions \
        --base-url http://127.0.0.1:8000 \
        --model "${MODEL}" \
        --dataset-name random \
        --random-input-len "${ISL}" --random-output-len "${OSL}" \
        --random-range-ratio 0.0 --random-prefix-len 0 \
        --ignore-eos --seed "$(_seed_for "$1")" \
        --max-concurrency "$1" --num-prompts "$2" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,95,99 \
        --save-result --result-dir /var/log/aic-vllm --result-filename "$3" \
        2>&1
}

# Warm-up: the first requests pay HIP graph capture, tokenizer load, and (on a
# cold prefix cache) full prefill.  Discarded -- it is not part of any result.
say "warm-up (32 requests @ c=8)"
bench 8 32 warmup.json > "${_logdir}/warmup.txt" 2>&1 || true

RESULTS=""
for C in ${CONC_LIST}; do
    N=$(( NUM_MULT * C )); [ "${N}" -lt 64 ] && N=64
    say "measuring c=${C} n=${N}"
    if ! bench "${C}" "${N}" "c${C}.json" > "${_logdir}/bench-c${C}.txt" 2>&1; then
        say "  c=${C}: bench FAILED (see bench-c${C}.txt)"
        RESULTS="${RESULTS} c${C}=failed"
        continue
    fi
    _j="${_logdir}/vllm/c${C}.json"
    if [ ! -s "${_j}" ]; then
        say "  c=${C}: no result json"; RESULTS="${RESULTS} c${C}=nojson"; continue
    fi
    read -r _line <<EOF
$(python3 - "${_j}" "${C}" "${TP}" <<'PY'
import json, sys
j = json.load(open(sys.argv[1])); c = int(sys.argv[2]); tp = int(sys.argv[3])
out   = j.get("output_throughput", 0.0)          # decode tokens/s, all users
total = j.get("total_token_throughput", 0.0)     # prompt + decode tokens/s
req   = j.get("request_throughput", 0.0)
# tokens/s/user is what a single interactive user perceives: 1/TPOT.
tpot  = j.get("mean_tpot_ms", 0.0)
per_user = (1000.0 / tpot) if tpot else 0.0
print("out=%.1f total=%.1f req=%.2f perGPU=%.1f perUser=%.1f "
      "ttft_p50=%.0f ttft_p95=%.0f tpot_p50=%.1f e2e_p50=%.0f e2e_p95=%.0f" % (
      out, total, req, out / tp, per_user,
      j.get("median_ttft_ms", 0), j.get("p95_ttft_ms", 0),
      j.get("median_tpot_ms", 0), j.get("median_e2el_ms", 0), j.get("p95_e2el_ms", 0)))
PY
)
EOF
    say "  c=${C}: ${_line}"
    RESULTS="${RESULTS} c${C}[${_line// /,}]"
done

echo "RESULT: name=${RUN_NAME} tp=${TP} devs=${DEVS} model=${MODEL} isl=${ISL} osl=${OSL} conn=${CONNECTOR} l2=${L2} l1_gb=${L1_GB} util=${GPU_UTIL} batched_tok=${BATCHED_TOK} prefix_cache=${PREFIX_CACHE} eager=${EAGER} ttr_s=${TTR} kv_tokens=${KVTOK:-?} max_conc=${CONCX:-?} sweep=${RESULTS# } status=ok"
