#!/usr/bin/env bash
set -euo pipefail

# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Build-time contract:
#   - IMAGE_REF points at a locally-loaded emulate image
#   - docker compose is available
#   - ROCJITSU_OUT_DIR is a writable host directory
#
# Runtime contract:
#   - boots the emulate compose profile on a CPU-only runner
#   - runs a fixed vllm bench serve sweep against the emulator
#   - copies the resulting emu-*.json files into $ROCJITSU_OUT_DIR/results

IMAGE_REF="${IMAGE_REF:?IMAGE_REF must be set}"
ROCJITSU_OUT_DIR="${ROCJITSU_OUT_DIR:?ROCJITSU_OUT_DIR must be set}"
ROCJITSU_MODEL="${ROCJITSU_MODEL:-Qwen/Qwen3-8B}"
ROCJITSU_PROFILE_PACK="${ROCJITSU_PROFILE_PACK:-/opt/llm-emu/profiles/MI300X-Qwen3-8B.json}"
ROCJITSU_SWEEP="${ROCJITSU_SWEEP:-1024,128,1,12 1024,128,16,64 4096,128,8,32}"
ROCJITSU_READY_TIMEOUT="${ROCJITSU_READY_TIMEOUT:-96}"
ROCJITSU_ORACLE_K="${ROCJITSU_ORACLE_K:-1}"

LOG_DIR="${ROCJITSU_OUT_DIR}/logs"
RESULTS_DIR="${ROCJITSU_OUT_DIR}/results"

mkdir -p "${LOG_DIR}/vllm-emulator/bench" "${RESULTS_DIR}"

compose() {
    docker compose -f docker/docker-compose.yml "$@"
}

cleanup() {
    compose --profile emulate logs --no-color --no-log-prefix vllm-emulator \
        > "${LOG_DIR}/emulate-bench.log" 2>&1 || true
    compose --profile emulate down --remove-orphans --timeout 10 >/dev/null 2>&1 || true
    docker rm -f aic-vllm-emulator >/dev/null 2>&1 || true
}
trap cleanup EXIT

export IMAGE_REF
export VLLM_MODEL="${ROCJITSU_MODEL}"
export VLLM_EMULATOR_PROFILE_PACK="${ROCJITSU_PROFILE_PACK}"
export VLLM_EMULATOR_ORACLE_K="${ROCJITSU_ORACLE_K}"
export HF_HOME="${HF_HOME:-${RUNNER_TEMP:-/tmp}/hf}"
export LOG="${LOG_DIR}"

echo "[rocjitsu-emulated] starting emulator for ${ROCJITSU_MODEL}"
compose --profile emulate up -d vllm-emulator

ready=0
for _i in $(seq 1 "${ROCJITSU_READY_TIMEOUT}"); do
    if curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1; then
        ready=1
        break
    fi
    if [ -z "$(docker ps -q -f name=aic-vllm-emulator)" ]; then
        echo "[rocjitsu-emulated] FAIL: emulator container exited during startup" >&2
        compose --profile emulate logs --tail 120 --no-color vllm-emulator 2>&1 | sed 's/^/  [emu] /'
        exit 1
    fi
    sleep 5
done

if [ "${ready}" != "1" ]; then
    echo "[rocjitsu-emulated] FAIL: endpoint never became ready" >&2
    compose --profile emulate logs --tail 120 --no-color vllm-emulator 2>&1 | sed 's/^/  [emu] /'
    exit 1
fi

compose --profile emulate logs --no-color --no-log-prefix vllm-emulator \
    > "${LOG_DIR}/emulate-startup.log" 2>&1 || true
if ! grep -q '\[ExecutorEmulatorHook\] Enabled' "${LOG_DIR}/emulate-startup.log"; then
    echo "[rocjitsu-emulated] FAIL: emulator hook never activated" >&2
    tail -60 "${LOG_DIR}/emulate-startup.log" | sed 's/^/  [emu] /'
    exit 1
fi

model_tag="$(printf '%s' "${ROCJITSU_MODEL}" | tr '/' '-')"
for point in ${ROCJITSU_SWEEP}; do
    isl="$(printf '%s' "${point}" | cut -d, -f1)"
    osl="$(printf '%s' "${point}" | cut -d, -f2)"
    conc="$(printf '%s' "${point}" | cut -d, -f3)"
    nprompts="$(printf '%s' "${point}" | cut -d, -f4)"
    outfile="emu-${model_tag}-isl${isl}-osl${osl}-c${conc}.json"
    echo "[rocjitsu-emulated] bench isl=${isl} osl=${osl} concurrency=${conc} prompts=${nprompts}"
    docker exec aic-vllm-emulator vllm bench serve \
        --host localhost --port 8000 --model "${ROCJITSU_MODEL}" \
        --dataset-name random \
        --random-input-len "${isl}" --random-output-len "${osl}" \
        --num-prompts "${nprompts}" --max-concurrency "${conc}" \
        --ignore-eos --seed 0 \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result --result-dir /var/log/aic-vllm-emulator/bench \
        --result-filename "${outfile}" \
        2>&1 | tail -25 | sed 's/^/  [bench] /'
done

find "${LOG_DIR}/vllm-emulator/bench" -maxdepth 1 -name 'emu-*.json' -type f -exec cp {} "${RESULTS_DIR}/" \;
count="$(find "${RESULTS_DIR}" -maxdepth 1 -name 'emu-*.json' | wc -l | tr -d ' ')"
[ "${count}" -gt 0 ] || { echo "[rocjitsu-emulated] FAIL: no result JSONs copied" >&2; exit 1; }
echo "[rocjitsu-emulated] wrote ${count} result files to ${RESULTS_DIR}"
