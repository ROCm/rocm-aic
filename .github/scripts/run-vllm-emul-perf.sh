#!/usr/bin/env bash
set -euo pipefail

# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Build-time contract:
#   - IMAGE_REF points at a locally-loaded emulate image
#   - docker compose is available
#   - VLLM_EMUL_OUT_DIR is a writable host directory
#
# Runtime contract:
#   - boots the emulate compose profile on a CPU-only runner
#   - runs a fixed vllm bench serve sweep against the emulator
#   - copies the resulting emu-*.json files into $VLLM_EMUL_OUT_DIR/results

IMAGE_REF="${IMAGE_REF:?IMAGE_REF must be set}"
VLLM_EMUL_OUT_DIR="${VLLM_EMUL_OUT_DIR:?VLLM_EMUL_OUT_DIR must be set}"
VLLM_EMUL_MODEL="${VLLM_EMUL_MODEL:-Qwen/Qwen3-8B}"
VLLM_EMUL_PROFILE_PACK="${VLLM_EMUL_PROFILE_PACK:-/opt/llm-emu/profiles/MI300X-Qwen3-8B.json}"
VLLM_EMUL_SWEEP="${VLLM_EMUL_SWEEP:-1024,128,1,12 1024,128,16,64 4096,128,8,32}"
VLLM_EMUL_READY_TIMEOUT="${VLLM_EMUL_READY_TIMEOUT:-96}"
VLLM_EMUL_ORACLE_K="${VLLM_EMUL_ORACLE_K:-1}"

LOG_DIR="${VLLM_EMUL_OUT_DIR}/logs"
RESULTS_DIR="${VLLM_EMUL_OUT_DIR}/results"

mkdir -p "${LOG_DIR}/vllm-emulator/bench" "${RESULTS_DIR}"
# shellcheck source=scripts/aic-endpoints.sh
source scripts/aic-endpoints.sh
EMU_C="${AIC_STACK}-vllm-emulator"

compose() {
    docker compose -f docker/docker-compose.yml "$@"
}

cleanup() {
    compose --profile emulate logs --no-color --no-log-prefix vllm-emulator \
        > "${LOG_DIR}/emulate-bench.log" 2>&1 || true
    compose --profile emulate down --remove-orphans --timeout 10 >/dev/null 2>&1 || true
    docker rm -f "${EMU_C}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

export IMAGE_REF
export VLLM_MODEL="${VLLM_EMUL_MODEL}"
export VLLM_EMULATOR_PROFILE_PACK="${VLLM_EMUL_PROFILE_PACK}"
export VLLM_EMULATOR_ORACLE_K="${VLLM_EMUL_ORACLE_K}"
export HF_HOME="${HF_HOME:-${RUNNER_TEMP:-/tmp}/hf}"
export LOG="${LOG_DIR}"

echo "[vllm-emul] starting emulator for ${VLLM_EMUL_MODEL}"
compose --profile emulate up -d vllm-emulator

ready=0
for _i in $(seq 1 "${VLLM_EMUL_READY_TIMEOUT}"); do
    if aic_probe "${EMU_C}" 8000 /v1/models; then
        ready=1
        break
    fi
    if [ -z "$(docker ps -q -f "name=^${EMU_C}$")" ]; then
        echo "[vllm-emul] FAIL: emulator container exited during startup" >&2
        compose --profile emulate logs --tail 120 --no-color vllm-emulator 2>&1 | sed 's/^/  [emu] /'
        exit 1
    fi
    sleep 5
done

if [ "${ready}" != "1" ]; then
    echo "[vllm-emul] FAIL: endpoint never became ready" >&2
    compose --profile emulate logs --tail 120 --no-color vllm-emulator 2>&1 | sed 's/^/  [emu] /'
    exit 1
fi

compose --profile emulate logs --no-color --no-log-prefix vllm-emulator \
    > "${LOG_DIR}/emulate-startup.log" 2>&1 || true
if ! grep -q '\[ExecutorEmulatorHook\] Enabled' "${LOG_DIR}/emulate-startup.log"; then
    echo "[vllm-emul] FAIL: emulator hook never activated" >&2
    tail -60 "${LOG_DIR}/emulate-startup.log" | sed 's/^/  [emu] /'
    exit 1
fi

model_tag="$(printf '%s' "${VLLM_EMUL_MODEL}" | tr '/' '-')"
for point in ${VLLM_EMUL_SWEEP}; do
    isl="$(printf '%s' "${point}" | cut -d, -f1)"
    osl="$(printf '%s' "${point}" | cut -d, -f2)"
    conc="$(printf '%s' "${point}" | cut -d, -f3)"
    nprompts="$(printf '%s' "${point}" | cut -d, -f4)"
    outfile="emu-${model_tag}-isl${isl}-osl${osl}-c${conc}.json"
    echo "[vllm-emul] bench isl=${isl} osl=${osl} concurrency=${conc} prompts=${nprompts}"
    docker exec "${EMU_C}" vllm bench serve \
        --host localhost --port 8000 --model "${VLLM_EMUL_MODEL}" \
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
[ "${count}" -gt 0 ] || { echo "[vllm-emul] FAIL: no result JSONs copied" >&2; exit 1; }
echo "[vllm-emul] wrote ${count} result files to ${RESULTS_DIR}"
