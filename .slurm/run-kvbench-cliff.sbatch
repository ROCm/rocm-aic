#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
#SBATCH --job-name=aic-kvbench-cliff
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#
# CPU-only KVBench cliff run. Builds/starts the compose `kvbench` profile plus the
# shared `client` container on the internal docker network, then runs the existing
# cliff harness against http://aic-kvbench:8000 from inside the client container.

set -uo pipefail
export PATH="/usr/local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

LOGDIR_REL="logs/${SLURM_JOB_ID:-manual}"
LOGDIR="${REPO}/${LOGDIR_REL}"
if mkdir -p "${LOGDIR}" 2>/dev/null; then
  exec >>"${LOGDIR}/kvbench-cliff.out" 2>&1
fi

KVBENCH_MODEL="${KVBENCH_MODEL:-llama-3.1-8b}"
KVBENCH_PORT="${KVBENCH_PORT:-8000}"
KVBENCH_HOST_PORT="${KVBENCH_HOST_PORT:-18000}"
KVBENCH_READY_TIMEOUT="${KVBENCH_READY_TIMEOUT:-120}"
BENCH_ISL="${BENCH_ISL:-32}"
BENCH_SHARED_TOK="${BENCH_SHARED_TOK:-16}"
BENCH_CONCUR="${BENCH_CONCUR:-1,2}"
BENCH_ITERS="${BENCH_ITERS:-1}"
KVBENCH_IMAGE_REF="${KVBENCH_IMAGE_REF:-aic-kvbench:177cf4e}"

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }
die() { log "FATAL: $*"; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker not found"
[[ -f "${REPO}/benchmarks/run_cliff.py" ]] ||
  die "benchmarks/run_cliff.py not found under REPO=${REPO}"

cd "${REPO}"
# shellcheck source=/dev/null
source "${REPO}/monitoring/monitoring-lib.sh"
ensure_compose || die "docker compose unavailable and could not be installed"

export KVBENCH_MODEL KVBENCH_PORT KVBENCH_HOST_PORT KVBENCH_IMAGE_REF
export LOG="${LOGDIR}"
compose() { docker compose -f "${REPO}/docker/docker-compose.yml" "$@"; }

cleanup() {
  timeout 60 compose --profile kvbench logs --no-color --no-log-prefix kvbench client \
    > "${LOGDIR}/kvbench-compose.log" 2>&1 || true
  timeout 60 compose --profile kvbench down --remove-orphans --timeout 10 >/dev/null 2>&1 || true
}
trap cleanup EXIT

log "bringing up KVBench + client on the internal compose network"
compose --profile kvbench up --build -d kvbench client || die "compose up failed"

log "waiting for aic-kvbench to serve /v1/models"
ready=0
for _i in $(seq 1 "${KVBENCH_READY_TIMEOUT}"); do
  if curl -fsS "http://127.0.0.1:${KVBENCH_HOST_PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
[[ "${ready}" == "1" ]] || die "KVBench endpoint never became ready"

log "waiting for client deps to finish installing"
client_ready=0
for _i in $(seq 1 60); do
  if docker exec aic-client python3 -c 'import openai' >/dev/null 2>&1; then
    client_ready=1
    break
  fi
  sleep 1
done
[[ "${client_ready}" == "1" ]] || die "client container never became ready"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="/logs/${SLURM_JOB_ID:-manual}/results/cliff-kvbench-${STAMP}.csv"
log "running cliff benchmark -> ${OUT}"
docker exec aic-client python3 -u benchmarks/run_cliff.py \
  --endpoint "http://aic-kvbench:${KVBENCH_PORT}" \
  --model "${KVBENCH_MODEL}" \
  --arm kvbench \
  --isl "${BENCH_ISL}" \
  --shared-prefix-tokens "${BENCH_SHARED_TOK}" \
  --concurrencies "${BENCH_CONCUR}" \
  --iters "${BENCH_ITERS}" \
  --warmup-iters 0 \
  --request-timeout 10 \
  --out "${OUT}" || die "cliff benchmark failed"

log "results: ${REPO}/logs/${OUT#/logs/}"
log "done"
