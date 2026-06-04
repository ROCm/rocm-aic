#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

# Launch N parallel run-long.sh workers, each with a distinct RUN_LONG_SEED.
#
# Usage:
#   ./run-long-parallel.sh [WORKERS]
#   WORKERS=8 ITERATIONS=10 ./run-long-parallel.sh
#
# Env (passed through to each worker):
#   BASE_URL, MODEL, BOOK_DATA_ROOT, BOOK_SLUG, BOOK_SLUGS, BOOK_SLUG_FILE,
#   ITERATIONS,
#   RUN_LONG_COMBINE_CHUNKS, RUN_LONG_MAX_TOKENS, etc.
#
# Wrapper-only env:
#   WORKERS         number of parallel workers (default: 4; or first argument)
#   BASE_SEED       seed for worker 0; worker i uses BASE_SEED + i (default: $RANDOM)
#   OUTPUT_DIR      directory for per-worker .jsonl and .log
#   STAGGER_SEC     delay between starting workers (default: 0)
#   PROGRESS        1=on, 0=off, unset=on when stderr is a TTY (iteration bar)
#   PROGRESS_WIDTH  bar width in characters (default: 40)

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_long="${here}/run-long.sh"
runtime_loader="${here}/../lib/load-runtime.sh"
if [[ -x "${runtime_loader}" ]]; then
  if ! runtime_exports="$("${runtime_loader}" llm-prefill "${here}")"; then
    exit 1
  fi
  if [[ -n "${runtime_exports}" ]]; then
    source /dev/stdin <<<"${runtime_exports}"
  fi
fi

if [[ ! -x "${run_long}" ]]; then
  echo "error: missing or non-executable ${run_long}" >&2
  exit 1
fi

WORKERS="${1:-${WORKERS:-4}}"
BASE_SEED="${BASE_SEED:-$RANDOM}"
OUTPUT_DIR="${OUTPUT_DIR:-${here}/logs/run-long-parallel}"
STAGGER_SEC="${STAGGER_SEC:-0}"
ITERATIONS="${ITERATIONS:-1}"
PROGRESS_WIDTH="${PROGRESS_WIDTH:-40}"

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: WORKERS must be a positive integer (got ${WORKERS})" >&2
  exit 1
fi
if ! [[ "${BASE_SEED}" =~ ^[0-9]+$ ]]; then
  echo "error: BASE_SEED must be a non-negative integer (got ${BASE_SEED})" >&2
  exit 1
fi
if ! [[ "${STAGGER_SEC}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "error: STAGGER_SEC must be a non-negative number (got ${STAGGER_SEC})" >&2
  exit 1
fi
if ! [[ "${PROGRESS_WIDTH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: PROGRESS_WIDTH must be a positive integer (got ${PROGRESS_WIDTH})" >&2
  exit 1
fi

progress_enabled() {
  case "${PROGRESS:-auto}" in
    0 | false | no | off) return 1 ;;
    1 | true | yes | on) return 0 ;;
    auto) [[ -t 2 ]] ;;
    *) return 0 ;;
  esac
}

TOTAL_ITERATIONS=$((WORKERS * ITERATIONS))
run_dir=""
pids=()

iterations_done() {
  local w f n=0 c
  for ((w = 0; w < WORKERS; w++)); do
    f="${run_dir}/worker-${w}.jsonl"
    if [[ -f "${f}" ]]; then
      c=$(grep -c '"run_long_iteration"' "${f}" 2>/dev/null || true)
      n=$((n + c))
    fi
  done
  echo "${n}"
}

workers_running() {
  local w
  for ((w = 0; w < WORKERS; w++)); do
    if kill -0 "${pids[w]}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

draw_progress_bar() {
  local done=$1 total=$2
  local width=${PROGRESS_WIDTH}
  local pct=0 filled=0 i bar=""
  if (( total > 0 )); then
    pct=$((done * 100 / total))
    filled=$((done * width / total))
  fi
  if (( filled > width )); then
    filled=${width}
  fi
  for ((i = 0; i < filled; i++)); do
    bar+='#'
  done
  for ((i = filled; i < width; i++)); do
    bar+='-'
  done
  if [[ -t 2 ]]; then
    printf '\r\033[Krun-long-parallel: [%s] %d/%d (%d%%)' \
      "${bar}" "${done}" "${total}" "${pct}" >&2
  else
    printf 'run-long-parallel: [%s] %d/%d (%d%%)\n' \
      "${bar}" "${done}" "${total}" "${pct}" >&2
  fi
}

progress_monitor() {
  local last_done=-1 done=0
  while workers_running; do
    done=$(iterations_done)
    if (( done != last_done )); then
      draw_progress_bar "${done}" "${TOTAL_ITERATIONS}"
      last_done=${done}
    fi
    sleep 0.25
  done
  done=$(iterations_done)
  draw_progress_bar "${done}" "${TOTAL_ITERATIONS}"
  if [[ -t 2 ]]; then
    printf '\n' >&2
  fi
}

mkdir -p "${OUTPUT_DIR}"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="${OUTPUT_DIR}/${stamp}"
mkdir -p "${run_dir}"

echo "run-long-parallel: workers=${WORKERS} base_seed=${BASE_SEED} iterations=${ITERATIONS}" >&2
echo "run-long-parallel: total_requests=${TOTAL_ITERATIONS} output=${run_dir}" >&2

failed=0
monitor_pid=""

for ((w = 0; w < WORKERS; w++)); do
  seed=$((BASE_SEED + w))
  out_jsonl="${run_dir}/worker-${w}.jsonl"
  out_log="${run_dir}/worker-${w}.log"

  (
    export RUN_LONG_SEED="${seed}"
    export RUN_LONG_WORKER="${w}"
    exec "${run_long}"
  ) >"${out_jsonl}" 2>"${out_log}" &
  pid=$!
  pids+=("${pid}")
  echo "run-long-parallel: started worker=${w} pid=${pid} seed=${seed}" >&2

  if [[ "${w}" -lt $((WORKERS - 1)) ]] && [[ "${STAGGER_SEC}" != "0" ]]; then
    sleep "${STAGGER_SEC}"
  fi
done

if progress_enabled; then
  progress_monitor &
  monitor_pid=$!
fi

for ((w = 0; w < WORKERS; w++)); do
  pid="${pids[w]}"
  if wait "${pid}"; then
    echo "run-long-parallel: worker=${w} pid=${pid} ok" >&2
  else
    echo "run-long-parallel: worker=${w} pid=${pid} FAILED (see ${run_dir}/worker-${w}.log)" >&2
    failed=$((failed + 1))
  fi
done

if [[ -n "${monitor_pid}" ]]; then
  wait "${monitor_pid}" 2>/dev/null || true
  if progress_enabled; then
    draw_progress_bar "$(iterations_done)" "${TOTAL_ITERATIONS}"
    if [[ -t 2 ]]; then
      printf '\n' >&2
    fi
  fi
fi

echo "run-long-parallel: done failed=${failed}/${WORKERS} dir=${run_dir}" >&2
if [[ "${failed}" -gt 0 ]]; then
  exit 1
fi
