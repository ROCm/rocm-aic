#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# TTFT cold/warm benchmark orchestrator for llama-server.
#
# Phase 1 (Cold):
#   Start llama-server, send a long prompt, measure TTFT,
#   save the slot to disk, stop the server.
#
# Phase 2 (Warm):
#   Restart llama-server (cold GPU), restore the slot from
#   disk, send the IDENTICAL prompt, measure TTFT.
#
set -euo pipefail

# ── defaults ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-8B-GGUF}"
CTX_SIZE="${CTX_SIZE:-16384}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
CONTEXT_CHARS="${CONTEXT_CHARS:-40000}"
SEED="${SEED:-42}"
REPEATS="${REPEATS:-3}"
SLOT_SAVE_PATH="${SLOT_SAVE_PATH:-/slots}"
RESULTS="${RESULTS:-${APP_DIR}/results.jsonl}"
SERVER_URL="${SERVER_URL:-http://localhost:8080}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
SLOT_FILE="cache.bin"

export MODEL CTX_SIZE N_GPU_LAYERS SLOT_SAVE_PATH

# ── usage ───────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Environment variables (all optional):
  MODEL             Path to GGUF model file          [${MODEL}]
  CTX_SIZE          Context window size               [${CTX_SIZE}]
  N_GPU_LAYERS      Layers to offload to GPU          [${N_GPU_LAYERS}]
  CONTEXT_CHARS     Prompt size in characters          [${CONTEXT_CHARS}]
  SEED              Master PRNG seed                   [${SEED}]
  REPEATS           Measurements per phase             [${REPEATS}]
  SLOT_SAVE_PATH    Directory for slot save files      [${SLOT_SAVE_PATH}]
  SERVER_URL        llama-server base URL              [${SERVER_URL}]
  STARTUP_TIMEOUT   Max seconds to wait for server     [${STARTUP_TIMEOUT}]

Example:
  MODEL=/models/qwen3-8b-q4.gguf REPEATS=5 ./scripts/run-bench.sh
EOF
    exit 0
}
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

# ── helpers ─────────────────────────────────────────────────────
log() { echo "[bench] $(date +%H:%M:%S) $*"; }

start_server() {
    log "starting llama-server ..."
    "${SCRIPT_DIR}/serve.sh" &
    SERVER_PID=$!

    local elapsed=0
    while ! curl -sf "${SERVER_URL}/health" >/dev/null 2>&1; do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            log "ERROR: llama-server exited unexpectedly"
            return 1
        fi
        if (( elapsed >= STARTUP_TIMEOUT )); then
            log "ERROR: server did not become ready within ${STARTUP_TIMEOUT}s"
            kill "${SERVER_PID}" 2>/dev/null || true
            return 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    log "llama-server ready (PID ${SERVER_PID}, took ~${elapsed}s)"
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "stopping llama-server (PID ${SERVER_PID}) ..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        log "llama-server stopped"
    fi
    SERVER_PID=""
}

run_bench() {
    local tag="$1"
    shift
    python3 "${APP_DIR}/bench_ttft.py" \
        --server-url "${SERVER_URL}" \
        --corpus-file "${APP_DIR}/corpus.txt" \
        --context-chars "${CONTEXT_CHARS}" \
        --seed "${SEED}" \
        --output "${RESULTS}" \
        --tag "${tag}" \
        "$@"
}

cleanup() {
    stop_server
}
trap cleanup EXIT

# ── main ────────────────────────────────────────────────────────
log "=========================================="
log "  llama-server TTFT Benchmark"
log "=========================================="
log "  MODEL         = ${MODEL}"
log "  CTX_SIZE      = ${CTX_SIZE}"
log "  CONTEXT_CHARS = ${CONTEXT_CHARS}"
log "  REPEATS       = ${REPEATS}"
log "  SEED          = ${SEED}"
log "  SLOT_SAVE_PATH= ${SLOT_SAVE_PATH}"
log "  RESULTS       = ${RESULTS}"
log "=========================================="

rm -f "${RESULTS}"
mkdir -p "${SLOT_SAVE_PATH}"

# ── phase 1: cold runs ─────────────────────────────────────────
log "--- Phase 1: Cold Runs ---"

for rep in $(seq 1 "${REPEATS}"); do
    log "cold run ${rep}/${REPEATS}"

    start_server

    if [[ "${rep}" -eq 1 ]]; then
        run_bench "cold-${rep}" --save-slot "${SLOT_FILE}"
    else
        run_bench "cold-${rep}"
    fi

    stop_server
done

# ── phase 2: warm runs ─────────────────────────────────────────
log "--- Phase 2: Warm Runs (slot restore) ---"

for rep in $(seq 1 "${REPEATS}"); do
    log "warm run ${rep}/${REPEATS}"

    start_server
    run_bench "warm-${rep}" --restore-slot "${SLOT_FILE}"
    stop_server
done

# ── results summary ────────────────────────────────────────────
log "--- Results Summary ---"

python3 -c "
import json, sys
from collections import defaultdict

results_path = '${RESULTS}'
records = []
with open(results_path) as fh:
    for line in fh:
        line = line.strip()
        if line:
            records.append(json.loads(line))

cold = [r['ttft_ms'] for r in records if r['tag'].startswith('cold')]
warm = [r['ttft_ms'] for r in records if r['tag'].startswith('warm')]

def stats(vals):
    if not vals:
        return 'N/A'
    n = len(vals)
    mean = sum(vals) / n
    lo = min(vals)
    hi = max(vals)
    return f'n={n}  mean={mean:.1f}ms  min={lo:.1f}ms  max={hi:.1f}ms'

print()
print('=== TTFT Summary ===')
print(f'  Cold (no cache):     {stats(cold)}')
print(f'  Warm (slot restore): {stats(warm)}')
if cold and warm:
    speedup = (sum(cold)/len(cold)) / (sum(warm)/len(warm))
    saving = (sum(cold)/len(cold)) - (sum(warm)/len(warm))
    print(f'  Speedup:             {speedup:.1f}x  ({saving:.0f}ms saved)')
print()
print(f'Full results: {results_path}')
"

log "done."
