#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# TTFT sweep orchestrator.
#
# Phase 1 -- Warming:
#   Start vLLM+LMCache, send a long prompt to populate all cache
#   chunks on disk, stop vLLM, snapshot the cache directory.
#
# Phase 2 -- Sweep:
#   For each target hit rate N%, restore the snapshot, randomly
#   delete (100-N)% of the .pt chunk files, restart vLLM, send
#   the IDENTICAL prompt and measure TTFT.
#
set -euo pipefail

# ── defaults ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${LMCACHE_CONFIG_FILE:-${APP_DIR}/configs/lmcache-disk.yaml}"
HIT_RATES="${HIT_RATES:-0 25 50 75 100}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-10000}"
REPEATS="${REPEATS:-3}"
SEED="${SEED:-42}"
CACHE_DIR="${CACHE_DIR:-/cache/lmcache}"
RESULTS="${RESULTS:-${APP_DIR}/results.jsonl}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
SERVER_URL="${SERVER_URL:-http://localhost:8000}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-300}"

export MODEL SEED LMCACHE_CONFIG_FILE="${CONFIG}"

SNAPSHOT_DIR="${CACHE_DIR}.snapshot"

# ── usage ───────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Environment variables (all optional):
  LMCACHE_CONFIG_FILE  Path to LMCache YAML config  [${CONFIG}]
  HIT_RATES            Space-separated hit rates     [${HIT_RATES}]
  CONTEXT_TOKENS       Prompt length in tokens       [${CONTEXT_TOKENS}]
  REPEATS              Measurements per hit rate     [${REPEATS}]
  SEED                 Master PRNG seed              [${SEED}]
  CACHE_DIR            LMCache disk cache path       [${CACHE_DIR}]
  MODEL                HuggingFace model name        [${MODEL}]
  SERVER_URL           vLLM base URL                 [${SERVER_URL}]
  VLLM_STARTUP_TIMEOUT Max seconds to wait for vLLM  [${VLLM_STARTUP_TIMEOUT}]

Example:
  SEED=123 HIT_RATES="0 50 100" REPEATS=5 ./scripts/run-sweep.sh
EOF
    exit 0
}
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

# ── helpers ─────────────────────────────────────────────────────
log() { echo "[sweep] $(date +%H:%M:%S) $*"; }

start_vllm() {
    log "starting vLLM (model=${MODEL}) ..."
    "${SCRIPT_DIR}/serve.sh" &
    VLLM_PID=$!

    local elapsed=0
    while ! curl -sf "${SERVER_URL}/v1/models" >/dev/null 2>&1; do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            log "ERROR: vLLM process exited unexpectedly"
            return 1
        fi
        if (( elapsed >= VLLM_STARTUP_TIMEOUT )); then
            log "ERROR: vLLM did not become ready within ${VLLM_STARTUP_TIMEOUT}s"
            kill "${VLLM_PID}" 2>/dev/null || true
            return 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    log "vLLM ready (PID ${VLLM_PID}, took ~${elapsed}s)"
}

stop_vllm() {
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        log "stopping vLLM (PID ${VLLM_PID}) ..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
        log "vLLM stopped"
    fi
    VLLM_PID=""
}

run_bench() {
    local tag="$1"
    python3 "${APP_DIR}/bench_ttft.py" \
        --server-url "${SERVER_URL}" \
        --model "${MODEL}" \
        --corpus-file "${APP_DIR}/configs/books.txt" \
        --context-tokens "${CONTEXT_TOKENS}" \
        --seed "${SEED}" \
        --output "${RESULTS}" \
        --tag "${tag}"
}

snapshot_cache() {
    log "snapshotting ${CACHE_DIR} -> ${SNAPSHOT_DIR} ..."
    rm -rf "${SNAPSHOT_DIR}"
    cp -a "${CACHE_DIR}" "${SNAPSHOT_DIR}"
    local count
    count=$(find "${SNAPSHOT_DIR}" -name '*.pt' 2>/dev/null | wc -l)
    log "snapshot contains ${count} chunk files"
}

restore_cache() {
    log "restoring cache from snapshot ..."
    rm -rf "${CACHE_DIR}"
    cp -a "${SNAPSHOT_DIR}" "${CACHE_DIR}"
}

randomly_delete_chunks() {
    local hit_rate="$1"
    local delete_seed="$2"

    python3 -c "
import os, random, sys, json

cache_dir = '${CACHE_DIR}'
hit_rate  = int(${hit_rate})
seed      = int(${delete_seed})

files = sorted(f for f in os.listdir(cache_dir)
               if f.endswith('.pt'))
total = len(files)
if total == 0:
    print('[sweep] WARNING: no .pt files found in cache dir',
          file=sys.stderr)
    sys.exit(0)

keep_count = max(0, round(total * hit_rate / 100))
delete_count = total - keep_count

rng = random.Random(seed)
to_delete = rng.sample(files, delete_count)

deleted_paths = []
for f in to_delete:
    p = os.path.join(cache_dir, f)
    os.remove(p)
    deleted_paths.append(f)

meta_files = [f for f in os.listdir(cache_dir)
              if f.endswith('.meta')]
orphaned = [m for m in meta_files
            if m.replace('.meta', '.pt') not in set(files) - set(to_delete)]
for m in orphaned:
    os.remove(os.path.join(cache_dir, m))

manifest = {
    'hit_rate': hit_rate,
    'seed': seed,
    'chunks_total': total,
    'chunks_kept': keep_count,
    'chunks_deleted': delete_count,
    'deleted_files': sorted(deleted_paths),
}
print(json.dumps(manifest))
"
}

cleanup() {
    stop_vllm
    rm -rf "${SNAPSHOT_DIR}"
}
trap cleanup EXIT

# ── main ────────────────────────────────────────────────────────
log "=========================================="
log "  TTFT LMCache Sweep"
log "=========================================="
log "  CONFIG         = ${CONFIG}"
log "  MODEL          = ${MODEL}"
log "  CONTEXT_TOKENS = ${CONTEXT_TOKENS}"
log "  HIT_RATES      = ${HIT_RATES}"
log "  REPEATS        = ${REPEATS}"
log "  SEED           = ${SEED}"
log "  CACHE_DIR      = ${CACHE_DIR}"
log "  RESULTS        = ${RESULTS}"
log "=========================================="

rm -f "${RESULTS}"
mkdir -p "${CACHE_DIR}"

# ── phase 1: warming ───────────────────────────────────────────
log "--- Phase 1: Warming ---"
start_vllm
run_bench "warmup"
stop_vllm

snapshot_cache

# ── phase 2: sweep ─────────────────────────────────────────────
log "--- Phase 2: Sweep ---"

for N in ${HIT_RATES}; do
    log ">>> hit rate = ${N}%"

    for rep in $(seq 1 "${REPEATS}"); do
        log "  repeat ${rep}/${REPEATS}"

        restore_cache

        DELETE_SEED=$(( SEED + N + rep ))
        manifest=$(randomly_delete_chunks "${N}" "${DELETE_SEED}")
        log "  cache manipulation: ${manifest}"

        start_vllm
        run_bench "hit-${N}-rep-${rep}"
        stop_vllm
    done
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

groups = defaultdict(list)
for r in records:
    groups[r['tag']].append(r['ttft_ms'])

hdr = f\"{'tag':<20} {'count':>5} {'mean_ms':>10} {'min_ms':>10} {'max_ms':>10}\"
print(hdr)
print('-' * len(hdr))
for tag in sorted(groups.keys()):
    vals = groups[tag]
    n = len(vals)
    mean = sum(vals) / n
    lo = min(vals)
    hi = max(vals)
    print(f'{tag:<20} {n:>5} {mean:>10.1f} {lo:>10.1f} {hi:>10.1f}')

print()
print(f'Full results: {results_path}')
"

log "done."
