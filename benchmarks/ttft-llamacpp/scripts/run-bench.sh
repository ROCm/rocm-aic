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
CONTEXT_CHARS_LIST="${CONTEXT_CHARS_LIST:-400 4000 40000}"
SEED="${SEED:-42}"
REPEATS="${REPEATS:-3}"
SLOT_SAVE_PATH="${SLOT_SAVE_PATH:-/slots}"
RESULTS="${RESULTS:-${APP_DIR}/results.jsonl}"
SERVER_URL="${SERVER_URL:-http://localhost:8080}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
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
  CONTEXT_CHARS_LIST Space-separated char sizes         [${CONTEXT_CHARS_LIST}]
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

get_disk_stats() {
    local slot_dev
    slot_dev=$(df "${SLOT_SAVE_PATH}" 2>/dev/null | awk 'NR==2{print $1}')
    slot_dev=$(basename "${slot_dev}" 2>/dev/null)
    if [[ -f /proc/diskstats ]] && [[ -n "${slot_dev}" ]]; then
        awk -v dev="${slot_dev}" '$3 == dev {print $6, $10}' /proc/diskstats
    else
        echo "0 0"
    fi
}

record_disk_delta() {
    local label="$1" before_r="$2" before_w="$3" after_r="$4" after_w="$5"
    local delta_r=$(( after_r - before_r ))
    local delta_w=$(( after_w - before_w ))
    local delta_r_mib
    delta_r_mib=$(awk "BEGIN{printf \"%.2f\", ${delta_r} * 512 / 1048576}")
    local delta_w_mib
    delta_w_mib=$(awk "BEGIN{printf \"%.2f\", ${delta_w} * 512 / 1048576}")
    log "  disk IO (${label}): read=${delta_r_mib} MiB  write=${delta_w_mib} MiB"

    # patch the last line of the results JSONL with disk IO and startup fields
    if [[ -f "${RESULTS}" ]]; then
        python3 -c "
import json
path = '${RESULTS}'
lines = open(path).readlines()
if lines:
    rec = json.loads(lines[-1])
    rec['disk_read_mib'] = float('${delta_r_mib}')
    rec['disk_write_mib'] = float('${delta_w_mib}')
    rec['startup_health_ms'] = int('${LAST_STARTUP_HEALTH_MS}')
    rec['startup_warmup_ms'] = int('${LAST_STARTUP_WARMUP_MS}')
    rec['startup_total_ms'] = int('${LAST_STARTUP_TOTAL_MS}')
    lines[-1] = json.dumps(rec) + '\n'
    open(path, 'w').writelines(lines)
"
    fi
}

drop_slot_page_cache() {
    # evict slot files from page cache without affecting the model file
    # this keeps model loading fast while ensuring slot restores hit disk
    local slot_dir="$1"
    sync
    if command -v vmtouch >/dev/null 2>&1; then
        vmtouch -e "${slot_dir}" 2>/dev/null && log "  page cache evicted (vmtouch) for ${slot_dir}" && return
    fi
    if sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
        log "  page cache dropped (global)"
    else
        log "  WARNING: cannot drop page cache; disk results may reflect cached IO"
    fi
}

start_server() {
    log "starting llama-server ..."
    local t_start
    t_start=$(date +%s%3N)

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

    local t_healthy
    t_healthy=$(date +%s%3N)
    local health_ms=$(( t_healthy - t_start ))

    # send a throwaway prompt to force full model load and GPU warmup
    curl -sf "${SERVER_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"stream\":false,\"extra_body\":{\"id_slot\":0}}" \
        >/dev/null 2>&1 || true

    local t_warm
    t_warm=$(date +%s%3N)
    local warmup_ms=$(( t_warm - t_healthy ))
    local total_ms=$(( t_warm - t_start ))

    LAST_STARTUP_HEALTH_MS="${health_ms}"
    LAST_STARTUP_WARMUP_MS="${warmup_ms}"
    LAST_STARTUP_TOTAL_MS="${total_ms}"

    log "llama-server ready (PID ${SERVER_PID})"
    log "  startup: health=${health_ms}ms  warmup=${warmup_ms}ms  total=${total_ms}ms"
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "stopping llama-server (PID ${SERVER_PID}) ..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""

    local port="${LLAMA_PORT:-8080}"
    local tries=0
    while curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; do
        if (( tries++ > 15 )); then
            log "WARNING: port ${port} still responding after 15s"
            break
        fi
        sleep 1
    done
    log "llama-server stopped (port ${port} free)"
}

run_bench() {
    local tag="$1"
    local chars="$2"
    shift 2
    python3 "${APP_DIR}/bench_ttft.py" \
        --server-url "${SERVER_URL}" \
        --corpus-file "${APP_DIR}/corpus.txt" \
        --context-chars "${chars}" \
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
log "  MODEL             = ${MODEL}"
log "  CTX_SIZE          = ${CTX_SIZE}"
log "  CONTEXT_CHARS_LIST= ${CONTEXT_CHARS_LIST}"
log "  REPEATS           = ${REPEATS}"
log "  SEED              = ${SEED}"
log "  SLOT_SAVE_PATH    = ${SLOT_SAVE_PATH}"
log "  RESULTS           = ${RESULTS}"
log "=========================================="

rm -f "${RESULTS}"
mkdir -p "${SLOT_SAVE_PATH}"

SLOT_DISK_PATH="${SLOT_DISK_PATH:-/tmp/ttft-slots-disk}"
SLOT_TMPFS_PATH="${SLOT_TMPFS_PATH:-/dev/shm/ttft-slots-tmpfs}"
SLOT_SAVE_PATH_ORIG="${SLOT_SAVE_PATH}"

copy_slot_to() {
    local src_file="$1"
    local dest_dir="$2"
    mkdir -p "${dest_dir}"
    cp "${src_file}" "${dest_dir}/$(basename "${src_file}")"
    log "  copied slot to ${dest_dir} ($(du -h "${src_file}" | cut -f1))"
}

for CCHARS in ${CONTEXT_CHARS_LIST}; do
    SLOT_FILE="cache-${CCHARS}.bin"

    log "================================================"
    log "  Context size: ~${CCHARS} chars"
    log "================================================"

    # ── cold runs ───────────────────────────────────────
    log "--- Cold Runs (${CCHARS} chars) ---"

    for rep in $(seq 1 "${REPEATS}"); do
        log "cold run ${rep}/${REPEATS} (${CCHARS} chars)"

        # purge all cached state so prefill is truly from scratch
        rm -rf "${SLOT_SAVE_PATH_ORIG:?}"/*
        rm -rf "${SLOT_TMPFS_PATH:?}"/* 2>/dev/null || true
        rm -rf "${SLOT_DISK_PATH:?}"/* 2>/dev/null || true

        export SLOT_SAVE_PATH="${SLOT_SAVE_PATH_ORIG}"
        start_server

        read -r before_r before_w <<< "$(get_disk_stats)"
        run_bench "cold-${CCHARS}c-${rep}" "${CCHARS}" --save-slot "${SLOT_FILE}"
        read -r after_r after_w <<< "$(get_disk_stats)"
        record_disk_delta "cold-${CCHARS}c-${rep}" "$before_r" "$before_w" "$after_r" "$after_w"

        stop_server
    done

    # ── warm runs: tmpfs ────────────────────────────────
    log "--- Warm Runs: tmpfs (${CCHARS} chars) ---"

    copy_slot_to "${SLOT_SAVE_PATH_ORIG}/${SLOT_FILE}" "${SLOT_TMPFS_PATH}"

    for rep in $(seq 1 "${REPEATS}"); do
        log "tmpfs warm run ${rep}/${REPEATS} (${CCHARS} chars)"

        export SLOT_SAVE_PATH="${SLOT_TMPFS_PATH}"
        start_server

        read -r before_r before_w <<< "$(get_disk_stats)"
        run_bench "warm-tmpfs-${CCHARS}c-${rep}" "${CCHARS}" --restore-slot "${SLOT_FILE}"
        read -r after_r after_w <<< "$(get_disk_stats)"
        record_disk_delta "warm-tmpfs-${CCHARS}c-${rep}" "$before_r" "$before_w" "$after_r" "$after_w"

        stop_server
    done

    # ── warm runs: disk ─────────────────────────────────
    log "--- Warm Runs: disk (${CCHARS} chars) ---"

    copy_slot_to "${SLOT_SAVE_PATH_ORIG}/${SLOT_FILE}" "${SLOT_DISK_PATH}"

    for rep in $(seq 1 "${REPEATS}"); do
        log "disk warm run ${rep}/${REPEATS} (${CCHARS} chars)"

        drop_slot_page_cache "${SLOT_DISK_PATH}"

        export SLOT_SAVE_PATH="${SLOT_DISK_PATH}"
        start_server

        read -r before_r before_w <<< "$(get_disk_stats)"
        run_bench "warm-disk-${CCHARS}c-${rep}" "${CCHARS}" --restore-slot "${SLOT_FILE}"
        read -r after_r after_w <<< "$(get_disk_stats)"
        record_disk_delta "warm-disk-${CCHARS}c-${rep}" "$before_r" "$before_w" "$after_r" "$after_w"

        stop_server
    done
done

# ── results summary ────────────────────────────────────────────
log "--- Results Summary ---"

python3 -c "
import json, re

results_path = '${RESULTS}'
records = []
with open(results_path) as fh:
    for line in fh:
        line = line.strip()
        if line:
            records.append(json.loads(line))

sizes = sorted(set(
    m.group(1) for r in records
    if (m := re.search(r'(\d+)c-', r['tag']))
))

def mean(v):
    return sum(v)/len(v) if v else 0

hdr = (f\"{'ctx_chars':<10} {'phase':<14} {'n':>3} {'mean_ms':>10}\"
       f\" {'min_ms':>10} {'max_ms':>10} {'speedup':>8}\"
       f\" {'rd_MiB':>8} {'wr_MiB':>8}\"
       f\" {'load_ms':>9}\")
print()
print(hdr)
print('-' * len(hdr))

for sz in sizes:
    cold_r  = [r for r in records if f'cold-{sz}c-'  in r['tag']]
    tmpfs_r = [r for r in records if f'tmpfs-{sz}c-' in r['tag']]
    disk_r  = [r for r in records if f'disk-{sz}c-'  in r['tag']]

    cold_mean = mean([r['ttft_ms'] for r in cold_r])

    for label, recs in [('cold', cold_r), ('warm-tmpfs', tmpfs_r), ('warm-disk', disk_r)]:
        if not recs:
            continue
        vals = [r['ttft_ms'] for r in recs]
        m = mean(vals)
        sp = f'{cold_mean/m:.1f}x' if cold_mean > 0 and m > 0 and label != 'cold' else ''
        rd = mean([r.get('disk_read_mib', 0) for r in recs])
        wr = mean([r.get('disk_write_mib', 0) for r in recs])
        ld = mean([r.get('startup_total_ms', 0) for r in recs])
        print(f'{sz:<10} {label:<14} {len(vals):>3} {m:>10.1f}'
              f' {min(vals):>10.1f} {max(vals):>10.1f} {sp:>8}'
              f' {rd:>8.1f} {wr:>8.1f}'
              f' {ld:>9.0f}')

print()
print(f'Full results: {results_path}')
"

log "done."
