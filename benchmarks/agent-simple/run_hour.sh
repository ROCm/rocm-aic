#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# About an hour of stress: the stack is restarted with two LMCache L1 sizes and
# three load shapes are run against each.  After every phase it records
# container health, LMCache allocation failures, log errors, vLLM's lazy-offload
# counters and host swap.  It runs `make down` and `make up-dev`, so it takes
# the stack over for the duration.
#
#   AIC_STRESS_MAKE_ARGS='ROCM_ARCH=gfx1201 VLLM_MODEL=... NVME_DATA=...' \
#       run_hour.sh [<log-file>]
#
# AIC_STRESS_MAKE_ARGS: the make variables this node needs for `make up-dev`
#   (required; LMCACHE_L1_SIZE_GB is set per half and must not be in it).
# AIC_STRESS_L1_SIZES: the two L1 sizes in GiB (default "2 8").
# AIC_BENCH_ENDPOINT: passed on to run_stress.sh (default http://aic-lmcache:8000).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../.." && pwd)"
log="${1:-/tmp/aic-agent-simple-hour.log}"
read -r -a make_args <<< "${AIC_STRESS_MAKE_ARGS:?set AIC_STRESS_MAKE_ARGS to the make variables for make up-dev}"
read -r -a l1_sizes <<< "${AIC_STRESS_L1_SIZES:-2 8}"

say() { echo "$(date +%H:%M:%S) $*" | tee -a "${log}"; }

up() {
    say "PHASE stack restart with LMCACHE_L1_SIZE_GB=$1"
    make -C "${repo}" down "${make_args[@]}" "LMCACHE_L1_SIZE_GB=$1" >/dev/null 2>&1 || true
    make -C "${repo}" up-dev "${make_args[@]}" "LMCACHE_L1_SIZE_GB=$1" >/dev/null 2>&1 \
        || { say "ERROR make up-dev failed"; return 1; }
    for _ in $(seq 1 60); do
        [[ "$(docker inspect aic-vllm-gpu0 --format '{{.State.Health.Status}}' 2>/dev/null)" == healthy ]] && break
        sleep 5
    done
    say "stack: vllm=$(docker inspect aic-vllm-gpu0 --format '{{.State.Health.Status}}' 2>&1)"
}

count() { docker logs --since 15m "$1" 2>&1 | grep -ci -- "$2" || true; }

run() {  # run-id documents concurrency seconds max-words
    say "RUN $1 docs=$2 concurrency=$3 seconds=$4 max_words=$5"
    "${here}/run_stress.sh" "$@" 2>&1 | grep --line-buffered -v '^t+' | sed -u "s/^/  [$1] /" | tee -a "${log}" || true
    local lazy
    lazy="$(docker logs --since 15m aic-vllm-gpu0 2>&1 | grep -o 'Lazy offload counters.*' | tail -1 \
        | grep -o 'admitted=[0-9]* emitted=[0-9]*\|dropped_evicted=[0-9]*\|pending=[0-9]*' | tr '\n' ' ' || true)"
    say "HEALTH $1 vllm=$(docker inspect aic-vllm-gpu0 --format '{{.State.Status}}/{{.State.Health.Status}}' 2>&1)" \
        "lmcache=$(docker inspect aic-lmcache --format '{{.State.Status}}' 2>&1)" \
        "allocfail=$(count aic-lmcache 'Failed to batched allocate')" \
        "vllmErr=$(count aic-vllm-gpu0 'traceback\|out of memory\| ERROR ')" \
        "lmcErr=$(count aic-lmcache 'traceback\| ERROR')" \
        "lazy=${lazy}" \
        "swapUsedMB=$(free -m | awk '/Swap/{print $3}') memAvailGB=$(free -g | awk '/Mem/{print $7}')"
}

: > "${log}"
n=0
for size in "${l1_sizes[@]}"; do
    n=$((n + 1))
    up "${size}"
    run "h${n}a" 220 16 480 1300
    run "h${n}b" 300 32 480 1300
    run "h${n}c" 120 "$((n == 1 ? 8 : 24))" 420 2600
done
say "DONE"
