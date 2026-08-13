#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Which L2 metric names does each backend actually emit?
#
# benchmarks/vllm_reset_test.py reads L2 occupancy from
#   lmcache_mp_l2_usage_bytes{l2_name="nixl_store"}
# -- an l2_name hardcoded to the NIXL storage backend.  Under AIC_L2_BACKEND=local_disk
# the gate reported "L2 stored 0 chunks" while the flood was plainly evicting, which
# is the signature of reading a label that backend never publishes rather than of a
# backend that never stores.  This brings each backend up, sends enough traffic to
# force a store, and dumps every `l2` metric line so the two cases can be told apart.
#
# Env: AIC_DAY_DIR (required), L2_BACKENDS (default: local_disk nixl_posix)
set -uo pipefail
: "${AIC_DAY_DIR:?}"
cd "${AIC_DAY_DIR}" || exit 1

L2_BACKENDS="${L2_BACKENDS:-local_disk nixl_posix}"
export HF_HOME="${HF_HOME:-/projects/hf_cache}"
export IMAGE_REF="${AIC_IMAGE:?}" IMAGE_NAME="${AIC_IMAGE%:*}" GPU=0
export AIC_VISIBLE_DEVICES=4 TENSOR_PARALLEL_SIZE=1
export VLLM_MODEL="Qwen/Qwen2.5-3B-Instruct"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LMCACHE_L1_SIZE_GB=1
export VLLM_IPC_MODE=service:lmcache VLLM_PID_MODE=service:lmcache
export KV_TRANSFER_ARG="--kv-transfer-config '{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://aic-lmcache\",\"lmcache.mp.port\":6555}}'"
PROM_UID="$(id -u)"; PROM_GID="$(id -g)"; export PROM_UID PROM_GID

# shellcheck source=/dev/null
source "${AIC_DAY_DIR}/monitoring/monitoring-lib.sh"; ensure_compose || exit 1
COMPOSE_FILE="${AIC_DAY_DIR}/docker/docker-compose.yml"
compose() { docker compose -f "${COMPOSE_FILE}" "$@"; }

NV=""
for d in /mnt/nixl-nvme-0 /mnt/nixl-nvme-* /mnt/lmcache-nvme; do
    [ -d "$d" ] && [ -w "$d" ] && { NV="$d"; break; }
done
[ -n "${NV}" ] || { echo "no writable NVMe mount" >&2; exit 1; }

OUT="${AIC_DAY_DIR}/logs/bench/l2-metric-names.txt"
mkdir -p "$(dirname "${OUT}")"; : > "${OUT}"

for be in ${L2_BACKENDS}; do
    pool="${NV}/aic-l2names-${SLURM_JOB_ID:-$$}-${be}"
    mkdir -p "${pool}" "${pool}-nfs"
    export NVME_DATA="${pool}" NFS_DATA="${pool}-nfs" GDS_SLAB_DATA=""
    export AIC_L2_BACKEND="${be}"
    export LOG="${AIC_DAY_DIR}/logs/bench/l2-names-${be}"
    export AIC_METRICS_DIR="${LOG}/prometheus"
    mkdir -p "${LOG}/vllm" "${AIC_METRICS_DIR}"
    _compat=false; [ "${be}" = "nixl" ] && _compat=true
    export HIPFILE_ALLOW_COMPAT_MODE="${_compat}"

    echo "=== ${be} $(date -Is)" | tee -a "${OUT}"
    if ! compose --profile cache up -d > "${LOG}/compose-up.log" 2>&1; then
        echo "  up_failed:" | tee -a "${OUT}"
        tail -15 "${LOG}/compose-up.log" | sed 's/^/    /' | tee -a "${OUT}"
    else
        ready=0
        for _ in $(seq 1 180); do
            docker exec aic-vllm-gpu0 curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1 \
                && { ready=1; break; }
            sleep 5
        done
        if [ "${ready}" != "1" ]; then
            echo "  not_ready" | tee -a "${OUT}"
        else
            # ~120 chunks per prompt-set; 40 x 4k-token prompts overflows a 1 GiB L1
            # several times over, so anything that stores to L2 has had to.
            for i in $(seq 1 40); do
                docker exec aic-vllm-gpu0 curl -fsS http://127.0.0.1:8000/v1/chat/completions \
                    -H 'Content-Type: application/json' \
                    -d "{\"model\":\"Qwen/Qwen2.5-3B-Instruct\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"run${i} $(head -c 12000 /dev/urandom | base64 | tr -d '\n' | head -c 12000)\"}]}" \
                    >/dev/null 2>&1
            done
            sleep 5
            echo "  --- lmcache metrics matching l2/usage/store ---" | tee -a "${OUT}"
            docker exec aic-lmcache curl -fsS http://127.0.0.1:8080/metrics 2>/dev/null \
                | grep -iE '^lmcache.*(l2|usage|store|disk|evict)' | grep -v '^#' \
                | sed 's/^/    /' | tee -a "${OUT}"
        fi
    fi

    # Every docker call here is wrapped in `timeout`.  Unwrapped, `compose down`
    # hung for 25 min on the vllm container in the first run of this script and had
    # to be cleared by hand from a second srun -- `--timeout 5` bounds how long
    # compose waits for the container, not how long compose itself takes.
    timeout 120 docker compose -f "${COMPOSE_FILE}" --profile cache down \
        --remove-orphans --timeout 5 >/dev/null 2>&1 || true
    for c in aic-vllm-gpu0 aic-lmcache aic-lmcache-coordinator aic-client; do
        timeout 40 docker rm -f "$c" >/dev/null 2>&1 || true
    done
    echo "  pool on disk: $(du -sh "${pool}" 2>/dev/null | cut -f1)" | tee -a "${OUT}"
    rm -rf "${pool}" "${pool}-nfs" 2>/dev/null || true
    sleep 5
done

echo "=== done"; cat "${OUT}"
