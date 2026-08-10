#!/bin/bash
#SBATCH --partition=amd-spur
#SBATCH --constraint=
#SBATCH --gres=
#SBATCH --time=01:30:00
#SBATCH --job-name=aic-ext-hit-test

set -euo pipefail
LOGDIR="/shared_nfs/stebates/logs/ext-hit-test-${SLURM_JOB_ID}"
mkdir -p "${LOGDIR}"
exec > >(tee -a "${LOGDIR}/sbatch.log") 2>&1

NODE=$(hostname)
echo "=== ext-hit diagnostic job ${SLURM_JOB_ID} on ${NODE} ==="
date

# --- Config ---
AIC_IMAGE=rocm-aic:7.14-latest
HF_HOME=/shared_nfs/stebates/hf
NVME_BASE=/mnt/m2m_nobackup/aic-cliff-exthit
NVME_DATA="${NVME_BASE}/nvme"
VLLM_MODEL="Qwen/Qwen2.5-3B-Instruct"
LMCACHE_PORT=6555
VLLM_PORT=8000
REPO=/home/stebates/Projects/rocm-aic

# --- Load image ---
echo "--- Loading image ---"
IMG_TAR=/shared_nfs/stebates/images/rocm-aic-7.14-latest-gfx950.tar.zst
if ! docker image inspect "${AIC_IMAGE}" >/dev/null 2>&1; then
    echo "Loading from tarball..."
    zstd -d -c "${IMG_TAR}" | docker load
fi
echo "Image ready: ${AIC_IMAGE}"

# --- Prep dirs ---
mkdir -p "${NVME_DATA}/lmcache" "${LOGDIR}"

# --- Cleanup stale containers ---
docker rm -f aic-lmcache aic-vllm-gpu0 2>/dev/null || true

# --- Start lmcache ---
echo "--- Starting lmcache (L1=1GB, pool=4096, POSIX) ---"
docker run -d --name aic-lmcache \
    --network host \
    --ipc shareable \
    --device /dev/kfd --device /dev/dri \
    --ulimit nofile=1048576:1048576 \
    -v "${NVME_DATA}:/data/nvme" \
    -e ROCR_VISIBLE_DEVICES=0 \
    -e PYTHONUNBUFFERED=1 \
    --entrypoint /bin/bash \
    "${AIC_IMAGE}" \
    -c "exec lmcache server \
        --host 0.0.0.0 \
        --port ${LMCACHE_PORT} \
        --l1-size-gb 1 \
        --eviction-policy LRU \
        --max-gpu-workers 1 \
        --worker-reap-timeout-seconds 3600 \
        --l2-prefetch-max-in-flight 32 \
        --l2-adapter '{\"type\":\"nixl_store\",\"backend\":\"POSIX\",\"backend_params\":{\"file_path\":\"/data/nvme/lmcache\",\"use_direct_io\":\"false\",\"file_size\":\"33554432\"},\"pool_size\":4096}'"

# Stream lmcache logs immediately
docker logs -f --timestamps aic-lmcache > "${LOGDIR}/lmcache.log" 2>&1 &
LMCACHE_LOG_PID=$!
echo "lmcache log streaming -> ${LOGDIR}/lmcache.log"

# Wait for lmcache ready
# LMCache ZMQ on LMCACHE_PORT, HTTP metrics on 8080
LMCACHE_HTTP_PORT=8080
echo "Waiting for lmcache HTTP on :${LMCACHE_HTTP_PORT}..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${LMCACHE_HTTP_PORT}/metrics" >/dev/null 2>&1; then
        echo "lmcache ready (${i}s)"; break
    fi
    [[ $i -eq 120 ]] && { echo "ERROR: lmcache never ready"; tail -30 "${LOGDIR}/lmcache.log"; exit 1; }
    sleep 1
done

# --- Start vLLM with DEBUG logging ---
echo "--- Starting vLLM (VLLM_LOGGING_LEVEL=DEBUG) ---"
KV_TRANSFER="{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.server_urls\":[\"tcp://localhost:${LMCACHE_PORT}\"]}}"

# --ipc: join lmcache's IPC ns (mirrors compose's ipc: service:lmcache).  torch's
# _share_cuda_ creates a /dev/shm/torch_* refcounter alongside each HIP IPC handle;
# with split IPC namespaces the consumer cannot see it, so those files and their
# ref counts leak for the life of the run.  Registration still succeeds either way
# (the refcounter is mapped only at teardown), so this is a leak fix, not a
# functional one -- but it restores parity with the compose stack.
docker run -d --name aic-vllm-gpu0 \
    --network host \
    --pid "container:aic-lmcache" \
    --ipc "container:aic-lmcache" \
    --device /dev/kfd --device /dev/dri \
    -v "${HF_HOME}:/hf" \
    -e ROCR_VISIBLE_DEVICES=0 \
    -e HF_HOME=/hf \
    -e HF_HUB_CACHE=/hf/hub \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_LOGGING_LEVEL=DEBUG \
    -e PYTHONUNBUFFERED=1 \
    -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:False \
    -e PYTORCH_ALLOC_CONF=expandable_segments:False \
    -e VLLM_TARGET_DEVICE=rocm \
    -e AMDGCN_USE_BUFFER_OPS=0 \
    "${AIC_IMAGE}" \
    --model "${VLLM_MODEL}" \
    --host 0.0.0.0 --port ${VLLM_PORT} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.40 \
    --max-model-len 32768 \
    --max-num-batched-tokens 4096 \
    --kv-transfer-config "${KV_TRANSFER}"

# Stream vLLM logs IMMEDIATELY
docker logs -f --timestamps aic-vllm-gpu0 > "${LOGDIR}/vllm.log" 2>&1 &
VLLM_LOG_PID=$!
echo "vLLM log streaming -> ${LOGDIR}/vllm.log"

# --- Wait for vLLM, print log tail every 60s ---
echo "Waiting for vLLM on :${VLLM_PORT} (up to 1200s)..."
VLLM_READY=0
for i in $(seq 1 240); do
    if curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
        VLLM_READY=1
        echo "vLLM READY after $((i*5))s"
        break
    fi
    if (( i % 12 == 0 )); then
        echo "--- vLLM log tail at $((i*5))s ---"
        tail -5 "${LOGDIR}/vllm.log" 2>/dev/null || true
        echo "---"
    fi
    sleep 5
done

if [[ ${VLLM_READY} -eq 0 ]]; then
    echo "ERROR: vLLM never became ready after 1200s"
    echo "=== Last 100 lines of vllm.log ==="
    tail -100 "${LOGDIR}/vllm.log"
    echo "=== Last 30 lines of lmcache.log ==="
    tail -30 "${LOGDIR}/lmcache.log"
    kill ${LMCACHE_LOG_PID} ${VLLM_LOG_PID} 2>/dev/null || true
    docker rm -f aic-lmcache aic-vllm-gpu0 2>/dev/null || true
    exit 1
fi

echo "=== Stack ready — running cliff: warmup-at-each-c + 1 timed iter ==="
# --warmup-at-each-c: warms c=1 first, then runs 1 timed iter at c=1
# This is the standard per_client flow — same as the real cliff run
docker run --rm --network host \
    -v "${REPO}:/work" -w /work \
    -v "${HF_HOME}:/hf" \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    --entrypoint python3 "${AIC_IMAGE}" \
    benchmarks/run_cliff.py \
        --endpoint "http://localhost:${VLLM_PORT}" \
        --model "${VLLM_MODEL}" \
        --arm kvd_v2 --isl 20000 --shared-prefix-tokens 18000 \
        --concurrencies 1 --iters 1 --warmup-at-each-c \
        --post-warmup-sleep-s 10 \
        --prefix-mode per_client --out "${LOGDIR}/results.csv" 2>&1 | tee "${LOGDIR}/cliff.log"

echo "--- Metrics snapshot (final state after all iters) ---"
# Capture baseline zeros for delta calculation
echo "" > "${LOGDIR}/metrics_before.txt"
echo "--- Metrics snapshot AFTER timed iter ---"
curl -s "http://localhost:${VLLM_PORT}/metrics" \
    | grep -E "external_prefix_cache|prefix_cache|connector" \
    | tee "${LOGDIR}/metrics_after.txt"

echo "=== KEY RESULTS ==="
echo "--- ext_hit counter delta ---"
python3 -c "
import re
def parse(f):
    r = {}
    try:
        for l in open(f):
            m = re.match(r'^([^\s{]+)[^\s]* ([\d.e+\-]+)', l.strip())
            if m: r[m.group(1)] = float(m.group(2))
    except: pass
    return r
b = parse('${LOGDIR}/metrics_before.txt')
a = parse('${LOGDIR}/metrics_after.txt')
for k in sorted(set(list(b)+list(a))):
    bv,av = b.get(k,0),a.get(k,0)
    print(f'  {k}: {bv:.0f} -> {av:.0f}  delta={av-bv:+.0f}')
"

echo "--- \"vLLM hit is\" log lines (from DEBUG connector) ---"
grep -i "vLLM hit is\|Need to load" "${LOGDIR}/vllm.log" | tail -20 \
    || echo "  (none found — check ${LOGDIR}/vllm.log)"

echo "--- lmcache L2 prefetch lines ---"
grep -i "prefetch\|l2_hit\|lookup.*found\|LOOKUP" "${LOGDIR}/lmcache.log" | tail -10 \
    || echo "  (none found)"

echo "=== Cleanup ==="
kill ${LMCACHE_LOG_PID} ${VLLM_LOG_PID} 2>/dev/null || true
docker rm -f aic-lmcache aic-vllm-gpu0 2>/dev/null || true
echo "Logs: ${LOGDIR}/"
