#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# overnight-test.sh — unattended overnight test run on gfx1151 / WSL2 / librocmdxg
#
# What it does:
#  Phase 0  — environment check + mkdir
#  Phase 1  — kick off gfx1151 image build in the background (bake target: lmcache)
#  Phase 2  — start Prometheus (standalone monitoring compose)
#  Phase 3  — download Qwen3-8B model weights if needed
#  Phase 4  — run test matrix against existing rocm-aic:latest (local_disk L2)
#               concurrencies: 1, 2, 4, 8, 16
#               ISL variants: 512, 2048, 8192
#  Phase 5  — wait for build; if successful re-run test matrix against new image
#  Phase 6  — dump prometheus TSDB snapshot + write summary to overnight MD file
#
# WSL2 / librocmdxg device args used throughout:
#   --device /dev/dxg
#   -v /usr/lib/wsl:/usr/lib/wsl:ro
#   -v /opt/rocm-7.2.4/lib:/opt/rocm-host-lib:ro
#   -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:/opt/rocm-host-lib:/opt/rocm/lib

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${REPO_ROOT}/logs/overnight-2026-08-12"
MDFILE="${REPO_ROOT}/overnight.2026-08-12.md"
POSIX_L2_DIR="${LOGDIR}/lmcache-posix-l2"
PROM_TSDB="${LOGDIR}/prometheus"
RESULTSDIR="${LOGDIR}/results"

MODEL="Qwen/Qwen3-8B"
LMCACHE_PORT=6555
VLLM_PORT=8000
EXISTING_IMAGE="rocm-aic:latest"
NEW_IMAGE="aic-lmcache:latest"

# WSL2 GPU device args
WSL2_STUB_DIR="/tmp/aic-wsl2-stubs"
GPU_ARGS=(
    --device /dev/dxg
    -v /usr/lib/wsl:/usr/lib/wsl:ro
    -v /opt/rocm-7.2.4/lib:/opt/rocm-host-lib:ro
    -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:/opt/rocm-host-lib:/opt/rocm/lib
    -e LD_PRELOAD=/usr/lib/wsl/lib/libdxcore.so:/opt/rocm-host-lib/librocdxg.so
    # PYTHONPATH stub: fake amdsmi.py so vLLM ROCm platform detection works in all subprocesses
    -v "${WSL2_STUB_DIR}:/wsl2-stubs:ro"
    -e PYTHONPATH=/wsl2-stubs
    # Required for cross-container HIP IPC KV tensor registration
    -e HSA_ENABLE_IPC_MODE_LEGACY=1
)

# Test matrix
CONCURRENCIES=(1 2 4 8 16)
ISLS=(512 2048 8192)

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOGDIR}/overnight.log"; }
md_append() { echo "$*" >> "${MDFILE}"; }

# --------------------------------------------------------------------------
# Phase 0: setup
# --------------------------------------------------------------------------
mkdir -p "${POSIX_L2_DIR}" "${PROM_TSDB}" "${RESULTSDIR}"
log "=== Overnight test run START ==="
log "LOGDIR: ${LOGDIR}"
log "MODEL: ${MODEL}"
log "GPU: AMD Radeon 8060S (gfx1151) via WSL2 librocmdxg"

md_append ""
md_append "---"
md_append "## Overnight test run — $(date '+%Y-%m-%d %H:%M:%S')"
md_append ""
md_append "### Environment"
md_append "- GPU: AMD Radeon 8060S (gfx1151), WSL2 librocmdxg"
md_append "- VRAM: ~79.7 GB shared"
md_append "- Model: ${MODEL}"
md_append "- L1: CPU DRAM"
md_append "- L2 Phase4 (existing image): lmcache 'fs' adapter (CPU-staged file writes — NIXL not in existing image)"
md_append "- L2 Phase5 (new build): NIXL POSIX (nixl_store POSIX backend, if build succeeds)"
md_append "- Concurrencies: ${CONCURRENCIES[*]}"
md_append "- ISL variants: ${ISLS[*]}"

# --------------------------------------------------------------------------
# Phase 1: kick off gfx1151 image build in background
# --------------------------------------------------------------------------
log "Phase 1: Starting gfx1151 image build in background..."
md_append ""
md_append "### Phase 1: Image build"
md_append "- Started: $(date '+%H:%M:%S')"
md_append "- Target: aic-lmcache (Dockerfile.lmcache, ROCM_ARCH=gfx1151)"

BUILD_LOG="${LOGDIR}/build-gfx1151.log"
(
    set +e
    cd "${REPO_ROOT}"
    # Build only lmcache (which includes NIXL) for gfx1151
    # Use existing rocm-aic:latest as base to skip the torch download
    WHEELS_DIR="/opt/pip-cache/wheels"
    [ -d "${WHEELS_DIR}" ] || WHEELS_DIR="/tmp/aic-pip-wheels-empty"
    mkdir -p "${WHEELS_DIR}"
    # Step 1: build aic-base (needed by nixl and lmcache as a local image)
    log "  Building aic-base for gfx1151..."
    AIC_VERSION="$(<VERSION)" \
    ROCM_ARCH=gfx1151 \
    BUILD_JOBS=4 \
    BUILDX_BAKE_ENTITLEMENTS_FS=0 \
    docker buildx bake \
        --allow fs.read=/dev/null \
        -f docker/docker-bake.hcl \
        --load \
        --set "base.contexts.pip-wheels=${WHEELS_DIR}" \
        base \
        >> "${BUILD_LOG}" 2>&1 || { echo "base build failed"; exit 1; }
    # Step 2: build aic-lmcache (depends on aic-base + aic-nixl, both now local)
    log "  Building aic-lmcache for gfx1151..."
    AIC_VERSION="$(<VERSION)" \
    ROCM_ARCH=gfx1151 \
    BUILD_JOBS=4 \
    BUILDX_BAKE_ENTITLEMENTS_FS=0 \
    docker buildx bake \
        --allow fs.read=/dev/null \
        -f docker/docker-bake.hcl \
        --load \
        --set "base.contexts.pip-wheels=${WHEELS_DIR}" \
        --set "lmcache.contexts.pip-wheels=${WHEELS_DIR}" \
        nixl lmcache \
        >> "${BUILD_LOG}" 2>&1
    RC=$?
    if [ $RC -eq 0 ]; then
        echo "BUILD_SUCCESS" > "${LOGDIR}/build.status"
        log "Phase 1: Build SUCCEEDED — aic-lmcache:latest ready"
    else
        echo "BUILD_FAILED rc=${RC}" > "${LOGDIR}/build.status"
        log "Phase 1: Build FAILED (rc=${RC}) — see ${BUILD_LOG}"
    fi
) &
BUILD_PID=$!
log "Phase 1: Build running in background (pid ${BUILD_PID})"

# --------------------------------------------------------------------------
# Phase 2: start Prometheus
# --------------------------------------------------------------------------
log "Phase 2: Starting Prometheus..."
docker rm -f aic-prometheus-overnight 2>/dev/null || true

PROM_UID="$(id -u)"
PROM_GID="$(id -g)"
# Create prometheus.yml for local scraping of lmcache + vllm
cat > "${LOGDIR}/prometheus.yml" <<'PROMYML'
global:
  scrape_interval: 15s
  scrape_timeout: 10s

scrape_configs:
  - job_name: vllm
    static_configs:
      - targets: ['host-gateway:8000']
        labels: {instance: 'vllm-gfx1151'}
    metrics_path: /metrics

  - job_name: lmcache
    static_configs:
      - targets: ['host-gateway:8080']
        labels: {instance: 'lmcache-gfx1151'}
    metrics_path: /metrics

  - job_name: overnight_tests
    static_configs:
      - targets: ['host-gateway:9091']
        labels: {instance: 'test-runner'}
    metrics_path: /metrics
PROMYML

docker run -d \
    --name aic-prometheus-overnight \
    --add-host host-gateway:host-gateway \
    -u "${PROM_UID}:${PROM_GID}" \
    -v "${LOGDIR}/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
    -v "${PROM_TSDB}:/prometheus" \
    -p 9292:9090 \
    prom/prometheus:v3.13.2 \
    --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/prometheus \
    --storage.tsdb.retention.time=7d \
    --web.enable-lifecycle \
    --web.listen-address=:9090

log "Phase 2: Prometheus running at http://localhost:9292"
md_append "- Prometheus: http://localhost:9292 (TSDB: ${PROM_TSDB})"

# --------------------------------------------------------------------------
# Phase 3: download model if needed
# --------------------------------------------------------------------------
log "Phase 3: Checking model ${MODEL}..."

HF_HOME="${HOME}/.cache/huggingface"

# Check if model weights are already present
WEIGHT_COUNT=$(find "${HF_HOME}/hub/models--Qwen--Qwen3-8B" -name "*.safetensors" 2>/dev/null | wc -l || echo 0)
if [ "${WEIGHT_COUNT}" -gt 0 ]; then
    log "Phase 3: Model already present (${WEIGHT_COUNT} safetensors files)"
else
    log "Phase 3: Downloading ${MODEL} weights on host (ZScaler SSL cert applied)..."
    md_append "- Model download: started $(date '+%H:%M:%S')"

    # Download on the host using the system CA bundle (ZScaler cert included)
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    /tmp/aic-bench-venv/bin/python3 -c "
import os
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
from huggingface_hub import snapshot_download
print('Downloading ${MODEL}...')
path = snapshot_download('${MODEL}', ignore_patterns=['*.gguf', '*.pt', 'original/'])
print('Downloaded to:', path)
import glob
w = glob.glob(path + '/*.safetensors')
print(f'{len(w)} weight shards')
" 2>&1 | tee "${LOGDIR}/model-download.log"

    WEIGHT_COUNT=$(find "${HF_HOME}/hub/models--Qwen--Qwen3-8B" -name "*.safetensors" 2>/dev/null | wc -l || echo 0)
    log "Phase 3: Download complete (${WEIGHT_COUNT} safetensors files)"
    md_append "- Model download: complete $(date '+%H:%M:%S'), ${WEIGHT_COUNT} shards"
fi

# --------------------------------------------------------------------------
# Helper: start_stack <image> <run_label> <l2_dir> [l1_gb]
# WSL2: uses LMCacheConnectorV1 (in-process) rather than LMCacheMPConnector
# because HIP IPC (hipIpcGetMemHandle) requires /dev/kfd which WSL2 lacks.
# LMCacheConnectorV1 runs inside vLLM's process — same GPU access, no IPC.
# L1 = GPU VRAM prefix cache (via enable-prefix-caching)
# L2 = local disk via lmcache local_disk_path config
# --------------------------------------------------------------------------
start_stack() {
    local image="$1"
    local label="$2"
    local l2dir="$3"
    local lmcache_l1_gb="${4:-20}"  # unused (L1 is VRAM via gpu-memory-utilization)

    log "  start_stack: image=${image} label=${label} L2=${l2dir} (LMCacheConnectorV1 in-process)"
    mkdir -p "${l2dir}"

    docker rm -f aic-vllm-test 2>/dev/null || true

    # Write LMCache single-process config with local disk L2
    local lmc_cfg="${LOGDIR}/lmcache-${label}.yaml"
    cat > "${lmc_cfg}" << YAMLEOF
chunk_size: 256
local_cpu: false
max_local_cpu_size: 0
local_disk: true
max_local_disk_size: 50
local_disk_path: /data/l2
remote_url: ~
remote_serde: ~
save_decode_cache: false
YAMLEOF

    local lmc_kv_json="{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache_config_file\":\"/lmcache-config.yaml\"}}"

    docker run -d \
        --name aic-vllm-test \
        "${GPU_ARGS[@]}" \
        -v "${HF_HOME}:/hf:ro" \
        -v "${REPO_ROOT}/scripts/vllm-wsl2-wrapper.py:/vllm-wrapper.py:ro" \
        -v "${lmc_cfg}:/lmcache-config.yaml:ro" \
        -v "${l2dir}:/data/l2" \
        -e HF_HOME=/hf \
        -e HF_HUB_CACHE=/hf/hub \
        -e HF_HUB_OFFLINE=1 \
        -e VLLM_TARGET_DEVICE=rocm \
        -e ROCR_VISIBLE_DEVICES=0 \
        -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:False \
        -e VLLM_ROCM_USE_AITER=1 \
        -e AMDGCN_USE_BUFFER_OPS=0 \
        -e MIOPEN_FIND_MODE=FAST \
        -e PYTHONHASHSEED=0 \
        -e VLLM_LOGGING_LEVEL=INFO \
        -e VLLM_SERVER_DEV_MODE=1 \
        -e LMCACHE_CONFIG_FILE=/lmcache-config.yaml \
        --entrypoint python3 \
        -p "${VLLM_PORT}:8000" \
        -p 8080:8080 \
        "${image}" \
        /vllm-wrapper.py \
        --model "${MODEL}" \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.50 \
        --max-model-len 16384 \
        --max-num-batched-tokens 4096 \
        --load-format safetensors \
        --block-size 64 \
        --enable-prefix-caching \
        --kv-cache-dtype auto \
        --attention-backend TRITON_ATTN \
        --stream-interval 20 \
        --async-scheduling \
        --kv-transfer-config "${lmc_kv_json}" \
        > "${LOGDIR}/vllm-${label}.log" 2>&1

    log "  vLLM+LMCache starting (model load ~4 min)..."
    local vretries=120
    while [ $vretries -gt 0 ]; do
        local ok
        ok=$(python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:${VLLM_PORT}/health', timeout=3); print('ok')" 2>/dev/null || echo "")
        if [ "${ok}" = "ok" ]; then
            log "  vLLM+LMCache healthy"
            return 0
        fi
        sleep 5
        vretries=$((vretries - 1))
    done
    log "  ERROR: vLLM+LMCache failed to start."
    docker logs --tail 30 aic-vllm-test 2>/dev/null || true
    return 1
}

stop_stack() {
    docker rm -f aic-vllm-test 2>/dev/null || true
    sleep 5
}

# --------------------------------------------------------------------------
# Helper: run_test_suite <image> <phase_label> <l2_dir>
# Runs the full concurrency × ISL matrix, writes CSV results.
# --------------------------------------------------------------------------
run_test_suite() {
    local image="$1"
    local phase_label="$2"
    local l2dir="$3"
    local suite_results="${RESULTSDIR}/${phase_label}"
    mkdir -p "${suite_results}"

    log "=== Test suite: ${phase_label} ==="
    md_append ""
    md_append "### Test suite: ${phase_label}"
    md_append "| ISL | shared_toks | concurrency | lat_p50_ms | lat_p95_ms | tput_tok/s | ext_hit_% | l1_hit_% |"
    md_append "|-----|-------------|-------------|------------|------------|------------|-----------|----------|"

    for isl in "${ISLS[@]}"; do
        local shared_toks=$(( isl * 7 / 8 ))

        for concur in "${CONCURRENCIES[@]}"; do
            local run_label="${phase_label}-isl${isl}-c${concur}"
            local out_csv="${suite_results}/cliff-${run_label}.csv"

            log "  run: ISL=${isl} shared=${shared_toks} c=${concur}"

            # Warmup pass (1 iter, not recorded)
            curl -sf -X POST "http://localhost:${VLLM_PORT}/reset_prefix_cache" \
                -H 'Content-Type: application/json' -d '{}' > /dev/null 2>&1 || true

            # Run cliff benchmark
            /tmp/aic-bench-venv/bin/python3 "${REPO_ROOT}/benchmarks/run_cliff.py" \
                --endpoint "http://localhost:${VLLM_PORT}" \
                --model "${MODEL}" \
                --arm kvd_v2 \
                --isl "${isl}" \
                --shared-prefix-tokens "${shared_toks}" \
                --concurrencies "${concur}" \
                --iters 3 \
                --warmup-iters 1 \
                --post-warmup-sleep-s 5 \
                --out "${out_csv}" \
                >> "${LOGDIR}/bench-${run_label}.log" 2>&1 || {
                    log "  WARNING: bench run failed for ${run_label}"
                    continue
                }

            # Extract key metrics from CSV
            if [ -f "${out_csv}" ]; then
                local metrics
                metrics=$(/tmp/aic-bench-venv/bin/python3 -c "
import csv, sys
with open('${out_csv}') as f:
    rows = list(csv.DictReader(f))
timed = [r for r in rows if r.get('iter','0') != '0']
if not timed:
    timed = rows
if not timed:
    print('no_data')
    sys.exit(0)
import statistics
lat50 = round(statistics.median(float(r['p50_latency_s'])*1000 for r in timed), 1)
lat95 = round(statistics.median(float(r['p95_latency_s'])*1000 for r in timed), 1)
tput  = round(statistics.mean(float(r['throughput_tok_s_total']) for r in timed), 1)
hits  = round(statistics.mean(float(r['ext_hit_rate_pct']) for r in timed), 1)
l1    = round(statistics.mean(float(r['l1_hit_rate_pct']) for r in timed), 1)
print(f'{lat50},{lat95},{tput},{hits},{l1}')
" 2>/dev/null || echo "?,?,?,?,?")
                IFS=',' read -r lat50 lat95 tput hits l1hit <<< "${metrics}"
                log "  RESULT ISL=${isl} c=${concur}: lat_p50=${lat50}ms lat_p95=${lat95}ms tput=${tput}tok/s ext_hit=${hits}% l1_hit=${l1hit}%"
                md_append "| ${isl} | ${shared_toks} | ${concur} | ${lat50} | ${lat95} | ${tput} | ${hits} | ${l1hit} |"
            fi

            sleep 3
        done

        # Between ISL groups: clear prefix cache and L2
        curl -sf -X POST "http://localhost:${VLLM_PORT}/reset_prefix_cache" \
            -H 'Content-Type: application/json' -d '{}' > /dev/null 2>&1 || true
        # LMCacheConnectorV1: cache lives in-process, cleared by prefix cache reset above
        sleep 5
    done

    log "=== Test suite ${phase_label} complete ==="
    md_append ""
    md_append "Suite complete: $(date '+%H:%M:%S')"
}

# --------------------------------------------------------------------------
# Phase 4: Test matrix against existing rocm-aic:latest
# --------------------------------------------------------------------------
log "Phase 4: Starting test matrix (existing image: ${EXISTING_IMAGE})"
md_append ""
md_append "### Phase 4: Test matrix — existing image (${EXISTING_IMAGE}, lmcache 'fs' L2)"
md_append "Note: existing image lacks libnixl.so; using lmcache fs adapter (CPU-staged writes)"
md_append "- Started: $(date '+%H:%M:%S')"

if start_stack "${EXISTING_IMAGE}" "existing" "${POSIX_L2_DIR}/run1" 20; then
    run_test_suite "${EXISTING_IMAGE}" "existing-rocm-aic" "${POSIX_L2_DIR}/run1"
    stop_stack
    md_append "- Completed: $(date '+%H:%M:%S')"
else
    log "Phase 4: Stack failed to start — skipping test matrix"
    md_append "- **FAILED**: Stack did not start. See ${LOGDIR}/lmcache-existing.log"
    stop_stack
fi

# --------------------------------------------------------------------------
# start_stack_nixl <image> <run_label> <l2_dir> <l1_gb>
# Like start_stack but uses the NIXL POSIX backend for L2 (requires aic-lmcache
# image with libnixl.so installed).
# --------------------------------------------------------------------------
start_stack_nixl() {
    local image="$1"
    local label="$2"
    local l2dir="$3"
    local lmcache_l1_gb="${4:-20}"
    local slot_size="${5:-16777216}"   # 16 MiB default
    local pool="${6:-4096}"

    log "  start_stack_nixl: image=${image} label=${label} L1=${lmcache_l1_gb}GB L2=${l2dir} (NIXL POSIX)"
    mkdir -p "${l2dir}"

    docker rm -f aic-lmcache-test aic-vllm-test 2>/dev/null || true
    docker network rm aic-test 2>/dev/null || true
    docker network create aic-test

    # The NIXL POSIX l2-adapter JSON for LMCache 0.5.3 (aic-lmcache image)
    local nixl_l2="{\"type\":\"nixl_store\",\"backend\":\"POSIX\",\"backend_params\":{\"file_path\":\"/data/l2/lmcache\",\"use_direct_io\":\"false\",\"file_size\":\"${slot_size}\"},\"pool_size\":${pool}}"

    docker run -d \
        --name aic-lmcache-test \
        --network aic-test \
        "${GPU_ARGS[@]}" \
        -v "${HF_HOME}:/hf:ro" \
        -v "${l2dir}:/data/l2" \
        -e HF_HOME=/hf \
        -e LMCACHE_PORT="${LMCACHE_PORT}" \
        -e ROCR_VISIBLE_DEVICES=0 \
        -e PYTHONHASHSEED=0 \
        -e NIXL_TELEMETRY_ENABLE=y \
        -e NIXL_TELEMETRY_EXPORTER=prometheus \
        -e NIXL_TELEMETRY_PROMETHEUS_PORT=19090 \
        --ipc shareable \
        --shm-size 16g \
        --cap-add CAP_SYS_ADMIN \
        --cap-add SYS_PTRACE \
        --cap-add IPC_LOCK \
        --security-opt seccomp:unconfined \
        --ulimit memlock=-1:-1 \
        --ulimit nofile=1048576:1048576 \
        -p 8080:8080 \
        -p 19090:19090 \
        --entrypoint lmcache \
        "${image}" \
        server \
        --host 0.0.0.0 \
        --http-port 8080 \
        --port "${LMCACHE_PORT}" \
        --l1-size-gb "${lmcache_l1_gb}" \
        --eviction-policy LRU \
        --max-gpu-workers 1 \
        --l2-adapter "${nixl_l2}" \
        > "${LOGDIR}/lmcache-${label}.log" 2>&1

    log "  LMCache server starting (NIXL POSIX)..."
    local retries=60
    while [ $retries -gt 0 ]; do
        if docker exec aic-lmcache-test python3 -c \
            "import socket; s=socket.create_connection(('127.0.0.1',${LMCACHE_PORT}),2); s.close()" \
            2>/dev/null; then
            log "  LMCache healthy"
            break
        fi
        sleep 5
        retries=$((retries - 1))
    done
    if [ $retries -eq 0 ]; then
        log "  ERROR: LMCache (NIXL POSIX) failed to start:"
        docker logs --tail 20 aic-lmcache-test 2>/dev/null || true
        return 1
    fi

    # Start vLLM (same as start_stack)
    local mp_json="{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://aic-lmcache-test\",\"lmcache.mp.port\":${LMCACHE_PORT}}}"

    docker run -d \
        --name aic-vllm-test \
        --network aic-test \
        "${GPU_ARGS[@]}" \
        -v "${HF_HOME}:/hf:ro" \
        -v "${REPO_ROOT}/scripts/vllm-wsl2-wrapper.py:/vllm-wrapper.py:ro" \
        -e HF_HOME=/hf \
        -e HF_HUB_CACHE=/hf/hub \
        -e HF_HUB_OFFLINE=1 \
        -e VLLM_TARGET_DEVICE=rocm \
        -e ROCR_VISIBLE_DEVICES=0 \
        -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:False \
        -e VLLM_ROCM_USE_AITER=1 \
        -e AMDGCN_USE_BUFFER_OPS=0 \
        -e MIOPEN_FIND_MODE=FAST \
        -e PYTHONHASHSEED=0 \
        -e VLLM_LOGGING_LEVEL=INFO \
        -e VLLM_SERVER_DEV_MODE=1 \
        --entrypoint python3 \
        --ipc container:aic-lmcache-test \
        --cap-add CAP_SYS_ADMIN \
        --cap-add SYS_PTRACE \
        --security-opt seccomp:unconfined \
        -p "${VLLM_PORT}:8000" \
        "${image}" \
        /vllm-wrapper.py \
        --model "${MODEL}" \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.50 \
        --max-model-len 16384 \
        --max-num-batched-tokens 4096 \
        --load-format safetensors \
        --block-size 64 \
        --enable-prefix-caching \
        --kv-cache-dtype auto \
        --attention-backend TRITON_ATTN \
        --stream-interval 20 \
        --async-scheduling \
        --kv-transfer-config "${mp_json}" \
        > "${LOGDIR}/vllm-${label}.log" 2>&1

    log "  vLLM starting (NIXL POSIX)..."
    local vretries=120
    while [ $vretries -gt 0 ]; do
        local ok
        ok=$(python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:${VLLM_PORT}/health', timeout=3); print('ok')" 2>/dev/null || echo "")
        if [ "${ok}" = "ok" ]; then
            log "  vLLM healthy"
            return 0
        fi
        sleep 5
        vretries=$((vretries - 1))
    done
    log "  ERROR: vLLM failed to start"
    docker logs --tail 30 aic-vllm-test 2>/dev/null || true
    return 1
}

# --------------------------------------------------------------------------
# Phase 5: Wait for build, then test new image with NIXL POSIX L2
# --------------------------------------------------------------------------
log "Phase 5: Waiting for background build (pid ${BUILD_PID})..."
md_append ""
md_append "### Phase 5: Waiting for gfx1151 build (NIXL POSIX L2)..."
md_append "- Build PID: ${BUILD_PID}"

wait "${BUILD_PID}" || true

BUILD_STATUS=$(cat "${LOGDIR}/build.status" 2>/dev/null || echo "UNKNOWN")
log "Phase 5: Build status: ${BUILD_STATUS}"
md_append "- Build status: ${BUILD_STATUS}"
md_append "- Build log: ${BUILD_LOG}"

if [[ "${BUILD_STATUS}" == "BUILD_SUCCESS" ]]; then
    log "Phase 5: Build succeeded — running NIXL POSIX test matrix on ${NEW_IMAGE}"

    # Run with a small L1 (4 GB) to force L2 eviction sooner and exercise L2 reads
    for l1_gb in 4 16; do
        md_append ""
        md_append "### Phase 5b: NIXL POSIX L2 — L1=${l1_gb}GB (${NEW_IMAGE})"
        md_append "- Started: $(date '+%H:%M:%S')"

        L2_RUN="${POSIX_L2_DIR}/nixl-l1-${l1_gb}gb"
        if start_stack_nixl "${NEW_IMAGE}" "nixl-l1${l1_gb}" "${L2_RUN}" "${l1_gb}" 16777216 4096; then
            run_test_suite "${NEW_IMAGE}" "nixl-posix-l1${l1_gb}gb" "${L2_RUN}"
            stop_stack
            md_append "- Completed: $(date '+%H:%M:%S')"
        else
            log "Phase 5b: Stack failed to start (NIXL POSIX, L1=${l1_gb}GB)"
            md_append "- **FAILED**: Stack did not start"
            stop_stack
        fi
    done
else
    log "Phase 5: Build failed — skipping NIXL POSIX tests"
    md_append "- Build failed; NIXL POSIX phase skipped"
fi

# --------------------------------------------------------------------------
# Phase 6: Prometheus snapshot + summary
# --------------------------------------------------------------------------
log "Phase 6: Taking Prometheus TSDB snapshot..."
curl -sf -X POST "http://localhost:9292/-/snapshot" > "${LOGDIR}/prom-snapshot.json" 2>/dev/null || true

md_append ""
md_append "### Phase 6: Summary"
md_append "- Prometheus TSDB: ${PROM_TSDB}"
md_append "- Results CSVs: ${RESULTSDIR}/"
md_append "- Build log: ${BUILD_LOG}"
md_append "- Overnight log: ${LOGDIR}/overnight.log"
md_append "- Completed: $(date '+%Y-%m-%d %H:%M:%S')"
md_append ""
md_append "View Prometheus: http://localhost:9292 (container aic-prometheus-overnight still running)"
md_append ""

log "=== Overnight run COMPLETE ==="
log "Results: ${RESULTSDIR}/"
log "Prometheus still running at http://localhost:9090"
