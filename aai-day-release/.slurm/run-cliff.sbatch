#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
#SBATCH --job-name=aai-day-cliff
#SBATCH --partition=defq
#SBATCH --constraint=MARKHAM&GFX942&NVME
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#
# Run the KV-cache cliff benchmark (benchmarks/run_cliff.py) end-to-end on ONE
# Markham MI300X + local-NVMe node, sweeping all three arms sequentially:
#
#   1. vram_only        plain vLLM, VRAM prefix cache only (the "cliff" baseline)
#   2. kvd_v2 (nvme)    LMCache: DRAM L1 + NIXL AIS_MT -> local NVMe L2a
#   3. kvd_v2 (gds)     LMCache: hipFile GDS NVMe slab as L1 (no L2)
#
# Each arm launches its own container stack via `docker run` (no compose/make
# dependency -- mirrors .slurm/run-build-distribute.sh's cmd_test), waits for the
# endpoint, runs the sweep in a sidecar container, then tears the stack down.
# The two kvd_v2 arms are plotted separately against the shared vram_only
# baseline so the NVMe-L2 and GDS-L1 backends can be compared.
#
# Submit from the aai-day-release/ tree root:
#
#   sbatch .slurm/run-cliff.sbatch
#   # pin a node:
#   sbatch --nodelist=ctr-s95-mi300x-3 .slurm/run-cliff.sbatch
#   # override the sweep / model:
#   AAI_BENCH_CONCUR=1,8,32,64,128,250 AAI_MODEL=openai/gpt-oss-120b \
#       sbatch .slurm/run-cliff.sbatch
#
# Storage note (from a live probe of ctr-cx63-mi300x-21, 2026-07-13): the MI300X
# nodes expose 4x 3.5T Samsung NVMe, but the spares (nvme1-3) are UNMOUNTED and
# /mnt is empty -- the compose default NVME_DATA=/mnt/lmcache-nvme does not exist.
# The OS root (/) is itself ext4-on-NVMe with ~3T free, so this script points the
# NVMe pools at node-local /tmp (physically the same local NVMe, writable by the
# job user). If a dedicated data NVMe gets mounted later, set AAI_NVME_DIR to it.
#
set -uo pipefail

# --- Config (all overridable via environment) --------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

AAI_IMAGE="${AAI_IMAGE:-rocm-aic-aai-day:latest}"
AAI_IMAGE_DIR="${AAI_IMAGE_DIR:-/scratch/${USER}/images}"
AAI_MODEL="${AAI_MODEL:-openai/gpt-oss-120b}"
# Shared HF cache that already holds gpt-oss-120b (see plan); offline so no token.
AAI_HF_HOME="${AAI_HF_HOME:-/scratch/${USER}/vllm-lmcache-hipfile/hf}"

# Node-local NVMe-backed scratch for the LMCache tiers (see storage note above).
AAI_LOCAL_ROOT="${AAI_LOCAL_ROOT:-/tmp/aai-cliff.${SLURM_JOB_ID:-manual}}"
AAI_NVME_DIR="${AAI_NVME_DIR:-${AAI_LOCAL_ROOT}/nvme}"   # L2a AIS_MT pool
AAI_NFS_DIR="${AAI_NFS_DIR:-${AAI_LOCAL_ROOT}/nfs}"      # L2b POSIX pool (local stand-in)
AAI_SLAB_DIR="${AAI_SLAB_DIR:-${AAI_LOCAL_ROOT}/slab}"   # GDS L1 slab

# Benchmark sweep knobs (defaults mirror the Makefile `cliff` target).
AAI_BENCH_ISL="${AAI_BENCH_ISL:-20000}"
AAI_BENCH_SHARED_TOK="${AAI_BENCH_SHARED_TOK:-18000}"
AAI_BENCH_CONCUR="${AAI_BENCH_CONCUR:-1,2,4,8,16,32,48,64,80,100,128,160,200,250}"
AAI_BENCH_ITERS="${AAI_BENCH_ITERS:-3}"
AAI_BENCH_PREFIX_MODE="${AAI_BENCH_PREFIX_MODE:-per_client}"  # per_client => a real cliff

# vLLM knobs.
AAI_TP="${AAI_TP:-1}"
AAI_GPU_MEM_UTIL="${AAI_GPU_MEM_UTIL:-0.90}"
AAI_MAX_MODEL_LEN="${AAI_MAX_MODEL_LEN:-32768}"
AAI_MAX_BATCHED_TOKENS="${AAI_MAX_BATCHED_TOKENS:-4096}"
AAI_L1_SIZE_GB="${AAI_L1_SIZE_GB:-20}"
AAI_NVME_POOL="${AAI_NVME_POOL:-4096}"
AAI_NVME_SLOT_SIZE="${AAI_NVME_SLOT_SIZE:-268435456}"
AAI_NFS_POOL="${AAI_NFS_POOL:-1024}"

VRAM_PORT=8001
KVD_PORT=8000
LMCACHE_PORT=6555
STAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS="${REPO}/results"
PLOTS="${REPO}/plots"

log() { printf '[cliff-job] %s\n' "$*" >&2; }
die() { printf '[cliff-job] ERROR: %s\n' "$*" >&2; exit 1; }

log "host        : $(hostname)  job=${SLURM_JOB_ID:-manual}"
log "image       : ${AAI_IMAGE}"
log "model       : ${AAI_MODEL}  (HF_HOME=${AAI_HF_HOME}, offline)"
log "local NVMe  : ${AAI_NVME_DIR}"
log "results     : ${RESULTS}"
command -v docker >/dev/null 2>&1 || die "docker not found on $(hostname)"

mkdir -p "${RESULTS}" "${PLOTS}" "${AAI_NVME_DIR}" "${AAI_NFS_DIR}" "${AAI_SLAB_DIR}"
[[ -d "${AAI_HF_HOME}/hub" ]] || die "HF cache not found at ${AAI_HF_HOME}/hub"

# --- Load the image from the shared tarball if the node doesn't have it -------
if [[ -z "$(docker images -q "${AAI_IMAGE}" 2>/dev/null)" ]]; then
    tarball="$(ls -t "${AAI_IMAGE_DIR}"/rocm-aic-aai-day-latest-*.tar.* 2>/dev/null | head -1 || true)"
    [[ -n "${tarball}" ]] || die "image ${AAI_IMAGE} absent and no tarball in ${AAI_IMAGE_DIR} (run the build first)"
    log "loading image from ${tarball}"
    case "${tarball}" in
        *.zst) zstd -dc "${tarball}" | docker load ;;
        *.gz)  gzip  -dc "${tarball}" | docker load ;;
        *)     docker load -i "${tarball}" ;;
    esac || die "docker load failed"
fi

# --- Container / GPU flags (mirror docker-compose.yml + cmd_test) -------------
GPU_FLAGS=(--device /dev/kfd --device /dev/dri --ipc host
           --cap-add SYS_PTRACE --cap-add CAP_SYS_ADMIN
           --security-opt seccomp=unconfined)
# --network host: the sidecar bench + the ZMQ connector reach services at
# localhost, so no compose bridge / service DNS is needed.
HF_ENV=(-e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub -e HF_HUB_OFFLINE=1
        -e TRANSFORMERS_OFFLINE=1 -e VLLM_CACHE_ROOT=/hf/vllm
        -e ROCR_VISIBLE_DEVICES=0 -e PYTHONHASHSEED=0
        -e AMDGCN_USE_BUFFER_OPS=0 -e MIOPEN_FIND_MODE=FAST)

# --- Helpers -----------------------------------------------------------------
wait_ready() {  # $1=port $2=timeout_s
    local port="$1" timeout="${2:-1200}" waited=0
    log "waiting for vLLM on :${port} (up to ${timeout}s)"
    while (( waited < timeout )); do
        if curl -fsS "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            log "vLLM :${port} is ready (${waited}s)"; return 0
        fi
        sleep 10; waited=$((waited + 10))
    done
    return 1
}

run_bench() {  # $1=arm $2=endpoint_port $3=out_csv $4=extra args...
    local arm="$1" port="$2" out="$3"; shift 3
    log "running sweep: arm=${arm} endpoint=:${port} -> ${out}"
    docker run --rm --network host \
        -v "${REPO}":/work -w /work \
        -v "${AAI_HF_HOME}":/hf "${HF_ENV[@]}" \
        --entrypoint python3 "${AAI_IMAGE}" \
        benchmarks/run_cliff.py \
            --endpoint "http://localhost:${port}" \
            --model "${AAI_MODEL}" \
            --arm "${arm}" \
            --isl "${AAI_BENCH_ISL}" \
            --shared-prefix-tokens "${AAI_BENCH_SHARED_TOK}" \
            --concurrencies "${AAI_BENCH_CONCUR}" \
            --iters "${AAI_BENCH_ITERS}" \
            --prefix-mode "${AAI_BENCH_PREFIX_MODE}" \
            --warmup-iters 1 \
            --out "results/$(basename "${out}")" "$@"
}

stop_containers() { docker rm -f "$@" >/dev/null 2>&1 || true; }

# =============================================================================
# Arm A — vram_only : plain vLLM, no LMCache connector
# =============================================================================
VRAM_CSV="${RESULTS}/cliff-vram_only-${STAMP}.csv"
run_arm_vram() {
    stop_containers aai-cliff-vram
    log "=== Arm A (vram_only): launching plain vLLM on :${VRAM_PORT} ==="
    docker run -d --name aai-cliff-vram --network host "${GPU_FLAGS[@]}" \
        -v "${AAI_HF_HOME}":/hf "${HF_ENV[@]}" \
        "${AAI_IMAGE}" \
            --model "${AAI_MODEL}" --host 0.0.0.0 --port "${VRAM_PORT}" \
            --tensor-parallel-size "${AAI_TP}" \
            --gpu-memory-utilization "${AAI_GPU_MEM_UTIL}" \
            --max-model-len "${AAI_MAX_MODEL_LEN}" \
            --max-num-batched-tokens "${AAI_MAX_BATCHED_TOKENS}" \
            --block-size 64 --enable-prefix-caching --kv-cache-dtype fp8 \
            --async-scheduling \
            --disable-access-log-for-endpoints "/health,/metrics,/v1/models" \
        >/dev/null || { log "Arm A: docker run failed"; return 1; }
    if ! wait_ready "${VRAM_PORT}"; then
        log "Arm A: server never became ready; last logs:"; docker logs --tail 40 aai-cliff-vram 2>&1 | sed 's/^/    /'
        stop_containers aai-cliff-vram; return 1
    fi
    run_bench vram_only "${VRAM_PORT}" "${VRAM_CSV}"
    local rc=$?
    stop_containers aai-cliff-vram
    return $rc
}

# =============================================================================
# Arm B — kvd_v2 : LMCache stack (lmcache-server + connector'd vLLM)
#   mode "nvme" -> DRAM L1 + AIS_MT NVMe L2a (+ POSIX L2b stand-in)
#   mode "gds"  -> hipFile GDS NVMe slab as L1, no L2
# =============================================================================
launch_lmcache() {  # $1=mode(nvme|gds)
    local mode="$1"
    stop_containers aai-cliff-lmcache
    local -a mounts=(-v "${AAI_NVME_DIR}":/data/nvme -v "${AAI_NFS_DIR}":/data/nfs
                     -v "${AAI_SLAB_DIR}":/data/slab
                     -v /lib/modules:/lib/modules:ro -v /usr/src:/usr/src:ro
                     -v /sys/kernel/debug:/sys/kernel/debug)
    local cmd
    if [[ "${mode}" == "gds" ]]; then
        cmd="exec lmcache server --host 0.0.0.0 --port ${LMCACHE_PORT} \
             --l1-size-gb ${AAI_L1_SIZE_GB} --eviction-policy LRU \
             --gds-l1-path /data/slab"
    else
        local ais="{\"type\":\"nixl_store\",\"backend\":\"AIS_MT\",\"backend_params\":{\"file_path\":\"/data/nvme/lmcache\",\"use_direct_io\":\"true\",\"file_size\":${AAI_NVME_SLOT_SIZE}},\"pool_size\":${AAI_NVME_POOL}}"
        local posix="{\"type\":\"nixl_store\",\"backend\":\"POSIX\",\"backend_params\":{\"file_path\":\"/data/nfs/lmcache\"},\"pool_size\":${AAI_NFS_POOL}}"
        cmd="exec lmcache server --host 0.0.0.0 --port ${LMCACHE_PORT} \
             --l1-size-gb ${AAI_L1_SIZE_GB} --eviction-policy LRU \
             --l2-adapter '${ais}' --l2-adapter '${posix}'"
    fi
    log "launching lmcache-server (mode=${mode}) on :${LMCACHE_PORT}"
    docker run -d --name aai-cliff-lmcache --network host \
        --device /dev/kfd --device /dev/dri --ipc host \
        --cap-add CAP_SYS_ADMIN --cap-add SYS_PTRACE --cap-add IPC_LOCK \
        --security-opt seccomp=unconfined --ulimit nofile=1048576:1048576 \
        "${mounts[@]}" \
        -e LMCACHE_PORT="${LMCACHE_PORT}" -e HIPFILE_ALLOW_COMPAT_MODE=false \
        -e HIPFILE_UNSUPPORTED_FILE_SYSTEMS=true -e PYTHONHASHSEED=0 \
        --entrypoint /bin/bash "${AAI_IMAGE}" -o pipefail -c "${cmd}" >/dev/null
}

launch_vllm_kvd() {
    stop_containers aai-cliff-kvd-vllm
    local kvcfg="{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://localhost\",\"lmcache.mp.port\":${LMCACHE_PORT}}}"
    log "launching connector'd vLLM on :${KVD_PORT}"
    docker run -d --name aai-cliff-kvd-vllm --network host "${GPU_FLAGS[@]}" \
        -v "${AAI_HF_HOME}":/hf "${HF_ENV[@]}" \
        "${AAI_IMAGE}" \
            --model "${AAI_MODEL}" --host 0.0.0.0 --port "${KVD_PORT}" \
            --tensor-parallel-size "${AAI_TP}" \
            --gpu-memory-utilization "${AAI_GPU_MEM_UTIL}" \
            --max-model-len "${AAI_MAX_MODEL_LEN}" \
            --max-num-batched-tokens "${AAI_MAX_BATCHED_TOKENS}" \
            --block-size 64 --enable-prefix-caching --kv-cache-dtype fp8 \
            --async-scheduling \
            --disable-access-log-for-endpoints "/health,/metrics,/v1/models" \
            --kv-transfer-config "${kvcfg}" >/dev/null
}

run_arm_kvd() {  # $1=mode(nvme|gds) $2=out_csv
    local mode="$1" out="$2"
    log "=== Arm B (kvd_v2, mode=${mode}) ==="
    launch_lmcache "${mode}" || { log "kvd/${mode}: lmcache failed to start"; return 1; }
    sleep 5
    launch_vllm_kvd     || { log "kvd/${mode}: vllm failed to start"; stop_containers aai-cliff-lmcache; return 1; }
    if ! wait_ready "${KVD_PORT}"; then
        log "kvd/${mode}: server never became ready; last logs:"
        docker logs --tail 40 aai-cliff-lmcache 2>&1 | sed 's/^/  [lmcache] /'
        docker logs --tail 40 aai-cliff-kvd-vllm 2>&1 | sed 's/^/  [vllm]    /'
        stop_containers aai-cliff-kvd-vllm aai-cliff-lmcache; return 1
    fi
    # Per-c warmup + drain: required for per_client mode and async-save kvd.
    run_bench kvd_v2 "${KVD_PORT}" "${out}" --warmup-at-each-c --post-warmup-sleep-s 5
    local rc=$?
    stop_containers aai-cliff-kvd-vllm aai-cliff-lmcache
    return $rc
}

# =============================================================================
# Run all arms; a failure in one is logged but does not abort the others.
# =============================================================================
KVD_NVME_CSV="${RESULTS}/cliff-kvd_v2-nvme-${STAMP}.csv"
KVD_GDS_CSV="${RESULTS}/cliff-kvd_v2-gds-${STAMP}.csv"
declare -A STATUS

run_arm_vram                    && STATUS[vram_only]=ok       || STATUS[vram_only]=FAIL
run_arm_kvd nvme "${KVD_NVME_CSV}" && STATUS[kvd_nvme]=ok     || STATUS[kvd_nvme]=FAIL
run_arm_kvd gds  "${KVD_GDS_CSV}"  && STATUS[kvd_gds]=ok      || STATUS[kvd_gds]=FAIL

# --- Plot: baseline vs each kvd backend separately ---------------------------
# Both kvd runs carry arm="kvd_v2", so plot them against the shared vram_only
# baseline in separate output dirs to keep the two backends distinct.
plot_pair() {  # $1=out_subdir $2=kvd_csv
    local sub="$1" kvd="$2"
    [[ -s "${VRAM_CSV}" && -s "${kvd}" ]] || { log "plot ${sub}: missing CSV(s), skipping"; return; }
    mkdir -p "${PLOTS}/${sub}"
    docker run --rm --network host -v "${REPO}":/work -w /work \
        --entrypoint python3 "${AAI_IMAGE}" \
        benchmarks/plot_cliff.py \
            --input "results/$(basename "${VRAM_CSV}")" "results/$(basename "${kvd}")" \
            --output-dir "plots/${sub}" \
        && log "plots (${sub}) -> ${PLOTS}/${sub}" || log "plot ${sub}: failed"
}
plot_pair nvme "${KVD_NVME_CSV}"
plot_pair gds  "${KVD_GDS_CSV}"

# --- Cleanup local NVMe scratch ---------------------------------------------
rm -rf "${AAI_LOCAL_ROOT}" 2>/dev/null || true

log "===================== SUMMARY ====================="
for k in vram_only kvd_nvme kvd_gds; do log "  ${k}: ${STATUS[$k]:-?}"; done
log "  CSVs  : ${RESULTS}"
log "  plots : ${PLOTS}/{nvme,gds}"
log "==================================================="
[[ "${STATUS[vram_only]}" == ok ]] || exit 1
exit 0
