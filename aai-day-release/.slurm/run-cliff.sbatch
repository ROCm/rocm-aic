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
# Discard Slurm's own stdout: the script re-redirects all output into
# logs/<job-id>/cliff.out as soon as it starts (see below), so this only ever
# sees the few lines before that redirect -- not worth a file.  (Slurm can't
# create logs/<job-id>/ itself, hence the in-script redirect rather than %j here.)
#SBATCH --output=/dev/null
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
#   BENCH_CONCUR=1,8,32,64,128,250 VLLM_MODEL=openai/gpt-oss-120b \
#       sbatch .slurm/run-cliff.sbatch
#
# Storage note (from a live probe of ctr-cx63-mi300x-21, 2026-07-13): the MI300X
# nodes expose 4x 3.5T Samsung NVMe, but the spares (nvme1-3) are UNMOUNTED and
# /mnt is empty -- the compose default NVME_DATA=/mnt/lmcache-nvme does not exist.
# The OS root (/) is itself ext4-on-NVMe with ~3T free, so this script points the
# NVMe pools at node-local /tmp (physically the same local NVMe, writable by the
# job user). If a dedicated data NVMe gets mounted later, set NVME_DATA to it.
#
set -uo pipefail

# --- Config (all overridable via environment) --------------------------------
# Under sbatch, Slurm copies this script to the node's spool dir and runs it from
# there, so ${BASH_SOURCE[0]} points at /var/spool/slurmd/... -- deriving REPO
# from it lands logs/<job-id> (and the /work bind-mount of benchmarks/) in an
# unwritable, wrong place.  Prefer SLURM_SUBMIT_DIR (the dir sbatch was invoked
# from; `make cliff-submit` cds to aai-day-release first), and fall back to
# BASH_SOURCE for a direct, non-sbatch invocation.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# Send all output to logs/<job-id>/cliff.out under the repo.  Slurm can't create
# the parent dir (and the job id isn't known at submit time), so we make it here
# and redirect the whole script's stdout+stderr into it.  Slurm's own --output is
# /dev/null, so if mkdir fails (e.g. a misresolved REPO) the few pre-redirect
# lines are lost -- acceptable; sacct still reports the failure.
# Everything this job produces -- cliff.out, container logs, the Prometheus TSDB,
# the generated LMCache config, and now the benchmark CSVs + plots -- lives under
# this one per-job dir, so the tree root stays clean (no stray results/ or plots/).
# LOGDIR_REL is the same path relative to REPO: containers bind-mount REPO at /work
# and write via this relative path (see run_bench / plot_pair).
LOGDIR_REL="logs/${SLURM_JOB_ID:-manual}"
LOGDIR="${REPO}/${LOGDIR_REL}"
if mkdir -p "${LOGDIR}" 2>/dev/null; then
    exec >>"${LOGDIR}/cliff.out" 2>&1
fi

AAI_IMAGE="${AAI_IMAGE:-rocm-aic-aai-day:latest}"
AAI_IMAGE_DIR="${AAI_IMAGE_DIR:-/scratch/${USER}/images}"
VLLM_MODEL="${VLLM_MODEL:-openai/gpt-oss-120b}"
# Shared HF cache that already holds gpt-oss-120b (see plan); offline so no token.
HF_HOME="${HF_HOME:-/scratch/${USER}/vllm-lmcache-hipfile/hf}"

# Node-local NVMe-backed scratch for the LMCache tiers (see storage note above).
AAI_LOCAL_ROOT="${AAI_LOCAL_ROOT:-/tmp/aai-cliff.${SLURM_JOB_ID:-manual}}"
NVME_DATA="${NVME_DATA:-${AAI_LOCAL_ROOT}/nvme}"   # L2a AIS_MT pool
NFS_DATA="${NFS_DATA:-${AAI_LOCAL_ROOT}/nfs}"      # L2b POSIX pool (local stand-in)
GDS_SLAB_DATA="${GDS_SLAB_DATA:-${AAI_LOCAL_ROOT}/slab}"   # GDS L1 slab

# Benchmark sweep knobs (defaults mirror the Makefile `cliff` target).
BENCH_ISL="${BENCH_ISL:-20000}"
BENCH_SHARED_TOK="${BENCH_SHARED_TOK:-18000}"
BENCH_CONCUR="${BENCH_CONCUR:-1,2,4,8,16,32,48,64,80,100,128,160,200,250}"
BENCH_ITERS="${BENCH_ITERS:-3}"
BENCH_PREFIX_MODE="${BENCH_PREFIX_MODE:-per_client}"  # per_client => a real cliff

# Arm selector: comma list of arms to run (vram, nvme, gds).  Default all three;
# narrow it for a quick check, e.g. `make cliff-submit AAI_CLIFF_ARMS=nvme`.
AAI_CLIFF_ARMS="${AAI_CLIFF_ARMS:-vram,nvme,gds}"

# vLLM knobs.
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLM_GPU_MEMORY_UTILIZATION="${VLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLM_MAX_MODEL_LEN="${VLM_MAX_MODEL_LEN:-32768}"
VLM_MAX_NUM_BATCHED_TOKENS="${VLM_MAX_NUM_BATCHED_TOKENS:-4096}"
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-20}"
LMCACHE_NVME_POOL="${LMCACHE_NVME_POOL:-4096}"
# hipFile GDS compat fallback.  Default 'false' fails loudly if true GPU-direct
# (P2PDMA) registration is unavailable -- so kvd arms measure real GDS.  Set to
# 'true' to allow hipFile's staged/bounce-buffer fallback when hipFileBufRegister
# fails (err 5013) on a node where GDS is not fully enabled -- lets the AIS_MT /
# GDS arms run, but NOT as true GPU-direct.
HIPFILE_ALLOW_COMPAT_MODE="${HIPFILE_ALLOW_COMPAT_MODE:-false}"
# VRAM NIXL staging buffer for the AIS_MT nvme arm (bytes).  MUST be a GPU (cuda)
# buffer so hipFileBufRegister succeeds -- registering a host/DRAM buffer returns
# err 5013.  512 MiB default (mirrors recipies/vllm-lmcache-nixl).
AAI_NIXL_BUFFER_SIZE="${AAI_NIXL_BUFFER_SIZE:-536870912}"
# Attention backend for all arms.  On vLLM 0.25 the ROCm backends (ROCM_ATTN,
# ROCM_AITER_UNIFIED_ATTN) report supports_kv_connector()=False, so the kvd arms
# (LMCache connector) MUST use TRITON_ATTN, which supports connectors.  It also
# requires a full-attention model (VLLM_MODEL): a sliding-window/hybrid model like
# gpt-oss can't unify its KV-cache specs under a connector on 0.25.  Same backend
# on all arms keeps baseline (vram) and kvd comparable.
AAI_ATTENTION_BACKEND="${AAI_ATTENTION_BACKEND:-TRITON_ATTN}"

VRAM_PORT=8001
KVD_PORT=8000
LMCACHE_PORT=6555
STAMP="$(date +%Y%m%d-%H%M%S)"
# CSVs and plots go under the per-job logs dir so the run is self-contained and
# the tree root needs no results/ or plots/ dirs.
RESULTS="${LOGDIR}/results"
PLOTS="${LOGDIR}/plots"

# --- Metrics capture (Prometheus sidecar) ------------------------------------
# Auto-on; set AAI_MONITORING=0 to skip.  AAI_METRICS_DIR is the Prometheus TSDB
# location -- defaults to logs/<job-id>/prometheus (job-id is the SLURM job, or
# "manual" off-Slurm); override with an NFS path to explore the run afterward.
# AAI_EXPORTERS=1 also launches the containerized node + AMD GPU exporters (for
# nodes without the host services); set 0 to rely on host-installed exporters
# and avoid :9100/:5000 clashes.
AAI_MONITORING="${AAI_MONITORING:-1}"
AAI_METRICS_DIR="${AAI_METRICS_DIR:-${REPO}/logs/${SLURM_JOB_ID:-manual}/prometheus}"
AAI_EXPORTERS="${AAI_EXPORTERS:-1}"
MON_COMPOSE="${REPO}/monitoring/docker-compose.monitoring.yml"
MON_DIR="${REPO}/monitoring"

log() { printf '[cliff-job] %s\n' "$*" >&2; }
die() { printf '[cliff-job] ERROR: %s\n' "$*" >&2; exit 1; }

log "host        : $(hostname)  job=${SLURM_JOB_ID:-manual}"
log "image       : ${AAI_IMAGE}"
log "model       : ${VLLM_MODEL}  (HF_HOME=${HF_HOME}, offline)"
log "local NVMe  : ${NVME_DATA}"
log "results     : ${RESULTS}"
command -v docker >/dev/null 2>&1 || die "docker not found on $(hostname)"
# REPO must be the aai-day-release tree (it is bind-mounted at /work so the
# benchmark can run benchmarks/run_cliff.py); fail early and clearly otherwise.
[[ -f "${REPO}/benchmarks/run_cliff.py" ]] || \
    die "benchmarks/run_cliff.py not found under REPO=${REPO}; submit from the aai-day-release/ dir (SLURM_SUBMIT_DIR)"

mkdir -p "${RESULTS}" "${PLOTS}" "${NVME_DATA}" "${NFS_DATA}" "${GDS_SLAB_DATA}"
[[ -d "${HF_HOME}/hub" ]] || die "HF cache not found at ${HF_HOME}/hub"

# --- Load images from shared tarballs when needed ----------------------------
# Reload when the node's image is absent, when the /scratch tarball is newer than
# what this node last loaded, or when AAI_FORCE_LOAD=1 -- so a rebuild is picked
# up automatically instead of a stale cached image being reused (mirrors cmd_test
# / cmd_push in run-build-distribute.sh).  A per-uid+image marker under /var/tmp
# records the tarball mtime last loaded here; comparing tarball-mtime vs the
# recorded tarball-mtime (both build-side values) avoids build/node clock skew.
#
#   $1=image name:tag  $2=tarball glob  $3=mode ("required" -> die if no tarball;
#   otherwise best-effort).  Returns 0 iff the image is present afterward.
load_image_if_needed() {
    local image="$1" glob="$2" mode="${3:-optional}"
    local tb marker have tarm loadedm
    tb="$(ls -t ${glob} 2>/dev/null | head -1 || true)"
    marker="/var/tmp/aai-day-loaded-$(id -u)-$(printf '%s' "${image}" | tr '/:' '__').mtime"
    have="$(docker images -q "${image}" 2>/dev/null)"
    tarm="$(stat -c %Y "${tb}" 2>/dev/null || echo 0)"
    loadedm="$(cat "${marker}" 2>/dev/null || echo 0)"
    if [[ "${AAI_FORCE_LOAD:-0}" == "1" || -z "${have}" || "${tarm}" -gt "${loadedm}" ]]; then
        if [[ -z "${tb}" ]]; then
            [[ "${mode}" == "required" ]] && die "image ${image} absent and no tarball matching ${glob} (run the build first)"
            [[ -n "${have}" ]] && return 0   # no tarball, but image already present
            log "  ${image}: absent and no tarball (${glob}); skipping"
            return 1
        fi
        log "loading ${image} from ${tb} (tarball=${tarm} last-loaded=${loadedm} present=$([[ -n "${have}" ]] && echo yes || echo no) force=${AAI_FORCE_LOAD:-0})"
        case "${tb}" in
            *.zst) zstd -dc "${tb}" | docker load ;;
            *.gz)  gzip  -dc "${tb}" | docker load ;;
            *)     docker load -i "${tb}" ;;
        esac || { [[ "${mode}" == "required" ]] && die "docker load failed"; log "  ${image}: docker load failed"; return 1; }
        echo "${tarm}" > "${marker}" 2>/dev/null || true
    else
        log "image up to date on $(hostname): ${image} (id ${have}); AAI_FORCE_LOAD=1 forces a reload"
    fi
    return 0
}

# Main inference image (required -- the whole run needs it).
_img_base="$(printf '%s' "${AAI_IMAGE}" | tr '/:' '--')"
load_image_if_needed "${AAI_IMAGE}" "${AAI_IMAGE_DIR}/${_img_base}-*.tar.*" required

# Fabric exporter images (optional).  When metrics exporters are on and the user
# has not pinned an explicit image, try the distributed tarball (built by
# run-build-distribute.sh build-exporters) and point the monitoring sidecar at it
# ONLY when the image is actually present.  On a node with the batesste host
# service the container is port-skipped anyway; with neither, node-exporter's
# nvme/infiniband collectors still cover it -- see _monitoring_run_up.
if [[ "${AAI_EXPORTERS}" == "1" ]]; then
    if [[ -z "${AAI_NVME_EXPORTER_IMAGE:-}" ]]; then
        _t="aai-day-nvme-exporter:latest"; _b="$(printf '%s' "${_t}" | tr '/:' '--')"
        load_image_if_needed "${_t}" "${AAI_IMAGE_DIR}/${_b}.tar.*" && export AAI_NVME_EXPORTER_IMAGE="${_t}"
    fi
    if [[ -z "${AAI_RDMA_EXPORTER_IMAGE:-}" ]]; then
        _t="aai-day-rdma-exporter:latest"; _b="$(printf '%s' "${_t}" | tr '/:' '--')"
        load_image_if_needed "${_t}" "${AAI_IMAGE_DIR}/${_b}.tar.*" && export AAI_RDMA_EXPORTER_IMAGE="${_t}"
    fi
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
        -e AMDGCN_USE_BUFFER_OPS=0 -e MIOPEN_FIND_MODE=FAST
        # AITER unified attention: required so a KV connector (LMCache) can unify
        # gpt-oss's hybrid (sliding-window + full) KV-cache specs.  Without
        # VLLM_ROCM_USE_AITER=1 the ROCM_AITER_UNIFIED_ATTN backend is rejected and
        # vLLM falls back to TRITON_ATTN -> "Hybrid KV cache manager is disabled but
        # failed to convert the KV cache specs" at engine init.
        -e VLLM_ROCM_USE_AITER=1)

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
        -v "${HF_HOME}":/hf "${HF_ENV[@]}" \
        --entrypoint python3 "${AAI_IMAGE}" \
        benchmarks/run_cliff.py \
            --endpoint "http://localhost:${port}" \
            --model "${VLLM_MODEL}" \
            --arm "${arm}" \
            --isl "${BENCH_ISL}" \
            --shared-prefix-tokens "${BENCH_SHARED_TOK}" \
            --concurrencies "${BENCH_CONCUR}" \
            --iters "${BENCH_ITERS}" \
            --prefix-mode "${BENCH_PREFIX_MODE}" \
            --warmup-iters 1 \
            --out "${LOGDIR_REL}/results/$(basename "${out}")" "$@"
}

# Pre-launch cleanup: drop any leftover container WITHOUT capturing its log --
# it belongs to a previous run, and capturing here would pollute this job's dir
# with a stale container-<name>.log (until the real teardown overwrites it).
rm_containers() { docker rm -f "$@" >/dev/null 2>&1 || true; }

# Teardown: capture each container's full stdout/stderr into the per-job logs dir
# before removing it.  The compute node is ephemeral, so `docker logs` is lost at
# teardown otherwise -- and Prometheus captures metrics, not container stdout
# (vLLM/LMCache/NIXL startup errors, stack traces, AIS debug lines live here).
stop_containers() {
    local c
    for c in "$@"; do
        docker inspect "${c}" >/dev/null 2>&1 || continue
        docker logs "${c}" > "${LOGDIR}/container-${c}.log" 2>&1 || true
    done
    docker rm -f "$@" >/dev/null 2>&1 || true
}

# --- Metrics-capture sidecar (Prometheus + optional exporters) ---------------
# The exporter/Prometheus stand-up, health checks, and teardown live in a shared
# lib so the cliff sweep and the build smoke-test (run-build-distribute.sh) drive
# metrics identically.  The lib's functions read log(), AAI_IMAGE, MON_DIR,
# AAI_METRICS_DIR, MON_COMPOSE, AAI_EXPORTERS, AAI_MONITORING -- all defined above.
source "${REPO}/monitoring/monitoring-lib.sh"
# Guarantee teardown (containers only; the NFS TSDB is retained) on any exit.
trap stop_monitoring EXIT

# =============================================================================
# Arm A — vram_only : plain vLLM, no LMCache connector
# =============================================================================
VRAM_CSV="${RESULTS}/cliff-vram_only-${STAMP}.csv"
run_arm_vram() {
    rm_containers aai-cliff-vram
    log "=== Arm A (vram_only): launching plain vLLM on :${VRAM_PORT} ==="
    docker run -d --name aai-cliff-vram --network host "${GPU_FLAGS[@]}" \
        -v "${HF_HOME}":/hf "${HF_ENV[@]}" \
        "${AAI_IMAGE}" \
            --model "${VLLM_MODEL}" --host 0.0.0.0 --port "${VRAM_PORT}" \
            --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
            --gpu-memory-utilization "${VLM_GPU_MEMORY_UTILIZATION}" \
            --max-model-len "${VLM_MAX_MODEL_LEN}" \
            --max-num-batched-tokens "${VLM_MAX_NUM_BATCHED_TOKENS}" \
            --block-size 64 --enable-prefix-caching --kv-cache-dtype fp8 \
            --async-scheduling \
            --attention-backend "${AAI_ATTENTION_BACKEND}" \
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
# Arm B — kvd_v2 : LMCache KV offload
#   mode "nvme" -> connector-side NIXL AIS_MT storage with a VRAM staging buffer
#                  (nixl_buffer_device=cuda) so hipFileBufRegister does TRUE GDS;
#                  in-process LMCacheConnectorV1, no standalone server, no L2b.
#                  Models recipies/vllm-lmcache-nixl (the proven AIS_MT path).
#                  A DRAM staging buffer -- what the old l2-adapter server used --
#                  makes hipFileBufRegister fail err 5013; cuda VRAM is required.
#   mode "gds"  -> hipFile GDS NVMe slab as L1 via a standalone lmcache-server
#                  (LMCache-native GDS; unchanged).
# =============================================================================
# hipFile/GDS extras any container doing hipFileBufRegister needs: kernel headers/
# modules + debugfs/tracefs for the ftrace kprobe, IPC_LOCK + unlimited memlock
# for pinning the registered buffer.
HIPFILE_FLAGS=(--cap-add IPC_LOCK --ulimit memlock=-1 --ulimit nofile=1048576:1048576
               -v /lib/modules:/lib/modules:ro -v /usr/src:/usr/src:ro
               -v /sys/kernel/debug:/sys/kernel/debug)
[[ -d /sys/kernel/tracing ]] && HIPFILE_FLAGS+=(-v /sys/kernel/tracing:/sys/kernel/tracing)

# Write the connector-side LMCache config for the nvme (AIS_MT) arm into the
# per-job logs dir (so it's captured), for mounting into the vLLM container.  The
# VRAM staging buffer (nixl_buffer_device=cuda) is the key to true GDS.
write_lmcache_nixl_config() {
    cat > "${LOGDIR}/lmcache-nixl-ais.yml" <<YML
# Generated by run-cliff.sbatch (kvd_v2 nvme arm) -- connector-side NIXL AIS_MT.
chunk_size: 256
local_cpu: false
save_decode_cache: true
# Deterministic, seed-independent chunk hashing.  The 'builtin' default uses
# Python hash() (PYTHONHASHSEED-dependent) and doesn't match across LMCache's
# separate lookup subprocess -> external lookups miss -> ext_hit stays 0 even
# under L1 overflow.  sha256_cbor (as in recipies/vllm-lmcache-nixl) is stable.
pre_caching_hash_algorithm: sha256_cbor
nixl_buffer_size: ${AAI_NIXL_BUFFER_SIZE}
nixl_buffer_device: cuda
extra_config:
  enable_nixl_storage: true
  nixl_backend: AIS_MT
  nixl_pool_size: ${LMCACHE_NVME_POOL}
  use_direct_io: true
  nixl_path: /data/nvme/lmcache
YML
}

# gds arm only: standalone lmcache-server with a hipFile GDS L1 slab.
launch_lmcache_gds() {
    rm_containers aai-cliff-lmcache
    log "launching lmcache-server (gds L1 slab) on :${LMCACHE_PORT}"
    docker run -d --name aai-cliff-lmcache --network host \
        --device /dev/kfd --device /dev/dri --ipc host \
        --cap-add CAP_SYS_ADMIN --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
        "${HIPFILE_FLAGS[@]}" -v "${GDS_SLAB_DATA}":/data/slab \
        -e LMCACHE_PORT="${LMCACHE_PORT}" -e HIPFILE_ALLOW_COMPAT_MODE="${HIPFILE_ALLOW_COMPAT_MODE}" \
        -e HIPFILE_UNSUPPORTED_FILE_SYSTEMS=true -e PYTHONHASHSEED=0 \
        --entrypoint /bin/bash "${AAI_IMAGE}" -o pipefail -c \
        "exec lmcache server --host 0.0.0.0 --port ${LMCACHE_PORT} \
             --l1-size-gb ${LMCACHE_L1_SIZE_GB} --eviction-policy LRU --gds-l1-path /data/slab" >/dev/null
}

# nvme arm: in-process LMCacheConnectorV1 with connector-side NIXL AIS_MT storage
# (VRAM staging buffer -> true GDS).  No standalone server.  NIXL telemetry runs
# in this process, so its exporter env lives here.
launch_vllm_kvd_nvme() {
    rm_containers aai-cliff-kvd-vllm
    write_lmcache_nixl_config
    local kvcfg="{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}"
    log "launching connector'd vLLM (LMCacheConnectorV1 + NIXL AIS_MT, cuda buffer) on :${KVD_PORT}"
    docker run -d --name aai-cliff-kvd-vllm --network host "${GPU_FLAGS[@]}" "${HIPFILE_FLAGS[@]}" \
        -v "${HF_HOME}":/hf "${HF_ENV[@]}" \
        -v "${NVME_DATA}":/data/nvme \
        -v "${LOGDIR}/lmcache-nixl-ais.yml":/config/lmcache-nixl-ais.yml:ro \
        -e LMCACHE_CONFIG_FILE=/config/lmcache-nixl-ais.yml \
        -e HIPFILE_ALLOW_COMPAT_MODE="${HIPFILE_ALLOW_COMPAT_MODE}" -e HIPFILE_UNSUPPORTED_FILE_SYSTEMS=true \
        -e NIXL_TELEMETRY_ENABLE=y -e NIXL_TELEMETRY_EXPORTER=prometheus \
        -e NIXL_TELEMETRY_PROMETHEUS_PORT="${NIXL_METRICS_PORT:-19090}" -e PYTHONHASHSEED=0 \
        "${AAI_IMAGE}" \
            --model "${VLLM_MODEL}" --host 0.0.0.0 --port "${KVD_PORT}" \
            --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
            --gpu-memory-utilization "${VLM_GPU_MEMORY_UTILIZATION}" \
            --max-model-len "${VLM_MAX_MODEL_LEN}" \
            --max-num-batched-tokens "${VLM_MAX_NUM_BATCHED_TOKENS}" \
            --block-size 64 --enable-prefix-caching --kv-cache-dtype fp8 \
            --async-scheduling \
            --attention-backend "${AAI_ATTENTION_BACKEND}" \
            --disable-access-log-for-endpoints "/health,/metrics,/v1/models" \
            --kv-transfer-config "${kvcfg}" >/dev/null
}

# gds arm: connector'd vLLM pointing at the standalone lmcache-server (MP).
launch_vllm_kvd_gds() {
    rm_containers aai-cliff-kvd-vllm
    local kvcfg="{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://localhost\",\"lmcache.mp.port\":${LMCACHE_PORT}}}"
    log "launching connector'd vLLM (LMCacheMPConnector) on :${KVD_PORT}"
    docker run -d --name aai-cliff-kvd-vllm --network host "${GPU_FLAGS[@]}" \
        -v "${HF_HOME}":/hf "${HF_ENV[@]}" \
        "${AAI_IMAGE}" \
            --model "${VLLM_MODEL}" --host 0.0.0.0 --port "${KVD_PORT}" \
            --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
            --gpu-memory-utilization "${VLM_GPU_MEMORY_UTILIZATION}" \
            --max-model-len "${VLM_MAX_MODEL_LEN}" \
            --max-num-batched-tokens "${VLM_MAX_NUM_BATCHED_TOKENS}" \
            --block-size 64 --enable-prefix-caching --kv-cache-dtype fp8 \
            --async-scheduling \
            --attention-backend "${AAI_ATTENTION_BACKEND}" \
            --disable-access-log-for-endpoints "/health,/metrics,/v1/models" \
            --kv-transfer-config "${kvcfg}" >/dev/null
}

run_arm_kvd() {  # $1=mode(nvme|gds) $2=out_csv
    local mode="$1" out="$2"
    log "=== Arm B (kvd_v2, mode=${mode}) ==="
    if [[ "${mode}" == "gds" ]]; then
        launch_lmcache_gds || { log "kvd/gds: lmcache failed to start"; return 1; }
        sleep 5
        launch_vllm_kvd_gds || { log "kvd/gds: vllm failed to start"; stop_containers aai-cliff-lmcache; return 1; }
        if ! wait_ready "${KVD_PORT}"; then
            log "kvd/gds: server never became ready; last logs:"
            docker logs --tail 40 aai-cliff-lmcache 2>&1 | sed 's/^/  [lmcache] /'
            docker logs --tail 40 aai-cliff-kvd-vllm 2>&1 | sed 's/^/  [vllm]    /'
            stop_containers aai-cliff-kvd-vllm aai-cliff-lmcache; return 1
        fi
        run_bench kvd_v2 "${KVD_PORT}" "${out}" --warmup-at-each-c --post-warmup-sleep-s 5
        local rc=$?
        stop_containers aai-cliff-kvd-vllm aai-cliff-lmcache
        return $rc
    fi
    # nvme: in-process LMCacheConnectorV1, no standalone server.
    launch_vllm_kvd_nvme || { log "kvd/nvme: vllm failed to start"; return 1; }
    if ! wait_ready "${KVD_PORT}"; then
        log "kvd/nvme: server never became ready; last logs:"
        docker logs --tail 60 aai-cliff-kvd-vllm 2>&1 | sed 's/^/  [vllm] /'
        stop_containers aai-cliff-kvd-vllm; return 1
    fi
    # Per-c warmup + drain: required for per_client mode and async-save kvd.
    run_bench kvd_v2 "${KVD_PORT}" "${out}" --warmup-at-each-c --post-warmup-sleep-s 5
    local rc=$?
    stop_containers aai-cliff-kvd-vllm
    return $rc
}

# =============================================================================
# Run all arms; a failure in one is logged but does not abort the others.
# =============================================================================
KVD_NVME_CSV="${RESULTS}/cliff-kvd_v2-nvme-${STAMP}.csv"
KVD_GDS_CSV="${RESULTS}/cliff-kvd_v2-gds-${STAMP}.csv"
declare -A STATUS

# Bring up the Prometheus metrics-capture sidecar before the first arm so it
# scrapes the whole sweep; torn down via the EXIT trap (TSDB retained on NFS).
start_monitoring

# Run only the arms named in AAI_CLIFF_ARMS (comma list of vram,nvme,gds).
_arm_on() { case ",${AAI_CLIFF_ARMS}," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
for _a in ${AAI_CLIFF_ARMS//,/ }; do
    case "${_a}" in vram|nvme|gds) ;; *) die "unknown arm '${_a}' in AAI_CLIFF_ARMS (valid: vram,nvme,gds)" ;; esac
done
log "arms        : ${AAI_CLIFF_ARMS}"

if _arm_on vram; then run_arm_vram && STATUS[vram_only]=ok || STATUS[vram_only]=FAIL; else STATUS[vram_only]=skip; fi
if _arm_on nvme; then run_arm_kvd nvme "${KVD_NVME_CSV}" && STATUS[kvd_nvme]=ok || STATUS[kvd_nvme]=FAIL; else STATUS[kvd_nvme]=skip; fi
if _arm_on gds;  then run_arm_kvd gds  "${KVD_GDS_CSV}"  && STATUS[kvd_gds]=ok  || STATUS[kvd_gds]=FAIL;  else STATUS[kvd_gds]=skip;  fi

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
            --input "${LOGDIR_REL}/results/$(basename "${VRAM_CSV}")" "${LOGDIR_REL}/results/$(basename "${kvd}")" \
            --output-dir "${LOGDIR_REL}/plots/${sub}" \
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
[[ "${AAI_MONITORING}" == "1" ]] && log "  metrics: ${AAI_METRICS_DIR} (Prometheus TSDB)"
log "==================================================="
# Fail the job if any arm that actually ran failed; skipped arms don't count.
rc=0
for k in vram_only kvd_nvme kvd_gds; do [[ "${STATUS[$k]:-skip}" == FAIL ]] && rc=1; done
exit "${rc}"
