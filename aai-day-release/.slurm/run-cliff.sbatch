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
# nodes expose 4x 3.5T Samsung NVMe -- nvme0 is the OS root, and the spares
# (nvme1-3) are typically RAW and UNMOUNTED.  To keep the LMCache NVMe tier off
# the busy root disk, AAI_NVME_AUTO=1 (default) auto-detects a dedicated local
# NVMe (needs passwordless sudo on the node), in priority order:
#   1) reuse an aai-lmcache volume already mounted at AAI_NVME_MOUNT (no format),
#   2) format + mount a raw, unmounted, non-root NVMe spare at AAI_NVME_MOUNT,
#   3) use a non-root NVMe filesystem that is already mounted somewhere, else
#   4) fall back to node-local /tmp on the root disk (the original behavior).
# Formatting (case 2) only ever touches a disk that is non-root, unmounted, has
# NO filesystem, is not swap, and is >=100G.  Set AAI_NVME_AUTO=0 to force (4);
# set NVME_DATA / AAI_LOCAL_ROOT explicitly to pin a path (auto-detect skipped).
# A provisioned mount is left mounted for reuse by later jobs on the same node.
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

# Storage for the LMCache tiers (see the storage note above).  With
# AAI_NVME_AUTO=1 (default) the tiers are pointed at a dedicated local NVMe;
# provision_storage() (below, once log() exists) finalizes the paths.  An
# explicit NVME_DATA / AAI_LOCAL_ROOT is always honored as-is.
AAI_NVME_AUTO="${AAI_NVME_AUTO:-1}"
AAI_NVME_MOUNT="${AAI_NVME_MOUNT:-/mnt/aai-lmcache}"   # where a provisioned spare is mounted
AAI_NVME_LABEL="${AAI_NVME_LABEL:-aai-lmcache}"        # ext4 label to recognize/reuse our fs
_AAI_NVME_DATA_PINNED="${NVME_DATA:+1}"
_AAI_LOCAL_ROOT_PINNED="${AAI_LOCAL_ROOT:+1}"

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
# Optional RoPE context extension (YaRN etc.) for long-ISL runs.  Only RoPE-based
# models support this; ALiBi/absolute-position models will error or ignore it.
# Two ways to set it (applied to ALL arms via vLLM --hf-overrides):
#   1. VLM_ROPE_SCALING = raw JSON for the HF rope_scaling dict, e.g.
#        VLM_ROPE_SCALING='{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
#   2. VLM_YARN_FACTOR (+ optional VLM_YARN_ORIG_LEN) = convenience shortcut that
#      builds the YaRN dict for you; VLM_YARN_ORIG_LEN defaults to 32768 (Qwen2.5).
# You MUST also raise VLM_MAX_MODEL_LEN to the extended length (e.g. factor 4.0 on a
# 32768 native window => set VLM_MAX_MODEL_LEN=131072).  Empty => no scaling.
VLM_ROPE_SCALING="${VLM_ROPE_SCALING:-}"
VLM_YARN_FACTOR="${VLM_YARN_FACTOR:-}"
VLM_YARN_ORIG_LEN="${VLM_YARN_ORIG_LEN:-32768}"
# Build the optional --hf-overrides arg once (empty array when no scaling asked).
if [[ -z "${VLM_ROPE_SCALING}" && -n "${VLM_YARN_FACTOR}" ]]; then
    VLM_ROPE_SCALING="{\"rope_type\":\"yarn\",\"factor\":${VLM_YARN_FACTOR},\"original_max_position_embeddings\":${VLM_YARN_ORIG_LEN}}"
fi
ROPE_ARGS=()
[[ -n "${VLM_ROPE_SCALING}" ]] && ROPE_ARGS=(--hf-overrides "{\"rope_scaling\": ${VLM_ROPE_SCALING}}")
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-20}"
LMCACHE_NVME_POOL="${LMCACHE_NVME_POOL:-4096}"
# CPU DRAM (L1) tier for the nvme arm.  Default 'false' = KV flows GPU VRAM
# staging -> NVMe (no host DRAM tier).  Set AAI_LOCAL_CPU=true to add an LMCache
# DRAM L1 in front of the NIXL NVMe L2 (a proper DRAM->NVMe tiered cache);
# LMCACHE_MAX_LOCAL_CPU_SIZE is that DRAM L1 size in GB.
AAI_LOCAL_CPU="${AAI_LOCAL_CPU:-false}"
LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-40}"
# vLLM KVTransferConfig.kv_load_failure_policy for the kvd arms.  Default 'fail'
# (a KV-load failure -> HTTP 500).  Set AAI_KV_LOAD_FAILURE_POLICY=recompute to
# gracefully recompute the prefill when the connector can't return the KV (e.g.
# DRAM L1 / NIXL staging pressure at high concurrency) instead of erroring.
# Empty -> omit the field entirely (vLLM's own default applies).
AAI_KV_LOAD_FAILURE_POLICY="${AAI_KV_LOAD_FAILURE_POLICY:-}"
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
# NIXL AIS_MT staging-buffer device: 'cuda' -> VRAM buffer -> hipFileBufRegister
# succeeds -> TRUE GDS (GPU-direct) reads/writes.  'cpu' -> DRAM buffer ->
# registration fails (err 5013) -> with HIPFILE_ALLOW_COMPAT_MODE=true this falls
# back to hipFile COMPAT mode = POSIX pread/pwrite through a host bounce buffer
# (i.e. NVMe L2 over POSIX, not GDS).  Use cpu+allow-compat to A/B POSIX vs GDS.
AAI_NIXL_BUFFER_DEVICE="${AAI_NIXL_BUFFER_DEVICE:-cuda}"
# O_DIRECT for the NIXL NVMe L2.  true mirrors the GDS path; set false if the POSIX
# (compat) path errors on unaligned direct I/O.
AAI_NIXL_USE_DIRECT_IO="${AAI_NIXL_USE_DIRECT_IO:-true}"
# nvme-arm L2 backend selector: 'nixl' (default) -> NIXL AIS_MT (GDS or POSIX/compat
# per AAI_NIXL_BUFFER_DEVICE).  'local_disk' -> LMCache's NATIVE POSIX LocalDiskBackend
# (one .pt file per chunk, no NIXL) -- a real indexed persistent tier that coexists
# with the DRAM L1.  Lets us A/B native-POSIX-disk vs NIXL for the L2.
AAI_L2_BACKEND="${AAI_L2_BACKEND:-nixl}"
# When AAI_L2_BACKEND=nixl(_posix): which NIXL plugin.  AIS_MT = hipFile (GDS w/ cuda
# buffer, or POSIX/compat w/ cpu buffer).  POSIX = NIXL's FIRST-CLASS POSIX plugin
# (requires cpu buffer).  AAI_L2_BACKEND=nixl_posix forces POSIX + cpu buffer below.
AAI_NIXL_BACKEND="${AAI_NIXL_BACKEND:-AIS_MT}"
# local_disk L2 knobs (only used when AAI_L2_BACKEND=local_disk):
AAI_LOCAL_DISK_SIZE_GB="${AAI_LOCAL_DISK_SIZE_GB:-320}"   # max_local_disk_size (GB)
# use_odirect=false -> buffered I/O through the PAGE CACHE (page cache acts as a fast
# second tier); true -> O_DIRECT (bypass page cache, needs block-aligned chunks).
AAI_L2_ODIRECT="${AAI_L2_ODIRECT:-false}"
# nixl_posix is shorthand for the NIXL first-class POSIX plugin: normalize it here (at
# top level) to POSIX + cpu buffer so the tiers log and the config writer agree.
if [[ "${AAI_L2_BACKEND}" == nixl_posix ]]; then
    AAI_NIXL_BACKEND=POSIX
    AAI_NIXL_BUFFER_DEVICE=cpu
fi
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

# --- Dedicated-NVMe auto-detection (see the storage note near the top) --------
# The disk backing / (e.g. nvme0n1); used to exclude the root disk everywhere.
_root_disk() {
    local src; src="$(findmnt -no SOURCE / 2>/dev/null)"
    [[ -n "${src}" ]] || return 0
    local pk; pk="$(lsblk -no PKNAME "${src}" 2>/dev/null | head -1)"
    printf '%s\n' "${pk:-$(basename "${src}")}"
}

# Echo a base directory on a dedicated local NVMe (or nothing -> caller uses
# /tmp).  Logs go to stderr so stdout carries only the chosen path.
_detect_or_provision_nvme() {
    command -v lsblk   >/dev/null 2>&1 || { log "  nvme-auto: lsblk missing; using /tmp" >&2; return 0; }
    command -v findmnt >/dev/null 2>&1 || { log "  nvme-auto: findmnt missing; using /tmp" >&2; return 0; }
    local rootdev; rootdev="$(_root_disk)"
    log "  nvme-auto: root disk = ${rootdev:-unknown}" >&2

    # (1) Reuse an aai-lmcache volume already mounted at AAI_NVME_MOUNT.
    if mountpoint -q "${AAI_NVME_MOUNT}" 2>/dev/null; then
        log "  nvme-auto: reusing already-mounted ${AAI_NVME_MOUNT}" >&2
        printf '%s\n' "${AAI_NVME_MOUNT}"; return 0
    fi

    # (2) Format + mount a raw, unmounted, non-root NVMe spare.
    local name fstype mnt cand="" szb
    while read -r name fstype mnt; do
        [[ "${name}" =~ ^nvme[0-9]+n[0-9]+$ ]] || continue
        [[ "${name}" == "${rootdev}" ]] && continue
        [[ -n "${fstype}" ]] && continue                                   # whole-disk has a fs -> not raw
        [[ -n "${mnt}" ]] && continue                                      # disk itself mounted
        [[ "$(lsblk -rno NAME "/dev/${name}" 2>/dev/null | wc -l)" -eq 1 ]] || continue  # has partitions
        grep -q "^/dev/${name}[[:space:]]" /proc/swaps 2>/dev/null && continue           # in swap
        szb="$(lsblk -brndo SIZE "/dev/${name}" 2>/dev/null | head -1)"
        [[ -n "${szb}" && "${szb}" -ge 100000000000 ]] || continue          # >= ~100G
        cand="/dev/${name}"; break
    done < <(lsblk -rndo NAME,FSTYPE,MOUNTPOINT 2>/dev/null)

    if [[ -n "${cand}" ]]; then
        log "  nvme-auto: raw spare ${cand} -> mkfs.ext4 (label ${AAI_NVME_LABEL}) + mount ${AAI_NVME_MOUNT}" >&2
        if sudo -n mkfs.ext4 -q -F -L "${AAI_NVME_LABEL}" "${cand}" >/dev/null 2>&1 \
           && sudo -n mkdir -p "${AAI_NVME_MOUNT}" >/dev/null 2>&1 \
           && sudo -n mount -o noatime "${cand}" "${AAI_NVME_MOUNT}" >/dev/null 2>&1; then
            sudo -n chown "$(id -u):$(id -g)" "${AAI_NVME_MOUNT}" >/dev/null 2>&1 \
                || sudo -n chmod 1777 "${AAI_NVME_MOUNT}" >/dev/null 2>&1 || true
            log "  nvme-auto: mounted ${cand} at ${AAI_NVME_MOUNT} (left mounted for reuse; 'sudo umount ${AAI_NVME_MOUNT}' to release)" >&2
            printf '%s\n' "${AAI_NVME_MOUNT}"; return 0
        fi
        log "  nvme-auto: provisioning ${cand} failed (passwordless sudo required?); trying already-mounted NVMe" >&2
    fi

    # (3) Use a non-root NVMe filesystem that is already mounted somewhere.
    local target source fstype2 pk base=""
    while read -r target source fstype2; do
        case "${fstype2}" in ext4|xfs) ;; *) continue ;; esac
        case "${target}" in /|/boot|/boot/*|/usr|/usr/*|/var|/var/*|/etc|/etc/*|/run|/run/*|/home) continue ;; esac
        pk="$(lsblk -no PKNAME "${source}" 2>/dev/null | head -1)"; pk="${pk:-$(basename "${source}")}"
        [[ "${pk}" == nvme* ]] || continue
        [[ "${pk}" == "${rootdev}" ]] && continue
        [[ -w "${target}" ]] || continue
        base="${target}"; break
    done < <(findmnt -rno TARGET,SOURCE,FSTYPE 2>/dev/null)
    if [[ -n "${base}" ]]; then
        log "  nvme-auto: using non-root mounted NVMe fs ${base}" >&2
        printf '%s\n' "${base}"; return 0
    fi

    log "  nvme-auto: no dedicated NVMe found; falling back to node-local /tmp" >&2
    return 0
}

# Finalize AAI_LOCAL_ROOT and the NVME/NFS/slab tier paths.
provision_storage() {
    if [[ -n "${_AAI_LOCAL_ROOT_PINNED}" || -n "${_AAI_NVME_DATA_PINNED}" ]]; then
        log "storage: honoring caller-pinned NVME_DATA/AAI_LOCAL_ROOT (auto-detect skipped)"
    elif [[ "${AAI_NVME_AUTO}" == "1" ]]; then
        log "storage: AAI_NVME_AUTO=1 -- detecting a dedicated local NVMe"
        local base; base="$(_detect_or_provision_nvme)"
        [[ -n "${base}" ]] && AAI_LOCAL_ROOT="${base}/aai-cliff.${SLURM_JOB_ID:-manual}"
    else
        log "storage: AAI_NVME_AUTO=0 -- using node-local /tmp on the root disk"
    fi
    AAI_LOCAL_ROOT="${AAI_LOCAL_ROOT:-/tmp/aai-cliff.${SLURM_JOB_ID:-manual}}"
    NVME_DATA="${NVME_DATA:-${AAI_LOCAL_ROOT}/nvme}"          # L2a AIS_MT pool
    NFS_DATA="${NFS_DATA:-${AAI_LOCAL_ROOT}/nfs}"             # L2b POSIX pool (local stand-in)
    GDS_SLAB_DATA="${GDS_SLAB_DATA:-${AAI_LOCAL_ROOT}/slab}"  # GDS L1 slab
}

log "host        : $(hostname)  job=${SLURM_JOB_ID:-manual}"
log "image       : ${AAI_IMAGE}"
log "model       : ${VLLM_MODEL}  (HF_HOME=${HF_HOME}, offline)"
provision_storage
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
            "${ROPE_ARGS[@]}" \
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
    # nixl_buffer_size is REQUIRED for a cuda (VRAM/GDS) staging buffer, but LMCache
    # REJECTS it for a cpu buffer -- in CPU/POSIX mode NIXL shares LocalCPUBackend's
    # pinned DRAM pool (sized by max_local_cpu_size), so it must be omitted.
    local _bufsize_line=""
    [[ "${AAI_NIXL_BUFFER_DEVICE}" == cuda ]] && _bufsize_line="nixl_buffer_size: ${AAI_NIXL_BUFFER_SIZE}"
    cat > "${LOGDIR}/lmcache-nixl-ais.yml" <<YML
# Generated by run-cliff.sbatch (kvd_v2 nvme arm) -- connector-side NIXL AIS_MT.
chunk_size: 256
# CPU DRAM (L1) tier: local_cpu=true adds an LMCache host-DRAM L1 in front of the
# NIXL NVMe L2 (max_local_cpu_size = DRAM L1 GB, ignored when local_cpu=false).
local_cpu: ${AAI_LOCAL_CPU}
max_local_cpu_size: ${LMCACHE_MAX_LOCAL_CPU_SIZE}
save_decode_cache: true
# Deterministic, seed-independent chunk hashing.  The 'builtin' default uses
# Python hash() (PYTHONHASHSEED-dependent) and doesn't match across LMCache's
# separate lookup subprocess -> external lookups miss -> ext_hit stays 0 even
# under L1 overflow.  sha256_cbor (as in recipies/vllm-lmcache-nixl) is stable.
pre_caching_hash_algorithm: sha256_cbor
${_bufsize_line}
nixl_buffer_device: ${AAI_NIXL_BUFFER_DEVICE}
extra_config:
  enable_nixl_storage: true
  nixl_backend: ${AAI_NIXL_BACKEND}
  nixl_pool_size: ${LMCACHE_NVME_POOL}
  use_direct_io: ${AAI_NIXL_USE_DIRECT_IO}
  nixl_path: /data/nvme/lmcache
YML
}

# nvme arm, AAI_L2_BACKEND=local_disk: LMCache NATIVE POSIX LocalDiskBackend as L2
# (no NIXL).  DRAM L1 (local_cpu) in front of a .pt-file-per-chunk disk tier at
# /data/nvme/lmcache-disk.  use_odirect=false -> buffered (page cache); true -> O_DIRECT.
write_lmcache_localdisk_config() {
    cat > "${LOGDIR}/lmcache-nixl-ais.yml" <<YML
# Generated by run-cliff.sbatch (kvd_v2 nvme arm) -- NATIVE LocalDiskBackend (POSIX), no NIXL.
chunk_size: 256
local_cpu: ${AAI_LOCAL_CPU}
max_local_cpu_size: ${LMCACHE_MAX_LOCAL_CPU_SIZE}
local_disk: "file:///data/nvme/lmcache-disk"
max_local_disk_size: ${AAI_LOCAL_DISK_SIZE_GB}
save_decode_cache: true
pre_caching_hash_algorithm: sha256_cbor
extra_config:
  use_odirect: ${AAI_L2_ODIRECT}
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
    local _l2desc
    case "${AAI_L2_BACKEND}" in
      local_disk)
        mkdir -p "${NVME_DATA}/lmcache-disk"          # host path -> /data/nvme/lmcache-disk in container
        write_lmcache_localdisk_config
        _l2desc="LMCache native LocalDiskBackend (POSIX, $([[ "${AAI_L2_ODIRECT}" == true ]] && echo O_DIRECT || echo buffered/page-cache))" ;;
      nixl_posix)
        # NIXL's first-class POSIX plugin (normalized to POSIX + cpu buffer up top).
        # cpu buffer shares LocalCPUBackend's pool (nixl_buffer_size omitted for cpu).
        write_lmcache_nixl_config
        _l2desc="NIXL POSIX plugin (cpu buffer, shares DRAM pool)" ;;
      *)
        write_lmcache_nixl_config
        local _iomode; [[ "${AAI_NIXL_BUFFER_DEVICE}" == cuda ]] && _iomode="GDS/GPU-direct" || _iomode="POSIX/compat"
        _l2desc="NIXL ${AAI_NIXL_BACKEND}, ${AAI_NIXL_BUFFER_DEVICE} buffer -> ${_iomode}" ;;
    esac
    local _fp=""; [[ -n "${AAI_KV_LOAD_FAILURE_POLICY}" ]] && _fp=",\"kv_load_failure_policy\":\"${AAI_KV_LOAD_FAILURE_POLICY}\""
    local kvcfg="{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"${_fp}}"
    log "launching connector'd vLLM (LMCacheConnectorV1 + ${_l2desc}) on :${KVD_PORT}"
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
            "${ROPE_ARGS[@]}" \
            --block-size 64 --enable-prefix-caching --kv-cache-dtype fp8 \
            --async-scheduling \
            --attention-backend "${AAI_ATTENTION_BACKEND}" \
            --disable-access-log-for-endpoints "/health,/metrics,/v1/models" \
            --kv-transfer-config "${kvcfg}" >/dev/null
}

# gds arm: connector'd vLLM pointing at the standalone lmcache-server (MP).
launch_vllm_kvd_gds() {
    rm_containers aai-cliff-kvd-vllm
    local _fp=""; [[ -n "${AAI_KV_LOAD_FAILURE_POLICY}" ]] && _fp=",\"kv_load_failure_policy\":\"${AAI_KV_LOAD_FAILURE_POLICY}\""
    local kvcfg="{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\"${_fp},\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://localhost\",\"lmcache.mp.port\":${LMCACHE_PORT}}}"
    log "launching connector'd vLLM (LMCacheMPConnector) on :${KVD_PORT}"
    docker run -d --name aai-cliff-kvd-vllm --network host "${GPU_FLAGS[@]}" \
        -v "${HF_HOME}":/hf "${HF_ENV[@]}" \
        "${AAI_IMAGE}" \
            --model "${VLLM_MODEL}" --host 0.0.0.0 --port "${KVD_PORT}" \
            --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
            --gpu-memory-utilization "${VLM_GPU_MEMORY_UTILIZATION}" \
            --max-model-len "${VLM_MAX_MODEL_LEN}" \
            --max-num-batched-tokens "${VLM_MAX_NUM_BATCHED_TOKENS}" \
            "${ROPE_ARGS[@]}" \
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
log "isl/prefix  : ISL=${BENCH_ISL} shared=${BENCH_SHARED_TOK} max_model_len=${VLM_MAX_MODEL_LEN}${VLM_ROPE_SCALING:+ rope_scaling=${VLM_ROPE_SCALING}}"
# Tier sizing — echo the EFFECTIVE values so a mis-propagated override is visible.
# nixl_pool_size is a SLOT count (~4.5 MiB/slot for Qwen2.5-3B); 4096 slots ≈ 18 GiB
# is far too small for long-ISL working sets (a silent 4096 caused the c>=32 cliff
# collapse in jobs 67536798/67537066 — the LMCACHE_NVME_POOL override didn't land).
log "tiers       : l2_backend=${AAI_L2_BACKEND} nixl_backend=${AAI_NIXL_BACKEND} local_cpu=${AAI_LOCAL_CPU} max_local_cpu_size=${LMCACHE_MAX_LOCAL_CPU_SIZE}GB nixl_pool_size=${LMCACHE_NVME_POOL} slots (~$(( LMCACHE_NVME_POOL * 45 / 10 / 1024 )) GiB) nixl_buffer_device=${AAI_NIXL_BUFFER_DEVICE} gds_l1=${LMCACHE_L1_SIZE_GB}GB local_disk=${AAI_LOCAL_DISK_SIZE_GB}GB(odirect=${AAI_L2_ODIRECT})"

if _arm_on vram; then run_arm_vram && STATUS[vram_only]=ok || STATUS[vram_only]=FAIL; else STATUS[vram_only]=skip; fi
if _arm_on nvme; then run_arm_kvd nvme "${KVD_NVME_CSV}" && STATUS[kvd_nvme]=ok || STATUS[kvd_nvme]=FAIL; else STATUS[kvd_nvme]=skip; fi
if _arm_on gds;  then run_arm_kvd gds  "${KVD_GDS_CSV}"  && STATUS[kvd_gds]=ok  || STATUS[kvd_gds]=FAIL;  else STATUS[kvd_gds]=skip;  fi

# --- Plot: baseline vs each kvd backend separately ---------------------------
# Both kvd runs carry arm="kvd_v2", so each is plotted in its own output dir to
# keep the backends distinct, with the shared vram_only baseline overlaid when
# present.  Any arm can be skipped: run_plot drops CSVs that don't exist and
# plots whatever remains (plot_cliff.py already handles one-or-more arms), so a
# single-arm run still produces a chart instead of being silently skipped.
run_plot() {  # $1=out_subdir  $2..=candidate CSV paths (absolute)
    local sub="$1"; shift
    local -a inputs=() csv
    for csv in "$@"; do
        [[ -s "${csv}" ]] && inputs+=( "${LOGDIR_REL}/results/$(basename "${csv}")" )
    done
    (( ${#inputs[@]} )) || { log "plot ${sub}: no CSVs, skipping"; return; }
    mkdir -p "${PLOTS}/${sub}"
    docker run --rm --network host -v "${REPO}":/work -w /work \
        --entrypoint python3 "${AAI_IMAGE}" \
        benchmarks/plot_cliff.py \
            --input "${inputs[@]}" \
            --output-dir "${LOGDIR_REL}/plots/${sub}" \
        && log "plots (${sub}) -> ${PLOTS}/${sub}" || log "plot ${sub}: failed"
}
# Plot each kvd backend that produced a CSV, baseline optional.  If only the
# vram arm ran, still emit the baseline-only cliff chart.
[[ -s "${KVD_NVME_CSV}" ]] && run_plot nvme "${VRAM_CSV}" "${KVD_NVME_CSV}"
[[ -s "${KVD_GDS_CSV}"  ]] && run_plot gds  "${VRAM_CSV}" "${KVD_GDS_CSV}"
if [[ ! -s "${KVD_NVME_CSV}" && ! -s "${KVD_GDS_CSV}" && -s "${VRAM_CSV}" ]]; then
    run_plot vram "${VRAM_CSV}"
fi

# --- Cleanup local NVMe scratch ---------------------------------------------
rm -rf "${AAI_LOCAL_ROOT}" 2>/dev/null || true

log "===================== SUMMARY ====================="
for k in vram_only kvd_nvme kvd_gds; do log "  ${k}: ${STATUS[$k]:-?}"; done
log "  CSVs  : ${RESULTS}"
log "  plots : ${PLOTS}/{vram,nvme,gds} (per arm that ran)"
[[ "${AAI_MONITORING}" == "1" ]] && log "  metrics: ${AAI_METRICS_DIR} (Prometheus TSDB)"
log "==================================================="
# Fail the job if any arm that actually ran failed; skipped arms don't count.
rc=0
for k in vram_only kvd_nvme kvd_gds; do [[ "${STATUS[$k]:-skip}" == FAIL ]] && rc=1; done
exit "${rc}"
