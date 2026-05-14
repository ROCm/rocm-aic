#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Slurm: MI300-class GPU (1), NVMe-capable nodes, vllm-from-kurt build/run/report.
#
# Edit partition/constraints for your site. Slurm does not expand env in #SBATCH;
# override at submit time: sbatch --partition=X --constraint=Y this.sbatch
#
# RUNTIME=docker (default): docker build in recipies/vllm-from-kurt, docker run.
# RUNTIME=pyxis: set CONTAINER_IMAGE to docker://registry/.../image:tag or a .sqsh
# path; no docker build in the job. Uses srun --container-image and
# --container-mounts (NVIDIA Pyxis + Enroot style; flag names may vary by site).
# Shared filesystem images (no registry): place a single .sqsh under e.g.
#   /cluster/images/rocm-aic/${IMAGE_NAME}.sqsh
# and either set CONTAINER_IMAGE to that path, or leave CONTAINER_IMAGE unset and
# rely on KURT_SHARED_IMAGE_DIR (default /cluster/images/rocm-aic) +
# KURT_SHARED_IMAGE_SQSH (default ${KURT_SHARED_IMAGE_DIR}/${IMAGE_NAME}.sqsh)
# when that file exists and is readable on compute nodes.
# vllm-kurt image ENTRYPOINT is /bin/bash: docker run uses --entrypoint /bin/bash … -lc;
# Pyxis uses --no-container-entrypoint so /bin/bash -lc is not prefixed twice.
#
# Optional NVMe block-layer bpftrace (host, CAP_BPF; per-I/O sizes from nr_sector):
#   KURT_NVME_BLK_BPFTRACE=1 traces tracepoint:block:block_rq_issue for the block
#   device backing KURT_NVME_BASE (from KURT_NVME_DEVICE basename, or findmnt).
#   Output: ${REPORT_DIR}/nvme_blk_io.tsv. Templates: bpf/nvme_block_io_trace.bt.in
#   (disk_name field) or bpf/nvme_block_io_trace_dev.bt.in (dev_t, Linux ~6.8+); chosen
#   from /sys/kernel/tracing/events/block/block_rq_issue/format when readable.
#   Not cgroup-scoped: all I/O to that disk (including page cache writeback).
#   KURT_NVME_BLK_BPFTRACE_APT_INSTALL=1 (default) may apt-get install bpftrace;
#   set to 0 on hosts without apt or to skip package install.
#
# Optional LMCache directory VFS bpftrace (host; same bpftrace install as above):
#   KURT_LMCACHE_FS_BPFTRACE defaults to KURT_NVME_BLK_BPFTRACE when unset (so
#   KURT_NVME_BLK_BPFTRACE=1 enables both). Set KURT_LMCACHE_FS_BPFTRACE=0 to keep
#   only block trace, or =1 with NVMe trace off to trace LMCache FS only.
#   kprobe:vfs_read / vfs_write filtered by realpath(${DATA_HOST}/lmcache_test).
#   Override prefix: KURT_LMCACHE_FS_BPFTRACE_PREFIX=/exact/host/path/lmcache_test
#   Output: ${REPORT_DIR}/lmcache_fs_io.tsv (+ lmcache_fs_io.bpftrace.log).
#   Not cgroup-scoped: any process under that path matches (container + host).
#
# Optional nvme-cli SMART snapshots (aggregate counters; not per-I/O sizes):
#   KURT_NVME_SMART_LOG=1 writes nvme_smart_log_job_{start,end}.json when a
#   namespace block device is known (KURT_NVME_DEVICE or findmnt on KURT_NVME_BASE).
#
# NVMe format/mount (DESTRUCTIVE; exclusive Slurm nodes only):
#   KURT_NVME_MKFS=1 KURT_NVME_DEVICE=/dev/nvme2n1 KURT_NVME_MOUNT=/mnt/kurt-nvme
#   Or with auto-pick of first unmounted whole-disk nvme* namespace:
#   KURT_NVME_MKFS=1 KURT_NVME_AUTO_DEVICE=1 KURT_NVME_MOUNT=/mnt/kurt-nvme
#   Discover candidates read-only: tools/kurt-nvme/discover_unmounted_nvme.sh
#
# Scratch for JOB_ROOT (HF + LMCache bind mounts): pick_nvme_base walks
#   /mnt/nvme, /local, /scratch, SLURM_TMPDIR, TMPDIR, /tmp
# and picks the first writable path with >= MIN_FREE_KB KiB free. On many
# nodes /tmp is on the **root filesystem** (same NVMe as /), so tens of GiB of
# Hugging Face cache land on the OS disk. Prefer **KURT_NVME_BASE=/mnt/nvme**
# (or your site’s local NVMe mount). KURT_SKIP_SCRATCH_ON_ROOTFS=1 (default)
# skips candidates whose st_dev matches / so /tmp-on-root is not chosen when
# a non-root candidate exists; set KURT_SKIP_SCRATCH_ON_ROOTFS=0 to allow rootfs.
#
#SBATCH --job-name=vllm-kurt-mi300
#SBATCH --gres=gpu:1
#SBATCH --constraint=MARKHAM&GFX942&NVME
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=.slurm/logs/vllm-from-kurt-mi300-%j.log

set -euo pipefail

# ── configuration (env overrides) ─────────────────────────────
RUNTIME="${RUNTIME:-docker}"
DOCKER="${DOCKER:-docker}"
IMAGE_NAME="${IMAGE_NAME:-vllm-kurt}"
# Default shared Pyxis/Enroot image directory (override per site).
KURT_SHARED_IMAGE_DIR="${KURT_SHARED_IMAGE_DIR:-/cluster/images/rocm-aic}"
KURT_SHARED_IMAGE_SQSH="${KURT_SHARED_IMAGE_SQSH:-${KURT_SHARED_IMAGE_DIR}/${IMAGE_NAME}.sqsh}"
REPO_DIR="${SLURM_SUBMIT_DIR}"
RECIPE_DIR="${REPO_DIR}/recipies/vllm-from-kurt"
LOG_DIR="${REPO_DIR}/.slurm/logs"
CONTAINER_HF_HOME="${CONTAINER_HF_HOME:-/hf}"
CONTAINER_DATA_DIR="${CONTAINER_DATA_DIR:-/data}"
BACKEND="${BACKEND:-native}"
ROCM_ARCH="${ROCM_ARCH:-}"
MIN_FREE_KB="${MIN_FREE_KB:-10485760}"
# 1: do not use /tmp (etc.) when it is the same filesystem as / — avoids filling OS disk.
KURT_SKIP_SCRATCH_ON_ROOTFS="${KURT_SKIP_SCRATCH_ON_ROOTFS:-1}"
READY_TIMEOUT="${READY_TIMEOUT:-7200}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HOME}/.batesste-hugging-face-read-march-2026.token}"
GPU_IDX="${GPU_IDX:-0}"
BENCH_EXTRA=(--repeat-count 3 --completions --trim-fraction 0.1)
if [[ -n "${BENCH_EXTRA_ARGS:-}" ]]; then
  read -r -a _bench_extra <<< "${BENCH_EXTRA_ARGS}"
  BENCH_EXTRA+=("${_bench_extra[@]}")
fi

mkdir -p "${LOG_DIR}"
PHASE_RC=0
BUILD_RC=""
RUN_RC=""
T_BUILD_START=""
T_BUILD_END=""
T_RUN_START=""
T_RUN_END=""
IMAGE_ID_LINE=""
KURT_NVME_BLK_BPFTRACE_PID=""
KURT_LMCACHE_FS_BPFTRACE_PID=""

# stderr: must not write to stdout — e.g. KURT_NVME_BASE="$(pick_nvme_base)" captures stdout.
log() { echo "[$(date -Iseconds)] $*" >&2; }

require_hf_token() {
  if [[ -z "${HF_TOKEN:-}" ]] && [[ -r "${HF_TOKEN_FILE}" ]]; then
    HF_TOKEN="$(tr -d '\r\n' < "${HF_TOKEN_FILE}")"
  fi
  if [[ -z "${HF_TOKEN:-}" ]]; then
    log "ERROR: set HF_TOKEN or a readable HF_TOKEN_FILE (default: ${HF_TOKEN_FILE})"
    exit 1
  fi
  export HF_TOKEN
}

detect_rocm_arch() {
  if [[ -n "${ROCM_ARCH}" ]]; then
    echo "${ROCM_ARCH}"
    return
  fi
  local a
  a="$(rocm_agent_enumerator 2>/dev/null | grep -E '^gfx' | grep -v '^gfx000' | head -1 || true)"
  if [[ -n "${a}" ]]; then
    echo "${a}"
    return
  fi
  echo "gfx942"
}

# Optional: KURT_NVME_MKFS=1 KURT_NVME_DEVICE=/dev/nvme0n1 KURT_NVME_MOUNT=/mnt/kurt-nvme (DANGEROUS on shared systems)
kurt_nvme_auto_pick_device() {
  if [[ "${KURT_NVME_AUTO_DEVICE:-0}" != "1" ]]; then
    return 0
  fi
  if [[ -n "${KURT_NVME_DEVICE:-}" ]]; then
    return 0
  fi
  local name dev
  while read -r name type _; do
    [[ "${type}" == "disk" ]] || continue
    [[ "${name}" =~ ^nvme[0-9]+n[0-9]+$ ]] || continue
    dev="/dev/${name}"
    [[ -b "${dev}" ]] || continue
    if lsblk -nr "${dev}" -o MOUNTPOINT 2>/dev/null | awk 'NF && $1 != "" && $1 != "-" { f = 1 } END { exit !f }'; then
      continue
    fi
    KURT_NVME_DEVICE="${dev}"
    export KURT_NVME_DEVICE
    log "KURT_NVME_AUTO_DEVICE: picked ${KURT_NVME_DEVICE} (exclusive nodes only)"
    return 0
  done < <(lsblk -d -n -o NAME,TYPE 2>/dev/null || true)
  log "WARN: KURT_NVME_AUTO_DEVICE=1 but no unmounted whole-disk nvme* namespace found"
  return 1
}

maybe_mkfs_mount_nvme() {
  kurt_nvme_auto_pick_device || true
  if [[ "${KURT_NVME_MKFS:-0}" != "1" ]]; then
    return 0
  fi
  if [[ -z "${KURT_NVME_DEVICE:-}" ]] || [[ -z "${KURT_NVME_MOUNT:-}" ]]; then
    log "ERROR: KURT_NVME_MKFS=1 requires KURT_NVME_DEVICE and KURT_NVME_MOUNT"
    exit 1
  fi
  mkdir -p "${KURT_NVME_MOUNT}"
  if ! mountpoint -q "${KURT_NVME_MOUNT}"; then
    if [[ ! -b "${KURT_NVME_DEVICE}" ]]; then
      log "ERROR: KURT_NVME_DEVICE is not a block device: ${KURT_NVME_DEVICE}"
      exit 1
    fi
    if ! blkid "${KURT_NVME_DEVICE}" &>/dev/null; then
      log "WARN: formatting ${KURT_NVME_DEVICE} as xfs (KURT_NVME_MKFS=1)"
      mkfs.xfs -f "${KURT_NVME_DEVICE}"
    fi
    mount "${KURT_NVME_DEVICE}" "${KURT_NVME_MOUNT}"
  fi
  export KURT_NVME_BASE="${KURT_NVME_MOUNT}"
}

free_kb() {
  local path="$1"
  df -Pk "${path}" 2>/dev/null | awk 'NR==2 {print $4}'
}

# True if path is on the same mounted filesystem as / (same st_dev).
kurt_path_same_stat_dev_as_root() {
  local p="$1"
  local dr dp
  dr="$(stat -c '%d' / 2>/dev/null)" || return 1
  dp="$(stat -c '%d' "${p}" 2>/dev/null)" || return 1
  [[ -n "${dr}" ]] && [[ "${dr}" == "${dp}" ]]
}

pick_nvme_base() {
  maybe_mkfs_mount_nvme
  if [[ -n "${KURT_NVME_BASE:-}" ]]; then
    if [[ ! -d "${KURT_NVME_BASE}" ]] || [[ ! -w "${KURT_NVME_BASE}" ]]; then
      log "ERROR: KURT_NVME_BASE not a writable directory: ${KURT_NVME_BASE}"
      exit 1
    fi
    echo "${KURT_NVME_BASE}"
    return
  fi
  local c free
  for c in /mnt/nvme /local /scratch "${SLURM_TMPDIR:-}" "${TMPDIR:-/tmp}" /tmp; do
    [[ -z "${c}" ]] && continue
    [[ -d "${c}" ]] || continue
    [[ -w "${c}" ]] || continue
    free="$(free_kb "${c}" || echo 0)"
    if [[ "${free}" =~ ^[0-9]+$ ]] && [[ "${free}" -ge "${MIN_FREE_KB}" ]]; then
      if [[ "${KURT_SKIP_SCRATCH_ON_ROOTFS:-1}" == "1" ]] && kurt_path_same_stat_dev_as_root "${c}"; then
        log "WARN: skipping scratch candidate ${c}: same filesystem as / (st_dev); use KURT_NVME_BASE for dedicated NVMe or KURT_SKIP_SCRATCH_ON_ROOTFS=0"
        continue
      fi
      echo "${c}"
      return
    fi
  done
  log "ERROR: no usable scratch (writable + >= ${MIN_FREE_KB} KiB free); set KURT_NVME_BASE to a local NVMe path, adjust MIN_FREE_KB, or KURT_SKIP_SCRATCH_ON_ROOTFS=0"
  exit 1
}

prepare_hf_cache_host() {
  local hf="$1"
  mkdir -p \
    "${hf}/hub" \
    "${hf}/datasets" \
    "${hf}/vllm" \
    "${hf}/vllm_config" \
    "${hf}/torch" \
    "${hf}/torch_inductor"
}

write_job_metadata() {
  local out="$1"
  {
    echo "=== job metadata ==="
    echo "hostname: $(hostname)"
    echo "date: $(date -Iseconds)"
    echo "RUNTIME: ${RUNTIME}"
    echo "CONTAINER_IMAGE: ${CONTAINER_IMAGE:-}"
    echo "KURT_SHARED_IMAGE_DIR: ${KURT_SHARED_IMAGE_DIR:-}"
    echo "KURT_SHARED_IMAGE_SQSH: ${KURT_SHARED_IMAGE_SQSH:-}"
    echo "ROCM_ARCH: ${ROCM_ARCH}"
    echo "IMAGE_NAME: ${IMAGE_NAME}"
    echo "BACKEND: ${BACKEND}"
    echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
    echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-}"
    echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-}"
    echo "JOB_ROOT: ${JOB_ROOT}"
    echo "IMAGE_ID: ${IMAGE_ID_LINE}"
    echo "BUILD_RC: ${BUILD_RC:-}"
    echo "RUN_RC: ${RUN_RC:-}"
    echo "T_BUILD: ${T_BUILD_START:-} .. ${T_BUILD_END:-}"
    echo "T_RUN: ${T_RUN_START:-} .. ${T_RUN_END:-}"
    echo "KURT_NVME_DEVICE: ${KURT_NVME_DEVICE:-}"
    echo "KURT_NVME_MOUNT: ${KURT_NVME_MOUNT:-}"
    echo "KURT_NVME_MKFS: ${KURT_NVME_MKFS:-0}"
    echo "KURT_NVME_AUTO_DEVICE: ${KURT_NVME_AUTO_DEVICE:-0}"
    echo "KURT_SKIP_SCRATCH_ON_ROOTFS: ${KURT_SKIP_SCRATCH_ON_ROOTFS:-1}"
    echo "KURT_NVME_BLK_BPFTRACE: ${KURT_NVME_BLK_BPFTRACE:-0}"
    echo "KURT_LMCACHE_FS_BPFTRACE: ${KURT_LMCACHE_FS_BPFTRACE:-0}"
    echo "KURT_LMCACHE_FS_BPFTRACE_PREFIX: ${KURT_LMCACHE_FS_BPFTRACE_PREFIX:-}"
    echo "KURT_NVME_SMART_LOG: ${KURT_NVME_SMART_LOG:-0}"
    if [[ -f "${REPORT_DIR}/nvme_blk_io_disk_name.txt" ]]; then
      echo "nvme_blk disk_name: $(tr -d '\r\n' <"${REPORT_DIR}/nvme_blk_io_disk_name.txt")"
    fi
    if [[ -f "${REPORT_DIR}/nvme_blk_io_lb_size.txt" ]]; then
      echo "nvme_blk logical_block_size: $(tr -d '\r\n' <"${REPORT_DIR}/nvme_blk_io_lb_size.txt")"
    fi
    if [[ -f "${REPORT_DIR}/nvme_blk_io.tsv" ]]; then
      echo "nvme_blk_io.tsv: ${REPORT_DIR}/nvme_blk_io.tsv lines=$(wc -l <"${REPORT_DIR}/nvme_blk_io.tsv" | tr -d ' ')"
    fi
    if [[ -f "${REPORT_DIR}/lmcache_fs_io_trace_prefix.txt" ]]; then
      echo "lmcache_fs path_prefix: $(tr -d '\r\n' <"${REPORT_DIR}/lmcache_fs_io_trace_prefix.txt")"
    fi
    if [[ -f "${REPORT_DIR}/lmcache_fs_io.tsv" ]]; then
      echo "lmcache_fs_io.tsv: ${REPORT_DIR}/lmcache_fs_io.tsv lines=$(wc -l <"${REPORT_DIR}/lmcache_fs_io.tsv" | tr -d ' ')"
    fi
    echo ""
    echo "=== lsblk (nvme lines) ==="
    lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,PKNAME,MODEL 2>/dev/null | grep -E '^NAME|^nvme' || echo "(no nvme lines from lsblk)"
    echo ""
    echo "=== rocm-smi ==="
    if command -v rocm-smi &>/dev/null; then
      rocm-smi --showproductname --showmeminfo vram --showuse 2>&1 || true
    else
      echo "(rocm-smi not in PATH)"
    fi
    echo ""
    echo "=== rocminfo (head) ==="
    if command -v rocminfo &>/dev/null; then
      rocminfo 2>&1 | head -80 || true
    else
      echo "(rocminfo not in PATH)"
    fi
  } >"${out}"
}

append_server_log() {
  local srv="${JOB_ROOT}/data/server.txt"
  local out="$1"
  {
    echo ""
    echo "=== server.txt (full) ==="
    if [[ -f "${srv}" ]]; then
      cat "${srv}"
    else
      echo "(missing ${srv})"
    fi
  } >>"${out}"
}

# Copy LMCache observability artifacts from the data mount into REPORT_DIR so
# report.md and Slurm log bundles reference one tree.
sync_lmcache_report_files() {
  local f
  for f in "${DATA_HOST}"/lmcache_internal_api_*.*; do
    [[ -e "${f}" ]] || continue
    cp -a "${f}" "${REPORT_DIR}/" 2>/dev/null || true
  done
  if [[ -d "${DATA_HOST}/lmcache_chunk_stats" ]]; then
    rm -rf "${REPORT_DIR}/lmcache_chunk_stats" 2>/dev/null || true
    cp -a "${DATA_HOST}/lmcache_chunk_stats" "${REPORT_DIR}/" 2>/dev/null || true
  fi
}

write_report_md() {
  local out="$1"
  local json="${REPORT_DIR}/long_doc_qa.json"
  {
    echo "# vllm-from-kurt job report"
    echo ""
    echo "- Job ID: ${SLURM_JOB_ID:-n/a}"
    echo "- RUNTIME: ${RUNTIME}"
    echo "- BACKEND: ${BACKEND}"
    echo "- ROCM_ARCH: ${ROCM_ARCH}"
    echo "- JOB_ROOT: ${JOB_ROOT}"
    echo "- long_doc_qa JSON: \`${json}\`"
    echo ""
    echo "## LMCache observability"
    echo ""
    if ls "${REPORT_DIR}"/lmcache_internal_api_*.* &>/dev/null; then
      echo "Internal API snapshots (**/chunk_statistics/status** JSON and **/metrics** text):"
      ls -1 "${REPORT_DIR}"/lmcache_internal_api_*.* | while IFS= read -r p; do
        echo "- \`${p}\`"
      done
    else
      echo "(No **lmcache_internal_api_*** files; see **\`${DATA_HOST}/\`** if the run used a different layout.)"
    fi
    if [[ -d "${REPORT_DIR}/lmcache_chunk_stats" ]]; then
      echo ""
      echo "- Chunk hash directory (**file_hash**): \`${REPORT_DIR}/lmcache_chunk_stats/\`"
    fi
    if [[ -f "${REPORT_DIR}/nvme_blk_io.tsv" ]]; then
      echo ""
      echo "- Optional **bpftrace** block I/O for the NVMe disk (see **nvme_blk_io_disk_name.txt**): \`${REPORT_DIR}/nvme_blk_io.tsv\`"
    fi
    if [[ -f "${REPORT_DIR}/lmcache_fs_io.tsv" ]]; then
      echo ""
      echo "- **bpftrace** VFS read/write under LMCache host dir (see **lmcache_fs_io_trace_prefix.txt**): \`${REPORT_DIR}/lmcache_fs_io.tsv\`"
    fi
    if ls "${REPORT_DIR}"/nvme_smart_log_job_*.json &>/dev/null; then
      echo ""
      echo "- **nvme-cli** SMART snapshots: \`${REPORT_DIR}/nvme_smart_log_job_*.json\` (aggregate counters; diff start vs end for totals)"
    fi
    echo ""
    echo "Per-call **store**/**retrieve** detail beyond chunk statistics uses **KURT_LMCACHE_ENABLE_KV_EVENTS** (default **1**) and **KURT_LMCACHE_LOG_LEVEL** (default **INFO** → **LMCACHE_LOG_LEVEL** in **server.txt**). Set **KURT_LMCACHE_ENABLE_KV_EVENTS=0** to disable KV events."
    echo ""
    echo "## Summary text"
    echo ""
    echo '```'
    cat "${REPORT_DIR}/summary.txt"
    echo '```'
    echo ""
    if command -v jq &>/dev/null && [[ -f "${json}" ]]; then
      echo "## long_doc_qa (jq .)"
      echo ""
      echo '```json'
      jq . "${json}" 2>/dev/null || cat "${json}"
      echo '```'
    fi
  } >"${out}"
}

copy_to_submit_logs() {
  local stamp="${LOG_DIR}/vllm-from-kurt-mi300-${SLURM_JOB_ID:-manual}"
  mkdir -p "${stamp}"
  cp -a "${REPORT_DIR}/summary.txt" "${stamp}/" 2>/dev/null || true
  cp -a "${REPORT_DIR}/metadata.txt" "${stamp}/" 2>/dev/null || true
  cp -a "${REPORT_DIR}/report.md" "${stamp}/" 2>/dev/null || true
  cp -a "${REPORT_DIR}/long_doc_qa.json" "${stamp}/" 2>/dev/null || true
  if ls "${REPORT_DIR}"/lmcache_internal_api_*.* &>/dev/null; then
    cp -a "${REPORT_DIR}"/lmcache_internal_api_*.* "${stamp}/" 2>/dev/null || true
  fi
  if [[ -d "${REPORT_DIR}/lmcache_chunk_stats" ]]; then
    cp -a "${REPORT_DIR}/lmcache_chunk_stats" "${stamp}/" 2>/dev/null || true
  fi
  if [[ -f "${REPORT_DIR}/nvme_blk_io.tsv" ]]; then
    cp -a "${REPORT_DIR}/nvme_blk_io.tsv" "${stamp}/" 2>/dev/null || true
  fi
  if [[ -f "${REPORT_DIR}/nvme_block_io_trace.gen.bt" ]]; then
    cp -a "${REPORT_DIR}/nvme_block_io_trace.gen.bt" "${stamp}/" 2>/dev/null || true
  fi
  if [[ -f "${REPORT_DIR}/nvme_blk_io.bpftrace.log" ]]; then
    cp -a "${REPORT_DIR}/nvme_blk_io.bpftrace.log" "${stamp}/" 2>/dev/null || true
  fi
  if [[ -f "${REPORT_DIR}/nvme_blk_io_disk_name.txt" ]]; then
    cp -a "${REPORT_DIR}/nvme_blk_io_disk_name.txt" "${stamp}/" 2>/dev/null || true
  fi
  if [[ -f "${REPORT_DIR}/nvme_blk_io_trace_variant.txt" ]]; then
    cp -a "${REPORT_DIR}/nvme_blk_io_trace_variant.txt" "${stamp}/" 2>/dev/null || true
  fi
  if [[ -f "${REPORT_DIR}/nvme_blk_io_dev_t.txt" ]]; then
    cp -a "${REPORT_DIR}/nvme_blk_io_dev_t.txt" "${stamp}/" 2>/dev/null || true
  fi
  if [[ -f "${REPORT_DIR}/nvme_blk_io_trace_dev_path.txt" ]]; then
    cp -a "${REPORT_DIR}/nvme_blk_io_trace_dev_path.txt" "${stamp}/" 2>/dev/null || true
  fi
  if [[ -f "${REPORT_DIR}/lmcache_fs_io.tsv" ]]; then
    cp -a "${REPORT_DIR}/lmcache_fs_io.tsv" "${stamp}/" 2>/dev/null || true
  fi
  if [[ -f "${REPORT_DIR}/lmcache_fs_io_trace.gen.bt" ]]; then
    cp -a "${REPORT_DIR}/lmcache_fs_io_trace.gen.bt" "${stamp}/" 2>/dev/null || true
  fi
  if [[ -f "${REPORT_DIR}/lmcache_fs_io.bpftrace.log" ]]; then
    cp -a "${REPORT_DIR}/lmcache_fs_io.bpftrace.log" "${stamp}/" 2>/dev/null || true
  fi
  if [[ -f "${REPORT_DIR}/lmcache_fs_io_trace_prefix.txt" ]]; then
    cp -a "${REPORT_DIR}/lmcache_fs_io_trace_prefix.txt" "${stamp}/" 2>/dev/null || true
  fi
  if ls "${REPORT_DIR}"/nvme_smart_log_job_*.json &>/dev/null; then
    cp -a "${REPORT_DIR}"/nvme_smart_log_job_*.json "${stamp}/" 2>/dev/null || true
  fi
  if [[ -f "${REPORT_DIR}/nvme_smart_log.log" ]]; then
    cp -a "${REPORT_DIR}/nvme_smart_log.log" "${stamp}/" 2>/dev/null || true
  fi
  log "Copied report artifacts to ${stamp}/"
}

# ── start ───────────────────────────────────────────────────────
log "RUNTIME=${RUNTIME} RECIPE_DIR=${RECIPE_DIR}"
require_hf_token

if [[ ! -f "${RECIPE_DIR}/Dockerfile" ]]; then
  log "ERROR: missing ${RECIPE_DIR}/Dockerfile (submit from repo root?)"
  exit 1
fi

ROCM_ARCH="$(detect_rocm_arch)"
log "Using ROCM_ARCH=${ROCM_ARCH}"

KURT_NVME_BASE="$(pick_nvme_base)"
log "Using KURT_NVME_BASE=${KURT_NVME_BASE}"
JOB_ROOT="${KURT_NVME_BASE}/vllm-kurt-${SLURM_JOB_ID:-0}"
DATA_HOST="${JOB_ROOT}/data"
HF_HOST="${JOB_ROOT}/hf"
REPORT_DIR="${JOB_ROOT}/report"
mkdir -p "${DATA_HOST}" "${REPORT_DIR}"
mkdir -p "${DATA_HOST}/lmcache_test"
prepare_hf_cache_host "${HF_HOST}"

KURT_NVME_BLK_BPFTRACE="${KURT_NVME_BLK_BPFTRACE:-0}"
KURT_LMCACHE_FS_BPFTRACE="${KURT_LMCACHE_FS_BPFTRACE:-${KURT_NVME_BLK_BPFTRACE}}"
KURT_NVME_BLK_BPFTRACE_APT_INSTALL="${KURT_NVME_BLK_BPFTRACE_APT_INSTALL:-1}"
KURT_NVME_SMART_LOG="${KURT_NVME_SMART_LOG:-0}"
export KURT_NVME_BLK_BPFTRACE KURT_LMCACHE_FS_BPFTRACE KURT_NVME_BLK_BPFTRACE_APT_INSTALL KURT_NVME_SMART_LOG
[[ -n "${KURT_NVME_BLK_DISK_NAME:-}" ]] && export KURT_NVME_BLK_DISK_NAME
[[ -n "${KURT_LMCACHE_FS_BPFTRACE_PREFIX:-}" ]] && export KURT_LMCACHE_FS_BPFTRACE_PREFIX

export ROCR_VISIBLE_DEVICES="${GPU_IDX}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# long_doc_qa (LMCache) default prompts + --output-len must fit under vLLM
# --max-model-len: the server rejects when prompt_tokens + max_new_tokens exceed
# the context cap. Job 302650 hit: ~16285 input + 100 output => need >= 16385
# (16384 failed by one token). Extra slack covers template overhead and longer
# rounds; raise further if you pass larger --document-length / --num-documents
# via BENCH_EXTRA_ARGS (README suggests e.g. 131072 for MI300 stress).
KURT_MAX_MODEL_LEN="${KURT_MAX_MODEL_LEN:-32768}"
KURT_MAX_NUM_BATCHED_TOKENS="${KURT_MAX_NUM_BATCHED_TOKENS:-8192}"

BOOTCFG="/boot/config-$(uname -r)"
BOOT_MOUNT=()
if [[ -r "${BOOTCFG}" ]]; then
  BOOT_MOUNT=( -v "${BOOTCFG}:${BOOTCFG}:ro" )
fi

# Command run inside the container (host-expanded args, safely quoted for bash -lc).
BENCH_CMD=(
  python3 /app/run_long_doc_qa.py
  --backend "${BACKEND}"
  --ready-timeout "${READY_TIMEOUT}"
  --json-output
)
BENCH_CMD+=("${BENCH_EXTRA[@]}")
BENCH_CMD_QUOTED="$(printf '%q ' "${BENCH_CMD[@]}")"

export CONTAINER_HF_HOME CONTAINER_DATA_DIR
export KURT_CONTAINER_DATA_DIR="${CONTAINER_DATA_DIR}"

export_for_container() {
  export HF_HOME="${CONTAINER_HF_HOME}"
  export HUGGINGFACE_HUB_CACHE="${CONTAINER_HF_HOME}/hub"
  export HF_HUB_CACHE="${CONTAINER_HF_HOME}/hub"
  export HF_DATASETS_CACHE="${CONTAINER_HF_HOME}/datasets"
  export VLLM_CACHE_ROOT="${CONTAINER_HF_HOME}/vllm"
  export VLLM_CONFIG_ROOT="${CONTAINER_HF_HOME}/vllm_config"
  export TORCH_HOME="${CONTAINER_HF_HOME}/torch"
  export TORCHINDUCTOR_CACHE_DIR="${CONTAINER_HF_HOME}/torch_inductor"
  export KURT_CONTAINER_DATA_DIR="${CONTAINER_DATA_DIR}"
  export ROCR_VISIBLE_DEVICES="${GPU_IDX}"
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
  export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
  export KURT_LMCACHE_ENABLE_CHUNK_STATISTICS="${KURT_LMCACHE_ENABLE_CHUNK_STATISTICS:-1}"
  export KURT_LMCACHE_CHUNK_STATISTICS_STRATEGY="${KURT_LMCACHE_CHUNK_STATISTICS_STRATEGY:-file_hash}"
  export KURT_LMCACHE_COLLECT_INTERNAL_API="${KURT_LMCACHE_COLLECT_INTERNAL_API:-1}"
  export KURT_LMCACHE_INTERNAL_API_MAX_OFFSET="${KURT_LMCACHE_INTERNAL_API_MAX_OFFSET:-3}"
  export KURT_LMCACHE_ENABLE_KV_EVENTS="${KURT_LMCACHE_ENABLE_KV_EVENTS:-1}"
  export KURT_LMCACHE_LOG_LEVEL="${KURT_LMCACHE_LOG_LEVEL:-INFO}"
  export KURT_LMCACHE_INTERNAL_API_VERBOSE="${KURT_LMCACHE_INTERNAL_API_VERBOSE:-0}"
  [[ -n "${VLLM_MODEL:-}" ]] && export VLLM_MODEL
  [[ -n "${KURT_MAX_MODEL_LEN:-}" ]] && export KURT_MAX_MODEL_LEN
  [[ -n "${KURT_MAX_NUM_BATCHED_TOKENS:-}" ]] && export KURT_MAX_NUM_BATCHED_TOKENS
  [[ -n "${KURT_ENFORCE_EAGER:-}" ]] && export KURT_ENFORCE_EAGER
  [[ -n "${KURT_GPU_MEMORY_UTILIZATION:-}" ]] && export KURT_GPU_MEMORY_UTILIZATION
}

# bpftrace optional apt install (for KURT_NVME_BLK_BPFTRACE=1 only).
kurt_nvme_bpftrace_apt_install() {
  if command -v bpftrace &>/dev/null; then
    return 0
  fi
  if [[ "${KURT_NVME_BLK_BPFTRACE_APT_INSTALL:-1}" != "1" ]]; then
    return 1
  fi
  if ! command -v apt-get &>/dev/null; then
    log "WARN: bpftrace not in PATH and apt-get missing; cannot apt install"
    return 1
  fi
  log "Installing bpftrace via apt-get (DEBIAN_FRONTEND=noninteractive) …"
  export DEBIAN_FRONTEND=noninteractive
  set +e
  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    apt-get update -qq
    apt-get install -y bpftrace
  else
    if ! sudo -n true 2>/dev/null; then
      log "WARN: bpftrace missing; need root or passwordless sudo for apt-get install"
      set -e
      return 1
    fi
    sudo -n env DEBIAN_FRONTEND=noninteractive apt-get update -qq
    sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y bpftrace
  fi
  local rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    log "WARN: apt-get install bpftrace failed rc=${rc}"
    return 1
  fi
  if ! command -v bpftrace &>/dev/null; then
    log "WARN: bpftrace still not in PATH after apt install"
    return 1
  fi
  log "bpftrace installed: $(command -v bpftrace)"
  return 0
}

kurt_nvme_bpftrace_ensure() {
  if command -v bpftrace &>/dev/null; then
    return 0
  fi
  kurt_nvme_bpftrace_apt_install || true
  if ! command -v bpftrace &>/dev/null; then
    log "WARN: bpftrace not in PATH; skipping NVMe block bpftrace"
    return 1
  fi
  return 0
}

resolve_kurt_nvme_blk_disk_name() {
  if [[ -n "${KURT_NVME_BLK_DISK_NAME:-}" ]]; then
    echo "${KURT_NVME_BLK_DISK_NAME}"
    return 0
  fi
  local src pk
  if [[ -n "${KURT_NVME_DEVICE:-}" ]]; then
    basename "$(readlink -f "${KURT_NVME_DEVICE}")"
    return 0
  fi
  src="$(findmnt -nro SOURCE --target "${KURT_NVME_BASE}" 2>/dev/null || true)"
  if [[ -z "${src}" ]]; then
    return 1
  fi
  pk="$(lsblk -ndo PKNAME "${src}" 2>/dev/null | head -1 || true)"
  if [[ -n "${pk}" ]]; then
    echo "${pk}"
  else
    basename "$(readlink -f "${src}")"
  fi
}

nvme_blk_io_trace_gen() {
  local tpl_disk="${RECIPE_DIR}/bpf/nvme_block_io_trace.bt.in"
  local tpl_dev="${RECIPE_DIR}/bpf/nvme_block_io_trace_dev.bt.in"
  local out="${REPORT_DIR}/nvme_block_io_trace.gen.bt"
  rm -f "${REPORT_DIR}/nvme_blk_io_dev_t.txt" "${REPORT_DIR}/nvme_blk_io_trace_variant.txt" \
    "${REPORT_DIR}/nvme_blk_io_trace_dev_path.txt"
  local dn lb variant devp devt
  local _tpf="/sys/kernel/tracing/events/block/block_rq_issue/format"
  if ! dn="$(resolve_kurt_nvme_blk_disk_name)"; then
    log "ERROR: set KURT_NVME_DEVICE or KURT_NVME_BLK_DISK_NAME for nvme block trace"
    return 1
  fi
  if [[ ! -d "/sys/block/${dn}" ]]; then
    log "WARN: /sys/block/${dn} missing; using logical_block_size 512"
    lb=512
  else
    lb="$(tr -d '\r\n' <"/sys/block/${dn}/queue/logical_block_size" 2>/dev/null || echo 512)"
  fi
  if [[ ! "${lb}" =~ ^[0-9]+$ ]] || [[ "${lb}" -lt 1 ]] || [[ "${lb}" -gt 1048576 ]]; then
    lb=512
  fi
  variant="dev"
  if sudo -n test -r "${_tpf}" 2>/dev/null && sudo -n grep -qE '[[:space:]]field:.*disk_name' "${_tpf}" 2>/dev/null; then
    variant="disk"
  elif ! sudo -n test -r "${_tpf}" 2>/dev/null; then
    log "WARN: cannot read ${_tpf} with sudo -n; using dev_t bpftrace template (Linux ~6.8+)"
  fi
  if [[ "${variant}" == "disk" ]]; then
    if [[ ! -f "${tpl_disk}" ]]; then
      log "ERROR: missing bpftrace template ${tpl_disk}"
      return 1
    fi
    python3 -c 'import json, pathlib, sys
dn, lb, tplp, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
lit = json.dumps(dn)
text = pathlib.Path(tplp).read_text()
text = text.replace("@@DISK_NAME_STR@@", lit)
text = text.replace("@@LB_SIZE@@", str(lb))
pathlib.Path(outp).write_text(text)' \
      "${dn}" "${lb}" "${tpl_disk}" "${out}"
  else
    if [[ ! -f "${tpl_dev}" ]]; then
      log "ERROR: missing bpftrace template ${tpl_dev}"
      return 1
    fi
    local src_mnt
    src_mnt="$(findmnt -nro SOURCE --target "${KURT_NVME_BASE}" 2>/dev/null || true)"
    devp=""
    if [[ -n "${src_mnt}" ]] && [[ -b "${src_mnt}" ]]; then
      devp="${src_mnt}"
    elif [[ -b "/dev/${dn}" ]]; then
      devp="/dev/${dn}"
    else
      log "ERROR: no block device for nvme bpftrace (dev filter); findmnt=${src_mnt:-empty} /dev/${dn}"
      return 1
    fi
    devt="$(python3 -c 'import os, sys; print(os.stat(sys.argv[1]).st_rdev)' "${devp}")"
    python3 -c 'import pathlib, sys
devt, tplp, outp = int(sys.argv[1]), sys.argv[2], sys.argv[3]
text = pathlib.Path(tplp).read_text()
text = text.replace("@@DEV_T@@", str(devt))
pathlib.Path(outp).write_text(text)' \
      "${devt}" "${tpl_dev}" "${out}"
    printf '%s\n' "${devt}" >"${REPORT_DIR}/nvme_blk_io_dev_t.txt"
    printf '%s\n' "${devp}" >"${REPORT_DIR}/nvme_blk_io_trace_dev_path.txt"
  fi
  printf '%s\n' "${dn}" >"${REPORT_DIR}/nvme_blk_io_disk_name.txt"
  printf '%s\n' "${lb}" >"${REPORT_DIR}/nvme_blk_io_lb_size.txt"
  printf '%s\n' "${variant}" >"${REPORT_DIR}/nvme_blk_io_trace_variant.txt"
  if [[ "${variant}" == "dev" ]]; then
    log "nvme block bpftrace variant=${variant} disk_name=${dn} dev_path=${devp} st_rdev=${devt} logical_block_size=${lb}"
  else
    log "nvme block bpftrace variant=${variant} disk_name=${dn} logical_block_size=${lb}"
  fi
}

nvme_blk_io_trace_start() {
  KURT_NVME_BLK_BPFTRACE_PID=""
  if [[ "${KURT_NVME_BLK_BPFTRACE:-0}" != "1" ]]; then
    return 0
  fi
  if ! kurt_nvme_bpftrace_ensure; then
    return 0
  fi
  if ! nvme_blk_io_trace_gen; then
    log "WARN: nvme_blk_io_trace_gen failed; skipping nvme block trace"
    return 0
  fi
  local bt_bin
  bt_bin="$(command -v bpftrace)"
  log "Starting bpftrace NVMe block trace → ${REPORT_DIR}/nvme_blk_io.tsv"
  : >"${REPORT_DIR}/nvme_blk_io.bpftrace.log"
  set +e
  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    "${bt_bin}" -q -o "${REPORT_DIR}/nvme_blk_io.tsv" "${REPORT_DIR}/nvme_block_io_trace.gen.bt" \
      2>>"${REPORT_DIR}/nvme_blk_io.bpftrace.log" &
    KURT_NVME_BLK_BPFTRACE_PID=$!
  elif sudo -n true 2>/dev/null; then
    sudo -n "${bt_bin}" -q -o "${REPORT_DIR}/nvme_blk_io.tsv" "${REPORT_DIR}/nvme_block_io_trace.gen.bt" \
      2>>"${REPORT_DIR}/nvme_blk_io.bpftrace.log" &
    KURT_NVME_BLK_BPFTRACE_PID=$!
  else
    log "WARN: no passwordless sudo; nvme block bpftrace may lack privileges"
    "${bt_bin}" -q -o "${REPORT_DIR}/nvme_blk_io.tsv" "${REPORT_DIR}/nvme_block_io_trace.gen.bt" \
      2>>"${REPORT_DIR}/nvme_blk_io.bpftrace.log" &
    KURT_NVME_BLK_BPFTRACE_PID=$!
  fi
  set -e
  sleep 0.3
  if ! kill -0 "${KURT_NVME_BLK_BPFTRACE_PID}" 2>/dev/null; then
    log "WARN: nvme bpftrace exited immediately; see ${REPORT_DIR}/nvme_blk_io.bpftrace.log"
    KURT_NVME_BLK_BPFTRACE_PID=""
    return 0
  fi
}

nvme_blk_io_trace_stop() {
  if [[ -z "${KURT_NVME_BLK_BPFTRACE_PID}" ]]; then
    return 0
  fi
  log "Stopping bpftrace NVMe block trace (pid=${KURT_NVME_BLK_BPFTRACE_PID})"
  kill -INT "${KURT_NVME_BLK_BPFTRACE_PID}" 2>/dev/null || true
  wait "${KURT_NVME_BLK_BPFTRACE_PID}" 2>/dev/null || true
  KURT_NVME_BLK_BPFTRACE_PID=""
  if [[ -f "${REPORT_DIR}/nvme_blk_io.tsv" ]]; then
    if [[ "$(stat -c '%u' "${REPORT_DIR}/nvme_blk_io.tsv" 2>/dev/null || echo 1)" == "0" ]]; then
      if sudo -n true 2>/dev/null; then
        sudo -n chown "$(id -u):$(id -g)" "${REPORT_DIR}/nvme_blk_io.tsv" 2>/dev/null \
          || log "WARN: sudo chown nvme_blk_io.tsv failed"
      fi
    fi
  fi
}

lmcache_fs_io_trace_gen() {
  local tpl="${RECIPE_DIR}/bpf/lmcache_fs_io_trace.bt.in"
  local out="${REPORT_DIR}/lmcache_fs_io_trace.gen.bt"
  local pref n
  rm -f "${REPORT_DIR}/lmcache_fs_io_trace_prefix.txt"
  mkdir -p "${DATA_HOST}/lmcache_test"
  if [[ -n "${KURT_LMCACHE_FS_BPFTRACE_PREFIX:-}" ]]; then
    pref="${KURT_LMCACHE_FS_BPFTRACE_PREFIX}"
  else
    pref="$(realpath "${DATA_HOST}/lmcache_test" 2>/dev/null || echo "${DATA_HOST}/lmcache_test")"
  fi
  printf '%s\n' "${pref}" >"${REPORT_DIR}/lmcache_fs_io_trace_prefix.txt"
  if [[ ! -f "${tpl}" ]]; then
    log "ERROR: missing bpftrace template ${tpl}"
    return 1
  fi
  n="$(python3 -c 'import pathlib, sys; p = pathlib.Path(sys.argv[1]).read_text().rstrip("\n"); print(min(len(p), 4096))' "${REPORT_DIR}/lmcache_fs_io_trace_prefix.txt")"
  python3 -c 'import json, pathlib, sys
pref, n, tplp, outp = pathlib.Path(sys.argv[1]).read_text().rstrip("\n"), int(sys.argv[2]), sys.argv[3], sys.argv[4]
lit = json.dumps(pref)
text = pathlib.Path(tplp).read_text()
text = text.replace("@@PATH_PREFIX_STR@@", lit)
text = text.replace("@@PATH_PREFIX_LEN@@", str(n))
pathlib.Path(outp).write_text(text)' \
    "${REPORT_DIR}/lmcache_fs_io_trace_prefix.txt" "${n}" "${tpl}" "${out}"
  log "lmcache fs bpftrace path_prefix=${pref} strncmp_len=${n}"
}

lmcache_fs_io_trace_start() {
  KURT_LMCACHE_FS_BPFTRACE_PID=""
  if [[ "${KURT_LMCACHE_FS_BPFTRACE:-0}" != "1" ]]; then
    return 0
  fi
  if ! kurt_nvme_bpftrace_ensure; then
    return 0
  fi
  if ! lmcache_fs_io_trace_gen; then
    log "WARN: lmcache_fs_io_trace_gen failed; skipping LMCache VFS trace"
    return 0
  fi
  local bt_bin
  bt_bin="$(command -v bpftrace)"
  log "Starting bpftrace LMCache VFS trace → ${REPORT_DIR}/lmcache_fs_io.tsv"
  : >"${REPORT_DIR}/lmcache_fs_io.bpftrace.log"
  set +e
  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    "${bt_bin}" -q -o "${REPORT_DIR}/lmcache_fs_io.tsv" "${REPORT_DIR}/lmcache_fs_io_trace.gen.bt" \
      2>>"${REPORT_DIR}/lmcache_fs_io.bpftrace.log" &
    KURT_LMCACHE_FS_BPFTRACE_PID=$!
  elif sudo -n true 2>/dev/null; then
    sudo -n "${bt_bin}" -q -o "${REPORT_DIR}/lmcache_fs_io.tsv" "${REPORT_DIR}/lmcache_fs_io_trace.gen.bt" \
      2>>"${REPORT_DIR}/lmcache_fs_io.bpftrace.log" &
    KURT_LMCACHE_FS_BPFTRACE_PID=$!
  else
    log "WARN: no passwordless sudo; LMCache VFS bpftrace may lack privileges"
    "${bt_bin}" -q -o "${REPORT_DIR}/lmcache_fs_io.tsv" "${REPORT_DIR}/lmcache_fs_io_trace.gen.bt" \
      2>>"${REPORT_DIR}/lmcache_fs_io.bpftrace.log" &
    KURT_LMCACHE_FS_BPFTRACE_PID=$!
  fi
  set -e
  sleep 0.3
  if ! kill -0 "${KURT_LMCACHE_FS_BPFTRACE_PID}" 2>/dev/null; then
    log "WARN: LMCache vfs bpftrace exited immediately; see ${REPORT_DIR}/lmcache_fs_io.bpftrace.log"
    KURT_LMCACHE_FS_BPFTRACE_PID=""
    return 0
  fi
}

lmcache_fs_io_trace_stop() {
  if [[ -z "${KURT_LMCACHE_FS_BPFTRACE_PID}" ]]; then
    return 0
  fi
  log "Stopping bpftrace LMCache VFS trace (pid=${KURT_LMCACHE_FS_BPFTRACE_PID})"
  kill -INT "${KURT_LMCACHE_FS_BPFTRACE_PID}" 2>/dev/null || true
  wait "${KURT_LMCACHE_FS_BPFTRACE_PID}" 2>/dev/null || true
  KURT_LMCACHE_FS_BPFTRACE_PID=""
  if [[ -f "${REPORT_DIR}/lmcache_fs_io.tsv" ]]; then
    if [[ "$(stat -c '%u' "${REPORT_DIR}/lmcache_fs_io.tsv" 2>/dev/null || echo 1)" == "0" ]]; then
      if sudo -n true 2>/dev/null; then
        sudo -n chown "$(id -u):$(id -g)" "${REPORT_DIR}/lmcache_fs_io.tsv" 2>/dev/null \
          || log "WARN: sudo chown lmcache_fs_io.tsv failed"
      fi
    fi
  fi
}

nvme_smart_log_snapshot() {
  local tag="$1"
  if [[ "${KURT_NVME_SMART_LOG:-0}" != "1" ]]; then
    return 0
  fi
  local dev="${KURT_NVME_DEVICE:-}"
  if [[ -z "${dev}" ]]; then
    dev="$(findmnt -nro SOURCE --target "${KURT_NVME_BASE}" 2>/dev/null || true)"
  fi
  if [[ -z "${dev}" ]] || [[ ! -b "${dev}" ]]; then
    log "WARN: nvme smart-log ${tag}: no block device (set KURT_NVME_DEVICE?)"
    return 0
  fi
  if ! command -v nvme &>/dev/null; then
    log "WARN: nvme-cli not in PATH; skipping smart-log ${tag}"
    return 0
  fi
  set +e
  nvme smart-log "${dev}" -o json >"${REPORT_DIR}/nvme_smart_log_${tag}.json" 2>>"${REPORT_DIR}/nvme_smart_log.log"
  local rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    log "WARN: nvme smart-log ${tag} failed rc=${rc} (root may be required)"
  else
    log "nvme smart-log ${tag} → ${REPORT_DIR}/nvme_smart_log_${tag}.json"
  fi
}

run_docker() {
  if ! command -v "${DOCKER}" &>/dev/null; then
    log "ERROR: ${DOCKER} not found (set DOCKER=podman or install Docker)"
    exit 1
  fi
  T_BUILD_START="$(date -Iseconds)"
  (
    cd "${RECIPE_DIR}"
    "${DOCKER}" build -t "${IMAGE_NAME}" --build-arg "ROCM_ARCH=${ROCM_ARCH}" .
  )
  BUILD_RC=$?
  T_BUILD_END="$(date -Iseconds)"
  if [[ "${BUILD_RC}" -ne 0 ]]; then
    log "ERROR: docker build failed rc=${BUILD_RC}"
    return "${BUILD_RC}"
  fi
  if IMAGE_ID_LINE="$("${DOCKER}" inspect --format='{{.Id}}' "${IMAGE_NAME}" 2>/dev/null)"; then
    true
  else
    IMAGE_ID_LINE="(inspect failed)"
  fi

  T_RUN_START="$(date -Iseconds)"
  nvme_blk_io_trace_start
  lmcache_fs_io_trace_start
  set +e
  "${DOCKER}" run --rm \
    --name="gpu${GPU_IDX}" \
    --hostname="gpu${GPU_IDX}" \
    --ipc=host \
    --network=host \
    --cap-add=CAP_SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --env HF_HOME="${CONTAINER_HF_HOME}" \
    --env HUGGINGFACE_HUB_CACHE="${CONTAINER_HF_HOME}/hub" \
    --env HF_HUB_CACHE="${CONTAINER_HF_HOME}/hub" \
    --env HF_DATASETS_CACHE="${CONTAINER_HF_HOME}/datasets" \
    --env VLLM_CACHE_ROOT="${CONTAINER_HF_HOME}/vllm" \
    --env VLLM_CONFIG_ROOT="${CONTAINER_HF_HOME}/vllm_config" \
    --env TORCH_HOME="${CONTAINER_HF_HOME}/torch" \
    --env TORCHINDUCTOR_CACHE_DIR="${CONTAINER_HF_HOME}/torch_inductor" \
    --env KURT_CONTAINER_DATA_DIR="${CONTAINER_DATA_DIR}" \
    --env HF_TOKEN="${HF_TOKEN}" \
    --env ROCR_VISIBLE_DEVICES="${GPU_IDX}" \
    --env PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
    --env VLLM_MODEL="${VLLM_MODEL:-}" \
    --env KURT_MAX_MODEL_LEN="${KURT_MAX_MODEL_LEN:-}" \
    --env KURT_MAX_NUM_BATCHED_TOKENS="${KURT_MAX_NUM_BATCHED_TOKENS:-}" \
    --env KURT_ENFORCE_EAGER="${KURT_ENFORCE_EAGER:-}" \
    --env KURT_GPU_MEMORY_UTILIZATION="${KURT_GPU_MEMORY_UTILIZATION:-}" \
    --env PYTHONHASHSEED=0 \
    --env KURT_LMCACHE_ENABLE_CHUNK_STATISTICS="${KURT_LMCACHE_ENABLE_CHUNK_STATISTICS:-1}" \
    --env KURT_LMCACHE_CHUNK_STATISTICS_STRATEGY="${KURT_LMCACHE_CHUNK_STATISTICS_STRATEGY:-file_hash}" \
    --env KURT_LMCACHE_COLLECT_INTERNAL_API="${KURT_LMCACHE_COLLECT_INTERNAL_API:-1}" \
    --env KURT_LMCACHE_INTERNAL_API_MAX_OFFSET="${KURT_LMCACHE_INTERNAL_API_MAX_OFFSET:-3}" \
    --env KURT_LMCACHE_ENABLE_KV_EVENTS="${KURT_LMCACHE_ENABLE_KV_EVENTS:-1}" \
    --env KURT_LMCACHE_LOG_LEVEL="${KURT_LMCACHE_LOG_LEVEL:-INFO}" \
    --env KURT_LMCACHE_INTERNAL_API_VERBOSE="${KURT_LMCACHE_INTERNAL_API_VERBOSE:-0}" \
    -v "${HF_HOST}:${CONTAINER_HF_HOME}" \
    -v "${DATA_HOST}:${CONTAINER_DATA_DIR}" \
    "${BOOT_MOUNT[@]}" \
    --entrypoint /bin/bash \
    "${IMAGE_NAME}" \
    -lc "${BENCH_CMD_QUOTED}" \
    | tee "${REPORT_DIR}/long_doc_qa.json"
  RUN_RC=${PIPESTATUS[0]}
  set -e
  lmcache_fs_io_trace_stop
  nvme_blk_io_trace_stop
  T_RUN_END="$(date -Iseconds)"
  return "${RUN_RC}"
}

run_pyxis() {
  local img="${CONTAINER_IMAGE:-}"
  if [[ -z "${img}" ]]; then
    log "ERROR: RUNTIME=pyxis requires CONTAINER_IMAGE (docker://registry/.../vllm-kurt:tag or /path/image.sqsh)"
    exit 1
  fi
  BUILD_RC="skipped"
  T_BUILD_START="$(date -Iseconds)"
  T_BUILD_END="${T_BUILD_START}"
  IMAGE_ID_LINE="${img}"

  local mounts="${HF_HOST}:${CONTAINER_HF_HOME}:rw,${DATA_HOST}:${CONTAINER_DATA_DIR}:rw"
  if [[ -r "${BOOTCFG}" ]]; then
    mounts="${mounts},${BOOTCFG}:${BOOTCFG}:ro"
  fi

  export_for_container
  T_RUN_START="$(date -Iseconds)"
  nvme_blk_io_trace_start
  lmcache_fs_io_trace_start
  set +e
  srun --ntasks=1 --export=ALL --container-image="${img}" \
    --container-mounts="${mounts}" \
    --container-writable \
    --no-container-entrypoint \
    /bin/bash -lc "${BENCH_CMD_QUOTED}" \
    | tee "${REPORT_DIR}/long_doc_qa.json"
  RUN_RC=${PIPESTATUS[0]}
  set -e
  lmcache_fs_io_trace_stop
  nvme_blk_io_trace_stop
  T_RUN_END="$(date -Iseconds)"
  return "${RUN_RC}"
}

# Pyxis: if CONTAINER_IMAGE is unset, use shared .sqsh when present (cluster-wide cache).
if [[ "${RUNTIME}" == "pyxis" && -z "${CONTAINER_IMAGE:-}" ]]; then
  if [[ -r "${KURT_SHARED_IMAGE_SQSH}" ]]; then
    CONTAINER_IMAGE="${KURT_SHARED_IMAGE_SQSH}"
    log "RUNTIME=pyxis: CONTAINER_IMAGE unset; using shared ${CONTAINER_IMAGE}"
  fi
fi

nvme_smart_log_snapshot job_start

case "${RUNTIME}" in
  docker) run_docker ;;
  pyxis) run_pyxis ;;
  *)
    log "ERROR: RUNTIME must be docker or pyxis (got ${RUNTIME})"
    exit 1
    ;;
esac
PHASE_RC=$?
RUN_RC="${RUN_RC:-${PHASE_RC}}"

nvme_smart_log_snapshot job_end

SUMMARY="${REPORT_DIR}/summary.txt"
{
  echo "RUNTIME=${RUNTIME} PHASE_RC=${PHASE_RC} RUN_RC=${RUN_RC} BUILD_RC=${BUILD_RC:-}"
  echo "JOB_ROOT=${JOB_ROOT}"
} >"${SUMMARY}"

META="${REPORT_DIR}/metadata.txt"
sync_lmcache_report_files
write_job_metadata "${META}"
append_server_log "${META}"
write_report_md "${REPORT_DIR}/report.md"
copy_to_submit_logs

if [[ "${PHASE_RC}" -ne 0 ]]; then
  log "ERROR: workload failed rc=${PHASE_RC}"
  exit "${PHASE_RC}"
fi
log "Done. Reports under ${REPORT_DIR} and ${LOG_DIR}/vllm-from-kurt-mi300-${SLURM_JOB_ID}/"
exit 0
