#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Shared helpers for vLLM + LMCache recipe Slurm jobs (sourced by recipe sbatch files).

if [[ -z "${REPO_DIR:-}" ]]; then
	REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
fi
# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/recipe-bench.sh"
_recipe_export_bench_root

_vlh_env_aliases() {
    local _k _vlh _kurt
    for _k in NVME_BASE NVME_BLK_BPFTRACE NVME_SMART_LOG NVME_MKFS \
        NVME_AUTO_DEVICE NVME_AUTO_USE NVME_MOUNT NVME_DEVICE VFS_BPFTRACE \
        VFS_BPFTRACE_APT_INSTALL LMCACHE_ENABLE_KV_EVENTS SKIP_BUILD BENCHMARK \
        LMCACHE_IO LMCACHE_LOG_LEVEL RUN_LONG_PARALLEL RUN_LONG_WORKERS \
        RUN_LONG_ITERATIONS RUN_LONG_BASE_SEED RUN_LONG_MAX_TOKENS \
        GUTENBERG_DATA_ROOT SHARED_ROOT \
        NIXL_BUFFER_SIZE; do
        _vlh="VLH_${_k}"
        _vln="VLN_${_k}"
        _kurt="KURT_${_k}"
        if [[ -z "${!_vlh:-}" && -n "${!_vln:-}" ]]; then
            export "${_vlh}=${!_vln}"
        fi
        if [[ -z "${!_vlh:-}" && -n "${!_kurt:-}" ]]; then
            export "${_vlh}=${!_kurt}"
        fi
    done
}

_vlh_log() {
    printf '[%s] %s\n' "$(date -Iseconds)" "$*"
}

_vlh_truthy() {
    local v="${1:-0}"
    case "${v,,}" in
        1 | true | yes | on) return 0 ;;
        *) return 1 ;;
    esac
}

# Single model knob: VLLM_MODEL (server + benchmarks). Unset → vllm-lmcache-hipfile.yaml default.
_vlh_model_default() {
    printf 'openai/gpt-oss-120b'
}

_vlh_resolve_model_env() {
    if [[ -z "${VLLM_MODEL:-}" && -n "${VLH_BENCH_MODEL:-}" ]]; then
        export VLLM_MODEL="${VLH_BENCH_MODEL}"
        _vlh_log "WARN: VLH_BENCH_MODEL is deprecated; use VLLM_MODEL instead"
    fi
}

_vlh_served_model() {
    if [[ -n "${VLLM_MODEL:-}" ]]; then
        printf '%s' "${VLLM_MODEL}"
        return 0
    fi
    _vlh_model_default
}

_vlh_resolve_hf_token() {
    if [[ -n "${HF_TOKEN:-}" ]]; then
        return 0
    fi
    if [[ -n "${HF_TOKEN_FILE:-}" && -r "${HF_TOKEN_FILE}" ]]; then
        HF_TOKEN="$(tr -d '\r\n' < "${HF_TOKEN_FILE}")"
        export HF_TOKEN
        return 0
    fi
    _vlh_log "ERROR: set HF_TOKEN or a readable HF_TOKEN_FILE"
    return 1
}

_vlh_detect_rocm_arch() {
    if [[ -n "${ROCM_ARCH:-}" ]]; then
        _vlh_log "Using ROCM_ARCH=${ROCM_ARCH}"
        return 0
    fi
    ROCM_ARCH="$(rocm_agent_enumerator 2>/dev/null | grep -E '^gfx' | head -1 || true)"
    if [[ -z "${ROCM_ARCH}" ]]; then
        ROCM_ARCH="$(rocm_agent_enumerator 2>/dev/null | grep -v '^gfx0' | sort -u | tail -1 || true)"
    fi
    if [[ -z "${ROCM_ARCH}" ]]; then
        _vlh_log "ERROR: ROCM_ARCH empty (set ROCM_ARCH or install rocm_agent_enumerator)"
        return 1
    fi
    export ROCM_ARCH
    _vlh_log "Using ROCM_ARCH=${ROCM_ARCH}"
}

_vlh_metadata_append() {
    local key="$1"
    local val="$2"
    printf '%s: %s\n' "${key}" "${val}" >> "${METADATA_FILE}"
}

_vlh_metadata_init() {
    : "${REPORT_DIR:?REPORT_DIR must be set before _vlh_metadata_init}"
    METADATA_FILE="${REPORT_DIR}/metadata.txt"
    : > "${METADATA_FILE}"
    _vlh_metadata_append "hostname" "$(hostname -f 2>/dev/null || hostname)"
    _vlh_metadata_append "date" "$(date -Iseconds)"
    _vlh_metadata_append "RUNTIME" "docker"
    _vlh_metadata_append "RECIPE_DIR" "${RECIPE_DIR}"
    _vlh_metadata_append "ROCM_ARCH" "${ROCM_ARCH}"
    _vlh_metadata_append "IMAGE_NAME" "${IMAGE_NAME:-vllm-lmcache-hipfile}"
    _vlh_metadata_append "SLURM_JOB_ID" "${SLURM_JOB_ID:-}"
    _vlh_metadata_append "SLURM_JOB_NODELIST" "${SLURM_JOB_NODELIST:-}"
    _vlh_metadata_append "SLURM_SUBMIT_DIR" "${SLURM_SUBMIT_DIR:-}"
    _vlh_metadata_append "JOB_ROOT" "${JOB_ROOT}"
    _vlh_metadata_append "VLH_NVME_BASE" "${VLH_NVME_BASE:-}"
    _vlh_metadata_append "VLH_LMCACHE_IO" "${VLH_LMCACHE_IO:-hipfile}"
    _vlh_metadata_append "VLH_BENCHMARK" "${VLH_BENCHMARK:-gutenberg}"
    _vlh_metadata_append "VLH_GUTENBERG_DATA_ROOT" "${VLH_GUTENBERG_DATA_ROOT:-}"
    _vlh_metadata_append "CONTAINER_NAME" "${CONTAINER_NAME}"
}

_vlh_nvme_disk_has_mounted_descendant() {
    # True when the namespace or any child partition/filesystem is mounted.
    local dev="$1"
    lsblk -rn -o MOUNTPOINT "${dev}" 2>/dev/null \
        | awk 'NF && $1 != "" { exit 0 } END { exit 1 }'
}

_vlh_find_unmounted_nvme_device() {
    # Spare whole-disk nvme namespace: no mounted descendants, not RAID/LVM.
    # Skips OS drives such as nvme0n1 when nvme0n1p* holds / or /boot.
    local name dev fstype
    while IFS= read -r name; do
        [[ -n "${name}" ]] || continue
        [[ "${name}" =~ ^nvme[0-9]+n[0-9]+$ ]] || continue
        dev="/dev/${name}"
        fstype="$(lsblk -dn -o FSTYPE "${dev}" 2>/dev/null | head -1 || true)"
        if [[ "${fstype}" =~ raid|LVM|linux_raid ]]; then
            _vlh_log "VLH_NVME_AUTO_DEVICE: skip ${dev} (fstype=${fstype:-unknown})"
            continue
        fi
        if _vlh_nvme_disk_has_mounted_descendant "${dev}"; then
            _vlh_log "VLH_NVME_AUTO_DEVICE: skip ${dev} (mounted partition or filesystem)"
            continue
        fi
        printf '%s' "${dev}"
        return 0
    done < <(lsblk -dn -o NAME,TYPE 2>/dev/null | awk '$2 == "disk" { print $1 }')
    return 1
}

_vlh_nvme_log_inventory() {
    _vlh_log "NVMe block devices on $(hostname -s 2>/dev/null || hostname):"
    if command -v lsblk >/dev/null 2>&1; then
        lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE,MODEL 2>/dev/null \
            | grep -E '^NAME|nvme' \
            | while IFS= read -r _line; do
                _vlh_log "  ${_line}"
            done || _vlh_log "  (none listed)"
    else
        _vlh_log "  lsblk not available"
    fi
}

_vlh_mount_is_excluded() {
    case "$1" in
        / | /boot | /boot/* | /proc | /proc/* | /sys | /sys/* | /dev | /dev/* \
            | /run | /run/*)
            return 0
            ;;
    esac
    return 1
}

_vlh_mount_is_nvme_backed() {
    local mnt="$1"
    local src
    src="$(findmnt -n -o SOURCE --target "${mnt}" 2>/dev/null || true)"
    [[ -z "${src}" ]] && return 1
    if [[ "${src}" == *nvme* ]]; then
        return 0
    fi
    # LVM/device-mapper on top of NVMe (e.g. /dev/mapper/data-data → nvme*n*).
    if command -v lsblk >/dev/null 2>&1; then
        lsblk -s -n -o NAME "${src}" 2>/dev/null | grep -qE '^nvme'
        return $?
    fi
    return 1
}

_vlh_mount_avail_bytes() {
    df -B1 --output=avail "${1}" 2>/dev/null | awk 'NR==2 { print $1 }'
}

_vlh_nvme_mount_candidates() {
    # All mount points; NVMe backing is checked via findmnt + lsblk (includes LVM on nvme).
    if command -v findmnt >/dev/null 2>&1; then
        findmnt -rn -o TARGET 2>/dev/null | sort -u
        return
    fi
    lsblk -rn -o MOUNTPOINT 2>/dev/null | awk '$1 != ""' | sort -u
}

_vlh_mount_is_shared_cluster_path() {
    # Site-wide pools (often root-owned LVM on NVMe). Do not use for LMCache unless opted in.
    case "$1" in
        /data | /data/* | /docker | /docker/*) return 0 ;;
    esac
    return 1
}

_vlh_find_mounted_nvme_mount() {
    # Writable NVMe-backed mount with enough free space (not shared /data or /docker by default).
    local min_avail_gb="${VLH_NVME_MIN_AVAIL_GB:-10}"
    local min_bytes=$((min_avail_gb * 1024 * 1024 * 1024))
    local pass mnt avail best_avail=-1 best_mnt=""

    for pass in preferred any; do
        best_avail=-1
        best_mnt=""
        while IFS= read -r mnt; do
            [[ -n "${mnt}" ]] || continue
            _vlh_mount_is_excluded "${mnt}" && continue
            if [[ "${pass}" == preferred ]]; then
                case "${mnt}" in
                    /mnt/* | /local/* | /localssd/* | /nvme/* | /ssd/* | /cache/*) ;;
                    *) continue ;;
                esac
            fi
            if _vlh_mount_is_shared_cluster_path "${mnt}" \
                && ! _vlh_truthy "${VLH_NVME_USE_SHARED_DATA_DOCKER:-0}"; then
                continue
            fi
            _vlh_mount_is_nvme_backed "${mnt}" || continue
            [[ -d "${mnt}" && -w "${mnt}" ]] || continue
            avail="$(_vlh_mount_avail_bytes "${mnt}")"
            [[ -n "${avail}" && "${avail}" -ge "${min_bytes}" ]] || continue
            if [[ "${avail}" -gt "${best_avail}" ]]; then
                best_avail="${avail}"
                best_mnt="${mnt}"
            fi
        done < <(_vlh_nvme_mount_candidates)
        if [[ -n "${best_mnt}" ]]; then
            printf '%s' "${best_mnt}"
            return 0
        fi
    done
    return 1
}

_vlh_use_mounted_nvme_path() {
    local mnt="$1"
    local base="${mnt}/vllm-lmcache-hipfile-${SLURM_JOB_ID:-local$$}"
    local avail_human src

    mkdir -p "${base}" || return 1
    [[ -w "${base}" ]] || return 1

    src="$(findmnt -n -o SOURCE --target "${mnt}" 2>/dev/null || true)"
    avail_human="$(df -h --output=avail "${mnt}" 2>/dev/null | awk 'NR==2 { print $1 }')"

    VLH_NVME_MOUNT="${mnt}"
    VLH_NVME_BASE="${base}"
    VLH_NVME_DEVICE="$(_vlh_block_device_for_path "${mnt}" 2>/dev/null || true)"
    if [[ -z "${VLH_NVME_DEVICE}" ]]; then
        VLH_NVME_DEVICE="${src}"
    fi
    export VLH_NVME_MOUNT VLH_NVME_BASE VLH_NVME_DEVICE

    _vlh_metadata_append "VLH_NVME_AUTO_MODE" "mounted"
    _vlh_metadata_append "VLH_NVME_MOUNT" "${mnt}"
    _vlh_metadata_append "VLH_NVME_SOURCE" "${src}"
    _vlh_log "VLH_NVME_AUTO_USE: NVMe mount ${mnt} (${src}, ${avail_human} avail) -> ${base}"
    return 0
}

_vlh_nvme_scratch_fallback() {
    if ! _vlh_truthy "${VLH_NVME_SCRATCH_FALLBACK:-1}"; then
        return 1
    fi
    local root="${VLH_NVME_SCRATCH_ROOT:-/scratch/${USER:-}}"
    local base
    [[ -n "${root}" && -d "${root}" && -w "${root}" ]] || return 1
    base="${root}/vllm-lmcache-hipfile/lmcache-${SLURM_JOB_ID:-local$$}"
    mkdir -p "${base}" || return 1
    [[ -w "${base}" ]] || return 1
    VLH_NVME_BASE="${base}"
    export VLH_NVME_BASE
    _vlh_metadata_append "VLH_NVME_AUTO_MODE" "scratch"
    _vlh_log "VLH_NVME_AUTO_USE: scratch fallback ${base} (no local NVMe path found)"
    return 0
}

_vlh_can_mkfs_mount_block_device() {
    # Format/mount of a whole-disk block device requires root on typical Slurm nodes.
    [[ "$(id -u)" -eq 0 ]]
}

_vlh_nvme_mount_device() {
    local device="$1"
    local mount="$2"
    local fstype

    if ! mkdir -p "${mount}"; then
        _vlh_log "WARN: cannot create mount point ${mount}"
        return 1
    fi
    fstype="$(blkid -o value -s TYPE "${device}" 2>/dev/null || true)"
    if [[ -n "${fstype}" ]]; then
        _vlh_log "mount ${device} (${fstype}) -> ${mount}"
        if ! mount "${device}" "${mount}"; then
            _vlh_log "WARN: mount ${device} -> ${mount} failed (root required?)"
            return 1
        fi
    elif _vlh_truthy "${VLH_NVME_MKFS:-0}"; then
        _vlh_log "mkfs.ext4 ${device} -> ${mount}"
        command -v mkfs.ext4 >/dev/null 2>&1 || {
            _vlh_log "ERROR: mkfs.ext4 not found"
            return 1
        }
        if ! mkfs.ext4 -F "${device}"; then
            _vlh_log "WARN: mkfs.ext4 ${device} failed (root required?)"
            return 1
        fi
        if ! mount "${device}" "${mount}"; then
            _vlh_log "WARN: mount ${device} -> ${mount} failed after mkfs"
            return 1
        fi
    else
        _vlh_log "WARN: ${device} has no filesystem; set VLH_NVME_MKFS=1 to format (default when auto-discovering)"
        return 1
    fi
    VLH_NVME_DEVICE="${device}"
    VLH_NVME_MOUNT="${mount}"
    VLH_NVME_BASE="${mount}"
    export VLH_NVME_DEVICE VLH_NVME_MOUNT VLH_NVME_BASE
    _vlh_metadata_append "VLH_NVME_DEVICE" "${device}"
    _vlh_metadata_append "VLH_NVME_MOUNT" "${mount}"
    if _vlh_chown_to_job_user "${mount}"; then
        _vlh_log "chown ${mount} -> ${SLURM_JOB_USER:-${USER}} (LMCache mount)"
    else
        _vlh_log "WARN: could not chown ${mount} to ${SLURM_JOB_USER:-${USER}}; LMCache writes may fail"
    fi
    return 0
}

# When VLH_NVME_BASE is unset: (1) mount blank nvme*n*, (2) use an existing
# NVMe-backed mount, (3) scratch fallback — see vllm-lmcache-hipfile.sbatch.
_vlh_nvme_auto_use() {
    local device mount mnt

    if ! _vlh_truthy "${VLH_NVME_AUTO_USE:-1}"; then
        return 1
    fi

    device="${VLH_NVME_DEVICE:-}"
    if [[ -z "${device}" ]] && _vlh_truthy "${VLH_NVME_AUTO_DEVICE:-1}"; then
        device="$(_vlh_find_unmounted_nvme_device || true)"
        if [[ -n "${device}" ]]; then
            _vlh_log "VLH_NVME_AUTO_DEVICE selected unmounted ${device}"
        fi
    fi
    if [[ -n "${device}" ]]; then
        if ! _vlh_can_mkfs_mount_block_device; then
            _vlh_log "VLH_NVME_AUTO_DEVICE: skip ${device} (mkfs/mount needs root; try mounted/scratch paths)"
        else
            mount="${VLH_NVME_MOUNT:-/mnt/vllm-lmcache-hipfile-${SLURM_JOB_ID:-local}}"
            if _vlh_nvme_mount_device "${device}" "${mount}"; then
                _vlh_metadata_append "VLH_NVME_AUTO_MODE" "mount-unmounted"
                return 0
            fi
        fi
    fi

    mnt="$(_vlh_find_mounted_nvme_mount || true)"
    if [[ -n "${mnt}" ]] && _vlh_use_mounted_nvme_path "${mnt}"; then
        return 0
    fi

    _vlh_nvme_log_inventory
    _vlh_log "WARN: no dedicated writable NVMe path (unmounted nvme*n*, or a job-local" \
        "mount under /mnt|/local|/nvme with >= ${VLH_NVME_MIN_AVAIL_GB:-10}G free)."
    _vlh_log "WARN: /data and /docker are shared site LVM pools — skipped unless" \
        "VLH_NVME_USE_SHARED_DATA_DOCKER=1. Using scratch fallback if enabled."
    return 1
}

# Explicit mkfs+mount (VLH_NVME_MKFS=1); used when VLH_NVME_BASE is preset.
_vlh_nvme_setup() {
    local device="${VLH_NVME_DEVICE:-}"
    local mount="${VLH_NVME_MOUNT:-}"

    if ! _vlh_truthy "${VLH_NVME_MKFS:-0}"; then
        return 0
    fi

    if [[ -z "${device}" ]] && _vlh_truthy "${VLH_NVME_AUTO_DEVICE:-1}"; then
        device="$(_vlh_find_unmounted_nvme_device || true)"
        if [[ -z "${device}" ]]; then
            _vlh_log "WARN: VLH_NVME_AUTO_DEVICE=1 but no unmounted whole-disk nvme* found"
        else
            VLH_NVME_DEVICE="${device}"
            export VLH_NVME_DEVICE
            _vlh_log "VLH_NVME_AUTO_DEVICE selected ${device}"
        fi
    fi

    if [[ -z "${device}" || -z "${mount}" ]]; then
        _vlh_log "ERROR: VLH_NVME_MKFS=1 requires VLH_NVME_DEVICE and VLH_NVME_MOUNT"
        return 1
    fi

    _vlh_nvme_mount_device "${device}" "${mount}"
}

_vlh_st_rdev_decimal() {
    local dev_path="$1"
    local maj min
    maj="$(stat -c '%t' "${dev_path}" 2>/dev/null || echo 0)"
    min="$(stat -c '%T' "${dev_path}" 2>/dev/null || echo 0)"
    printf '%d' $((16#${maj} * 256 + 16#${min}))
}

_vlh_block_device_for_path() {
    local path="$1"
    local src dev_path
    src="$(findmnt -n -o SOURCE --target "${path}" 2>/dev/null || true)"
    if [[ -z "${src}" ]]; then
        src="$(df -P "${path}" 2>/dev/null | awk 'NR==2 {print $1}')"
    fi
    if [[ -z "${src}" ]]; then
        return 1
    fi
    dev_path="${src}"
    if [[ ! -b "${dev_path}" ]]; then
        dev_path="/dev/$(lsblk -no PKNAME "${src}" 2>/dev/null | head -1)"
    fi
    if [[ ! -b "${dev_path}" ]]; then
        return 1
    fi
    printf '%s' "${dev_path}"
}

_vlh_nvme_smart_log() {
    local phase="$1"
    local dev="${VLH_NVME_DEVICE:-}"
    if ! _vlh_truthy "${VLH_NVME_SMART_LOG:-0}"; then
        return 0
    fi
    if ! command -v nvme >/dev/null 2>&1; then
        _vlh_log "WARN: nvme-cli not installed; skip smart-log ${phase}"
        return 0
    fi
    if [[ -z "${dev}" ]]; then
        dev="$(_vlh_block_device_for_path "${DATA_HOST}" 2>/dev/null || true)"
    fi
    if [[ -z "${dev}" ]]; then
        _vlh_log "WARN: nvme smart-log ${phase}: no block device for ${DATA_HOST}"
        return 0
    fi
    local out="${REPORT_DIR}/nvme_smart_log_job_${phase}.json"
    if nvme smart-log "${dev}" -o json > "${out}" 2>>"${REPORT_DIR}/nvme_smart_log.log"; then
        _vlh_metadata_append "nvme_smart_log_${phase}" "${out}"
    else
        _vlh_log "WARN: nvme smart-log ${phase} failed (root may be required)"
    fi
}

_vlh_bpftrace_ok() {
    local probe_log="${REPORT_DIR}/bpftrace-probe.log"
    if bpftrace -e 'BEGIN { exit() }' >"${probe_log}" 2>&1; then
        return 0
    fi
    _vlh_log "WARN: bpftrace probe failed (root/CAP_BPF or tracefs); skip trace"
    if [[ -s "${probe_log}" ]]; then
        sed -n '1,3p' "${probe_log}" | while IFS= read -r _line; do
            _vlh_log "  bpftrace: ${_line}"
        done
    fi
    return 1
}

_vlh_bpftrace_log_tail() {
    local log_file="$1"
    local label="${2:-bpftrace}"
    if [[ -s "${log_file}" ]]; then
        sed -n '1,5p' "${log_file}" | while IFS= read -r _line; do
            _vlh_log "  ${label}: ${_line}"
        done
    fi
}

_vlh_bpftrace_nvme_start() {
    if ! _vlh_truthy "${VLH_NVME_BLK_BPFTRACE:-0}"; then
        return 0
    fi
    if ! command -v bpftrace >/dev/null 2>&1; then
        _vlh_log "WARN: bpftrace not installed; skip NVMe block trace"
        return 0
    fi
    if ! _vlh_bpftrace_ok; then
        return 0
    fi

    local dev_path disk_name st_rdev bt_out tsv_out
    dev_path="$(_vlh_block_device_for_path "${DATA_HOST}" 2>/dev/null || true)"
    if [[ -z "${dev_path}" ]]; then
        disk_name="$(lsblk -dn -o NAME,MOUNTPOINT 2>/dev/null | awk '$2=="" && $1 ~ /^nvme/ {print $1; exit}')"
        if [[ -n "${disk_name}" ]]; then
            dev_path="/dev/${disk_name}"
        fi
    fi
    if [[ -z "${dev_path}" || ! -b "${dev_path}" ]]; then
        _vlh_log "WARN: NVMe bpftrace: no block device for ${DATA_HOST}"
        return 0
    fi

    disk_name="$(basename "${dev_path}")"
    st_rdev="$(_vlh_st_rdev_decimal "${dev_path}")"
    bt_out="${REPORT_DIR}/nvme_block_io_trace.gen.bt"
    tsv_out="${REPORT_DIR}/nvme_blk_io.tsv"
    NVME_BLK_TSV="${tsv_out}"

    sed "s/__ST_RDEV__/${st_rdev}/g" "${SLURM_LIB}/nvme_block_io_trace.bt.in" > "${bt_out}"
    _vlh_log "nvme block bpftrace disk_name=${disk_name} dev_path=${dev_path} st_rdev=${st_rdev}"
    _vlh_log "Starting bpftrace NVMe block trace -> ${tsv_out}"

    bpftrace -q "${bt_out}" > "${tsv_out}" 2>"${REPORT_DIR}/nvme_blk_io.bpftrace.log" &
    NVME_BLK_BPFTRACE_PID=$!
    sleep 1
    if ! kill -0 "${NVME_BLK_BPFTRACE_PID}" 2>/dev/null; then
        _vlh_log "WARN: nvme bpftrace exited immediately"
        _vlh_bpftrace_log_tail "${REPORT_DIR}/nvme_blk_io.bpftrace.log" "nvme-bpftrace"
        NVME_BLK_BPFTRACE_PID=
    fi
    _vlh_metadata_append "nvme_blk disk_name" "${disk_name}"
    _vlh_metadata_append "nvme_blk dev_path" "${dev_path}"
    _vlh_metadata_append "nvme_blk st_rdev" "${st_rdev}"
}

_vlh_bpftrace_nvme_stop() {
    if [[ -n "${NVME_BLK_BPFTRACE_PID:-}" ]] && kill -0 "${NVME_BLK_BPFTRACE_PID}" 2>/dev/null; then
        kill "${NVME_BLK_BPFTRACE_PID}" 2>/dev/null || true
        wait "${NVME_BLK_BPFTRACE_PID}" 2>/dev/null || true
    fi
    if [[ -f "${NVME_BLK_TSV:-}" ]]; then
        _vlh_metadata_append "nvme_blk_io.tsv" "${NVME_BLK_TSV} lines=$(wc -l < "${NVME_BLK_TSV}")"
    fi
}

_vlh_cgroup_path_for_pid() {
    local pid="$1"
    local line path
    line="$(grep -E '^0::' "/proc/${pid}/cgroup" 2>/dev/null | head -1 || true)"
    path="${line#0::}"
    if [[ -z "${path}" ]]; then
        return 1
    fi
    printf '/sys/fs/cgroup%s' "${path}"
}

_vlh_bpftrace_vfs_start() {
    if ! _vlh_truthy "${VLH_VFS_BPFTRACE:-0}"; then
        return 0
    fi
    if ! command -v bpftrace >/dev/null 2>&1; then
        if _vlh_truthy "${VLH_VFS_BPFTRACE_APT_INSTALL:-0}"; then
            _vlh_log "Installing bpftrace (apt) ..."
            if apt-get update -qq; then
                DEBIAN_FRONTEND=noninteractive apt-get install -y -qq bpftrace \
                    || true
            fi
        fi
    fi
    if ! command -v bpftrace >/dev/null 2>&1; then
        _vlh_log "WARN: bpftrace not installed; skip VFS trace"
        return 0
    fi
    if ! _vlh_bpftrace_ok; then
        return 0
    fi

    local cid cgroup_path bt_out tsv_out escaped
    cid="$(docker inspect -f '{{.State.Pid}}' "${CONTAINER_NAME}" 2>/dev/null || true)"
    if [[ -z "${cid}" || "${cid}" == "0" ]]; then
        _vlh_log "WARN: VFS bpftrace: container pid not available yet"
        return 0
    fi
    cgroup_path="$(_vlh_cgroup_path_for_pid "${cid}" || true)"
    if [[ -z "${cgroup_path}" ]]; then
        _vlh_log "WARN: VFS bpftrace: cgroup path not found for pid ${cid}"
        return 0
    fi

    VFS_TRACE_PREFIX="${DATA_HOST}/lmcache"
    bt_out="${REPORT_DIR}/vfs_dir_io_trace.gen.bt"
    tsv_out="${REPORT_DIR}/vfs_dir_io.tsv"
    VFS_DIR_TSV="${tsv_out}"

    escaped="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "${cgroup_path}")"
    sed "s|__CGROUP_PATH__|${escaped}|g" "${SLURM_LIB}/vfs_dir_io_trace.bt.in" > "${bt_out}"

    _vlh_log "Starting bpftrace VFS trace cgroup=${cgroup_path} -> ${tsv_out}"
    bpftrace -q "${bt_out}" > "${tsv_out}" 2>"${REPORT_DIR}/vfs_dir_io.bpftrace.log" &
    VFS_BPFTRACE_PID=$!
    _vlh_metadata_append "VFS_TRACE_PREFIX" "${VFS_TRACE_PREFIX}"
    _vlh_metadata_append "VFS_TRACE_CGROUP_PATH" "${cgroup_path}"
}

_vlh_bpftrace_vfs_stop() {
    if [[ -n "${VFS_BPFTRACE_PID:-}" ]] && kill -0 "${VFS_BPFTRACE_PID}" 2>/dev/null; then
        kill "${VFS_BPFTRACE_PID}" 2>/dev/null || true
        wait "${VFS_BPFTRACE_PID}" 2>/dev/null || true
    fi
    if [[ -f "${VFS_DIR_TSV:-}" ]]; then
        _vlh_metadata_append "vfs_dir_io.tsv" "${VFS_DIR_TSV} lines=$(wc -l < "${VFS_DIR_TSV}")"
    fi
}

_vlh_docker_cleanup() {
    if [[ -n "${CONTAINER_NAME:-}" ]]; then
        docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi
}

_vlh_docker_preflight() {
    if docker info >/dev/null 2>&1; then
        return 0
    fi
    _vlh_log "ERROR: cannot access Docker on $(hostname -s 2>/dev/null || hostname)"
    _vlh_log "  (${DOCKER_HOST:-unix:///var/run/docker.sock}: permission denied or daemon down)"
    _vlh_log "  Fix: add \$USER to the docker group on GPU nodes, or build the image on"
    _vlh_log "  this host once (where docker works), then submit with VLH_SKIP_BUILD=1:"
    _vlh_log "    make -C recipies/vllm-lmcache-hipfile build ROCM_ARCH=<gfx>"
    _vlh_log "    export VLH_SKIP_BUILD=1 && ./run-slurm.sh"
    return 1
}

_vlh_build_image() {
    local rc=0
    if ! _vlh_docker_preflight; then
        BUILD_RC=1
        return 1
    fi
    if _vlh_truthy "${VLH_SKIP_BUILD:-0}"; then
        if docker image inspect "${IMAGE_NAME:-vllm-lmcache-hipfile}" >/dev/null 2>&1; then
            _vlh_log "VLH_SKIP_BUILD=1 and image exists; skipping build"
            BUILD_RC=0
            return 0
        fi
        _vlh_log "VLH_SKIP_BUILD=1 but image missing; building anyway"
    fi
    T_BUILD_START="$(date -Iseconds)"
    if make -C "${RECIPE_DIR}" build ROCM_ARCH="${ROCM_ARCH}" IMAGE_NAME="${IMAGE_NAME:-vllm-lmcache-hipfile}"; then
        BUILD_RC=0
    else
        BUILD_RC=$?
        rc=${BUILD_RC}
    fi
    T_BUILD_END="$(date -Iseconds)"
    IMAGE_ID="$(docker image inspect -f '{{.Id}}' "${IMAGE_NAME:-vllm-lmcache-hipfile}" 2>/dev/null || true)"
    _vlh_metadata_append "IMAGE_ID" "${IMAGE_ID}"
    _vlh_metadata_append "BUILD_RC" "${BUILD_RC}"
    _vlh_metadata_append "T_BUILD" "${T_BUILD_START} .. ${T_BUILD_END}"
    return "${rc}"
}

_vlh_wait_vllm() {
    local port="${VLLM_PORT:-8000}"
    local url="http://127.0.0.1:${port}/v1/models"
    local timeout="${VLH_VLLM_READY_TIMEOUT:-1800}"
    local waited=0
    _vlh_log "Waiting for vLLM at ${url} (timeout ${timeout}s) ..."
    while (( waited < timeout )); do
        if curl -sf "${url}" >/dev/null 2>&1; then
            _vlh_log "vLLM HTTP API is up"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    _vlh_log "ERROR: vLLM not ready after ${timeout}s"
    return 1
}

_vlh_golden_hf_home() {
    if [[ -n "${VLH_HF_HOME:-}" ]]; then
        printf '%s' "${VLH_HF_HOME}"
        return 0
    fi
    if [[ -n "${VLH_SHARED_ROOT:-}" ]]; then
        printf '%s' "${VLH_SHARED_ROOT}/hf"
        return 0
    fi
    printf '/scratch/%s/vllm-lmcache-hipfile/hf' "${USER:-unknown}"
}

_vlh_resolve_hf_home() {
  # Golden cache only — never ${VLH_NVME_BASE}/hf (per-job LMCache storage).
    HF_HOST="$(_vlh_golden_hf_home)"
    export HF_HOST
    _vlh_metadata_append "VLH_HF_HOME" "${HF_HOST}"
}

_vlh_chown_to_job_user() {
    local path="$1"
    local who grp
    [[ -e "${path}" ]] || return 0
    who="${SLURM_JOB_USER:-${USER}}"
    grp="$(id -gn "${who}" 2>/dev/null || echo "${who}")"
    if chown -R "${who}:${grp}" "${path}" 2>/dev/null; then
        return 0
    fi
    if command -v sudo >/dev/null 2>&1 \
        && sudo chown -R "${who}:${grp}" "${path}" 2>/dev/null; then
        return 0
    fi
    return 1
}

_vlh_hf_fix_ownership() {
    local hf="${HF_HOST:-$(_vlh_golden_hf_home)}"
    [[ -d "${hf}" ]] || return 0
    if ! find "${hf}" -maxdepth 4 ! -user "${SLURM_JOB_USER:-${USER}}" \
        -print -quit 2>/dev/null | grep -q .; then
        return 0
    fi
    _vlh_log "Fixing HF cache ownership under ${hf} (docker wrote as root)"
    if _vlh_chown_to_job_user "${hf}"; then
        return 0
    fi
    _vlh_log "WARN: could not chown ${hf}; use: sudo chown -R \$USER:\$(id -gn) ${hf}"
}

_vlh_resolve_gutenberg_data_root() {
    if [[ -n "${VLH_GUTENBERG_DATA_ROOT:-}" ]]; then
        GUTENBERG_DATA_ROOT="${VLH_GUTENBERG_DATA_ROOT}"
    elif [[ -n "${VLH_SHARED_ROOT:-}" ]]; then
        GUTENBERG_DATA_ROOT="${VLH_SHARED_ROOT}/gutenberg"
    else
        GUTENBERG_DATA_ROOT="${REPO_DIR}/data/gutenberg"
    fi
    export GUTENBERG_DATA_ROOT
}

_vlh_gutenberg_fixtures_present() {
    local root="$1"
    [[ -d "${root}" ]] || return 1
    find "${root}" -maxdepth 2 -name '*.questions.json' -print -quit 2>/dev/null | grep -q .
}

_vlh_gutenberg_prereqs() {
    _vlh_resolve_gutenberg_data_root
    if ! _vlh_gutenberg_fixtures_present "${GUTENBERG_DATA_ROOT}"; then
        _vlh_log "ERROR: no Gutenberg fixtures under ${GUTENBERG_DATA_ROOT}"
        _vlh_log "Generate once, e.g.:"
        _vlh_log "  make -C ${REPO_DIR}/benchmarks/llm-prefill-benchmark data-all"
        _vlh_log "  # or on shared storage:"
        _vlh_log "  export VLH_GUTENBERG_DATA_ROOT=/scratch/\$USER/vllm-lmcache-hipfile/gutenberg"
        _vlh_log "  make -C ${LLM_PREFILL_BENCH_ROOT} data-all BOOK_DATA_ROOT=\${VLH_GUTENBERG_DATA_ROOT}"
        return 1
    fi
    command -v jq >/dev/null 2>&1 || {
        _vlh_log "ERROR: jq required for run-long.sh (install on compute nodes)"
        return 1
    }
    _vlh_metadata_append "BOOK_DATA_ROOT" "${GUTENBERG_DATA_ROOT}"
    return 0
}

_vlh_run_long_env() {
    local port="${VLLM_PORT:-8000}"
    local model
    model="$(_vlh_served_model)"
    export BOOK_DATA_ROOT="${GUTENBERG_DATA_ROOT}"
    # run-long.sh appends /v1/chat/completions; do not include /v1 here.
    export BASE_URL="http://127.0.0.1:${port}"
    export MODEL="${model}"
    export ITERATIONS="${VLH_RUN_LONG_ITERATIONS:-1}"
    export BOOK_SLUG="${BOOK_SLUG:-}"
    export BOOK_SLUGS="${BOOK_SLUGS:-}"
    export BOOK_SLUG_FILE="${BOOK_SLUG_FILE:-}"
    export RUN_LONG_COMBINE_CHUNKS="${RUN_LONG_COMBINE_CHUNKS:-1}"
    export RUN_LONG_MAX_TOKENS="${VLH_RUN_LONG_MAX_TOKENS:-512}"
}

_vlh_run_gutenberg_serial() {
    local out="${REPORT_DIR}/run-long.jsonl"
    _vlh_gutenberg_prereqs || return 1
    _vlh_run_long_env
    _vlh_log "Running run-long.sh BOOK_DATA_ROOT=${GUTENBERG_DATA_ROOT}"
    : > "${out}"
    if RUN_LONG_SEED="${RUN_LONG_SEED:-}" \
        bash "${LLM_PREFILL_BENCH_ROOT}/run-long.sh" >> "${out}" 2>&1; then
        return 0
    fi
    return 1
}

_vlh_run_gutenberg_parallel() {
    local workers="${VLH_RUN_LONG_WORKERS:-4}"
    local log="${REPORT_DIR}/run-long-parallel.log"
    _vlh_gutenberg_prereqs || return 1
    _vlh_run_long_env
    _vlh_log "Running run-long-parallel.sh workers=${workers} BOOK_DATA_ROOT=${GUTENBERG_DATA_ROOT}"
    _vlh_metadata_append "VLH_RUN_LONG_WORKERS" "${workers}"
    _vlh_metadata_append "VLH_RUN_LONG_ITERATIONS" "${ITERATIONS}"
    if OUTPUT_DIR="${REPORT_DIR}/run-long-parallel" \
        WORKERS="${workers}" \
        BASE_SEED="${VLH_RUN_LONG_BASE_SEED:-${BASE_SEED:-$RANDOM}}" \
        STAGGER_SEC="${VLH_RUN_LONG_STAGGER_SEC:-0}" \
        PROGRESS="${VLH_RUN_LONG_PROGRESS:-0}" \
        RUN_LONG_SEED="${RUN_LONG_SEED:-}" \
        bash "${LLM_PREFILL_BENCH_ROOT}/run-long-parallel.sh" >> "${log}" 2>&1; then
        _vlh_metadata_append "run-long-parallel log" "${log}"
        _vlh_metadata_append "run-long-parallel dir" "${REPORT_DIR}/run-long-parallel"
        return 0
    fi
    return 1
}

_vlh_run_gutenberg() {
    if _vlh_truthy "${VLH_RUN_LONG_PARALLEL:-1}"; then
        _vlh_run_gutenberg_parallel
    else
        _vlh_run_gutenberg_serial
    fi
}

_vlh_run_long_doc_qa() {
    local port="${VLLM_PORT:-8000}"
    local model
    model="$(_vlh_served_model)"
    local out="${REPORT_DIR}/long_doc_qa.json"
    _vlh_log "Running long_doc_qa.py (model=${model})"
    docker exec "${CONTAINER_NAME}" python3 \
        /app/LMCache/benchmarks/long_doc_qa/long_doc_qa.py \
        --port "${port}" \
        --model "${model}" \
        --num-documents "${VLH_BENCH_NUM_DOCUMENTS:-40}" \
        --document-length "${VLH_BENCH_DOCUMENT_LENGTH:-24000}" \
        --output-len "${VLH_BENCH_OUTPUT_LEN:-128}" \
        --repeat-count "${VLH_BENCH_REPEAT_COUNT:-4}" \
        --repeat-mode "${VLH_BENCH_REPEAT_MODE:-tile}" \
        --hit-miss-ratio "${VLH_BENCH_HIT_MISS_RATIO:-1:2}" \
        --max-inflight-requests "${VLH_BENCH_MAX_INFLIGHT:-4}" \
        --sleep-time-after-warmup "${VLH_BENCH_SLEEP_AFTER_WARMUP:-10}" \
        --visualize \
        --completions \
        --json-output \
        --trim-fraction "${VLH_BENCH_TRIM_FRACTION:-0.1}" \
        > "${out}" 2>&1
}

_vlh_run_test_aic() {
    local out="${REPORT_DIR}/test_aic.json"
    local -a _aic_extra=()
    local log_subdir="${CONTAINER_LOG_SUBDIR:-vllm-lmcache-hipfile}"
    _vlh_log "Running test-aic.py"
    if [[ -n "${VLH_TEST_AIC_EXTRA_ARGS:-}" ]]; then
        read -r -a _aic_extra <<< "${VLH_TEST_AIC_EXTRA_ARGS}"
    fi
    docker exec "${CONTAINER_NAME}" python3 /app/scripts/test-aic.py \
        --json -o "/var/log/${log_subdir}/test_aic.json" \
        "${_aic_extra[@]}" \
        > "${out}" 2>&1 || true
    docker cp "${CONTAINER_NAME}:/var/log/${log_subdir}/test_aic.json" "${out}" 2>/dev/null || true
}

_vlh_collect_lmcache_api() {
    local gpu="${GPU:-0}"
    local sched_port="699${gpu}"
    local worker_port=$((sched_port + 1))
    local p path dest
    for p in "${sched_port}" "${worker_port}"; do
        for path in metrics "chunk_statistics/status"; do
            dest="${REPORT_DIR}/lmcache_internal_api_${p}_${path//\//_}"
            curl -sf "http://127.0.0.1:${p}/${path}" -o "${dest}.txt" 2>/dev/null || true
        done
    done
    if [[ -d "${DATA_HOST}/lmcache_chunk_stats" ]]; then
        mkdir -p "${REPORT_DIR}/lmcache_chunk_stats"
        cp -a "${DATA_HOST}/lmcache_chunk_stats/." "${REPORT_DIR}/lmcache_chunk_stats/" 2>/dev/null || true
    fi
}

_vlh_collect_artifacts() {
    local dest="${SLURM_LOG_DIR}"
    mkdir -p "${dest}"
    cp -a "${REPORT_DIR}/." "${dest}/" 2>/dev/null || true
    if [[ -f "${CONTAINER_LOG_HOST}/server.txt" ]]; then
        cp "${CONTAINER_LOG_HOST}/server.txt" "${dest}/server.txt" 2>/dev/null || true
    elif [[ -f "${REPORT_DIR}/logs/server.txt" ]]; then
        cp "${REPORT_DIR}/logs/server.txt" "${dest}/server.txt" 2>/dev/null || true
    fi
    _vlh_log "Artifacts copied to ${dest}"
}

_vlh_write_report() {
    local report="${REPORT_DIR}/report.md"
    {
        echo "# vllm-lmcache-hipfile Slurm job report"
        echo ""
        echo "- Job ID: ${SLURM_JOB_ID:-unknown}"
        echo "- JOB_ROOT: ${JOB_ROOT}"
        echo "- CONTAINER_NAME: ${CONTAINER_NAME}"
        echo "- BUILD_RC: ${BUILD_RC:-?} RUN_RC: ${RUN_RC:-?} PHASE_RC: ${PHASE_RC:-?}"
        echo ""
        if [[ -f "${REPORT_DIR}/results-summary.md" ]]; then
            cat "${REPORT_DIR}/results-summary.md"
            echo ""
        else
            echo "## Summary"
            echo ""
            echo '```'
            cat "${REPORT_DIR}/summary.txt" 2>/dev/null || true
            echo '```'
            echo ""
        fi
        echo "## Artifact paths"
        echo ""
        echo "- Report dir: \`${REPORT_DIR}\`"
        echo "- Slurm copy: \`${SLURM_LOG_DIR}\`"
        if [[ -d "${REPORT_DIR}/run-long-parallel" ]]; then
            echo "- Gutenberg: \`${REPORT_DIR}/run-long-parallel/\`"
        fi
        if [[ -f "${REPORT_DIR}/run-long.jsonl" ]]; then
            echo "- Gutenberg serial: \`${REPORT_DIR}/run-long.jsonl\`"
        fi
        if [[ -f "${REPORT_DIR}/long_doc_qa.json" ]]; then
            echo "- long_doc_qa: \`${REPORT_DIR}/long_doc_qa.json\`"
        fi
        if [[ -f "${REPORT_DIR}/logs/server.txt" ]] || [[ -f "${REPORT_DIR}/../server.txt" ]]; then
            echo "- server log: \`server.txt\` (under Slurm log dir)"
        fi
    } > "${report}"
}

_vlh_write_summary() {
    {
        echo "RUNTIME=docker PHASE_RC=${PHASE_RC:-?} RUN_RC=${RUN_RC:-?} BUILD_RC=${BUILD_RC:-?}"
        echo "JOB_ROOT=${JOB_ROOT}"
    } > "${REPORT_DIR}/summary.txt"
}

_vlh_summarize_job() {
    local script="${SLURM_LIB}/summarize-recipe-job.py"
    local recipe="${IMAGE_NAME:-vllm-lmcache-hipfile}"
    if [[ ! -f "${script}" ]]; then
        _vlh_log "WARN: ${script} missing; skip results summary"
        return 0
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        _vlh_log "WARN: python3 not found; skip results summary"
        return 0
    fi
    if python3 "${script}" --recipe-name "${recipe}" "${REPORT_DIR}"; then
        _vlh_metadata_append "results-summary.md" "${REPORT_DIR}/results-summary.md"
        _vlh_metadata_append "results-summary.json" "${REPORT_DIR}/results-summary.json"
        _vlh_log "Results summary: ${REPORT_DIR}/results-summary.md"
        echo ""
        echo "========== ${recipe} results summary =========="
        sed -n '1,80p' "${REPORT_DIR}/results-summary.md" 2>/dev/null || true
        echo "================================================="
        echo ""
    else
        _vlh_log "WARN: summarize-recipe-job.py failed"
    fi
}

_vlh_job_cleanup() {
    _vlh_bpftrace_vfs_stop
    _vlh_bpftrace_nvme_stop
    _vlh_nvme_smart_log "end"
    if [[ -n "${CONTAINER_NAME:-}" ]]; then
        docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
        docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
        _vlh_hf_fix_ownership
    fi
}
