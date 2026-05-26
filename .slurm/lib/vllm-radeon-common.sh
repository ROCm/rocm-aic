#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Shared helpers for vllm-radeon Slurm jobs (sourced by vllm-radeon-mi300.sbatch).

_radeon_env_aliases() {
    local _k _radeon _kurt
    for _k in NVME_BASE NVME_BLK_BPFTRACE NVME_SMART_LOG NVME_MKFS \
        NVME_AUTO_DEVICE NVME_MOUNT NVME_DEVICE VFS_BPFTRACE \
        VFS_BPFTRACE_APT_INSTALL LMCACHE_ENABLE_KV_EVENTS SKIP_BUILD BENCHMARK; do
        _radeon="RADEON_${_k}"
        _kurt="KURT_${_k}"
        if [[ -z "${!_radeon:-}" && -n "${!_kurt:-}" ]]; then
            export "${_radeon}=${!_kurt}"
        fi
    done
}

_radeon_log() {
    printf '[%s] %s\n' "$(date -Iseconds)" "$*"
}

_radeon_truthy() {
    local v="${1:-0}"
    case "${v,,}" in
        1 | true | yes | on) return 0 ;;
        *) return 1 ;;
    esac
}

_radeon_resolve_hf_token() {
    if [[ -n "${HF_TOKEN:-}" ]]; then
        return 0
    fi
    if [[ -n "${HF_TOKEN_FILE:-}" && -r "${HF_TOKEN_FILE}" ]]; then
        HF_TOKEN="$(tr -d '\r\n' < "${HF_TOKEN_FILE}")"
        export HF_TOKEN
        return 0
    fi
    _radeon_log "ERROR: set HF_TOKEN or a readable HF_TOKEN_FILE"
    return 1
}

_radeon_detect_rocm_arch() {
    if [[ -n "${ROCM_ARCH:-}" ]]; then
        _radeon_log "Using ROCM_ARCH=${ROCM_ARCH}"
        return 0
    fi
    ROCM_ARCH="$(rocm_agent_enumerator 2>/dev/null | grep -E '^gfx' | head -1 || true)"
    if [[ -z "${ROCM_ARCH}" ]]; then
        ROCM_ARCH="$(rocm_agent_enumerator 2>/dev/null | grep -v '^gfx0' | sort -u | tail -1 || true)"
    fi
    if [[ -z "${ROCM_ARCH}" ]]; then
        _radeon_log "ERROR: ROCM_ARCH empty (set ROCM_ARCH or install rocm_agent_enumerator)"
        return 1
    fi
    export ROCM_ARCH
    _radeon_log "Using ROCM_ARCH=${ROCM_ARCH}"
}

_radeon_metadata_append() {
    local key="$1"
    local val="$2"
    printf '%s: %s\n' "${key}" "${val}" >> "${METADATA_FILE}"
}

_radeon_metadata_init() {
    METADATA_FILE="${REPORT_DIR}/metadata.txt"
    : > "${METADATA_FILE}"
    _radeon_metadata_append "hostname" "$(hostname -f 2>/dev/null || hostname)"
    _radeon_metadata_append "date" "$(date -Iseconds)"
    _radeon_metadata_append "RUNTIME" "docker"
    _radeon_metadata_append "RECIPE_DIR" "${RECIPE_DIR}"
    _radeon_metadata_append "ROCM_ARCH" "${ROCM_ARCH}"
    _radeon_metadata_append "IMAGE_NAME" "${IMAGE_NAME:-vllm-radeon}"
    _radeon_metadata_append "SLURM_JOB_ID" "${SLURM_JOB_ID:-}"
    _radeon_metadata_append "SLURM_JOB_NODELIST" "${SLURM_JOB_NODELIST:-}"
    _radeon_metadata_append "SLURM_SUBMIT_DIR" "${SLURM_SUBMIT_DIR:-}"
    _radeon_metadata_append "JOB_ROOT" "${JOB_ROOT}"
    _radeon_metadata_append "RADEON_NVME_BASE" "${RADEON_NVME_BASE}"
    _radeon_metadata_append "RADEON_LMCACHE_IO" "${RADEON_LMCACHE_IO:-hipfile}"
    _radeon_metadata_append "RADEON_BENCHMARK" "${RADEON_BENCHMARK:-long_doc_qa}"
    _radeon_metadata_append "CONTAINER_NAME" "${CONTAINER_NAME}"
}

_radeon_nvme_setup() {
    local mkfs="${RADEON_NVME_MKFS:-0}"
    local auto="${RADEON_NVME_AUTO_DEVICE:-0}"
    local device="${RADEON_NVME_DEVICE:-}"
    local mount="${RADEON_NVME_MOUNT:-}"

    if ! _radeon_truthy "${mkfs}"; then
        return 0
    fi

    if _radeon_truthy "${auto}" && [[ -z "${device}" ]]; then
        device="$(
            lsblk -dn -o NAME,MOUNTPOINT,FSTYPE 2>/dev/null | awk '
                $1 ~ /^nvme[0-9]+n[0-9]+$/ && $2 == "" && $3 == "" { print "/dev/" $1; exit }
            ' || true
        )"
        if [[ -z "${device}" ]]; then
            _radeon_log "WARN: RADEON_NVME_AUTO_DEVICE=1 but no unmounted whole-disk nvme* found"
        else
            RADEON_NVME_DEVICE="${device}"
            export RADEON_NVME_DEVICE
            _radeon_log "RADEON_NVME_AUTO_DEVICE selected ${device}"
        fi
    fi

    if [[ -z "${device}" || -z "${mount}" ]]; then
        _radeon_log "ERROR: RADEON_NVME_MKFS=1 requires RADEON_NVME_DEVICE and RADEON_NVME_MOUNT"
        return 1
    fi

    _radeon_log "mkfs + mount ${device} -> ${mount}"
    if command -v mkfs.ext4 >/dev/null 2>&1; then
        mkfs.ext4 -F "${device}"
    else
        _radeon_log "ERROR: mkfs.ext4 not found"
        return 1
    fi
    mkdir -p "${mount}"
    mount "${device}" "${mount}"
    RADEON_NVME_BASE="${mount}"
    export RADEON_NVME_BASE
    _radeon_metadata_append "RADEON_NVME_DEVICE" "${device}"
    _radeon_metadata_append "RADEON_NVME_MOUNT" "${mount}"
}

_radeon_st_rdev_decimal() {
    local dev_path="$1"
    local maj min
    maj="$(stat -c '%t' "${dev_path}" 2>/dev/null || echo 0)"
    min="$(stat -c '%T' "${dev_path}" 2>/dev/null || echo 0)"
    printf '%d' $((16#${maj} * 256 + 16#${min}))
}

_radeon_block_device_for_path() {
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

_radeon_nvme_smart_log() {
    local phase="$1"
    local dev="${RADEON_NVME_DEVICE:-}"
    if ! _radeon_truthy "${RADEON_NVME_SMART_LOG:-0}"; then
        return 0
    fi
    if ! command -v nvme >/dev/null 2>&1; then
        _radeon_log "WARN: nvme-cli not installed; skip smart-log ${phase}"
        return 0
    fi
    if [[ -z "${dev}" ]]; then
        dev="$(_radeon_block_device_for_path "${DATA_HOST}" 2>/dev/null || true)"
    fi
    if [[ -z "${dev}" ]]; then
        _radeon_log "WARN: nvme smart-log ${phase}: no block device for ${DATA_HOST}"
        return 0
    fi
    local out="${REPORT_DIR}/nvme_smart_log_job_${phase}.json"
    if nvme smart-log "${dev}" -o json > "${out}" 2>>"${REPORT_DIR}/nvme_smart_log.log"; then
        _radeon_metadata_append "nvme_smart_log_${phase}" "${out}"
    else
        _radeon_log "WARN: nvme smart-log ${phase} failed (root may be required)"
    fi
}

_radeon_bpftrace_nvme_start() {
    if ! _radeon_truthy "${RADEON_NVME_BLK_BPFTRACE:-0}"; then
        return 0
    fi
    if ! command -v bpftrace >/dev/null 2>&1; then
        _radeon_log "WARN: bpftrace not installed; skip NVMe block trace"
        return 0
    fi

    local dev_path disk_name st_rdev bt_out tsv_out
    dev_path="$(_radeon_block_device_for_path "${DATA_HOST}" 2>/dev/null || true)"
    if [[ -z "${dev_path}" ]]; then
        disk_name="$(lsblk -dn -o NAME,MOUNTPOINT 2>/dev/null | awk '$2=="" && $1 ~ /^nvme/ {print $1; exit}')"
        if [[ -n "${disk_name}" ]]; then
            dev_path="/dev/${disk_name}"
        fi
    fi
    if [[ -z "${dev_path}" || ! -b "${dev_path}" ]]; then
        _radeon_log "WARN: NVMe bpftrace: no block device for ${DATA_HOST}"
        return 0
    fi

    disk_name="$(basename "${dev_path}")"
    st_rdev="$(_radeon_st_rdev_decimal "${dev_path}")"
    bt_out="${REPORT_DIR}/nvme_block_io_trace.gen.bt"
    tsv_out="${REPORT_DIR}/nvme_blk_io.tsv"
    NVME_BLK_TSV="${tsv_out}"

    sed "s/__ST_RDEV__/${st_rdev}/g" "${SLURM_LIB}/nvme_block_io_trace.bt.in" > "${bt_out}"
    _radeon_log "nvme block bpftrace disk_name=${disk_name} dev_path=${dev_path} st_rdev=${st_rdev}"
    _radeon_log "Starting bpftrace NVMe block trace -> ${tsv_out}"

    bpftrace -q "${bt_out}" > "${tsv_out}" 2>"${REPORT_DIR}/nvme_blk_io.bpftrace.log" &
    NVME_BLK_BPFTRACE_PID=$!
    sleep 1
    if ! kill -0 "${NVME_BLK_BPFTRACE_PID}" 2>/dev/null; then
        _radeon_log "WARN: nvme bpftrace exited immediately; see ${REPORT_DIR}/nvme_blk_io.bpftrace.log"
        NVME_BLK_BPFTRACE_PID=
    fi
    _radeon_metadata_append "nvme_blk disk_name" "${disk_name}"
    _radeon_metadata_append "nvme_blk dev_path" "${dev_path}"
    _radeon_metadata_append "nvme_blk st_rdev" "${st_rdev}"
}

_radeon_bpftrace_nvme_stop() {
    if [[ -n "${NVME_BLK_BPFTRACE_PID:-}" ]] && kill -0 "${NVME_BLK_BPFTRACE_PID}" 2>/dev/null; then
        kill "${NVME_BLK_BPFTRACE_PID}" 2>/dev/null || true
        wait "${NVME_BLK_BPFTRACE_PID}" 2>/dev/null || true
    fi
    if [[ -f "${NVME_BLK_TSV:-}" ]]; then
        _radeon_metadata_append "nvme_blk_io.tsv" "${NVME_BLK_TSV} lines=$(wc -l < "${NVME_BLK_TSV}")"
    fi
}

_radeon_cgroup_path_for_pid() {
    local pid="$1"
    local line path
    line="$(grep -E '^0::' "/proc/${pid}/cgroup" 2>/dev/null | head -1 || true)"
    path="${line#0::}"
    if [[ -z "${path}" ]]; then
        return 1
    fi
    printf '/sys/fs/cgroup%s' "${path}"
}

_radeon_bpftrace_vfs_start() {
    if ! _radeon_truthy "${RADEON_VFS_BPFTRACE:-0}"; then
        return 0
    fi
    if ! command -v bpftrace >/dev/null 2>&1; then
        if _radeon_truthy "${RADEON_VFS_BPFTRACE_APT_INSTALL:-0}"; then
            _radeon_log "Installing bpftrace (apt) ..."
            apt-get update -qq && DEBIAN_FRONTEND=noninteractive \
                apt-get install -y -qq bpftrace || true
        fi
    fi
    if ! command -v bpftrace >/dev/null 2>&1; then
        _radeon_log "WARN: bpftrace not installed; skip VFS trace"
        return 0
    fi

    local cid docker_pid cgroup_path bt_out tsv_out escaped
    cid="$(docker inspect -f '{{.State.Pid}}' "${CONTAINER_NAME}" 2>/dev/null || true)"
    if [[ -z "${cid}" || "${cid}" == "0" ]]; then
        _radeon_log "WARN: VFS bpftrace: container pid not available yet"
        return 0
    fi
    cgroup_path="$(_radeon_cgroup_path_for_pid "${cid}" || true)"
    if [[ -z "${cgroup_path}" ]]; then
        _radeon_log "WARN: VFS bpftrace: cgroup path not found for pid ${cid}"
        return 0
    fi

    VFS_TRACE_PREFIX="${DATA_HOST}/lmcache"
    VFS_TRACE_CGROUP_PATH="${cgroup_path}"
    bt_out="${REPORT_DIR}/vfs_dir_io_trace.gen.bt"
    tsv_out="${REPORT_DIR}/vfs_dir_io.tsv"
    VFS_DIR_TSV="${tsv_out}"

    escaped="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "${cgroup_path}")"
    sed "s|__CGROUP_PATH__|${escaped}|g" "${SLURM_LIB}/vfs_dir_io_trace.bt.in" > "${bt_out}"

    _radeon_log "Starting bpftrace VFS trace cgroup=${cgroup_path} -> ${tsv_out}"
    bpftrace -q "${bt_out}" > "${tsv_out}" 2>"${REPORT_DIR}/vfs_dir_io.bpftrace.log" &
    VFS_BPFTRACE_PID=$!
    _radeon_metadata_append "VFS_TRACE_PREFIX" "${VFS_TRACE_PREFIX}"
    _radeon_metadata_append "VFS_TRACE_CGROUP_PATH" "${cgroup_path}"
}

_radeon_bpftrace_vfs_stop() {
    if [[ -n "${VFS_BPFTRACE_PID:-}" ]] && kill -0 "${VFS_BPFTRACE_PID}" 2>/dev/null; then
        kill "${VFS_BPFTRACE_PID}" 2>/dev/null || true
        wait "${VFS_BPFTRACE_PID}" 2>/dev/null || true
    fi
    if [[ -f "${VFS_DIR_TSV:-}" ]]; then
        _radeon_metadata_append "vfs_dir_io.tsv" "${VFS_DIR_TSV} lines=$(wc -l < "${VFS_DIR_TSV}")"
    fi
}

_radeon_docker_cleanup() {
    if [[ -n "${CONTAINER_NAME:-}" ]]; then
        docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi
}

_radeon_build_image() {
    local rc=0
    if _radeon_truthy "${RADEON_SKIP_BUILD:-0}"; then
        if docker image inspect "${IMAGE_NAME:-vllm-radeon}" >/dev/null 2>&1; then
            _radeon_log "RADEON_SKIP_BUILD=1 and image exists; skipping build"
            BUILD_RC=0
            return 0
        fi
        _radeon_log "RADEON_SKIP_BUILD=1 but image missing; building anyway"
    fi
    T_BUILD_START="$(date -Iseconds)"
    if make -C "${RECIPE_DIR}" build ROCM_ARCH="${ROCM_ARCH}" IMAGE_NAME="${IMAGE_NAME:-vllm-radeon}"; then
        BUILD_RC=0
    else
        BUILD_RC=$?
        rc=${BUILD_RC}
    fi
    T_BUILD_END="$(date -Iseconds)"
    IMAGE_ID="$(docker image inspect -f '{{.Id}}' "${IMAGE_NAME:-vllm-radeon}" 2>/dev/null || true)"
    _radeon_metadata_append "IMAGE_ID" "${IMAGE_ID}"
    _radeon_metadata_append "BUILD_RC" "${BUILD_RC}"
    _radeon_metadata_append "T_BUILD" "${T_BUILD_START} .. ${T_BUILD_END}"
    return "${rc}"
}

_radeon_wait_vllm() {
    local port="${VLLM_PORT:-8000}"
    local url="http://127.0.0.1:${port}/v1/models"
    local timeout="${RADEON_VLLM_READY_TIMEOUT:-1800}"
    local waited=0
    _radeon_log "Waiting for vLLM at ${url} (timeout ${timeout}s) ..."
    while (( waited < timeout )); do
        if curl -sf "${url}" >/dev/null 2>&1; then
            _radeon_log "vLLM HTTP API is up"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    _radeon_log "ERROR: vLLM not ready after ${timeout}s"
    return 1
}

_radeon_run_long_doc_qa() {
    local port="${VLLM_PORT:-8000}"
    local model="${RADEON_BENCH_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
    local out="${REPORT_DIR}/long_doc_qa.json"
    _radeon_log "Running long_doc_qa.py (model=${model})"
    docker exec "${CONTAINER_NAME}" python3 \
        /app/LMCache/benchmarks/long_doc_qa/long_doc_qa.py \
        --port "${port}" \
        --model "${model}" \
        --num-documents "${RADEON_BENCH_NUM_DOCUMENTS:-40}" \
        --document-length "${RADEON_BENCH_DOCUMENT_LENGTH:-24000}" \
        --output-len "${RADEON_BENCH_OUTPUT_LEN:-128}" \
        --repeat-count "${RADEON_BENCH_REPEAT_COUNT:-4}" \
        --repeat-mode "${RADEON_BENCH_REPEAT_MODE:-tile}" \
        --hit-miss-ratio "${RADEON_BENCH_HIT_MISS_RATIO:-1:2}" \
        --max-inflight-requests "${RADEON_BENCH_MAX_INFLIGHT:-4}" \
        --sleep-time-after-warmup "${RADEON_BENCH_SLEEP_AFTER_WARMUP:-10}" \
        --visualize \
        --completions \
        --json-output \
        --trim-fraction "${RADEON_BENCH_TRIM_FRACTION:-0.1}" \
        > "${out}" 2>&1
}

_radeon_run_test_aic() {
    local out="${REPORT_DIR}/test_aic.json"
    _radeon_log "Running test-aic.py"
    docker exec "${CONTAINER_NAME}" python3 /app/scripts/test-aic.py \
        --json -o "/var/log/vllm-radeon/test_aic.json" \
        ${RADEON_TEST_AIC_EXTRA_ARGS:-} \
        > "${out}" 2>&1 || true
    docker cp "${CONTAINER_NAME}:/var/log/vllm-radeon/test_aic.json" "${out}" 2>/dev/null || true
}

_radeon_collect_lmcache_api() {
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

_radeon_collect_artifacts() {
    local dest="${SLURM_LOG_DIR}"
    mkdir -p "${dest}"
    cp -a "${REPORT_DIR}/." "${dest}/" 2>/dev/null || true
    if [[ -f "${CONTAINER_LOG_HOST}/server.txt" ]]; then
        cp "${CONTAINER_LOG_HOST}/server.txt" "${dest}/server.txt" 2>/dev/null || true
    elif [[ -f "${REPORT_DIR}/logs/server.txt" ]]; then
        cp "${REPORT_DIR}/logs/server.txt" "${dest}/server.txt" 2>/dev/null || true
    fi
    _radeon_log "Artifacts copied to ${dest}"
}

_radeon_write_report() {
    local report="${REPORT_DIR}/report.md"
    {
        echo "# vllm-radeon Slurm job report"
        echo ""
        echo "- Job ID: ${SLURM_JOB_ID:-unknown}"
        echo "- JOB_ROOT: ${JOB_ROOT}"
        echo "- CONTAINER_NAME: ${CONTAINER_NAME}"
        echo "- BUILD_RC: ${BUILD_RC:-?} RUN_RC: ${RUN_RC:-?} PHASE_RC: ${PHASE_RC:-?}"
        echo ""
        echo "## Summary"
        echo ""
        echo '```'
        cat "${REPORT_DIR}/summary.txt" 2>/dev/null || true
        echo '```'
        echo ""
        if [[ -f "${REPORT_DIR}/long_doc_qa.json" ]]; then
            echo "## long_doc_qa"
            echo ""
            echo "See \`${REPORT_DIR}/long_doc_qa.json\`"
        fi
    } > "${report}"
}

_radeon_write_summary() {
    {
        echo "RUNTIME=docker PHASE_RC=${PHASE_RC:-?} RUN_RC=${RUN_RC:-?} BUILD_RC=${BUILD_RC:-?}"
        echo "JOB_ROOT=${JOB_ROOT}"
    } > "${REPORT_DIR}/summary.txt"
}

_radeon_job_cleanup() {
    _radeon_bpftrace_vfs_stop
    _radeon_bpftrace_nvme_stop
    _radeon_nvme_smart_log "end"
    if [[ -n "${CONTAINER_NAME:-}" ]]; then
        docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
        docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi
}
