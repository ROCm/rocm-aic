#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# accuracy-test: differential KV-integrity gate -- the COMPUTE-NODE half.
#
# Answers "does routing KV through DRAM/NVMe change the model's answers?", which
# neither tiny-test (serves one completion) nor cliff (measures throughput) can.
#
# Five phases.  The two arms share a container name, a port and the GPU, so they
# cannot coexist -- the baseline is scored, torn down, and its number carried
# forward to the tiered arm's pytest run via AIC_ACCURACY_BASELINE_SCORE.
#
#   Phase 1  vram_only arm: plain vLLM, no LMCache, KV never leaves VRAM.
#            Score gsm8k, record the number, tear down.
#   Phase 2  kvd arm: LMCache + NIXL POSIX NVMe L2, constrained VRAM so blocks
#            actually evict.  Score gsm8k, run both assertions.
#   Phase 3  Liveness: assert the NVMe pool grew during phase 2.  Without this a
#            tiered arm that never tiered would pass the differential trivially
#            -- the run would be vacuous rather than green.
#   Phase 4  Restart vLLM only (LMCache keeps its DRAM/NVMe state), re-score the
#            full split, and assert within DELTA of the phase-2 score.  The
#            prompts are guaranteed cache hits, so this isolates
#            retrieval-from-NVMe.
#   Phase 5  Prove that re-score was SERVED from the tier, not recomputed.
#
# There is deliberately no committed golden: no reference.json, no md5 manifest
# of pool files.  Which pool slot a block lands in depends on allocation order,
# so a checksum would be a flake generator; see tests/accuracy/README.md.
#
# pytest runs host-side (it needs .venv and lm_eval; the image stays lean and the
# gsm8k download stays off the compute node), reaching vLLM on its `aic` bridge
# IP.  vLLM publishes no ports, so a bridge IP -- not localhost -- is the only
# host-side route; readiness probes use the `docker exec ... 127.0.0.1` form that
# the rest of the repo uses.
#
# Every scoring pass asks the full gsm8k split; there is no item cap to set.
#
# ---------------------------------------------------------------------------
# HOW THIS IS LAUNCHED
# ---------------------------------------------------------------------------
# Deliberately NOT a .sbatch file and deliberately carrying no #SBATCH header:
# `run-build-distribute.sh accuracy-test` submits a shim that exports the
# variables below and then execs this script, so any #SBATCH directive here
# would be inert (sbatch never parses this file).  Job sizing -- --time, --cpus-
# per-task, --mem, node selection -- lives on the _sbatch_run call in
# cmd_accuracy_test, alongside the AIC_ACCURACY_TIME/CPUS/MEM defaults.
#
# That submission path BLOCKS and propagates this script's exit code, which is
# what makes the gate usable from CI (`make accuracy-test`).  It is not the
# fire-and-forget model run-cliff.sbatch uses.
#
# Every value arrives as an environment variable, so the script is also runnable
# by hand on a GPU node that has the repo:
#
#   AIC_DAY_DIR=/path/to/repo AIC_LOG_DIR=/tmp/acc AIC_IMAGE=rocm-aic:latest \
#   AIC_TARBALL=/shared/rocm-aic-latest-gfx950.tar.zst AIC_DECOMPRESS_CMD='zstd -dc' \
#   AIC_ROCM_ARCH=gfx950 HF_HOME=/shared/hf AIC_ACCURACY_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
#   AIC_ACCURACY_DELTA=0.02 AIC_ACCURACY_READY_TIMEOUT=120 \
#       bash .slurm/run-accuracy.sh
#
# Required, all set by the submitter:
#   AIC_DAY_DIR                 repo root, absolute, on storage the node can see
#   AIC_LOG_DIR                 per-job log dir; scores and container logs land here
#   AIC_IMAGE                   image ref to load and run
#   AIC_TARBALL                 image tarball to load from
#   AIC_DECOMPRESS_CMD          matching decompressor (e.g. "zstd -dc")
#   AIC_ROCM_ARCH               arch tag forwarded to compose
#   HF_HOME                     persistent HF cache
#   AIC_ACCURACY_MODEL          model to serve
#   AIC_ACCURACY_DELTA          allowed tiered-vs-baseline gap, two-sided
#   AIC_ACCURACY_READY_TIMEOUT  x5s waits for the endpoint
# Optional:
#   HF_TOKEN                    gated-model download token (default: empty)
#   AIC_FORCE_LOAD              1 forces an image reload (default: 0)
#   AIC_ACCURACY_POOL_ROOT      L2 pool root (default: /tmp/aic-accuracy.$SLURM_JOB_ID)
#   AIC_ACCURACY_ALLOW_L1_ONLY  1 downgrades phase 5's L2-read gate to a warning

set -uo pipefail

# Fail loudly on a missing input.  Host-side interpolation used to guarantee
# these were present; now that they cross the process boundary as environment,
# an unset one would otherwise surface as an empty path deep inside a phase.
: "${AIC_DAY_DIR:?run-accuracy.sh: AIC_DAY_DIR must be set by the submitter}"
: "${AIC_LOG_DIR:?run-accuracy.sh: AIC_LOG_DIR must be set by the submitter}"
: "${AIC_IMAGE:?run-accuracy.sh: AIC_IMAGE must be set by the submitter}"
: "${AIC_TARBALL:?run-accuracy.sh: AIC_TARBALL must be set by the submitter}"
: "${AIC_DECOMPRESS_CMD:?run-accuracy.sh: AIC_DECOMPRESS_CMD must be set by the submitter}"
: "${AIC_ROCM_ARCH:?run-accuracy.sh: AIC_ROCM_ARCH must be set by the submitter}"
: "${HF_HOME:?run-accuracy.sh: HF_HOME must be set by the submitter}"
: "${AIC_ACCURACY_MODEL:?run-accuracy.sh: AIC_ACCURACY_MODEL must be set by the submitter}"
: "${AIC_ACCURACY_DELTA:?run-accuracy.sh: AIC_ACCURACY_DELTA must be set by the submitter}"
: "${AIC_ACCURACY_READY_TIMEOUT:?run-accuracy.sh: AIC_ACCURACY_READY_TIMEOUT must be set by the submitter}"

command -v docker >/dev/null 2>&1 || { echo "$(hostname): docker not found" >&2; exit 1; }
echo "[accuracy-test] host=$(hostname) docker=$(docker --version)"

# Load the image from the shared tarball only when needed (same marker logic as
# tiny-test): reload when forced, absent, or the tarball is newer.
_marker="/var/tmp/aic-loaded-$(id -u)-$(echo "${AIC_IMAGE}" | tr '/:' '__').mtime"
_tar_mtime="$(stat -c %Y "${AIC_TARBALL}" 2>/dev/null || echo 0)"
_have_img="$(docker images -q "${AIC_IMAGE}")"
_loaded_mtime="$(cat "${_marker}" 2>/dev/null || echo 0)"
if [ "${AIC_FORCE_LOAD:-0}" = "1" ] || [ -z "${_have_img}" ] || [ "${_tar_mtime}" -gt "${_loaded_mtime}" ]; then
    echo "[accuracy-test] loading ${AIC_IMAGE} from ${AIC_TARBALL}"
    ${AIC_DECOMPRESS_CMD} "${AIC_TARBALL}" | docker load >/dev/null
    echo "${_tar_mtime}" > "${_marker}" 2>/dev/null || true
else
    echo "[accuracy-test] image up to date on $(hostname) (id ${_have_img})"
fi

cd "${AIC_DAY_DIR}" || { echo "[accuracy-test] FAIL: cannot cd to ${AIC_DAY_DIR}" >&2; exit 1; }
# shellcheck source=/dev/null
source "${AIC_DAY_DIR}/monitoring/monitoring-lib.sh"
ensure_compose || { echo "[accuracy-test] docker compose unavailable" >&2; exit 1; }

# --- host-side pytest venv ---------------------------------------------------
# lm_eval is deliberately NOT in the image: the gsm8k download does not belong
# on a compute node's container and the image stays lean.  PYTHONNOUSERSITE=1
# because ~/.local shadows venv site-packages on these boxes.
export PYTHONNOUSERSITE=1
VENV="${AIC_DAY_DIR}/.venv"
if [ ! -x "${VENV}/bin/pytest" ]; then
    echo "[accuracy-test] creating host venv at ${VENV}"
    python3 -m venv "${VENV}" || { echo "[accuracy-test] FAIL: venv creation failed" >&2; exit 1; }
    "${VENV}/bin/pip" -q install --upgrade pip
    "${VENV}/bin/pip" -q install -e "${AIC_DAY_DIR}[accuracy]" || {
        echo "[accuracy-test] FAIL: could not install the accuracy extra" >&2; exit 1; }
fi
PYTEST="${VENV}/bin/pytest"
PYBIN="${VENV}/bin/python"

export AIC_ACCURACY_MODEL="${AIC_ACCURACY_MODEL}"
export AIC_ACCURACY_DELTA="${AIC_ACCURACY_DELTA}"
# This is CI: an endpoint the fixtures cannot reach must fail the job, not skip
# it green.  The fixtures default to skipping so the package stays runnable on a
# laptop with no GPU; here we have brought the arms up ourselves and every skip
# for unreachability means the gate scored nothing.  A deliberate absence (an
# unknown model's floor) remains a skip.
export AIC_ACCURACY_REQUIRED=1

# --- shared compose env ------------------------------------------------------
export IMAGE_REF="${AIC_IMAGE}"
export IMAGE_NAME="${AIC_IMAGE%:*}"
export ROCM_ARCH="${AIC_ROCM_ARCH}"
export GPU=0
export VLLM_MODEL="${AIC_ACCURACY_MODEL}"
export HF_HOME="${HF_HOME}"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export LOG="${AIC_LOG_DIR}"
# Pools are namespaced by job id and pre-cleared, the same idiom run-cliff.sbatch
# uses.  The fixed /tmp/aic-accuracy-{nvme,nfs} pair they replace was removed only
# by the EXIT trap, so a wall-clock SIGKILL (untrapped) stranded the whole pool --
# ~15 GB of NIXL slot files -- and two runs landing on one node shared it.
AIC_ACCURACY_POOL_ROOT="${AIC_ACCURACY_POOL_ROOT:-/tmp/aic-accuracy.${SLURM_JOB_ID:-manual}}"
export NVME_DATA="${AIC_ACCURACY_POOL_ROOT}/nvme"
export NFS_DATA="${AIC_ACCURACY_POOL_ROOT}/nfs"
# 5-shot gsm8k prompts land around 900-1300 tokens plus generation; 4096 leaves
# headroom without inflating the KV footprint.  Watch the vLLM log for
# truncation warnings if the model or shot count changes.
export VLM_MAX_MODEL_LEN=4096
export VLM_MAX_NUM_BATCHED_TOKENS=4096
export VLM_ATTENTION_BACKEND=TRITON_ATTN
export VLM_KV_CACHE_DTYPE=auto
# Pre-clear our own namespace: a requeued job keeps its id, so without this a
# retry would compute POOL_BEFORE against the killed attempt's slot files.
rm -rf "${AIC_ACCURACY_POOL_ROOT}"
mkdir -p "${NVME_DATA}" "${NFS_DATA}" "${HF_HOME}"

# Namespacing stops runs colliding but on its own it turns one reused directory
# into an unbounded set of abandoned ones, so reap what earlier kills left.  A
# day is far outside AIC_ACCURACY_TIME (2h default), so this cannot touch a live
# run's pool.  Report rather than swallow: a pool we cannot remove is why the
# next run finds /tmp full.
_reaped="$(find /tmp -maxdepth 1 -name 'aic-accuracy.*' -type d -mtime +0 2>/dev/null | wc -l)"
if [ "${_reaped}" -gt 0 ]; then
    echo "[accuracy-test] reaping ${_reaped} abandoned accuracy pool(s) from /tmp"
    find /tmp -maxdepth 1 -name 'aic-accuracy.*' -type d -mtime +0 \
        -exec rm -rf {} + 2>&1 | sed 's/^/[accuracy-test]   reap: /' || true
fi

compose() { docker compose -f "${AIC_DAY_DIR}/docker/docker-compose.yml" "$@"; }

# --- teardown ----------------------------------------------------------------
# vLLM shares lmcache's PID ns (for cross-container HIP IPC), which blocks docker
# from reaping vLLM's EngineCore children -> compose down/docker rm hang.  Force-
# kill the stack first, then tear down.  Load-bearing; see cmd_tiny_test.
_teardown() {
    local tag="$1" svc c
    for svc in vllm lmcache; do
        # 'timeout' needs a real executable, and compose() is a shell function --
        # 'timeout 30 compose ...' fails with "No such file or directory" and
        # captures nothing.  Invoke docker compose directly instead: these logs
        # are the only post-mortem for a failed arm, so silently empty files are
        # worse than useless.
        timeout 30 docker compose -f "${AIC_DAY_DIR}/docker/docker-compose.yml" \
            logs --no-color --no-log-prefix "$svc" \
            > "${AIC_LOG_DIR}/accuracy-${tag}-${svc}.log" 2>&1 || true
    done
    pkill -9 -f 'vllm.entrypoints.openai' 2>/dev/null || true
    pkill -9 -f 'EngineCore'              2>/dev/null || true
    pkill -9 -f 'lmcache server'          2>/dev/null || true
    sleep 2
    timeout 60 compose --profile cache down --remove-orphans --timeout 5 >/dev/null 2>&1 || true
    for c in aic-vllm-gpu0 aic-lmcache; do timeout 30 docker rm -f "$c" >/dev/null 2>&1 || true; done
}
# shellcheck disable=SC2317  # reached via the EXIT trap below, not inline
cleanup() {
    _teardown final
    # Best-effort only: this trap does not run on SIGKILL, which is why the pool
    # is namespaced and both pre-cleared and reaped above.
    rm -rf "${AIC_ACCURACY_POOL_ROOT}" 2>/dev/null || true
}
trap cleanup EXIT

# --- helpers -----------------------------------------------------------------
# Readiness is probed from inside the container: vLLM publishes no ports, so the
# Slurm host's loopback cannot reach it (the convention used across the repo).
_wait_ready() {
    local tag="$1"
    for _ in $(seq 1 "${AIC_ACCURACY_READY_TIMEOUT}"); do
        if docker exec aic-vllm-gpu0 curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
            echo "[accuracy-test] ${tag}: endpoint ready"
            return 0
        fi
        sleep 5
    done
    echo "[accuracy-test] FAIL: ${tag}: vLLM never became ready" >&2
    compose logs --tail 80 --no-color vllm 2>&1 | sed 's/^/  [vllm] /'
    return 1
}

# pytest is host-side but vLLM lives on the aic bridge, so hand pytest the
# container's bridge IP.  Verified host-routable before this was relied on; if a
# node ever fails this check the fallback is to run pytest inside the client
# container (already on aic, repo mounted), at the cost of a gsm8k download on
# the compute node.
_endpoint_url() {
    local ip
    ip="$(docker inspect -f '{{(index .NetworkSettings.Networks "aic").IPAddress}}' aic-vllm-gpu0 2>/dev/null)"
    [ -n "${ip}" ] || { echo "[accuracy-test] FAIL: no aic bridge IP for aic-vllm-gpu0" >&2; return 1; }
    curl -fsS --max-time 10 "http://${ip}:8000/v1/models" >/dev/null 2>&1 || {
        echo "[accuracy-test] FAIL: host cannot reach ${ip}:8000 on the aic bridge" >&2
        echo "[accuracy-test]       (fallback: run pytest inside the client container)" >&2
        return 1; }
    echo "http://${ip}:8000/v1"
}

# Apparent size: the sum of file LENGTHS.  The NIXL POSIX adapter carves the
# pool into fixed LMCACHE_NIXL_POSIX_SLOT_SIZE slots, so a slot's length is that
# size no matter how much KV it holds -- this number measures slot allocation,
# not KV volume.  Keep it for liveness; use the L2 counters below for volume.
_pool_bytes() { du -sb "${NVME_DATA}" 2>/dev/null | cut -f1 || echo 0; }
# Allocated size: blocks actually on disk.  Differs from the apparent size when
# slots are preallocated sparse (seek+truncate), which would let a pool that was
# never written to look like it grew.
_pool_alloc_bytes() { du -s --block-size=1 "${NVME_DATA}" 2>/dev/null | cut -f1 || echo 0; }

# Sum one Prometheus counter across all its label sets, from a container-local
# scrape.  Neither endpoint is published to the host, so scrape via docker exec.
# Prints 0 when the metric or the endpoint is absent -- callers distinguish
# "counter absent" from "counter zero" by checking the scrape succeeded first.
_metric_sum() {
    local container="$1" url="$2" metric="$3"
    docker exec "${container}" curl -fsS --max-time 15 "${url}" 2>/dev/null \
        | awk -v m="${metric}" '
            /^#/ { next }
            {
                name = $1; sub(/\{.*/, "", name)
                if (name == m) { s += $NF }
            }
            END { printf "%.0f", s + 0 }' \
        || echo 0
}

# vLLM's own view of the connector: how many blocks it asked the external tier
# for, and how many it got.  These are the numbers that decide whether a
# re-score was actually served from the tier or silently recomputed.
_vllm_ext() { _metric_sum aic-vllm-gpu0 http://127.0.0.1:8000/metrics "$1"; }
# LMCache's own view, one tier down: which level answered.
_lmc() { _metric_sum aic-lmcache http://127.0.0.1:8080/metrics "$1"; }

# ===========================================================================
# Phase 1: vram_only arm -- plain vLLM, KV never leaves VRAM
# ===========================================================================
# This arm is not optional.  It used to be skippable to halve the bringups on a
# PR path, but skipping it skipped test_tiered_matches_baseline -- the
# differential the whole gate exists to run -- and left a run that could only
# fail on catastrophe.
echo "[accuracy-test] === Phase 1: vram_only arm ==="
# No LMCache: vllm owns its IPC ns, no KV_TRANSFER_ARG, no --profile cache.
export VLLM_IPC_MODE=shareable
export VLLM_PID_MODE=
export KV_TRANSFER_ARG=
export AIC_L2_BACKEND=none
export VLM_GPU_MEMORY_UTILIZATION=0.90
if ! compose up -d vllm; then
    echo "[accuracy-test] FAIL: baseline compose up failed" >&2
    compose logs --tail 60 --no-color vllm 2>&1 | sed 's/^/  [vllm] /'
    exit 1
fi
_wait_ready vram_only || exit 1
BASE_URL="$(_endpoint_url)" || exit 1
echo "[accuracy-test] scoring baseline at ${BASE_URL}"
"${PYBIN}" "${AIC_DAY_DIR}/tests/accuracy/score_endpoint.py" "${BASE_URL}" \
    --out "${AIC_LOG_DIR}/baseline-score.json" || {
    echo "[accuracy-test] FAIL: baseline scoring failed" >&2; exit 1; }
BASELINE_SCORE="$("${PYBIN}" -c "import json;print(json.load(open('${AIC_LOG_DIR}/baseline-score.json'))['score'])")"
echo "[accuracy-test] baseline score: ${BASELINE_SCORE}"
_teardown vram_only
sleep 5

# ===========================================================================
# Phase 2: kvd arm -- LMCache + NIXL POSIX NVMe L2
# ===========================================================================
echo "[accuracy-test] === Phase 2: kvd arm (LMCache + NIXL POSIX L2) ==="
# A constrained VRAM budget is what makes this test non-vacuous: without eviction
# pressure the KV never reaches L2 and the differential compares two VRAM runs.
# Phase 3 asserts the eviction actually happened.
export VLM_GPU_MEMORY_UTILIZATION=0.15
export LMCACHE_L1_SIZE_GB=1
# NIXL POSIX: no GPUDirect/hipFile requirement, so this runs on any node.
export AIC_L2_BACKEND=nixl_posix
export AIC_NIXL_BACKEND=POSIX
export VLLM_IPC_MODE=service:lmcache
export VLLM_PID_MODE=service:lmcache
# Sharing lmcache's PID/IPC namespaces does not share its network namespace;
# resolve the peer over the compose network rather than vLLM's own loopback.
export KV_TRANSFER_ARG="--kv-transfer-config '{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://aic-lmcache\",\"lmcache.mp.port\":6555}}'"

POOL_BEFORE="$(_pool_bytes)"
POOL_ALLOC_BEFORE="$(_pool_alloc_bytes)"
if ! compose --profile cache up -d; then
    echo "[accuracy-test] FAIL: tiered compose up failed" >&2
    compose logs --tail 60 --no-color lmcache 2>&1 | sed 's/^/  [lmcache] /'
    compose logs --tail 60 --no-color vllm    2>&1 | sed 's/^/  [vllm]    /'
    exit 1
fi
_wait_ready kvd || exit 1
TIERED_URL="$(_endpoint_url)" || exit 1
echo "[accuracy-test] tiered endpoint: ${TIERED_URL}"

# Phase 3 baseline for the L2 counters.  This container is fresh, so these
# should all read zero -- sample anyway and difference, so a reused container
# (or a scrape that picks up a warm-up store) cannot inflate the phase-3 deltas.
L2_STORE_SUB_REQ_BEFORE="$(_lmc lmcache_mp_l2_store_submitted_requests_total)"
L2_STORE_DONE_REQ_BEFORE="$(_lmc lmcache_mp_l2_store_completed_requests_total)"
L2_STORE_DONE_CHUNKS_BEFORE="$(_lmc lmcache_mp_l2_store_completed_objects_chunks_total)"
echo "[accuracy-test] pre-score L2 counters: store_sub_req=${L2_STORE_SUB_REQ_BEFORE} store_done_req=${L2_STORE_DONE_REQ_BEFORE} store_done_chunks=${L2_STORE_DONE_CHUNKS_BEFORE}"

export AIC_ACCURACY_TIERED_URL="${TIERED_URL}"
export AIC_ACCURACY_SCORE_OUT="${AIC_LOG_DIR}/tiered-score.json"
export AIC_ACCURACY_BASELINE_SCORE="${BASELINE_SCORE}"

rc=0
"${PYTEST}" "${AIC_DAY_DIR}/tests/accuracy" -v -rs --no-header || rc=$?

# ===========================================================================
# Phase 3: liveness -- prove the tiered arm actually tiered
# ===========================================================================
# Two independent views, for the same reason phase 5 uses two:
#
#   * The filesystem says slots were allocated under the pool.  It cannot say
#     how much KV is in them -- the NIXL POSIX adapter allocates fixed
#     LMCACHE_NIXL_POSIX_SLOT_SIZE slots, so file length tracks slot count, not
#     bytes stored.  (Job 6235's "15,489,564,672 B of growth" is ~923 x 16 MiB
#     slots; the KV in them is bounded by 923 x LMCACHE_CHUNK_SIZE x per-token
#     KV bytes, which for a 0.5B model at bf16 is several times smaller.)
#   * LMCache's L2 counters say how many chunks it actually wrote through.
#     They cannot say the bytes reached this filesystem.
#
# The counters also close a hole the filesystem view cannot see: a pool where
# every store was submitted and then FAILED still has its slots allocated, so
# the old growth-only assertion would pass it.
echo "[accuracy-test] === Phase 3: NVMe pool liveness ==="
POOL_AFTER="$(_pool_bytes)"
POOL_ALLOC_AFTER="$(_pool_alloc_bytes)"
POOL_FILES="$(find "${NVME_DATA}" -type f -size +0c 2>/dev/null | wc -l)"
POOL_GROWTH=$(( POOL_AFTER - POOL_BEFORE ))
POOL_ALLOC_GROWTH=$(( POOL_ALLOC_AFTER - POOL_ALLOC_BEFORE ))

L2_STORE_SUB_REQ=$(( $(_lmc lmcache_mp_l2_store_submitted_requests_total) - L2_STORE_SUB_REQ_BEFORE ))
L2_STORE_DONE_REQ=$(( $(_lmc lmcache_mp_l2_store_completed_requests_total) - L2_STORE_DONE_REQ_BEFORE ))
L2_STORE_DONE_CHUNKS=$(( $(_lmc lmcache_mp_l2_store_completed_objects_chunks_total) - L2_STORE_DONE_CHUNKS_BEFORE ))
# Gauge, not a counter: LMCache's own view of L2 occupancy right now.  Read
# absolute, not differenced.
L2_USAGE_BYTES="$(_lmc lmcache_mp_l2_usage_bytes)"

echo "[accuracy-test] pool apparent: before=${POOL_BEFORE} after=${POOL_AFTER} growth=${POOL_GROWTH}"
echo "[accuracy-test] pool allocated: before=${POOL_ALLOC_BEFORE} after=${POOL_ALLOC_AFTER} growth=${POOL_ALLOC_GROWTH}"
echo "[accuracy-test] pool non-empty-files=${POOL_FILES}"
echo "[accuracy-test] lmcache L2: store_req submitted=${L2_STORE_SUB_REQ} completed=${L2_STORE_DONE_REQ} chunks=${L2_STORE_DONE_CHUNKS} usage_bytes=${L2_USAGE_BYTES}"

if [ "${POOL_FILES}" -eq 0 ] || [ "${POOL_GROWTH}" -le 0 ]; then
    echo "[accuracy-test] FAIL Phase 3: the NVMe pool did not grow -- KV never reached L2." >&2
    echo "[accuracy-test]   before=${POOL_BEFORE} after=${POOL_AFTER} growth=${POOL_GROWTH} files=${POOL_FILES}" >&2
    echo "[accuracy-test]   The differential compared two VRAM-only runs, so a pass here" >&2
    echo "[accuracy-test]   would be vacuous.  Lower VLM_GPU_MEMORY_UTILIZATION or" >&2
    echo "[accuracy-test]   LMCACHE_L1_SIZE_GB to force eviction." >&2
    rc=1
elif [ "${L2_STORE_DONE_CHUNKS}" -le 0 ]; then
    echo "[accuracy-test] FAIL Phase 3: slots were allocated but LMCache completed zero L2 chunk stores." >&2
    echo "[accuracy-test]   submitted_requests=${L2_STORE_SUB_REQ} completed_requests=${L2_STORE_DONE_REQ}" >&2
    echo "[accuracy-test]   The ${POOL_GROWTH} bytes of pool growth are slot-file allocation," >&2
    echo "[accuracy-test]   not stored KV.  Either every store failed, or the adapter does not" >&2
    echo "[accuracy-test]   report this counter -- check the lmcache log before adjusting." >&2
    rc=1
elif [ "${L2_STORE_SUB_REQ}" -gt 0 ] && [ "${L2_STORE_DONE_REQ}" -le 0 ]; then
    echo "[accuracy-test] FAIL Phase 3: ${L2_STORE_SUB_REQ} L2 store requests submitted, zero completed." >&2
    echo "[accuracy-test]   Writes are being issued and dropped on the floor." >&2
    rc=1
else
    echo "[accuracy-test] OK Phase 3: ${L2_STORE_DONE_CHUNKS} chunks stored to L2 across ${POOL_FILES} slot files"
    echo "[accuracy-test]   (usage_bytes=${L2_USAGE_BYTES}, slot growth=${POOL_GROWTH} B)"
fi

# Sparse-slot warning.  du -sb reports file LENGTH; if the adapter preallocates
# slots with seek+truncate, an unwritten pool still shows growth.  A large gap
# between apparent and allocated means the growth number above is mostly holes.
# Not fatal: filesystems differ, and the completed-chunks assertion above is the
# real gate.  Threshold is a tenth -- deliberately loose, since slot padding
# alone puts the honest ratio well under 1.
if [ "${POOL_GROWTH}" -gt 0 ] && [ "${POOL_ALLOC_GROWTH}" -lt $(( POOL_GROWTH / 10 )) ]; then
    echo "[accuracy-test] WARN Phase 3: pool slots look sparse -- allocated ${POOL_ALLOC_GROWTH} B vs apparent ${POOL_GROWTH} B."
    echo "[accuracy-test]   Treat the apparent-growth figure as slot reservation, not stored bytes."
fi

if [ "${rc}" != "0" ]; then
    echo "[accuracy-test] FAIL: phases 1-3 did not pass; skipping the restart phase" >&2
    exit "${rc}"
fi

# ===========================================================================
# Phase 4: restart vLLM only -- LMCache keeps DRAM/NVMe state
# ===========================================================================
echo "[accuracy-test] === Phase 4: restart vLLM, re-score from the tier ==="
TIERED_SCORE="$("${PYBIN}" -c "import json;print(json.load(open('${AIC_LOG_DIR}/tiered-score.json'))['score'])" 2>/dev/null)" || {
    echo "[accuracy-test] FAIL: no tiered score recorded in phase 2" >&2; exit 1; }

pkill -9 -f 'vllm.entrypoints.openai' 2>/dev/null || true
pkill -9 -f 'EngineCore'              2>/dev/null || true
sleep 3
if ! compose restart vllm; then
    echo "[accuracy-test] FAIL: compose restart vllm failed" >&2; exit 1
fi
_wait_ready kvd-restarted || exit 1
TIERED_URL="$(_endpoint_url)" || exit 1

# Counters are read AFTER the restart: vLLM's process is new, so its
# vllm:external_prefix_cache_* counters start at zero and everything they
# accumulate from here belongs to the re-score.  LMCache was NOT restarted, so
# its counters carry phase-2 history and must be differenced.
EXT_Q_BEFORE="$(_vllm_ext vllm:external_prefix_cache_queries_total)"
EXT_H_BEFORE="$(_vllm_ext vllm:external_prefix_cache_hits_total)"
L1_READ_BEFORE="$(_lmc lmcache_mp_l1_read_chunks_total)"
L2_HIT_BEFORE="$(_lmc lmcache_mp_l2_prefetch_hit_chunks_total)"
L2_LOAD_BEFORE="$(_lmc lmcache_mp_l2_prefetch_load_completed_chunks_total)"
echo "[accuracy-test] pre-rescore counters: ext_q=${EXT_Q_BEFORE} ext_h=${EXT_H_BEFORE} l1_read=${L1_READ_BEFORE} l2_hit=${L2_HIT_BEFORE} l2_load=${L2_LOAD_BEFORE}"

# Same prompts against a warm tier, so the blocks SHOULD come back from
# DRAM/NVMe rather than being recomputed.  Phase 5 checks whether they actually
# did -- the score alone cannot tell the two apart, since a full recompute
# produces the same answers.
#
# Full split, matching phase 2.  This used to re-score a 200-item prefix, which
# meant comparing a 200-item number against a 1319-item reference: the tolerance
# had to widen to +/-0.085 to cover the sampling noise, four times looser than
# the gate it was meant to be.  Re-scoring everything costs ~7 min against a
# warm cache and buys back the +/-0.02 comparison.
export AIC_ACCURACY_TIERED_URL="${TIERED_URL}"
export AIC_ACCURACY_REFERENCE_SCORE="${TIERED_SCORE}"
unset AIC_ACCURACY_BASELINE_SCORE AIC_ACCURACY_SCORE_OUT
"${PYTEST}" "${AIC_DAY_DIR}/tests/accuracy" -v -rs --no-header \
    -k 'restart' || {
    echo "[accuracy-test] FAIL: phase 4 -- score regressed after restart" >&2
    exit 1; }

# ===========================================================================
# Phase 5: prove the re-score was SERVED, not recomputed
# ===========================================================================
# Without this, phase 4 is nearly vacuous.  If the restart lost the cache
# entirely, vLLM would recompute every prompt from scratch and produce the very
# same answers -- phase 4 passes, and the "survives restart" claim is untested.
# Phase 3 does not cover it either: that shows KV was WRITTEN to the pool during
# phase 2, not that anything was READ BACK afterwards.
#
# Two independent views must agree, since either alone has a failure mode:
#   * vLLM's connector counters say the engine asked the tier and got blocks
#     back -- but not which tier answered.
#   * LMCache's counters say which level served them -- but are cumulative
#     across phase 2, hence the differencing.
echo "[accuracy-test] === Phase 5: verify the re-score hit the cache ==="
EXT_Q_AFTER="$(_vllm_ext vllm:external_prefix_cache_queries_total)"
EXT_H_AFTER="$(_vllm_ext vllm:external_prefix_cache_hits_total)"
L1_READ_AFTER="$(_lmc lmcache_mp_l1_read_chunks_total)"
L2_HIT_AFTER="$(_lmc lmcache_mp_l2_prefetch_hit_chunks_total)"
L2_LOAD_AFTER="$(_lmc lmcache_mp_l2_prefetch_load_completed_chunks_total)"

EXT_Q=$(( EXT_Q_AFTER - EXT_Q_BEFORE ))
EXT_H=$(( EXT_H_AFTER - EXT_H_BEFORE ))
L1_READ=$(( L1_READ_AFTER - L1_READ_BEFORE ))
L2_HIT=$(( L2_HIT_AFTER - L2_HIT_BEFORE ))
L2_LOAD=$(( L2_LOAD_AFTER - L2_LOAD_BEFORE ))
echo "[accuracy-test] re-score deltas: ext_q=${EXT_Q} ext_h=${EXT_H}"
echo "[accuracy-test]   lmcache: l1_read_chunks=${L1_READ} l2_hit_chunks=${L2_HIT} l2_load_chunks=${L2_LOAD}"

hit_rc=0
if [ "${EXT_Q}" -le 0 ]; then
    echo "[accuracy-test] FAIL Phase 5: vLLM never queried the external tier." >&2
    echo "[accuracy-test]   The connector is not wired up post-restart; the re-score" >&2
    echo "[accuracy-test]   recomputed everything and phase 4 proved nothing." >&2
    hit_rc=1
elif [ "${EXT_H}" -le 0 ]; then
    echo "[accuracy-test] FAIL Phase 5: ${EXT_Q} external queries, zero hits." >&2
    echo "[accuracy-test]   LMCache did not survive the restart with usable state," >&2
    echo "[accuracy-test]   so phase 4's matching score is a recompute, not retrieval." >&2
    hit_rc=1
else
    echo "[accuracy-test] OK Phase 5: ${EXT_H}/${EXT_Q} external prefix-cache hits"
fi

# Which tier answered.  L1 (DRAM) survives a vLLM restart on its own, so DRAM
# hits alone would leave NVMe retrieval -- the thing this gate exists to check
# -- still untested, and the run would be as vacuous as the cases phases 3 and 5
# already fail on.
#
# This is a hard failure.  It used to be a warning on the grounds that the tier
# split "depends on the model and the item cap", but there is no item cap any
# more -- every pass scores the full split -- so the only remaining variable is
# the model/L1 pairing, which CI pins.  AIC_ACCURACY_ALLOW_L1_ONLY=1 downgrades
# it for a configuration that is legitimately DRAM-heavy; the counters are
# logged either way, so a drift shows up in the job output regardless.
if [ "${hit_rc}" = "0" ] && [ "${L2_HIT}" -le 0 ] && [ "${L2_LOAD}" -le 0 ]; then
    if [ "${AIC_ACCURACY_ALLOW_L1_ONLY:-0}" = "1" ]; then
        echo "[accuracy-test] WARN Phase 5: every hit came from L1 (DRAM); no L2/NVMe reads." >&2
        echo "[accuracy-test]   NVMe retrieval is NOT exercised by this run, but" >&2
        echo "[accuracy-test]   AIC_ACCURACY_ALLOW_L1_ONLY=1 so this is not fatal." >&2
    else
        echo "[accuracy-test] FAIL Phase 5: every hit came from L1 (DRAM); no L2/NVMe reads." >&2
        echo "[accuracy-test]   ${EXT_H}/${EXT_Q} external hits were all served from DRAM, so" >&2
        echo "[accuracy-test]   NVMe retrieval -- the thing this gate exists to check -- was" >&2
        echo "[accuracy-test]   never exercised.  Shrink LMCACHE_L1_SIZE_GB (keep it 4096-" >&2
        echo "[accuracy-test]   aligned; a bad value surfaces as NIXL_ERR_NOT_FOUND) to push" >&2
        echo "[accuracy-test]   the working set past DRAM, or set" >&2
        echo "[accuracy-test]   AIC_ACCURACY_ALLOW_L1_ONLY=1 if this split is expected." >&2
        hit_rc=1
    fi
elif [ "${hit_rc}" = "0" ]; then
    echo "[accuracy-test] OK Phase 5: L2/NVMe served ${L2_HIT} hit chunks (${L2_LOAD} loads completed)"
fi

if [ "${hit_rc}" != "0" ]; then
    exit "${hit_rc}"
fi

echo "[accuracy-test] ALL PHASES PASSED"
exit 0
