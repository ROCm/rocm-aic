#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Run `agent-kv nvme-sweep` from aic-client with read latency injected on the
# LMCache L2 device: TTFT of L2 hits against NVMe read delay and fill fraction.
#
# agent-kv changes the delay with `sudo dmsetup`.  aic-client is unprivileged,
# so shim/sudo there forwards each change to this script, which applies it on
# the host with sudo and then drops the L2 files from the page cache: the
# stack's L2 writes are buffered, and a cached read never reaches the device.
#
# Before the first run, with the stack down:
#   benchmarks/agent-kv/dm_delay_setup.sh up <L2 mount point>
# and start the stack with LMCACHE_LAZY_OFFLOAD=false: the sweep resets the GPU
# prefix cache, which drops stores that lazy offload has deferred, so nothing
# would reach L2 and every measured request would be a recompute.
#
#   nvme_delay_sweep.sh [agent-kv nvme-sweep options]
#   nvme_delay_sweep.sh --delays 0,10,50 --fill-fractions 0.5,1.0 --num-books 4
#
# Environment:
#   NVME_DATA       host directory of the L2 tier (required; make passes it)
#   GPU             index in the vLLM container name <stack>-vllm-gpu<N> (default: 0)
#   AIC_STACK       the stack to drive (scripts/aic-endpoints.sh)
#   AIC_DM_DEVICE   mapper name (default: kv-delay)
#   LOG             host logs directory, /logs in aic-client (default: <repo>/logs)
#   AIC_BENCH_ENDPOINT, AIC_BENCH_MODEL   passed to aic-bench when set

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/aic-endpoints.sh
source "${REPO_ROOT}/scripts/aic-endpoints.sh"
CLIENT="${AIC_STACK}-client"
NAME="${AIC_DM_DEVICE:-kv-delay}"
LOG_DIR="${LOG:-${REPO_ROOT}/logs}"
CTL_HOST="${LOG_DIR}/bench/agent-kv/dm-ctl"
CTL_CLIENT="/logs/bench/agent-kv/dm-ctl"

die() { echo "nvme_delay_sweep: $*" >&2; exit 1; }

[[ -n "${NVME_DATA:-}" ]] || die "set NVME_DATA to the L2 directory the stack was started with"
[[ "$(findmnt -no SOURCE -T "${NVME_DATA}")" == "/dev/mapper/${NAME}" ]] \
    || die "${NVME_DATA} is not on /dev/mapper/${NAME}: run dm_delay_setup.sh up <mount point> with the stack down"
[[ "$(docker inspect -f '{{.State.Running}}' "${CLIENT}" 2>/dev/null)" == true ]] \
    || die "${CLIENT} is not running: start the stack with 'make up'"
vllm="${AIC_STACK}-vllm-gpu${GPU:-0}"
vllm_args="$(docker inspect -f '{{json .Config.Entrypoint}} {{json .Config.Cmd}}' "${vllm}" 2>/dev/null)" \
    || die "${vllm} is not running"
if grep -Eq 'lazy_offload[\\"]*: *true' <<< "${vllm_args}"; then
    die "${vllm} runs with lazy offload: restart the stack with LMCACHE_LAZY_OFFLOAD=false"
fi

evict_l2() {
    sync -f "${NVME_DATA}"
    python3 - "${NVME_DATA}" <<'PY'
import os, sys
for root, _, files in os.walk(sys.argv[1]):
    for name in files:
        try:
            fd = os.open(os.path.join(root, name), os.O_RDONLY)
        except OSError:
            continue
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
PY
}

# "0 <sectors> delay <dev> 0 <read_ms> <dev> 0 <write_ms>": only the read delay
# of the request is taken, so the container cannot retarget the mapping.
apply_request() {
    local current want read_ms
    current="$(sudo dmsetup table "${NAME}")"
    read_ms="$(awk '{print $6}' "${CTL_HOST}/request")"
    [[ "${read_ms}" =~ ^[0-9]+$ ]] || { echo "nvme_delay_sweep: ignoring a malformed delay request" >&2; return 0; }
    want="$(awk -v r="${read_ms}" '{$6 = r; print}' <<< "${current}")"
    sudo dmsetup load "${NAME}" <<< "${want}"
    sudo dmsetup resume "${NAME}"
    printf '%s\n' "${want}" > "${CTL_HOST}/table"
    echo "${read_ms}" > "${CTL_HOST}/read-ms"
    evict_l2
}

serve_requests() {
    while :; do
        if [[ -e "${CTL_HOST}/request" ]]; then
            apply_request
            rm -f "${CTL_HOST}/request"
        elif (( $(cat "${CTL_HOST}/read-ms") > 0 )); then
            # A measuring round refills the page cache as it reads.
            evict_l2
        fi
        sleep 0.2
    done
}

mkdir -p "${CTL_HOST}"
rm -f "${CTL_HOST}/request" "${CTL_HOST}/request.tmp"
table="$(sudo dmsetup table "${NAME}")"
printf '%s\n' "${table}" > "${CTL_HOST}/table"
awk '{print $6}' <<< "${table}" > "${CTL_HOST}/read-ms"

serve_requests &
server=$!
restore() {
    kill "${server}" 2>/dev/null || true
    awk '{$6 = 0; print}' <<< "$(sudo dmsetup table "${NAME}")" | sudo dmsetup load "${NAME}"
    sudo dmsetup resume "${NAME}"
}
trap restore EXIT

client_path="$(docker exec "${CLIENT}" printenv PATH)"
env_args=(-e "AIC_DM_CTL=${CTL_CLIENT}" -e "PATH=/repo/benchmarks/agent-kv/shim:${client_path}")
for _var in AIC_BENCH_ENDPOINT AIC_BENCH_MODEL; do
    [[ -z "${!_var:-}" ]] || env_args+=(-e "${_var}=${!_var}")
done

docker exec "${env_args[@]}" "${CLIENT}" aic-bench agent-kv nvme-sweep --dm-device "${NAME}" "$@"
