#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
#SBATCH --job-name=aic-p2p
#SBATCH --partition=defq
# Two nodes on the SAME InfiniBand fabric.  The SUPERMICRO cx6x MI300X pool is
# the validated target: 8x ConnectX-7 (400 Gb, link layer InfiniBand), local
# NVMe, and a shared NFS $HOME that both nodes can read/write.  Override on the
# command line for another pool, e.g.
#   sbatch --constraint='RDMA&IB' .slurm/run-p2p.sbatch
#   sbatch --nodelist=ctr-cx65-mi300x-30,ctr-cx66-mi300x-31 .slurm/run-p2p.sbatch
# NOTE: the `rccl` partition also groups these nodes but requires a Slurm
# account this job does not have ("Invalid account or account/partition
# combination"), so stay on defq.
#SBATCH --constraint=SUPERMICRO&CX7&NVME
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
# The cx6x nodes are shared (Slurm state `mix`), so a co-tenant can collide on
# the fixed ports below AND perturb the timings.  Add --exclusive on the sbatch
# command line for numbers you intend to publish:
#   sbatch --exclusive .slurm/run-p2p.sbatch
# Discarded: the script re-redirects everything into logs/<job-id>/p2p.out as
# soon as it starts (Slurm cannot create that dir itself).
#SBATCH --output=/dev/null
#
# Two-node LMCache P2P (cross-node KV sharing) benchmark over RDMA.
#
# Topology -- one container per node, both processes inside it (this is Ivan's
# tools/lmcache-p2p-test.sh layout, which sidesteps the cross-container HIP IPC
# dance the single-node compose stack needs):
#
#      node A (batch host)                       node B (via ssh)
#   +-----------------------------+        +-----------------------------+
#   | lmcache coordinator :9300   |<-------| lmcache server  (instance   |
#   | lmcache server (node-a)     |        |   node-b) :5555 /:8080      |
#   |   :5555 zmq /:8080 http     |<==RDMA=>|  p2p transfer chan :8500   |
#   |   p2p transfer chan :8500   |        |                             |
#   | vllm serve :8000            |        | vllm serve :8000            |
#   |   LMCacheMPConnector        |        |   LMCacheMPConnector        |
#   +-----------------------------+        +-----------------------------+
#
# Both lmcache servers register with the coordinator on node A.  When node B
# misses locally it asks the coordinator, learns node A holds the chunks, and
# pulls them directly.  The pull goes through LMCache's NIXL transfer channel
# (`--p2p-transfer-engine nixl`, the default), whose NixlAgent is created with
# backends=["UCX"] -- so the bulk KV moves over IB verbs while the coordinator
# lookup and the NIXL metadata handshake stay on TCP.
#
# What this job measures, per prompt length, using TWO distinct prompts so a
# local hit can never be mistaken for a P2P hit:
#
#   cold   : prompt X -> node B, nothing cached anywhere        (full prefill)
#   warm   : prompt Y -> node A, populates node A's LMCache L1
#   p2p    : prompt Y -> node B, B must fetch Y from A          (THE number)
#   local  : prompt Y -> node A again, A serves from its own L1 (lower bound)
#
# and, around the p2p step, the delta on every mlx5 port counter on both nodes.
# That counter delta is the proof that the transfer went over RDMA and not over
# TCP -- if it stays flat while TTFT still improves, UCX fell back to Ethernet
# and the bandwidth numbers are meaningless.
#
# Prompts are generated ONCE on the batch host into logs/<job-id>/payloads/ on
# the shared NFS $HOME, and both nodes POST the byte-identical file.  That
# removes any dependence on `vllm bench serve --seed` producing the same random
# prompt on two hosts.
#
# Submit from the tree root:
#
#   sbatch .slurm/run-p2p.sbatch
#   sbatch --exclusive .slurm/run-p2p.sbatch                    # publishable
#   AIC_P2P_ISL_LIST=4096,16384 sbatch .slurm/run-p2p.sbatch    # narrow sweep
#   VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct sbatch .slurm/run-p2p.sbatch
#
# Results land in logs/<job-id>/: p2p.out (this log), results.csv, the per-role
# container logs, and the /status + /instances JSON captured at each phase.
#
set -uo pipefail

# --- Repo + log dir ----------------------------------------------------------
# Same reasoning as run-cliff.sbatch: under sbatch, ${BASH_SOURCE[0]} points at
# the node spool copy, so prefer SLURM_SUBMIT_DIR.  logs/ must be on the shared
# NFS $HOME -- node B writes its container logs there too.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LOGDIR_REL="logs/${SLURM_JOB_ID:-manual}"
LOGDIR="${REPO}/${LOGDIR_REL}"
if mkdir -p "${LOGDIR}/payloads" 2>/dev/null; then
    exec >>"${LOGDIR}/p2p.out" 2>&1
fi

log()  { printf '[p2p-job] %s\n' "$*" >&2; }
warn() { printf '[p2p-job] WARN: %s\n' "$*" >&2; }
die()  { printf '[p2p-job] ERROR: %s\n' "$*" >&2; exit 1; }

# --- Config (all overridable via environment) --------------------------------
AIC_IMAGE="${AIC_IMAGE:-rocm-aic:latest}"
# Storage roots must land on a filesystem BOTH nodes can read.  AIC_SHARED_NFS is
# a cluster-wide root that the Makefile suffixes with $USER; $HOME already is
# per-user (and is NFS here: ags-pure-mkm-01:/home_mkm), so appending $USER to it
# would give .../stebates/stebates/images.  Keep the two cases separate.
if [[ -n "${AIC_SHARED_NFS:-}" ]]; then
    _shared_root="${AIC_SHARED_NFS}/${USER}"
else
    _shared_root="${HOME}"
fi
AIC_IMAGE_DIR="${AIC_IMAGE_DIR:-${_shared_root}/images}"
CONTAINER="${AIC_P2P_CONTAINER:-aic-p2p}"

# Model.  Default to the small full-attention model the cliff already stages:
# gpt-oss is a hybrid (sliding-window) model and cannot unify its KV-cache specs
# under a KV connector, so it is NOT a valid default here.
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
HF_HOME="${HF_HOME:-${_shared_root}/hf}"
HF_TOKEN="${HF_TOKEN:-}"
# Offline by default only when the cache looks populated; otherwise let vLLM
# fetch (Qwen2.5-3B is not gated).  Both nodes share the NFS cache, so only one
# download happens in practice.
AIC_P2P_OFFLINE="${AIC_P2P_OFFLINE:-auto}"

# Ports (host network on both nodes).  All fixed -- see the --exclusive note in
# the header; preflight_ports() below fails fast on a collision.
COORDINATOR_PORT="${AIC_P2P_COORDINATOR_PORT:-9300}"
LMCACHE_ZMQ_PORT="${AIC_P2P_ZMQ_PORT:-5555}"
LMCACHE_HTTP_PORT="${AIC_P2P_HTTP_PORT:-8080}"
P2P_PORT="${AIC_P2P_PORT:-8500}"
VLLM_PORT="${AIC_P2P_VLLM_PORT:-8000}"
LMCACHE_PROM_PORT="${AIC_P2P_LMCACHE_PROM_PORT:-9400}"

# LMCache sizing.  A pure DRAM L1 keeps this experiment about the network: no
# NVMe/GDS tier, so a miss on B can only be served by the peer or recomputed.
L1_SIZE_GB="${AIC_P2P_L1_SIZE_GB:-50}"
L1_ALIGN_BYTES="${AIC_P2P_L1_ALIGN_BYTES:-65536}"
P2P_LOOKUP_TIMEOUT="${AIC_P2P_LOOKUP_TIMEOUT:-30}"
P2P_LOAD_TIMEOUT="${AIC_P2P_LOAD_TIMEOUT:-30}"
P2P_TRANSFER_ENGINE="${AIC_P2P_TRANSFER_ENGINE:-nixl}"

# UCX.  Empty UCX_NET_DEVICES = let UCX auto-select (it prefers rc_mlx5 over tcp
# when IB is up and reachable).  Pin to e.g. mlx5_0:1 for a reproducible
# single-NIC number.  UCX_LOG_LEVEL=info makes the chosen transport appear in
# the lmcache logs, which is the cheapest way to confirm rc_mlx5 vs tcp.
UCX_NET_DEVICES="${AIC_P2P_UCX_NET_DEVICES:-}"
UCX_TLS="${AIC_P2P_UCX_TLS:-}"
UCX_LOG_LEVEL="${AIC_P2P_UCX_LOG_LEVEL:-info}"

# vLLM knobs.  Mirror the validated cliff defaults for gfx942: TRITON_ATTN is
# required because the ROCm attention backends report supports_kv_connector()
# =False on vLLM 0.25, and expandable_segments MUST be off so the KV tensors can
# be exported over HIP IPC to the MP server.
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLM_GPU_MEMORY_UTILIZATION="${VLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLM_MAX_MODEL_LEN="${VLM_MAX_MODEL_LEN:-40960}"
VLM_MAX_NUM_BATCHED_TOKENS="${VLM_MAX_NUM_BATCHED_TOKENS:-4096}"
VLM_ATTENTION_BACKEND="${AIC_ATTENTION_BACKEND:-TRITON_ATTN}"
VLM_KV_CACHE_DTYPE="${AIC_KV_CACHE_DTYPE:-fp8}"
AIC_ROCM_USE_AITER="${AIC_ROCM_USE_AITER:-1}"
VLM_LOAD_FORMAT="${AIC_LOAD_FORMAT:-fastsafetensors}"
GPU="${GPU:-0}"

# Sweep: approximate prompt lengths in tokens.  Each entry costs two distinct
# prompts.  Keep the largest comfortably under VLM_MAX_MODEL_LEN.
AIC_P2P_ISL_LIST="${AIC_P2P_ISL_LIST:-4096,8192,16384,32512}"
AIC_P2P_REPEATS="${AIC_P2P_REPEATS:-3}"      # timed repeats of the p2p/local steps
AIC_VLLM_READY_TIMEOUT="${AIC_VLLM_READY_TIMEOUT:-1800}"

# Validate placement (2 nodes, shared FS, active IB, free ports, image present)
# and exit WITHOUT starting containers or loading a model.  Cheap way to confirm
# a node pair is usable before committing a multi-hour GPU allocation:
#   AIC_P2P_PREFLIGHT_ONLY=1 sbatch --time=00:10:00 .slurm/run-p2p.sbatch
AIC_P2P_PREFLIGHT_ONLY="${AIC_P2P_PREFLIGHT_ONLY:-0}"

# --- Metrics capture (multi-node) --------------------------------------------
# ONE Prometheus (on node A) scrapes EVERY allocated node.  The checked-in
# monitoring/prometheus/prometheus.yml is single-node (all targets localhost), so
# this job GENERATES its own config with <node-ip>:<port> targets and node/role
# labels -- see write_prom_config().  The exporter fleet, including the
# rdma-exporter that reports the per-port counters this benchmark lives or dies
# by, runs on every node from monitoring/docker-compose.monitoring.yml.
#
# The TSDB lands in logs/<job-id>/tsdb and is kept after teardown.  Prometheus on
# NFS is single-writer only (mmap + POSIX locks); one Prometheus for the whole
# allocation satisfies that.
AIC_MONITORING="${AIC_MONITORING:-1}"
AIC_EXPORTERS="${AIC_EXPORTERS:-1}"
AIC_METRICS_DIR="${AIC_METRICS_DIR:-${LOGDIR}/tsdb}"
AIC_PROM_PORT="${AIC_PROM_PORT:-9090}"
AIC_PROM_RETENTION="${AIC_PROM_RETENTION:-90d}"
AIC_PROM_SCRAPE_INTERVAL="${AIC_PROM_SCRAPE_INTERVAL:-1s}"   # peer fetches are short
# Fabric exporters.  These have no upstream published image; run-build-distribute.sh
# `build-exporters` builds them and saves tarballs next to the main image, which
# load_exporter_images() below picks up.
AIC_NVME_EXPORTER_IMAGE="${AIC_NVME_EXPORTER_IMAGE:-aic-nvme-exporter:latest}"
AIC_RDMA_EXPORTER_IMAGE="${AIC_RDMA_EXPORTER_IMAGE:-aic-rdma-exporter:latest}"
# Exporter ports (mirrors the batesste Ansible roles + prometheus.yml defaults).
NODE_EXPORTER_PORT="${NODE_EXPORTER_PORT:-9100}"
AMDGPU_EXPORTER_PORT="${AMDGPU_EXPORTER_PORT:-5000}"
NVME_EXPORTER_PORT="${NVME_EXPORTER_PORT:-9998}"
RDMA_EXPORTER_PORT="${RDMA_EXPORTER_PORT:-9879}"
HSA_SNOOP_PORT="${HSA_SNOOP_PORT:-9488}"
# Which exporter services to run on every node.  node-exporter and rdma-exporter
# are the two that matter here: both expose IB port counters (node-exporter via
# its infiniband collector, rdma-exporter via yuuki-style rdma_port_*_data_total),
# giving two independent readings of the same transfer.  amdgpu-exporter and
# hsa-snoop are heavier and off by default; add them with AIC_P2P_GPU_EXPORTERS=1.
AIC_P2P_GPU_EXPORTERS="${AIC_P2P_GPU_EXPORTERS:-0}"

# --- Node discovery ----------------------------------------------------------
command -v scontrol >/dev/null 2>&1 || die "scontrol not found; run this under Slurm"
mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST:-}" 2>/dev/null)
(( ${#NODES[@]} >= 2 )) || die "need 2 nodes, got ${#NODES[@]} (${SLURM_JOB_NODELIST:-unset}); submit with --nodes=2"
NODE_A="${NODES[0]}"
NODE_B="${NODES[1]}"

# The advertise IPs must be reachable from the peer.  Resolve the peer by name
# first (DNS), then ask each node which local source address it would use to
# reach that peer -- that picks the right interface even on hosts with a pile of
# docker/veth/usb addresses, which `hostname -I` does not.
_dns_ip() { getent hosts "$1" 2>/dev/null | awk 'NR==1{print $1}'; }
NODE_A_DNS="$(_dns_ip "${NODE_A}")"
NODE_B_DNS="$(_dns_ip "${NODE_B}")"
[[ -n "${NODE_A_DNS}" && -n "${NODE_B_DNS}" ]] || die "could not resolve ${NODE_A} / ${NODE_B} via DNS"

# ssh into an allocated node.  Verified to work password-less inside an
# allocation on this cluster.  BatchMode so a prompt fails instead of hanging.
#
# ControlMaster multiplexing matters for more than speed: the IB counter
# snapshots BRACKET each timed request, so per-call ssh setup latency (~0.5-1 s
# on a fresh TCP+auth handshake) would widen the counter window and fold a
# co-tenant's traffic into the delta.  A persistent control socket cuts each
# subsequent call to a round trip.  The socket dir is per-job under /tmp (local
# to the batch host) -- NOT the NFS logdir, where a unix socket cannot live.
SSH_CTL_DIR="$(mktemp -d "/tmp/aic-p2p-ssh-${SLURM_JOB_ID:-$$}-XXXX" 2>/dev/null || echo /tmp)"
sshx() {
    local host="$1"; shift
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        -o ControlMaster=auto -o "ControlPath=${SSH_CTL_DIR}/%r@%h:%p" \
        -o ControlPersist=600 "${host}" "$@"
}
on_a() { sshx "${NODE_A}" "$@"; }
on_b() { sshx "${NODE_B}" "$@"; }
on_node() { local n="$1"; shift; sshx "${n}" "$@"; }

_src_ip_to() {  # $1=node to ask, $2=peer ip it should route to
    on_node "$1" "ip -4 route get $2 2>/dev/null | awk '{for(i=1;i<=NF;i++) if (\$i==\"src\") print \$(i+1)}' | head -1"
}
NODE_A_IP="$(_src_ip_to "${NODE_A}" "${NODE_B_DNS}")"; NODE_A_IP="${NODE_A_IP:-${NODE_A_DNS}}"
NODE_B_IP="$(_src_ip_to "${NODE_B}" "${NODE_A_DNS}")"; NODE_B_IP="${NODE_B_IP:-${NODE_B_DNS}}"

# The workload uses exactly two nodes (A serves, B fetches), but monitoring
# covers the WHOLE allocation -- so a 4-node run submitted for a future
# many-peer topology still gets every node scraped without touching this code.
# ALL_NODES / ALL_IPS stay index-aligned; role is the a/b/c... suffix used as a
# Prometheus label.
ALL_NODES=("${NODES[@]}")
ALL_IPS=("${NODE_A_IP}" "${NODE_B_IP}")
for ((i=2; i<${#ALL_NODES[@]}; i++)); do
    _ip="$(_src_ip_to "${ALL_NODES[i]}" "${NODE_A_DNS}")"
    ALL_IPS+=("${_ip:-$(_dns_ip "${ALL_NODES[i]}")}")
done
_role_of() { printf '%s' "$(printf "\\$(printf '%03o' $((97 + $1)))")"; }   # 0->a, 1->b, ...

log "job ${SLURM_JOB_ID:-manual}  logs -> ${LOGDIR}"
log "node A = ${NODE_A} (${NODE_A_IP})   [coordinator lives here]"
log "node B = ${NODE_B} (${NODE_B_IP})"
(( ${#ALL_NODES[@]} > 2 )) && log "extra allocated nodes (monitored only): ${ALL_NODES[*]:2}"
log "model  = ${VLLM_MODEL}   image = ${AIC_IMAGE}"

# --- Teardown ----------------------------------------------------------------
# Runs on any exit path.  Capture logs BEFORE removing the containers, and bound
# every remote call so a wedged node cannot hang the job past its walltime.
_torn_down=0
_containers_started=0
_monitoring_up=0     # set by start_monitoring_fleet; read here, so init before the trap
teardown() {
    local rc=$?
    (( _torn_down )) && return
    _torn_down=1
    # Record what the TSDB captured before anything is stopped.
    monitoring_targets_summary 2>/dev/null || true
    stop_monitoring_fleet 2>/dev/null || true
    if (( _containers_started == 0 )); then
        log "nothing to tear down (exit ${rc})"
        return
    fi
    log "tearing down (exit ${rc})"
    local n role
    for n in "${NODE_A}" "${NODE_B}"; do
        role=$([[ "${n}" == "${NODE_A}" ]] && echo a || echo b)
        timeout 60 sshx "${n}" "docker logs ${CONTAINER} >/dev/null 2>&1" || true
        timeout 120 sshx "${n}" "docker rm -f ${CONTAINER} >/dev/null 2>&1" || true
        log "  ${n}: container removed"
    done
    log "results: ${LOGDIR}/results.csv"
    log "log:     ${LOGDIR}/p2p.out"
    # Close the multiplexed ssh masters and drop the socket dir.
    for n in "${ALL_NODES[@]:-}"; do
        [[ -n "${n}" ]] && ssh -O exit -o "ControlPath=${SSH_CTL_DIR}/%r@%h:%p" "${n}" 2>/dev/null || true
    done
    [[ "${SSH_CTL_DIR}" == /tmp/aic-p2p-ssh-* ]] && rm -rf "${SSH_CTL_DIR}" 2>/dev/null || true
}
trap teardown EXIT INT TERM

# --- Preflight ---------------------------------------------------------------
# Sum of a per-port IB counter across every mlx5 device, in BYTES.  The sysfs
# port_{rcv,xmit}_data counters are in 4-byte lanes -- multiply by 4.  Defined
# here (rather than with the other measurement helpers) so preflight can print
# the baseline values and prove the read path works before anything is launched.
ib_bytes() {  # $1=node $2=rcv|xmit
    local n="$1" which="$2"
    on_node "${n}" "cat /sys/class/infiniband/*/ports/*/counters/port_${which}_data 2>/dev/null \
        | awk '{s+=\$1} END{printf \"%.0f\", s*4}'"
}

preflight_ports() {
    local n p busy=0
    for n in "${NODE_A}" "${NODE_B}"; do
        for p in "${LMCACHE_ZMQ_PORT}" "${LMCACHE_HTTP_PORT}" "${P2P_PORT}" "${VLLM_PORT}" "${LMCACHE_PROM_PORT}"; do
            if on_node "${n}" "ss -ltn 2>/dev/null | awk '{print \$4}' | grep -qE '[:.]${p}\$'"; then
                warn "${n}: port ${p} already in use (co-tenant?); resubmit with --exclusive or override the port"
                busy=1
            fi
        done
    done
    on_node "${NODE_A}" "ss -ltn 2>/dev/null | awk '{print \$4}' | grep -qE '[:.]${COORDINATOR_PORT}\$'" \
        && { warn "${NODE_A}: coordinator port ${COORDINATOR_PORT} in use"; busy=1; }
    (( busy == 0 )) || die "port conflict; refusing to produce numbers that are not ours"
}

preflight_ib() {
    local n devs act
    for n in "${NODE_A}" "${NODE_B}"; do
        devs="$(on_node "${n}" "ls /sys/class/infiniband 2>/dev/null | tr '\n' ' '")"
        [[ -n "${devs}" ]] || die "${n}: no /sys/class/infiniband devices -- this node has no RDMA NIC"
        act="$(on_node "${n}" "grep -l ACTIVE /sys/class/infiniband/*/ports/*/state 2>/dev/null | wc -l")"
        log "  ${n}: ib devices [${devs}], ${act} active port(s)"
        (( act > 0 )) || die "${n}: no ACTIVE IB port"
        on_node "${n}" "test -e /dev/infiniband/uverbs0" \
            || die "${n}: /dev/infiniband/uverbs0 missing -- cannot pass verbs into the container"
        # Also exercises the counter read path the measurement depends on.
        log "  ${n}: IB counters baseline rx=$(ib_bytes "${n}" rcv) B tx=$(ib_bytes "${n}" xmit) B"
    done
}

preflight_shared_fs() {
    local marker="${LOGDIR}/.shared-fs-probe"
    date +%s > "${marker}"
    on_b "test -r '${marker}'" \
        || die "${LOGDIR} is not visible on ${NODE_B}; logs/ must live on the shared NFS \$HOME"
    rm -f "${marker}"
    log "  shared fs OK: ${LOGDIR} readable on both nodes"
}

log "preflight"
preflight_shared_fs
preflight_ib
preflight_ports

# --- Image load --------------------------------------------------------------
# Mirrors load_image_if_needed() in run-cliff.sbatch, but fanned out over ssh.
# Reload when absent or when the shared tarball is newer than the marker.  Used
# for the AIC image (required) and for the fabric exporter images (optional --
# run-build-distribute.sh `build-exporters` writes their tarballs alongside it).
_tarball_glob() {  # $1=image:tag -> the shared-dir glob its tarball matches
    printf '%s/%s-*.tar.*' "${AIC_IMAGE_DIR}" "$(printf '%s' "$1" | tr '/:' '--')"
}
_glob="$(_tarball_glob "${AIC_IMAGE}")"
load_image_on() {  # $1=node $2=image (default AIC_IMAGE)
    local n="$1" image="${2:-${AIC_IMAGE}}" glob
    glob="$(_tarball_glob "${image}")"
    on_node "${n}" "bash -s" <<EOF
set -uo pipefail
tb="\$(ls -t ${glob} 2>/dev/null | head -1 || true)"
marker="/var/tmp/aic-loaded-\$(id -u)-\$(printf '%s' '${image}' | tr '/:' '__').mtime"
have="\$(docker images -q '${image}' 2>/dev/null)"
tarm="\$(stat -c %Y "\${tb}" 2>/dev/null || echo 0)"
loadedm="\$(cat "\${marker}" 2>/dev/null || echo 0)"
if [ "${AIC_FORCE_LOAD:-0}" = "1" ] || [ -z "\${have}" ] || [ "\${tarm}" -gt "\${loadedm}" ]; then
    if [ -z "\${tb}" ]; then
        [ -n "\${have}" ] && { echo "image present, no tarball"; exit 0; }
        echo "ERROR: ${image} absent and no tarball matching ${glob}" >&2; exit 1
    fi
    echo "loading \${tb}"
    case "\${tb}" in
        *.zst) zstd -dc "\${tb}" | docker load ;;
        *.gz)  gzip  -dc "\${tb}" | docker load ;;
        *)     docker load -i "\${tb}" ;;
    esac || exit 1
    echo "\${tarm}" > "\${marker}" 2>/dev/null || true
else
    echo "image up to date (\${have})"
fi
EOF
}
# --- Multi-node metrics capture ----------------------------------------------
MON_COMPOSE="${REPO}/monitoring/docker-compose.monitoring.yml"
PROM_CONFIG="${LOGDIR}/prometheus/prometheus.yml"

# Generate the scrape config across every allocated node.  One static_configs
# entry per node per job, each carrying node/role labels, so a PromQL query can
# say `rdma_port_xmit_data_total{role="a"}` and mean "the serving node".
write_prom_config() {
    mkdir -p "$(dirname "${PROM_CONFIG}")" "${AIC_METRICS_DIR}" 2>/dev/null || return 1
    local spec="" i
    # job:port pairs.  vllm/lmcache are the workload; the rest are the fleet.
    local -a jobs=(
        "vllm:${VLLM_PORT}"
        "lmcache:${LMCACHE_PROM_PORT}"
        "node_exporter:${NODE_EXPORTER_PORT}"
        "rdma_exporter:${RDMA_EXPORTER_PORT}"
        "nvme_exporter:${NVME_EXPORTER_PORT}"
    )
    if [[ "${AIC_P2P_GPU_EXPORTERS}" == "1" ]]; then
        jobs+=("amd_metrics_exporter:${AMDGPU_EXPORTER_PORT}" "hsa_snoop:${HSA_SNOOP_PORT}")
    fi
    for ((i=0; i<${#ALL_NODES[@]}; i++)); do
        spec+="${ALL_NODES[i]}|${ALL_IPS[i]}|$(_role_of "${i}") "
    done
    SPEC="${spec}" JOBS="${jobs[*]}" \
    PROM_SELF="${AIC_PROM_PORT}" SCRAPE="${AIC_PROM_SCRAPE_INTERVAL}" \
    python3 - "${PROM_CONFIG}" <<'PY'
import os, sys
nodes = [s.split('|') for s in os.environ['SPEC'].split()]
jobs  = [j.split(':') for j in os.environ['JOBS'].split()]
out = [
    "# GENERATED by .slurm/run-p2p.sbatch -- do not edit; regenerated per job.",
    "# Multi-node: one Prometheus scrapes every allocated node by IP (the",
    "# checked-in monitoring/prometheus/prometheus.yml is localhost-only).",
    "global:",
    f"  scrape_interval: {os.environ['SCRAPE']}",
    "  evaluation_interval: 15s",
    "  external_labels:",
    "    cluster: AIC",
    "    job_kind: p2p",
    "",
    "rule_files:",
    "  - /etc/prometheus/rules/aic.yml",
    "",
    "scrape_configs:",
    "  - job_name: prometheus",
    "    static_configs:",
    f"      - targets: [\"localhost:{os.environ['PROM_SELF']}\"]",
]
for name, port in jobs:
    out += ["", f"  - job_name: {name}", "    metrics_path: /metrics",
            "    static_configs:"]
    for host, ip, role in nodes:
        out += [f"      - targets: [\"{ip}:{port}\"]",
                "        labels:",
                f"          node: \"{host}\"",
                f"          role: \"{role}\""]
open(sys.argv[1], "w").write("\n".join(out) + "\n")
print(f"{len(jobs)} jobs x {len(nodes)} nodes = {len(jobs)*len(nodes)} targets")
PY
}

# The exporter fleet on EVERY node.  Services are named explicitly rather than
# selected purely by profile: `up -d` with no service list would also start the
# prometheus service on every node, and we want exactly one (on node A).
mon_services() {
    local -a svcs=(node-exporter rdma-exporter nvme-exporter)
    [[ "${AIC_P2P_GPU_EXPORTERS}" == "1" ]] && svcs+=(amdgpu-exporter hsa-snoop)
    printf '%s\n' "${svcs[@]}"
}
MON_CONTAINERS=(aic-prometheus aic-node-exporter aic-amdgpu-exporter
                aic-nvme-exporter aic-rdma-exporter aic-hsa-snoop)

# Compose invocation on a remote node.  AIC_METRICS_DIR must be set even where
# only exporters start: compose interpolates the `:?` on the prometheus service
# at parse time, for the whole file, before profile/service filtering.
mon_compose_on() {  # $1=node, rest = compose args
    local n="$1"; shift
    on_node "${n}" "cd '${REPO}/monitoring' && \
        AIC_METRICS_DIR='${AIC_METRICS_DIR}' \
        AIC_PROM_CONFIG='${PROM_CONFIG}' \
        AIC_PROM_RULES='${REPO}/monitoring/prometheus/rules' \
        AIC_PROM_PORT='${AIC_PROM_PORT}' \
        AIC_PROM_RETENTION='${AIC_PROM_RETENTION}' \
        PROM_UID='$(id -u)' PROM_GID='$(id -g)' \
        IMAGE_NAME='${AIC_IMAGE}' \
        AIC_NVME_EXPORTER_IMAGE='${AIC_NVME_EXPORTER_IMAGE}' \
        AIC_RDMA_EXPORTER_IMAGE='${AIC_RDMA_EXPORTER_IMAGE}' \
        HSA_SNOOP_PORT='${HSA_SNOOP_PORT}' \
        docker compose -f '${MON_COMPOSE}' $*"
}

start_monitoring_fleet() {
    [[ "${AIC_MONITORING}" == "1" ]] || { log "monitoring disabled (AIC_MONITORING=0)"; return 0; }
    [[ -f "${MON_COMPOSE}" ]] || { log "monitoring: ${MON_COMPOSE} missing, skipping"; return 0; }

    local out
    out="$(write_prom_config)" || { log "monitoring: could not write ${PROM_CONFIG}, skipping"; return 0; }
    log "prometheus config: ${LOGDIR_REL}/prometheus/prometheus.yml (${out})"

    # Fabric exporter images are optional; without them those two services fail
    # to start and their targets simply read down.  Say so rather than failing.
    local n img
    for img in "${AIC_RDMA_EXPORTER_IMAGE}" "${AIC_NVME_EXPORTER_IMAGE}"; do
        for n in "${ALL_NODES[@]}"; do
            load_image_on "${n}" "${img}" >/dev/null 2>&1 \
                || log "  WARN ${n}: ${img} unavailable (run 'make dist-build-exporters')"
        done
    done

    _monitoring_up=1   # from here teardown must stop the fleet
    if [[ "${AIC_EXPORTERS}" == "1" ]]; then
        local -a svcs; mapfile -t svcs < <(mon_services)
        # --no-build: the nvme/rdma services carry a `build:` section, and compose
        # would otherwise silently start a multi-minute Debian build (needing
        # registry egress) on every node whose image tarball was missing.  Fail
        # that service fast instead; the health check reports it.
        for n in "${ALL_NODES[@]}"; do
            mon_compose_on "${n}" --profile exporters --profile exporters-fabric \
                up -d --no-build "${svcs[@]}" >/dev/null 2>&1 \
                && log "  ${n}: exporters up (${svcs[*]})" \
                || log "  WARN ${n}: exporter compose up failed (continuing)"
        done
    fi

    # Exactly one Prometheus, on node A, scraping every node.
    mon_compose_on "${NODE_A}" up -d prometheus >/dev/null 2>&1 \
        && log "  ${NODE_A}: prometheus up on :${AIC_PROM_PORT}, TSDB -> ${AIC_METRICS_DIR}" \
        || log "  WARN ${NODE_A}: prometheus failed to start"
}

stop_monitoring_fleet() {
    (( _monitoring_up )) || return 0
    _monitoring_up=0
    log "stopping metrics capture (TSDB retained at ${AIC_METRICS_DIR})"
    local n
    for n in "${ALL_NODES[@]}"; do
        mon_compose_on "${n}" --profile exporters --profile exporters-fabric down \
            >/dev/null 2>&1 || true
        # Sweep leftovers from a crashed compose run (mirrors monitoring-lib.sh).
        timeout 60 sshx "${n}" "docker rm -f ${MON_CONTAINERS[*]} >/dev/null 2>&1" || true
    done
}

# Probe every exporter on every node and report which serve their series.  This
# is what tells you a `down` target is a missing exporter rather than a missing
# metric.
monitoring_healthcheck() {
    (( _monitoring_up )) || return 0
    local n i port name re spec cnt
    log "exporter health check:"
    for ((i=0; i<${#ALL_NODES[@]}; i++)); do
        n="${ALL_NODES[i]}"
        for spec in "node_exporter:${NODE_EXPORTER_PORT}:^node_" \
                    "rdma_exporter:${RDMA_EXPORTER_PORT}:^rdma_" \
                    "nvme_exporter:${NVME_EXPORTER_PORT}:^nvme_"; do
            name="${spec%%:*}"; port="$(cut -d: -f2 <<<"${spec}")"; re="${spec##*:}"
            cnt="$(on_node "${n}" "curl -fsS --max-time 3 http://127.0.0.1:${port}/metrics 2>/dev/null | grep -cE '${re}'" 2>/dev/null || echo 0)"
            if [[ "${cnt:-0}" -gt 0 ]]; then
                log "  OK   ${n} ${name} :${port} -- ${cnt} ${re} series"
            else
                log "  WARN ${n} ${name} :${port} -- no ${re} series"
            fi
        done
    done
}

# Ask Prometheus which targets are up, so the log records exactly what the TSDB
# does and does not contain.
monitoring_targets_summary() {
    (( _monitoring_up )) || return 0
    log "Prometheus scrape targets (up, by node):"
    on_a "curl -fsS --max-time 10 'http://127.0.0.1:${AIC_PROM_PORT}/api/v1/query?query=up'" \
        > "${LOGDIR}/prometheus-up.json" 2>/dev/null || {
            log "  WARN could not query Prometheus at :${AIC_PROM_PORT}"; return 0; }
    python3 - "${LOGDIR}/prometheus-up.json" <<'PY' 2>/dev/null || log "  WARN could not parse up query"
import json, sys
d = json.load(open(sys.argv[1]))
res = d.get("data", {}).get("result", [])
if not res:
    print("    (no up series yet)"); raise SystemExit
for r in sorted(res, key=lambda x: (x["metric"].get("node", ""), x["metric"].get("job", ""))):
    m = r["metric"]
    print("    up=%s  node=%-24s role=%-3s job=%-22s instance=%s" % (
        r["value"][1], m.get("node", "-"), m.get("role", "-"),
        m.get("job", "?"), m.get("instance", "?")))
PY
    log "  TSDB: ${AIC_METRICS_DIR} ($(du -sh "${AIC_METRICS_DIR}" 2>/dev/null | cut -f1))"
}

if [[ "${AIC_P2P_PREFLIGHT_ONLY}" == "1" ]]; then
    # Report image availability rather than loading it -- preflight must not
    # spend minutes decompressing a multi-GB tarball.
    for n in "${ALL_NODES[@]}"; do
        for _i in "${AIC_IMAGE}" "${AIC_RDMA_EXPORTER_IMAGE}" "${AIC_NVME_EXPORTER_IMAGE}"; do
            _have="$(on_node "${n}" "docker images -q '${_i}' 2>/dev/null | head -1")"
            _tb="$(on_node "${n}" "ls -t $(_tarball_glob "${_i}") 2>/dev/null | head -1")"
            if [[ -n "${_have}" ]]; then
                log "  ${n}: ${_i} present (${_have})"
            elif [[ -n "${_tb}" ]]; then
                log "  ${n}: ${_i} absent, will load ${_tb}"
            elif [[ "${_i}" == "${AIC_IMAGE}" ]]; then
                warn "${n}: ${_i} -- no image and no tarball; run 'make dist-build'"
            else
                warn "${n}: ${_i} -- no image and no tarball; run 'make dist-build-exporters'"
            fi
        done
    done
    # Generate + validate the multi-node scrape config without starting anything.
    if [[ "${AIC_MONITORING}" == "1" ]]; then
        if _out="$(write_prom_config)"; then
            log "  prometheus config OK: ${LOGDIR_REL}/prometheus/prometheus.yml (${_out})"
        else
            warn "could not write ${PROM_CONFIG}"
        fi
    fi
    log "preflight OK: ${NODE_A} + ${NODE_B} are a usable pair"
    exit 0
fi

for n in "${NODE_A}" "${NODE_B}"; do
    log "image on ${n}:"
    load_image_on "${n}" || die "image load failed on ${n} (run 'make dist-build' first)"
done

log "starting multi-node metrics capture"
start_monitoring_fleet
monitoring_healthcheck

# --- Launch the containers ---------------------------------------------------
# One long-lived container per node; the roles are started inside it with
# `docker exec -d`.  Differences from tools/lmcache-p2p-test.sh, all required
# for RDMA: /dev/infiniband is passed through, IPC_LOCK + unlimited memlock let
# UCX pin and register memory, and nofile is raised for NIXL's per-slot FDs.
mkdir -p "${LOGDIR}/roles"
docker_run_cmd() {
    cat <<EOF
docker run -d --rm --name ${CONTAINER} \
  --network host --ipc host \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 \
  -v ${HF_HOME}:/hf \
  -v ${LOGDIR}:/joblog \
  -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub -e VLLM_CACHE_ROOT=/hf/vllm \
  -e VLLM_CONFIG_ROOT=/hf/vllm_config -e TORCH_HOME=/hf/torch \
  -e HF_TOKEN=${HF_TOKEN} \
  -e HF_HUB_OFFLINE=${_offline} -e TRANSFORMERS_OFFLINE=${_offline} \
  -e ROCR_VISIBLE_DEVICES=${GPU} \
  -e AMDGCN_USE_BUFFER_OPS=0 -e MIOPEN_FIND_MODE=FAST -e MIOPEN_USER_DB_PATH=/app/miopen \
  -e VLLM_ROCM_USE_AITER=${AIC_ROCM_USE_AITER} \
  -e PYTORCH_ALLOC_CONF=expandable_segments:False \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:False \
  -e PYTHONHASHSEED=0 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/opt/nixl/lib/x86_64-linux-gnu \
  -e UCX_LOG_LEVEL=${UCX_LOG_LEVEL} \
  ${UCX_NET_DEVICES:+-e UCX_NET_DEVICES=${UCX_NET_DEVICES}} \
  ${UCX_TLS:+-e UCX_TLS=${UCX_TLS}} \
  --entrypoint sleep ${AIC_IMAGE} infinity
EOF
}

# Offline only if the model already looks staged in the shared cache.
_offline=0
if [[ "${AIC_P2P_OFFLINE}" == "1" ]]; then
    _offline=1
elif [[ "${AIC_P2P_OFFLINE}" == "auto" ]]; then
    _model_dir="models--$(printf '%s' "${VLLM_MODEL}" | tr '/' '-')"
    [[ -d "${HF_HOME}/hub/${_model_dir}" ]] && _offline=1
fi
log "HF cache ${HF_HOME} (offline=${_offline})"
mkdir -p "${HF_HOME}" 2>/dev/null || true

_containers_started=1   # from here on, teardown must clean up
for n in "${NODE_A}" "${NODE_B}"; do
    on_node "${n}" "docker rm -f ${CONTAINER} >/dev/null 2>&1" || true
    on_node "${n}" "$(docker_run_cmd)" >/dev/null \
        || die "${n}: docker run failed"
    on_node "${n}" "docker ps --filter name=${CONTAINER} --format '{{.Names}}'" | grep -q "${CONTAINER}" \
        || die "${n}: container did not start"
    log "  ${n}: ${CONTAINER} up"
done

# Confirm the container can actually see the verbs devices.  A silent failure
# here is exactly how you end up benchmarking TCP.
for n in "${NODE_A}" "${NODE_B}"; do
    _ibv="$(on_node "${n}" "docker exec ${CONTAINER} ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE" || echo 0)"
    log "  ${n}: ${_ibv} PORT_ACTIVE verbs port(s) visible inside the container"
    (( _ibv > 0 )) || die "${n}: container sees no active verbs port -- RDMA would silently fall back to TCP"
done

# --- Start the roles ---------------------------------------------------------
# Each role's command is WRITTEN TO A SCRIPT under the shared LOGDIR (visible in
# the container as /joblog) and then run with `docker exec -d bash /joblog/...`.
# Passing the command inline instead would have to survive three levels of shell
# quoting -- local -> ssh -> `docker exec bash -lc` -- and the vLLM invocation
# carries single-quoted JSON (--kv-transfer-config) that does not.  The script
# file also doubles as an artifact: logs/<job-id>/roles/<role>.sh is exactly what
# ran, reproducible by hand.
exec_role() {  # $1=node $2=role-name $3=command (may span lines)
    local n="$1" role="$2" cmd="$3"
    { echo "#!/usr/bin/env bash"; echo "set -uo pipefail"; echo "${cmd}"; } \
        > "${LOGDIR}/roles/${role}.sh"
    chmod +x "${LOGDIR}/roles/${role}.sh"
    # `bash -c`, not `bash -lc`: a login shell sources /etc/profile, which can
    # overwrite the LD_LIBRARY_PATH (ROCm + NIXL) set by `docker run -e`.
    on_node "${n}" "docker exec -d ${CONTAINER} bash -c 'bash /joblog/roles/${role}.sh > /joblog/roles/${role}.log 2>&1'" \
        || die "${n}: failed to start ${role}"
    log "  started ${role} on ${n} (cmd: ${LOGDIR_REL}/roles/${role}.sh)"
}

wait_http() {  # $1=node $2=url $3=timeout_s $4=label
    local n="$1" url="$2" timeout="${3:-60}" label="${4:-$2}" waited=0
    log "waiting for ${label} (${url}, up to ${timeout}s)"
    while (( waited < timeout )); do
        if on_node "${n}" "curl -sf -o /dev/null --max-time 5 '${url}'"; then
            log "  ${label} ready (${waited}s)"; return 0
        fi
        sleep 5; waited=$((waited + 5))
    done
    return 1
}

log "starting coordinator on ${NODE_A}"
exec_role "${NODE_A}" coordinator \
    "lmcache coordinator --host 0.0.0.0 --port ${COORDINATOR_PORT}"
wait_http "${NODE_A}" "http://${NODE_A_IP}:${COORDINATOR_PORT}/instances" 60 "coordinator" \
    || die "coordinator never came up (see ${LOGDIR}/roles/coordinator.log)"

lmcache_cmd() {  # $1=instance-id $2=advertise-ip
    cat <<EOF
lmcache server --host 0.0.0.0 --port ${LMCACHE_ZMQ_PORT} \
  --http-port ${LMCACHE_HTTP_PORT} \
  --prometheus-port ${LMCACHE_PROM_PORT} \
  --l1-size-gb ${L1_SIZE_GB} --eviction-policy LRU \
  --l1-align-bytes ${L1_ALIGN_BYTES} \
  --instance-id $1 \
  --coordinator-url http://${NODE_A_IP}:${COORDINATOR_PORT} \
  --coordinator-advertise-ip $2 \
  --p2p-advertise-url $2:${P2P_PORT} \
  --p2p-transfer-engine ${P2P_TRANSFER_ENGINE} \
  --p2p-lookup-timeout ${P2P_LOOKUP_TIMEOUT} \
  --p2p-load-timeout ${P2P_LOAD_TIMEOUT}
EOF
}
log "starting lmcache servers"
exec_role "${NODE_A}" lmcache-a "$(lmcache_cmd node-a "${NODE_A_IP}")"
exec_role "${NODE_B}" lmcache-b "$(lmcache_cmd node-b "${NODE_B_IP}")"
wait_http "${NODE_A}" "http://${NODE_A_IP}:${LMCACHE_HTTP_PORT}/status" 120 "lmcache node-a" \
    || die "lmcache node-a never came up (see ${LOGDIR}/roles/lmcache-a.log)"
wait_http "${NODE_B}" "http://${NODE_B_IP}:${LMCACHE_HTTP_PORT}/status" 120 "lmcache node-b" \
    || die "lmcache node-b never came up (see ${LOGDIR}/roles/lmcache-b.log)"

# Both instances must appear in the coordinator registry, or there is no peer to
# fetch from and every "p2p" number below would silently be a recompute.
on_a "curl -sf --max-time 10 http://${NODE_A_IP}:${COORDINATOR_PORT}/instances" \
    > "${LOGDIR}/instances.json" 2>/dev/null || true
if grep -q 'node-a' "${LOGDIR}/instances.json" 2>/dev/null && \
   grep -q 'node-b' "${LOGDIR}/instances.json" 2>/dev/null; then
    log "coordinator sees both instances (${LOGDIR}/instances.json)"
else
    warn "coordinator does NOT list both instances; P2P will not happen"
    warn "  $(cat "${LOGDIR}/instances.json" 2>/dev/null | head -c 400)"
fi

KV_TRANSFER_ARG='{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"lmcache.mp.port":'"${LMCACHE_ZMQ_PORT}"'}}'
vllm_cmd() {
    cat <<EOF
vllm serve ${VLLM_MODEL} --host 0.0.0.0 --port ${VLLM_PORT} \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --gpu-memory-utilization ${VLM_GPU_MEMORY_UTILIZATION} \
  --max-model-len ${VLM_MAX_MODEL_LEN} \
  --max-num-batched-tokens ${VLM_MAX_NUM_BATCHED_TOKENS} \
  --load-format ${VLM_LOAD_FORMAT} \
  --block-size 64 --enable-prefix-caching \
  --kv-cache-dtype ${VLM_KV_CACHE_DTYPE} \
  --attention-backend ${VLM_ATTENTION_BACKEND} \
  --disable-access-log-for-endpoints "/health,/metrics,/v1/models" \
  --kv-transfer-config '${KV_TRANSFER_ARG}'
EOF
}
log "starting vLLM on both nodes (this loads weights; can take a while)"
exec_role "${NODE_A}" vllm-a "$(vllm_cmd)"
exec_role "${NODE_B}" vllm-b "$(vllm_cmd)"
wait_http "${NODE_A}" "http://${NODE_A_IP}:${VLLM_PORT}/v1/models" "${AIC_VLLM_READY_TIMEOUT}" "vLLM node-a" \
    || die "vLLM A never became ready (see ${LOGDIR}/roles/vllm-a.log)"
wait_http "${NODE_B}" "http://${NODE_B_IP}:${VLLM_PORT}/v1/models" "${AIC_VLLM_READY_TIMEOUT}" "vLLM node-b" \
    || die "vLLM B never became ready (see ${LOGDIR}/roles/vllm-b.log)"

# --- Payload generation ------------------------------------------------------
# Deterministic prompts, generated once on the shared FS, POSTed byte-identically
# by both nodes.  That removes any dependence on two hosts independently
# producing the same "random" prompt.
#
# Per prompt length we emit:
#   x        -- the cold baseline.  Served exactly ONCE, on node A, and never
#               touched again by anything.
#   y1..yR   -- one DISTINCT prompt per timed repeat.  Each is warmed on A, then
#               fetched by B.  They must be distinct: once B has fetched y1, y1
#               sits in B's own L1 *and* in B's vLLM prefix cache, so re-timing
#               y1 on B would measure a local hit while calling it P2P.
#
# Each prompt gets a streaming variant (for the TTFT measurement -- the first SSE
# chunk is the first token) and a non-streaming variant (for the warm-up write).
# max_tokens=1 keeps the measurement about prefill and KV movement, not decode.
gen_payloads() {
    python3 - "$@" <<'PY'
import json, random, sys
outdir, isl_list, model, repeats = sys.argv[1], sys.argv[2].split(','), sys.argv[3], int(sys.argv[4])
VOCAB = ["alpha","bravo","delta","echo","forest","gamma","harbor","ivory","jungle",
         "kernel","lumen","meadow","nectar","opal","prairie","quartz","ridge",
         "summit","tundra","umber","vector","willow","xenon","yonder","zephyr"]
def write(name, prompt, stream):
    body = {"model": model, "prompt": prompt, "max_tokens": 1,
            "temperature": 0.0, "stream": stream}
    with open(name, "w") as f:
        json.dump(body, f)
for isl in isl_list:
    n = int(isl)
    tags = ["x"] + [f"y{i}" for i in range(1, repeats + 1)]
    for tag in tags:
        rng = random.Random(f"aic-p2p-{n}-{tag}")
        # ~1 token per word for this vocabulary; the true count comes from
        # /tokenize at run time and is recorded in results.csv.
        prompt = f"context {n} {tag}: " + " ".join(rng.choice(VOCAB) for _ in range(n))
        write(f"{outdir}/isl{n}-{tag}-stream.json", prompt, True)
        write(f"{outdir}/isl{n}-{tag}.json", prompt, False)
        if tag == "x":
            # Token-count probe: /tokenize needs only the prompt, and unlike a
            # completion it does NOT populate any KV cache -- critical, because
            # anything that caches x would turn the cold baseline into a hit.
            with open(f"{outdir}/isl{n}-tokenize.json", "w") as f:
                json.dump({"model": model, "prompt": prompt}, f)
print("ok")
PY
}
command -v python3 >/dev/null 2>&1 || die "python3 not found on the batch host (needed to build payloads)"
IFS=',' read -r -a ISLS <<< "${AIC_P2P_ISL_LIST}"
gen_payloads "${LOGDIR}/payloads" "${AIC_P2P_ISL_LIST}" "${VLLM_MODEL}" "${AIC_P2P_REPEATS}" >/dev/null \
    || die "payload generation failed"
log "payloads: $(ls "${LOGDIR}"/payloads/*.json | wc -l) files in ${LOGDIR}/payloads"

# --- Measurement helpers -----------------------------------------------------
# TTFT via curl's time_starttransfer on a streaming completion: the first SSE
# chunk is the first token.  Runs ON the target node against its own loopback so
# the number excludes any client-side network hop.
ttft_ms() {  # $1=node $2=ip $3=payload-file(host path) -> milliseconds
    local n="$1" ip="$2" pf="$3"
    on_node "${n}" "curl -s -o /dev/null --max-time 600 \
        -w '%{time_starttransfer}' \
        -H 'Content-Type: application/json' \
        -X POST --data-binary @'${pf}' \
        http://${ip}:${VLLM_PORT}/v1/completions" \
      | awk '{printf "%.1f", $1*1000}'
}

# Exact prompt length via vLLM's /tokenize.  Deliberately NOT a completion call:
# a completion would populate the KV cache for that prompt and destroy whichever
# baseline it was meant to describe.
prompt_tokens() {  # $1=node $2=ip $3=tokenize-payload-file
    local n="$1" ip="$2" pf="$3"
    on_node "${n}" "curl -s --max-time 60 \
        -H 'Content-Type: application/json' \
        -X POST --data-binary @'${pf}' \
        http://${ip}:${VLLM_PORT}/tokenize" \
      | python3 -c "import json,sys
try:
    d = json.load(sys.stdin)
    print(d.get('count') or len(d.get('tokens', [])))
except Exception: print(0)" 2>/dev/null
}

# Issue a completion without timing it (used to warm a prompt into a node's L1).
warm() {  # $1=node $2=ip $3=payload-file
    local n="$1" ip="$2" pf="$3"
    on_node "${n}" "curl -s -o /dev/null --max-time 600 \
        -H 'Content-Type: application/json' \
        -X POST --data-binary @'${pf}' \
        http://${ip}:${VLLM_PORT}/v1/completions" || true
}

capture_status() {  # $1=tag
    local tag="$1"
    on_a "curl -sf --max-time 10 http://${NODE_A_IP}:${LMCACHE_HTTP_PORT}/status" \
        > "${LOGDIR}/status-a-${tag}.json" 2>/dev/null || true
    on_b "curl -sf --max-time 10 http://${NODE_B_IP}:${LMCACHE_HTTP_PORT}/status" \
        > "${LOGDIR}/status-b-${tag}.json" 2>/dev/null || true
}

# --- The sweep ---------------------------------------------------------------
# Long format, one row per measurement, so the phases stay unambiguous and the
# result is trivial to pivot.  ib_* are only meaningful on the p2p rows (and only
# if the nodes are exclusive -- the sysfs counters are node-wide, so a co-tenant's
# traffic lands in the same delta; see the --exclusive note in the header).
# t_start/t_end are Unix epoch seconds bracketing each measurement, so a phase can
# be sliced straight out of the Prometheus TSDB, e.g.
#   rdma_port_xmit_data_total{role="a"} @ <t_end> - ... @ <t_start>
CSV="${LOGDIR}/results.csv"
echo "isl_requested,prompt_tokens,repeat,phase,node,ttft_ms,ib_rx_bytes_b,ib_tx_bytes_a,implied_gbps,t_start,t_end" > "${CSV}"
now() { date +%s.%N; }

log "=== sweep: ISL ${AIC_P2P_ISL_LIST}, ${AIC_P2P_REPEATS} repeat(s) ==="
capture_status "start"

for isl in "${ISLS[@]}"; do
    log "--- ISL ${isl} ---"

    # Exact prompt length.  /tokenize leaves no cache footprint.
    ntok="$(prompt_tokens "${NODE_A}" "${NODE_A_IP}" "${LOGDIR}/payloads/isl${isl}-tokenize.json")"

    # (1) cold baseline: prompt X, served once on A.  Neither node has seen X,
    #     and nothing below ever touches it again -> a genuine full prefill.
    ts0="$(now)"
    t_cold="$(ttft_ms "${NODE_A}" "${NODE_A_IP}" "${LOGDIR}/payloads/isl${isl}-x-stream.json")"
    ts1="$(now)"
    log "  cold prefill  (x  on ${NODE_A}): ${t_cold} ms   prompt_tokens=${ntok}"
    echo "${isl},${ntok},0,cold,${NODE_A},${t_cold},,,,${ts0},${ts1}" >> "${CSV}"

    # (2..) one distinct prompt per repeat: warm on A, fetch on B, then re-serve
    #       on A as the local-hit floor.
    for ((r=1; r<=AIC_P2P_REPEATS; r++)); do
        pw="${LOGDIR}/payloads/isl${isl}-y${r}.json"
        ps="${LOGDIR}/payloads/isl${isl}-y${r}-stream.json"

        warm "${NODE_A}" "${NODE_A_IP}" "${pw}"
        sleep "${AIC_P2P_SETTLE_S:-5}"   # let the async store + coordinator report land

        # P2P: B has never seen y{r}; avoiding a full prefill requires pulling it
        # from node-a.  IB counters bracket exactly this call.
        ts0="$(now)"
        rx0="$(ib_bytes "${NODE_B}" rcv)"; tx0="$(ib_bytes "${NODE_A}" xmit)"
        t_p2p="$(ttft_ms "${NODE_B}" "${NODE_B_IP}" "${ps}")"
        rx1="$(ib_bytes "${NODE_B}" rcv)"; tx1="$(ib_bytes "${NODE_A}" xmit)"
        ts1="$(now)"
        d_rx=$(( ${rx1:-0} - ${rx0:-0} ))
        d_tx=$(( ${tx1:-0} - ${tx0:-0} ))
        # Implied wire rate over the p2p step.  For scale only, not a link
        # benchmark: the interval also covers the coordinator lookup RTT, the
        # NIXL handshake, and cache bookkeeping.
        gbps="$(awk -v b="${d_rx}" -v ms="${t_p2p}" \
            'BEGIN{ if (ms>0) printf "%.2f", (b*8)/(ms/1000)/1e9; else print "0" }')"
        log "  p2p fetch     (y${r} on ${NODE_B}): ${t_p2p} ms   IB rx(B)=${d_rx} B tx(A)=${d_tx} B  ~${gbps} Gb/s"
        echo "${isl},${ntok},${r},p2p,${NODE_B},${t_p2p},${d_rx},${d_tx},${gbps},${ts0},${ts1}" >> "${CSV}"

        # Local hit on A for the same prompt: the floor this whole thing is
        # trying to approach.
        ts0="$(now)"
        t_local="$(ttft_ms "${NODE_A}" "${NODE_A_IP}" "${ps}")"
        ts1="$(now)"
        log "  local hit     (y${r} on ${NODE_A}): ${t_local} ms"
        echo "${isl},${ntok},${r},local,${NODE_A},${t_local},,,,${ts0},${ts1}" >> "${CSV}"
    done

    capture_status "isl${isl}"
done

# --- Evidence ----------------------------------------------------------------
# Let the last scrapes land before teardown stops Prometheus, so the TSDB covers
# the final measurement rather than ending mid-phase.
sleep "${AIC_PROM_DRAIN_S:-15}"
capture_status "end"
on_a "curl -sf --max-time 10 http://${NODE_A_IP}:${COORDINATOR_PORT}/instances" \
    > "${LOGDIR}/instances-end.json" 2>/dev/null || true
for n_role in "${NODE_A}:a" "${NODE_B}:b"; do
    n="${n_role%%:*}"; r="${n_role##*:}"
    on_node "${n}" "curl -sf --max-time 10 http://localhost:${LMCACHE_PROM_PORT}/metrics" \
        > "${LOGDIR}/lmcache-prom-${r}.txt" 2>/dev/null || true
    on_node "${n}" "curl -sf --max-time 10 http://localhost:${VLLM_PORT}/metrics" \
        > "${LOGDIR}/vllm-prom-${r}.txt" 2>/dev/null || true
done

# Per-ISL medians across repeats: the table worth pasting into a report.
log ""
log "=== summary: median over ${AIC_P2P_REPEATS} repeat(s) (${LOGDIR_REL}/summary.csv) ==="
python3 - "${CSV}" "${LOGDIR}/summary.csv" <<'PY' || warn "summary generation failed"
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
def med(vals):
    vals = [float(v) for v in vals if v not in ("", None)]
    return statistics.median(vals) if vals else 0.0
isls, seen = [], set()
for r in rows:
    if r["isl_requested"] not in seen:
        seen.add(r["isl_requested"]); isls.append(r["isl_requested"])
out = [("isl", "tokens", "cold_ms", "p2p_ms", "local_ms",
        "p2p_speedup", "p2p_vs_local", "ib_rx_MiB", "gbps")]
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f); w.writerow(out[0])
    for isl in isls:
        sel = lambda ph, col: [r[col] for r in rows
                               if r["isl_requested"] == isl and r["phase"] == ph]
        tok  = sel("cold", "prompt_tokens")[:1] or ["0"]
        cold = med(sel("cold", "ttft_ms"))
        p2p  = med(sel("p2p", "ttft_ms"))
        loc  = med(sel("local", "ttft_ms"))
        rx   = med(sel("p2p", "ib_rx_bytes_b"))
        gb   = med(sel("p2p", "implied_gbps"))
        row = (isl, tok[0], f"{cold:.1f}", f"{p2p:.1f}", f"{loc:.1f}",
               f"{cold/p2p:.2f}x" if p2p else "-",
               f"{p2p/loc:.2f}x" if loc else "-",
               f"{rx/1048576:.1f}", f"{gb:.2f}")
        out.append(row); w.writerow(row)
width = [max(len(str(r[i])) for r in out) for i in range(len(out[0]))]
for i, r in enumerate(out):
    print("  " + "  ".join(str(c).rjust(width[j]) for j, c in enumerate(r)))
    if i == 0:
        print("  " + "  ".join("-" * width[j] for j in range(len(width))))
PY

log ""
log "=== per-measurement results (${CSV}) ==="
column -t -s, "${CSV}" 2>/dev/null || cat "${CSV}"
log ""
log "=== transport evidence ==="
# UCX prints the selected transport at agent init; rc_mlx5/rc_verbs = RDMA, tcp = fallback.
for r in a b; do
    _tl="$(grep -ioE 'rc_mlx5|rc_verbs|dc_mlx5|\btcp\b' "${LOGDIR}/roles/lmcache-${r}.log" 2>/dev/null \
           | sort | uniq -c | sort -rn | head -4 | tr '\n' ' ')"
    log "  lmcache-${r} UCX transports mentioned: ${_tl:-none found (raise AIC_P2P_UCX_LOG_LEVEL)}"
done
_p2pcount="$(grep -c 'P2P transfer finished' "${LOGDIR}/roles/lmcache-b.log" 2>/dev/null || echo 0)"
log "  'P2P transfer finished' lines in lmcache-b.log: ${_p2pcount}"
if (( _p2pcount == 0 )); then
    warn "no P2P transfers recorded on node B -- the 'p2p' column is probably just recompute."
    warn "check ${LOGDIR}/roles/lmcache-b.log and ${LOGDIR}/instances.json"
fi

log ""
log "=== metrics ==="
log "  TSDB:   ${AIC_METRICS_DIR}"
log "  config: ${LOGDIR_REL}/prometheus/prometheus.yml"
log "  Replay it locally (the TSDB outlives the allocation):"
log "    docker run --rm -p 9090:9090 -v ${AIC_METRICS_DIR}:/prometheus \\"
log "      prom/prometheus:v2.55.1 --storage.tsdb.path=/prometheus \\"
log "      --storage.tsdb.retention.time=${AIC_PROM_RETENTION}"
log "  Every series is labelled node=<hostname> role=<a|b|...>; results.csv carries"
log "  t_start/t_end per measurement so a phase can be sliced exactly.  Useful queries:"
log "    sum by (node) (rate(rdma_port_xmit_data_total[15s]))   # bytes/s off each node"
log "    sum by (node) (rate(rdma_port_rcv_data_total[15s]))    # bytes/s into each node"
log "    node_infiniband_port_data_transmitted_bytes_total      # node-exporter cross-check"
log "    vllm:time_to_first_token_seconds                       # vLLM's own TTFT view"
log ""
log "done."
