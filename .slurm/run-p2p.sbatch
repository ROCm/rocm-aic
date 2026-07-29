#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
#SBATCH --job-name=aic-p2p
#SBATCH --partition=defq
# Two nodes on the SAME InfiniBand fabric.  What the job actually needs is
# ConnectX-7 (400 Gb, link layer InfiniBand), local NVMe, and a shared NFS $HOME
# both nodes can read/write -- CX7&NVME expresses exactly that.  The vendor is
# not a requirement: constraining to SUPERMICRO as well only shrinks the pool
# (19 -> 21 eligible nodes when dropped) and lengthens the queue wait on defq.
# Override on the command line for another pool, e.g.
#   sbatch --constraint='RDMA&IB' .slurm/run-p2p.sbatch
#   sbatch --nodelist=ctr-cx65-mi300x-30,ctr-cx66-mi300x-31 .slurm/run-p2p.sbatch
# NOTE: the `rccl` partition also groups these nodes but requires a Slurm
# account this job does not have ("Invalid account or account/partition
# combination"), so stay on defq.
#SBATCH --constraint=CX7&NVME
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
# SINGLE-NODE MODE (AIC_P2P_SINGLE_NODE, auto-detected on a 1-node allocation)
# puts BOTH instances on one host: two containers (aic-p2p-a / aic-p2p-b), two
# GPUs, and role B's ports shifted by AIC_P2P_PORT_OFFSET because both keep
# --network host.  It exercises the entire stack -- coordinator registration,
# lookup, the NIXL handshake, the KV pull, the vLLM MP connector -- without
# waiting for a two-node allocation.  What it does NOT prove is RDMA: with both
# peers on one host UCX picks shm/self, so the IB counters stay flat and
# implied_gbps is suppressed rather than fabricated.  Use it as a plumbing
# smoke test before committing to the real 2-node run.  (AIC_P2P_UCX_TLS=
# rc_mlx5,ud forces even that loopback through the HCA if you want the verbs
# path exercised too.)
#
# Submit from the tree root:
#
#   sbatch .slurm/run-p2p.sbatch
#   sbatch --exclusive .slurm/run-p2p.sbatch                    # publishable
#   AIC_P2P_ISL_LIST=4096,16384 sbatch .slurm/run-p2p.sbatch    # narrow sweep
#   VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct sbatch .slurm/run-p2p.sbatch
#   sbatch --nodes=1 --gres=gpu:2 .slurm/run-p2p.sbatch         # single-node
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

# Single-node mode: both peers on one host.  `auto` decides from the allocation
# size; `1` forces it (and then requires >=2 GPUs); `0` keeps the strict
# two-node requirement.  Resolved just below node discovery, once ${#NODES[@]}
# is known -- see the SINGLE_NODE branch there.
AIC_P2P_SINGLE_NODE="${AIC_P2P_SINGLE_NODE:-auto}"
# Amount role B's ports are shifted by when both roles share a host.  Both
# containers run --network host, so every listening port must be distinct.
AIC_P2P_PORT_OFFSET="${AIC_P2P_PORT_OFFSET:-100}"

# --- No-GPU mode -------------------------------------------------------------
# AIC_P2P_NO_GPU=1 measures the TRANSPORT ONLY and needs no GPU at all.
#
# Nothing in the KV path between two LMCache instances touches a GPU: the L1
# cache is host DRAM (POSIX shm), and a P2P fetch is DRAM -> NIC -> DRAM via
# NIXL/UCX.  The GPU in the default mode exists solely so vLLM can run a real
# model and produce a TTFT number.  Drop vLLM and the whole thing is a CPU job.
#
# So this mode replaces the coordinator + lmcache + vLLM + TTFT sweep with
# LMCache's own upstream transfer-channel benchmark
# (`lmcache tool transfer-channel-benchmark`, lmcache/tools/transfer_channel_
# benchmark/): role `server` registers an L1 host buffer and serves its object
# catalog, role `client` reads a random subset over the SAME nixl/UCX channel
# the P2P path uses, and reports best/median/mean GB/s.  --verify checks the
# bytes against a known pattern, so it is a correctness test as well.
#
# What you get:   real RDMA bandwidth + IB counter deltas + payload verification
# What you lose:  TTFT, speedup vs cold prefill, the vLLM MP connector path
#
# Cost: no --gres=gpu, so Slurm can backfill it onto the CPU side of a `mix`
# node instead of queueing behind whole-GPU allocations.
#   sbatch --nodes=2 --constraint='CX7&IB' .slurm/run-p2p.sbatch   # AIC_P2P_NO_GPU=1
AIC_P2P_NO_GPU="${AIC_P2P_NO_GPU:-0}"
# Object sizes to sweep (bytes, or the tool's size strings: 1MB, 64MB, 1GB...).
# Each is one server+client pair; object_size must be a multiple of the page.
AIC_P2P_BENCH_OBJ_LIST="${AIC_P2P_BENCH_OBJ_LIST:-1MB,4MB,16MB,64MB}"
AIC_P2P_BENCH_NUM_OBJECTS="${AIC_P2P_BENCH_NUM_OBJECTS:-100}"   # objects per read
AIC_P2P_BENCH_ITERS="${AIC_P2P_BENCH_ITERS:-10}"                # measured reads
AIC_P2P_BENCH_WARMUP="${AIC_P2P_BENCH_WARMUP:-2}"
AIC_P2P_BENCH_BUFFER="${AIC_P2P_BENCH_BUFFER:-16GB}"            # server L1 region
AIC_P2P_BENCH_PAGE="${AIC_P2P_BENCH_PAGE:-524288}"              # align_bytes
AIC_P2P_BENCH_VERIFY="${AIC_P2P_BENCH_VERIFY:-1}"
AIC_P2P_BENCH_TIMEOUT="${AIC_P2P_BENCH_TIMEOUT:-900}"           # per client run

# Model.  Default to the small full-attention model the cliff already stages:
# gpt-oss is a hybrid (sliding-window) model and cannot unify its KV-cache specs
# under a KV connector, so it is NOT a valid default here.
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
# The staged cache on this cluster is $HOME/aic-hf (that is where run-cliff.sbatch
# put Qwen2.5-3B-Instruct); ${_shared_root}/hf does not exist.  Prefer a populated
# aic-hf, else fall back so an AIC_SHARED_NFS deployment still works unchanged.
if [[ -z "${HF_HOME:-}" && -d "${_shared_root}/aic-hf/hub" ]]; then
    HF_HOME="${_shared_root}/aic-hf"
fi
HF_HOME="${HF_HOME:-${_shared_root}/hf}"
HF_TOKEN="${HF_TOKEN:-}"
# Offline by default only when the cache looks populated; otherwise let vLLM
# fetch (Qwen2.5-3B is not gated).  Both nodes share the NFS cache, so only one
# download happens in practice.
AIC_P2P_OFFLINE="${AIC_P2P_OFFLINE:-auto}"

# Ports (host network on both nodes).  All fixed -- see the --exclusive note in
# the header; preflight_ports() below fails fast on a collision.
#
# These are the ROLE A values.  Role B's are derived after node discovery: the
# same numbers in two-node mode (different hosts, no conflict), shifted by
# AIC_P2P_PORT_OFFSET in single-node mode.  The coordinator is a singleton and
# is never shifted.
COORDINATOR_PORT="${AIC_P2P_COORDINATOR_PORT:-9300}"
LMCACHE_ZMQ_PORT="${AIC_P2P_ZMQ_PORT:-5555}"
LMCACHE_HTTP_PORT="${AIC_P2P_HTTP_PORT:-8080}"
P2P_PORT="${AIC_P2P_PORT:-8500}"
VLLM_PORT="${AIC_P2P_VLLM_PORT:-8000}"
LMCACHE_PROM_PORT="${AIC_P2P_LMCACHE_PROM_PORT:-9400}"
# No-GPU (transfer-channel benchmark) ports.  The bench's own defaults are
# 7600/7601/7610; keep them so a hand-run matches the job.  BENCH_XFER is the
# server's nixl listen/advertise url, BENCH_LISTEN the client's own mandatory
# (never-read) channel server, BENCH_CTRL the ZMQ catalog side-channel.
BENCH_XFER_PORT="${AIC_P2P_BENCH_XFER_PORT:-7600}"
BENCH_LISTEN_PORT="${AIC_P2P_BENCH_LISTEN_PORT:-7601}"
BENCH_CTRL_PORT="${AIC_P2P_BENCH_CTRL_PORT:-7610}"

# LMCache sizing.  A pure DRAM L1 keeps this experiment about the network: no
# NVMe/GDS tier, so a miss on B can only be served by the peer or recomputed.
L1_SIZE_GB="${AIC_P2P_L1_SIZE_GB:-50}"
L1_ALIGN_BYTES="${AIC_P2P_L1_ALIGN_BYTES:-65536}"
P2P_LOOKUP_TIMEOUT="${AIC_P2P_LOOKUP_TIMEOUT:-30}"
P2P_LOAD_TIMEOUT="${AIC_P2P_LOAD_TIMEOUT:-30}"
P2P_TRANSFER_ENGINE="${AIC_P2P_TRANSFER_ENGINE:-nixl}"
# NIXL backend for the no-GPU transfer-channel benchmark.  UCX is what
# NixlAgent(backends=[...]) uses on the P2P path, so keep them the same.
AIC_P2P_NIXL_BACKEND="${AIC_P2P_NIXL_BACKEND:-UCX}"

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
(( ${#NODES[@]} >= 1 )) || die "no nodes in SLURM_JOB_NODELIST (${SLURM_JOB_NODELIST:-unset})"

# Resolve the mode now that the allocation size is known.
case "${AIC_P2P_SINGLE_NODE}" in
    auto) SINGLE_NODE=$(( ${#NODES[@]} < 2 ? 1 : 0 )) ;;
    1)    SINGLE_NODE=1 ;;
    0)    SINGLE_NODE=0 ;;
    *)    die "AIC_P2P_SINGLE_NODE must be auto, 0 or 1 (got '${AIC_P2P_SINGLE_NODE}')" ;;
esac
(( SINGLE_NODE || ${#NODES[@]} >= 2 )) \
    || die "need 2 nodes, got ${#NODES[@]} (${SLURM_JOB_NODELIST:-unset}); submit with --nodes=2, or set AIC_P2P_SINGLE_NODE=1"

case "${AIC_P2P_NO_GPU}" in
    1) NO_GPU=1 ;;
    0) NO_GPU=0 ;;
    *) die "AIC_P2P_NO_GPU must be 0 or 1 (got '${AIC_P2P_NO_GPU}')" ;;
esac
if (( NO_GPU )); then
    log "mode: NO-GPU transport benchmark (no vLLM, no model, host DRAM buffers)"
    # amdgpu-exporter and hsa-snoop both open /dev/kfd; on a node with no ROCm
    # devices they crash-loop and the healthcheck fails the whole job.
    if [[ "${AIC_P2P_GPU_EXPORTERS}" == "1" ]]; then
        warn "AIC_P2P_GPU_EXPORTERS=1 ignored: no-gpu mode has no /dev/kfd to scrape"
        AIC_P2P_GPU_EXPORTERS=0
    fi
    if (( SINGLE_NODE )); then
        warn "no-gpu + single-node: server and client share a host, so UCX will"
        warn "  pick shm/self and the result is a memcpy rate, not a fabric rate."
    fi
fi

NODE_A="${NODES[0]}"
if (( SINGLE_NODE )); then
    NODE_B="${NODE_A}"
else
    NODE_B="${NODES[1]}"
fi

# Per-role identity.  In two-node mode every B value equals its A value -- the
# roles are separated by the host, exactly as before.  In single-node mode they
# are separated by container name, GPU and port instead.
if (( SINGLE_NODE )); then
    CONTAINER_A="${CONTAINER}-a"
    CONTAINER_B="${CONTAINER}-b"
    _off="${AIC_P2P_PORT_OFFSET}"
    GPU_A="${AIC_P2P_GPU_A:-0}"
    GPU_B="${AIC_P2P_GPU_B:-1}"
else
    CONTAINER_A="${CONTAINER}"
    CONTAINER_B="${CONTAINER}"
    _off=0
    GPU_A="${AIC_P2P_GPU_A:-${GPU}}"
    GPU_B="${AIC_P2P_GPU_B:-${GPU}}"
fi
ZMQ_A="${LMCACHE_ZMQ_PORT}";       ZMQ_B=$(( LMCACHE_ZMQ_PORT  + _off ))
HTTP_A="${LMCACHE_HTTP_PORT}";     HTTP_B=$(( LMCACHE_HTTP_PORT + _off ))
P2PP_A="${P2P_PORT}";              P2PP_B=$(( P2P_PORT          + _off ))
VLLM_A="${VLLM_PORT}";             VLLM_B=$(( VLLM_PORT         + _off ))
LMPROM_A="${LMCACHE_PROM_PORT}";   LMPROM_B=$(( LMCACHE_PROM_PORT + _off ))
# No-GPU bench.  The server (role A) owns XFER + CTRL, the client (role B) owns
# LISTEN; they are on distinct hosts in two-node mode, so only single-node needs
# the shift -- and there only the client's, since server and client already use
# different port numbers.
BXFER_A="${BENCH_XFER_PORT}";      BCTRL_A="${BENCH_CTRL_PORT}"
BLISTEN_B=$(( BENCH_LISTEN_PORT + _off ))

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
if (( SINGLE_NODE )); then
    # Asking a host which source address it would use to reach ITSELF yields
    # 127.0.0.1, which NIXL cannot usefully advertise.  Use the node's real
    # routable address for both roles so the config shape matches two-node mode.
    NODE_A_IP="${NODE_A_DNS}"
    NODE_B_IP="${NODE_A_DNS}"
else
    NODE_A_IP="$(_src_ip_to "${NODE_A}" "${NODE_B_DNS}")"; NODE_A_IP="${NODE_A_IP:-${NODE_A_DNS}}"
    NODE_B_IP="$(_src_ip_to "${NODE_B}" "${NODE_A_DNS}")"; NODE_B_IP="${NODE_B_IP:-${NODE_B_DNS}}"
fi

# The workload uses exactly two nodes (A serves, B fetches), but monitoring
# covers the WHOLE allocation -- so a 4-node run submitted for a future
# many-peer topology still gets every node scraped without touching this code.
# ALL_NODES / ALL_IPS stay index-aligned; role is the a/b/c... suffix used as a
# Prometheus label.
ALL_NODES=("${NODES[@]}")
ALL_IPS=("${NODE_A_IP}")
(( SINGLE_NODE )) || ALL_IPS+=("${NODE_B_IP}")
for ((i=${#ALL_IPS[@]}; i<${#ALL_NODES[@]}; i++)); do
    _ip="$(_src_ip_to "${ALL_NODES[i]}" "${NODE_A_DNS}")"
    ALL_IPS+=("${_ip:-$(_dns_ip "${ALL_NODES[i]}")}")
done
_role_of() { printf '%s' "$(printf "\\$(printf '%03o' $((97 + $1)))")"; }   # 0->a, 1->b, ...

# The two workload instances, as node|container|gpu|role records.  Every
# per-role loop below iterates THIS rather than a (NODE_A, NODE_B) pair, so
# single-node mode -- where the two records share a host but differ in
# container, GPU and ports -- needs no special-casing at each site.
INSTANCES=(
    "${NODE_A}|${CONTAINER_A}|${GPU_A}|a"
    "${NODE_B}|${CONTAINER_B}|${GPU_B}|b"
)
# Nodes to act on once each (image load, IB probe, exporter fleet).  Distinct
# from INSTANCES: in single-node mode both instances live on one node.
UNIQ_NODES=("${ALL_NODES[@]}")

log "job ${SLURM_JOB_ID:-manual}  logs -> ${LOGDIR}"
if (( SINGLE_NODE )); then
    log "SINGLE-NODE mode: both instances on ${NODE_A} (${NODE_A_IP})"
    log "  role a: container ${CONTAINER_A} gpu ${GPU_A}  vllm :${VLLM_A} lmcache :${ZMQ_A}/:${HTTP_A} p2p :${P2PP_A}"
    log "  role b: container ${CONTAINER_B} gpu ${GPU_B}  vllm :${VLLM_B} lmcache :${ZMQ_B}/:${HTTP_B} p2p :${P2PP_B}"
    log "  NOTE: same-host peers -- UCX will pick shm/self, so IB counters stay"
    log "        flat and implied_gbps is suppressed.  This validates the P2P"
    log "        plumbing, NOT RDMA.  Use a 2-node run for fabric numbers."
else
    log "node A = ${NODE_A} (${NODE_A_IP})   [coordinator lives here]"
    log "node B = ${NODE_B} (${NODE_B_IP})"
fi
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
    local inst n c
    for inst in "${INSTANCES[@]}"; do
        IFS='|' read -r n c _ _ <<< "${inst}"
        timeout 60 sshx "${n}" "docker logs ${c} >/dev/null 2>&1" || true
        timeout 120 sshx "${n}" "docker rm -f ${c} >/dev/null 2>&1" || true
        log "  ${n}: ${c} removed"
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

# Every port this job will listen on, as "node port role" triples.  In
# single-node mode that is BOTH role port sets on the one host -- the union the
# collision check has to cover.
_all_listen_ports() {
    # No-GPU mode runs neither lmcache nor vLLM, so only the bench's three
    # sockets exist -- checking the others would reject a perfectly good node
    # because some co-tenant happens to hold :8000.
    if (( NO_GPU )); then
        printf '%s %s a\n' "${NODE_A}" "${BXFER_A}" "${NODE_A}" "${BCTRL_A}"
        printf '%s %s b\n' "${NODE_B}" "${BLISTEN_B}"
        return 0
    fi
    printf '%s %s a\n' "${NODE_A}" "${ZMQ_A}"    "${NODE_A}" "${HTTP_A}" \
                       "${NODE_A}" "${P2PP_A}"   "${NODE_A}" "${VLLM_A}" \
                       "${NODE_A}" "${LMPROM_A}"
    printf '%s %s b\n' "${NODE_B}" "${ZMQ_B}"    "${NODE_B}" "${HTTP_B}" \
                       "${NODE_B}" "${P2PP_B}"   "${NODE_B}" "${VLLM_B}" \
                       "${NODE_B}" "${LMPROM_B}"
}

preflight_ports() {
    local n p role busy=0
    while read -r n p role; do
        [[ -n "${n}" ]] || continue
        if on_node "${n}" "ss -ltn 2>/dev/null | awk '{print \$4}' | grep -qE '[:.]${p}\$'"; then
            warn "${n}: port ${p} (role ${role}) already in use (co-tenant?); resubmit with --exclusive or override the port"
            busy=1
        fi
    done < <(_all_listen_ports)
    if (( ! NO_GPU )); then
        on_node "${NODE_A}" "ss -ltn 2>/dev/null | awk '{print \$4}' | grep -qE '[:.]${COORDINATOR_PORT}\$'" \
            && { warn "${NODE_A}: coordinator port ${COORDINATOR_PORT} in use"; busy=1; }
    fi
    (( busy == 0 )) || die "port conflict; refusing to produce numbers that are not ours"
}

preflight_ib() {
    local n devs act
    for n in "${UNIQ_NODES[@]}"; do
        devs="$(on_node "${n}" "ls /sys/class/infiniband 2>/dev/null | tr '\n' ' '")"
        act="$(on_node "${n}" "grep -l ACTIVE /sys/class/infiniband/*/ports/*/state 2>/dev/null | wc -l")"
        # In single-node mode the peers never cross the fabric, so a missing or
        # down HCA is not fatal -- the gate below exists to stop a TWO-node run
        # silently benchmarking TCP, and that failure mode cannot occur here.
        if (( SINGLE_NODE )); then
            [[ -n "${devs}" ]] || { warn "${n}: no /sys/class/infiniband devices (single-node: not fatal)"; return 0; }
            log "  ${n}: ib devices [${devs}], ${act} active port(s)"
            (( act > 0 )) || warn "${n}: no ACTIVE IB port (single-node: not fatal, transfers stay on shm)"
            log "  ${n}: IB counters baseline rx=$(ib_bytes "${n}" rcv) B tx=$(ib_bytes "${n}" xmit) B"
            continue
        fi
        [[ -n "${devs}" ]] || die "${n}: no /sys/class/infiniband devices -- this node has no RDMA NIC"
        log "  ${n}: ib devices [${devs}], ${act} active port(s)"
        (( act > 0 )) || die "${n}: no ACTIVE IB port"
        on_node "${n}" "test -e /dev/infiniband/uverbs0" \
            || die "${n}: /dev/infiniband/uverbs0 missing -- cannot pass verbs into the container"
        # Also exercises the counter read path the measurement depends on.
        log "  ${n}: IB counters baseline rx=$(ib_bytes "${n}" rcv) B tx=$(ib_bytes "${n}" xmit) B"
    done
}

preflight_shared_fs() {
    if (( SINGLE_NODE )); then
        log "  shared fs: single-node, ${LOGDIR} is local to ${NODE_A} (no cross-node check)"
        return 0
    fi
    local marker="${LOGDIR}/.shared-fs-probe"
    date +%s > "${marker}"
    on_b "test -r '${marker}'" \
        || die "${LOGDIR} is not visible on ${NODE_B}; logs/ must live on the shared NFS \$HOME"
    rm -f "${marker}"
    log "  shared fs OK: ${LOGDIR} readable on both nodes"
}

# Single-node mode needs two GPUs -- one per instance -- or the two vLLM
# servers land on the same device and fight for HBM.
preflight_gpus() {
    if (( NO_GPU )); then
        log "  gpus: none required (AIC_P2P_NO_GPU=1; transport-only, host DRAM buffers)"
        return 0
    fi
    (( SINGLE_NODE )) || return 0
    [[ "${GPU_A}" != "${GPU_B}" ]] \
        || die "single-node mode needs distinct GPUs (GPU_A=${GPU_A} GPU_B=${GPU_B})"
    log "  gpus: role a -> ${GPU_A}, role b -> ${GPU_B} (SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset})"
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]] && (( SLURM_GPUS_ON_NODE < 2 )); then
        die "single-node mode needs >=2 GPUs, allocation has ${SLURM_GPUS_ON_NODE}; submit with --gres=gpu:2"
    fi
}

log "preflight"
preflight_shared_fs
preflight_gpus
preflight_ib
preflight_ports

# --- Image load --------------------------------------------------------------
# Mirrors load_image_if_needed() in run-cliff.sbatch, but fanned out over ssh.
# Reload when absent or when the shared tarball is newer than the marker.  Used
# for the AIC image (required) and for the fabric exporter images (optional --
# run-build-distribute.sh `build-exporters` writes their tarballs alongside it).
_tarball_glob() {  # $1=image:tag -> the shared-dir glob(s) its tarball matches
    # Two patterns, because the two producers name their tarballs differently:
    # the main image carries an arch-list suffix (rocm-aic-latest-gfx942-...tar.zst)
    # while the fabric exporters do not (aic-rdma-exporter-latest.tar.zst).  A
    # lone `-*.tar.*` silently misses the second and the job reports "no tarball"
    # for images that are sitting right there.
    local base
    base="$(printf '%s' "$1" | tr '/:' '--')"
    printf '%s/%s-*.tar.* %s/%s.tar.*' \
        "${AIC_IMAGE_DIR}" "${base}" "${AIC_IMAGE_DIR}" "${base}"
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
# Targets are emitted as explicit job|ip|port|node|role records rather than a
# jobs x nodes cross product.  The cross product cannot express single-node
# mode, where vllm and lmcache exist TWICE on one IP at different ports; an
# explicit list handles both modes with one code path and leaves the two-node
# output byte-identical to what the cross product produced.
#
#   workload jobs (vllm, lmcache)  -> one target per INSTANCE (role a, role b)
#   fleet jobs    (node/rdma/nvme) -> one target per NODE
write_prom_config() {
    mkdir -p "$(dirname "${PROM_CONFIG}")" "${AIC_METRICS_DIR}" 2>/dev/null || return 1
    local spec="" i inst n c g role
    # Per-instance: the workload's own endpoints, at that role's ports.
    for inst in "${INSTANCES[@]}"; do
        IFS='|' read -r n c g role <<< "${inst}"
        case "${role}" in
            a) spec+="vllm|${NODE_A_IP}|${VLLM_A}|${n}|a lmcache|${NODE_A_IP}|${LMPROM_A}|${n}|a " ;;
            b) spec+="vllm|${NODE_B_IP}|${VLLM_B}|${n}|b lmcache|${NODE_B_IP}|${LMPROM_B}|${n}|b " ;;
        esac
    done
    # Per-node: the exporter fleet, one instance of each per host.
    local -a fleet=(
        "node_exporter:${NODE_EXPORTER_PORT}"
        "rdma_exporter:${RDMA_EXPORTER_PORT}"
        "nvme_exporter:${NVME_EXPORTER_PORT}"
    )
    if [[ "${AIC_P2P_GPU_EXPORTERS}" == "1" ]]; then
        fleet+=("amd_metrics_exporter:${AMDGPU_EXPORTER_PORT}" "hsa_snoop:${HSA_SNOOP_PORT}")
    fi
    local jp
    for ((i=0; i<${#ALL_NODES[@]}; i++)); do
        for jp in "${fleet[@]}"; do
            spec+="${jp%%:*}|${ALL_IPS[i]}|${jp##*:}|${ALL_NODES[i]}|$(_role_of "${i}") "
        done
    done
    SPEC="${spec}" PROM_SELF="${AIC_PROM_PORT}" SCRAPE="${AIC_PROM_SCRAPE_INTERVAL}" \
    python3 - "${PROM_CONFIG}" <<'PY'
import os, sys
from collections import OrderedDict
targets = [s.split('|') for s in os.environ['SPEC'].split()]
byjob = OrderedDict()
for job, ip, port, node, role in targets:
    byjob.setdefault(job, []).append((ip, port, node, role))
out = [
    "# GENERATED by .slurm/run-p2p.sbatch -- do not edit; regenerated per job.",
    "# One Prometheus scrapes every allocated node by IP (the checked-in",
    "# monitoring/prometheus/prometheus.yml is localhost-only).  Workload jobs",
    "# carry one target per INSTANCE, so single-node runs -- both roles on one",
    "# host at different ports -- stay distinguishable by the role label.",
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
for name, entries in byjob.items():
    out += ["", f"  - job_name: {name}", "    metrics_path: /metrics",
            "    static_configs:"]
    for ip, port, node, role in entries:
        out += [f"      - targets: [\"{ip}:{port}\"]",
                "        labels:",
                f"          node: \"{node}\"",
                f"          role: \"{role}\""]
open(sys.argv[1], "w").write("\n".join(out) + "\n")
print(f"{len(byjob)} jobs, {len(targets)} targets")
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
    if (( SINGLE_NODE )); then
        log "preflight OK: ${NODE_A} can host both instances (single-node mode)"
    else
        log "preflight OK: ${NODE_A} + ${NODE_B} are a usable pair"
    fi
    exit 0
fi

for n in "${UNIQ_NODES[@]}"; do
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
docker_run_cmd() {  # $1=container name $2=gpu index
    # No-GPU mode: omit /dev/kfd and /dev/dri so the container starts on a node
    # with no ROCm devices at all (and so Slurm never had to grant a gres/gpu).
    # /dev/infiniband stays -- the transport is the entire point.
    local _gpudev="--device=/dev/kfd --device=/dev/dri"
    local _gpuenv="-e ROCR_VISIBLE_DEVICES=$2"
    if (( NO_GPU )); then _gpudev=""; _gpuenv=""; fi
    cat <<EOF
docker run -d --rm --name $1 \
  --network host --ipc host \
  ${_gpudev} --device=/dev/infiniband \
  --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 \
  -v ${HF_HOME}:/hf \
  -v ${LOGDIR}:/joblog \
  -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub -e VLLM_CACHE_ROOT=/hf/vllm \
  -e VLLM_CONFIG_ROOT=/hf/vllm_config -e TORCH_HOME=/hf/torch \
  -e HF_TOKEN=${HF_TOKEN} \
  -e HF_HUB_OFFLINE=${_offline} -e TRANSFORMERS_OFFLINE=${_offline} \
  ${_gpuenv} \
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
for _inst in "${INSTANCES[@]}"; do
    IFS='|' read -r n c g role <<< "${_inst}"
    on_node "${n}" "docker rm -f ${c} >/dev/null 2>&1" || true
    on_node "${n}" "$(docker_run_cmd "${c}" "${g}")" >/dev/null \
        || die "${n}: docker run ${c} failed"
    on_node "${n}" "docker ps --filter name=${c} --format '{{.Names}}'" | grep -qx "${c}" \
        || die "${n}: container ${c} did not start"
    log "  ${n}: ${c} up (role ${role}, $( (( NO_GPU )) && echo 'no gpu' || echo "gpu ${g}" ))"
done

# Confirm the container can actually see the verbs devices.  A silent failure
# here is exactly how you end up benchmarking TCP -- fatal for a two-node run.
# Single-node peers never cross the fabric, so this is informational there.
for _inst in "${INSTANCES[@]}"; do
    IFS='|' read -r n c g role <<< "${_inst}"
    _ibv="$(on_node "${n}" "docker exec ${c} ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE" || echo 0)"
    log "  ${n}/${c}: ${_ibv} PORT_ACTIVE verbs port(s) visible inside the container"
    if (( _ibv == 0 )); then
        (( SINGLE_NODE )) \
            && warn "${n}/${c}: no active verbs port (single-node: transfers stay on shm)" \
            || die "${n}: container sees no active verbs port -- RDMA would silently fall back to TCP"
    fi
done

# --- Start the roles ---------------------------------------------------------
# Each role's command is WRITTEN TO A SCRIPT under the shared LOGDIR (visible in
# the container as /joblog) and then run with `docker exec -d bash /joblog/...`.
# Passing the command inline instead would have to survive three levels of shell
# quoting -- local -> ssh -> `docker exec bash -lc` -- and the vLLM invocation
# carries single-quoted JSON (--kv-transfer-config) that does not.  The script
# file also doubles as an artifact: logs/<job-id>/roles/<role>.sh is exactly what
# ran, reproducible by hand.
exec_role() {  # $1=node $2=container $3=role-name $4=command (may span lines)
    local n="$1" c="$2" role="$3" cmd="$4"
    { echo "#!/usr/bin/env bash"; echo "set -uo pipefail"; echo "${cmd}"; } \
        > "${LOGDIR}/roles/${role}.sh"
    chmod +x "${LOGDIR}/roles/${role}.sh"
    # `bash -c`, not `bash -lc`: a login shell sources /etc/profile, which can
    # overwrite the LD_LIBRARY_PATH (ROCm + NIXL) set by `docker run -e`.
    on_node "${n}" "docker exec -d ${c} bash -c 'bash /joblog/roles/${role}.sh > /joblog/roles/${role}.log 2>&1'" \
        || die "${n}: failed to start ${role} in ${c}"
    log "  started ${role} on ${n}/${c} (cmd: ${LOGDIR_REL}/roles/${role}.sh)"
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

# --- No-GPU transport benchmark ----------------------------------------------
# Runs INSTEAD of coordinator + lmcache + vLLM + the TTFT sweep.  See the
# AIC_P2P_NO_GPU block in the config section for why this needs no GPU.

# Synchronous sibling of exec_role: the client must be waited on and its stdout
# is the measurement, so `docker exec -d` will not do.  Output is tee'd to the
# same roles/<role>.log artifact and echoed to our stdout.
# Defined here as well as in the (GPU-path) sweep below, because this block
# returns before that definition is ever reached.
now() { date +%s.%N; }

exec_role_sync() {  # $1=node $2=container $3=role-name $4=command
    local n="$1" c="$2" role="$3" cmd="$4"
    { echo "#!/usr/bin/env bash"; echo "set -uo pipefail"; echo "${cmd}"; } \
        > "${LOGDIR}/roles/${role}.sh"
    chmod +x "${LOGDIR}/roles/${role}.sh"
    on_node "${n}" "timeout ${AIC_P2P_BENCH_TIMEOUT} docker exec ${c} bash -c 'bash /joblog/roles/${role}.sh 2>&1' | tee /joblog/roles/${role}.log"
}

bench_server_cmd() {  # $1=object-size
    cat <<EOF
lmcache tool transfer-channel-benchmark \
  --role server \
  --transfer-channel-type ${P2P_TRANSFER_ENGINE} \
  --nixl-backend ${AIC_P2P_NIXL_BACKEND} \
  --url ${NODE_A_IP}:${BXFER_A} \
  --control-url 0.0.0.0:${BCTRL_A} \
  --buffer-size ${AIC_P2P_BENCH_BUFFER} \
  --page-size ${AIC_P2P_BENCH_PAGE} \
  --object-size $1 \
  --num-objects ${AIC_P2P_BENCH_NUM_OBJECTS} \
  --server-timeout ${AIC_P2P_BENCH_TIMEOUT}
EOF
}

bench_client_cmd() {  # $1=object-size
    cat <<EOF
lmcache tool transfer-channel-benchmark \
  --role client \
  --transfer-channel-type ${P2P_TRANSFER_ENGINE} \
  --nixl-backend ${AIC_P2P_NIXL_BACKEND} \
  --url ${NODE_A_IP}:${BXFER_A} \
  --listen-url ${NODE_B_IP}:${BLISTEN_B} \
  --control-url ${NODE_A_IP}:${BCTRL_A} \
  --page-size ${AIC_P2P_BENCH_PAGE} \
  --object-size $1 \
  --num-objects ${AIC_P2P_BENCH_NUM_OBJECTS} \
  --iters ${AIC_P2P_BENCH_ITERS} \
  --warmup ${AIC_P2P_BENCH_WARMUP} \
  ${_bench_verify}
EOF
}

run_nogpu_bench() {
    local CSV="${LOGDIR}/results.csv"
    echo "object_bytes,num_objects,payload_bytes,iters,best_ms,median_ms,mean_ms,best_gbps,median_gbps,ib_rx_bytes_b,ib_tx_bytes_a,wire_gbps,verified,t_start,t_end" \
        > "${CSV}"

    log "=== no-gpu transfer-channel sweep: objects ${AIC_P2P_BENCH_OBJ_LIST} x ${AIC_P2P_BENCH_NUM_OBJECTS}, ${AIC_P2P_BENCH_ITERS} iters ==="
    log "    channel=${P2P_TRANSFER_ENGINE}/${AIC_P2P_NIXL_BACKEND}  server=${NODE_A}:${BXFER_A}  client=${NODE_B}:${BLISTEN_B}"

    local obj osz ts0 ts1 rx0 rx1 tx0 tx1 d_rx d_tx
    IFS=',' read -r -a _OBJS <<< "${AIC_P2P_BENCH_OBJ_LIST}"
    for obj in "${_OBJS[@]}"; do
        obj="$(printf '%s' "${obj}" | tr -d '[:space:]')"
        [[ -n "${obj}" ]] || continue
        log "--- object size ${obj} ---"

        # Server first; it must be listening before the client dials the
        # catalog socket.  One server per object size: --object-size is baked
        # into the registered pool, and the client refuses a mismatch.
        exec_role "${NODE_A}" "${CONTAINER_A}" "bench-server-${obj}" "$(bench_server_cmd "${obj}")"
        local waited=0 ready=0
        while (( waited < 180 )); do
            if grep -q '\[server\] ready' "${LOGDIR}/roles/bench-server-${obj}.log" 2>/dev/null; then
                ready=1; break
            fi
            sleep 3; waited=$((waited + 3))
        done
        if (( ! ready )); then
            warn "bench server never reported ready for object size ${obj}; skipping"
            warn "  see ${LOGDIR_REL}/roles/bench-server-${obj}.log"
            continue
        fi
        log "  server ready (${waited}s)"

        # The client run IS the measurement; bracket it with the IB counters so
        # the tool's self-reported rate can be cross-checked against bytes the
        # HCA actually moved.
        ts0="$(now)"
        rx0="$(ib_bytes "${NODE_B}" rcv)"; tx0="$(ib_bytes "${NODE_A}" xmit)"
        exec_role_sync "${NODE_B}" "${CONTAINER_B}" "bench-client-${obj}" "$(bench_client_cmd "${obj}")" \
            || warn "bench client exited non-zero for object size ${obj}"
        rx1="$(ib_bytes "${NODE_B}" rcv)"; tx1="$(ib_bytes "${NODE_A}" xmit)"
        ts1="$(now)"
        d_rx=$(( ${rx1:-0} - ${rx0:-0} ))
        d_tx=$(( ${tx1:-0} - ${tx0:-0} ))

        # The server exits on its own once the client is done reading, but do
        # not leave it holding the pool if the client died.
        on_node "${NODE_A}" "docker exec ${CONTAINER_A} pkill -f 'transfer-channel-benchmark.*--role server'" >/dev/null 2>&1 || true

        # Parse the tool's own report out of the client log.
        local clog="${LOGDIR}/roles/bench-client-${obj}.log"
        local osz_b best_ms med_ms mean_ms best_gbps med_gbps verified
        osz_b="$(awk -v s="${obj}" 'BEGIN{
            u["B"]=1;u["KB"]=1024;u["MB"]=1048576;u["GB"]=1073741824;
            s=toupper(s);
            for (k in u) if (index(s,k) && substr(s,length(s)-length(k)+1)==k) {
                printf "%d", substr(s,1,length(s)-length(k)) * u[k]; exit }
            printf "%d", s }')"
        best_ms="$(awk '/^ *best *:/ {print $3}'   "${clog}" 2>/dev/null | tail -1)"
        med_ms="$(awk  '/^ *median *:/ {print $3}' "${clog}" 2>/dev/null | tail -1)"
        mean_ms="$(awk '/^ *mean *:/ {print $3}'   "${clog}" 2>/dev/null | tail -1)"
        best_gbps="$(awk '/^ *best *:/ {print $5}'   "${clog}" 2>/dev/null | tail -1)"
        med_gbps="$(awk  '/^ *median *:/ {print $5}' "${clog}" 2>/dev/null | tail -1)"
        if grep -qi 'verif' "${clog}" 2>/dev/null; then
            grep -qiE 'mismatch|FAIL' "${clog}" && verified=FAIL || verified=OK
        else
            verified="n/a"
        fi

        local payload=$(( osz_b * AIC_P2P_BENCH_NUM_OBJECTS ))
        local wire_gbps=""
        if (( SINGLE_NODE )); then
            log "  [same-host: IB delta is not a fabric measurement]"
        else
            # Bytes the receiving HCA actually took in over the whole client
            # run, divided by that wall time.  Lower than the tool's per-read
            # rate by construction (it includes connect + warmup + idle), but
            # it is the number that cannot be faked by a shm fallback.
            wire_gbps="$(awk -v b="${d_rx}" -v t0="${ts0}" -v t1="${ts1}" \
                'BEGIN{ d=t1-t0; if (d>0) printf "%.2f", (b*8)/d/1e9; else print "0" }')"
            log "  IB rx(B)=${d_rx} B  tx(A)=${d_tx} B over $(awk -v a="${ts0}" -v b="${ts1}" 'BEGIN{printf "%.1f", b-a}')s  ~${wire_gbps} Gb/s end-to-end"
            # A read of N objects must put ~N*object_size on the wire.  If it
            # did not, UCX did not use the HCA and the tool's GB/s is a
            # host-memory number wearing a fabric costume.
            awk -v got="${d_rx}" -v want="$(( payload * (AIC_P2P_BENCH_ITERS + AIC_P2P_BENCH_WARMUP) ))" \
                'BEGIN{ exit (got >= want*0.5) ? 0 : 1 }' \
                || warn "IB rx delta (${d_rx} B) is far below the ${payload} B x $(( AIC_P2P_BENCH_ITERS + AIC_P2P_BENCH_WARMUP )) reads expected -- transfers may not have crossed the fabric"
        fi
        log "  ${obj}: best ${best_ms:-?} ms / ${best_gbps:-?} GB/s   median ${med_ms:-?} ms / ${med_gbps:-?} GB/s   verify=${verified}"

        echo "${osz_b},${AIC_P2P_BENCH_NUM_OBJECTS},${payload},${AIC_P2P_BENCH_ITERS},${best_ms},${med_ms},${mean_ms},${best_gbps},${med_gbps},${d_rx},${d_tx},${wire_gbps},${verified},${ts0},${ts1}" \
            >> "${CSV}"
    done

    log ""
    log "=== summary (${LOGDIR_REL}/results.csv) ==="
    column -t -s, "${CSV}" 2>/dev/null || cat "${CSV}"

    log ""
    log "=== transport evidence ==="
    for obj in "${_OBJS[@]}"; do
        obj="$(printf '%s' "${obj}" | tr -d '[:space:]')"
        [[ -n "${obj}" ]] || continue
        local _tl
        _tl="$(grep -hioE 'rc_mlx5|rc_verbs|dc_mlx5|\bshm\b|\bself\b|\btcp\b' \
               "${LOGDIR}/roles/bench-client-${obj}.log" "${LOGDIR}/roles/bench-server-${obj}.log" 2>/dev/null \
               | sort | uniq -c | sort -rn | head -5 | tr '\n' ' ')"
        log "  ${obj}: UCX transports mentioned: ${_tl:-none found (raise AIC_P2P_UCX_LOG_LEVEL)}"
    done
    log ""
    log "  NOTE: this run measured the TRANSPORT only.  There was no model, no"
    log "        vLLM and no GPU, so there is no TTFT or cold-vs-warm speedup"
    log "        here -- those need the default (GPU) mode.  What it does prove"
    log "        is the nixl/UCX channel LMCache P2P rides on, end to end,"
    log "        including payload verification."
}

if (( NO_GPU )); then
    _bench_verify=""
    [[ "${AIC_P2P_BENCH_VERIFY}" == "1" ]] && _bench_verify="--verify"
    run_nogpu_bench
    sleep "${AIC_PROM_DRAIN_S:-15}"
    log ""
    log "=== metrics ==="
    log "  TSDB:   ${AIC_METRICS_DIR}"
    log "  config: ${LOGDIR_REL}/prometheus/prometheus.yml"
    log "done."
    exit 0
fi

log "starting coordinator on ${NODE_A}"
exec_role "${NODE_A}" "${CONTAINER_A}" coordinator \
    "lmcache coordinator --host 0.0.0.0 --port ${COORDINATOR_PORT}"
wait_http "${NODE_A}" "http://${NODE_A_IP}:${COORDINATOR_PORT}/instances" 60 "coordinator" \
    || die "coordinator never came up (see ${LOGDIR}/roles/coordinator.log)"

lmcache_cmd() {  # $1=instance-id $2=advertise-ip $3=zmq $4=http $5=prom $6=p2p
    cat <<EOF
lmcache server --host 0.0.0.0 --port $3 \
  --http-port $4 \
  --prometheus-port $5 \
  --l1-size-gb ${L1_SIZE_GB} --eviction-policy LRU \
  --l1-align-bytes ${L1_ALIGN_BYTES} \
  --instance-id $1 \
  --coordinator-url http://${NODE_A_IP}:${COORDINATOR_PORT} \
  --coordinator-advertise-ip $2 \
  --p2p-advertise-url $2:$6 \
  --p2p-transfer-engine ${P2P_TRANSFER_ENGINE} \
  --p2p-lookup-timeout ${P2P_LOOKUP_TIMEOUT} \
  --p2p-load-timeout ${P2P_LOAD_TIMEOUT}
EOF
}
log "starting lmcache servers"
exec_role "${NODE_A}" "${CONTAINER_A}" lmcache-a \
    "$(lmcache_cmd node-a "${NODE_A_IP}" "${ZMQ_A}" "${HTTP_A}" "${LMPROM_A}" "${P2PP_A}")"
exec_role "${NODE_B}" "${CONTAINER_B}" lmcache-b \
    "$(lmcache_cmd node-b "${NODE_B_IP}" "${ZMQ_B}" "${HTTP_B}" "${LMPROM_B}" "${P2PP_B}")"
wait_http "${NODE_A}" "http://${NODE_A_IP}:${HTTP_A}/status" 120 "lmcache node-a" \
    || die "lmcache node-a never came up (see ${LOGDIR}/roles/lmcache-a.log)"
wait_http "${NODE_B}" "http://${NODE_B_IP}:${HTTP_B}/status" 120 "lmcache node-b" \
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

vllm_cmd() {  # $1=vllm port $2=lmcache zmq port
    local kv='{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"lmcache.mp.port":'"$2"'}}'
    cat <<EOF
vllm serve ${VLLM_MODEL} --host 0.0.0.0 --port $1 \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --gpu-memory-utilization ${VLM_GPU_MEMORY_UTILIZATION} \
  --max-model-len ${VLM_MAX_MODEL_LEN} \
  --max-num-batched-tokens ${VLM_MAX_NUM_BATCHED_TOKENS} \
  --load-format ${VLM_LOAD_FORMAT} \
  --block-size 64 --enable-prefix-caching \
  --kv-cache-dtype ${VLM_KV_CACHE_DTYPE} \
  --attention-backend ${VLM_ATTENTION_BACKEND} \
  --disable-access-log-for-endpoints "/health,/metrics,/v1/models" \
  --kv-transfer-config '${kv}'
EOF
}
log "starting vLLM for both instances (this loads weights; can take a while)"
exec_role "${NODE_A}" "${CONTAINER_A}" vllm-a "$(vllm_cmd "${VLLM_A}" "${ZMQ_A}")"
exec_role "${NODE_B}" "${CONTAINER_B}" vllm-b "$(vllm_cmd "${VLLM_B}" "${ZMQ_B}")"
wait_http "${NODE_A}" "http://${NODE_A_IP}:${VLLM_A}/v1/models" "${AIC_VLLM_READY_TIMEOUT}" "vLLM node-a" \
    || die "vLLM A never became ready (see ${LOGDIR}/roles/vllm-a.log)"
wait_http "${NODE_B}" "http://${NODE_B_IP}:${VLLM_B}/v1/models" "${AIC_VLLM_READY_TIMEOUT}" "vLLM node-b" \
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
ttft_ms() {  # $1=node $2=ip $3=vllm-port $4=payload-file(host path) -> milliseconds
    local n="$1" ip="$2" port="$3" pf="$4"
    on_node "${n}" "curl -s -o /dev/null --max-time 600 \
        -w '%{time_starttransfer}' \
        -H 'Content-Type: application/json' \
        -X POST --data-binary @'${pf}' \
        http://${ip}:${port}/v1/completions" \
      | awk '{printf "%.1f", $1*1000}'
}

# Exact prompt length via vLLM's /tokenize.  Deliberately NOT a completion call:
# a completion would populate the KV cache for that prompt and destroy whichever
# baseline it was meant to describe.
prompt_tokens() {  # $1=node $2=ip $3=vllm-port $4=tokenize-payload-file
    local n="$1" ip="$2" port="$3" pf="$4"
    on_node "${n}" "curl -s --max-time 60 \
        -H 'Content-Type: application/json' \
        -X POST --data-binary @'${pf}' \
        http://${ip}:${port}/tokenize" \
      | python3 -c "import json,sys
try:
    d = json.load(sys.stdin)
    print(d.get('count') or len(d.get('tokens', [])))
except Exception: print(0)" 2>/dev/null
}

# Issue a completion without timing it (used to warm a prompt into a node's L1).
warm() {  # $1=node $2=ip $3=vllm-port $4=payload-file
    local n="$1" ip="$2" port="$3" pf="$4"
    on_node "${n}" "curl -s -o /dev/null --max-time 600 \
        -H 'Content-Type: application/json' \
        -X POST --data-binary @'${pf}' \
        http://${ip}:${port}/v1/completions" || true
}

capture_status() {  # $1=tag
    local tag="$1"
    on_a "curl -sf --max-time 10 http://${NODE_A_IP}:${HTTP_A}/status" \
        > "${LOGDIR}/status-a-${tag}.json" 2>/dev/null || true
    on_b "curl -sf --max-time 10 http://${NODE_B_IP}:${HTTP_B}/status" \
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
    ntok="$(prompt_tokens "${NODE_A}" "${NODE_A_IP}" "${VLLM_A}" "${LOGDIR}/payloads/isl${isl}-tokenize.json")"

    # (1) cold baseline: prompt X, served once on A.  Neither node has seen X,
    #     and nothing below ever touches it again -> a genuine full prefill.
    ts0="$(now)"
    t_cold="$(ttft_ms "${NODE_A}" "${NODE_A_IP}" "${VLLM_A}" "${LOGDIR}/payloads/isl${isl}-x-stream.json")"
    ts1="$(now)"
    log "  cold prefill  (x  on ${NODE_A}): ${t_cold} ms   prompt_tokens=${ntok}"
    echo "${isl},${ntok},0,cold,${NODE_A},${t_cold},,,,${ts0},${ts1}" >> "${CSV}"

    # (2..) one distinct prompt per repeat: warm on A, fetch on B, then re-serve
    #       on A as the local-hit floor.
    for ((r=1; r<=AIC_P2P_REPEATS; r++)); do
        pw="${LOGDIR}/payloads/isl${isl}-y${r}.json"
        ps="${LOGDIR}/payloads/isl${isl}-y${r}-stream.json"

        warm "${NODE_A}" "${NODE_A_IP}" "${VLLM_A}" "${pw}"
        sleep "${AIC_P2P_SETTLE_S:-5}"   # let the async store + coordinator report land

        # P2P: B has never seen y{r}; avoiding a full prefill requires pulling it
        # from node-a.  IB counters bracket exactly this call.
        ts0="$(now)"
        rx0="$(ib_bytes "${NODE_B}" rcv)"; tx0="$(ib_bytes "${NODE_A}" xmit)"
        t_p2p="$(ttft_ms "${NODE_B}" "${NODE_B_IP}" "${VLLM_B}" "${ps}")"
        rx1="$(ib_bytes "${NODE_B}" rcv)"; tx1="$(ib_bytes "${NODE_A}" xmit)"
        ts1="$(now)"
        d_rx=$(( ${rx1:-0} - ${rx0:-0} ))
        d_tx=$(( ${tx1:-0} - ${tx0:-0} ))
        if (( SINGLE_NODE )); then
            # Both roles read the SAME node's counters, and a same-host UCX
            # transfer does not touch the HCA at all.  Emitting a Gb/s here
            # would be inventing a fabric number out of a co-tenant's traffic.
            gbps=""
            log "  p2p fetch     (y${r} on ${NODE_B}:${VLLM_B}): ${t_p2p} ms   [same-host: IB delta not a fabric measurement]"
        else
            # Implied wire rate over the p2p step.  For scale only, not a link
            # benchmark: the interval also covers the coordinator lookup RTT, the
            # NIXL handshake, and cache bookkeeping.
            gbps="$(awk -v b="${d_rx}" -v ms="${t_p2p}" \
                'BEGIN{ if (ms>0) printf "%.2f", (b*8)/(ms/1000)/1e9; else print "0" }')"
            log "  p2p fetch     (y${r} on ${NODE_B}): ${t_p2p} ms   IB rx(B)=${d_rx} B tx(A)=${d_tx} B  ~${gbps} Gb/s"
        fi
        echo "${isl},${ntok},${r},p2p,${NODE_B},${t_p2p},${d_rx},${d_tx},${gbps},${ts0},${ts1}" >> "${CSV}"

        # Local hit on A for the same prompt: the floor this whole thing is
        # trying to approach.
        ts0="$(now)"
        t_local="$(ttft_ms "${NODE_A}" "${NODE_A_IP}" "${VLLM_A}" "${ps}")"
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
for n_role in "${NODE_A}:a:${LMPROM_A}:${VLLM_A}" "${NODE_B}:b:${LMPROM_B}:${VLLM_B}"; do
    IFS=':' read -r n r _lp _vp <<< "${n_role}"
    on_node "${n}" "curl -sf --max-time 10 http://localhost:${_lp}/metrics" \
        > "${LOGDIR}/lmcache-prom-${r}.txt" 2>/dev/null || true
    on_node "${n}" "curl -sf --max-time 10 http://localhost:${_vp}/metrics" \
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
if (( SINGLE_NODE )); then
    log ""
    log "  NOTE: single-node run.  cold/p2p/local and the speedups above are real"
    log "        -- they measure the coordinator lookup, the NIXL handshake and the"
    log "        KV pull -- but the transfer never crossed the fabric, so ib_rx_MiB"
    log "        and gbps are blank by design.  Re-run on 2 nodes for RDMA numbers."
fi

log ""
log "=== transport evidence ==="
# UCX prints the selected transport at agent init; rc_mlx5/rc_verbs = RDMA, tcp =
# fallback, shm/self = same-host (the expected pick in single-node mode).
for r in a b; do
    _tl="$(grep -ioE 'rc_mlx5|rc_verbs|dc_mlx5|\bshm\b|\bself\b|\btcp\b' "${LOGDIR}/roles/lmcache-${r}.log" 2>/dev/null \
           | sort | uniq -c | sort -rn | head -5 | tr '\n' ' ')"
    log "  lmcache-${r} UCX transports mentioned: ${_tl:-none found (raise AIC_P2P_UCX_LOG_LEVEL)}"
done
_p2pcount="$(grep -c 'P2P transfer finished' "${LOGDIR}/roles/lmcache-b.log" 2>/dev/null || echo 0)"
log "  'P2P transfer finished' lines in lmcache-b.log: ${_p2pcount}"
if (( _p2pcount == 0 )); then
    warn "no P2P transfers recorded on node B -- the 'p2p' column is probably just recompute."
    warn "check ${LOGDIR}/roles/lmcache-b.log and ${LOGDIR}/instances.json"
else
    log "  PASS: instance b served ${_p2pcount} transfer(s) from instance a's cache"
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
