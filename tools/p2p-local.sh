#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# p2p-local.sh -- the simplest possible LMCache P2P test.
#
# N+1 docker containers on ONE machine: one coordinator, N lmcache servers.
# Answers two questions and nothing else:
#
#   1. CAN THEY SEE EACH OTHER?  Every server registers with the coordinator,
#      the coordinator lists them all, and each server reports p2p_peer_count
#      == N-1 (it built a P2P L2 adapter for every peer).
#   2. CAN THEY MOVE VECTORS?    One container registers a buffer of known
#      bytes and another reads it back over the same nixl/UCX transfer channel
#      the real P2P path uses, with --verify checking every byte.
#
# What this deliberately does NOT need:
#
#   GPU     -- nothing in the KV path between two LMCache instances touches one.
#              L1 is host DRAM (POSIX shm); a peer fetch is DRAM -> DRAM.  The
#              GPU only ever existed so vLLM could run a model and give a TTFT.
#   CX7/IB  -- an RDMA NIC is only needed to prove bytes crossed a FABRIC.  Two
#              containers on one host never reach the NIC; UCX picks shm/self.
#   NVMe    -- there is no L2 disk tier here at all.  L1 is pure DRAM.  (The
#              NVME constraint on the Slurm jobs is inherited from the cliff
#              work and is dead weight for P2P.)
#   Slurm   -- runs anywhere docker does, including a login node.
#
# So the only requirements are: docker, this image, and some RAM.
#
# Usage:
#   tools/p2p-local.sh                 # 2 servers + coordinator, full test
#   tools/p2p-local.sh -n 4            # 4 servers (tests N-way discovery)
#   tools/p2p-local.sh --see-only      # discovery check, skip the transfer
#   tools/p2p-local.sh --keep          # leave the containers up to poke at
#   tools/p2p-local.sh --down          # tear down a --keep run and exit
#
# Env:
#   AIC_IMAGE          image to run                (default rocm-aic:latest)
#   AIC_P2P_L1_SIZE_GB per-server DRAM L1          (default 4)
#   AIC_P2P_OBJ_SIZE   transfer object size        (default 4MB)
#   AIC_P2P_NUM_OBJECTS objects per read           (default 64)
#   AIC_P2P_ITERS      measured read iterations    (default 5)
#   AIC_LOGDIR         where logs land             (default ./logs/p2p-local)

set -uo pipefail

IMAGE="${AIC_IMAGE:-rocm-aic:latest}"
NSERVERS=2
SEE_ONLY=0
KEEP=0
DOWN_ONLY=0

PREFIX="aic-p2p-local"
L1_SIZE_GB="${AIC_P2P_L1_SIZE_GB:-4}"
OBJ_SIZE="${AIC_P2P_OBJ_SIZE:-4MB}"
NUM_OBJECTS="${AIC_P2P_NUM_OBJECTS:-64}"
ITERS="${AIC_P2P_ITERS:-5}"
PAGE_SIZE="${AIC_P2P_PAGE_SIZE:-524288}"
BUFFER_SIZE="${AIC_P2P_BUFFER_SIZE:-4GB}"

# Base ports.  Server i gets base + i*STRIDE, so servers never collide and the
# whole block is easy to shift out of the way of a co-tenant.
COORD_PORT="${AIC_P2P_COORD_PORT:-19300}"
ZMQ_BASE="${AIC_P2P_ZMQ_BASE:-15555}"
HTTP_BASE="${AIC_P2P_HTTP_BASE:-18080}"
P2P_BASE="${AIC_P2P_P2P_BASE:-18500}"
XFER_BASE="${AIC_P2P_XFER_BASE:-17600}"
STRIDE=10

# PID namespace shared by every peer.  See the run_container comment for why
# this is mandatory.  shared = join a dedicated holder container's namespace
# (default, no host PID exposure); host = --pid host; none = per-container
# namespaces, which is the configuration that FAILS.  `none` exists only so the
# failure can be reproduced on demand.
PIDNS_MODE="${AIC_P2P_PIDNS:-shared}"
PIDNS_HOLDER="${PREFIX}-pidns"
case "${PIDNS_MODE}" in
    shared|host|none) ;;
    *) echo "ERROR: AIC_P2P_PIDNS must be shared|host|none (got '${PIDNS_MODE}')" >&2
       exit 2 ;;
esac

while (( $# )); do
    case "$1" in
        -n|--servers)  NSERVERS="$2"; shift 2 ;;
        --see-only)    SEE_ONLY=1; shift ;;
        --keep)        KEEP=1; shift ;;
        --down)        DOWN_ONLY=1; shift ;;
        -h|--help)     sed -n '8,48p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
    esac
done

LOGDIR="${AIC_LOGDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/logs/p2p-local}"

log()  { printf '\033[1;34m>>> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  PASS\033[0m %s\n' "$*"; }
bad()  { printf '\033[1;31m  FAIL\033[0m %s\n' "$*"; FAILURES=$((FAILURES + 1)); }
warn() { printf '\033[1;33m  WARN\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

FAILURES=0

srv_name() { printf '%s-%s' "${PREFIX}" "$(printf "\\$(printf '%03o' $((97 + $1)))")"; }  # 0->a
srv_zmq()  { echo $(( ZMQ_BASE  + $1 * STRIDE )); }
srv_http() { echo $(( HTTP_BASE + $1 * STRIDE )); }
srv_p2p()  { echo $(( P2P_BASE  + $1 * STRIDE )); }
srv_xfer() { echo $(( XFER_BASE + $1 * STRIDE )); }

teardown() {
    local c
    # Joiners first, PID-namespace holder last: removing the owner out from
    # under a live joiner is the one ordering docker complains about.
    for c in $(docker ps -aq --filter "name=^${PREFIX}-" 2>/dev/null); do
        [[ "$(docker inspect -f '{{.Name}}' "${c}" 2>/dev/null)" == "/${PIDNS_HOLDER}" ]] \
            && continue
        docker rm -f "${c}" >/dev/null 2>&1
    done
    docker rm -f "${PIDNS_HOLDER}" >/dev/null 2>&1
}

if (( DOWN_ONLY )); then
    log "tearing down any ${PREFIX}-* containers"
    teardown
    log "done."
    exit 0
fi

docker image inspect "${IMAGE}" >/dev/null 2>&1 \
    || die "image ${IMAGE} not present. Load it first, e.g.
    zstd -dc \$HOME/images/rocm-aic-latest-*.tar.zst | docker load"

mkdir -p "${LOGDIR}"
rm -f "${LOGDIR}"/*.log 2>/dev/null

log "LMCache P2P -- ${NSERVERS} servers + 1 coordinator, all local containers"
log "  image  ${IMAGE}"
log "  host   $(hostname)  ($(nproc) cpus, $(free -g | awk 'NR==2{print $2}') GB ram)"
log "  logs   ${LOGDIR}"
echo

log "removing any leftovers from a previous run"
teardown

# --- Containers --------------------------------------------------------------
# --network host: every role talks over 127.0.0.1, exactly like LMCache's own
#   multiprocess CI test -- no bridge NAT in the middle of the measurement.
# --ipc host:     REQUIRED.  The L1 cache is POSIX shared memory and nixl/UCX
#   moves bytes between containers through it.  Without a shared IPC namespace
#   each container gets a private /dev/shm and the peers cannot reach each
#   other's buffers -- this is the single most common way this test "mysteriously"
#   fails.
# Shared PID namespace: REQUIRED, and the subtler of the two.  On a same-host
#   pair UCX has no RDMA NIC to use, so it selects `cma/memory` -- Cross Memory
#   Attach, i.e. process_vm_readv() straight into the peer process's address
#   space.  That call takes a *PID*, and a PID is only meaningful inside one PID
#   namespace.  Give each container its own namespace and the peer advertises a
#   PID the reader cannot resolve, so NIXL fails the handshake with
#     loadRemoteMD: error loading connection info for backend 'UCX'
#     ... NIXL_ERR_BACKEND
#   and the connecting side just sits there until its 60s handshake timeout.
#   Nothing in that message mentions PIDs, which makes it a nasty one to chase.
#   Measured on this host: shared namespace -> 11.3 GB/s, verify OK; private
#   namespaces -> handshake timeout, 100% of the time.
#   Note that forcing UCX_TLS=tcp is NOT a workaround: UCX then enumerates every
#   docker veth on the box (dozens, on a shared login node) and none of those
#   bridge legs are mutually routable, so it hangs a different way.
# No --device flags at all: no /dev/kfd, no /dev/dri, no /dev/infiniband.
run_container() {  # $1=name
    local _pidns=()
    case "${PIDNS_MODE}" in
        shared) _pidns=(--pid "container:${PIDNS_HOLDER}") ;;
        host)   _pidns=(--pid host) ;;
        none)   _pidns=() ;;
    esac
    docker run -d --rm --name "$1" \
        --network host --ipc host "${_pidns[@]}" \
        --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 \
        -v "${LOGDIR}:/joblog" \
        -e LD_LIBRARY_PATH=/opt/rocm/lib:/opt/nixl/lib/x86_64-linux-gnu \
        -e UCX_LOG_LEVEL="${AIC_P2P_UCX_LOG_LEVEL:-info}" \
        --entrypoint sleep "${IMAGE}" infinity >/dev/null \
        || die "failed to start container $1"
}

exec_bg() {  # $1=container $2=logname $3=command
    { echo '#!/usr/bin/env bash'; echo 'set -uo pipefail'; echo "$3"; } > "${LOGDIR}/$2.sh"
    chmod +x "${LOGDIR}/$2.sh"
    docker exec -d "$1" bash -c "bash /joblog/$2.sh > /joblog/$2.log 2>&1"
}

wait_url() {  # $1=url $2=timeout $3=label
    local waited=0
    while (( waited < $2 )); do
        curl -sf -o /dev/null --max-time 3 "$1" && { ok "$3 up (${waited}s)"; return 0; }
        sleep 2; waited=$((waited + 2))
    done
    bad "$3 never came up ($1)"
    return 1
}

(( KEEP )) || trap teardown EXIT

log "starting ${NSERVERS} server containers + coordinator"
# The PID-namespace holder must exist before anything can join it.  It is a bare
# `sleep`, owns nothing, and exists purely so the peers share one PID namespace
# without being handed the host's.
if [[ "${PIDNS_MODE}" == "shared" ]]; then
    docker run -d --rm --name "${PIDNS_HOLDER}" \
        --entrypoint sleep "${IMAGE}" infinity >/dev/null \
        || die "failed to start PID-namespace holder ${PIDNS_HOLDER}"
    log "  pid namespace: shared via ${PIDNS_HOLDER} (host PIDs not exposed)"
else
    log "  pid namespace: ${PIDNS_MODE}"
fi
run_container "${PREFIX}-coord"
for ((i=0; i<NSERVERS; i++)); do run_container "$(srv_name "${i}")"; done
docker ps --filter "name=^${PREFIX}-" --format '  {{.Names}}' | sort

# --- Coordinator -------------------------------------------------------------
echo
log "step 1/3: coordinator"
exec_bg "${PREFIX}-coord" coordinator \
    "lmcache coordinator --host 0.0.0.0 --port ${COORD_PORT}"
wait_url "http://127.0.0.1:${COORD_PORT}/healthz" 60 "coordinator :${COORD_PORT}" \
    || die "coordinator failed; see ${LOGDIR}/coordinator.log"

# --- Servers -----------------------------------------------------------------
echo
log "step 2/3: ${NSERVERS} lmcache servers -- can they see each other?"
for ((i=0; i<NSERVERS; i++)); do
    name="$(srv_name "${i}")"
    exec_bg "${name}" "${name}" "$(cat <<EOF
lmcache server --host 0.0.0.0 --port $(srv_zmq "${i}") \
  --http-port $(srv_http "${i}") \
  --l1-size-gb ${L1_SIZE_GB} --eviction-policy LRU \
  --instance-id ${name} \
  --coordinator-url http://127.0.0.1:${COORD_PORT} \
  --coordinator-advertise-ip 127.0.0.1 \
  --p2p-advertise-url 127.0.0.1:$(srv_p2p "${i}") \
  --p2p-transfer-engine nixl
EOF
)"
done

for ((i=0; i<NSERVERS; i++)); do
    wait_url "http://127.0.0.1:$(srv_http "${i}")/status" 180 "$(srv_name "${i}") :$(srv_http "${i}")"
done
(( FAILURES == 0 )) || die "not all servers came up; see ${LOGDIR}/*.log"

# Registration is asynchronous: the servers heartbeat into the coordinator and
# then discover each other. Poll rather than sleep a magic number.
echo
log "waiting for peer discovery (each server should see $(( NSERVERS - 1 )) peer(s))"
waited=0; discovered=0
while (( waited < 120 )); do
    allgood=1
    for ((i=0; i<NSERVERS; i++)); do
        n="$(curl -sf --max-time 5 "http://127.0.0.1:$(srv_http "${i}")/status" 2>/dev/null \
             | python3 -c 'import json,sys; print(json.load(sys.stdin).get("p2p_peer_count",0))' 2>/dev/null)"
        [[ "${n:-0}" == "$(( NSERVERS - 1 ))" ]] || allgood=0
    done
    (( allgood )) && { discovered=1; break; }
    sleep 3; waited=$((waited + 3))
done

curl -sf --max-time 10 "http://127.0.0.1:${COORD_PORT}/instances" > "${LOGDIR}/instances.json" 2>/dev/null
registered=0
for ((i=0; i<NSERVERS; i++)); do
    grep -q "$(srv_name "${i}")" "${LOGDIR}/instances.json" 2>/dev/null && registered=$((registered + 1))
done
if (( registered == NSERVERS )); then
    ok "coordinator lists all ${NSERVERS} instances (${LOGDIR}/instances.json)"
else
    bad "coordinator lists ${registered}/${NSERVERS} instances"
fi

for ((i=0; i<NSERVERS; i++)); do
    n="$(curl -sf --max-time 5 "http://127.0.0.1:$(srv_http "${i}")/status" 2>/dev/null \
         | python3 -c 'import json,sys; print(json.load(sys.stdin).get("p2p_peer_count",0))' 2>/dev/null)"
    if [[ "${n:-0}" == "$(( NSERVERS - 1 ))" ]]; then
        ok "$(srv_name "${i}") sees ${n} peer(s)"
    else
        bad "$(srv_name "${i}") sees ${n:-0} peer(s), expected $(( NSERVERS - 1 ))"
    fi
done
(( discovered )) || warn "discovery timed out after ${waited}s"

# --- Vector movement ---------------------------------------------------------
if (( SEE_ONLY )); then
    echo
    log "--see-only: skipping the transfer test"
else
    echo
    log "step 3/3: moving vectors between containers ($(srv_name 0) -> $(srv_name 1))"
    log "  ${NUM_OBJECTS} x ${OBJ_SIZE} objects over nixl/UCX, every byte verified"

    # Server side registers a host buffer full of a known per-object pattern and
    # publishes its catalog; the client reads a random subset and checks the
    # bytes.  Same transfer channel the P2P L2 adapter uses.
    exec_bg "$(srv_name 0)" xfer-server "$(cat <<EOF
lmcache tool transfer-channel-benchmark --role server \
  --transfer-channel-type nixl --nixl-backend UCX \
  --url 127.0.0.1:$(srv_xfer 0) \
  --control-url 0.0.0.0:$(( $(srv_xfer 0) + 1 )) \
  --buffer-size ${BUFFER_SIZE} --page-size ${PAGE_SIZE} \
  --object-size ${OBJ_SIZE} --num-objects ${NUM_OBJECTS} \
  --server-timeout 300
EOF
)"

    waited=0; ready=0
    while (( waited < 120 )); do
        grep -q '\[server\] ready' "${LOGDIR}/xfer-server.log" 2>/dev/null && { ready=1; break; }
        sleep 2; waited=$((waited + 2))
    done
    if (( ! ready )); then
        bad "transfer-channel server never became ready (${LOGDIR}/xfer-server.log)"
    else
        ok "transfer-channel server ready in $(srv_name 0) (${waited}s)"
        { echo '#!/usr/bin/env bash'; echo 'set -uo pipefail'; cat <<EOF
lmcache tool transfer-channel-benchmark --role client \
  --transfer-channel-type nixl --nixl-backend UCX \
  --url 127.0.0.1:$(srv_xfer 0) \
  --listen-url 127.0.0.1:$(srv_xfer 1) \
  --control-url 127.0.0.1:$(( $(srv_xfer 0) + 1 )) \
  --page-size ${PAGE_SIZE} --object-size ${OBJ_SIZE} \
  --num-objects ${NUM_OBJECTS} --iters ${ITERS} --warmup 1 --verify
EOF
        } > "${LOGDIR}/xfer-client.sh"
        chmod +x "${LOGDIR}/xfer-client.sh"

        echo
        docker exec "$(srv_name 1)" bash -c 'bash /joblog/xfer-client.sh 2>&1' \
            | tee "${LOGDIR}/xfer-client.log"
        rc=${PIPESTATUS[0]}
        echo

        if (( rc != 0 )); then
            bad "transfer client exited ${rc}"
        elif ! grep -q 'verify OK' "${LOGDIR}/xfer-client.log"; then
            bad "transfer completed but the client never reported 'verify OK'"
        # Filter the two benign UCX DIAG lines emitted on any box without an
        # RDMA NIC before looking for real trouble.  They contain the word
        # "error" and are unrelated to the transfer:
        #   rdma_create_event_channel failed: No such device
        #   failed to open CM on component rdmacm with status Input/output error
        elif grep -viE 'rdma_create_event_channel|open CM on component rdmacm' \
                "${LOGDIR}/xfer-client.log" \
             | grep -qiE 'mismatch|traceback|error'; then
            bad "transfer completed but the log reports an error/mismatch"
        elif grep -q 'read throughput' "${LOGDIR}/xfer-client.log"; then
            ok "vectors moved $(srv_name 0) -> $(srv_name 1) and verified byte-for-byte"
        else
            bad "no throughput report in the client log"
        fi

        tl="$(grep -hioE 'rc_mlx5|rc_verbs|dc_mlx5|\bshm\b|\bself\b|\btcp\b' \
              "${LOGDIR}/xfer-client.log" "${LOGDIR}/xfer-server.log" 2>/dev/null \
              | sort | uniq -c | sort -rn | head -4 | tr '\n' ' ')"
        log "  UCX transports mentioned: ${tl:-none (raise AIC_P2P_UCX_LOG_LEVEL)}"
        log "  (shm/self is CORRECT here -- same host.  rc_mlx5 would mean RDMA,"
        log "   which only happens when the peers are on different machines.)"
    fi
fi

# --- Verdict -----------------------------------------------------------------
echo
if (( FAILURES == 0 )); then
    log "ALL CHECKS PASSED -- ${NSERVERS} containers registered, discovered each other,"
    log "and moved verified data between them.  No GPU, no RDMA NIC, no NVMe, no Slurm."
else
    log "${FAILURES} CHECK(S) FAILED -- see ${LOGDIR}/"
fi

if (( KEEP )); then
    echo
    log "--keep: containers left running.  Poke at them:"
    for ((i=0; i<NSERVERS; i++)); do
        log "  curl -s http://127.0.0.1:$(srv_http "${i}")/status | python3 -m json.tool"
    done
    log "  curl -s http://127.0.0.1:${COORD_PORT}/instances | python3 -m json.tool"
    log "  tear down with: $0 --down"
fi

exit $(( FAILURES > 0 ))
