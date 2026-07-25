#!/usr/bin/env bash
set -euo pipefail

# --- Configuration -----------------------------------------------------------
SESSION="lmcache-p2p"

NODE_A_SSH="mi355-21"
NODE_B_SSH="mi355-29"
NODE_A_IP="10.235.192.135"
NODE_B_IP="10.235.192.57"

CONTAINER="lmcache-mp"
DOCKER_IMAGE="rocm-aic:latest"
DOCKER_RUN="docker run -d --rm --name ${CONTAINER} \
  --device=/dev/kfd --device=/dev/dri \
  --network=host \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/opt/nixl/lib/x86_64-linux-gnu \
  --ipc host \
  -v /data:/data \
  --env-file \$HOME/docker.env \
  --entrypoint sleep ${DOCKER_IMAGE} infinity"

COORDINATOR_PORT=9300
LMCACHE_ZMQ_PORT=5555
LMCACHE_HTTP_PORT=8080
P2P_PORT=8500
VLLM_PORT=8000
L1_SIZE_GB=50
MODEL="Qwen/Qwen2.5-3B-Instruct"

DOCKER_EXEC="docker exec ${CONTAINER}"

# --- Helpers -----------------------------------------------------------------
log()  { printf '\033[1;34m>>> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*"; }

wait_for_http() {
    local ssh_host=$1 ip=$2 port=$3 path=${4:-/} timeout=${5:-60}
    log "Waiting for http://${ip}:${port}${path} via ${ssh_host} (${timeout}s timeout)..."
    for ((i=0; i<timeout; i++)); do
        if ssh_cmd "$ssh_host" "curl -sf -o /dev/null http://${ip}:${port}${path}" 2>/dev/null; then
            log "  ...ready"
            return 0
        fi
        sleep 1
    done
    warn "Timed out waiting for http://${ip}:${port}${path}"
    return 1
}

ssh_cmd() { ssh -o ConnectTimeout=5 "$@"; }

# --- Teardown ----------------------------------------------------------------
do_teardown() {
    log "Tearing down..."
    for node in "$NODE_A_SSH" "$NODE_B_SSH"; do
        ssh_cmd "$node" "docker rm -f ${CONTAINER} 2>/dev/null" || true
    done
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    log "Done."
}

# --- Argument handling -------------------------------------------------------
case "${1:-start}" in
    stop|teardown|down|kill)
        do_teardown
        exit 0
        ;;
    start|up)
        ;;
    *)
        echo "Usage: $0 [start|stop]"
        exit 1
        ;;
esac

# --- Pre-flight --------------------------------------------------------------
log "Cleaning up any previous run..."
for node in "$NODE_A_SSH" "$NODE_B_SSH"; do
    ssh_cmd "$node" "docker rm -f ${CONTAINER} 2>/dev/null" || true
done
tmux kill-session -t "$SESSION" 2>/dev/null || true

log "Starting containers..."
for node in "$NODE_A_SSH" "$NODE_B_SSH"; do
    ssh_cmd "$node" "${DOCKER_RUN}"
done

log "Waiting for containers to come up..."
for node in "$NODE_A_SSH" "$NODE_B_SSH"; do
    for ((i=0; i<30; i++)); do
        if ssh_cmd "$node" "docker ps --filter name=${CONTAINER} --format '{{.Names}}'" 2>/dev/null \
            | grep -q "${CONTAINER}"; then
            log "  ${node}: container running"
            break
        fi
        sleep 1
    done
    if ((i == 30)); then
        warn "${node}: container not running after 30s"
        exit 1
    fi
done

# --- Create tmux session -----------------------------------------------------
# Layout:
#   ┌─────────────────────────────────┐
#   │          coordinator            │  pane 0 (top)
#   ├────────────────┬────────────────┤
#   │  lmcache n21   │  lmcache n29   │  pane 1, pane 3
#   ├────────────────┼────────────────┤
#   │   vllm n21     │   vllm n29     │  pane 2, pane 4
#   └────────────────┴────────────────┘

TERM_COLS="${COLUMNS:-$(tput cols 2>/dev/null)}"
TERM_ROWS="${LINES:-$(tput lines 2>/dev/null)}"
: "${TERM_COLS:=200}" "${TERM_ROWS:=50}"
tmux new-session -d -s "$SESSION" -n main -x "$TERM_COLS" -y "$TERM_ROWS"

# Split: coordinator (top 33%) / grid (bottom 67%)
tmux split-window -v -l 67% -t "${SESSION}:main"
# Split bottom left/right (50/50)
tmux split-window -h -l 50% -t "${SESSION}:main.1"
# Split bottom-left top/bottom (pane 1 → 1+2, old right becomes 3)
tmux split-window -v -l 50% -t "${SESSION}:main.1"
# Split bottom-right top/bottom (pane 3 → 3+4)
tmux split-window -v -l 50% -t "${SESSION}:main.3"
# Pane map: 0=coordinator, 1=lmcache-A, 2=vllm-A, 3=lmcache-B, 4=vllm-B

# Pane 0: coordinator
tmux send-keys -t "${SESSION}:main.0" \
    "ssh ${NODE_A_SSH} '${DOCKER_EXEC} lmcache coordinator --host 0.0.0.0 --port ${COORDINATOR_PORT}'" Enter

wait_for_http "$NODE_A_SSH" "$NODE_A_IP" "$COORDINATOR_PORT" "/instances" 30

# Pane 1: lmcache-A
tmux send-keys -t "${SESSION}:main.1" \
    "ssh ${NODE_A_SSH} '${DOCKER_EXEC} lmcache server \
  --host 0.0.0.0 --port ${LMCACHE_ZMQ_PORT} \
  --http-port ${LMCACHE_HTTP_PORT} \
  --l1-size-gb ${L1_SIZE_GB} --eviction-policy LRU \
  --l1-align-bytes 65536 \
  --instance-id node-a \
  --coordinator-url http://${NODE_A_IP}:${COORDINATOR_PORT} \
  --coordinator-advertise-ip ${NODE_A_IP} \
  --p2p-advertise-url ${NODE_A_IP}:${P2P_PORT}'" Enter

# Pane 3: lmcache-B
tmux send-keys -t "${SESSION}:main.3" \
    "ssh ${NODE_B_SSH} '${DOCKER_EXEC} lmcache server \
  --host 0.0.0.0 --port ${LMCACHE_ZMQ_PORT} \
  --http-port ${LMCACHE_HTTP_PORT} \
  --l1-size-gb ${L1_SIZE_GB} --eviction-policy LRU \
  --l1-align-bytes 65536 \
  --instance-id node-b \
  --coordinator-url http://${NODE_A_IP}:${COORDINATOR_PORT} \
  --coordinator-advertise-ip ${NODE_B_IP} \
  --p2p-advertise-url ${NODE_B_IP}:${P2P_PORT}'" Enter

wait_for_http "$NODE_A_SSH" "$NODE_A_IP" "$LMCACHE_HTTP_PORT" "/status" 60
wait_for_http "$NODE_B_SSH" "$NODE_B_IP" "$LMCACHE_HTTP_PORT" "/status" 60

# Pane 2: vllm-A
tmux send-keys -t "${SESSION}:main.2" \
    "ssh ${NODE_A_SSH} '${DOCKER_EXEC} vllm serve ${MODEL} \
  --port ${VLLM_PORT} \
  --kv-transfer-config '\"'\"'{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":${LMCACHE_ZMQ_PORT}}}'\"'\"''" Enter

# Pane 4: vllm-B
tmux send-keys -t "${SESSION}:main.4" \
    "ssh ${NODE_B_SSH} '${DOCKER_EXEC} vllm serve ${MODEL} \
  --port ${VLLM_PORT} \
  --kv-transfer-config '\"'\"'{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":${LMCACHE_ZMQ_PORT}}}'\"'\"''" Enter

# --- Done --------------------------------------------------------------------
log "Launched all services in tmux session '${SESSION}'."
log ""
log "  ┌─────────────────────────────────┐"
log "  │       lmcache coordinator       │  pane 0"
log "  ├────────────────┬────────────────┤"
log "  │ lmcache node A │ lmcache node B │  pane 1, pane 3"
log "  ├────────────────┼────────────────┤"
log "  │  vllm node A   │  vllm node B   │  pane 2, pane 4"
log "  └────────────────┴────────────────┘"
log "  node A = ${NODE_A_SSH} (${NODE_A_IP})"
log "  node B = ${NODE_B_SSH} (${NODE_B_IP})"
log ""
log "Attach with:    tmux attach -t ${SESSION}"
log "Navigate:       Ctrl-b arrow keys"
log "Teardown with:  $0 stop"
log ""
BENCH_CMD="vllm bench serve --backend openai --dataset-name random --model ${MODEL} --num-prompts 1 --input-len $((127*256)) --random-output-len 1 --seed 0"
log "Once vLLM is ready, test P2P:"
log "  # Warm node-a"
log "  ssh ${NODE_A_SSH} '${DOCKER_EXEC} ${BENCH_CMD} --base-url http://${NODE_A_IP}:${VLLM_PORT}'"
log "  # Hit node-b (should P2P-fetch from node-a)"
log "  ssh ${NODE_B_SSH} '${DOCKER_EXEC} ${BENCH_CMD} --base-url http://${NODE_B_IP}:${VLLM_PORT}'"
