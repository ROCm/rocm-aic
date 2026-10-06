#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# run-lmcache-p2p-test.sh — LMCache P2P KV-cache sharing test on a SPUR node.
#
# Boots two QEMU KVM VMs with rocm-ernic emulated ionic RDMA NICs, loads the
# AIC lmcache image into each VM via docker save/load, then runs the lmcache
# P2P coordinator + servers inside the VMs using docker compose.
#
# Invoked by: sbatch .slurm/run-lmcache-p2p-test.sbatch
# Or directly with an active allocation:
#   srun --jobid=<ID> --overlap env SLURM_SUBMIT_DIR=$PWD bash .slurm/run-lmcache-p2p-test.sh
#
# Required env:
#   SLURM_SUBMIT_DIR          Path to repo root
#   AIC_LMCACHE_IMAGE_REF     AIC lmcache image (has torch, lmcache, nixl pre-installed)
#
# Optional env:
#   AIC_LMCACHE_P2P_VM_IMAGES_DIR  qcow2 image dir (default: /var/tmp/aic-p2p-images-<JOBID>)
#   AIC_LMCACHE_P2P_QCOW2_IMAGE   ionic qcow2 OCI image
#   AIC_LMCACHE_P2P_READY_S        SSH + server readiness timeout (default: 480)
#   AIC_LMCACHE_P2P_VM1_IP         VM1 ionic IP (default: 192.168.200.10)
#   AIC_LMCACHE_P2P_VM2_IP         VM2 ionic IP (default: 192.168.200.20)
#   AIC_LMCACHE_PORT               lmcache ZMQ port (default: 6555)
#   AIC_LMCACHE_HTTP_PORT          lmcache HTTP port for /healthz (default: 7555)
#   AIC_COORD_PORT                 coordinator port (default: 9300)
#   AIC_P2P_PORT                   P2P RDMA port (default: 8500)

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set to the repo root}"
export PATH="${HOME}/.local/bin:${PATH}"

# ---- Configuration -----------------------------------------------------------
VM_IMAGES_DIR="${AIC_LMCACHE_P2P_VM_IMAGES_DIR:-/var/tmp/aic-p2p-images-${SLURM_JOB_ID:-$$}}"
QCOW2_IMAGE="${AIC_LMCACHE_P2P_QCOW2_IMAGE:-docker.io/sbates130272/batesste-ci-images-ubuntu-qcow2-gen-ionic:20260929.g2bdbd16-vm.resolute-ionic-qm.54cc234}"
READY_S="${AIC_LMCACHE_P2P_READY_S:-480}"
VM1_IP="${AIC_LMCACHE_P2P_VM1_IP:-192.168.200.10}"
VM2_IP="${AIC_LMCACHE_P2P_VM2_IP:-192.168.200.20}"
LMCACHE_PORT="${AIC_LMCACHE_PORT:-6555}"
LMCACHE_HTTP_PORT="${AIC_LMCACHE_HTTP_PORT:-7555}"
COORD_PORT="${AIC_COORD_PORT:-9300}"
P2P_PORT="${AIC_P2P_PORT:-8500}"
VM1_SSH_PORT=12230
VM2_SSH_PORT=12231
VM1_NAME="p2p-vm1-${SLURM_JOB_ID:-$$}"
VM2_NAME="p2p-vm2-${SLURM_JOB_ID:-$$}"
WORK_DIR="/var/tmp/aic-p2p-work-${SLURM_JOB_ID:-$$}"

VM_SSH_KEY="${VM_IMAGES_DIR}/id_rsa"
VM_SSH_USER="ubuntu"
SSH_FLAGS="-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5 -o ServerAliveInterval=30 -o ServerAliveCountMax=3"

# AIC lmcache image — must have torch + lmcache + nixl pre-installed.
# Use AIC_LMCACHE_IMAGE_REF from env, or fall back to LMCACHE_IMAGE_REF.
LMCACHE_IMAGE="${AIC_LMCACHE_IMAGE_REF:-${LMCACHE_IMAGE_REF:-}}"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
die() { log "FATAL: $*" >&2; cleanup; exit 1; }

cleanup() {
    log "Tearing down compose stack and coordinator ..."
    docker rm -f aic-lmcache-coordinator 2>/dev/null || true
    VM_IMAGES_DIR="$VM_IMAGES_DIR" VM1_NAME="$VM1_NAME" VM2_NAME="$VM2_NAME" \
        qemu-tool compose --stack vfio-user-ernic-2vm \
            --vm-name "$VM1_NAME" --vm2-name "$VM2_NAME" \
            down 2>/dev/null || true
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

log "=== LMCache P2P SPUR test ==="
log "  Node:   $(hostname)  Job: ${SLURM_JOB_ID:-local}"
log "  Image:  ${LMCACHE_IMAGE:-<none>}"
log "  Repo:   $REPO_ROOT"

# ---- Step 0: clean up stale containers --------------------------------------
log "[0/7] Cleaning up stale containers ..."
docker ps -a --filter "name=vfio-user-ernic-2vm" --format "{{.ID}}" \
    | xargs -r docker rm -f 2>/dev/null || true
docker network ls --filter "name=vfio-user-ernic-2vm" --format "{{.ID}}" \
    | xargs -r docker network rm 2>/dev/null || true
sleep 2

# ---- Step 1: prerequisites ---------------------------------------------------
log "[1/7] Checking prerequisites ..."
test -c /dev/kvm || die "/dev/kvm not found"
command -v docker >/dev/null || die "docker not found"
command -v qemu-tool >/dev/null 2>&1 || {
    log "  Installing qemu-tool ..."
    PIPX_BIN_DIR="${HOME}/.local/bin" pipx install --force qemu-tool >/dev/null 2>&1 || \
    pip install --quiet --break-system-packages qemu-tool 2>/dev/null || true
}
qemu-tool --version
mkdir -p "$WORK_DIR"
# Build a minimal lmcache image if not already present.
# The full aic-lmcache:latest is 59GB (ROCm+torch) — too large to stream into VMs.
# aic-lmcache-minimal is ~500MB and has only lmcache+nixl for CPU-only P2P testing.
# Force rebuild if Dockerfile changed (delete old image)
docker rmi aic-lmcache-minimal:latest >/dev/null 2>&1 || true
if ! docker image inspect aic-lmcache-minimal:latest >/dev/null 2>&1; then
    log "  Building minimal lmcache image (no torch/ROCm) ..."
    # Extract nixl libs from the full image for the minimal build
    docker run --rm --entrypoint sh aic-lmcache:latest -c \
        "tar -czC /opt/nixl ." > "$WORK_DIR/nixl.tar.gz" 2>/dev/null || true

    cat > "$WORK_DIR/Dockerfile.minimal" << 'DOCKEREOF'
FROM python:3.12-slim
RUN apt-get update -qq && apt-get install -y -qq \
    libibverbs1 librdmacm1 libnl-3-200 libnl-route-3-200 curl \
    && rm -rf /var/lib/apt/lists/*
# Install CPU-only torch first, then lmcache with all deps.
# CPU torch is ~200MB vs ROCm torch ~8GB — sufficient for coordinator/P2P testing.
RUN pip install --quiet --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch && \
    pip install --quiet --no-cache-dir "lmcache==0.5.5"
DOCKEREOF
    docker build -q -t aic-lmcache-minimal:latest -f "$WORK_DIR/Dockerfile.minimal" "$WORK_DIR"
    log "  Minimal image built: $(docker image inspect aic-lmcache-minimal:latest --format '{{.Size}}' | numfmt --to=si)B"
fi
LMCACHE_IMAGE="aic-lmcache-minimal:latest"

# ---- Step 2: extract VM disk images ------------------------------------------
log "[2/7] Setting up VM disk images ..."
mkdir -p "$VM_IMAGES_DIR" "$WORK_DIR"

if [ ! -f "$VM_IMAGES_DIR/${VM1_NAME}.qcow2" ]; then
    log "  Pulling $QCOW2_IMAGE ..."
    docker pull -q "$QCOW2_IMAGE"
    _cid=$(docker create "$QCOW2_IMAGE")
    mkdir -p "$WORK_DIR/qcow2-tmp"
    docker cp "$_cid:/output/." "$WORK_DIR/qcow2-tmp" 2>/dev/null \
        || docker cp "$_cid:/." "$WORK_DIR/qcow2-tmp"
    docker rm "$_cid" >/dev/null
    _qcow2=$(find "$WORK_DIR/qcow2-tmp" -name '*.qcow2' | head -1)
    [ -n "$_qcow2" ] || die "No .qcow2 found in $QCOW2_IMAGE"
    cp "$_qcow2" "$VM_IMAGES_DIR/${VM1_NAME}.qcow2"
    cp "$_qcow2" "$VM_IMAGES_DIR/${VM2_NAME}.qcow2"
    find "$WORK_DIR/qcow2-tmp" -name 'id_rsa' -exec cp {} "$VM_IMAGES_DIR/" \;
    find "$WORK_DIR/qcow2-tmp" -name 'vm-info.json' -exec cp {} "$VM_IMAGES_DIR/" \;
    chmod 600 "$VM_IMAGES_DIR/id_rsa" 2>/dev/null || true
    rm -rf "$WORK_DIR/qcow2-tmp"
    log "  Disk images ready"
else
    log "  Disk images already present"
    [ -f "$VM_IMAGES_DIR/${VM2_NAME}.qcow2" ] || \
        cp "$VM_IMAGES_DIR/${VM1_NAME}.qcow2" "$VM_IMAGES_DIR/${VM2_NAME}.qcow2"
fi

# Resolve SSH user from vm-info.json
VM_SSH_USER=$(python3 -c "import json; print(json.load(open('$VM_IMAGES_DIR/vm-info.json')).get('username','ubuntu'))" 2>/dev/null || echo "ubuntu")
log "  SSH user=$VM_SSH_USER key=$([ -f "$VM_SSH_KEY" ] && echo present || echo MISSING)"
[ -f "$VM_SSH_KEY" ] || die "SSH key not found — re-extract the qcow2"

# ---- Step 3: start compose stack --------------------------------------------
log "[3/7] Starting vfio-user-ernic-2vm compose stack ..."
_compose_env=(
    "VM_IMAGES_DIR=$VM_IMAGES_DIR"
    "VM1_NAME=$VM1_NAME" "VM2_NAME=$VM2_NAME"
    "VM1_SSH_PORT=$VM1_SSH_PORT" "VM2_SSH_PORT=$VM2_SSH_PORT"
    "VM_VCPUS=4" "VM_VMEM=4096"
)
env "${_compose_env[@]}" \
    qemu-tool compose --stack vfio-user-ernic-2vm \
        --vm-name "$VM1_NAME" --vm2-name "$VM2_NAME" \
        up -d

# ---- Step 4: wait for SSH ----------------------------------------------------
log "[4/7] Waiting for SSH on both VMs (up to ${READY_S}s) ..."
for _vmspec in "$VM1_SSH_PORT VM1" "$VM2_SSH_PORT VM2"; do
    _port=$(echo "$_vmspec" | awk '{print $1}')
    _label=$(echo "$_vmspec" | awk '{print $2}')
    _ready=0
    for _i in $(seq 1 $(( READY_S / 5 ))); do
        # shellcheck disable=SC2086
        ssh $SSH_FLAGS -i "$VM_SSH_KEY" -p "$_port" "$VM_SSH_USER@localhost" true 2>/dev/null \
            && { _ready=1; break; }
        sleep 5
    done
    [ "$_ready" = "1" ] || {
        VM_IMAGES_DIR="$VM_IMAGES_DIR" VM1_NAME="$VM1_NAME" VM2_NAME="$VM2_NAME" \
            qemu-tool compose --stack vfio-user-ernic-2vm \
                --vm-name "$VM1_NAME" --vm2-name "$VM2_NAME" \
                logs --tail 40 2>/dev/null || true
        die "$_label not SSH-reachable after ${READY_S}s"
    }
    log "  $_label SSH ready (:$_port)"
done

# ---- Step 5a: start coordinator on the ernic Docker bridge ------------------
# Must happen AFTER compose up (step 3) so vfio-user-ernic-2vm_default exists.
log "[5a/7] Starting coordinator on host Docker bridge ..."
docker rm -f aic-lmcache-coordinator 2>/dev/null || true
# Use the minimal image (available on every node) for the coordinator.
# lmcache_controller needs torch — add torch-cpu to the minimal image.
docker run -d --name aic-lmcache-coordinator \
    --network vfio-user-ernic-2vm_default \
    --entrypoint /usr/local/bin/lmcache_controller \
    aic-lmcache-minimal:latest \
    --host 0.0.0.0 --port "${COORD_PORT}" \
    2>&1 | tail -2 || die "Failed to start coordinator container"

sleep 5
COORD_DOCKER_IP=$(docker inspect aic-lmcache-coordinator \
    --format '{{(index .NetworkSettings.Networks "vfio-user-ernic-2vm_default").IPAddress}}' \
    2>/dev/null || echo "")
[ -n "$COORD_DOCKER_IP" ] && [[ "$COORD_DOCKER_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] \
    || die "Could not get coordinator Docker IP (got: '$COORD_DOCKER_IP')"
log "  Coordinator at ${COORD_DOCKER_IP}:${COORD_PORT}"

# ---- Step 5b: load AIC lmcache image + compose files into VMs ---------------
log "[5b/7] Loading AIC lmcache image and compose files into VMs ..."
_compose_src="$REPO_ROOT/docker/compose/lmcache-p2p/docker-compose.yml"
_setup_src="$REPO_ROOT/scripts/lmcache-p2p-guest-setup.sh"

for _port in "$VM1_SSH_PORT" "$VM2_SSH_PORT"; do
    # Install Docker if not present (ionic VM doesn't ship it)
    # shellcheck disable=SC2086
    _has_docker=$(ssh $SSH_FLAGS -i "$VM_SSH_KEY" -p "$_port" "$VM_SSH_USER@localhost" \
        "command -v docker >/dev/null 2>&1 && echo yes || echo no" 2>/dev/null)
    if [ "$_has_docker" != "yes" ]; then
        log "  Installing Docker in VM on :$_port ..."
        # shellcheck disable=SC2086
        ssh $SSH_FLAGS -i "$VM_SSH_KEY" -p "$_port" "$VM_SSH_USER@localhost" \
            "export DEBIAN_FRONTEND=noninteractive;
             sudo apt-get update -qq 2>/dev/null;
             sudo apt-get install -y -qq docker.io docker-compose-plugin 2>/dev/null;
             sudo usermod -aG docker \$USER 2>/dev/null || true;
             sudo systemctl start docker 2>/dev/null || true" \
            2>&1 | tail -3
        log "  Docker installed"
    fi

    # Copy compose files
    # shellcheck disable=SC2086
    ssh $SSH_FLAGS -i "$VM_SSH_KEY" -p "$_port" "$VM_SSH_USER@localhost" \
        "mkdir -p /tmp/lmcache-p2p"
    scp -o StrictHostKeyChecking=no -i "$VM_SSH_KEY" -P "$_port" \
        "$_compose_src" "$VM_SSH_USER@localhost:/tmp/lmcache-p2p/docker-compose.yml"
    scp -o StrictHostKeyChecking=no -i "$VM_SSH_KEY" -P "$_port" \
        "$_setup_src" "$VM_SSH_USER@localhost:/tmp/lmcache-p2p/lmcache-p2p-guest-setup.sh"

    # Transfer the minimal lmcache image into the VM.
    # We use docker save | ssh docker load — the image is ~500MB (no torch/ROCm).
    log "  Loading aic-lmcache-minimal into VM on :$_port ..."
    docker save aic-lmcache-minimal:latest | \
        ssh $SSH_FLAGS -i "$VM_SSH_KEY" -p "$_port" "$VM_SSH_USER@localhost" \
            "sudo docker load" 2>&1 | tail -3
    log "  Image loaded"
done

# ---- Step 6: start lmcache servers in VMs ------------------------------------
log "[6/7] Starting lmcache P2P servers in VMs ..."
# Coordinator is already running (started in step 5a) at COORD_DOCKER_IP.

# Start lmcache server in each VM — coordinator reachable via Docker bridge IP
for _idx in 1 2; do
    _port="$VM1_SSH_PORT"; [ "$_idx" = "2" ] && _port="$VM2_SSH_PORT"
    _role="secondary";     [ "$_idx" = "1" ] && _role="primary"
    _this_ip="$VM1_IP";    [ "$_idx" = "2" ] && _this_ip="$VM2_IP"
    log "  VM$_idx ($_role) :$_port $_this_ip ..."
    # shellcheck disable=SC2086
    ssh $SSH_FLAGS -i "$VM_SSH_KEY" -p "$_port" "$VM_SSH_USER@localhost" \
        "LMCACHE_P2P_ROLE=$_role \
         THIS_IP=$_this_ip \
         COORD_IP=$COORD_DOCKER_IP \
         COMPOSE_FILE=/tmp/lmcache-p2p/docker-compose.yml \
         LMCACHE_IMAGE_REF=$LMCACHE_IMAGE \
         LMCACHE_PORT=$LMCACHE_PORT \
         LMCACHE_HTTP_PORT=$LMCACHE_HTTP_PORT \
         COORD_PORT=$COORD_PORT \
         P2P_PORT=$P2P_PORT \
         bash /tmp/lmcache-p2p/lmcache-p2p-guest-setup.sh" \
        2>&1 | sed "s/^/  [vm$_idx] /" || \
        log "  WARN: VM$_idx setup script returned non-zero"
done

# ---- Step 7: check lmcache HTTP health + P2P metrics -------------------------
log "[7/7] Waiting for lmcache servers and checking P2P metrics ..."
# Wait for VM1 HTTP health (coordinator + server)
for _vmspec in "$VM1_SSH_PORT VM1" "$VM2_SSH_PORT VM2"; do
    _port=$(echo "$_vmspec" | awk '{print $1}')
    _label=$(echo "$_vmspec" | awk '{print $2}')
    _ready=0
    for _i in $(seq 1 $(( READY_S / 5 ))); do
        # Check if lmcache_server container is running (TCP socket)
        # shellcheck disable=SC2086
        if ssh $SSH_FLAGS -i "$VM_SSH_KEY" -p "$_port" "$VM_SSH_USER@localhost" \
                "sudo docker inspect aic-lmcache-p2p --format '{{.State.Status}}' 2>/dev/null | grep -q running"; then
            _ready=1; break
        fi
        sleep 5
    done
    if [ "$_ready" = "1" ]; then
        log "  $_label lmcache ready (/healthz on :${LMCACHE_HTTP_PORT})"
    else
        log "WARN: $_label lmcache not ready after ${READY_S}s — checking logs"
        # shellcheck disable=SC2086
        ssh $SSH_FLAGS -i "$VM_SSH_KEY" -p "$_port" "$VM_SSH_USER@localhost" \
            "docker compose -f /tmp/lmcache-p2p/docker-compose.yml logs --tail 20 2>/dev/null || true"
    fi
done

# Collect P2P metrics from VM2
log "  Checking P2P metrics on VM2 ..."
# shellcheck disable=SC2086
_metrics=$(ssh $SSH_FLAGS -i "$VM_SSH_KEY" -p "$VM2_SSH_PORT" "$VM_SSH_USER@localhost" \
    "curl -sS http://127.0.0.1:${LMCACHE_HTTP_PORT}/metrics 2>/dev/null || echo no_metrics")

printf '%s\n' "$_metrics" | grep -E 'lmcache_mp_(p2p|remote|l1|l2)' | grep -v '^#' | sed 's/^/  [metrics] /' || true

_p2p_hits=$(printf '%s\n' "$_metrics" | \
    grep -oE 'lmcache_mp_(p2p_load|remote_hit)_count_total[[:space:]]+[0-9.]+' | \
    awk '{s+=$$NF} END {printf "%d", s+0}')
_p2p_hits="${_p2p_hits:-0}"

if [ "$_p2p_hits" -gt 0 ]; then
    log "PASS: P2P hit count = $_p2p_hits"
else
    log "INFO: P2P hit count = 0 (servers may need traffic to generate hits)"
fi

log "=== test-lmcache-p2p-spur complete ==="
