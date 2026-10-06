#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# run-lmcache-p2p-test.sh — LMCache P2P KV-cache sharing test on a SPUR node.
#
# Boots two QEMU KVM VMs with rocm-ernic emulated ionic RDMA NICs using
# qemu-tool compose (vfio-user-ernic-2vm stack), then runs the Python-level
# P2P test (tests/test_lmcache_p2p.py) against those VMs.
#
# Invoked by: make test-lmcache-p2p-spur
# Or directly: srun --partition=amd-spur --gpus=1 bash .slurm/run-lmcache-p2p-test.sh
#
# Required env:
#   SLURM_SUBMIT_DIR    Path to repo root (set by make target)
#   HF_TOKEN            HuggingFace token (for model download inside VMs)
#   AIC_LMCACHE_IMAGE   LMCache+NIXL image ref
#   AIC_VLLM_IMAGE      vLLM image ref (used only if secondary needs vllm)
#
# Optional env:
#   AIC_LMCACHE_P2P_VM_IMAGES_DIR  Where to store qcow2 images (default: /tmp/aic-p2p-images)
#   AIC_LMCACHE_P2P_QCOW2_IMAGE    Docker image containing the ionic guest qcow2
#   AIC_LMCACHE_P2P_READY_S        SSH readiness timeout in seconds (default: 480)
#   AIC_LMCACHE_P2P_VM1_IP         Ionic interface IP for VM1 (default: 192.168.200.10)
#   AIC_LMCACHE_P2P_VM2_IP         Ionic interface IP for VM2 (default: 192.168.200.20)

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set to the repo root}"
VM_IMAGES_DIR="${AIC_LMCACHE_P2P_VM_IMAGES_DIR:-/tmp/aic-p2p-images-${SLURM_JOB_ID:-$$}}"
QCOW2_IMAGE="${AIC_LMCACHE_P2P_QCOW2_IMAGE:-docker.io/sbates130272/batesste-ci-images-ubuntu-qcow2-gen-ionic:20260929.g2bdbd16-vm.resolute-ionic-qm.54cc234}"
READY_S="${AIC_LMCACHE_P2P_READY_S:-480}"
VM1_IP="${AIC_LMCACHE_P2P_VM1_IP:-192.168.200.10}"
VM2_IP="${AIC_LMCACHE_P2P_VM2_IP:-192.168.200.20}"
VM1_SSH_PORT=12230
VM2_SSH_PORT=12231
VM1_NAME="p2p-vm1-${SLURM_JOB_ID:-$$}"
VM2_NAME="p2p-vm2-${SLURM_JOB_ID:-$$}"
WORK_DIR="/tmp/aic-p2p-work-${SLURM_JOB_ID:-$$}"

VM_SSH_KEY="${VM_IMAGES_DIR}/id_rsa"
VM_SSH_USER="ubuntu"   # overridden from vm-info.json after extraction
SSH_FLAGS="-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
die() { log "FATAL: $*" >&2; cleanup; exit 1; }

cleanup() {
    log "Tearing down compose stack ..."
    VM_IMAGES_DIR="$VM_IMAGES_DIR" \
    VM1_NAME="$VM1_NAME" \
    VM2_NAME="$VM2_NAME" \
        qemu-tool compose \
            --stack vfio-user-ernic-2vm \
            --vm-name "$VM1_NAME" \
            --vm2-name "$VM2_NAME" \
            down 2>/dev/null || true
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

log "=== LMCache P2P SPUR test ==="
log "  Node:     $(hostname)"
log "  Repo:     $REPO_ROOT"
log "  Images:   $VM_IMAGES_DIR"

# ---- Step 0: clean up any stale containers from a prior run ---------------
# Force-remove containers that share our VM names to avoid compose collision.
log "[0/8] Cleaning up any stale containers ..."
docker ps -a --filter "name=vfio-user-ernic-2vm" --format "{{.ID}}" \
  | xargs -r docker rm -f 2>/dev/null || true
docker network ls --filter "name=vfio-user-ernic-2vm" --format "{{.ID}}" \
  | xargs -r docker network rm 2>/dev/null || true
sleep 2

# ---- Step 1: prerequisites -----------------------------------------------
log "[1/8] Checking prerequisites ..."
test -c /dev/kvm || die "/dev/kvm not available — nested virtualisation required"
command -v docker >/dev/null || die "docker not found"

# ---- Step 2: install qemu-tool -------------------------------------------
log "[2/8] Ensuring qemu-tool is installed ..."
# Always prefer ~/.local/bin (user-writable, no sudo needed).
QEMU_TOOL_BIN="${HOME}/.local/bin/qemu-tool"
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v qemu-tool >/dev/null 2>&1; then
    log "  Installing qemu-tool via pipx ..."
    if command -v pipx >/dev/null 2>&1; then
        PIPX_BIN_DIR="${HOME}/.local/bin" pipx install --force qemu-tool >/dev/null
    else
        pip install --quiet --break-system-packages qemu-tool 2>/dev/null \
            || pip install --quiet qemu-tool
    fi
fi
qemu-tool --version

# ---- Step 3: pull and extract guest qcow2 --------------------------------
log "[3/8] Setting up VM disk images ..."
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
    # Copy the SSH key and vm-info.json that ship alongside the disk
    find "$WORK_DIR/qcow2-tmp" -name 'id_rsa' -exec cp {} "$VM_IMAGES_DIR/" \;
    find "$WORK_DIR/qcow2-tmp" -name 'vm-info.json' -exec cp {} "$VM_IMAGES_DIR/" \;
    chmod 600 "$VM_IMAGES_DIR/id_rsa" 2>/dev/null || true
    rm -rf "$WORK_DIR/qcow2-tmp"
    log "  Disk images ready (key: $(ls "$VM_IMAGES_DIR/id_rsa" 2>/dev/null && echo present || echo MISSING))"
else
    log "  Disk images already present"
    [ -f "$VM_IMAGES_DIR/${VM2_NAME}.qcow2" ] || \
        cp "$VM_IMAGES_DIR/${VM1_NAME}.qcow2" "$VM_IMAGES_DIR/${VM2_NAME}.qcow2"
fi

# Resolve SSH key and username from the vm-info.json bundled with the disk image.
VM_SSH_KEY="$VM_IMAGES_DIR/id_rsa"
if [ -f "$VM_IMAGES_DIR/vm-info.json" ]; then
    VM_SSH_USER=$(python3 -c "import json; print(json.load(open('$VM_IMAGES_DIR/vm-info.json')).get('username','ubuntu'))" 2>/dev/null || echo "ubuntu")
else
    VM_SSH_USER="ubuntu"
fi
log "  SSH user=$VM_SSH_USER key=$([ -f "$VM_SSH_KEY" ] && echo present || echo MISSING)"
[ -f "$VM_SSH_KEY" ] || die "SSH key not found at $VM_SSH_KEY — re-run to re-extract"

# ---- Step 4: start compose stack -----------------------------------------
log "[4/8] Starting vfio-user-ernic-2vm compose stack ..."
VM_IMAGES_DIR="$VM_IMAGES_DIR" \
VM1_NAME="$VM1_NAME" \
VM2_NAME="$VM2_NAME" \
VM1_SSH_PORT="$VM1_SSH_PORT" \
VM2_SSH_PORT="$VM2_SSH_PORT" \
VM_VCPUS=4 \
VM_VMEM=4096 \
    qemu-tool compose \
        --stack vfio-user-ernic-2vm \
        --vm-name "$VM1_NAME" \
        --vm2-name "$VM2_NAME" \
        up -d

# ---- Step 5: wait for SSH ------------------------------------------------
log "[5/8] Waiting for SSH on both VMs (up to ${READY_S}s) ..."
for _vmspec in "$VM1_SSH_PORT VM1" "$VM2_SSH_PORT VM2"; do
    _port=$(echo "$_vmspec" | awk '{print $1}')
    _label=$(echo "$_vmspec" | awk '{print $2}')
    _ready=0
    for _i in $(seq 1 $(( READY_S / 5 ))); do
        # shellcheck disable=SC2086
        if ssh $SSH_FLAGS -i "$VM_SSH_KEY" -p "$_port" "$VM_SSH_USER@localhost" true 2>/dev/null; then
            _ready=1; break
        fi
        sleep 5
    done
    [ "$_ready" = "1" ] || {
        log "Compose logs on failure:"
        VM_IMAGES_DIR="$VM_IMAGES_DIR" VM1_NAME="$VM1_NAME" VM2_NAME="$VM2_NAME" \
            qemu-tool compose --stack vfio-user-ernic-2vm \
            --vm-name "$VM1_NAME" --vm2-name "$VM2_NAME" \
            logs --tail 40 2>/dev/null || true
        die "$_label not SSH-reachable after ${READY_S}s"
    }
    log "  $_label SSH ready (:$_port)"
done

# ---- Step 6: run the Python P2P test ------------------------------------
# test_lmcache_p2p.py handles: ionic RDMA device detection, lmcache[nixl]
# install via pip, coordinator + P2P server startup, cache population, and
# the P2P hit assertion — all without requiring Docker inside the VMs.
log "[6/6] Running LMCache P2P test ..."
python3 "$REPO_ROOT/tests/test_lmcache_p2p.py" \
    --vm1-host localhost --vm1-port "$VM1_SSH_PORT" \
    --vm2-host localhost --vm2-port "$VM2_SSH_PORT" \
    --vm1-ip "$VM1_IP" --vm2-ip "$VM2_IP" \
    --ssh-user "$VM_SSH_USER" \
    --ssh-key "$VM_SSH_KEY" \
    --timeout 300

log "=== test-lmcache-p2p-spur complete ==="
