#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# lmcache-p2p-guest-setup.sh — configure the ionic RDMA NIC and launch the
# LMCache P2P compose stack inside a rocm-ernic VM.
#
# Run over SSH from test-lmcache-p2p-local (mk/bench.mk).
#
# Environment variables:
#   LMCACHE_P2P_ROLE    primary | secondary (required)
#   THIS_IP             Ionic interface IP for this VM (required)
#   COORD_IP            IP of the primary VM coordinator (required)
#   COMPOSE_FILE        Path to docker-compose.yml inside the VM (required)
#   LMCACHE_IMAGE_REF   LMCache image (required)
#   VLLM_IMAGE_REF      vLLM image (secondary role only)
#   VLLM_MODEL          Model name (default: HuggingFaceTB/SmolLM2-135M-Instruct)
#   HF_TOKEN            HuggingFace token (secondary role only)
#   COORD_PORT          Coordinator port (default: 9300)
#   P2P_PORT            P2P transfer-channel port (default: 18200)
#   LMCACHE_PORT        LMCache HTTP port (default: 6555)
#   LMCACHE_L1_SIZE_GB  L1 DRAM cap in GiB (default: 0.5)
#   LMCACHE_CHUNK_SIZE  Tokens per KV chunk (default: 256)
#   VLLM_PORT           vLLM serve port (default: 8000)
#   IONIC_IFACE         Ionic ethernet interface name (default: auto-detect)

set -euo pipefail

LMCACHE_P2P_ROLE="${LMCACHE_P2P_ROLE:?LMCACHE_P2P_ROLE is required}"
THIS_IP="${THIS_IP:?THIS_IP is required}"
COORD_IP="${COORD_IP:?COORD_IP is required}"
COMPOSE_FILE="${COMPOSE_FILE:?COMPOSE_FILE is required}"
LMCACHE_IMAGE_REF="${LMCACHE_IMAGE_REF:?LMCACHE_IMAGE_REF is required}"
COORD_PORT="${COORD_PORT:-9300}"
P2P_PORT="${P2P_PORT:-8500}"
LMCACHE_PORT="${LMCACHE_PORT:-6555}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-7555}"
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-0.5}"

# ---------------------------------------------------------------------------
# Load ionic modules
# ---------------------------------------------------------------------------
echo "=== lmcache-p2p-guest-setup: ROLE=${LMCACHE_P2P_ROLE} THIS_IP=${THIS_IP} ==="

if ! lsmod | grep -qw ionic_rdma; then
    echo "Loading ionic and ionic_rdma modules ..."
    sudo modprobe ionic  2>/dev/null || true
    sudo modprobe ionic_rdma 2>/dev/null || true
    sleep 2
fi

# ---------------------------------------------------------------------------
# Configure ionic interface IP
# ---------------------------------------------------------------------------
# The ionic NIC appears as a netdev named by the driver (rocm-ernic0 or
# similar).  Find it by its ionic driver.
if [ -z "${IONIC_IFACE:-}" ]; then
    IONIC_IFACE=$(for iface in /sys/class/net/*; do
        drv=$(readlink "$iface/device/driver" 2>/dev/null | xargs basename 2>/dev/null || true)
        [ "$drv" = "ionic" ] && basename "$iface" && break
    done)
fi

if [ -n "${IONIC_IFACE:-}" ]; then
    echo "Configuring ${IONIC_IFACE} with ${THIS_IP}/24 ..."
    sudo ip addr show "${IONIC_IFACE}" | grep -q "${THIS_IP}" \
        || sudo ip addr add "${THIS_IP}/24" dev "${IONIC_IFACE}" 2>/dev/null || true
    sudo ip link set "${IONIC_IFACE}" up 2>/dev/null || true

    # Wait for the IP to be reachable (up to 10s)
    for _i in $(seq 1 10); do
        ip addr show "${IONIC_IFACE}" 2>/dev/null | grep -q "${THIS_IP}" && break
        sleep 1
    done

    echo "RDMA devices:"
    ibv_devices 2>/dev/null || echo "  (ibv_devices not available)"
else
    echo "WARNING: no ionic interface found; using management NIC for P2P" >&2
    # Fall back to management NIC IP for coordinator/P2P (SLIRP: 10.0.2.15)
    THIS_IP=$(ip -4 addr show scope global 2>/dev/null | grep -oP '(?<=inet )[0-9.]+' | grep -v '192\.168\.200' | head -1)
    echo "Using fallback IP: ${THIS_IP:-unknown}"
fi

# ---------------------------------------------------------------------------
# Launch compose
# ---------------------------------------------------------------------------
echo "Starting docker compose (role=${LMCACHE_P2P_ROLE} coord=${COORD_IP}:${COORD_PORT:-9300}) ..."

# Both primary and secondary run the same lmcache service.
# The coordinator runs on the host Docker bridge (not in this VM).
LMCACHE_IMAGE_REF="${LMCACHE_IMAGE_REF}" \
COORD_IP="${COORD_IP}" \
THIS_IP="${THIS_IP}" \
COORD_PORT="${COORD_PORT:-9300}" \
P2P_PORT="${P2P_PORT:-8500}" \
LMCACHE_PORT="${LMCACHE_PORT:-6555}" \
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-7555}" \
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-0.5}" \
    docker compose \
        -f "${COMPOSE_FILE}" \
        up -d
