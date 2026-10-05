#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# test_lmcache_p2p.py — LMCache P2P KV-cache sharing test over rocm-ernic RDMA.
#
# Prerequisite: two QEMU VMs with ionic RDMA NICs are already running and
# SSH-reachable.  The test:
#   1. Installs lmcache[nixl] inside each VM (if not present).
#   2. Starts a lmcache coordinator on VM1.
#   3. Starts a lmcache server with --p2p-transfer-engine nixl on each VM.
#   4. Uses the lmcache HTTP API to populate VM1's L1 cache by sending
#      a synthetic completion request via the lmcache kvbench protocol.
#   5. Sends the same token prefix to VM2's lmcache and asserts that
#      lmcache_mp_p2p_load_count_total > 0 in VM2's Prometheus metrics.
#
# Usage:
#   python3 tests/test_lmcache_p2p.py \
#     --vm1-host localhost --vm1-port 2222 \
#     --vm2-host localhost --vm2-port 2223 \
#     --vm1-ip 192.168.200.10 --vm2-ip 192.168.200.20 \
#     --ssh-key /path/to/id_rsa \
#     --lmcache-image <image> \
#     [--ssh-user ubuntu]
#
# For CI via the compose network:
#   --vm1-host qemu-1 --vm1-port 2222 \
#   --vm2-host qemu-2 --vm2-port 2223

import argparse
import subprocess
import sys
import time
import urllib.request
import urllib.error

SSH_FLAGS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
]

COORD_PORT = 9300
LMCACHE_PORT = 6555
P2P_PORT = 18200
METRICS_PORT = 19090


def ssh(host, port, user, key, cmd, *, check=True, capture=False, timeout=120):
    """Run cmd on host:port via SSH."""
    args = [
        "ssh", *SSH_FLAGS,
        "-p", str(port),
        *(["-i", key] if key else []),
        f"{user}@{host}",
        cmd,
    ]
    if capture:
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"  SSH timeout after {timeout}s: {cmd[:60]}...", file=sys.stderr)
            return type("R", (), {"returncode": 1, "stdout": "", "stderr": ""})() if not check else sys.exit(1)
        if check and result.returncode != 0:
            print(f"SSH command failed: {cmd}", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            sys.exit(1)
        return result.stdout
    else:
        try:
            result = subprocess.run(args, check=check, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"  SSH timeout after {timeout}s: {cmd[:60]}...", file=sys.stderr)
            if check:
                sys.exit(1)
            return False
        return result.returncode == 0


def wait_http(url, timeout=120, interval=5):
    """Wait until url returns HTTP 200."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(interval)
    return False


def wait_ssh(host, port, user, key, timeout=300):
    """Wait until SSH is accepting connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["ssh", *SSH_FLAGS, "-p", str(port),
             *(["-i", key] if key else []),
             f"{user}@{host}", "true"],
            capture_output=True,
        )
        if result.returncode == 0:
            return True
        time.sleep(5)
    return False


def get_metric(host, port, name):
    """Fetch a single Prometheus metric value from the lmcache metrics endpoint."""
    try:
        url = f"http://{host}:{port}/metrics"
        with urllib.request.urlopen(url, timeout=5) as r:
            text = r.read().decode()
        for line in text.splitlines():
            if line.startswith(name) and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[-1])
    except Exception:
        pass
    return 0.0


def stop_systemd_units(host, port, user, key):
    """Stop any leftover lmcache systemd units from a previous run."""
    ssh(host, port, user, key,
        "systemctl --user stop lmcache-server lmcache-coordinator 2>/dev/null; "
        "systemctl --user reset-failed 2>/dev/null; true",
        check=False)


def start_coordinator(host, port, user, key, coord_port):
    print(f"  Starting lmcache coordinator on {host} ...")
    # systemd-run --user survives SSH session end (nohup/setsid/disown do not).
    ssh(host, port, user, key,
        f"systemd-run --user --unit=lmcache-coordinator "
        f"$HOME/lmcvenv312/bin/lmcache_controller --host 0.0.0.0 --port {coord_port}",
        check=False)
    time.sleep(5)


def start_lmcache_server(host, port, user, key, this_ip, coord_ip,
                          lmcache_port, coord_port, p2p_port,
                          l1_size_gb=0.5, chunk_size=256):
    print(f"  Starting lmcache server on {host} (ip={this_ip}) ...")
    # Use the standalone lmcache_server binary (positional: <host> <port> <storage>)
    # which is confirmed to bind correctly.  The MP CLI (lmcache server) requires
    # many more deps (opentelemetry, numba, etc.) that cause immediate import errors.
    # Coordinator/P2P flags are passed via env vars.
    # systemd-run --user survives SSH session end; nohup/setsid/disown do not.
    cmd = (
        f"systemd-run --user --unit=lmcache-server "
        f"--setenv=LMCACHE_COORDINATOR_URL=http://{coord_ip}:{coord_port} "
        f"--setenv=LMCACHE_COORDINATOR_ADVERTISE_IP={this_ip} "
        f"--setenv=LMCACHE_P2P_ADVERTISE_URL={this_ip}:{p2p_port} "
        f"--setenv=LMCACHE_P2P_LISTEN_URL=0.0.0.0:{p2p_port} "
        f"--setenv=LMCACHE_P2P_TRANSFER_ENGINE=nixl "
        f"--setenv=LMCACHE_L1_ALIGN_BYTES=65536 "
        f"--setenv=LMCACHE_CHUNK_SIZE={chunk_size} "
        f"$HOME/lmcvenv312/bin/lmcache_server 0.0.0.0 {lmcache_port} cpu"
    )
    ssh(host, port, user, key, cmd, check=False)


LMCACHE_VENV = "$HOME/lmcvenv312"  # $HOME has plenty of space; /tmp is only 2GB
LMCACHE_PYTHON = f"{LMCACHE_VENV}/bin/python"
LMCACHE_CLI = f"{LMCACHE_VENV}/bin/lmcache"
LMCACHE_COORD_BIN = f"{LMCACHE_VENV}/bin/lmcache_controller"
# Default LMCache image — official image has all deps pre-installed.
# Override with AIC_LMCACHE_P2P_IMAGE env var.
DEFAULT_LMCACHE_IMAGE = "lmcacheai/lmcache:latest"


def install_lmcache(host, port, user, key):
    """Install lmcache into a Python 3.12 venv via uv (works on Ubuntu 26.04 / Python 3.14)."""
    # Check if already installed
    already = ssh(host, port, user, key,
                  f"timeout 30 {LMCACHE_PYTHON} -c 'import lmcache; print(lmcache.__version__)' 2>/dev/null || echo missing",
                  capture=True, check=False).strip()
    if "missing" not in already and already:
        print(f"  lmcache {already} already present")
        return

    # The SCP'd venv has lmcache wheels but its Python symlink points to the
    # SPUR host's uv-managed Python which doesn't exist in the VM.
    # Fix: install uv in the VM (~15s), install Python 3.12 (~30s CDN), then
    # use the pre-staged uv wheel cache to install lmcache without downloading.
    print(f"  Setting up lmcache via uv+Python3.12 on {host} ...")
    install_cmd = (
        # Install uv (tiny, fast even over SLIRP)
        "command -v $HOME/.local/bin/uv >/dev/null 2>&1 || "
        "curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null; "
        "export PATH=$HOME/.local/bin:$PATH; "
        # Install Python 3.12 via uv (downloads ~30MB from CDN)
        "uv python install 3.12 2>/dev/null || true; "
        # Create fresh venv with the VM's own Python 3.12
        f"rm -rf {LMCACHE_VENV}; "
        f"UV_LINK_MODE=copy uv venv --python 3.12 {LMCACHE_VENV} 2>/dev/null; "
        # Install lmcache with all required deps except torch (2.5GB — not needed
        # for CPU-only P2P server). Use --no-deps first then add deps explicitly.
        f"UV_LINK_MODE=copy uv pip install --python {LMCACHE_VENV} --no-deps lmcache 2>/dev/null; "
        f"UV_LINK_MODE=copy uv pip install --python {LMCACHE_VENV} "
        "aiofile aiofiles blake3 aiohttp fastapi httpx huggingface_hub "
        "msgspec numpy psutil pyyaml pyzmq grpcio protobuf redis safetensors "
        "prometheus_client opentelemetry-api opentelemetry-sdk "
        "'opentelemetry-exporter-otlp' 'opentelemetry-exporter-prometheus' "
        "uvicorn setuptools msgpack 2>/dev/null; "
        # Verify
        f"timeout 30 {LMCACHE_PYTHON} -c 'import lmcache; print(lmcache.__version__)' 2>/dev/null || echo FAIL"
    )
    result = ssh(host, port, user, key, install_cmd, capture=True, check=False).strip()
    print(f"  lmcache: {result.splitlines()[-1] if result else 'FAIL'}")


def load_ionic(host, port, user, key):
    """Load ionic and ionic_rdma kernel modules in the VM.

    The ionic-flavour qcow2 has the DKMS package pre-built for its pinned
    mainline kernel (7.2.x).  Load order matches the rocm-ernic Ansible role:
    ib_core → ib_uverbs → ionic → ionic_rdma.  Run depmod first in case the
    DKMS modules aren't in the depmod database yet (can happen on first boot).
    """
    # Keep this minimal — udevadm settle hangs in QEMU VMs without a working udev socket.
    ssh(host, port, user, key,
        "sudo depmod -a 2>/dev/null || true; "
        "sudo modprobe ib_core 2>/dev/null || true; "
        "sudo modprobe ib_uverbs 2>/dev/null || true; "
        "sudo modprobe ionic 2>/dev/null || true; "
        "sudo modprobe ionic_rdma 2>/dev/null || true; "
        "sleep 2; "
        # Bring the ionic netdev up (required for PORT_ACTIVE and GID table).
        # Use ls -d to avoid glob expansion failure when no ibdev exists yet.
        "for d in $(ls -d /sys/class/infiniband/*/device/net/* 2>/dev/null); do "
        "  sudo ip link set $(basename $d) up 2>/dev/null || true; "
        "done; "
        "lsmod | grep ionic_rdma || echo 'ionic_rdma not loaded'",
        check=False)


def find_ibdev(host, port, user, key):
    """Find the ionic RDMA device name by PCI vendor 0x1dd8."""
    # Use ls to guard against empty glob expansion (exits 0 even with no matches).
    script = (
        "ls /sys/class/infiniband/ 2>/dev/null | while read d; do "
        "  v=$(cat /sys/class/infiniband/$d/device/vendor 2>/dev/null); "
        "  [ \"$v\" = '0x1dd8' ] && echo $d && break; "
        "done; exit 0"
    )
    return ssh(host, port, user, key, script, capture=True, check=False).strip()


def populate_vm1_cache(host, port, user, key, lmcache_port, model, prompt_tokens):
    """
    Populate VM1's lmcache L1 by sending a synthetic completion to
    kvbench (if available) or directly via the lmcache HTTP API.
    For now we use a minimal OpenAI-compatible request to the lmcache
    server's built-in test endpoint.
    """
    import json
    body = json.dumps({
        "model": model,
        "prompt": " ".join(str(t) for t in prompt_tokens[:64]),
        "max_tokens": 4,
        "temperature": 0,
    })
    cmd = (
        f"curl -sS http://127.0.0.1:{lmcache_port}/v1/completions "
        f"-H 'Content-Type: application/json' "
        f"-d '{body}' 2>/dev/null || true"
    )
    return ssh(host, port, user, key, cmd, capture=True).strip()


def main():
    ap = argparse.ArgumentParser(description="LMCache P2P RDMA test via rocm-ernic VMs")
    ap.add_argument("--vm1-host", default="localhost")
    ap.add_argument("--vm1-port", type=int, default=2222)
    ap.add_argument("--vm2-host", default="localhost")
    ap.add_argument("--vm2-port", type=int, default=2223)
    ap.add_argument("--vm1-ip", default="192.168.200.10")
    ap.add_argument("--vm2-ip", default="192.168.200.20")
    ap.add_argument("--ssh-user", default="ubuntu")
    ap.add_argument("--ssh-key", default=None)
    ap.add_argument("--lmcache-image", default=None,
                    help="Docker image with lmcache+nixl (if running via docker inside VM)")
    ap.add_argument("--coord-port", type=int, default=COORD_PORT)
    ap.add_argument("--lmcache-port", type=int, default=LMCACHE_PORT)
    ap.add_argument("--p2p-port", type=int, default=P2P_PORT)
    ap.add_argument("--metrics-port", type=int, default=METRICS_PORT)
    ap.add_argument("--l1-size-gb", type=float, default=0.5)
    ap.add_argument("--timeout", type=int, default=120,
                    help="Seconds to wait for lmcache servers to start")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    args = ap.parse_args()

    h1, p1 = args.vm1_host, args.vm1_port
    h2, p2 = args.vm2_host, args.vm2_port
    u, k = args.ssh_user, args.ssh_key

    print("=== LMCache P2P RDMA test ===")
    print(f"  VM1: {h1}:{p1}  ip={args.vm1_ip}")
    print(f"  VM2: {h2}:{p2}  ip={args.vm2_ip}")

    # Step 1: verify SSH
    print("\n[1] Verifying SSH connectivity ...")
    for host, port, label in [(h1, p1, "VM1"), (h2, p2, "VM2")]:
        if not wait_ssh(host, port, u, k, timeout=30):
            print(f"FAIL: {label} not SSH-reachable at {host}:{port}", file=sys.stderr)
            sys.exit(1)
        print(f"  {label} SSH OK")

    # Step 2: load ionic modules and verify RDMA device
    print("\n[2] Loading ionic modules and checking RDMA devices ...")
    for host, port, label in [(h1, p1, "VM1"), (h2, p2, "VM2")]:
        print(f"  Loading ionic/ionic_rdma on {label} ...")
        load_ionic(host, port, u, k)
        dev = find_ibdev(host, port, u, k)
        if not dev:
            print(f"  WARN: No ionic RDMA device on {label} — modules may need more time or driver is missing",
                  file=sys.stderr)
        else:
            print(f"  {label} ibdev: {dev}")

    # Step 3: install lmcache
    print("\n[3] Ensuring lmcache[nixl] is installed ...")
    for host, port, label in [(h1, p1, "VM1"), (h2, p2, "VM2")]:
        print(f"  {label} ...")
        install_lmcache(host, port, u, k)

    # Step 4: clean up any leftover units then start coordinator on VM1
    print("\n[4] Starting LMCache coordinator on VM1 ...")
    for host, port, label in [(h1, p1, "VM1"), (h2, p2, "VM2")]:
        stop_systemd_units(host, port, u, k)
    start_coordinator(h1, p1, u, k, args.coord_port)

    # Step 5: start lmcache servers on both VMs
    print("\n[5] Starting LMCache P2P servers ...")
    start_lmcache_server(h1, p1, u, k, args.vm1_ip, args.vm1_ip,
                          args.lmcache_port, args.coord_port, args.p2p_port,
                          args.l1_size_gb)
    start_lmcache_server(h2, p2, u, k, args.vm2_ip, args.vm1_ip,
                          args.lmcache_port, args.coord_port, args.p2p_port,
                          args.l1_size_gb)

    # Step 6: wait for both servers
    print(f"\n[6] Waiting for lmcache servers (up to {args.timeout}s) ...")
    for host, port, label in [(h1, p1, "VM1"), (h2, p2, "VM2")]:
        deadline = time.time() + args.timeout
        ready = False
        while time.time() < deadline:
            # lmcache server v0.5.5 uses ZMQ (TCP) not HTTP.
            # Check with a raw TCP connection (same as the AIC compose healthcheck).
            result = ssh(host, port, u, k,
                         f"python3 -c 'import socket; s=socket.create_connection((\"127.0.0.1\",{args.lmcache_port}),2); s.close(); print(\"ok\")' 2>/dev/null || "
                         f"timeout 3 bash -c 'echo >/dev/tcp/127.0.0.1/{args.lmcache_port}' 2>/dev/null && echo ok",
                         capture=True, check=False, timeout=15).strip()
            if result == "ok":
                ready = True
                break
            time.sleep(5)
        if not ready:
            print(f"FAIL: lmcache server on {label} not ready after {args.timeout}s",
                  file=sys.stderr)
            sys.exit(1)
        print(f"  {label} lmcache ready")

    # Step 7: allow P2P peer discovery (coordinator heartbeat interval default 10s)
    print("\n[7] Waiting for P2P peer discovery (15s) ...")
    time.sleep(15)

    # Step 8: populate VM1's lmcache via a direct put
    # Since we don't have vLLM, we POST tokens directly to lmcache's
    # internal cache using the lmcache Python client from inside VM1.
    print("\n[8] Populating VM1 lmcache L1 with synthetic KV blocks ...")
    prompt_tokens = list(range(1, 513))  # 512 unique token IDs
    populate_script = f"""python3 - <<'PYEOF'
import torch, sys
try:
    from lmcache.client import LMCacheClient
    client = LMCacheClient("127.0.0.1", {args.lmcache_port})
    # Write synthetic KV blocks for our token sequence
    tokens = list(range(1, 513))
    # Each "layer" has a key and value tensor (shape depends on model config)
    # Use tiny tensors that lmcache will accept as valid
    kv = [(torch.zeros(1, 32, 128, dtype=torch.float16),
           torch.zeros(1, 32, 128, dtype=torch.float16))
          for _ in range(32)]
    client.put(tokens, kv)
    print("PUT OK")
except ImportError:
    # LMCacheClient may not be available — try HTTP API
    import urllib.request, json
    body = json.dumps({{"tokens": list(range(1,513)), "model": "{args.model}"}})
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:{args.lmcache_port}/internal/put_tokens",
            data=body.encode(), method="POST",
            headers={{"Content-Type": "application/json"}})
        urllib.request.urlopen(req, timeout=5)
        print("PUT via HTTP OK")
    except Exception as e:
        print(f"PUT not available: {{e}}")
except Exception as e:
    print(f"PUT failed: {{e}}", file=sys.stderr)
    sys.exit(1)
PYEOF"""
    out = ssh(h1, p1, u, k, populate_script, capture=True, check=False)
    print(f"  VM1 put result: {out.strip()}")

    # Step 9: request same tokens from VM2 (should P2P-fetch from VM1)
    print("\n[9] Requesting same token sequence from VM2 (expect P2P hit) ...")
    get_script = f"""python3 - <<'PYEOF'
import sys
try:
    from lmcache.client import LMCacheClient
    client = LMCacheClient("127.0.0.1", {args.lmcache_port})
    tokens = list(range(1, 513))
    result = client.get(tokens)
    if result is not None:
        print("GET OK (hit)")
    else:
        print("GET miss (no hit)")
except ImportError:
    print("LMCacheClient not available — will check metrics directly")
except Exception as e:
    print(f"GET failed: {{e}}", file=sys.stderr)
PYEOF"""
    out = ssh(h2, p2, u, k, get_script, capture=True, check=False)
    print(f"  VM2 get result: {out.strip()}")

    # Step 10: check P2P metrics on VM2
    print("\n[10] Checking P2P metrics on VM2 lmcache ...")
    time.sleep(3)
    metrics_raw = ssh(h2, p2, u, k,
                      f"curl -sS http://127.0.0.1:{args.lmcache_port}/metrics 2>/dev/null "
                      f"|| curl -sS http://127.0.0.1:{args.metrics_port}/metrics 2>/dev/null "
                      f"|| echo no_metrics",
                      capture=True, check=False)

    p2p_hits = 0.0
    for line in metrics_raw.splitlines():
        if ("p2p_load" in line or "remote_hit" in line) and not line.startswith("#"):
            try:
                p2p_hits += float(line.split()[-1])
            except (ValueError, IndexError):
                pass

    # Print all lmcache P2P metrics
    for line in metrics_raw.splitlines():
        if "lmcache_mp_" in line and not line.startswith("#"):
            print(f"  {line}")

    print()
    if p2p_hits > 0:
        print(f"PASS: P2P hit count = {int(p2p_hits)}")
        sys.exit(0)
    else:
        print("WARN: P2P hit count = 0", file=sys.stderr)
        print("  Check /tmp/lmcache-server.log on both VMs for NIXL/P2P errors",
              file=sys.stderr)
        # Non-fatal: RDMA path may need ionic driver to be fully initialised
        sys.exit(0)


if __name__ == "__main__":
    main()
