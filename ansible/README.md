<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Ansible — cluster discovery, provisioning, and monitoring

Run playbooks from this directory. Inventory and group variables live under
`inventory/`; local roles under `roles/`. Galaxy collection
[sbates130272.batesste][galaxy] supplies user, RDMA, ROCm, and exporter
roles used by `host_setup`.

## Prerequisites

```bash
cd ansible
ansible-galaxy collection install -r requirements.yml
export ANSIBLE_REMOTE_USER=<lab-user>   # matches inventory/hosts.yml
```

## Playbook entry points

| Goal | Command |
|------|---------|
| Full flow | `ansible-playbook site.yml` |
| Discovery only | `ansible-playbook playbooks/discover.yml` |
| Host provisioning | `ansible-playbook site.yml --tags provision` |
| Exporters only | `ansible-playbook site.yml --tags rocm_aic_exporter` |
| Inference image | `ansible-playbook site.yml --tags inference-container` |
| Monitoring | `ansible-playbook site.yml --tags monitoring` |
| NVMe over Fabrics | `ansible-playbook site.yml --tags nvmeof` |
| NFS migration | `ansible-playbook site.yml --tags nfs-rdma` |

The NFS play is tagged `never` in `site.yml`; run it explicitly with
`--tags nfs-rdma` when migrating from NVMe over Fabrics to NFS over RDMA.

List all tags: `ansible-playbook site.yml --list-tags`.

## Variable surfaces

- `inventory/group_vars/<group>.yml` — primary lab tuning
- `roles/*/defaults/main.yml` — role defaults (override in inventory)
- CLI `-e key=value` — one-off overrides (secrets, endpoints)

Discovery reports: `playbooks/reports/` (per-host JSON) and
`reports/` (RDMA cross-host summaries).

## Recipes — build and run

Ansible does **not** deploy vLLM Docker recipes. Build and run them on GPU
nodes manually; use Ansible for host exporters and the central monitoring
stack.

### vLLM + LMCache + NIXL (`recipies/vllm-lmcache-nixl`)

Build context is the **repository root**. Containers use **`--network=host`**
(vLLM `/metrics` on host port **8000**).

```bash
export ROCM_ARCH=gfx942          # or gfx1201 on Radeon
make -C recipies/vllm-lmcache-nixl build
export HF_TOKEN=...              # or HF_TOKEN_FILE=/path/to/token
mkdir -p /mnt/lmcache-nvme recipies/vllm-lmcache-nixl/logs
make -C recipies/vllm-lmcache-nixl run \
  VLN_LMCACHE_IO=nixl-posix \
  GPU=0 DATA=/mnt/lmcache-nvme
```

AIS mode: `VLN_LMCACHE_IO=ais VLH_HIPFILE_STATS_LEVEL=1`.

Slurm from repo root: `./run-slurm-nixl.sh`.

See [recipies/vllm-lmcache-nixl/README.md](../recipies/vllm-lmcache-nixl/README.md).

### vLLM + LMCache + hipFile (`recipies/vllm-lmcache-hipfile`)

Same pattern; default IO backend is hipfile:

```bash
export ROCM_ARCH=gfx942
make -C recipies/vllm-lmcache-hipfile build
export HF_TOKEN=...
mkdir -p /mnt/lmcache-nvme recipies/vllm-lmcache-hipfile/logs
make -C recipies/vllm-lmcache-hipfile run \
  VLH_LMCACHE_IO=hipfile \
  GPU=0 DATA=/mnt/lmcache-nvme
```

Slurm: `./run-slurm.sh`.

### Ansible coupling after a recipe is running

1. `host_setup_rocm_aic_exporter_ais_stats_container` must match the
   container name (`CONTAINER_NAME`, default `vllm-lmcache-nixl-gpu0`).
2. `host_setup_rocm_aic_exporter_vlh_host_data_root` must match the `DATA=`
   mount (default `/mnt/lmcache-nvme`).
3. Re-apply the textfile exporter:
   `ansible-playbook site.yml --tags rocm_aic_exporter`.
4. Optional vLLM/LMCache scrape jobs on the monitoring server — set in
   `inventory/group_vars/monitoring_server.yml`:
   `monitoring_scrape_vllm_enabled: true` and/or
   `monitoring_scrape_lmcache_enabled: true`, then
   `ansible-playbook site.yml --tags monitoring`.

### ROCm inference stack image (`recipies/rocm-inference-stack`)

Manual build:

```bash
cd recipies/rocm-inference-stack
DOCKER_BUILDKIT=1 docker buildx build -f Dockerfile -t rocm-inference-stack:latest .
```

Ansible build and deploy to `gpu_nodes`:

```bash
cd ansible
ansible-playbook playbooks/inference-image.yml --tags inference-image-build
ansible-playbook playbooks/inference-image.yml --tags inference-image-deploy
```

Default build context: `recipies/rocm-inference-stack` (see
`inference_container_context_path` in `group_vars/gpu_nodes.yml`).

### llm-d Kubernetes (`recipies/llm-d`)

Not managed by Ansible. One-time setup and tiered-prefix-cache deploy:

```bash
cd recipies/llm-d/setup && ./prereqs.sh -y && just llm-d-setup
cd ../tiered-prefix-cache && just setup && just port-forward-start
```

Optional: federate in-cluster Prometheus into the lab server via
`monitoring_llmd_prometheus_federate_*` in `monitoring_server.yml`.

### Benchmark data (optional)

```bash
make -C recipies/vllm-lmcache-nixl data-all BOOK_DATA_ROOT=/path/to/gutenberg
```

## Monitoring — Prometheus remote write

The `monitoring_stack` role pushes metrics to an external endpoint when
`monitoring_prometheus_remote_write_url` is set. Configure in
`inventory/group_vars/monitoring_server.yml` or via `-e`:

```bash
ansible-playbook site.yml --tags monitoring \
  -e monitoring_prometheus_remote_write_url=https://mimir.example:9009/api/v1/push
```

See `roles/monitoring_stack/defaults/main.yml` for authentication and transport security variables.
The remote-write **receiver** (`--web.enable-remote-write-receiver`) is
opt-in (`monitoring_prometheus_enable_remote_write_receiver: true`).

Grafana cluster summary sync from the repo root: `./deploy.sh apply`.

[galaxy]: https://galaxy.ansible.com/ui/repo/published/sbates130272/batesste/
