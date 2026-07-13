# aai-day-release Ansible

Ansible playbooks to provision bare-metal GPU nodes and deploy the aai-day-release Docker Compose stack (vLLM + LMCache + hipFile + NIXL).

## Prerequisites

- Ansible 2.15 or later (`pip install ansible`)
- Target hosts running Ubuntu 22.04 or 24.04
- AMD GPU with ROCm drivers already installed (or set `host_setup_enable_rocm_install: true`)
- SSH access to all nodes with `sudo` rights

Install required Ansible collections:

```bash
ansible-galaxy collection install -r requirements.yml
```

## Quick start

1. Copy `inventory/hosts.yml` and replace all `192.0.2.x` placeholders with real IPs.
2. Set required group_vars (search for `# REPLACE`):
   - `inventory/group_vars/gpu_nodes.yml` — `rocm_arch`, `aai_day_vllm_model`, `aai_day_hf_token`
3. Run the full stack:

```bash
export HF_TOKEN=hf_...
ansible-playbook site.yml --tags "provision,build-image,deploy-stack"
```

4. Verify the API:

```bash
curl http://<node-ip>:8000/v1/models
```

## Playbook tags

| Tag | What it does |
|-----|-------------|
| `provision` | Install Docker, packages, node_exporter |
| `build-image` | Build aai-day image on controller, ship to nodes |
| `deploy-stack` | Render `.env`, run docker compose up |
| `monitoring` | Install Prometheus + Grafana |
| `nfs-rdma` | Configure NFS-over-RDMA (L2b KV cache tier) |
| `nvmeof` | Configure NVMe-oF (optional; local NVMe is simpler) |

Run individual tags:

```bash
ansible-playbook site.yml --tags provision
ansible-playbook site.yml --tags nfs-rdma   # explicitly opt in
```

## Pull a pre-built image instead of building

```bash
ansible-playbook playbooks/build-image.yml \
  -e aai_day_pull_image=true \
  -e aai_day_image_registry=ghcr.io/your-org \
  -e aai_day_image_tag=latest
```

## Key variables

See `inventory/group_vars/gpu_nodes.yml` for the full list.
Key variables you must set:

| Variable | Description |
|----------|-------------|
| `rocm_arch` | GPU arch string (e.g. `gfx942` for MI300X) |
| `aai_day_vllm_model` | HuggingFace model repo ID |
| `aai_day_hf_token` | HuggingFace access token |
| `aai_day_nvme_data` | Host path for NVMe L2a pool |
| `aai_day_nfs_data` | Host path for NFS L2b pool |

## NFS-over-RDMA (optional)

The LMCache L2b pool uses an NFS-over-RDMA mount for shared KV cache
across restarts. To provision it:

1. Set `nfs_rdma_server_ip` and `nfs_rdma_export_clients` in
   `group_vars/nfs_rdma_server.yml` and `group_vars/nfs_rdma_clients.yml`.
2. Add hosts to the `nfs_rdma_server` and `nfs_rdma_clients` inventory groups.
3. Run: `ansible-playbook site.yml --tags nfs-rdma`

With `nfs_rdma_plain_dir_mode: true` (default), the server simply exports
a directory — no block device or LVM is required.

## ROCm installation

By default, this role assumes ROCm is already installed. To install it:

```bash
ansible-playbook site.yml --tags provision -e host_setup_enable_rocm_install=true
```

ROCm installation is hardware and distro version dependent. Refer to the
[AMD ROCm installation guide](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html)
if automatic installation fails.
