<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Grafana dashboards

Sample Grafana dashboards for ROCm-AIC monitoring. Import JSON from this
directory into your Grafana server (Dashboards → New → Import).

## `rocm-aic-dashboard.json`

**ROCm(tm) AMD Infinity Context Dashboard** — vLLM, LMCache, GPU, storage,
and `rocm_aic_*` textfile metrics from
[`rocm-aic-exporter.py`](../recipies/common/scripts/rocm-aic-exporter.py).

Expects Prometheus scrape jobs named `node_exporter`, `nvme_exporter`,
`vllm-exporter`, `lmcache-exporter`, and `amd_metrics_exporter` (as
provisioned by the Ansible `monitoring_stack` role).

### Local vs NFS AIC storage

The AIC cache tier may be **local** (NVMe, dm-crypt, hipfile on block
device) or **NFS-mounted** depending on the deployment. The dashboard supports
both:

| Storage mode | Primary panels | Optional / NFS-only |
| --- | --- | --- |
| Local NVMe, hipfile, direct block | KV Block I/O, `rocm_aic_*` gauges, RDMA (if used) | NFS panels show **No data** (expected) |
| NFS-backed cache | Same `rocm_aic_*` panels plus NFS client when configured | Set **`nfs_mount`** to the cache mount path |

Leave **`nfs_mount`** at the default **`^$`** when AIC is not NFS-backed. That
disables NFS client series cleanly instead of showing misleading empty queries.
Set **`nfs_mount`** to your mount path (for example `/mnt/rocm-icms-cache`)
only when `rocm_aic_nfs_mount_*` metrics exist.

The **NFS Server** panel only applies to nodes that **export** the cache; GPU
clients running local or NFS client mounts will not populate server metrics.

### Variables

| Variable | Purpose |
| --- | --- |
| `datasource` | Prometheus data source (hidden; pick on import if needed) |
| `host` | node_exporter + amd_metrics_exporter instance (inventory hostname on Ansible clusters) |
| `vllm_instance` | vLLM exporter target (`host:port`) for latency and token metrics |
| `model` | LLM `model_name` for vLLM panels |
| `gpu_id` | GPU id from `amd_metrics_exporter` |
| `blkdevice` | Block device(s) for local KV cache block I/O panel |
| `rdma_device` | RDMA port(s) when storage uses RoCE (optional) |
| `nfs_mount` | Optional NFS client mount regex; default `^$` = not NFS |
| `server_power_metric` | Optional PDU metric regex; default `^$` disables |

### Per-cluster setup

1. Import `rocm-aic-dashboard.json` and select your Prometheus data source.
2. Set **`host`** to the Prometheus `instance` label for your machine (same value
   on Ansible clusters for node and GPU scrapes). Set **`vllm_instance`** to the
   vLLM scrape target (`host:port`) — always separate from **`host`**.
3. **Local AIC:** leave **`nfs_mount`** as **`^$`**. Pick **`blkdevice`** for
   the backing volume on the KV Block I/O panel.
4. **NFS AIC:** set **`nfs_mount`** to your cache mount path. NFS client
   panels populate when `rocm_aic_nfs_mount_*` metrics are scraped.
5. Set **`server_power_metric`** when a site PDU metric exists (for example
   `snoc_pinewood_plug_d_power`). Default **`^$`** disables server power;
   GPU power still shows.
6. Pick **`rdma_device`** when RDMA storage panels should filter to your NIC
   port names.

Most query variables default to **All** on load so panels populate before you
narrow scope. Pick a single **Host** or **vLLM Instance** when comparing
machines.

### Overview row

The top stat row summarizes cluster health:

- **Online GPUs** — GPUs on selected nodes
- **TTFT p99** — tail latency for selected vLLM targets
- **AIC Prefix Hit Ratio** — external KV cache effectiveness
- **KV Chunk Bytes** — on-disk LMCache footprint
- **AIC Storage Free %** — data filesystem headroom

### Git workflow (normalize metadata)

Grafana UI exports include server-owned fields (`resourceVersion`,
`grafana.app/updatedBy`, internal numeric IDs, and so on). Before committing,
run from the repo root:

```bash
make grafana-normalize
```

That runs [`scripts/normalize-dashboard.py`](scripts/normalize-dashboard.py),
which:

- Keeps a stable **`metadata.uid`** (never auto-regenerated on normalize).
- Strips **`resourceVersion`**, **`generation`**, **`labels`**, and
  **`grafana.app/*`** annotations.
- Optionally records **`rocm-aic.git.revision`**, **`rocm-aic.git.author`**, and
  **`rocm-aic.git.normalizedAt`** (Grafana may ignore these; git is the audit
  trail).

CI runs `make grafana-check` (same script with `--check`) on pull requests
that touch `grafana/**`.

After editing in the Grafana UI, export or API-pull the dashboard, then
normalize before commit. Assign a new UID only once:

```bash
python3 grafana/scripts/normalize-dashboard.py --ensure-uid
```

Classic JSON dashboards (top-level `panels`, no `apiVersion`) are also
supported: normalize removes the Grafana-assigned numeric **`id`** only.
