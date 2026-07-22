# Metrics & Observability

A host-network Prometheus sidecar can capture the whole run so it can be
explored afterward. It scrapes, all at `localhost`:

| Source | Port | Notes |
| --- | --- | --- |
| vLLM `/metrics` | 8000, 8001 | 8000 = kvd arm, 8001 = vram_only baseline |
| LMCache `/metrics` | 8080 | includes NIXL-backed tier counters |
| NIXL telemetry `/metrics` | 19090 | native NIXL exporter on the LMCache process (see below) |
| node_exporter | 9100 | CPU/mem/net + **NVMe I/O** (diskstats/nvme) + **RDMA** (infiniband) |
| nvme_exporter | 9998 | dedicated NVMe exporter (host service, or container — see below) |
| rdma_exporter | 9879 | dedicated RDMA exporter (host service, or container — see below) |
| amd_metrics_exporter | 5000 | AMD GPU device-metrics-exporter (`amd_*` metrics) |
| hsa_snoop | 9488 | HSA AQL queue/dispatch telemetry (`hsa_kernel_launches_total`, `hsa_kernel_duration_seconds`, `hsa_active_queues`) |

The TSDB is written to `AIC_METRICS_DIR`. For cliff sbatch runs (`make
cliff-submit`) it defaults to `logs/<job-id>/prometheus` (the SLURM job id,
or `manual` off-Slurm); **bind-mount / point it at an NFS directory** to
explore the capture later by pointing a Prometheus at it. Job names/ports
mirror the Ansible monitoring roles, so the same Grafana dashboards and
recording rules apply.

**NIXL native telemetry (`:19090`).** NIXL ships its own (experimental/beta)
Prometheus exporter — `agent_tx_bytes_total`, `agent_errors_total{status=...}`,
etc. It's compiled into the image (the `prometheus-cpp` plugin is built by
[docker/scripts/build-nixl.sh](../docker/scripts/build-nixl.sh)) and enabled
at runtime on the **LMCache** process — the one that runs the NIXL agent —
via `NIXL_TELEMETRY_ENABLE=y NIXL_TELEMETRY_EXPORTER=prometheus
NIXL_TELEMETRY_PROMETHEUS_PORT=19090` (set by default in
`docker/docker-compose.yml` and the cliff sbatch). Under LMCache MP mode
only one worker process wins the port; the rest run without a sink. Metric
names may change between NIXL versions. Set `NIXL_TELEMETRY_ENABLE=` (empty)
to disable, or `NIXL_METRICS_PORT` to move the port.

**During cliff sbatch runs** it is auto-started.
**Standalone / with `make up`:**

```bash
# scrape-only: exporters already installed on the host (Ansible)
make monitoring-up AIC_METRICS_DIR=/mnt/lmcache-nfs/metrics

# bare node: also launch containerized node + AMD GPU exporters
make monitoring-up \
    AIC_METRICS_DIR=/mnt/lmcache-nfs/metrics AIC_EXPORTERS=1

make monitoring-down     # stop (TSDB retained)
```

`nvme_exporter` / `rdma_exporter` are normally host services (Ansible roles).
For bare nodes without those services there are container images built from the
same upstream release binaries so the `nvme_*` / `rdma_port_*` series match:

> **Note:** these exporter images are **optional** and their Dockerfiles pull
> `debian:12-slim` from Docker Hub, so the build node needs registry egress.
> `make dist-build` builds them after the main image but treats a failure as a
> non-fatal warning — the main image is unaffected. Skip with
> `AIC_BUILD_EXPORTERS=0`; if absent the cliff job falls back to host exporters
> / node-exporter's nvme+infiniband collectors.

```bash
# build both fabric-exporter images (plain docker build)
make monitoring-build-exporters

# run them via the exporters-fabric compose profile
AIC_METRICS_DIR=/mnt/lmcache-nfs/metrics \
  docker compose -f monitoring/docker-compose.monitoring.yml \
    --profile exporters --profile exporters-fabric up -d
```

The whole metrics path is `docker compose` (v2) only. The cliff sbatch and the
smoke-test run `ensure_compose` first, which installs the plugin user-locally
(`~/.docker/cli-plugins`, shared `$HOME`) on any node that lacks it — so there is
no longer a `docker run` sidecar fallback. When the fabric-exporter images are
present on the node, set `AIC_NVME_EXPORTER_IMAGE=aic-nvme-exporter:latest` /
`AIC_RDMA_EXPORTER_IMAGE=aic-rdma-exporter:latest` and the cliff enables the
`exporters-fabric` compose profile automatically. On a node without them, NVMe
I/O still comes from node-exporter's `diskstats`/`nvme` collectors and RDMA from
its `infiniband` collector.

**hsa-snoop (`:9488`).**
[sbates130272/hsa-snoop](https://github.com/sbates130272/hsa-snoop)
is compiled into the `rocm-aic` image with its Prometheus exporter
(`-DHSA_SNOOP_PROMETHEUS=ON`; see [docker/Dockerfile](../docker/Dockerfile)
step 6b — the build fails if the resulting binary lacks `--prometheus`).
The snooper is host-only C++ and **architecture-independent** — one binary
runs on every gfx target we ship — so the build disables the optional HIP
`examples/` (`CMAKE_DISABLE_FIND_PACKAGE_hip=ON`) rather than pinning them
to a single arch. It runs in the `exporters` profile as
`hsa-snoop --all --prometheus`, and because it snoops HSA AQL queues from
userspace (ftrace kprobe + pagemap + `process_vm_readv`) it needs
**`privileged: true`, `pid: host`, and root** to see the vLLM/LMCache GPU
processes. It's a sampling snoop (very short kernels between poll intervals
can be missed) and is upstream-verified on gfx90a / ROCm 7.1.0.

> **NFS caveat:** Prometheus' TSDB uses `mmap` + POSIX file locks, which NFS
> handles poorly. Keep to a single writer; this is fine for lab/demo capture,
> not a durable production store.
