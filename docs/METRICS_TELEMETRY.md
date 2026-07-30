# Metrics & Observability

A host-network Prometheus sidecar captures the whole run for post-hoc
exploration.  It scrapes the following targets, all at `localhost`:

| Source | Port | Notes |
| --- | --- | --- |
| vLLM `/metrics` | 8000, 8001 | 8000 = kvd arm, 8001 = vram\_only baseline |
| LMCache `/metrics` | **8080** | HTTP API frontend; includes lmcache\_mp\_\* tier counters (see below) |
| NIXL telemetry `/metrics` | 19090 | native NIXL exporter on the LMCache process (see below) |
| node\_exporter | 9100 | CPU/mem/net + **NVMe I/O** (diskstats/nvme) + **RDMA** (infiniband) |
| nvme\_exporter | 9998 | dedicated NVMe exporter (host service, or container — see below) |
| rdma\_exporter | 9879 | dedicated RDMA exporter (host service, or container — see below) |
| amd\_metrics\_exporter | 5000 | AMD GPU device-metrics-exporter (`amd_*` metrics) |
| hsa\_snoop | 9488 | HSA AQL queue/dispatch + AIS P2P storage telemetry (see below) |

The TSDB is written to `AIC_METRICS_DIR`.  For cliff sbatch runs
(`make cliff-submit`) it defaults to `logs/<job-id>/prometheus`; **bind-mount /
point it at an NFS directory** to explore afterward by pointing a Prometheus at
it.  Job names/ports mirror the Ansible monitoring roles so the same Grafana
dashboards and recording rules apply.

---

## LMCache metrics (`:8080`)

LMCache v0.5.x exposes its Prometheus metrics through the FastAPI HTTP API
server at `--http-port` (default **8080**), not on a separate dedicated port.
The OpenTelemetry initialisation log says:

> `OTel MeterProvider initialised with Prometheus fallback (standalone metrics
> HTTP server disabled; /metrics must be exposed by the caller)`

So `/metrics` lives at `localhost:8080/metrics` and is scraped by the `lmcache`
Prometheus job.  Key LMCache counter families:

| Metric | Meaning |
| --- | --- |
| `lmcache_mp_l1_usage_ratio` | DRAM L1 fill fraction (0–1) |
| `lmcache_mp_l1_memory_usage_bytes` | Bytes in the DRAM L1 pool |
| `lmcache_mp_l1_eviction_loop_ticks_total` | L1 LRU eviction cycles (non-zero → L1 overflowing to L2) |
| `lmcache_mp_l2_prefetch_lookup_requests_total` | L2 prefetch attempts triggered |
| `lmcache_mp_l2_prefetch_lookup_objects_chunks_total` | Chunks looked up in L2 |
| `lmcache_mp_l2_prefetch_hit_chunks_total` | Chunks successfully retrieved from L2 |
| `lmcache_mp_l2_usage_bytes` | Bytes written to the L2 pool |

> **Note:** `vllm:external_prefix_cache_hits_total` (vLLM side) counts hits
> only when the KV arrives *before* the request completes its prefill.  The
> NIXL `agent_rx_bytes_total` counter is more reliable for confirming that L2
> reads are actually happening at the transfer layer.

---

## NIXL telemetry (`:19090`)

NIXL ships an experimental Prometheus exporter — `agent_tx_bytes_total`
(bytes written to NVMe L2), `agent_rx_bytes_total` (bytes read back),
`agent_tx/rx_requests_num_total`, `agent_xfer_time_total`, etc.  It is
compiled into the image (`prometheus-cpp` plugin in
[docker/scripts/build-nixl.sh](../docker/scripts/build-nixl.sh)) and enabled
on the LMCache process by default:

```
NIXL_TELEMETRY_ENABLE=y
NIXL_TELEMETRY_EXPORTER=prometheus
NIXL_TELEMETRY_PROMETHEUS_PORT=19090
```

Under LMCache MP mode only one worker process wins the port; others run without
a sink.  Metric names may change between NIXL versions.  Set
`NIXL_TELEMETRY_ENABLE=` (empty) to disable, or `NIXL_METRICS_PORT` to move
the port.

---

## Exporter modes (`AIC_EXPORTERS`)

The `AIC_EXPORTERS` knob controls which containerised exporters are launched
alongside Prometheus:

| Value | What starts | Use case |
| --- | --- | --- |
| `0` | Nothing — Prometheus only | Minimal; rely on host-installed exporters |
| `safe` | amdgpu-exporter, rdma-exporter, hsa-snoop (container PID) | **SPUR default** — avoids `--pid=host` authz block |
| `1` | Full fleet: node-exporter, amdgpu-exporter, hsa-snoop (host PID), + fabric exporters if images present | Non-SPUR bare nodes |

The three compose profiles that back these modes:

| Profile | Services |
| --- | --- |
| `exporters` | node-exporter, amdgpu-exporter, hsa-snoop (`pid:host`) |
| `exporters-safe` | amdgpu-exporter, rdma-exporter, hsa-snoop (`pid:container`) |
| `exporters-fabric` | nvme-exporter, rdma-exporter |

**Usage:**

```bash
# scrape-only: exporters already installed on the host (Ansible)
make monitoring-up AIC_METRICS_DIR=/mnt/lmcache-nfs/metrics

# safe mode: amdgpu + rdma + hsa-snoop via container PID (SPUR / authz-restricted)
make monitoring-up AIC_METRICS_DIR=/mnt/lmcache-nfs/metrics AIC_EXPORTERS=safe

# full fleet: node + amdgpu + hsa-snoop via host PID (bare node, no authz block)
make monitoring-up AIC_METRICS_DIR=/mnt/lmcache-nfs/metrics AIC_EXPORTERS=1

make monitoring-down     # stop (TSDB retained)
```

---

## hsa-snoop (`:9488`)

[sbates130272/hsa-snoop](https://github.com/sbates130272/hsa-snoop) is compiled
into the `rocm-aic` image (`-DHSA_SNOOP_PROMETHEUS=ON`; see
[docker/Dockerfile](../docker/Dockerfile)).  It exports HSA AQL queue/dispatch
metrics (`hsa_kernel_launches_total`, `hsa_kernel_duration_seconds`,
`hsa_active_queues`) and, from v1.0.0, AIS (AMD Infinity Storage) P2P storage
counters (`ais_rx_ops_total`, `ais_tx_bytes_total`, etc.).

**PID namespace:** hsa-snoop uses ftrace kprobes + pagemap + `process_vm_readv`
to snoop GPU processes.  Two modes are supported via `AIC_HSA_SNOOP_PID_MODE`:

| Mode | `AIC_HSA_SNOOP_PID_MODE` | When to use |
| --- | --- | --- |
| **Container PID** | `container:aic-lmcache` | **SPUR default** — vLLM uses `pid:service:lmcache` so the lmcache container already contains both the lmcache server and the vLLM EngineCore; hsa-snoop joining that namespace sees both without needing `pid:host` |
| **Host PID** | `host` | Non-SPUR bare nodes where `--pid=host` is permitted |

> **spur-authz note:** the SPUR cluster's Docker authz plugin blocks
> `--pid=host` as a host-namespace escape.  Use `AIC_EXPORTERS=safe` (the SPUR
> default) to launch hsa-snoop with `pid:container` instead.

---

## fabric exporters (nvme\_exporter / rdma\_exporter)

Normally host services (Ansible roles).  For bare nodes without them, container
images are built from the same upstream release binaries so the `nvme_*` /
`rdma_port_*` series match:

> **Note:** these exporter images are **optional** — their Dockerfiles pull
> `debian:12-slim` from Docker Hub so the build node needs registry egress.
> `make dist-build` builds them after the main image but treats failure as a
> non-fatal warning.  Skip with `AIC_BUILD_EXPORTERS=0`; if absent the cliff
> job falls back to host exporters / node-exporter's nvme+infiniband collectors.

```bash
# build both fabric-exporter images
make monitoring-build-exporters

# run via exporters-fabric compose profile
AIC_METRICS_DIR=/mnt/lmcache-nfs/metrics \
  docker compose -f monitoring/docker-compose.monitoring.yml \
    --profile exporters --profile exporters-fabric up -d
```

When fabric-exporter images are present on the node, set
`AIC_NVME_EXPORTER_IMAGE=aic-nvme-exporter:latest` /
`AIC_RDMA_EXPORTER_IMAGE=aic-rdma-exporter:latest` and the cliff sbatch enables
`exporters-fabric` automatically.

---

## Container log streaming

During each kvd arm the cliff sbatch streams `docker logs -f --timestamps` for
the lmcache and vLLM containers into:

```
logs/<job-id>/lmcache/lmcache-stream.log
logs/<job-id>/vllm/vllm-stream.log
```

These complement the Prometheus TSDB for fine-grained timing — store times per
token, NIXL error messages, and EngineCore stack traces are visible in real time
rather than only at teardown.

---

## Prometheus image pre-pull

On fresh nodes the `prom/prometheus:v2.55.1` image is pulled from Docker Hub
(~75 MB) before `start_monitoring()`.  The cliff sbatch now runs a `docker pull`
beforehand so the download is logged and is a no-op on nodes that already have
the image cached.

---

## NFS caveat

Prometheus' TSDB uses `mmap` + POSIX file locks, which NFS handles poorly.
Keep to a single writer — fine for lab/demo capture, not a durable production
store.
