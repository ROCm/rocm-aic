# ROCm AMD Infinity Context — AAI Day Release

Self-contained inference stack and benchmarking bundle for the AMD AAI Day (July 2026) demonstration of the **ROCm AMD Infinity Context** platform.

## Stack overview

```text
Ubuntu 24.04  (rocm/dev-ubuntu-24.04:7.2.4-complete, ROCm 7.2.4, Python 3.12)
  └── vLLM v0.25.0+rocm723  (pre-built wheel — bundles torch/triton/flash-attn)
        └── LMCacheMPConnector (ZMQ)
              └── LMCache server (standalone MP mode)  [dev @ 21b3341 + 7 AMD patches]
                    ├── L1:  GPU / CPU DRAM   (--l1-size-gb)
                    │    or  hipFile NVMe slab (GDS L1 mode)
                    ├── L2a: NIXL AIS_MT → local NVMe   (hipFile P2PDMA, GDS)
                    └── L2b: NIXL POSIX  → NFS-over-RDMA
```

Component versions (pinned SHAs — update to latest branch heads before each release):

| Component | Source | Ref |
| --- | --- | --- |
| Base OS | `rocm/dev-ubuntu-24.04:7.2.4-complete` | Ubuntu 24.04, ROCm 7.2.4, Python 3.12 |
| vLLM | `wheels.vllm.ai/rocm/0.25.0/rocm723` | v0.25.0+rocm723 (pre-built wheel, bundles torch) |
| LMCache | `LMCache/LMCache` (upstream) | `dev` @ `21b3341` (> v0.5.1) + 7 AMD patches |
| NIXL | `ai-dynamo/nixl` (upstream) | `main` @ `644facf0` + `nixl-rocm-ais-mt.patch` |
| hipFile | `ROCm/rocm-systems` | `develop` @ `6901b670` |

## Pip-installable nightly wheels

The patched **LMCache** and **ROCm NIXL** are published as pip wheels on a rolling
`nightly` GitHub Release, rebuilt automatically whenever the `aai-day` branch
changes (see `.github/workflows/aai-day-release-nightly-wheels.yml`). Install them
into a matching ROCm environment:

```bash
pip install \
  https://github.com/ROCm/rocm-aic/releases/download/nightly/lmcache-<ver>-cp312-cp312-linux_x86_64.whl \
  https://github.com/ROCm/rocm-aic/releases/download/nightly/nixl_rocm-<ver>-cp312-cp312-linux_x86_64.whl
```

(Grab the exact filenames from the [nightly release](https://github.com/ROCm/rocm-aic/releases/tag/nightly).)

**Compatibility — read before installing.** These are **not** manylinux wheels:

- ROCm **7.2.x**, Python **3.12**, **x86_64** only. They match the
  `rocm/dev-ubuntu-24.04:7.2.4-complete` base; other ROCm/Python/arch combos will
  fail to import.
- The wheels are built for the image's full multi-arch set (`gfx90a … gfx1201`);
  LMCache's HIP extension is compiled for all of them.
- The `nixl_rocm` wheel bundles `libnixl` + the NIXL/UCX plugin `.so`s, but the
  **ROCm runtime (`libamdhip64`) and hipFile are external dependencies** — they
  must already be present on the host (they are, inside this image). It installs
  the `nixl_rocm` import package; the `nixl` compatibility shim is applied only
  inside the image.

These wheels are a convenience for reproducing the stack outside the container;
the supported deployment is still the Docker image built below.

> The wheels are produced by the `wheels` stage of the Dockerfile
> (`docker build --target wheels --output type=local,dest=./wheels aai-day-release`);
> the default build target is unchanged and still yields the full runtime image.

## Prerequisites

- ROCm-capable host (MI300X recommended; `gfx942`). The default build is
  multi-arch — it bakes in every gfx the vLLM wheel supports (`gfx90a`, `gfx942`,
  `gfx950`, and the RDNA `gfx11xx`/`gfx12xx` line), so one image runs on any of
  them. Narrow with `ROCM_ARCH=gfx942` for a faster single-arch build.
- Docker with BuildKit and the `docker compose` plugin (Docker 23+)
- Host mounts: local NVMe (`NVME_DATA`) and NFS-over-RDMA (`NFS_DATA`) pre-mounted
- HuggingFace token with access to the target model
- Python 3.10+ for host-side benchmarks

### AMD network (Zscaler proxy)

Pass your corporate CA cert as a BuildKit secret — it is mounted read-only during the
build and is **never baked into the image or any cache layer**:

```bash
make -C aai-day-release build TLS_CERT=/path/to/corp-ca.crt
```

Or with plain `docker build`:

```bash
# Context is the aai-day-release tree (the Dockerfile COPYs scripts/, patches/,
# benchmarks/, monitoring/ from it); the Dockerfile itself lives in docker/.
DOCKER_BUILDKIT=1 docker build \
  --build-arg ROCM_ARCH=gfx942 \
  --secret id=tls_cert,src=/path/to/corp-ca.crt \
  -f aai-day-release/docker/Dockerfile \
  -t rocm-aic-aai-day aai-day-release
```

Omit `TLS_CERT` / `--secret` entirely when building outside AMD's network.

## Quick start

All commands run from the repo root with `make -C aai-day-release <target>`, or from inside `aai-day-release/` with `make <target>`.

### 1. Build the image

```bash
make -C aai-day-release build ROCM_ARCH=gfx942
```

### 2. Start the stack (standard mode: DRAM L1 + NVMe L2a + NFS L2b)

```bash
make -C aai-day-release up \
    HF_TOKEN=hf_... \
    NVME_DATA=/mnt/lmcache-nvme \
    NFS_DATA=/mnt/lmcache-nfs \
    VLLM_MODEL=openai/gpt-oss-120b
```

### 3. Start in GDS L1 mode (hipFile NVMe slab as L1, no L2)

```bash
make -C aai-day-release up-gds-l1 \
    HF_TOKEN=hf_... \
    GDS_SLAB_DATA=/mnt/lmcache-nvme \
    VLLM_MODEL=openai/gpt-oss-120b
```

### 4. Install host-side benchmark dependencies

```bash
make -C aai-day-release venv
source .venv/bin/activate
```

### 5. Run the cliff benchmark

Run the non-AIC baseline arm first, then the AIC arm against separate endpoints:

```bash
# Arm A: baseline (vram_only) — must be run against a plain vLLM endpoint
make -C aai-day-release cliff \
    BENCH_ARM=vram_only \
    BENCH_ENDPOINT=http://localhost:8001 \
    BENCH_MODEL=openai/gpt-oss-120b

# Arm B: AIC (kvd_v2) — run against the vllm-lmcache stack on port 8000
make -C aai-day-release cliff \
    BENCH_ARM=kvd_v2 \
    BENCH_ENDPOINT=http://localhost:8000 \
    BENCH_MODEL=openai/gpt-oss-120b
```

### 6. Generate cliff charts

```bash
make -C aai-day-release plot
# → logs/manual/plots/cliff-throughput.png
# → logs/manual/plots/cliff-latency-p50.png
# → logs/manual/plots/cliff-latency-p95.png
```

### 7. Slurm build & cliff sweep (multi-node)

Sections 1–6 drive the stack locally. On a Slurm cluster, the `dist-*` / `cliff-*`
targets build/distribute the image and run the full three-arm sweep end-to-end
(they wrap `.slurm/run-build-distribute.sh` and `sbatch .slurm/run-cliff.sbatch`):

```bash
# Build the image (+ fabric exporters) on a CPU build node and save tarballs to
# the shared /scratch image dir; chain push + smoke-test like the old run-this.sh:
make -C aai-day-release dist-build dist-push smoke-test AAI_PUSH_REF=<registry>/aai-day:latest

# Submit the full sweep (vram_only + kvd_v2 nvme + kvd_v2 gds) on a GPU+NVMe node.
# Output lands in logs/<job-id>/. Pin a node / narrow arms / override the sweep via env:
make -C aai-day-release cliff-submit AAI_CLIFF_NODE=ctr-s95-mi300x-3
make -C aai-day-release cliff-submit AAI_CLIFF_ARMS=nvme BENCH_CONCUR=1,8,64
make -C aai-day-release cliff-short          # 1-point smoke test of the whole flow
```

`smoke-test` validates the *image* on a GPU+NVMe node (GPU/arch, vLLM + LMCache
imports, `ais-check`, `nvme list`, the NIXL AIS_MT plugin). After those checks it
also stands up the full exporter fleet + Prometheus (the same
`monitoring/monitoring-lib.sh` the cliff uses), scrapes briefly, health-checks
each `/metrics` endpoint, and leaves a TSDB under `logs/<job-id>/prometheus` to
sanity-check — all **informational** (only the in-image checks affect the exit
code). Tune with `AAI_SMOKE_EXPORTERS=0` (skip) and `AAI_SMOKE_SCRAPE_S=<secs>`
(default 45).

## Metrics & observability

A host-network Prometheus sidecar can capture the whole run so it can be
explored afterward. It scrapes, all at `localhost`:

| Source | Port | Notes |
| --- | --- | --- |
| vLLM `/metrics` | 8000, 8001 | 8000 = kvd arm, 8001 = vram_only baseline |
| LMCache `/metrics` | 8080 | includes NIXL-backed tier counters |
| NIXL telemetry `/metrics` | 19090 | native NIXL exporter on the LMCache process (see below) |
| node_exporter | 9100 | CPU/mem/net + **NVMe I/O** (diskstats/nvme) + **RDMA** (infiniband) |
| nvme_exporter | 9998 | dedicated NVMe exporter (batesste host service, or container — see below) |
| rdma_exporter | 9879 | dedicated RDMA exporter (batesste host service, or container — see below) |
| amd_metrics_exporter | 5000 | AMD GPU device-metrics-exporter (`amd_*` metrics) |
| hsa_snoop | 9488 | HSA AQL queue/dispatch telemetry (`hsa_kernel_launches_total`, `hsa_kernel_duration_seconds`, `hsa_active_queues`) |

The TSDB is written to `AAI_METRICS_DIR`. For cliff sbatch runs (`make
cliff-submit`) it defaults to
`logs/<job-id>/prometheus` (the SLURM job id, or `manual` off-Slurm);
**bind-mount / point it at an NFS directory** to explore the capture later by
pointing a Prometheus at it. Job names/ports
mirror the batesste Ansible monitoring roles, so the same Grafana dashboards
and recording rules apply.

**NIXL native telemetry (`:19090`).** NIXL ships its own (experimental/beta)
Prometheus exporter — `agent_tx_bytes_total`, `agent_errors_total{status=...}`,
etc. It's compiled into the image (the `prometheus-cpp` plugin is built by
[scripts/nixl/build-nixl.sh](scripts/nixl/build-nixl.sh)) and enabled at runtime
on the **LMCache** process — the one that runs the NIXL agent — via
`NIXL_TELEMETRY_ENABLE=y NIXL_TELEMETRY_EXPORTER=prometheus
NIXL_TELEMETRY_PROMETHEUS_PORT=19090` (set by default in `docker/docker-compose.yml`
and the cliff sbatch). Under LMCache MP mode only one worker process wins the
port; the rest run without a sink. Metric names may change between NIXL
versions. Set `NIXL_TELEMETRY_ENABLE=` (empty) to disable, or
`NIXL_METRICS_PORT` to move the port.

**During cliff sbatch runs** it is auto-started (see below). **Standalone / with
`make up`:**

```bash
# scrape-only: exporters already installed on the host (Ansible)
make -C aai-day-release monitoring-up AAI_METRICS_DIR=/mnt/lmcache-nfs/metrics

# bare node: also launch containerized node + AMD GPU exporters
make -C aai-day-release monitoring-up \
    AAI_METRICS_DIR=/mnt/lmcache-nfs/metrics AAI_EXPORTERS=1

make -C aai-day-release monitoring-down     # stop (TSDB retained)
```

`nvme_exporter` / `rdma_exporter` are normally host services (batesste galaxy
roles). For bare nodes without those services there are now container images too,
built from the same upstream release binaries so the `nvme_*` / `rdma_port_*`
series match:

> **Note:** these exporter images are **optional** and their Dockerfiles pull
> `debian:12-slim` from Docker Hub, so the build node needs registry egress
> (Zscaler proxy / DNS). `make dist-build` builds them after the main image but
> treats a failure as a non-fatal warning — the main image (which builds entirely
> from the shared `/scratch` BuildKit cache, no registry pull) is unaffected. Skip
> the step with `AAI_BUILD_EXPORTERS=0`; if the images are absent the cliff job
> falls back to host exporters / node-exporter's nvme+infiniband collectors.

```bash
# build both fabric-exporter images (works without the compose plugin)
make -C aai-day-release monitoring-build-exporters

# run them alongside the sidecar via the exporters-fabric compose profile
AAI_METRICS_DIR=/mnt/lmcache-nfs/metrics \
  docker compose -f aai-day-release/monitoring/docker-compose.monitoring.yml \
    --profile exporters --profile exporters-fabric up -d
```

For the cliff sbatch docker-run path (nodes without the compose plugin), set
`AAI_NVME_EXPORTER_IMAGE=aai-day-nvme-exporter:local` /
`AAI_RDMA_EXPORTER_IMAGE=aai-day-rdma-exporter:local` after building. Each
container is skipped automatically if a host service already serves its port.
On a node lacking both, NVMe I/O still comes from node-exporter's
`diskstats`/`nvme` collectors and RDMA from its `infiniband` collector.

**hsa-snoop (`:9488`).** [sbates130272/hsa-snoop](https://github.com/sbates130272/hsa-snoop)
is compiled into the `rocm-aic-aai-day` image with its Prometheus exporter
(`-DHSA_SNOOP_PROMETHEUS=ON`; see [docker/Dockerfile](docker/Dockerfile) step 6b — the build
fails if the resulting binary lacks `--prometheus`). The snooper is host-only
C++ and **architecture-independent** — one binary runs on every gfx target we
ship — so the build disables the optional HIP `examples/`
(`CMAKE_DISABLE_FIND_PACKAGE_hip=ON`) rather than pinning them to a single arch.
It runs in the `exporters` profile as `hsa-snoop --all --prometheus`, and because
it snoops HSA AQL queues from userspace (ftrace kprobe + pagemap +
`process_vm_readv`) it needs **`privileged: true`, `pid: host`, and root** to see
the vLLM/LMCache GPU processes. It's a sampling snoop (very short kernels between
poll intervals can be missed) and is upstream-verified on gfx90a / ROCm 7.1.0.

> **NFS caveat:** Prometheus' TSDB uses `mmap` + POSIX file locks, which NFS
> handles poorly. Keep to a single writer; this is fine for lab/demo capture,
> not a durable production store.

## Key environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `ROCM_ARCH` | auto-detected | GPU arch, e.g. `gfx942` |
| `HF_TOKEN` | — | HuggingFace access token (required) |
| `VLLM_MODEL` | `openai/gpt-oss-120b` | Model to serve |
| `NVME_DATA` | `/mnt/lmcache-nvme` | Host path for NVMe L2a pool |
| `NFS_DATA` | `/mnt/lmcache-nfs` | Host path for NFS-over-RDMA L2b pool |
| `GDS_SLAB_DATA` | — | Host path for GDS NVMe slab (GDS L1 mode only) |
| `GDS_MODE` | — | Set to `1` to enable GDS L1 mode |
| `LMCACHE_L1_SIZE_GB` | `20` | L1 memory cap in GiB |
| `LMCACHE_NVME_POOL` | `4096` | NIXL pool slots for NVMe adapter |
| `LMCACHE_NFS_POOL` | `1024` | NIXL pool slots for NFS adapter |
| `TENSOR_PARALLEL_SIZE` | `1` | vLLM tensor parallel degree |
| `GPU` | `0` | ROCR_VISIBLE_DEVICES for the vllm container |
| `AAI_MONITORING` | `1` | Auto-start the Prometheus sidecar in cliff sbatch runs (`0` to skip) |
| `AAI_METRICS_DIR` | `logs/<job-id>/prometheus` (cliff) | Prometheus TSDB dir — bind-mount an NFS path here |
| `AAI_EXPORTERS` | `1` (cliff) / `0` (make) | Also launch containerized node + AMD GPU exporters |

## Directory layout

```text
aai-day-release/
├── Makefile                # Local stack + benchmark + distribute/cliff targets
├── pyproject.toml          # Host-side bench + plot Python deps
├── docker/
│   ├── Dockerfile          # ROCm inference stack image (build context = tree root)
│   └── docker-compose.yml  # lmcache + vllm services (standard + GDS L1)
├── benchmarks/
│   ├── run_cliff.py        # KV cache cliff benchmark runner
│   └── plot_cliff.py       # Cliff chart plotter (matplotlib)
├── .slurm/                 # Slurm entrypoints (see `make dist-* / cliff-*`)
│   ├── run-cliff.sbatch    # Full 3-arm cliff sweep as a batch job
│   └── run-build-distribute.sh  # Build/push/test the image across nodes
├── scripts/
│   └── nixl/               # NIXL clone/build helpers + defaults.mk (used by Dockerfile)
├── patches/
│   ├── lmcache/            # LMCache source patches applied at build
│   └── nixl/               # NIXL ROCm/AIS_MT patch applied at build
├── monitoring/             # Prometheus metrics-capture sidecar
│   ├── docker-compose.monitoring.yml
│   ├── prometheus/
│   │   ├── prometheus.yml  # localhost scrape config (vLLM/LMCache/exporters)
│   │   └── rules/aai_day.yml
│   ├── amdgpu-exporter/config.json
│   ├── nvme-exporter/      # containerized NVMe exporter (bare-node fallback)
│   ├── rdma-exporter/      # containerized RDMA exporter (bare-node fallback)
│   └── ais-snoop/          # AIS/hipFile KFD kprobe exporter
├── logs/                   # Per-job output: logs/<job-id>/{cliff.out,results,plots,prometheus} (gitignored)
└── certs/
    └── corp-ca.crt         # Gitignored; add AMD/Zscaler CA cert here (create as needed)
```

## Upgrading component versions

To pick up the latest commits from each AMD branch before a release:

```bash
# Check latest SHAs for pinned components:
git ls-remote https://github.com/LMCache/LMCache.git refs/tags/operator-v0.5.0
git ls-remote https://github.com/ai-dynamo/nixl.git refs/heads/main
git ls-remote https://github.com/ROCm/rocm-systems.git refs/tags/rocm-7.2.3

# Then update LMCACHE_SHA, NIXL_SHA, HIPFILE_SHA, VLLM_VERSION, and
# the FROM base image tag in docker/Dockerfile.
```

## Related recipes

- [`recipies/vllm-lmcache-mp/`](../recipies/vllm-lmcache-mp/) — development recipe this bundle is based on
- [`benchmarks/kv-cache-cliff/`](../benchmarks/kv-cache-cliff/) — upstream cliff benchmark (canonical source for `run_cliff.py`)
- [`docs/aai-day/`](../docs/aai-day/) — blog post and white paper
