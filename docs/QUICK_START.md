# Quick Start

All commands run from the repo root with `make <target>`.

## 1. Build the images

```bash
make build ROCM_ARCH=gfx942
```

This builds three images in sequence: `aic-base` (PyTorch + torchvision), then
`aic-vllm` and `aic-lmcache`, which both inherit from it.

> **Low on RAM?**
>
> ```bash
> make build ROCM_ARCH=gfx942 BUILD_JOBS=8
> ```
>
> **Rebuilding often?** Set `AIC_CACHE_DIR` to persist BuildKit layers across
> `docker system prune`, so the expensive PyTorch stage is not recompiled:
>
> ```bash
> make build ROCM_ARCH=gfx942 AIC_CACHE_DIR=~/.cache/rocm-aic-buildx
> ```
>
> The build stages a multi-GB OCI copy of `aic-base` under `TMPDIR`; point
> `TMPDIR` at a real filesystem if `/tmp` is a small tmpfs.

## 2. Start the stack (standard mode: DRAM L1 + NVMe L2a + NFS L2b)

```bash
make up \
    HF_TOKEN=hf_... \
    NVME_DATA=/mnt/lmcache-nvme \
    NFS_DATA=/mnt/lmcache-nfs \
    VLLM_MODEL=openai/gpt-oss-120b
```

## 3. Start in GDS L1 mode (hipFile NVMe slab as L1, no L2)

```bash
make up-gds-l1 \
    HF_TOKEN=hf_... \
    GDS_SLAB_DATA=/mnt/lmcache-nvme \
    VLLM_MODEL=openai/gpt-oss-120b
```

## 4. Ports, and running LMCache without compose

`make up` puts the containers on a private docker bridge (`aic`). The ports
below are container ports there, and the few that reach the host are published
on `127.0.0.1` with a free port Docker picks (`make endpoints` prints them), so
the stack never claims a fixed host port. If your environment offers only host
networking (or you run `lmcache server` by hand), the processes share the host's
port space and every port below must be chosen explicitly.

| Port | Flag / variable | Compose default (container port) |
| --- | --- | --- |
| LMCache ZMQ (vLLM connector) | `--port`, `LMCACHE_PORT` | 6555 |
| LMCache HTTP API | `--http-port` | 8080 |
| LMCache Prometheus `/metrics` | `--prometheus-port` | 9091 |
| NIXL telemetry (NIXL L2 only) | `NIXL_METRICS_PORT` | 19090 |
| vLLM OpenAI API | `--port` | 8000 |

> **Always pass `--prometheus-port`.** The AIC image patches LMCache to serve
> `/metrics` from a dedicated listener on that port, and LMCache's own default
> is **9090**. Leave it unset on a shared host and the server dies at startup
> with `OSError: [Errno 98] Address already in use` from
> `prometheus_client.start_http_server`, no matter what `--http-port` is.
> `--disable-metrics` also avoids the error, but turns off every
> `lmcache_mp_*` metric.

> **Also pass `--http-host 0.0.0.0`** if anything outside the container needs
> the HTTP API: LMCache binds it to `127.0.0.1` by default.

A DRAM-only server with the same flags CI uses:

```bash
lmcache server \
    --no-separate-object-groups \
    --host 0.0.0.0 \
    --http-host 0.0.0.0 \
    --http-port 8080 \
    --port 6555 \
    --l1-size-gb 20 \
    --eviction-policy LRU \
    --max-gpu-workers 1 \
    --worker-reap-timeout-seconds 3600 \
    --prometheus-port 9091
```

The flags for the L2 configurations (NVMe, NFS, `nixl_posix`) are in the
`x-lmcache-command` block of `docker/docker-compose.yml`; copy from there rather
than reconstructing them. Point vLLM at the ZMQ port with the `lmcache.mp.port`
entry of its `--kv-transfer-config` (the `LMCacheMPConnector` JSON in the
`Makefile`).

## 5. Install host-side benchmark dependencies

```bash
make venv
source .venv/bin/activate
```

## 6. Run the cliff benchmark

Run the baseline arm first, then the AIC arm against separate endpoints. Host
ports are ephemeral, so `make endpoints` prints the URL of each running service;
`BENCH_ENDPOINT` defaults to the running vLLM's URL when you leave it out.

```bash
# Arm A: baseline (vram_only) — must be run against a plain vLLM endpoint
make cliff \
    BENCH_ARM=vram_only \
    BENCH_ENDPOINT=<vLLM URL from 'make endpoints'> \
    BENCH_MODEL=openai/gpt-oss-120b

# Arm B: AIC (kvd_v2) — run against the vllm-lmcache stack
make cliff \
    BENCH_ARM=kvd_v2 \
    BENCH_MODEL=openai/gpt-oss-120b

# Arm C: KVBench — build/start the compose KVBench container, then bench it
# over the internal docker network (the cliff harness automatically skips
# vLLM-only prefix-cache metrics for this arm)
make cliff-kvbench-local \
    KVBENCH_MODEL=llama-3.1-8b \
    BENCH_CONCUR=1,2
```

## 7. Generate cliff charts

```bash
make plot
# → logs/manual/plots/cliff-throughput.png
# → logs/manual/plots/cliff-latency-p50.png
# → logs/manual/plots/cliff-latency-p95.png
```
