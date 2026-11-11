# Quick Start

Build the images, start the stack, run the KV-cache cliff benchmark and plot
the result. Run every command from the repository root, on a host that meets
the [prerequisites](../README.md#prerequisites). `make help` lists all targets,
and [ENVIRONMENT.md](ENVIRONMENT.md) documents every variable.

## 1. Build the images

```bash
make build ROCM_ARCH=gfx950
```

This builds `aic-base` (PyTorch and shared dependencies), then `aic-vllm` and
`aic-lmcache`, which inherit from it. Set `ROCM_ARCH` to your GPU architecture;
the build compiles kernels for exactly the architectures you list.

## 2. Start the stack

```bash
make up \
    HF_TOKEN=hf_... \
    NVME_DATA=/mnt/lmcache-nvme \
    NFS_DATA=/mnt/lmcache-nfs \
    VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

This starts LMCache, vLLM and the monitoring stack, with DRAM as the first cache
tier and NVMe and NFS as the second. `make up-batch` runs it in the background;
`make logs` follows the logs, `make endpoints` prints the Grafana URL, and
`make down` stops everything.

`AIC_STACK` names the stack: the compose project, its private Docker bridge and
the prefix of every container name (`<stack>-lmcache`, ...). Inside a Slurm job
it defaults to `aic-<job id>`, so jobs sharing a node never meet. To run a
second stack on one host by hand, give it its own name and keep passing it,
e.g. `make up AIC_STACK=aic-mine` and later `make down AIC_STACK=aic-mine`;
`make up` refuses a name another checkout's stack is already using.

The default NVMe tier uses
[AMD Infinity Storage (AIS)](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-10.1-blog/README.html),
peer-to-peer DMA between NVMe and the GPU through the
[hipFile](https://rocm.docs.amd.com/projects/hipFile/en/latest/) library. If
LMCache exits at startup with a `hipFileBufRegister` error, your host does not
support it; add `AIC_L2_BACKEND=nixl_posix` to use ordinary file I/O instead.

To use an AIS NVMe slab as the first tier instead:

```bash
make up-gds-l1 HF_TOKEN=hf_... GDS_SLAB_DATA=/mnt/lmcache-nvme VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

## 3. Run the cliff benchmark

The cliff benchmark compares a plain vLLM against vLLM with AIC as the number of
concurrent long-context requests grows. Each arm needs its own stack, so stop
one with `make down` before starting the next. The benchmark runs in the
`aic-client` container and finds the running vLLM by itself.

```bash
export HF_TOKEN=hf_... VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# AIC: the tiered stack from step 2
make up-batch NVME_DATA=/mnt/lmcache-nvme NFS_DATA=/mnt/lmcache-nfs
make cliff BENCH_ARM=kvd_v2
make down

# Baseline: vLLM with no LMCache
make up-vllm-only
make cliff BENCH_ARM=vram_only
make down

# KVBench: starts its own server
make cliff-kvbench-local KVBENCH_MODEL=llama-3.1-8b
make kvbench-down
```

Results land in `logs/manual/results/`, one CSV per run. `BENCH_CONCUR` sets the
concurrency levels to sweep. The same `aic-client` container runs the other
benchmarks: `make bench-list` lists them, and
[`benchmarks/README.md`](../benchmarks/README.md) describes them.

## 4. Plot the results

```bash
make venv    # once: creates .venv with the plotting dependencies
make plot
```

Charts are written to `logs/manual/plots/` (`cliff-throughput.png`,
`cliff-latency-p50.png`, `cliff-latency-p95.png`).

## Next steps

- [SLURM_SPUR.md](SLURM_SPUR.md): full sweeps on a Slurm cluster.
- [METRICS_TELEMETRY.md](METRICS_TELEMETRY.md): Prometheus metrics and dashboards.
- [EMULATE.md](EMULATE.md): run without a GPU using the CPU emulator.
