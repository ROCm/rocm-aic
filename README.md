# ROCm AMD Infinity Context

[![MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ROCm/rocm-aic/blob/main/LICENSE.md)
[![Platform](https://img.shields.io/badge/platform-linux-lightgrey.svg)](README.md)
[![ROCm](https://img.shields.io/badge/ROCm-7.2.4-green.svg)](https://rocm.docs.amd.com)
[![vLLM](https://img.shields.io/badge/vLLM-0.25.0+rocm723-blue.svg)](https://github.com/vllm-project/vllm)
[![LMCache](https://img.shields.io/badge/LMCache-v0.5.1-blue.svg)](https://github.com/LMCache/LMCache)
[![NIXL](https://img.shields.io/badge/NIXL-v1.3.1-blue.svg)](https://github.com/ai-dynamo/nixl)
[![Spelling](https://github.com/ROCm/rocm-aic/actions/workflows/spellcheck.yml/badge.svg)](https://github.com/ROCm/rocm-aic/actions/workflows/spellcheck.yml)
[![Nightly Dist Build](https://github.com/ROCm/rocm-aic/actions/workflows/aic-amd-nightly-dist-build.yml/badge.svg)](https://github.com/ROCm/rocm-aic/actions/workflows/aic-amd-nightly-dist-build.yml)
[![Nightly Smoke Test](https://github.com/ROCm/rocm-aic/actions/workflows/aic-amd-nightly-smoke-test.yml/badge.svg)](https://github.com/ROCm/rocm-aic/actions/workflows/aic-amd-nightly-smoke-test.yml)
[![Nightly Cliff](https://github.com/ROCm/rocm-aic/actions/workflows/aic-amd-nightly-cliff.yml/badge.svg)](https://github.com/ROCm/rocm-aic/actions/workflows/aic-amd-nightly-cliff.yml)

> [!CAUTION]
> This release is an *early-access* software technology preview. Running
> production workloads is *not* recommended.

ROCm(tm) AMD Infinity Context (AIC) is a disaggregated KV-cache inference stack
for large language models on AMD Instinct GPUs. It combines an LLM serving
framework with a KV Cache block manager to offload KV tensors across a tiered
memory hierarchy. GPU VRAM, CPU DRAM, local NVMe and NFS-over-RDMA.

It focuses on enabling a low-latency, shared level of KV Cache that can be
accessed by O(1000) GPUs.

ROCm AIC enables long-context serving at scale without recomputation. This
repository provides a Docker image build and test harness, benchmarking
harness, and Slurm/Spur automation used to validate and demonstrate the
platform.

## Stack overview

```text
Ubuntu 24.04  (rocm/dev-ubuntu-24.04:7.2.4-complete, ROCm 7.2.4, Python 3.12)
  └── vLLM v0.25.0+rocm723  (pre-built wheel — bundles torch/triton/flash-attn)
        └── LMCacheMPConnector (ZMQ)
              └── LMCache server (standalone MP mode)  [v0.5.1 + 7 AMD patches]
                    ├── L1:  GPU / CPU DRAM   (--l1-size-gb)
                    │    or  hipFile NVMe slab (GDS L1 mode)
                    ├── L2a: NIXL AIS_MT → local NVMe   (hipFile P2PDMA, GDS)
                    └── L2b: NIXL POSIX  → NFS-over-RDMA
```

Component versions (pinned in `docker/Dockerfile` — update ARGs there before
each release):

| Component | Source | Ref |
| --- | --- | --- |
| Base OS | `rocm/dev-ubuntu-24.04:7.2.4-complete` | Ubuntu 24.04, ROCm 7.2.4, Python 3.12 |
| vLLM | `wheels.vllm.ai/rocm/0.25.0/rocm723` | v0.25.0+rocm723 (pre-built wheel, bundles torch) |
| LMCache | `LMCache/LMCache` (upstream) | `v0.5.1` + 7 AMD patches |
| NIXL | `ai-dynamo/nixl` (upstream) | `v1.3.1` + `nixl-rocm-ais-mt.patch` |
| hipFile | `ROCm/rocm-systems` | `develop` @ `6901b670` |

## Pip-installable nightly wheels

See [docs/PIP_WHEELS.md](docs/PIP_WHEELS.md) for installation instructions and
compatibility notes. Wheels are rebuilt nightly from `main` and published to the
[nightly release](https://github.com/ROCm/rocm-aic/releases/tag/nightly).

## Prerequisites

- ROCm-capable host (MI300X recommended; `gfx942`). The default build is
  multi-arch — it bakes in every gfx the vLLM wheel supports (`gfx90a`, `gfx942`,
  `gfx950`, and the RDNA `gfx11xx`/`gfx12xx` line), so one image runs on any of
  them. Narrow with `ROCM_ARCH=gfx942` for a faster single-arch build.
- Docker with BuildKit and the `docker compose` plugin (Docker 23+)
- Host mounts: local NVMe (`NVME_DATA`) and NFS-over-RDMA (`NFS_DATA`) pre-mounted
- HuggingFace token with access to the target model
- Python 3.10+ for host-side benchmarks

## Quick start

See [docs/QUICK_START.md](docs/QUICK_START.md) for the full step-by-step guide
(build, start, benchmark, plot). For Slurm / SPUR cluster usage see
[docs/SLURM_SPUR.md](docs/SLURM_SPUR.md).

## Metrics & observability

See [docs/METRICS_TELEMETRY.md](docs/METRICS_TELEMETRY.md) for the full
Prometheus scrape config, exporter details, NIXL telemetry, hsa-snoop, and
NFS caveats.

## Key environment variables

See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for the full reference.

## License

[MIT](LICENSE.md)
