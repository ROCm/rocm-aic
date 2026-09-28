# ROCm AMD Infinity Context

[![MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ROCm/rocm-aic/blob/main/LICENSE.md)
[![Platform](https://img.shields.io/badge/platform-linux-lightgrey.svg)](README.md)
[![ROCm](https://img.shields.io/badge/ROCm-7.14.1-green.svg)](https://rocm.docs.amd.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-release%2F2.13-ee4c2c.svg)](https://github.com/ROCm/pytorch/tree/release/2.13)
[![AITER](https://img.shields.io/badge/AITER-v0.1.22.post1-blue.svg)](https://github.com/ROCm/aiter/tree/v0.1.22.post1)
[![vLLM](https://img.shields.io/badge/vLLM-v0.29.0-blue.svg)](https://github.com/vllm-project/vllm)
[![LMCache](https://img.shields.io/badge/LMCache-v0.5.5-blue.svg)](https://github.com/LMCache/LMCache)
[![NIXL](https://img.shields.io/badge/NIXL-v1.4.1-blue.svg)](https://github.com/ai-dynamo/nixl)
[![hsa-snoop](https://img.shields.io/badge/hsa--snoop-v1.1.1-blue.svg)](https://github.com/sbates130272/hsa-snoop)
[![rocjitsu](https://img.shields.io/badge/rocjitsu-8e01a5a-red.svg)](https://github.com/sbates130272/batesste-ci-images/tree/main/ubuntu-rocm-rocjitsu)
[![qemu-vfu](https://img.shields.io/badge/qemu--vfu-7794baa-red.svg)](https://github.com/sbates130272/batesste-ci-images/tree/main/ubuntu-qemu-libvfio-user)
[![rocjitsu-guest](https://img.shields.io/badge/guest-resolute%2F63cc0bc-red.svg)](https://github.com/sbates130272/batesste-ci-images/tree/main/ubuntu-qcow2-gen)
[![Spelling](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-spell-check.yml/badge.svg)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-spell-check.yml)
[![Lint](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-lint-check.yml/badge.svg)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-lint-check.yml)
[![Export Tarball](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-export-tarball.yml/badge.svg)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-export-tarball.yml)
[![Nightly Dist Build](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-self-hosted-nightly-dist-build.yml/badge.svg)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-self-hosted-nightly-dist-build.yml)
[![Nightly Smoke Test](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-self-hosted-nightly-smoke-test.yml/badge.svg)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-self-hosted-nightly-smoke-test.yml)
[![Nightly Tiny Test](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-self-hosted-nightly-tiny-test.yml/badge.svg)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-self-hosted-nightly-tiny-test.yml)
[![Nightly Accuracy Test](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-self-hosted-accuracy-test.yml/badge.svg)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-self-hosted-accuracy-test.yml)
[![Nightly Cliff](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-self-hosted-nightly-cliff-test.yml/badge.svg)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-self-hosted-nightly-cliff-test.yml)
[![Nightly Wheels](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-nightly-wheels-publication.yml/badge.svg)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-nightly-wheels-publication.yml)
[![Check PRs + Patches](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-check-prs-and-patches.yml/badge.svg)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-check-prs-and-patches.yml)
[![Rocjitsu VM Test](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-rocjitsu-test.yml/badge.svg)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-rocjitsu-test.yml)
[![Metrics Reference](https://img.shields.io/badge/Prometheus-Metrics_Reference-E05D00?logo=prometheus&logoColor=white)](https://rocm.github.io/rocm-aic/prometheus/)
[![Cliff Perf Dashboard](https://img.shields.io/badge/Cliff-Perf_Dashboard-2980b9?logo=github&logoColor=white)](https://rocm.github.io/rocm-aic/cliff/)

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

## Stack Overview

| Component | Source | Ref |
| --- | --- | --- |
| Base OS | `rocm/dev-ubuntu-24.04:7.14.1-full` | Ubuntu 24.04, ROCm 7.14, Python 3.12 |
| PyTorch | `ROCm/pytorch` (source build) | `release/2.13` |
| vLLM | `github.com/vllm-project/vllm` (source build) | `v0.29.0` + 3 AMD patches |
| AITER | `ROCm/aiter` (source build) | `v0.1.22.post1` (vLLM ROCm-validated) |
| FlashAttention | `Dao-AILab/flash-attention` (source build) | `0e60e394` (vLLM ROCm-validated) |
| LMCache | `LMCache/LMCache` (upstream) | `v0.5.5` + 16 AMD patches |
| NIXL | `ai-dynamo/nixl` (upstream) | `v1.4.1` + `nixl-rocm-ais-mt.patch` |
| hsa-snoop | `sbates130272/hsa-snoop` (source build) | `v1.1.1` |
| hipFile | ROCm 7.14 base image | GA in ROCm 7.14 — no separate source build |

## Pip Nightly Wheels

See [docs/PIP_WHEELS.md](docs/PIP_WHEELS.md) for installation instructions and
compatibility notes. Wheels are rebuilt nightly from `main` and published to the
[nightly release](https://github.com/ROCm/rocm-aic/releases/tag/nightly).

## Prerequisites

- ROCm-capable host. Pass `ROCM_ARCH` as a `;`-separated list (e.g. `gfx90a;gfx942;gfx950`)
  to build a multi-arch image, or a single arch (e.g. `ROCM_ARCH=gfx942`) for a faster build.
  The vLLM source build compiles GPU kernels for exactly the archs specified.
- Docker with BuildKit and the `docker compose` (v2) plugin (Docker 23+). On a
  node that lacks it, `make ensure-compose` installs the plugin user-locally
  (`~/.docker/cli-plugins`); the Slurm cliff / smoke / tiny-test jobs self-install
  it automatically.
- Host mounts: local NVMe (`NVME_DATA`) and NFS-over-RDMA (`NFS_DATA`) pre-mounted
- HuggingFace token with access to the target model
- Python 3.10+ for host-side benchmarks

## Quick Start

See [docs/QUICK_START.md](docs/QUICK_START.md) for the full step-by-step guide
(build, start, benchmark, plot). For Slurm / SPUR cluster usage see
[docs/SLURM_SPUR.md](docs/SLURM_SPUR.md).

## Metrics & Observability

See [docs/METRICS_TELEMETRY.md](docs/METRICS_TELEMETRY.md) for the full
Prometheus scrape config, exporter details, NIXL telemetry, hsa-snoop, and
NFS caveats.

## Key Environment Variables

See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for the full reference.

## License

[MIT](LICENSE.md)
