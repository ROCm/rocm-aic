# ROCm AMD Infinity Context

[![MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ROCm/rocm-aic/blob/main/LICENSE.md)
[![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey.svg)](README.md)
[![ROCm](https://img.shields.io/badge/ROCm-10.2-green.svg)](https://rocm.docs.amd.com/en/docs-10.2.0/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-blue.svg)](https://github.com/ROCm/pytorch/tree/abecaa87238d2d43d2641a22d89a51de185779b2)
[![AITER](https://img.shields.io/badge/AITER-v0.1.29.post1-blue.svg)](https://github.com/ROCm/aiter/tree/v0.1.29.post1)
[![vLLM](https://img.shields.io/badge/vLLM-v0.34.0-blue.svg)](https://github.com/vllm-project/vllm/tree/v0.34.0)
[![LMCache](https://img.shields.io/badge/LMCache-v0.5.8rc1-blue.svg)](https://github.com/LMCache/LMCache/tree/v0.5.8rc1)
[![NIXL](https://img.shields.io/badge/NIXL-v1.5.2-blue.svg)](https://github.com/ai-dynamo/nixl/tree/v1.5.2)
[![HSA-Snoop](https://img.shields.io/badge/HSA--Snoop-aba6a306-blue.svg)](https://github.com/sbates130272/hsa-snoop/tree/aba6a306)
[![Rocjitsu](https://img.shields.io/badge/Rocjitsu-98f6a72-blue.svg)](https://github.com/ROCm/rocm-systems/commit/98f6a72dcc49a5f809593deacbe5c9df89cde3b3)
[![Qemu-Vfu](https://img.shields.io/badge/Qemu--Vfu-f91f120-blue.svg)](https://github.com/sbates130272/qemu/commit/f91f12068854848cdb248f2a4636ddfe41f3283b)
[![Rocjitsu Guest](https://img.shields.io/badge/Rocjitsu_Guest-resolute%2Fc079195-blue.svg)](https://github.com/sbates130272/qemu-minimal/commit/c0791954aec508bfd3f26d11d6960f279d4b175a)
[![Spelling](https://img.shields.io/github/actions/workflow/status/ROCm/rocm-aic/rocm-aic-spell-check.yml?label=Spelling)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-spell-check.yml)
[![Lint](https://img.shields.io/github/actions/workflow/status/ROCm/rocm-aic/rocm-aic-lint-check.yml?label=Lint)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-lint-check.yml)
[![Export Tarball](https://img.shields.io/github/actions/workflow/status/ROCm/rocm-aic/rocm-aic-export-tarball.yml?label=Export+Tarball)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-export-tarball.yml)
[![Nightly Dist Build](https://img.shields.io/github/actions/workflow/status/ROCm/rocm-aic/rocm-aic-self-hosted-nightly-dist-build.yml?label=Nightly+Dist+Build)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-self-hosted-nightly-dist-build.yml)
[![Nightly Smoke + Tiny Test](https://img.shields.io/github/actions/workflow/status/ROCm/rocm-aic/rocm-aic-self-hosted-nightly-smoke-test.yml?label=Nightly+Smoke+%2B+Tiny+Test)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-self-hosted-nightly-smoke-test.yml)
[![Nightly Accuracy Test](https://img.shields.io/github/actions/workflow/status/ROCm/rocm-aic/rocm-aic-self-hosted-accuracy-test.yml?label=Nightly+Accuracy+Test)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-self-hosted-accuracy-test.yml)
[![Nightly Cliff](https://img.shields.io/github/actions/workflow/status/ROCm/rocm-aic/rocm-aic-self-hosted-nightly-cliff-test.yml?label=Nightly+Cliff)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-self-hosted-nightly-cliff-test.yml)
[![Nightly Wheels](https://img.shields.io/github/actions/workflow/status/ROCm/rocm-aic/rocm-aic-nightly-wheels-publication.yml?label=Nightly+Wheels)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-nightly-wheels-publication.yml)
[![Check PRs + Patches](https://img.shields.io/github/actions/workflow/status/ROCm/rocm-aic/rocm-aic-check-prs-and-patches.yml?label=Check+PRs+%2B+Patches)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-check-prs-and-patches.yml)
[![vLLM Emul Test](https://img.shields.io/github/actions/workflow/status/ROCm/rocm-aic/rocm-aic-emulate-test.yml?label=vLLM+Emul+Test)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-emulate-test.yml)
[![vLLM Emul Perf](https://img.shields.io/github/actions/workflow/status/ROCm/rocm-aic/rocm-aic-vllm-emul-performance.yml?label=vLLM+Emul+Perf)](https://github.com/ROCm/rocm-aic/actions/workflows/rocm-aic-vllm-emul-performance.yml)
[![Metrics Reference](https://img.shields.io/badge/Prometheus-Metrics_Reference-E05D00?logo=prometheus&logoColor=white)](https://rocm.github.io/rocm-aic/prometheus/)
[![Cliff Perf Dashboard](https://img.shields.io/badge/Cliff-Perf_Dashboard-2980b9?logo=github&logoColor=white)](https://rocm.github.io/rocm-aic/cliff/)
[![vLLM Emul Dashboard](https://img.shields.io/badge/vLLM_Emul-Perf_Dashboard-7c3aed?logo=github&logoColor=white)](https://rocm.github.io/rocm-aic/vllm-emul-performance/)

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
| Base OS | `rocm/dev-ubuntu-26.04:10.2.0-full` | Ubuntu 26.04, ROCm 10.2, Python 3.14 |
| PyTorch | `ROCm/pytorch` (source build) | `2.15 (abecaa8)` |
| Triton | `ROCm/triton` (source build) | `release/internal/3.8.x` (669b31ac, vLLM v0.34.0 validated) |
| vLLM | `github.com/vllm-project/vllm` (source build) | `v0.34.0` + 2 AMD patches |
| AITER | `ROCm/aiter` (source build) | `v0.1.29.post1` |
| FlashAttention | `Dao-AILab/flash-attention` (source build) | `0e60e394` |
| LMCache | `LMCache/LMCache` (upstream) | `v0.5.8rc1` + 6 AMD patches |
| NIXL | `ai-dynamo/nixl` (upstream) | `v1.5.2` + `nixl-rocm-ais-mt.patch` |
| HSA Snoop | `sbates130272/hsa-snoop` (source build) | `aba6a306` |

`aic-vllm` runs OpenTelemetry 1.40.0, below the 1.45.1 that vLLM v0.34.0
declares: LMCache v0.5.8rc1 caps it there, and the vLLM process loads the
LMCache client. The image build checks that vLLM's tracing exporters still
import; `pip check` in the image reports the conflict.

The vLLM `--kv-transfer-config` must carry `"lmcache.mp.use_vmm_api": false`
(the `Makefile` JSON does). Without it LMCache v0.5.8rc1 asks vLLM for
fd-exported KV memory, which vLLM v0.34.0 refuses on ROCm at startup.

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

## Environment Variables

[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) is the reference for the variables:
each default, the file that owns it, and what it does. The tables there are
generated from code (`make env-docs`); `make list-vars` prints the values a
`make` run would use on this host.

## License

[MIT](LICENSE.md)
