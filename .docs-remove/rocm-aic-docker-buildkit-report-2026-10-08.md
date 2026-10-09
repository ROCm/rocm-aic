# ROCm AIC Docker BuildKit Report — 2026-10-08

## Overview

The ROCm AIC stack ships four Docker images built from three Dockerfiles plus a
sidecar. All images use **multi-stage BuildKit builds** that separate a
heavyweight ROCm compilation environment from a minimal runtime image, reducing
deployed image sizes by 60–95% relative to the full ROCm base.

---

## Image Tag Format

Every image is tagged with all pinned component versions and the target GPU
architecture slug, computed by `docker/scripts/aic-image-tag.sh`:

```
<AIC_VERSION>-rocm<ROCM_VERSION>-pytorch<PYTORCH_SERIES>-vllm<VLLM_REF>
  -aiter<AITER_REF>-fa<FLASH_ATTN_SHA>-lmcache<LMCACHE_REF>
  -nixl<NIXL_REF>-hsasnoop<HSA_SNOOP_REF>[-<arch_slug>]
```

Example (gfx1201 single-arch build, 2026-10-08):
```
0.1.0-rocm7.14.1-pytorch2.13-vllm0.29.0-aiter0.1.22.post1-lmcache0.5.5-nixl1.4.1-hsasnoop1.1.1-gfx1201
```

The arch slug `gfx1201` is omitted for multi-arch builds. This tagging scheme
ensures any two images with the same tag are bit-for-bit equivalent, making
reproducing a deployment trivial.

---

## Deployed Image Sizes (gfx1201, 2026-10-08)

| Image | Deployed Size | Build Base | Runtime Base |
|---|---|---|---|
| `aic-base` | **5.95 GB** | rocm/dev-ubuntu-24.04:7.14.1-full (28.2 GB) | ubuntu:24.04 |
| `aic-vllm` | **12.3 GB** | rocm/dev-ubuntu-24.04:7.14.1-full (28.2 GB) | aic-base |
| `aic-lmcache` | **7.24 GB** | rocm/dev-ubuntu-24.04:7.14.1-full (28.2 GB) | aic-base |
| `aic-hsa-snoop` | **1.59 GB** | rocm/dev-ubuntu-24.04:7.14.1-full (28.2 GB) | ubuntu:24.04 |

**Total deployed stack: ~27.1 GB** vs **~113 GB** if all four images were based
on the full ROCm image — a **76% reduction**.

---

## Build Stage Architecture

### aic-base (`docker/base/Dockerfile`)

```
rocm/dev-ubuntu-24.04:7.14.1-full (28.2 GB)
  ├─ base          System deps (cmake, git, libboost, ibverbs, bpftrace, uv)
  ├─ build_pytorch PyTorch 2.13 from source (SHA ec5f3b1)
  │     └─ build_torchvision  torchvision v0.28.0
  ├─ build_nixl    NIXL v1.4.1 KV-transfer transport
  └─ build_clr_patch  Patched libamdhip64.so (IPC event ring fix ROCm#11511)

ubuntu:24.04 (119 MB)
  └─ runtime ─── aic-base (7.05 GB)
       ├─ 13 ROCm APT packages (~2.0 GB total):
       │    amdrocm-runtime, base, profiler-base, dnn-host (MIOpen 463 MB),
       │    blas-host (273 MB), fft-host, rand-host, rccl-host, solver-host,
       │    sparse-host, amdsmi, llvm (839 MB), math-common
       ├─ Per-arch GPU kernel data: amdrocm-blas7.14-gfxNNNN
       │    (Tensile .dat files; separate RUN to avoid apt cache stale issue)
       ├─ torch + torchvision wheels (COPYed from build_torchvision)
       ├─ NIXL: /opt/nixl/ + /opt/rocnixl-ucx/ (COPYed from build_nixl)
       └─ patched libamdhip64.so (COPYed from build_clr_patch)
```

**What stays in build stages only (not shipped):**
- PyTorch full source tree (~8 GB after clone + submodules)
- torchvision source
- NIXL source + UCX build artifacts
- CLR source (projects/clr + projects/hip), cmake build dir
- All intermediate .o files, cmake caches

**Key ENV in runtime:**
- `HSA_ENABLE_IPC_MODE_LEGACY=1` — selects KFD native signal IPC path
- `NIXL_PLUGIN_DIR=/opt/nixl/lib/x86_64-linux-gnu/plugins`
- `ROCM_PATH=/opt/rocm/core-7.14`

---

### aic-vllm (`docker/vllm/Dockerfile`)

```
rocm/dev-ubuntu-24.04:7.14.1-full (28.2 GB)
  └─ build_vllm  ── COPYs torch from aic-base (via --build-context)
       ├─ AITER v0.1.22.post1 (GPU attention/GEMM kernels)
       ├─ FlashAttention v2.8.3.post1 (CDNA only; skipped on gfx1201)
       ├─ vLLM v0.29.0 (patched, wheel build with hipcc)
       └─ llm-emu b71c602 (CPU-only emulation plugin)

aic-base (7.05 GB, via OCI layout)
  └─ vllm runtime (13.4 GB)
       ├─ All packages COPYed from build_vllm
       │    (vllm, aiter, flash_attn, llm-emu, all deps)
       ├─ lmcache=0.5.5 installed from PyPI (MP connector client)
       ├─ profiles/*.json (emulator profile packs)
       └─ /etc/rocm-aic-attention-backends.env (records compiled backends)
```

**AITER size:** ~800 MB installed (compiled ROCm attention kernels for gfx1201)  
**vLLM size:** ~400 MB installed  
**Delta over aic-base:** ~6.35 GB

**FlashAttention policy:** Only compiled when all ROCM_ARCH values are CDNA
(`gfx9xx`). For gfx1201 (RDNA4) it is skipped; Triton attention is used instead
(no performance loss on consumer GPUs, avoids CDNA-specific memory patterns).

---

### aic-lmcache (`docker/lmcache/Dockerfile`)

```
rocm/dev-ubuntu-24.04:7.14.1-full (28.2 GB)
  └─ build_lmcache_wheel ── COPYs torch from aic-base
       └─ LMCache v0.5.5 HIP wheel
            (CXX=hipcc, BUILD_WITH_HIP=1, patches/lmcache/*.patch)

aic-base (7.05 GB, via OCI layout)
  └─ build / runtime (8.39 GB)
       ├─ LMCache v0.5.5 wheel (COPYed from build_lmcache_wheel, then installed)
       ├─ cupy-rocm-7-0 (GPU buffer allocator — avoids loading librocblas/MIOpen)
       ├─ matplotlib, openai client
       └─ paged_cupy_memory_allocator.py (CuPy-backed NIXL buffer pool)
```

**Key ENV:** `LMCACHE_NIXL_USE_CUPY=1` activates CuPy GPU allocator, preventing
lmcache from dlopen-ing librocblas/MIOpen (which would add ~700 MB RSS overhead
for a service that runs no matrix kernels itself).

**Delta over aic-base:** ~1.34 GB

---

### aic-hsa-snoop (`docker/hsa-snoop/Dockerfile`)

```
rocm/dev-ubuntu-24.04:7.14.1-full (28.2 GB)
  └─ build
       └─ hsa-snoop v1.2.0 (cmake, stripped binary)
            -DHSA_SNOOP_PROMETHEUS=ON (embeds prometheus-cpp v1.2.4)

ubuntu:24.04 (119 MB)
  └─ runtime (1.59 GB)
       ├─ amdrocm-runtime7.14 (libhsa-runtime64.so for AQL interception)
       ├─ AMD apt key + sources.list (COPYed from build stage — no hardcoded URL)
       └─ /usr/local/bin/hsa-snoop (single stripped binary)
```

**v1.2.0 additions over v1.1.1:** SDMA v7 support (gfx12 / gfx1201). Previously
all SDMA queues on gfx1201 were logged as "unsupported SDMA v7" and skipped;
v1.2.0 maps v7 to the v6 packet layout (correct for gfx1201's sdma_v7_0 kernel
driver).

---

## Build Orchestration (`mk/local.mk`)

The `make build` target runs four sequential `docker buildx build` invocations:

```
[1/4] aic-base  ──→  OCI tar → OCI dir (hand-off artifact)
[2/4] aic-vllm  ──→  --build-context base=oci-layout://<OCI_DIR>
[3/4] aic-lmcache → --build-context base=oci-layout://<OCI_DIR>
                     OCI dir deleted after this step
[4/4] aic-hsa-snoop  (no base dependency)
```

The OCI layout hand-off avoids loading aic-base into the Docker daemon between
steps — it stays as a directory on disk and BuildKit reads it directly, saving
a `docker load` + `docker save` round-trip of ~7 GB.

---

## Layer Cache Strategy

### What BuildKit caches and why

BuildKit caches each RUN/COPY/ADD instruction as an independent layer keyed by
a hash of: the instruction text, the FROM image digest, and (for COPY) the
content hash of the source files. A cache hit skips the instruction entirely;
a miss re-executes it and invalidates all downstream layers.

**Critical invariant:** if layer N misses, layers N+1, N+2, ... all miss too,
regardless of whether their inputs changed. This makes the order and granularity
of COPY instructions matter enormously for expensive compile steps.

### APT layer caching (`--mount=type=cache`)

All `apt-get install` RUN steps use BuildKit cache mounts:

```dockerfile
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y ...
```

The APT package cache (`/var/cache/apt`) and package list cache
(`/var/lib/apt/lists`) are stored in a BuildKit-managed volume that persists
across builds. The `sharing=locked` mode allows only one build at a time to
write, with others waiting. This means:
- The package index (`apt-get update`) only re-downloads when the package lists
  are stale, not on every build
- Downloaded `.deb` files are reused across builds
- The layer hash still changes if the installed package list changes (triggering
  a re-run), but the download is served from local cache rather than the network

### ccache for C++/HIP compilation (`--mount=type=cache`)

PyTorch and AITER compilation use ccache to avoid recompiling unchanged `.cpp`
and `.hip` files even within the same build:

```dockerfile
RUN --mount=type=cache,target=/root/.cache/ccache \
    CCACHE_DIR=/root/.cache/ccache \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    python3 setup.py bdist_wheel ...
```

This helps when a rebuild is triggered (e.g. by a COPY hash miss) but only a
small number of files actually changed — ccache hits on the unchanged objects.

### Local disk cache (development)

```makefile
AIC_CACHE_DIR ?= ~/.cache/rocm-aic-buildx   # set by user
AIC_CACHE_MODE ?= max
```

When `AIC_CACHE_DIR` is set every buildx invocation adds:
```
--cache-from type=local,src=<AIC_CACHE_DIR>/<arch_slug>
--cache-to   type=local,dest=<AIC_CACHE_DIR>/<arch_slug>,mode=max,ignore-error=true
```

`mode=max` exports **every intermediate layer** (not just the final image layers)
so individual RUN steps are cached even if downstream layers change. This is
critical for the PyTorch compile steps (~90 min) which would otherwise re-run
every time an unrelated Dockerfile line changes.

`ignore-error=true` on `--cache-to` ensures a disk-full or permission error
during cache export never fails the actual build.

### Registry cache (SPUR cluster)

On the SPUR cluster (`AIC_SPUR_CLUSTER=1`), `AIC_CACHE_REF` points to a shared
registry (e.g. `ghcr.io/rocm/aic-buildcache`). This enables:
- Any build node to resume from any other node's last good layer
- Parallel multi-node builds sharing compile artifacts
- Cache persists across Slurm job boundaries

### GitHub Actions cache (CI)

The emulate-only CI workflow uses `type=gha,scope=aic-emulate` with `mode=max`.
The emulate-base image (`python:3.12-bookworm` + CPU-only torch) is separately
cached under `scope=aic-emulate-base` since it changes rarely.

### Surgical COPY: the most impactful cache optimisation

The most expensive single failure mode is a spurious PyTorch re-compile triggered
by an unrelated change to the `patches/` directory. Originally all Dockerfiles
used:

```dockerfile
COPY patches/ /app/patches/   # ❌ WRONG — entire patches/ dir as one layer
```

BuildKit computes a recursive hash of the source directory. Renaming or adding
any patch file in **any** subdirectory — even `patches/lmcache/` — changes the
hash, causing the COPY layer to miss, which cascades to a full PyTorch recompile
(~150 min) even though pytorch patches were untouched.

**The fix (2026-10-08):** each stage only copies the subdirectory it uses:

```dockerfile
# docker/base/Dockerfile — build_pytorch stage
COPY patches/pytorch/ /app/patches/pytorch/   # ✅ isolated

# docker/vllm/Dockerfile — build_vllm stage
COPY patches/vllm/     /app/patches/vllm/     # ✅ isolated
COPY patches/llm-emu/  /app/patches/llm-emu/  # ✅ isolated

# docker/lmcache/Dockerfile — build_lmcache_wheel and build stages
COPY patches/lmcache/  /app/patches/lmcache/  # ✅ isolated

# docker/base/Dockerfile — build_nixl stage (was already correct)
COPY patches/nixl/ /tmp/nixl-patches/         # ✅ isolated

# docker/base/Dockerfile — build_clr_patch stage (single file — correct)
COPY patches/hip-ipc-event-ring-fix.patch ... # ✅ single file
```

Now renaming a lmcache patch only busts lmcache build layers (~5 min). PyTorch,
vLLM, AITER, NIXL, and CLR patch caches are fully independent. This change
reduced the blast radius of any patch modification from ~150 min to <10 min.

---

## Build Timing (observed, gfx1201, 24-core node)

Cumulative image size grows as each stage adds layers. "Start size" is the
FROM base; "End size" is the layer exported/committed at stage completion.
Build stages are ephemeral and not shipped — only runtime stages are deployed.

| Phase | Stage | Start Size | End Size | Cached | Uncached |
|---|---|---|---|---|---|
| ROCm -full pull | base (FROM) | 0 | 28.2 GB | 0s (present) | ~90s |
| Base apt + uv | base | 28.2 GB | 28.6 GB | ~5s | ~30s |
| PyTorch submodule clone | build_pytorch | 28.6 GB | 30.1 GB | ~5s | ~3 min |
| PyTorch CPU compile (3800 TUs) | build_pytorch | 30.1 GB | 33.8 GB | ~15s | ~90 min |
| PyTorch HIP compile | build_pytorch | 33.8 GB | 36.2 GB | ~15s | ~60 min |
| torchvision build | build_torchvision | 36.2 GB | 36.5 GB | ~10s | ~5 min |
| NIXL compile (UCX + plugins) | build_nixl | 28.6 GB | 29.0 GB | ~10s | ~8 min |
| CLR patch + libamdhip64 build | build_clr_patch | 28.6 GB | 28.7 GB | ~10s | ~12 min |
| **aic-base runtime APT** | **runtime** | **119 MB** | **~2.2 GB** | ~10s | ~3 min |
| **+ torch/torchvision COPY** | **runtime** | **2.2 GB** | **~5.7 GB** | ~30s | ~30s |
| **+ NIXL COPY + patched .so** | **runtime** | **5.7 GB** | **~7.05 GB** | ~10s | ~10s |
| AITER HIP compile | build_vllm | 28.2 GB | 29.5 GB | ~15s | ~20 min |
| FlashAttention (CDNA only) | build_vllm | 29.5 GB | 30.2 GB | ~10s | ~15 min |
| vLLM wheel build | build_vllm | 30.2 GB | 30.8 GB | ~15s | ~15 min |
| llm-emu install | build_vllm | 30.8 GB | 30.9 GB | ~5s | ~2 min |
| **aic-vllm runtime COPY** | **vllm** | **7.05 GB** | **~13.0 GB** | ~60s | ~60s |
| **+ lmcache PyPI client** | **vllm** | **13.0 GB** | **~13.4 GB** | ~10s | ~10s |
| lmcache HIP wheel build | build_lmcache_wheel | 28.6 GB | 29.0 GB | ~10s | ~5 min |
| **aic-lmcache runtime install** | **build/runtime** | **7.05 GB** | **~8.39 GB** | ~20s | ~20s |
| hsa-snoop compile + strip | hsa-snoop build | 28.2 GB | 28.2 GB | ~5s | ~2 min |
| **aic-hsa-snoop runtime** | **runtime** | **119 MB** | **~1.59 GB** | ~10s | ~10s |
| **Total (cache warm, 74 CACHED hits)** | | | | **~15 min** | — |
| **Total (cold)** | | | | — | **~3.5 hours** |

> **Observed 2026-10-08** (surgical COPY build, 74/~90 stages cached): aic-base ~3 min
> (all CACHED), aic-vllm ~3 min (all CACHED), aic-lmcache ~8 min (patches + wheel),
> aic-hsa-snoop ~2 min. Total wall-clock: **~15 min** for a patch-only change.
>
> Build stages (build_pytorch, build_vllm, etc.) are never shipped — their
> intermediate sizes are transient on-disk artifacts only. The "End Size" for
> runtime stages is what's stored in the local Docker daemon.

The most expensive uncached phases are PyTorch CPU and HIP compile. With a warm
local cache these are reduced to layer manifest checks (~15s each). The NIXL
and CLR patch builds are expensive on cold but relatively fast (~8-12 min each).

---

## Layer Size Decomposition (aic-base, 7.05 GB)

| Component | Approx Size | Source |
|---|---|---|
| Ubuntu 24.04 base | 119 MB | ubuntu:24.04 |
| ROCm runtime APT packages | ~2.0 GB | AMD apt repo |
| Per-arch GPU kernel data | ~200 MB | amdrocm-blas7.14-gfxNNNN |
| PyTorch 2.13 + deps | ~3.5 GB | Built from source |
| torchvision 0.28.0 | ~100 MB | Built from source |
| NIXL + UCX transport | ~200 MB | Built from source |
| patched libamdhip64.so | ~7 MB | Built from CLR source |
| System packages (git, nvme-cli, etc.) | ~150 MB | ubuntu apt |

---

## Key Optimization Decisions

### Why build PyTorch from source?

AMD's PyPI wheel for ROCm 7.14 shipped Triton 3.7.1 which broke vLLM 0.25.x on
gfx950 (MI355X). Building from `ROCm/pytorch@release/2.13` against the 7.14
headers avoids the Triton version conflict and gives full control over compiler
flags (e.g. `CK_BUFFER_RESOURCE_3RD_DWORD` for CK BGEMM).

### Why split build/runtime stages for vllm and lmcache?

AITER, FlashAttention, vLLM, and the LMCache HIP extension all require `hipcc`
and HIP CMake files (`hip-config.cmake`) which are only present in the full ROCm
image. The slim `aic-base` runtime (ubuntu:24.04 + 13 packages) does not have
these. Splitting into `build_vllm` / `build_lmcache_wheel` stages (FROM -full)
with COPY to the slim runtime stage achieves GPU-accelerated package builds
without shipping the 28 GB toolchain.

### Why NIXL has its own build stage in aic-base?

NIXL's POSIX backend requires libaio + liburing at runtime (for NVMe DMA) and
UCX headers at build time. Building NIXL in aic-base means both aic-vllm and
aic-lmcache inherit it automatically — no per-image NIXL build.

### Why is `libamdhip64.so` patched at build time?

ROCm/rocm-systems PRs #11511 and #11749 fix critical IPC event bugs affecting
cross-container GPU IPC on ROCm 7.14.1. These PRs are merged to `develop` but
not yet in any released ROCm package. The `build_clr_patch` stage applies the
one-line modulo fix (`read_index % IPC_SIGNALS_PER_EVENT`) and rebuilds only the
`amdhip64` CMake target, producing a patched 7 MB `.so` that replaces the
system-installed one.

### Why separate OCI layout hand-off?

`docker buildx build --build-context base=oci-layout://...` lets BuildKit use a
pre-built image as a named build context without requiring it to be loaded into
the Docker daemon. This avoids the `docker load` + `docker push` + `docker pull`
round-trip (3 × 7 GB I/O) between the aic-base build and the aic-vllm/lmcache
builds. The OCI tar is extracted to a directory on disk (~5s) and BuildKit reads
it directly via the content-addressable blob store.

---

## Emulate (CPU-only) Build

For CI and rocjitsu VM testing, a separate `docker/emulate-base.Dockerfile`
provides a CPU-only base:

```
python:3.12-bookworm
  └─ emulate-base
       ├─ CPU-only torch + torchvision (pytorch.org/whl/cpu)
       └─ vLLM build tools (setuptools, ninja, etc.)

emulate-base (via OCI layout)
  └─ aic-vllm --target emulate
       ├─ vLLM built with VLLM_TARGET_DEVICE=empty (no HIP kernels)
       └─ llm-emu plugin (profile-driven emulation)
```

The `--target emulate` build skips AITER, FlashAttention, and all HIP kernel
compilation, reducing build time from ~3.5 hours to ~15 minutes. The `emulate`
stage also overrides `PYTHONPATH` to exclude NIXL entries (the CI runner has no
ROCm libraries).

---

## Version Pinning Strategy

All framework versions are pinned as ARGs in the Dockerfiles. The Makefile reads
them via `sed` for consistency:

```makefile
# Makefile reads ROCM_VERSION from base/Dockerfile ARG line
ROCM_VERSION := $(shell grep '^ARG ROCM_VERSION=' docker/base/Dockerfile | ...)
# Similarly for LMCACHE_REF from lmcache/Dockerfile
```

This ensures `make build` always uses the versions declared in the Dockerfiles
without any risk of accidental mismatch from environment variables.

PyTorch is pinned to a specific commit SHA (`ec5f3b1...`) rather than a branch
tag to guarantee byte-for-bit reproducibility. VLLM, AITER, lmcache, NIXL, and
hsa-snoop use semver tags. llm-emu uses a short SHA (pre-release code).

---

*Generated: 2026-10-08 from commit `bb3ba70` on branch `feat/build-improvements`*
