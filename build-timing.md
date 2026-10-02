# AIC Build Timing Reference

Host: `snoc-thinkstation` — Threadripper PRO 7955WX (16c/32t), RX 9070 XT (gfx1201, 16 GB)
Target arches: `gfx1201,gfx1250` (dual-arch; gfx1250 for rocjitsu VM testing)
Build command: `make build AIC_CACHE_DIR=~/.cache/rocm-aic-buildx BUILD_JOBS=24`

---

## Stage structure

```
make build
  └─ [1/3] docker buildx build → aic-base      (docker/base/Dockerfile)
       ├─ base              — ROCm base image + system packages + uv
       ├─ build_pytorch     — PyTorch source build (release/2.13) → wheel
       ├─ build_torchvision — torchvision source build → wheel
       └─ runtime           — install wheels, strip build residue

  └─ [2/3] docker buildx build → aic-vllm      (docker/vllm/Dockerfile)
       └─ vllm              — vLLM v0.29.0 + lmcache + llm-emu

  └─ [3/3] docker buildx build → aic-lmcache   (docker/lmcache/Dockerfile)
       └─ build             — AITER, Flash-Attention, NIXL, LMCache, hsa-snoop
```

Stages 2 and 3 depend on the OCI export of stage 1 and run **sequentially** in
`make build`. On the SPUR cluster, `make dist-build-parallel` runs stages 2 and 3
concurrently after stage 1, roughly halving wall time.

---

## Run log

### Run 1 — 2026-10-02 (~11:40) — cold, no cache, plain progress

| Setting | Value |
|---------|-------|
| `AIC_CACHE_DIR` | _(not set — no layer cache)_ |
| `BUILD_JOBS` | _(not set — all 32 threads)_ |
| `BUILD_PROGRESS` | `plain` (hardcoded at the time) |
| Result | **Failed** — Docker truncated `--progress=plain` output at 2 MiB during HIP kernel compile; torchvision failed with silent `bdist_wheel` error (masked by `; true`) |

| Dockerfile step | BuildKit step | Observed time |
|-----------------|--------------|---------------|
| base — apt + uv | #1–#12 | ~30s |
| build_pytorch — git clone + submodules | #12 | ~90s (many submodules) |
| build_pytorch — pip deps | #13 | ~60s |
| build_pytorch — cmake + ninja (CXX) | #13/#14 | ~580s to reach 3826/3826 CXX |
| build_pytorch — HIP kernel compile | #14 | ~767s observed, then log clipped |
| build_pytorch — bdist_wheel | #14 (cont.) | unknown — masked by `; true` bug |
| build_torchvision | #16 | **FAILED** — wheel glob matched nothing |

> Log hit Docker's 2 MiB per-step limit during HIP. Fixed in this session by
> switching `--progress` default to `auto` via `BUILD_PROGRESS ?= auto`.

---

### Run 2 — 2026-10-02 (~12:08) — cold layer cache, auto progress, BUILD_JOBS=24

| Setting | Value |
|---------|-------|
| `AIC_CACHE_DIR` | `~/.cache/rocm-aic-buildx/gfx1201-gfx1250` — set, but **empty** (first run with cache enabled) |
| `BUILD_JOBS` | `24` |
| `BUILD_PROGRESS` | `auto` |
| Result | **Failed** — bdist_wheel silently failed (masked by `; true` shell bug); layer cached with empty `dist/` |

| Dockerfile step | BuildKit step | Time (from log) |
|-----------------|--------------|-----------------|
| base + build_pytorch 1–3 | #1–#13 | ~34s (BuildKit builder internal cache hits) |
| build_pytorch 4/4 — compile + bdist_wheel | #14 | **1224.8s** |
| build_torchvision — git clone | #15 | 1.8s |
| build_torchvision — pip install + build | #16 | **FAILED** — `/app/pytorch/dist/*.whl` empty |

> The `; true` at the end of the build_pytorch `RUN` caused the shell to exit 0
> even when `bdist_wheel` failed, silently committing a layer with no wheel.
> Fixed by replacing `; true` with `|| true` on the cleanup `find` commands only.
> See `docker/base/Dockerfile:156-157`.

---

### Run 3 — 2026-10-02 (~12:35) — warm cache, || true fix applied (in progress)

| Setting | Value |
|---------|-------|
| `AIC_CACHE_DIR` | `~/.cache/rocm-aic-buildx/gfx1201-gfx1250` — populated from run 2 |
| `BUILD_JOBS` | `24` |
| `BUILD_PROGRESS` | `auto` |
| Result | **In progress** |

| Dockerfile step | BuildKit step | Time |
|-----------------|--------------|------|
| base — all steps | #5–#12 | CACHED (~1s) |
| build_pytorch 1/4 — gcc-13 | #8 | CACHED |
| build_pytorch 2/4 — git clone | #6 | CACHED |
| build_pytorch 3/4 — pip deps + submodules | #13 | CACHED |
| build_pytorch 4/4 — compile + bdist_wheel | #14 | **~49s to 70%** (ccache warm) |
| build_torchvision | #15+ | pending |
| aic-vllm [2/3] | — | pending |
| aic-lmcache [3/3] | — | pending |

> ccache mount (`--mount=type=cache,target=/root/.cache/ccache`) is providing
> hits across runs. The PyTorch HIP compile went from ~1200s cold to est. <120s warm.

---

## Key bottlenecks

| Step | Cold time | Warm (ccache) | Arch multiplier |
|------|-----------|---------------|-----------------|
| build_pytorch — cmake configure | ~24s | ~24s | 1× |
| build_pytorch — ninja CXX (3826 obj) | ~580s | ~30s est. | 1× |
| build_pytorch — ninja HIP kernels | ~600s+ | ~60s est. | **2× for gfx1201+gfx1250** |
| build_pytorch — bdist_wheel | ~60s est. | ~30s est. | 1× |
| build_torchvision | ~300s est. | ~60s est. | 2× |
| aic-vllm — cmake + ninja | ~900s est. | unknown | 2× |
| aic-lmcache — AITER + Flash-Attention + NIXL | ~1200s est. | unknown | 2× |

The dual-arch build (`gfx1201,gfx1250`) roughly doubles every HIP compilation
step. Single-arch (`ROCM_ARCH=gfx1201`) would approximately halve build_pytorch
HIP time and all downstream HIP steps.

---

## Speedup opportunities

| Opportunity | Effort | Estimated gain |
|-------------|--------|----------------|
| **Layer cache** (`AIC_CACHE_DIR`) | Low — env var | Skips all unchanged stages entirely |
| **ccache** (already on via `--mount=type=cache`) | None — already active | ~10× HIP recompile speedup |
| **Single-arch dev build** (`ROCM_ARCH=gfx1201`) | Low — pass on CLI | ~2× HIP step speedup |
| **Parallel stages 2+3** (`make dist-build-parallel`) | Medium — needs Slurm or manual orchestration | ~1.5–2× total wall time |
| **Pre-built PyTorch wheel** (pull from registry instead of source build) | High — needs wheel publishing pipeline | Eliminates build_pytorch entirely (~1200s cold) |
| **`PYTORCH_REF` SHA pin** | Low — one-line Dockerfile change | Makes the git clone cacheable across machines (moving branch busts cache) |

---

## Cache notes

- **BuildKit layer cache** (`--cache-from/--cache-to type=local`): opt-in via
  `AIC_CACHE_DIR`. Populated on first successful build per arch slug. Skips
  entire stages when inputs are unchanged.
- **ccache** (`--mount=type=cache,target=/root/.cache/ccache`): always active;
  scoped to the BuildKit docker-container builder instance. Survives across
  `make build` runs on the same host. Not shared across machines.
- **BuildKit internal layer cache**: always active within the builder instance
  (`aic-local`). Lost if `docker buildx rm aic-local` is run.
- **`PYTORCH_REF=release/2.13`** is a moving branch — a new commit upstream
  invalidates the git clone layer and forces a full PyTorch recompile. Pin to
  a SHA for reproducible caching.
