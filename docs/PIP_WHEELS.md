# Pip-installable Nightly Wheels

The stack's source-built Python packages are published as pip wheels on
rolling GitHub pre-release pages, rebuilt automatically whenever `main` changes (see
`.github/workflows/rocm-aic-nightly-wheels-publication.yml`):

| Wheel | Package |
| --- | --- |
| `vllm-<ver>+rocm-…` | vLLM, built from source with the AIC patches |
| `amd_aiter-…` | AITER, the ROCm attention/GEMM kernel library vLLM uses |
| `flash_attn-…` | FlashAttention (CDNA releases only) |
| `lmcache-…` | LMCache with the AIC patches and its HIP extension |
| `nixl_rocm-…` | NIXL with the ROCm AIS_MT plugin |

Each GPU architecture group has its own release, because the wheels carry
compiled GPU code but the same filenames:

| Release | GPU arch set | Built |
| --- | --- | --- |
| [`nightly`](https://github.com/ROCm/rocm-aic/releases/tag/nightly) | `gfx90a;gfx942;gfx950` (CDNA: MI200, MI300X, MI355X) | every night `main` changed |
| [`nightly-rdna`](https://github.com/ROCm/rocm-aic/releases/tag/nightly-rdna) | `gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201` (RDNA3/4) | on manual dispatch only; may lag `nightly` |

Install them into a matching ROCm environment, taking the exact filenames from
the release page (its notes carry a ready-made `pip install` line):

```bash
pip install \
  https://github.com/ROCm/rocm-aic/releases/download/nightly/vllm-<ver>+rocm-<tags>.whl \
  https://github.com/ROCm/rocm-aic/releases/download/nightly/amd_aiter-<ver>-cp312-cp312-linux_x86_64.whl \
  https://github.com/ROCm/rocm-aic/releases/download/nightly/flash_attn-<ver>-cp312-cp312-linux_x86_64.whl \
  https://github.com/ROCm/rocm-aic/releases/download/nightly/lmcache-<ver>-cp312-cp312-linux_x86_64.whl \
  https://github.com/ROCm/rocm-aic/releases/download/nightly/nixl_rocm-<ver>-cp312-cp312-linux_x86_64.whl
```

A versioned `v*` release carries the same wheel set, built for both groups'
architectures in one wheel each (and therefore without FlashAttention).

**Compatibility — read before installing.** These are **not** manylinux wheels:

- ROCm **7.14.x**, Python **3.12**, **x86_64** only. They are built on the
  `rocm/dev-ubuntu-24.04:7.14.1-full` base; other ROCm/Python/arch
  combos will fail to import.
- GPU code is compiled only for the release's arch set; use the release that
  matches your GPU.
- The C++ extensions are linked against the stack's PyTorch, built from source
  from the commit named in the release notes. That PyTorch is not published;
  a different PyTorch build may not be ABI-compatible.
- The `nixl_rocm` wheel bundles `libnixl` + the NIXL/UCX plugin `.so`s,
  but the **ROCm runtime (`libamdhip64`) and hipFile are external
  dependencies** — they must already be present on the host (they are,
  inside this image). It installs the `nixl_rocm` import package; the
  `nixl` compatibility shim is applied only inside the image.
- The AITER wheel is AITER as built. Inside the image the AITER JIT is also
  patched to target the image's arch list and its core module is
  pre-compiled; the wheel instead JIT-compiles for the local GPU on first use.

These wheels are a convenience for reproducing the stack outside the
container; the supported deployment is still the Docker image built below.

> The wheels are the `wheels` stages of `docker/base/Dockerfile` (NIXL),
> `docker/lmcache/Dockerfile` and `docker/vllm/Dockerfile` (vLLM, AITER,
> FlashAttention), built by the `wheels` group of `docker/docker-bake.hcl`:
>
> ```bash
> ROCM_ARCH='gfx90a;gfx942;gfx950' AIC_VERSION="$(<VERSION)" \
>   docker buildx bake -f docker/docker-bake.hcl wheels   # -> wheels-out/{nixl,lmcache,vllm}/
> ```
>
> Those stages are not part of any runtime image, and the default build
> targets are unchanged.
