# Pip-installable Nightly Wheels

The source-built **vLLM**, patched **LMCache**, source-built **Mooncake ROCm**,
and **ROCm NIXL** are published as pip wheels on a rolling `nightly` GitHub
Release, rebuilt automatically whenever `main` changes (see
`.github/workflows/aic-nightly-wheels.yml`). Install them together into a
matching ROCm environment:

```bash
python3 -m pip install \
  "torch==2.13.0+rocm7.2" \
  "torchvision==0.28.0+rocm7.2" \
  --index-url https://download.pytorch.org/whl/rocm7.2

python3 -m pip install \
  https://github.com/ROCm/rocm-aic/releases/download/nightly/vllm-<ver>-cp312-cp312-linux_x86_64.whl \
  https://github.com/ROCm/rocm-aic/releases/download/nightly/mooncake_transfer_engine_rocm-<ver>-cp312-cp312-manylinux_2_39_x86_64.whl \
  https://github.com/ROCm/rocm-aic/releases/download/nightly/lmcache-<ver>-cp312-cp312-linux_x86_64.whl \
  https://github.com/ROCm/rocm-aic/releases/download/nightly/nixl_rocm-<ver>-cp312-cp312-linux_x86_64.whl
```

(Grab the exact filenames from the
[nightly release](https://github.com/ROCm/rocm-aic/releases/tag/nightly).)

**Compatibility — read before installing.** These are **not** manylinux wheels:

- ROCm **7.14.x**, Python **3.12**, **x86_64** only. They match the
  `rocm/dev-ubuntu-24.04:7.14.0-full` base; other ROCm/Python/arch
  combos will fail to import.
- Install the pinned ROCm Torch pair first as shown above. LMCache deliberately
  leaves its Torch dependency open, so resolving all four wheels from a clean
  environment without that prerequisite can select PyPI's CUDA Torch build.
- The wheels are built for the image's full multi-arch set (`gfx90a … gfx1201`);
  LMCache's HIP extension is compiled for all of them.
- The LMCache Mooncake extension resolves `libmooncake_store` from the companion
  `mooncake-transfer-engine-rocm` wheel, so those two wheels must be installed
  together. The exported pair does not depend on the image-only
  `/opt/mooncake-sdk` path.
- The `nixl_rocm` wheel bundles `libnixl` + the NIXL/UCX plugin `.so`s,
  but the **ROCm runtime (`libamdhip64`) and hipFile are external
  dependencies** — they must already be present on the host (they are,
  inside this image). It installs the `nixl_rocm` import package; the
  `nixl` compatibility shim is applied only inside the image.

These wheels are a convenience for reproducing the stack outside the
container; the supported deployment is still the Docker image built below.

> The wheels are produced by the `wheels` stage of the Dockerfile
> (`docker build --build-arg AIC_VERSION="$(<VERSION)" --target wheels
> --output type=local,dest=./wheels`);
> the default build target is unchanged and still yields the full runtime
> image.
