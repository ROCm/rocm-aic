# Pip-installable Nightly Wheels

The patched **LMCache** and **ROCm NIXL** are published as pip wheels on a
rolling `nightly` GitHub Release, rebuilt automatically whenever `main`
changes (see `.github/workflows/aic-nightly-wheels.yml`). Install them
into a matching ROCm environment:

```bash
pip install \
  https://github.com/ROCm/rocm-aic/releases/download/nightly/lmcache-<ver>-cp312-cp312-linux_x86_64.whl \
  https://github.com/ROCm/rocm-aic/releases/download/nightly/nixl_rocm-<ver>-cp312-cp312-linux_x86_64.whl
```

(Grab the exact filenames from the
[nightly release](https://github.com/ROCm/rocm-aic/releases/tag/nightly).)

**Compatibility — read before installing.** These are **not** manylinux wheels:

- ROCm **7.2.x**, Python **3.12**, **x86_64** only. They match the
  `rocm/dev-ubuntu-24.04:7.2.4-complete` base; other ROCm/Python/arch
  combos will fail to import.
- The wheels are built for the image's full multi-arch set (`gfx90a … gfx1201`);
  LMCache's HIP extension is compiled for all of them.
- The `nixl_rocm` wheel bundles `libnixl` + the NIXL/UCX plugin `.so`s,
  but the **ROCm runtime (`libamdhip64`) and hipFile are external
  dependencies** — they must already be present on the host (they are,
  inside this image). It installs the `nixl_rocm` import package; the
  `nixl` compatibility shim is applied only inside the image.

These wheels are a convenience for reproducing the stack outside the
container; the supported deployment is still the Docker image built below.

> The wheels are produced by the `wheels` stage of the Dockerfile
> (`docker build --target wheels --output type=local,dest=./wheels`);
> the default build target is unchanged and still yields the full runtime
> image.
