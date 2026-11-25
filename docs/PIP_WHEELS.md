# Pip-installable Nightly Wheels

The source-built **vLLM**, patched **LMCache** and ROCm **NIXL** are published
as pip wheels on a rolling `nightly` GitHub Release, rebuilt automatically
whenever `main` changes (see
[`.github/workflows/rocm-aic-nightly-wheels-publication.yml`](../.github/workflows/rocm-aic-nightly-wheels-publication.yml)).
Install them into a matching ROCm environment:

```bash
pip install \
  https://github.com/ROCm/rocm-aic/releases/download/nightly/vllm-<ver>-cp312-cp312-linux_x86_64.whl \
  https://github.com/ROCm/rocm-aic/releases/download/nightly/lmcache-<ver>-cp314-cp314-linux_x86_64.whl \
  https://github.com/ROCm/rocm-aic/releases/download/nightly/nixl_rocm-<ver>-cp314-cp314-linux_x86_64.whl
```

(Grab the exact filenames from the
[nightly release](https://github.com/ROCm/rocm-aic/releases/tag/nightly).)

`amd_aiter` and — on CDNA arch sets only — `flash_attn` wheels are attached to
the same release. vLLM picks them up at runtime when they are installed; they
are optional.

## Pick the release that matches your GPU

Every arch group ships wheels with **identical filenames** but different GPU
code, so each has its own rolling release. Installing from the wrong one gives
you a wheel that imports and then fails on the first kernel launch:

| Release tag | Arch group | Hardware | Built |
| --- | --- | --- | --- |
| `nightly` | `gfx90a;gfx942;gfx950;gfx1250` | MI200 / MI300X / MI355X, and the rocjitsu-emulated gfx1250 | every night |
| `nightly-rdna` | `gfx1151;gfx1201` | RDNA3 / RDNA4 | on request |

The on-request group is built by dispatching the workflow with the `groups`
input set to `cdna,rdna`.

## One NIXL for the whole stack

There is a single `nixl_rocm` wheel, not one per image. NIXL is built once in
`docker/base/Dockerfile` (the `build_nixl` stage) and both `aic-vllm` and
`aic-lmcache` inherit that one build, so the wheel the release publishes is the
same artifact both images run.

## Docker images from the wheels

`make build-from-wheels` assembles `aic-vllm` and `aic-lmcache` from a release's
wheels instead of compiling them, for a node that should run the stack without
the source builds. It takes minutes, and the images carry the tags `make build`
gives them, so `make up` runs either.

```bash
make ensure-buildx                        # once, if docker has no buildx plugin
make build-from-wheels ROCM_ARCH=gfx950   # the nightly release
```

It downloads the release's wheels to `wheels-release/<tag>/`, checks that they
carry GPU code for `ROCM_ARCH`, and bakes the `from-wheels` group: an `aic-base`
from the `torch`, `torchvision`, `triton` and `nixl_rocm` wheels on the same
ROCm runtime packages as the source build, then the two images on top of it.
The full ROCm image is not pulled: the apt key of the runtime packages is
downloaded from AMD instead. No wheel holds the sidecars. The two exporters
build as in `make build`; `aic-hsa-snoop`, which compiles on the full image,
is pulled from `AIC_PUBLISHED_PREFIX` and built only if its tag is not there.

| Variable | Default | |
| --- | --- | --- |
| `WHEELS_RELEASE` | `nightly` | Release tag to download; pick the one that covers your GPU (table above). Empty skips the download |
| `WHEELS_REPO` | `ROCm/rocm-aic` | Repository of the release |
| `AIC_WHEELS_SRC` | `wheels-release/<tag>` | Directory the wheels are downloaded to and installed from |

To build from wheels you already have (the run artifact of a PR, or `make wheels`):

```bash
make build-from-wheels WHEELS_RELEASE= AIC_WHEELS_SRC=$PWD/wheels-out
```

An image built this way is not the source-built one:

- `libamdhip64` is the packaged library, not the patched build that fixes the
  signal ring of cross-process GPU events. Both images replace those events
  with a producer-side sync, so the stack does not use the ring.
- NIXL is the `nixl_rocm` wheel, without UCX. The POSIX and AIS_MT backends
  load; the UCX backend does not. The image adds the library links and loader
  path the wheel's telemetry exporters need; a plain `pip install` of the
  wheel does not have them, and a NIXL agent fails to start there with
  `NIXL_TELEMETRY_ENABLE=y`.
- AITER kernels are compiled on first use, as in a default `make build`.

## Compatibility — read before installing

These are **not** manylinux wheels:

- ROCm **10.2**, Python **3.14**, **x86_64** only. They match the
  `rocm/dev-ubuntu-26.04:10.2.0-full` base; other ROCm/Python/arch
  combos will fail to import.
- Each wheel is compiled for one **arch group** (see the table above), named in
  the release notes. LMCache's HIP extension and vLLM's kernels are compiled for
  every arch in the group. `flash_attn` is CDNA-only: it is compiled for the gfx9
  archs in a group, so `nightly` carries it (without gfx1250 code) and
  `nightly-rdna` does not; vLLM falls back to Triton attention where it is absent
  or the GPU is not gfx9.
- The `nixl_rocm` wheel bundles `libnixl` + the NIXL/UCX plugin `.so`s,
  but the **ROCm runtime (`libamdhip64`) and hipFile are external
  dependencies** — they must already be present on the host (they are,
  inside this image). It installs the `nixl_rocm` import package; the
  `nixl` compatibility shim is applied only inside the image.

These wheels are a convenience for reproducing the stack outside the
container; the supported deployment is still the Docker image.

## Building the wheels yourself

The wheels come out of the bake `wheels` group — three scratch stages, one per
image, that hold nothing but `.whl` files:

| bake target | Dockerfile | wheels |
| --- | --- | --- |
| `wheels-base` | `docker/base/Dockerfile` | `nixl_rocm`, `torch`, `torchvision`, `triton` |
| `wheels-lmcache` | `docker/lmcache/Dockerfile` | `lmcache` |
| `wheels-vllm` | `docker/vllm/Dockerfile` | `vllm`, `amd_aiter`, `flash_attn` |

Locally:

```bash
make wheels ROCM_ARCH='gfx942;gfx950' WHEELS_DIR=./wheels-out
```

On a Slurm cluster (what CI does — the PyTorch and vLLM source builds behind
these wheels do not fit in a GitHub-hosted runner's 6h job limit):

```bash
make dist-wheels AIC_ROCM_ARCH='gfx942;gfx950'
```

Either way the result is a flat directory of `.whl` files, checked by
`.github/scripts/workflows/check-wheels.sh` — the one definition of the
published wheel set, run on the build node and again before upload.

## Testing a change to the wheels from a PR

Comment `/run-wheels` on the PR. The workflow builds the PR head SHA for the
CDNA group on the SPUR cluster and uploads the wheels as a run artifact with
7-day retention.

A PR run never publishes: no rolling release is touched, and nothing becomes
`pip install`-able from a URL. Only scheduled runs and manual
`workflow_dispatch` publish. Download the artifact from the run page to test
the wheels from a PR on a node.
