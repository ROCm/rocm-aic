# Pip-installable Nightly Wheels

The source-built **vLLM**, patched **LMCache** and ROCm **NIXL** are published
as pip wheels on a rolling `nightly` GitHub Release, rebuilt automatically
whenever `main` changes (see
[`.github/workflows/rocm-aic-nightly-wheels-publication.yml`](../.github/workflows/rocm-aic-nightly-wheels-publication.yml)).
Install them into a matching ROCm environment:

```bash
pip install \
  https://github.com/ROCm/rocm-aic/releases/download/nightly/vllm-<ver>-cp312-cp312-linux_x86_64.whl \
  https://github.com/ROCm/rocm-aic/releases/download/nightly/lmcache-<ver>-cp312-cp312-linux_x86_64.whl \
  https://github.com/ROCm/rocm-aic/releases/download/nightly/nixl_rocm-<ver>-cp312-cp312-linux_x86_64.whl
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
| `nightly` | `gfx90a;gfx942;gfx950` | MI200 / MI300X / MI355X | every night |
| `nightly-rdna` | `gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201` | RDNA3 / RDNA4 | on request |
| `nightly-gfx1250` | `gfx1250` | rocjitsu-emulated GPU | on request |

The on-request groups are built by dispatching the workflow with the `groups`
input (`cdna,rdna`, `cdna,gfx1250`, or `cdna,rdna,gfx1250`).

## One NIXL for the whole stack

There is a single `nixl_rocm` wheel, not one per image. NIXL is built once in
`docker/base/Dockerfile` (the `build_nixl` stage) and both `aic-vllm` and
`aic-lmcache` inherit that one build, so the wheel the release publishes is the
same artifact both images run.

## Compatibility — read before installing

These are **not** manylinux wheels:

- ROCm **7.14.x**, Python **3.12**, **x86_64** only. They match the
  `rocm/dev-ubuntu-24.04:7.14.1-full` base; other ROCm/Python/arch
  combos will fail to import.
- Each wheel is compiled for one **arch group** (see the table above), named in
  the release notes. LMCache's HIP extension and vLLM's kernels are compiled for
  every arch in the group. `flash_attn` is CDNA-only and is absent from the RDNA
  and gfx1250 sets by design; vLLM falls back to Triton attention there.
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
| `wheels-nixl` | `docker/base/Dockerfile` | `nixl_rocm` |
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
