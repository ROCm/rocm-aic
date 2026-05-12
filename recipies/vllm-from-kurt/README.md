<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# vLLM ROCm image (Kurt stack)

This recipe builds **`vllm-kurt`**, a ROCm container derived from
`vllm/vllm-openai-rocm:v0.19.0`. It layers Riley Dixon’s LMCache fork
(build-time `LMCACHE_SHA1`), builds **hipFile** and **fio** with the
hipFile I/O engine, installs the bundled **hipfile** Python wheel
(`cp312`, `linux_x86_64`), and copies helper scripts into `/app/`.

## Makefile (local checks and build)

Run from **`recipies/vllm-from-kurt/`**. **`make`** (or **`make help`**)
lists targets.

| Target | Purpose |
| ------ | ------- |
| **`test`** / **`lint`** | Run **Hadolint** on `Dockerfile` (Hadolint runs in Docker against the repo **`.hadolint.yaml`**) and **shellcheck** on `vllm-container` and the `scripts/` shell helpers. |
| **`lint-hadolint`** | Dockerfile only. |
| **`lint-shell`** | Shell scripts only (requires **`shellcheck`** on the host). |
| **`build`** / **`vllm-kurt`** | `docker build -t vllm-kurt` with **`ROCM_ARCH`** (and optional **`LMCACHE_SHA1`**). |

Examples:

```bash
make test
export ROCM_ARCH=gfx942
make build
```

## Prerequisites

- Docker on a host with AMD GPUs (`/dev/kfd`, `/dev/dri`).
- **`shellcheck`** installed on the host if you run **`make test`** or
  **`make lint-shell`** (e.g. `apt install shellcheck`).
- **`ROCM_ARCH`** matching your GPU (for example `gfx942` for MI300,
  or your GCN/CDNA target string for `PYTORCH_ROCM_ARCH` / hipFile
  CMake).

## Build

From this directory (after **`make test`** if you want the same static
checks as CI for the Dockerfile and host scripts):

```bash
export ROCM_ARCH=gfx942   # adjust for your GPU
docker build -t vllm-kurt --build-arg ROCM_ARCH="${ROCM_ARCH}" .
```

Or use the Makefile (same requirement on **`ROCM_ARCH`**):

```bash
export ROCM_ARCH=gfx942
make build
# or: make vllm-kurt
```

Optional: override the LMCache commit when building:

```bash
export ROCM_ARCH=gfx942
make build LMCACHE_SHA1=your_git_sha_here
```

## Run the container

The **`vllm-container`** script is meant to be run on the **host** (not
inside the image). It expects:

| Variable   | Purpose |
| ---------- | ------- |
| `GPU`      | GPU index passed to Docker as `ROCR_VISIBLE_DEVICES` |
| `DATA`     | Host path mounted read-write at `/data` in the container |
| `HF_HOME`  | Hugging Face cache directory (mounted at `/hf`) |
| `HF_TOKEN` | Hugging Face token (passed into the container) |

Example:

```bash
export GPU=0
export DATA=/srv/vllm-data
export HF_HOME="${HOME}/.cache/huggingface"
export HF_TOKEN="$(cat ~/.hf_token)"   # however you store the token

./vllm-container
```

That starts an interactive shell (`ENTRYPOINT` is `bash`) with host
networking, `ipc=host`, GPU devices, and the mounts above. From that
shell you can run the scripts under `/app/`.

## In-container scripts (under `/app/`)

- **`vllm-server-hipfile`** — LMCache GDS backend via hipFile; uses
  `ROCR_VISIBLE_DEVICES` to pick ports (`699${PORT}` / `800${PORT}`).
  Runs **`ais-stats`** from the hipFile build ahead of **`vllm serve`**
  (dummy `openai/gpt-oss-120b`, prefix caching, AITER attention). Log:
  `/app/server.txt`.
- **`vllm-server-native-disk`** — Same vLLM/LMCache wiring but local
  disk cache under `file:///data/lmcache_test/`.
- **`vllm-benchmark`** — Drives the long-doc LMCache benchmark against
  the server on port `800${ROCR_VISIBLE_DEVICES}`.
- **`fio-benchmark`** — hipFile **fio** read job on `/data/rand1G.dat`
  (creates a 1 GiB random file if missing).
- **`hipfile-bench.py`** — Small Python hipfile/HIP exercise (not the
  full vLLM stack).

Ensure **`ROCR_VISIBLE_DEVICES`** is set inside the container when you
use the server or benchmark scripts (the host wrapper sets it from
`GPU`).
