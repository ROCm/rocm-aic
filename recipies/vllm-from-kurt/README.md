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
inside the image). Defaults apply when variables are unset; override as
needed:

| Variable              | Default | Purpose |
| --------------------- | ------- | ------- |
| `GPU`                 | `0` | GPU index for `ROCR_VISIBLE_DEVICES` |
| `DATA`                | `/mnt/lmcache-nvme` | Host path mounted at **`${CONTAINER_DATA_DIR}`** (see below) |
| `HF_HOME`             | `${HOME}/.cache/huggingface` | Host Hugging Face cache mounted at **`${CONTAINER_HF_HOME}`** |
| `CONTAINER_DATA_DIR`  | `/data` | In-container mount **target** for **`DATA`**; passed as **`KURT_CONTAINER_DATA_DIR`** |
| `CONTAINER_HF_HOME`   | `/hf` | Mount target for host **`HF_HOME`**; in-container **`HF_HOME`** matches it |
| `HF_TOKEN`            | *(empty)* | If empty, read from `HF_TOKEN_FILE` |
| `HF_TOKEN_FILE`       | `${HOME}/.batesste-hugging-face-read-march-2026.token` | Token file when `HF_TOKEN` is unset |

**Inside the container,** **`./vllm-container`** mounts host **`HF_HOME`** at
**`/hf`** and sets **`HF_HOME=/hf`** plus **`HUGGINGFACE_HUB_CACHE`**,
**`HF_HUB_CACHE`**, **`HF_DATASETS_CACHE`**, **`VLLM_CACHE_ROOT`**,
**`VLLM_CONFIG_ROOT`**, **`TORCH_HOME`**, and **`TORCHINDUCTOR_CACHE_DIR`**
under that same tree so Hub weights, vLLM assets, and common compile caches
stay on the host (Transformers uses **`HF_HOME`**; we do not set deprecated
**`TRANSFORMERS_CACHE`**). The data-root env is **`KURT_CONTAINER_DATA_DIR`**
(not a **`VLLM_*`** name) so vLLM’s env parser does not warn about unknown
variables. **`./vllm-container`** also sets **`PYTORCH_ALLOC_CONF`** to
**`expandable_segments:True`** by default (host can set **`PYTORCH_ALLOC_CONF`**
before launch to override). It sets **`KURT_CONTAINER_DATA_DIR`** to **`${CONTAINER_DATA_DIR}`**
for **`/app/`** scripts: LMCache, **fio**, and **`server.txt`** share that data
root. Align **`CONTAINER_*`**, **`-v`**, and **`KURT_CONTAINER_DATA_DIR`** to one path.

**Llama weights:** default id **`meta-llama/Llama-3.1-8B-Instruct`**; set env
**`VLLM_MODEL`** inside the container to change it. The Hub checkpoint is gated:
accept the license on the model card; keep **`HF_TOKEN`** or **`HF_TOKEN_FILE`**
on the host so gated pulls use the **`/hf`** cache mount.

The script runs **`mkdir -p`** on **`DATA`** and **`HF_HOME`** before **`docker run`**.

Minimal example (defaults only; ensure the default token file exists or set
**`HF_TOKEN`** / **`HF_TOKEN_FILE`**):

```bash
./vllm-container
```

Override example:

```bash
export GPU=1
export DATA=/srv/vllm-data
export HF_HOME="${HOME}/.cache/huggingface"
export HF_TOKEN="$(cat ~/.hf_token)"

./vllm-container
```

That starts an interactive shell (`ENTRYPOINT` is `bash`) with host
networking, `ipc=host`, GPU devices, and the mounts above. From that
shell you can run the scripts under `/app/`.

## In-container scripts (under `/app/`)

- **`vllm-server-hipfile`** — LMCache GDS backend via hipFile; uses
  `ROCR_VISIBLE_DEVICES` to pick ports (`699${PORT}` / `800${PORT}`).
  Runs **`ais-stats`** from the hipFile build ahead of **`vllm serve`**
  (dummy **`meta-llama/Llama-3.1-8B-Instruct`** by default, prefix caching,
  AITER attention). Defaults: **`--max-model-len 8192`**, **`--max-num-batched-tokens 4096`**,
  **`--enforce-eager`** (override with **`KURT_MAX_MODEL_LEN`**, **`KURT_MAX_NUM_BATCHED_TOKENS`**,
  **`KURT_ENFORCE_EAGER=0`**). Log: **`${KURT_CONTAINER_DATA_DIR}/server.txt`**.
- **`vllm-server-native-disk`** — Same vLLM/LMCache wiring but local disk cache
  under **`file://${KURT_CONTAINER_DATA_DIR}/lmcache_test/`** (default
  **`file:///data/lmcache_test/`**). Model: **`VLLM_MODEL`** or
  **`meta-llama/Llama-3.1-8B-Instruct`** (real weights; needs HF token).
  Same **`KURT_*`** limits as **`vllm-server-hipfile`** so **`profile_run`**
  fits ~16 GiB; raise **`KURT_MAX_MODEL_LEN`** (e.g. **`131072`**) on large GPUs.
- **`vllm-benchmark`** — Drives **`long_doc_qa.py`** against the server on port
  **`800${ROCR_VISIBLE_DEVICES}`**; **`--model`** matches **`vllm-server-*`**.
  Defaults **`KURT_BENCH_NUM_DOCUMENTS=8`**, **`KURT_BENCH_DOCUMENT_LENGTH=6000`**
  (fits default **`KURT_MAX_MODEL_LEN=8192`**); restore long-doc stress with
  higher env values when **`KURT_MAX_MODEL_LEN`** is raised.
- **`run_long_doc_qa.py`** — Starts **`vllm serve`** (hipfile or native path),
  waits for **`/v1/models`**, then runs **`long_doc_qa.py`**; see
  **`python3 /app/run_long_doc_qa.py --help`**. Uses **`VLLM_MODEL`** (or Llama
  3.1 8B Instruct) for both server and benchmark unless you override **`--model`**.
- **`fio-benchmark`** — hipFile **fio** read on **`${KURT_CONTAINER_DATA_DIR}/rand1G.dat`**
  (default **`/data/rand1G.dat`**; creates 1 GiB random file if missing).
- **`hipfile-bench.py`** — Small Python hipfile/HIP exercise (not the full
  vLLM stack); pass a chunk directory under **`${KURT_CONTAINER_DATA_DIR}`** after a run.

### GPU memory (HIP OOM while **`vllm serve`** starts)

Bf16 **Llama‑3.1‑8B** uses ~**15 GiB** for weights alone; vLLM’s **KV
`profile_run`** still needs a little more VRAM. **`vllm-server-*`** therefore
defaults **`--max-model-len 8192`**, **`--max-num-batched-tokens 4096`**, and
**`--enforce-eager`**. On MI300-class GPUs raise limits, e.g. **`export
KURT_MAX_MODEL_LEN=131072`**, **`export
KURT_MAX_NUM_BATCHED_TOKENS=8192`**, and **`export KURT_ENFORCE_EAGER=0`**.
Prefer **`KURT_*`** for recipe-only overrides.
**`vllm-server-*`** still reads legacy **`VLLM_GPU_MEMORY_UTILIZATION`** and
**`VLLM_ENFORCE_EAGER`** for the initial value, then **unsets** them before
**`vllm`** so the engine does not log unknown **`VLLM_*`** env vars.
**`./vllm-container`** sets **`PYTORCH_ALLOC_CONF`** to
**`expandable_segments:True`** by default; override on the host if needed.

- **`export KURT_GPU_MEMORY_UTILIZATION=0.55`** — fraction passed to
  **`vllm serve --gpu-memory-utilization`** (default **`0.72`**).
- **`export KURT_PYTORCH_ALLOC_CONF=expandable_segments:True`** in the
  container if you disabled the host default from **`vllm-container`**.
- Use a **smaller** **`VLLM_MODEL`** or a **quantized** checkpoint if bf16 8B
  still does not fit.

Ensure **`ROCR_VISIBLE_DEVICES`** is set inside the container when you
use the server or benchmark scripts (the host wrapper sets it from
`GPU`).
