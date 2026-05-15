<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# vllm-radeon

ROCm **vLLM** + **LMCache** image (base **`vllm/vllm-openai-rocm:v0.19.0`**), with
**hipFile** from **ROCm/rocm-systems**, **fio** with **libhipfile**, and **`RADEON_*`**
naming. Work from **`recipies/vllm-radeon/`**.

## Where things live

| What you need | File |
| --- | --- |
| **`make build` / `make run`**, **`ROCM_ARCH`**, **`CONTAINER_NAME`**, mounts (**`DATA`**, **`LOG`**),
**`TZ`**, **`HF_TOKEN`**, **`HF_TOKEN_FILE`**, **`RADEON_LMCACHE_IO`**, **`ARGS`**, **`EXTRA_DOCKER_RUN_FLAGS`** | **`Makefile`** (see **`make help`**) |
| Image layers, LMCache / hipFile / **fio** build; **`ENTRYPOINT`** **`/app/scripts/vllm-server`** (**`Dockerfile`** **`COPY`**; **`make run`** overlays repo **`configs/`** + **`scripts/`**) | **`Dockerfile`** |
| vLLM + LMCache (**`--kv-transfer-config`**); **`RADEON_LMCACHE_IO`** selects template | **`scripts/vllm-server`** |
| LMCache **hipfile** (**GdsBackend**, **`gds_path`**) vs **posix** (**`fs`**
plugin, same **`DATA`/`gds_subdir`** as **`gds_path`**, no **`gds_path`** key) |
**`configs/lmcache-hipfile.yml`**, **`configs/lmcache-posix.yml`** |
| LMCache subdir + **`serve`** (load format, ais-stats, clear GDS) | **`configs/vllm-radeon.yaml`**, **`scripts/vllm_radeon_defaults.py`**, **`Makefile`** **`CONTAINER_DATA_DIR`**, **`CONTAINER_LOG_DIR`** |

## Quick start

```bash
export ROCM_ARCH=gfx1201   # e.g. RX 9070 XT; required for make build
make build
export HF_TOKEN=your_hf_token_here   # or HF_TOKEN_FILE in Makefile / env
make run
```

The **Makefile** bind-mounts **`configs/`** and **`scripts/`** to **`/app/configs`**
and **`/app/scripts`**, so YAML and Python helpers update without **`docker build`**.
Run **`make run`** from **`recipies/vllm-radeon/`** (so **`$(CURDIR)`** is correct),
or add matching **`-v`** flags with **`EXTRA_DOCKER_RUN_FLAGS`**.

Prepare the host path you mount as LMCache data (default host **`DATA`**
in **`Makefile`**: **`/mnt/lmcache-nvme`** → container **`/data`**). That
volume should hold only LMCache on-disk state (**`gds_subdir`**, runtime
YAML, chunk statistics, etc.). vLLM tee logs go under host **`LOG`**
(default **`recipies/vllm-radeon/logs`** → container **`/var/log/vllm-radeon`**,
file **`server.txt`**). Override with **`make run LOG=/other/host/dir`** or
**`CONTAINER_LOG_DIR`**. **`make run`** also passes **`TZ=America/Edmonton`**
(Edmonton, Alberta). **tzdata** uses that **IANA** id, not **`Canada/Edmonton`**;
override with **`make run TZ=...`**. vLLM and LMCache log timestamps follow
**`TZ`** in the container. For **`docker exec`**, use **`CONTAINER_NAME`**
(default **`vllm-radeon-gpu0`**, i.e. **`IMAGE_NAME`** + **`gpu`** + **`GPU`**);
override with **`make run CONTAINER_NAME=...`**.

## LMCache disk mode (**`RADEON_LMCACHE_IO`**)

**`make run`** passes **`RADEON_LMCACHE_IO`** (default **`hipfile`**). **`hipfile`**
uses LMCache **GdsBackend** + hipFile (**`gds_path`** under **`DATA`/`gds_subdir`**).
**`posix`** uses LMCache **`remote_storage_plugins: [fs]`** (POSIX filesystem
backend): **`extra_config.remote_storage_plugin.fs.base_path`** points at the
same directory as **`hipfile`** (**`DATA`/`gds_subdir`**). No **`gds_path`** key
in the runtime YAML (normal path; not **`fs://`**).

```bash
make run RADEON_LMCACHE_IO=posix
make run RADEON_LMCACHE_IO=hipfile   # default
```

## LMCache **long_doc_qa** benchmark

After vLLM is listening (e.g. **`curl -sS http://127.0.0.1:8000/v1/models`**),
run the upstream script from the image (**not** bind-mounted; it lives only
under **`/app/LMCache`** in the container). With **`GPU=0`**, vLLM listens on port
**`8000`**. Rebuild the image (**`make build`**) so the Dockerfile patch
applies;
otherwise **`--help`** hits upstream **`ValueError: incomplete format`** (a
**`%`** in **`--trim-fraction`** help text). Until rebuilt, skip **`--help`**
or inspect the script in the container.

```bash
docker exec -it vllm-radeon-gpu0 python3 \
  /app/LMCache/benchmarks/long_doc_qa/long_doc_qa.py --help
docker exec -it vllm-radeon-gpu0 python3 \
  /app/LMCache/benchmarks/long_doc_qa/long_doc_qa.py \
  --port 8000 --model Qwen/Qwen2.5-3B-Instruct \
  --num-documents 2 --hit-miss-ratio 1:1
```

Use the same name as **`make run`** (**`CONTAINER_NAME`**, default **`vllm-radeon-gpu0`** if **`IMAGE_NAME`** and **`GPU`** match defaults).

Match **`--model`** to **`VLLM_MODEL`** / **`vllm-radeon.yaml`** **`model_default`**;
match **`--port`** to **`800{GPU}`** from **`ROCR_VISIBLE_DEVICES`**.

## Grafana **`grafana/vllm-lmcache-prometheus.json`**

A sample Grafana dashboard. Import into your Grafana server. This may need 
adjusting to match your exporter naming.

