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
**`TZ`**, **`HF_TOKEN`**, **`HF_TOKEN_FILE`**, **`RADEON_LMCACHE_IO`**, **`VLLM_SERVER_DEV_MODE`**, **`ARGS`**, **`EXTRA_DOCKER_RUN_FLAGS`** | **`Makefile`** (see **`make help`**) |
| Image layers, LMCache / hipFile / **fio** build; **`patches/`** applies [LMCache#3008][lmcache-pr-3008] (`cache_salt` in V1 keys); **`ENTRYPOINT`** **`/app/scripts/vllm-server`** (**`make run`** overlays **`configs/`** + **`scripts/`**) | **`Dockerfile`**, **`patches/`** |
| vLLM + LMCache (**`--kv-transfer-config`**); **`RADEON_LMCACHE_IO`** selects template | **`scripts/vllm-server`** |
| LMCache **hipfile** (**GdsBackend**, **`gds_path`**) vs **posix** (**`fs`**
plugin, same **`DATA`/`subdir`** as **`gds_path`**, no **`gds_path`** key) |
**`configs/lmcache-hipfile.yml`**, **`configs/lmcache-posix.yml`** |
| LMCache subdir + **`serve`** (load format, ais-stats, clear GDS) | **`configs/vllm-radeon.yaml`**, **`scripts/vllm-radeon-defaults.py`**, **`Makefile`** **`CONTAINER_DATA_DIR`**, **`CONTAINER_LOG_DIR`** |
| Gutenberg chunks + questions + load / AIC A/B test | **`make data`**, **`scripts/test-aic.py`**, **`run-long.sh`** |
| Parse engine log → CSV/SVG | **`scripts/parse-vllm-engine-log-timeseries.py`** |

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
volume should hold only LMCache on-disk state (**`subdir`**, runtime
YAML, chunk statistics, etc.). vLLM tee logs go under host **`LOG`**
(default **`recipies/vllm-radeon/logs`** → container **`/var/log/vllm-radeon`**,
file **`server.txt`**). Override with **`make run LOG=/other/host/dir`** or
**`CONTAINER_LOG_DIR`**. **`make run`** also passes **`TZ=America/Edmonton`**
(Edmonton, Alberta). **tzdata** uses that **IANA** id, not **`Canada/Edmonton`**;
override with **`make run TZ=...`**. vLLM and LMCache log timestamps follow
**`TZ`** in the container. For **`docker exec`**, use **`CONTAINER_NAME`**
(default **`vllm-radeon-gpu0`**, i.e. **`IMAGE_NAME`** + **`gpu`** + **`GPU`**);
override with **`make run CONTAINER_NAME=...`**.

## vLLM dev mode (**`VLLM_SERVER_DEV_MODE`**)

**`make run`** sets **`VLLM_SERVER_DEV_MODE=1`** by default (also defaulted in
**`scripts/vllm-server`**). That enables dev-only HTTP routes such as
**`POST /reset_prefix_cache`** to clear the GPU prefix cache without restarting
the container. Disable with **`make run VLLM_SERVER_DEV_MODE=0`**. Restart vLLM
after changing this variable so the server picks it up.

```bash
curl -sS -X POST "http://127.0.0.1:8000/reset_prefix_cache"
```

Port **`800{GPU}`** matches **`ROCR_VISIBLE_DEVICES`** (e.g. **`8000`** for
**`GPU=0`**).

## LMCache disk mode (**`RADEON_LMCACHE_IO`**)

**`make run`** passes **`RADEON_LMCACHE_IO`** (default **`hipfile`**). **`hipfile`**
uses LMCache **GdsBackend** + hipFile (**`gds_path`** under **`DATA`/`subdir`**).
**`posix`** uses LMCache **`remote_storage_plugins: [fs]`** (POSIX filesystem
backend): **`extra_config.remote_storage_plugin.fs.base_path`** points at the
same directory as **`hipfile`** (**`DATA`/`subdir`**). No **`gds_path`** key
in the runtime YAML (normal path; not **`fs://`**).

```bash
make run RADEON_LMCACHE_IO=posix
make run RADEON_LMCACHE_IO=hipfile   # default
```

### Runtime storage mode (no vLLM restart)

After **`make build`** (applies **`lmcache-storage-mode-switch.patch`**), the
LMCache worker HTTP API on port **`699{GPU}+1`** (e.g. **`6991`** for **`GPU=0`**)
exposes **`GET|POST /storage/mode`**. This closes the active disk backend,
updates config to match the hipfile or posix profile (same fields as
**`vllm-server`** materialization), and recreates backends. **KV on disk is not
portable** between modes; repopulate after switching.

```bash
curl -sS "http://127.0.0.1:6991/storage/mode"
curl -sS -X POST "http://127.0.0.1:6991/storage/mode" \
  -H 'Content-Type: application/json' \
  -d '{"mode":"posix","fs_base_path":"/data/lmcache/"}'
curl -sS -X POST "http://127.0.0.1:6991/storage/mode" \
  -H 'Content-Type: application/json' \
  -d '{"mode":"hipfile","gds_path":"/data/lmcache"}'
```

Startup mode still comes from **`RADEON_LMCACHE_IO`** at **`make run`**; use
**`/storage/mode`** only when you need to flip layouts on a live server.

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

## Gutenberg long-context fixtures

**`data/`** is **not** tracked (see **`.gitignore`**); generate fixtures locally.
From **`recipies/vllm-radeon/`**:

```bash
make data
```

Defaults: **War and Peace** (**`BOOK_PG_ID=2600`**, **`BOOK_SLUG=war-and-peace`**).
Override book, chunk size, or count, e.g.:

```bash
make data BOOK_SLUG=pride-and-prejudice BOOK_PG_ID=1342 \
  BOOK_TITLE='Pride and Prejudice' BOOK_AUTHOR='Jane Austen'
```

Equivalent manual steps (hyphenated scripts under **`scripts/`**):

```bash
python3 scripts/split-gutenberg-random-chunks.py \
  --pg-id 2600 --slug war-and-peace \
  -o data/war-and-peace --count 100

python3 scripts/gen-questions-json.py \
  --slug war-and-peace --title "War and Peace" --author "Leo Tolstoy" \
  --pg-id 2600

# Optional: supply your own question list (.json array or one question per line):
python3 scripts/gen-questions-json.py \
  --slug war-and-peace --title "War and Peace" --author "Leo Tolstoy" \
  --extra-questions /path/to/my-questions.json

# Load test (run after data/<slug>/ exists):
BOOK_SLUG=war-and-peace ./run-long.sh

# LMCache populate / cold / warm A/B (repo root: pip install -r requirements.txt):
python3 scripts/test-aic.py -o logs/test-aic.json
# Same chunk + cache_salt; reset_prefix_cache before cold/warm; cold bypasses GDS.
# Fresh NVMe store: new --run-id (or --skip-populate if already stored).
```

Chunk files are **`data/<slug>/<slug>-<chunk-label>.<offset>.txt`** (default
label **`10k`** for 10 000 words). **`run-long.sh`** honors **`BOOK_SLUG`**,
**`BOOK_DATA_DIR`**, and **`QUESTIONS_FILE`**.

## Grafana **`grafana/vllm-lmcache-prometheus.json`**

A sample Grafana dashboard. Import into your Grafana server. This may need
adjusting to match your exporter naming.

<!-- References -->
[lmcache-pr-3008]: https://github.com/LMCache/LMCache/pull/3008

