<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# vllm-lmcache-hipfile

ROCm **vLLM** + **LMCache** image (base **`vllm/vllm-openai-rocm:v0.19.0`**), with
**hipFile** from **ROCm/rocm-systems**, **fio** with **libhipfile**, and **`VLH_*`**
naming. Work from **`recipies/vllm-lmcache-hipfile/`**.

Part of [rocm-aic](../../README.md). Host Python deps for Gutenberg benchmarks
and **`test-aic.py`**: from the repo root, `pip install -r requirements.txt`
(see [benchmarks/llm-prefill-benchmark](../../benchmarks/llm-prefill-benchmark/README.md)).

## Contents

- [Where things live](#where-things-live)
- [Quick start](#quick-start)
- [rocm-aic-exporter.py](#rocm-aic-exporterpy-prometheus-textfile)
- [MFU metrics](#mfu-metrics---enable-mfu-metrics)
- [vLLM dev mode](#vllm-dev-mode-vllm_server_dev_mode)
- [LMCache disk mode](#lmcache-disk-mode-vlh_lmcache_io)
- [LMCache logging and hipFile buffer](#lmcache-logging-and-hipfile-buffer)
- [LMCache long_doc_qa benchmark](#lmcache-long_doc_qa-benchmark)
- [Gutenberg long-context fixtures](#gutenberg-long-context-fixtures)
- [GitHub Actions CI](#github-actions-ci)
- [Grafana dashboard](#grafana-dashboard)

## Where things live

| What you need | File |
| --- | --- |
| **`make build` / `make run`**, **`ROCM_ARCH`**, **`CONTAINER_NAME`**, mounts (**`DATA`**, **`LOG`**),
**`TZ`**, **`HF_TOKEN`**, **`HF_TOKEN_FILE`**, **`VLH_LMCACHE_IO`**, **`VLLM_SERVER_DEV_MODE`**, **`ARGS`**, **`EXTRA_DOCKER_RUN_FLAGS`** | **`Makefile`** (see **`make help`**) |
| Image layers, LMCache / hipFile / **fio** build; **`patches/`** (cache_salt, storage mode, log noise, sha256_cbor); **`ENTRYPOINT`** **`/app/scripts/vllm-server`** (**`make run`** overlays **`configs/`** + **`scripts/`**) | **`Dockerfile`**, **`patches/`** |
| vLLM + LMCache (**`--kv-transfer-config`**); **`VLH_LMCACHE_IO`** selects template | **`scripts/vllm-server`** |
| LMCache **hipfile** (**GdsBackend**, **`gds_path`**) vs **posix** (**`fs`**
plugin, same **`DATA`/`subdir`** as **`gds_path`**, no **`gds_path`** key) |
**`configs/lmcache-hipfile.yml`**, **`configs/lmcache-posix.yml`** |
| LMCache subdir + **`serve`** (load format, ais-stats, clear GDS,
**`enable_mfu_metrics`**) | **`configs/vllm-lmcache-hipfile.yaml`**, **`scripts/vllm-lmcache-hipfile-defaults.py`**, **`Makefile`** **`CONTAINER_DATA_DIR`**, **`CONTAINER_LOG_DIR`** |
| Gutenberg chunks + questions + load / AIC A/B test | **`make data`** (delegates to [llm-prefill-benchmark](../../benchmarks/llm-prefill-benchmark/)); **`run-long.sh`** / **`test-aic.py`** there |
| LMCache / NVMe textfile metrics for Grafana | [recipies/common/scripts/rocm-aic-exporter.py](../common/scripts/rocm-aic-exporter.py) |
| Parse engine log → CSV/SVG | [recipies/common/scripts/parse-vllm-engine-log-timeseries.py](../common/scripts/parse-vllm-engine-log-timeseries.py) |
| Slurm + Gutenberg **`run-long.sh`** | **`run-slurm.sh`**, **`.slurm/vllm-lmcache-hipfile.sbatch`** (benchmark via **`LLM_PREFILL_BENCH_ROOT`**) |

## Quick start

```bash
export ROCM_ARCH=gfx1201   # e.g. RX 9070 XT; required for make build
make build
export HF_TOKEN=your_hf_token_here   # or HF_TOKEN_FILE in Makefile / env
make run
```

The **Makefile** bind-mounts **`configs/`** and **`scripts/`** to **`/app/configs`**
and **`/app/scripts`**, so YAML and Python helpers update without **`docker build`**.
Run **`make run`** from **`recipies/vllm-lmcache-hipfile/`** (so **`$(CURDIR)`** is correct),
or add matching **`-v`** flags with **`EXTRA_DOCKER_RUN_FLAGS`**.

Prepare the host path you mount as LMCache data (default host **`DATA`**
in **`Makefile`**: **`/mnt/lmcache-nvme`** → container **`/data`**). That
volume should hold only LMCache on-disk state (**`subdir`**, runtime
YAML, chunk statistics, etc.). vLLM tee logs go under host **`LOG`**
(default **`recipies/vllm-lmcache-hipfile/logs`** → container **`/var/log/vllm-lmcache-hipfile`**,
file **`server.txt`**). Override with **`make run LOG=/other/host/logs`** or
**`CONTAINER_LOG_DIR`**. **`make run`** also passes **`TZ=America/Edmonton`**
(Edmonton, Alberta). **tzdata** uses that **IANA** id, not **`Canada/Edmonton`**;
override with **`make run TZ=...`**. vLLM and LMCache log timestamps follow
**`TZ`** in the container. For **`docker exec`**, use **`CONTAINER_NAME`**
(default **`vllm-lmcache-hipfile-gpu0`**, i.e. **`IMAGE_NAME`** + **`gpu`** + **`GPU`**);
override with **`make run CONTAINER_NAME=...`**.

## Slurm (ROCm GPU cluster)

Submit from the **repository root** on a node with Docker and ROCm (default
**`gres=gpu:1`**, no GPU architecture constraint — see
**`.slurm/vllm-lmcache-hipfile.sbatch`**). The allocated GPU node must allow
**`docker`** for **`$USER`** (member of the **`docker`** group, or root). If
**`docker build`** fails with *permission denied* on **`/var/run/docker.sock`**
on compute nodes, build once on a node where Docker works (or via
**`srun --pty`** on a GPU host), then set **`VLH_SKIP_BUILD=1`** for later jobs
on that host. Narrow nodes with **`VLH_SLURM_CONSTRAINT`** (e.g.
**`MARKHAM&GFX942`** for MI300X, **`MARKHAM`** for any Markham ROCm GPU).

```bash
./run-slurm.sh
```

**`run-slurm.sh`** sets defaults (HF token file if present, Gutenberg path under
**`/scratch/$USER/vllm-lmcache-hipfile/gutenberg`** and Hub weights under
**`/scratch/$USER/vllm-lmcache-hipfile/hf/hub`**, **`VLH_LMCACHE_IO=posix`**, parallel
**`run-long-parallel.sh`** with **`4`** workers). You do not need to know NVMe
mount paths in advance: the job discovers storage on the allocated node (blank
**`nvme*n*`**, then mounted NVMe under **`/mnt`** / **`/local`** / similar, then
**`/scratch/$USER/vllm-lmcache-hipfile/lmcache-<jobid>`**, then **`/tmp`**). Override any
variable, then run again. Low-level submit: **`bash .slurm/run-vllm-lmcache-hipfile.sh`**.

If you have data from the old **`vllm-radeon`** layout, symlink or move
**`/scratch/$USER/vllm-radeon`** to **`/scratch/$USER/vllm-lmcache-hipfile`**
and update exports from **`RADEON_*`** to **`VLH_*`**.

**One-time** Gutenberg library on shared storage (same path every job):

```bash
mkdir -p "$VLH_GUTENBERG_DATA_ROOT"
make -C recipies/vllm-lmcache-hipfile data-all BOOK_DATA_ROOT="$VLH_GUTENBERG_DATA_ROOT"
# smoke: make -C recipies/vllm-lmcache-hipfile data BOOK_DATA_DIR=$VLH_GUTENBERG_DATA_ROOT/war-and-peace ...
```

The job **`docker build`**s the image, starts vLLM via **`make run-batch`**, then
runs host **`run-long-parallel.sh`** (default **`4`** workers, distinct seeds;
each worker calls **`run-long.sh`**). Set **`VLH_RUN_LONG_PARALLEL=0`** or
**`VLH_BENCHMARK=gutenberg_serial`** for a single **`run-long.sh`** stream.
Artifacts land under **`.slurm/logs/vllm-lmcache-hipfile-<jobid>/`**
(**`run-long-parallel/<timestamp>/worker-*.jsonl`**, server log, LMCache API
snapshots). Requires **`jq`** on compute nodes. LMCache **`long_doc_qa`** is
optional (**`VLH_BENCHMARK=long_doc_qa`**).

| Variable | Default | Purpose |
| --- | --- | --- |
| **`VLH_NVME_BASE`** | auto | Unset: discover on node (see below); else **`/tmp/...`** job dir |
| **`VLH_NVME_AUTO_USE`** | **`1`** | When **`VLH_NVME_BASE`** unset, probe the compute node at runtime |
| **`VLH_NVME_AUTO_DEVICE`** | **`1`** | Pick first spare **`nvme*n*`** (skips OS disk when **`nvme*n*p*`** is mounted) |
| **`VLH_NVME_SCRATCH_FALLBACK`** | **`1`** | Use **`/scratch/$USER/vllm-lmcache-hipfile/lmcache-<jobid>`** if no NVMe path |
| **`VLH_NVME_USE_SHARED_DATA_DOCKER`** | **`0`** | Allow LMCache under site **`/data`** / **`/docker`** (shared LVM; usually off) |
| **`VLH_NVME_MIN_AVAIL_GB`** | **`10`** | Minimum free space on a mounted NVMe path |
| **`VLH_NVME_MKFS`** | **`1`** if **`VLH_NVME_BASE`** unset, else **`0`** | **`mkfs.ext4`** blank **`nvme*n*`** only (destructive); needs **root** |
| **`VLH_NVME_MOUNT`** | **`/mnt/vllm-lmcache-hipfile-<jobid>`** | Mount point when mounting a blank **`nvme*n*`** |
| **`VLH_GUTENBERG_DATA_ROOT`** | **`VLH_SHARED_ROOT/gutenberg`** or recipe **`data/`** | Shared Gutenberg chunks + **`*.questions.json`** |
| **`VLH_SHARED_ROOT`** | **`/scratch/$USER/vllm-lmcache-hipfile`** (via **`run-slurm.sh`**) | Shared parent on scratch |
| **`VLH_HF_HOME`** | **`/scratch/$USER/vllm-lmcache-hipfile/hf`** | Golden HF Hub cache (always; never **`lmcache-<jobid>/hf`**) |
| **`VLLM_MODEL`** | **`model_default`** in yaml | Served model (vLLM + Gutenberg / **`long_doc_qa`**) |
| **`VLH_VLLM_READY_TIMEOUT`** | **`1800`** | Seconds to wait for **`/v1/models`** (raise for **`gpt-oss-120b`**) |
| **`VLH_LMCACHE_IO`** | **`hipfile`** | **`hipfile`** or **`posix`** disk backend |
| **`VLH_BENCHMARK`** | **`gutenberg`** | **`gutenberg`**, **`none`**, **`long_doc_qa`**, **`test_aic`** |
| **`VLH_RUN_LONG_PARALLEL`** | **`1`** | **`1`** = **`run-long-parallel.sh`**; **`0`** = serial **`run-long.sh`** |
| **`VLH_RUN_LONG_WORKERS`** | **`4`** | Parallel workers ( **`WORKERS`** ) |
| **`VLH_RUN_LONG_ITERATIONS`** | **`1`** | Per-worker **`ITERATIONS`** (total ≈ workers × iterations) |
| **`VLH_RUN_LONG_BASE_SEED`** | **`$RANDOM`** | Worker **`i`** uses seed **`BASE_SEED + i`** |
| **`VLH_RUN_LONG_STAGGER_SEC`** | **`0`** | Delay between starting workers |
| **`BOOK_SLUG`** / **`BOOK_SLUGS`** | (library mode) | Single book or subset; see **`run-long.sh`** |
| **`VLH_SLURM_CONSTRAINT`** | (none) | Slurm **`--constraint`** (e.g. **`MARKHAM`**, **`MARKHAM&GFX942`**) |
| **`VLH_SLURM_EXCLUDE`** | (none) | Slurm **`--exclude`** comma-separated node names |
| **`VLH_SLURM_MEM`** | **`64G`** (via sbatch) | Override with **`128G`** on large-memory nodes |
| **`ROCM_ARCH`** | auto on node | Force image build (**`gfx942`**, **`gfx90a`**, **`gfx1201`**, …) |
| **`VLH_SKIP_BUILD`** | **`0`** | **`1`** skips **`make build`** if image exists |
| **`VLH_NVME_BLK_BPFTRACE`** | **`1`** (via wrapper) | NVMe block I/O bpftrace tab-separated trace |
| **`VLH_NVME_SMART_LOG`** | **`1`** (via wrapper) | **`nvme smart-log`** at job start/end |
| **`VLH_LMCACHE_ENABLE_KV_EVENTS`** | **`1`** | LMCache KV events in **`server.txt`** |

Build + server only (no Gutenberg load):

```bash
sbatch --export=ALL,VLH_BENCHMARK=none,VLH_NVME_BASE=/mnt/nvme \
  .slurm/vllm-lmcache-hipfile.sbatch
```

Monitor: **`tail -f .slurm/logs/vllm-lmcache-hipfile-<jobid>.log`**. After the job,
read **`.slurm/logs/vllm-lmcache-hipfile-<jobid>/results-summary.md`** (per-worker TTFT,
prefill/decode tok/s, LMCache store totals, bpftrace NVMe/VFS I/O, NVMe SMART
delta, engine stats from **`server.txt`**, exit codes). Machine-readable:
**`results-summary.json`**. Re-run post-processing:
**`python3 .slurm/lib/summarize-recipe-job.py .slurm/logs/vllm-lmcache-hipfile-<jobid>`**.
**`.slurm/run-slurm.sh`** delegates to the top-level **`run-slurm.sh`**.

## **`rocm-aic-exporter.py`** (Prometheus textfile)

Standalone exporter for LMCache / vLLM host metrics. Today it reports:

1. **KV inventory** — ``.data`` file count and total bytes per **`model_name`**
2. **Filesystem** — total, used, and free bytes on the mount hosting **`$DATA`**
3. **Hit histogram** — from **`$DATA/lmcache_chunk_stats/chunk_hashes_*.jsonl`**
   over **current** ``$DATA/lmcache/*.data`` only (deleted chunks excluded).
   JSONL hashes must use the same **`pre_caching_hash_algorithm`** as KV storage
   (**`lmcache-chunk-statistics-hash.patch`**). Stats collected with the default
   **builtin** hasher cannot be matched to **`sha256_cbor`** on-disk keys.
4. **NFS** (only with **`--prometheus-textfile`**) — runs **`nfsiostat`** when
   installed; cumulative RX/TX bytes per NFS client mount from
   **`/proc/self/mountstats`** (label **`mount_point`**).
   **`rocm_aic_nfsiostat_present`** is 0 when **`nfsiostat`** is absent.
5. **ROCm** (only with **`--prometheus-textfile`**) — **`hipconfig`** for
   HIP/ROCm version; **`rocm_aic_hipconfig_present`** is 0 when
   **`hipconfig`** is absent.

```bash
# Host path matches make run DATA= (default /mnt/lmcache-nvme)
python3 ../common/scripts/rocm-aic-exporter.py
python3 ../common/scripts/rocm-aic-exporter.py --top 20 --json

# node_exporter textfile collector → Grafana (job=node_exporter)
python3 ../common/scripts/rocm-aic-exporter.py --prometheus-textfile
python3 ../common/scripts/rocm-aic-exporter.py \
  --prometheus-textfile /var/lib/prometheus/node-exporter/rocm_aic_exporter.prom \
  --textfile-only

# Example Grafana bar chart:
#   rocm_aic_kv_files_by_hit_count{hit_count=~".+"}
```

Metrics use the **`rocm_aic_*`** prefix. Default textfile path:
**`/var/lib/prometheus/node-exporter/rocm_aic_exporter.prom`**, or set
**`ROCM_AIC_EXPORTER_TEXTFILE`** / **`ROCM_ICMS_TEXTFILE_DIR`** (legacy:
**`VLH_LMCACHE_CHUNK_HIST_TEXTFILE`**).

Example Grafana queries (``job="node_exporter"``):

- ``rocm_aic_kv_files{model_name=~".+"}``
- ``rocm_aic_kv_chunk_bytes_total``
- ``rocm_aic_data_fs_free_bytes``
- ``rate(rocm_aic_nfs_mount_rx_bytes_total{mount_point="/mnt/..."}[5m])``
- ``rocm_aic_rocm_version_info``

## MFU metrics (**`--enable-mfu-metrics`**)

**`scripts/vllm-server`** passes **`--enable-mfu-metrics`** by default
(**`serve.enable_mfu_metrics: true`** in **`configs/vllm-lmcache-hipfile.yaml`**).
vLLM exposes analytic FLOPs / MFU on Prometheus (e.g.
**`vllm:estimated_flops_per_gpu_total`**). Disable without editing YAML:

```bash
make run VLH_ENABLE_MFU_METRICS=0
```

Set **`VLLM_DEBUG_MFU_METRICS=1`** in the container for extra MFU debug
logging. Restart vLLM after changing MFU-related env vars.

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

## LMCache disk mode (**`VLH_LMCACHE_IO`**)

**`make run`** passes **`VLH_LMCACHE_IO`** (default **`hipfile`**). **`hipfile`**
uses LMCache **GdsBackend** + hipFile (**`gds_path`** under **`DATA`/`subdir`**).
**`posix`** uses LMCache **`remote_storage_plugins: [fs]`** (POSIX filesystem
backend): **`extra_config.remote_storage_plugin.fs.base_path`** points at the
same directory as **`hipfile`** (**`DATA`/`subdir`**). No **`gds_path`** key
in the runtime YAML (normal path; not **`fs://`**).

```bash
make run VLH_LMCACHE_IO=posix
make run VLH_LMCACHE_IO=hipfile   # default
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

Startup mode still comes from **`VLH_LMCACHE_IO`** at **`make run`**; use
**`/storage/mode`** only when you need to flip layouts on a live server.

## LMCache logging and hipFile buffer

The image applies **`patches/lmcache-gds-eviction-log.patch`**,
**`lmcache-sha256-cbor-int.patch`**, and **`lmcache-controller-log.patch`**
(see **`patches/README.md`**). Rebuild after changing patches:

```bash
make build
```

**Log level:** **`scripts/vllm-server`** sets **`LMCACHE_LOG_LEVEL`** from
**`VLH_LMCACHE_LOG_LEVEL`** (default **`INFO`**). Use **`ERROR`** only if
you need to hide all LMCache warnings (including real allocation failures):

```bash
make run VLH_LMCACHE_LOG_LEVEL=ERROR
```

**hipFile pool (MiB):** **`configs/lmcache-hipfile.yml`** defaults
**`gds_buffer_size: 1024`**. Override at run time without editing YAML:

```bash
make run VLH_LMCACHE_GDS_BUFFER_SIZE=2048
```

Raise the pool if **`logs/server.txt`** shows
**`Failed to allocate memory block of size 9437184`** during parallel long-
context retrieve (~9 MiB per chunk; many concurrent loads can exhaust a 512
MiB pool).

**Validate warnings** after rebuild and a short workload:

```bash
grep -c 'LMCache WARNING' logs/server.txt
grep 'LMCache WARNING' logs/server.txt | sed 's/.*LMCache WARNING://' \
  | sort | uniq -c | sort -rn
```

Expect no repeated **GDS Backend does not support eviction** lines at INFO;
**builtin** hash warnings should be gone (**`pre_caching_hash_algorithm:
sha256_cbor`**).

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
docker exec -it vllm-lmcache-hipfile-gpu0 python3 \
  /app/LMCache/benchmarks/long_doc_qa/long_doc_qa.py --help
docker exec -it vllm-lmcache-hipfile-gpu0 python3 \
  /app/LMCache/benchmarks/long_doc_qa/long_doc_qa.py \
  --port 8000 --model openai/gpt-oss-120b \
  --num-documents 2 --hit-miss-ratio 1:1
```

Use the same name as **`make run`** (**`CONTAINER_NAME`**, default **`vllm-lmcache-hipfile-gpu0`** if **`IMAGE_NAME`** and **`GPU`** match defaults).

Set **`VLLM_MODEL`** for Slurm and **`make run`** (same name for server and
benchmarks). Unset uses **`vllm-lmcache-hipfile.yaml`** **`model_default`**. Legacy
**`VLH_BENCH_MODEL`** is copied to **`VLLM_MODEL`** with a warning.
Match **`--port`** to **`800{GPU}`** from **`ROCR_VISIBLE_DEVICES`**.

## Gutenberg long-context fixtures

Gutenberg data prep and **`run-long*.sh`** live in
[benchmarks/llm-prefill-benchmark](../../benchmarks/llm-prefill-benchmark/).
This recipe's **`make data`** / **`make data-all`** delegate there; Slurm sets
**`LLM_PREFILL_BENCH_ROOT`** automatically.

**`data/`** is **not** tracked (see **`.gitignore`**); generate fixtures locally.
From **`recipies/vllm-lmcache-hipfile/`**:

```bash
make data
```

Defaults: **War and Peace** (**`BOOK_PG_ID=2600`**, **`BOOK_SLUG=war-and-peace`**).
Override book, chunk size, or count, e.g.:

```bash
make data BOOK_SLUG=pride-and-prejudice BOOK_PG_ID=1342 \
  BOOK_TITLE='Pride and Prejudice' BOOK_AUTHOR='Jane Austen'
```

Equivalent manual steps (from repo root or **`benchmarks/llm-prefill-benchmark/`**):

```bash
python3 benchmarks/llm-prefill-benchmark/scripts/split-gutenberg-random-chunks.py \
  --pg-id 2600 --slug war-and-peace \
  -o data/war-and-peace --count 100

python3 benchmarks/llm-prefill-benchmark/scripts/gen-questions-json.py \
  --slug war-and-peace --title "War and Peace" --author "Leo Tolstoy" \
  --pg-id 2600

# Optional: supply your own question list (.json array or one question per line):
python3 benchmarks/llm-prefill-benchmark/scripts/gen-questions-json.py \
  --slug war-and-peace --title "War and Peace" --author "Leo Tolstoy" \
  --extra-questions /path/to/my-questions.json

# Load test (run after data/<slug>/ exists):
BOOK_SLUG=war-and-peace benchmarks/llm-prefill-benchmark/run-long.sh

# LMCache populate / cold / warm A/B (host: pip install 'openai>=1.40.0' or full requirements.txt):
python3 benchmarks/llm-prefill-benchmark/scripts/test-aic.py -o logs/test-aic.json
# Same chunk + cache_salt; reset_prefix_cache before cold/warm; cold bypasses GDS.
# Fresh NVMe store: new --run-id (or --skip-populate if already stored).
```

Chunk files are **`data/<slug>/<slug>-<chunk-label>.<offset>.txt`** (default
label **`10k`** for 10 000 words). Each split run also writes
**`data/<slug>/<slug>.book-stats.json`** with **`book_word_count`** (full
text after Gutenberg boilerplate removal, whitespace-split). **`gen-questions-json.py`**
copies those fields into **`<slug>.questions.json`**. Backfill stats without
rewriting chunks:

```bash
python3 benchmarks/llm-prefill-benchmark/scripts/split-gutenberg-random-chunks.py --pg-id 2600 \
  --slug war-and-peace -o data/war-and-peace --stats-only
python3 benchmarks/llm-prefill-benchmark/scripts/gen-questions-json.py --slug war-and-peace \
  --title "War and Peace" --author "Leo Tolstoy" --pg-id 2600
```

**`run-long.sh`** (under **`benchmarks/llm-prefill-benchmark/`**) honors **`BOOK_SLUG`**, **`BOOK_DATA_DIR`**, and
**`QUESTIONS_FILE`**.

### 100-book library

A curated manifest lives at
**`benchmarks/llm-prefill-benchmark/configs/gutenberg-library.json`** (100 English novels: PG id, slug,
title, author). Build every book locally (network required; **`data/`** stays
gitignored):

```bash
make data-all
```

Smoke-test the first three books:

```bash
make data-all DATA_ALL_LIMIT=3
```

Re-runs skip complete directories by default
(**`DATA_ALL_SKIP_EXISTING=0`** forces a rebuild). Expect on the order of
**500 MB–1 GB+** under **`data/`** for the full library (100 books × 100
10 k-word chunks plus questions JSON per book).

**`run-long.sh`** library mode (default when **`BOOK_SLUG`** is unset) picks a
random book, chunk, and question on each iteration from every complete
**`data/<slug>/`** directory:

```bash
make -C benchmarks/llm-prefill-benchmark run
make -C benchmarks/llm-prefill-benchmark run ITERATIONS=20
```

Single-book mode is unchanged:

```bash
BOOK_SLUG=war-and-peace make -C benchmarks/llm-prefill-benchmark run
```

Limit library mode to a subset of books with a comma-separated list or a file
(one slug per line; **`#`** starts a comment):

```bash
BOOK_SLUGS=war-and-peace,pride-and-prejudice make -C benchmarks/llm-prefill-benchmark run
BOOK_SLUG_FILE=configs/my-slugs.txt make -C benchmarks/llm-prefill-benchmark run-parallel WORKERS=4
```

**`BOOK_SLUGS`** and **`BOOK_SLUG_FILE`** may be combined (union). Do not set
**`BOOK_SLUG`** to a single slug when using **`BOOK_SLUGS`** / **`BOOK_SLUG_FILE`**.

Optional env: **`BOOK_DATA_ROOT`** (default **`data/`**), **`CONTEXT_FILE`**,
**`QUESTION`**, **`RUN_LONG_SEED`** (deterministic **`$RANDOM`** sequence),
**`RUN_LONG_COMBINE_CHUNKS`** (default **`1`**; set **`2`** to concatenate two
random 10 k-word chunks into ~20 k words without new fixture files).

```bash
RUN_LONG_COMBINE_CHUNKS=2 BOOK_SLUG=war-and-peace make -C benchmarks/llm-prefill-benchmark run
RUN_LONG_COMBINE_CHUNKS=2 make -C benchmarks/llm-prefill-benchmark run-parallel WORKERS=4
```

### Parallel load (**`run-long-parallel.sh`**)

Run **`N`** workers in parallel; worker **`i`** uses **`RUN_LONG_SEED =
BASE_SEED + i`** so each picks a different book/chunk/question stream:

```bash
make -C benchmarks/llm-prefill-benchmark run-parallel WORKERS=8
WORKERS=8 ITERATIONS=50 BASE_SEED=42 make -C benchmarks/llm-prefill-benchmark run-parallel
```

Per-worker JSON lines go under **`logs/run-long-parallel/<timestamp>/`**
(**`worker-<n>.jsonl`**, **`worker-<n>.log`**). Response JSON includes
**`run_long_worker`** and **`run_long_seed`**. Optional **`STAGGER_SEC`** delays
worker starts.

When stderr is a TTY, an iteration progress bar runs automatically
(**`WORKERS * ITERATIONS`** completions, polled from **`worker-*.jsonl`**).
Disable with **`PROGRESS=0`**; force on in CI with **`PROGRESS=1`**.

## GitHub Actions CI

Path-filtered workflows under [`.github/workflows/vllm-lmcache-hipfile-*.yml`][gh-vr]
cover patches, Python scripts, shell fixtures, config lint, and an optional
manual Docker build. PRs do not run a full image build (runner disk); use
**`vllm-lmcache-hipfile-docker`** via **workflow_dispatch** when you need an end-to-end
compile. See also [rocm-aic CI][root-ci] in the root README.

## Grafana dashboard

Cluster dashboard: [`grafana/rocm-aic-dashboard.json`](../../grafana/rocm-aic-dashboard.json).
Import into your Grafana server and adjust variables for your Prometheus
labels and mount paths. See [`grafana/README.md`](../../grafana/README.md).

<!-- References -->

[gh-vr]: ../../.github/workflows/
[root-ci]: ../../README.md#ci
[lmcache-pr-3008]: https://github.com/LMCache/LMCache/pull/3008

