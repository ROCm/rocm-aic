<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# TTFT LMCache Benchmark

Measure **Time-To-First-Token (TTFT)** on AMD ROCm GPUs as a
function of KV-cache hit rate. The benchmark uses [vLLM][vllm]
for inference and [LMCache][lmcache] with cache blending to
serve KV chunks from an offloaded cache backend.

## How it works

```
Phase 1 -- Warming
  Start vLLM + LMCache  -->  send long prompt  -->  all KV
  chunks written to disk as .pt files  -->  stop vLLM  -->
  snapshot the cache directory.

Phase 2 -- Sweep  (for each hit rate N%)
  Restore snapshot  -->  randomly delete (100-N)% of .pt
  chunk files (seeded PRNG)  -->  restart vLLM  -->  send
  the IDENTICAL prompt  -->  measure TTFT  -->  stop vLLM.
```

Cache **blending** (`enable_blending: true`) lets LMCache load
non-contiguous surviving chunks.  Without blending, only the
longest contiguous prefix of cached chunks would be usable,
making random deletion equivalent to a simple prefix
truncation.

Every source of randomness is governed by a single `SEED`
value (default `42`).  The seed controls corpus excerpt
selection, chunk deletion patterns, and `PYTHONHASHSEED` for
vLLM/LMCache.  Deleted filenames are recorded in the results
JSON so any run can be replayed exactly.

## Prerequisites

* Docker with GPU pass-through (ROCm `/dev/kfd`, `/dev/dri`)
* Host tools: optional `pip install -r requirements.txt` from the [repo root][root-readme]
  (full stack). The Docker image uses **local** `requirements.txt` here
  (`openai`, `transformers` only).
* `/dev/infiniband` access (only for AIS backend)
* An NVMe or disk mount for cache storage
* A HuggingFace token if using a gated model

## Quick start

### 1. Build the Docker image

```bash
./scripts/docker-build.sh
```

To skip hipFile / AIS support (faster build):

```bash
WITH_HIPFILE=0 ./scripts/docker-build.sh
```

### 2. Launch the container

```bash
HF_TOKEN="hf_..." CACHE_MOUNT=/mnt/nvme/bench \
    ./scripts/docker-run.sh
```

### 3. Run the sweep

Inside the container:

```bash
./scripts/run-sweep.sh
```

Override defaults via env vars:

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct \
CONTEXT_TOKENS=15000 \
HIT_RATES="0 10 25 50 75 90 100" \
REPEATS=5 \
SEED=123 \
    ./scripts/run-sweep.sh
```

### Runtime YAML

For repeated runs, put the common settings in YAML instead of exporting each
variable. Copy the shared example, edit the `ttft_lmcache` section, then run
the sweep normally:

```bash
cp ../runtime.yaml.example ../runtime.yaml
./scripts/run-sweep.sh
```

Use `RUNTIME_CONFIG_FILE=/path/to/runtime.yaml` to select another file.
Environment variables still override YAML values for one-off changes.

### 4. Read results

Results are written as JSON-lines to `/app/results.jsonl`
inside the container.  A summary table is printed at the end
of the sweep:

```
tag                  count    mean_ms      min_ms      max_ms
----------------------------------------------------------------
hit-0-rep-1              1     6200.3      6200.3      6200.3
hit-0-rep-2              1     6180.1      6180.1      6180.1
hit-50-rep-1             1     3100.5      3100.5      3100.5
hit-100-rep-1            1      148.2       148.2       148.2
warmup                   1     6314.0      6314.0      6314.0
```

## Cache backends

Five LMCache configurations are provided under `configs/`.
Select one by setting `LMCACHE_CONFIG_FILE` before running the
sweep:

| Config | Backend | Mount | Notes |
|--------|---------|-------|-------|
| `lmcache-cpu.yaml` | Host RAM | -- | No persistence; cold/warm only |
| `lmcache-disk.yaml` | Local disk | `-v /path:/cache` | ext4, xfs, tmpfs |
| `lmcache-nvme.yaml` | NVMe (O\_DIRECT) | `-v /path:/cache` | Bypasses page cache |
| `lmcache-ais.yaml` | hipFile / AIS | `-v /path:/data` | GPU Direct Storage |
| `lmcache-nfs.yaml` | NFS / remote FS | `-v /path:/cache` | Network filesystem |

Example with NVMe backend:

```bash
LMCACHE_CONFIG_FILE=/app/configs/lmcache-nvme.yaml \
    ./scripts/run-sweep.sh
```

## Reproducibility

| What | Controlled by |
|------|---------------|
| Corpus excerpt offset | `bench_ttft.py --seed` |
| Chunk deletion pattern | `random.Random(SEED + N + rep)` |
| vLLM / LMCache hashing | `PYTHONHASHSEED=$SEED` |
| Deletion manifest | Logged to `results.jsonl` |

Two runs with the same `SEED`, `HIT_RATES`, and `REPEATS`
produce identical deletion patterns.  Changing the seed
explores different random layouts of cache holes.

## Directory layout

```
Dockerfile                    ROCm + vLLM + LMCache (+hipFile)
README.md                     This file
requirements.txt              Image-only deps (openai, transformers)
bench_ttft.py                 TTFT measurement script (host + container)
scripts/
  docker-build.sh             Build the image
  docker-run.sh               Launch container with ROCm flags
  fetch-corpus.sh             Download Gutenberg corpus
  serve.sh                    Start vLLM + LMCache
  run-sweep.sh                Full sweep orchestrator
configs/
  lmcache-cpu.yaml            CPU RAM backend
  lmcache-disk.yaml           Local disk backend
  lmcache-nvme.yaml           NVMe O_DIRECT backend
  lmcache-ais.yaml            hipFile / AIS backend
  lmcache-nfs.yaml            NFS / remote FS backend
```

<!-- References -->

[root-readme]: ../../README.md
[vllm]: https://github.com/vllm-project/vllm
[lmcache]: https://github.com/LMCache/LMCache
