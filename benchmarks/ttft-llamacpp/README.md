# llama.cpp TTFT Benchmark

Measure **Time-To-First-Token (TTFT)** on AMD ROCm GPUs using
[llama.cpp][llamacpp]'s built-in slot save/restore API.  Compares
cold prefill (no cache) against warm restore (slot loaded from
disk or RAM-backed tmpfs).

This benchmark works on both AMD Instinct (CDNA) and Radeon
(RDNA) GPUs, since llama.cpp supports both via HIP.

## How it works

1. **Cold run** -- start llama-server, send a long prompt, measure
   TTFT from scratch, then save the slot to disk via
   `POST /slots/0?action=save`.
2. **Warm run** -- restart llama-server (cold GPU), restore the
   slot via `POST /slots/0?action=restore`, send the
   **identical** prompt, measure TTFT.

The storage tier for saved slots is controlled by the Docker
mount at `/slots`:

| Tier | Docker flag | Latency |
|------|-------------|---------|
| CPU RAM | `--tmpfs /slots:rw,size=4g` | Sub-ms |
| Local disk | `-v /path:/slots` | ~10-50 ms |
| NVMe | `-v /mnt/nvme:/slots` | ~10-20 ms |
| NFS | `-v /nfs/share:/slots` | Network-dependent |

## Prerequisites

* Docker with ROCm GPU pass-through (`/dev/kfd`, `/dev/dri`)
* A GGUF model file (e.g., from HuggingFace)
* Sufficient VRAM for the model + context

## Quick start

### 1. Build the Docker image

```bash
./scripts/docker-build.sh
```

To include the experimental `--cache-disk` patch:

```bash
APPLY_CACHE_DISK_PATCH=1 ./scripts/docker-build.sh
```

### 2. Launch the container

```bash
MODEL_DIR=$HOME/models ./scripts/docker-run.sh
```

For RAM-speed slot storage (tmpfs):

```bash
MODEL_DIR=$HOME/models SLOT_TMPFS=1 ./scripts/docker-run.sh
```

### 3. Run the benchmark

Inside the container:

```bash
MODEL=/models/qwen3-8b-q4_k_m.gguf ./scripts/run-bench.sh
```

Override defaults via env vars:

```bash
MODEL=/models/llama-3.1-8b-q4.gguf \
CONTEXT_CHARS=60000 \
REPEATS=5 \
SEED=123 \
    ./scripts/run-bench.sh
```

### 4. Read results

A summary table is printed at the end:

```
=== TTFT Summary ===
  Cold (no cache):     n=3  mean=6200.3ms  min=6100.1ms  max=6300.5ms
  Warm (slot restore): n=3  mean=148.2ms   min=142.0ms   max=155.1ms
  Speedup:             41.8x  (6052ms saved)

Full results: /app/results.jsonl
```

## cache-disk patch

The `patches/0001-cache-disk.patch` adds two new CLI flags
to llama-server:

- `--cache-disk <path>` -- directory for disk-backed prompt
  cache; evicted RAM entries are written here instead of
  being destroyed
- `--cache-disk-max <MiB>` -- maximum disk cache size
  (default: 0 = unlimited)

### What it does

When the RAM prompt cache (`--cache-ram`) is full and must
evict an entry, instead of destroying the KV state, the
patch serialises it to a file in the `--cache-disk`
directory.  On a subsequent prefix match, the state is
read back from disk and restored.

This is useful on:

- **UMA systems** (AMD Strix Halo, Apple Silicon) where RAM
  is VRAM and prompt caching competes with model weights
- **Multi-user servers** where many distinct conversations
  need cached context
- **Large context windows** (100k+ tokens) where KV state
  is hundreds of MiB per slot

### Serialisation format

Each disk file uses a simple binary layout:

```
uint32  magic    = 0x4C4C4344  ("LLCD")
uint32  version  = 1
uint64  n_tokens
int32[] tokens[n_tokens]
uint64  data_size
uint8[] data[data_size]
uint32  n_checkpoints
Per checkpoint:
  int32   pos_min
  int32   pos_max
  int64   n_tokens
  uint64  ckpt_data_size
  uint8[] ckpt_data[ckpt_data_size]
```

### Files modified

- `common/common.h` -- `cache_disk`, `cache_disk_max_mib`
- `common/arg.cpp` -- `--cache-disk`, `--cache-disk-max`
- `tools/server/server-task.h` -- disk-tier fields/methods
- `tools/server/server-task.cpp` -- offload/reload/eviction
- `tools/server/server-context.cpp` -- initialisation

### Building with the patch

```bash
APPLY_CACHE_DISK_PATCH=1 ./scripts/docker-build.sh
```

Then use it inside the container:

```bash
llama-server \
    --model /models/my-model.gguf \
    --cache-ram 4096 \
    --cache-disk /slots/disk-cache \
    --cache-disk-max 8192 \
    --slot-save-path /slots
```

## Reproducibility

| What | Controlled by |
|------|---------------|
| Corpus excerpt offset | `bench_ttft.py --seed` |
| Server hashing | Same prompt each run |
| Slot save/restore | Deterministic binary state |

Two runs with the same `SEED` and `MODEL` produce identical
prompts and slot saves.

## Directory layout

```
Dockerfile                    ROCm + llama.cpp (HIP) build
README.md                     This file
requirements.txt              Python deps (openai)
bench_ttft.py                 TTFT measurement script
corpus.txt                    (generated, gitignored)
scripts/
  docker-build.sh             Build the image
  docker-run.sh               Launch with ROCm flags
  fetch-corpus.sh             Download Gutenberg corpus
  serve.sh                    Start llama-server
  run-bench.sh                Cold/warm orchestrator
patches/
  0001-cache-disk.patch       --cache-disk feature patch
```

<!-- References -->

[llamacpp]: https://github.com/ggml-org/llama.cpp
