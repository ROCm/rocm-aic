<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# vllm-optimus-kvd

ROCm **vLLM** + **AMD Optimus**. Base **`vllm/vllm-openai-rocm:v0.21.0`**, with
hipFile from **ROCm/rocm-systems**, fio with libhipfile.

## Contents

- [Where things live](#where-things-live)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Cliff sweep](#cliff-sweep)
- [Tuning knobs](#tuning-knobs)
- [Correctness check](#correctness-check)
- [Known issues / follow-ups](#known-issues--follow-ups)

## Where things live

| What you need | File |
| --- | --- |
| `make build` / `make run` / `make cliff` / `make correctness`, `ROCM_ARCH`, `CONTAINER_NAME`, mounts (`DATA`, `LOG`), `HF_TOKEN`, knobs (`VLLM_MODEL`, `TENSOR_PARALLEL_SIZE`, `KV_CACHE_MEMORY_BYTES`, all `ROCSERVE_KVD_*`) | `Makefile` (`make help`) |
| Image build (vLLM base → hipFile → fio → async patch → Optimus clone) | `Dockerfile` |
| Container entrypoint: start `python3 -m rocserve.kvd` daemon, then `exec vllm serve --kv-transfer-config '{...RocserveKvdConnector...}'` | `scripts/vllm-server` |
| Reference defaults (also documented inline in Makefile + scripts) | `configs/kvd-defaults.yaml` |
| Cliff sweep | `benchmarks/kv-cache-cliff/run_cliff.py` (repo root; mounted into the container at `/app/cliff`) |
| Standalone smoke cliff (against running container) | `run-this.sh` |

## Quick start

```bash
# 1. Build (one-time; pulls rocm-systems hipFile, fio).
export ROCM_ARCH=gfx942        # MI300X / MI325X family
make build

# 2. Provision an NVMe mount for the kvd SSD tiers (host path).
sudo mkdir -p /mnt/optimus-nvme && sudo chown $USER /mnt/optimus-nvme
export DATA=/mnt/optimus-nvme

# 3. Start the engine.
export HF_TOKEN=hf_...                    # or HF_TOKEN_FILE=/path
make run                                  # foreground; Ctrl-C to stop
# or: make run-batch                      # background (-d)

# 4. In another terminal: cliff sweep
make cliff                                # writes CSV under LOG dir
```

`make build` runs from `recipies/vllm-optimus-ais/`. The Dockerfile
`COPY`s from the rocm-aic repo root, so the Makefile passes
`$(REPO_ROOT)` as build context automatically.

## Architecture

```
container vllm-optimus-kvd-gpu0
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   ┌────────────────┐   Unix Domain Socket   ┌──────────────────────┐ │
│   │  vllm serve    │◄─────────────────────► │  rocserve.kvd daemon │ │
│   │  + Rocserve    │                        │  RAM hash → file idx │ │
│   │    KvdConnector│                        └──────────────────────┘ │
│   │  (Python)      │                                                 │
│   └────────┬───────┘                                                 │
│            │  hipFileReadAsync (P2PDMA → device VRAM)                │
│            ▼                                                         │
│   ┌───────────────────────────────────────────────────────┐          │
│   │  Content-keyed file tier (bind-mounted DATA=/data)    │          │
│   │   {root}/{h[:2]}/{h[2:4]}/<urlencoded(...)>.kvcache   │          │
│   │   path = sha256(model|compat|key) → 2-level shard     │          │
│   └───────────────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────────────┘
```

vLLM's L1 (GPU prefix cache) stays the hot tier. When the working set
exceeds VRAM budget, evicted prefixes spill to the kvd file tier; new
admissions probe the file tier via local `os.stat` (no daemon round-trip
on the data path) and stream chunks back via hipFile P2PDMA.

## Cliff sweep

`make cliff` (or `benchmarks/kv-cache-cliff/run_cliff.py` directly) sweeps concurrency at
a long deterministic per-client prefix (default 60k tokens, suffix=0,
greedy decoding). For each `c`:

1. `--warmup-iters` cold iters populate the file tier (not measured).
2. `--iters` measured iters report per-iter wall, throughput, p50/p95
   request latency, and L1 / external prefix-cache hit-rate **deltas**
   computed from `/metrics`.

Read the resulting CSV (`/var/log/vllm-optimus-kvd/cliff-*.csv`) for:

- **`tput_total`** — total tokens/s (prompt + completion). Rises with
  `c` until the working set spills L1; that's where vanilla cliffs and
  this recipe should hold (kvd serves evicted prefixes from the file
  tier via hipFile direct).
- **`ext_hit_pct`** — external (kvd file tier) hit rate as % of total
  prefix-cache queries. ≈0 % when L1 covers; rises sharply once
  eviction starts.
- **`miss_pct > 0`** — KV restoration failure: vLLM expected
  external/L1 to cover those tokens but neither did, so it re-prefills.
  Investigate with `tests/correctness_check.py`.
- **`p50_s` / `p95_s`** — per-request wall (HTTP request to last
  token). The cliff shows up as a step jump in p50.

The cliff point depends on model size, TP, and `KV_CACHE_MEMORY_BYTES`
— size your sweep to cross it.

## Tuning knobs

All set via Makefile vars (`make run KEY=VAL`) or env on `make run`:

| Var | Default | What it does |
| --- | --- | --- |
| `VLLM_MODEL` | `openai/gpt-oss-120b` | served model |
| `TENSOR_PARALLEL_SIZE` | 1 | TP world size (must match `--device` count) |
| `KV_CACHE_MEMORY_BYTES` | 80 GiB | L1 cap; size by card |
| `MAX_MODEL_LEN` | 131072 | vLLM `--max-model-len` |
| `DATA` | `/mnt/optimus-nvme` | host NVMe for kvd file tier |
| `ROCSERVE_KVD_LONG_BYTES` | 512 GiB | persistent SSD tier size budget |
| `ROCSERVE_KVD_SHORT_BYTES` | 64 GiB | short SSD tier (L1-evict spillover) |
| `ROCSERVE_KVD_GPU_DIRECT` | `auto` | `0`/`1`/`auto` (ais-check P2PDMA) |

## Correctness check

After `make run` starts the kvd arm, point a **vanilla** vLLM (no
`--kv-transfer-config`, same model + same TP) at any other port, then:

```bash
make correctness VANILLA_ENDPOINT=http://other-host:8000
```

This sends identical prompts (greedy, temperature=0) to both arms over
multiple iters. Any token-stream divergence = KV restoration bug in
the chunked-fusion / hipFile load path. The check exits non-zero on
divergence so it composes with CI.

This catches the class of bugs `Optimus@fe056f8` fixed (coverage
arithmetic / staging buffer race / fp8 dtype name mismatch) and the
file-tier durability bugs (`Optimus@b35a6f3`: tmp filename collision /
unverified file size on AIS load / no fsync on publish).

