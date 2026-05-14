# LMCache Simulation Tool

A Python CLI tool for running LMCache cache operations in-process, configuring
storage backends, and generating workload traffic to test cache performance.

The tool creates an `LMCacheEngine` directly in the simulator process and
calls `engine.store()` and `engine.retrieve()` with token ID lists. When you
pass `--hf-model-name` or `--model-path` (and optional
`--tokenizer-mode text-to-tokens`), those IDs come from the model tokenizer;
otherwise workloads use synthetic ranges derived from operation keys. It
reports latency, IO bytes, cache outcomes, and optional per-chunk hit
histograms.

## Installation

Install the required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running the Tool

Run the tool as a Python module from this directory (`tools/lmcache-io-tester`):

```bash
python -m src.lmcache-sim --help
```

### Retrieve-only (warmed cache)

After store/lookup has populated the sidecar under your `--storage-path`,
`run-retrieve-only.sh` runs **retrieve-only** on the NFS-backed data dir for
**120 seconds** (override with `RETRIEVE_DURATION_SEC`). It does not pass
`--fs-odirect` (page cache path). From this directory:

```bash
./run-retrieve-only.sh
```

## Defaults when no Hugging Face model is passed

If you omit both `--hf-model-name` and `--model-path`, no Transformers model is
loaded. The engine still starts using CLI defaults:

| Setting | Default | Meaning |
|---------|---------|---------|
| `--model-name` | `lmcache_model` | LMCache model tag in cache paths |
| `--kv-shape` | `2,2,256,4,16` | KV tensor shape string passed to LMCache |
| `--kv-dtype` | `float16` | KV element type |
| `--chunk-size` | `256` | KV config and token rows per op in workloads |
| `--tokenizer-mode` | `vocab-only` | No HF tokenizer unless a model is loaded |
| `--device` | `cpu` | Connector device (`cpu`, `cuda`, `xpu`) |

Pass `--hf-model-name …` and `--auto-kv-shape` (and optionally
`--tokenizer-mode text-to-tokens`) when you want KV layout and tokens to
match a real checkpoint.

## Project Layout

```
src/                  Core simulator modules
data/                 Conversation schemas, sample data
tests/                Test scripts
scripts/              Optional helpers (e.g. RDMA throughput plot from bench logs)
configs/              Generated YAML configs (runtime)
```

### RDMA throughput plot (1 s, SI GB/s)

After an NFS/RDMA bench, `rdma-statistic.sample.log` contains timestamped
`rdma statistic show` blocks. To chart **RX and TX throughput in SI GB/s**
(1e9 bytes per second) with **1 second** resolution (linear interpolation
between samples, then per-second deltas):

```bash
./scripts/plot_rdma_throughput_1s.py \
  -i /path/to/rdma-statistic.sample.log \
  --iface rocep159s0 \
  -o /tmp/rdma-throughput-1s-gbs.svg
```

Open the SVG in a browser or an image viewer. Use `--iface` to match the
`link …/1` line for the NIC you care about.

## Architecture

The simulator creates the LMCache engine in-process and calls its methods
directly:

```
CLI (lmcache-sim run)
  |
  +-> ModelLoader (optional)
  |     -> KV shape, dtype from HF model config
  |     -> TokenizerWrapper for text-to-tokens
  |
  +-> ConfigGenerator -> YAML config
  +-> StorageManager  -> validate/mount storage
  |
  +-> EngineManager.create_engine()
  |     -> LMCacheEngineConfig.from_file()
  |     -> LMCacheMetadata(kv_shape, kv_dtype, ...)
  |     -> MockGPUConnector (no real GPU needed)
  |     -> LMCacheEngineBuilder.get_or_create()
  |     -> engine.post_init()
  |
  +-> WorkloadGenerator.run_workload()
        -> pattern.execute_operation()
             -> engine.store(tokens=token_ids)
             -> engine.retrieve(tokens=token_ids)
```

With a loaded tokenizer, token IDs are real vocabulary indices (e.g.
`[15496, 11, 616, 1438, …]`). Without one, patterns use deterministic synthetic
ranges. Either path exercises chunking and the rolling-prefix-hash pipeline
used by vLLM and SGLang integrations.

## Hugging Face Model Integration

The simulator supports integration with Hugging Face models to use real token
dictionaries and KV cache formats without performing inference.

### Features

- **Model Loading**: Download and load models from Hugging Face Hub or use
  local models
- **KV Cache Extraction**: Automatically extract KV cache parameters (shape,
  size, dtype) from model configs
- **Tokenizer Support**: Use model tokenizers for realistic token generation
- **Two Modes**:
  - `vocab-only`: Use vocabulary size only
  - `text-to-tokens`: Full tokenization support for text inputs

### Usage

#### Basic Model Integration

```bash
python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path /tmp/lmcache \
    --hf-model-name gpt2 \
    --auto-kv-shape \
    --pattern random \
    --duration 10
```

#### Text-to-Tokens Mode

```bash
python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path /tmp/lmcache \
    --hf-model-name gpt2 \
    --tokenizer-mode text-to-tokens \
    --text-input "data/sample-text.txt" \
    --auto-kv-shape \
    --pattern random \
    --duration 10
```

#### Local Models

```bash
python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path /tmp/lmcache \
    --model-path /path/to/local/model \
    --auto-kv-shape \
    --pattern random \
    --duration 10
```

### Model Options

Examples in this README often use **`gpt2`** for a small public checkpoint;
that is documentation convention, not a built-in default. Omitting HF flags
uses the fixed KV defaults in the table above.

- `--hf-model-name`: Hugging Face model identifier
  (e.g., `gpt2`, `meta-llama/Llama-2-7b-hf`)
- `--model-path`: Local path to model (overrides
  `--hf-model-name`)
- `--tokenizer-mode`: `vocab-only` (default) or
  `text-to-tokens`
- `--cache-dir`: Directory to cache downloaded models
  (default: `~/.cache/huggingface`)
- `--auto-kv-shape`: Automatically calculate KV shape
  from model config
- `--local-only`: Only use local models, don't download
- `--hf-token-file`: Path to Hugging Face token file
  (auto-detected if not specified)
- `--text-input`: Text file or inline text to tokenize
  (requires `--tokenizer-mode text-to-tokens`)

### Authentication

The tool automatically searches for Hugging Face token
files in the current directory (files matching `*.token`
or `.batesste-hugging-face-*.token`). You can also:

- Specify a token file with `--hf-token-file`
- Set the `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`
  environment variable
- Place a `.token` file in the project directory

### Supported Architectures

The tool automatically extracts KV cache parameters from
common transformer architectures:

- **Llama/Mistral**: Uses `num_hidden_layers`,
  `num_attention_heads`, `hidden_size`
- **GPT-2**: Uses `n_layer`, `n_head`, `n_embd`
- **Generic**: Falls back to common attribute names

### KV Cache Calculation

KV cache shape is calculated as:

```
[num_layers, 2, chunk_size, num_heads, head_dim]
```

Where:
- `num_layers`: Number of transformer layers
- `2`: K and V tensors
- `chunk_size`: Cache chunk size (from CLI or config)
- `num_heads`: Number of attention heads
- `head_dim`: Hidden size / num_heads

## Engine Manager

The `EngineManager` class (`src/engine-manager.py`)
wraps `LMCacheEngine` creation and provides simplified
methods:

| Method / Property | Description |
|-------------------|-------------|
| `create_engine()` | Create engine in-process |
| `store(token_ids)` | Store KV cache for tokens |
| `retrieve(token_ids)` | Retrieve cached KV data |
| `lookup(token_ids)` | Check prefix cache hits |
| `clear(token_ids)` | Clear cache entries |
| `freeze(enabled)` | Toggle freeze mode |
| `set_hot_cache(enabled)` | Toggle CPU hot cache |
| `is_healthy()` | Health check |
| `close()` | Destroy engine and free resources |
| `bytes_per_chunk` | KV bytes per chunk (property) |
| `bytes_per_token` | KV bytes per token (property) |

These methods accept arbitrary `list[int]` token IDs
(not limited to sequential ranges), matching how real
vLLM/SGLang integrations interact with LMCache.

## Workload Patterns

All workload patterns call the engine directly with
actual token ID lists. When a model tokenizer is loaded,
patterns slice real token IDs from the tokenized text.
Without a tokenizer, patterns fall back to sequential
token ranges.

### Random

Random read/write operations across a key range. Keys
are hashed to select token slices from the tokenized
text.

### Store-only (`--pattern store-only`)

Only `engine.store()` operations: each chunk is written
to the cache and one JSON line `{"tokens":[...]}` is
appended to a sidecar file. Default path is
`<storage-path>/.lmcache_io_chunk_tokens.jsonl`; set
`--chunk-index PATH` to choose the file explicitly.
Use this phase to populate storage before measuring
reads. The `workload` subcommand requires
`--chunk-index` because there is no `--storage-path`.

**Store timing and size stats**: Pass
`--per-op-store-log /path/to/file.jsonl` to append one
JSON object per successful store (`op_index`,
`ts_unix`, `ts_iso`, `latency_ms`, `bytes_written`).
Pass `--per-op-log /path/to/file.jsonl` for one JSON
object per operation (every op type): `op_index`,
`ts_unix`, `ts_iso`, `op_type`, `success`, `cache_hit`,
`latency_ms`, `kv_blocks`, `data_bytes`. If both
`--per-op-log` and `--per-op-store-log` are set,
`--per-op-log` is used.
Printed metrics (default `--output-format text`) list
per-operation-type averages, min/max, and when samples exist
`latency_std_ms`, `latency_p99_ms`, and `latency_p999_ms` under
`store_operations`, `retrieve_operations`, and `lookup_operations`
(when the lookup-only pattern runs).
Use `--output-format json` for machine-readable summaries.
Latency is wall time around `engine.store()` /
`engine.retrieve()`, or `engine.lookup()` for lookup-only;
`bytes_written` is logical KV bytes
for that chunk (same basis as the workload summary), not
a host OS block IO counter. Very large `--num-operations`
values keep all samples in RAM for percentiles.

### Retrieve-only (`--pattern retrieve-only`)

Random `engine.retrieve()` operations. Token sequences
are read from the JSONL sidecar produced by store-only
(same `--storage-path` / `--chunk-index` as the store
run, or copy the sidecar for remote backends). LMCache
needs the original token IDs to retrieve; listing blob
files under the cache directory is not enough.

Example two-phase run on a filesystem backend: **store-only** writes chunks
and appends each chunk’s `tokens` to
`<storage-path>/.lmcache_io_chunk_tokens.jsonl`. **Retrieve-only** reads
random lines from that sidecar and calls `retrieve` with the same token IDs.
Use the same `--storage-path` (and model/KV flags) for both so the cache and
metadata line up. Distinct chunk counts in metrics use **unique token
fingerprints** in the JSONL, so duplicate lines collapse to one bucket.

```bash
ST=/tmp/lmcache_sim_example

python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path "$ST" \
    --device cpu \
    --hf-model-name gpt2 \
    --auto-kv-shape \
    --pattern store-only \
    --num-operations 500

python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path "$ST" \
    --device cpu \
    --hf-model-name gpt2 \
    --auto-kv-shape \
    --pattern retrieve-only \
    --num-operations 500
```

#### Maximizing retrieve IOPS

Reported **IOPS** is `throughput_ops_per_sec` in JSON output (successful
ops divided by wall time). For retrieve-only that is effectively retrieves
per second in a **single Python thread**; omit **`--rate`** so the driver
does not sleep between ops. **`--chunk-size`** and KV flags apply to both
store-only and token-building patterns: keep the same values across store
and retrieve phases. For **A/B** comparisons across backends, `O_DIRECT`,
dtype, or chunk size, run
[`tests/run_retrieve_throughput_matrix.sh`](tests/run_retrieve_throughput_matrix.sh)
from this directory (set `LMCACHE_IO_MATRIX_*` env vars to tune op counts).
To approximate **aggregate** cluster RPS, run several **process** sessions
in parallel. The conservative pattern in
[`tests/run_retrieve_multi_process.sh`](tests/run_retrieve_multi_process.sh)
copies the populated cache so each worker has a **disjoint**
`--storage-path` (clear isolation for summed-throughput experiments). For
the **filesystem** backend, LMCache supports multiple **read-only**
retrievers against the **same** on-disk tree from **separate processes**
(one `LMCacheEngine` per process); do not run writers against that path
concurrently and give each process its own `--per-op-log` path. Do **not**
assume `LMCacheEngine.retrieve` is safe to call concurrently from **multiple
threads** on a single engine in one process. CI runs
[`tests/verify_retrieve_throughput_behavior.py`](tests/verify_retrieve_throughput_behavior.py)
to assert `--rate` caps throughput and that `run-this.sh` does not pass
`--rate` or `--fs-odirect` on lookup/retrieve lines.

**Lookup-only** (`--pattern lookup-only`) uses the same JSONL sidecar as
retrieve-only but calls `engine.lookup(tokens=...)` only. It measures
prefix-hit latency and full-chunk hit rate without loading KV tensors.
Add a third phase after the retrieve-only example:

```bash
python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path "$ST" \
    --device cpu \
    --hf-model-name gpt2 \
    --auto-kv-shape \
    --pattern lookup-only \
    --num-operations 500
```

Omit `--hf-model-name` / `--auto-kv-shape` if you want the fixed default KV
shape only; keep flags identical across store, retrieve, and lookup phases
when you use a model.

Continuous integration (on `main`) runs a small **store-only**, then
**retrieve-only**, then **lookup-only** smoke job when files under
`tools/lmcache-io-tester/` change; see `.github/workflows/lmcache-io-tester.yml`.

### Steady-State

Mix of read and write operations with configurable
read ratio. Simulates steady-state cache behavior.

### Conversation

Replays multi-turn conversations through the KV cache
engine, modelling real LLM prefix caching. Each
conversation builds a cumulative token context: user
turns trigger a retrieve (prefix cache lookup) and
assistant turns trigger a store (cache the full
context so far).

```bash
python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path /tmp/lmcache \
    --hf-model-name gpt2 \
    --local-only \
    --auto-kv-shape \
    --tokenizer-mode text-to-tokens \
    --conversation-file \
        data/sample-conversations.json \
    --pattern conversation \
    --duration 10
```

Conversation data must conform to the schema in
`data/conversation-schema.json`. A sample dataset is
provided in `data/sample-conversations.json`. To
download larger datasets from Hugging Face, use the
`download` subcommand:

```bash
python -m src.lmcache-sim download \
    --dataset sharegpt \
    --output data/sharegpt-5k.json \
    --max-conversations 5000
```

Supported datasets: `sharegpt` (ShareGPT52K, free),
`lmsys` (LMSYS-Chat-1M, gated), `wildchat`
(WildChat-1M, gated), `longbench` (LongBench,
long-context Q&A, free), `vicuna` (ShareGPT Vicuna
unfiltered, free), `ultrachat` (UltraChat multi-turn,
free), and `oasst` (OpenAssistant oasst1, free).

#### Concurrent Conversations

The `--concurrency N` option simulates N users
chatting simultaneously. Operations from each active
conversation are interleaved round-robin, so the
cache sees mixed access patterns that mirror a real
inference server:

```bash
python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path /tmp/lmcache \
    --hf-model-name gpt2 --local-only \
    --auto-kv-shape \
    --tokenizer-mode text-to-tokens \
    --conversation-file data/sharegpt-5k.json \
    --pattern conversation \
    --concurrency 32 \
    --duration 60
```

#### Multi-Pass Mode

Use `--passes N` to replay the conversation dataset
N times against the same engine without restarting.
Per-pass metrics show how the cache hit rate
improves as the cache warms:

```bash
python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path /tmp/lmcache \
    --hf-model-name gpt2 --local-only \
    --auto-kv-shape \
    --tokenizer-mode text-to-tokens \
    --conversation-file data/sharegpt-5k.json \
    --pattern conversation \
    --concurrency 16 \
    --passes 3 \
    --duration 30
```

#### Persistent Cache

Add `--persist-cache` to report warm-cache state on
startup. LMCache's filesystem backend writes `.data`
files that survive between runs. Point subsequent
runs at the same `--storage-path` (and omit
`--cleanup`) to build on the prior cache state:

```bash
python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path /tmp/lmcache \
    --hf-model-name gpt2 --local-only \
    --auto-kv-shape \
    --tokenizer-mode text-to-tokens \
    --conversation-file data/sharegpt-5k.json \
    --pattern conversation \
    --persist-cache \
    --duration 30
```

#### Large Dataset Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-conversations` | 0 (all) | Cap on conversations loaded |
| `--shuffle-conversations` | off | Randomize order |
| `--seed` | none | RNG seed for reproducible shuffle |

## Workload Generation

The workload generation flow:

1. **Key Generation**: Patterns generate keys (e.g.,
   `key_0`, `key_1`)

2. **Token ID Selection**:

   **Without tokenizer (fallback)**:
   ```python
   start = abs(hash(key)) % 10000
   token_ids = list(range(start, start + 256))
   ```

   **With tokenizer**:
   ```python
   start_idx = abs(hash(key)) % len(tokenized_text)
   token_ids = tokenized_text[start_idx:start_idx+256]
   ```

3. **Direct Engine Calls**:
   ```python
   engine.store(token_ids)      # store KV cache
   engine.retrieve(token_ids)   # retrieve cached data
   engine.lookup(token_ids)     # lookup-only: prefix hit count, no KV load
   ```

4. **Metrics Collection** (default human-readable `text` output):

   - **Header**: `Workload Metrics` then one line: duration, operation counts,
     ok/fail, and **IOPS** (successful ops / elapsed time),
     semicolon-separated.
   - **Table**: one row each for `store` and `retrieve`, plus `lookup` when the
     run used **lookup-only**, with op counts, KV blocks, **IO Bytes** (logical
     KV bytes; zero for lookup), average / P99 / P99.9 ms where samples exist,
     and retrieve / lookup **Hit%** (full-chunk hit for lookup: returned prefix
     covers all sidecar tokens).
   - **Hits per KV block**: histogram of how many logical blocks (by default,
     distinct chunk token fingerprints in the JSONL sidecar for
     **retrieve-only** / **lookup-only**) had 0, 1, … successful hits (retrieve
     tokens loaded; lookup full prefix). Prints **Total
     Unique Blocks** for that index. Only non-zero buckets are shown (up to
     nine interior rows); the last row is **`>N`** where `N` is the last printed
     bucket index and its count sums every higher bucket plus the internal
     overflow tally for hit counts above 10.
   - Optional **`--output-format json`**: full metrics object including
     `kv_block_hit_histogram`, `chunk_index_distinct_chunks`, and related
     fields when applicable.

## Configuration

The tool generates YAML configuration files based on
storage type. Config files can be customized and reused.

[Full configuration docs][ref-lmcache-config].

### Storage backends (`--storage-type`)

| Type | Required flags | LMCache wiring / notes |
|------|----------------|------------------------|
| `filesystem` | `--storage-path` | ``remote_storage_plugins: [fs]`` and
``extra_config.remote_storage_plugin.fs.base_path`` (no legacy ``remote_url``) |
| `local-disk` | `--storage-path` | ``local_disk`` + ``file://`` path; uses
[local disk][ref-lmcache-local] backend (not ``fs://``) |
| `block-device` | `--block-device` | Mounts device; same ``fs`` plugin +
``base_path`` as ``filesystem`` |
| `gds` | `--storage-path` | Sets `gds_path` and CuFile options |
| `redis` | `--remote-url` | `redis://` or `redis-sentinel://` |
| `s3` | `--remote-url`, `--s3-region` | `s3://bucket`; region and AWS keys
in `extra_config` (see [S3 backend][ref-lmcache-s3]) |
| `remote` | `--remote-url` | Any other scheme (`lm://`, `mooncakestore://`, …) |

Optional **`--extra-config PATH`**: merge a YAML or JSON object into the
generated config (nested `extra_config` keys merge with existing S3 or
other backend settings). Optional **`--probe-remote`**: TCP reachability
for host/port URLs before starting the engine (skipped for `s3://`).

**`--fs-odirect`** (`start`, `run`, `verify`): after `--extra-config`, merges
LMCache `extra_config` entries to turn on POSIX `O_DIRECT` for the ``fs``
remote connector on **`filesystem`** and **`block-device`** backends
(`fs_connector_use_odirect`, `save_chunk_meta: false`). Other storage types
print a warning and ignore the flag. Keys follow upstream LMCache; adjust
with `--extra-config` if your package version uses different names.

### LMCache stderr warnings (what they mean)

At ``LMCACHE_LOG_LEVEL=WARNING`` (the tester default), you may still see:

- **Fallback to python backend ``lmcache.non_cuda_equivalents``** — normal on
  CPU-only hosts or when CUDA is unavailable; use a GPU build with CUDA to
  pick ``lmcache.c_ops`` instead.
- **Could not load ``builtin`` from vLLM** — the sim does not require vLLM;
  LMCache then uses Python ``hash``. The tester sets ``PYTHONHASHSEED`` by
  default so follow-on hash-seed warnings are avoided; install vLLM if you
  need its hash functions.
- **Controller message sender is not initialized** — expected for this
  in-process engine: there is no LMCache cache-controller worker. Harmless for
  local IO testing.
- **``remote_url`` is deprecated** — still emitted for ``redis``, ``s3``, and
  generic ``remote`` backends until those configs migrate to
  ``remote_storage_plugins`` upstream; ``filesystem`` / ``block-device`` use
  the plugin path and should not log this.

**`verify` subcommand**: one chunk `store` then `retrieve`/`lookup`; exits
0 when hit counts match the chunk size. Set **`LMVERIFY_RELAXED=1`** to
allow partial hits; **`LMVERIFY_MIN_RETRIEVE`** sets the minimum retrieve
token count (defaults to chunk size).

## Cache File Format

When LMCache stores KV cache data to disk (filesystem or
block device backends), it creates `.data` files.

### File Naming Convention

Files are named using the pattern:

```
<model>@<world_size>@<worker_id>@<hash>@<dtype>.data
```

Example:
`lmcache_model@1@0@3991436492686501@half.data`

**Components:**
- `<model>`: Model identifier
- `<world_size>`: Total number of workers
- `<worker_id>`: Worker ID
- `<hash>`: Chunk hash (rolling prefix hash of tokens)
- `<dtype>`: Data type (`half`, `float`, etc.)

### Serialization Format

The tool uses `remote_serde: naive` format, which stores
KV cache data as uncompressed binary with shape metadata
in the file header.

<!-- References -->

[ref-lmcache-config]: https://docs.lmcache.ai/api_reference/configurations.html
[ref-lmcache-local]: https://docs.lmcache.ai/kv_cache/storage_backends/local_storage.html
[ref-lmcache-s3]: https://docs.lmcache.ai/kv_cache/storage_backends/s3.html
