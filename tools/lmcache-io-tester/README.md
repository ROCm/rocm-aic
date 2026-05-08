# LMCache Simulation Tool

A Python CLI tool for running LMCache cache operations in-process, configuring
storage backends, and generating workload traffic to test cache performance.

The tool creates an `LMCacheEngine` directly in the simulator process, calls
`engine.store()` and `engine.retrieve()` with actual token ID lists from a
Hugging Face tokenizer, and measures cache performance under various traffic
patterns.

## Installation

Install the required dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Running the Tool

Run the tool as a Python module:

```bash
python -m src.lmcache-sim --help
```

## Project Layout

```
src/                  Core simulator modules
data/                 Conversation schemas, sample data
tests/                Test scripts
configs/              Generated YAML configs (runtime)
```

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
             -> engine.store(tokens=actual_token_ids)
             -> engine.retrieve(tokens=actual_token_ids)
```

Token IDs passed to the engine are actual vocabulary indices from the HF
tokenizer (e.g. `[15496, 11, 616, 1438, ...]`), not sequential ranges. This
exercises the same chunking and rolling-prefix-hash pipeline that real vLLM
and SGLang integrations use.

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

### Sequential

Sequential write operations with advancing token
offsets, walking through the tokenized text chunk by
chunk.

### Burst

Bursts of store operations at regular intervals.
Simulates traffic spikes during inference serving.

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
   engine.store(token_ids)     # store KV cache
   engine.retrieve(token_ids)  # retrieve cached data
   ```

4. **Metrics Collection**: Measures per-operation:
   - Latency (min, max, average) per operation type
   - Throughput (operations per second)
   - Cache hit/miss rates
   - KV blocks written/read
   - Storage I/O (total bytes written/read, scaled
     to KiB/MiB/GiB as appropriate)
   - Store vs retrieve breakdown

## Configuration

The tool generates YAML configuration files based on
storage type. Config files can be customized and reused.

[Full configuration docs][ref-lmcache-config].

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

[ref-lmcache-config]: https://docs.lmcache.ai/api_reference/configurations.html
