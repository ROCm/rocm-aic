<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# LMCache IO Tester

A Python CLI for running LMCache cache operations in-process, configuring
storage backends, and generating workload traffic to test cache performance.

The tool creates an `LMCacheEngine` in the simulator process and calls
`engine.store()` and `engine.retrieve()` with token ID lists. With
`--hf-model-name` or `--model-path` (and optional
`--tokenizer-mode text-to-tokens`), IDs come from the model tokenizer;
otherwise workloads use synthetic ranges from operation keys. It reports
latency, IO bytes, cache outcomes, and optional per-chunk hit histograms.

## Installation

From the **repository root** (not this directory alone):

```bash
cd ../..   # repository root if you are in tools/lmcache-io-tester
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick start

Run from `tools/lmcache-io-tester/`:

```bash
python -m src.lmcache-sim --help
python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path /tmp/lmcache-ci \
    --device cpu \
    --pattern store-only \
    --num-operations 32 \
    --config configs/lmcache-config.yml
```

## Project layout

```
src/                  Core simulator modules
data/                 Conversation schemas, sample data
tests/                Test scripts
configs/              Generated YAML configs (runtime)
docs/USAGE.md         Patterns, HF models, storage, metrics
```

## Architecture

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
  |     -> LMCacheEngineBuilder.get_or_create()
  |
  +-> WorkloadGenerator.run_workload()
        -> engine.store() / retrieve() / lookup()
```

With a loaded tokenizer, token IDs are real vocabulary indices. Without one,
patterns use deterministic synthetic ranges. Either path exercises chunking
and the rolling-prefix-hash pipeline used by vLLM and SGLang integrations.

## Further reading

See [docs/USAGE.md](docs/USAGE.md) for Hugging Face integration, workload
patterns (store/retrieve/lookup, conversation, steady-state), storage
backends, LMCache warnings, and on-disk cache file format.
