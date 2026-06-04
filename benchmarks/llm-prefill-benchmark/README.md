<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# LLM prefill benchmark (Gutenberg long-context)

Engine-agnostic workload for measuring **time-to-first-token (TTFT)** and
sustained prefill against any OpenAI-compatible vLLM server. Used by
[recipies/vllm-lmcache-hipfile](../../recipies/vllm-lmcache-hipfile) and
[recipies/vllm-lmcache-nixl](../../recipies/vllm-lmcache-nixl) Slurm jobs.

Part of [rocm-aic](../../README.md).

## Compared to ttft-lmcache

| Benchmark | Focus |
|-----------|-------|
| [ttft-lmcache](../ttft-lmcache) | Controlled hit-rate sweep on one fixed prompt |
| **llm-prefill-benchmark** | Random Gutenberg chunks + questions, parallel workers |

## Prerequisites

- Host: `python3`, `jq` (Slurm jobs), optional venv with `pip install -r requirements.txt`
- Running server: `curl -sS http://127.0.0.1:8000/v1/models` (or your `BASE_URL`)

## Quick start

```bash
cd benchmarks/llm-prefill-benchmark
make data BOOK_SLUG=war-and-peace BOOK_PG_ID=2600 \
  BOOK_TITLE='War and Peace' BOOK_AUTHOR='Leo Tolstoy'
export BASE_URL=http://127.0.0.1:8000
export MODEL=openai/gpt-oss-120b
make run ITERATIONS=5
make run-parallel WORKERS=4 ITERATIONS=10
```

## Runtime YAML

The benchmark can read defaults from a YAML file so common run settings do not
need to be exported before every test. Copy the repo-level example, edit the
`llm_prefill` section, then run the scripts or Make targets normally:

```bash
cp ../runtime.yaml.example ../runtime.yaml
make run-parallel
```

Use `RUNTIME_CONFIG_FILE=/path/to/runtime.yaml` to select a different file.
Environment variables and command-line Make variables still win over YAML
values, so one-off overrides keep working:

```bash
make run-parallel ITERATIONS=20
```

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `BASE_URL` | `http://127.0.0.1:8000` | OpenAI-compatible API root (no `/v1` suffix) |
| `MODEL` | `openai/gpt-oss-120b` | Model id for chat completions |
| `BOOK_DATA_ROOT` | `../../data/gutenberg` | Gutenberg chunk + `*.questions.json` tree |
| `BOOK_SLUG` | unset | Single-book mode when set |
| `ITERATIONS` | `1` | Requests per worker |
| `RUN_LONG_SEED` | random | Reproducible chunk/question selection |
| `RUN_LONG_MAX_TOKENS` | `512` | Max completion tokens per chat request |
| `BOOK_SLUG_FILE` | unset | Path to one slug per line (library subset) |
| `LLM_PREFILL_BENCH_ROOT` | auto | Override benchmark tree (Slurm sets this) |

Set a longer decode cap before `make run` or `make run-parallel`:

```bash
export RUN_LONG_MAX_TOKENS=2048
make -C benchmarks/llm-prefill-benchmark run-parallel WORKERS=12 ITERATIONS=100
```

Ensure `prompt_tokens + RUN_LONG_MAX_TOKENS` stays within the server's
`max_model_len`.

## Slurm

From the repo root, recipe jobs set `LLM_PREFILL_BENCH_ROOT` and
`VLH_GUTENBERG_DATA_ROOT` (or `VLN_*` for the NIXL recipe). See
[run-slurm.sh](../../run-slurm.sh) and [run-slurm-nixl.sh](../../run-slurm-nixl.sh).

## Storage A/B matrix (recipe comparison)

| Mode | Storage path |
|------|----------------|
| vllm-lmcache-hipfile | LMCache GdsBackend + hipFile |
| vllm-lmcache-nixl posix | NIXL POSIX |
| vllm-lmcache-nixl ais | NIXL AIS + hipFile |

All three use this benchmark tree for TTFT measurement.

## Layout

| Path | Role |
|------|------|
| `run-long.sh` | Serial benchmark loop |
| `run-long-parallel.sh` | Parallel workers with distinct seeds |
| `scripts/stream-chat-completion.py` | TTFT measurement client |
| `scripts/test-aic.py` | A/B cache test helper |
| `configs/gutenberg-library.json` | 100-book manifest for `make data-all` |
