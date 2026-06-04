<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# LLM Agent-X benchmark (CC trace replay)

Engine-agnostic workload for replaying [SemiAnalysis Claude Code agent
traces][cc-traces] against any OpenAI-compatible text LLM server. Measures
per-request TTFT, wall time, and token usage under realistic agentic context
growth (main-agent turns plus nested sub-agent requests).

Part of [rocm-aic](../../README.md).

Dataset source:
[semianalysisai/cc-traces-weka-with-subagents-052726-256k][hf-dataset]
(470 traces, in+out ≤ 256k proxy tokens, Apache-2.0). Pinned in
[`configs/cc-traces-hf.yaml`](configs/cc-traces-hf.yaml).

Traces ship **metadata only** (`in`, `out`, `hash_ids`, timing) — no prompt
text. The replay client synthesizes filler prompts sized with the dataset's
proxy tokenizer (`o200k_base`) to match each request's input length.

## Compared to other benchmarks

| Benchmark | Focus |
|-----------|-------|
| [llm-prefill-benchmark](../llm-prefill-benchmark) | Long text prefill (Gutenberg) |
| **llm-agentx** | Real agentic CC traces + KV-oriented ISL growth |
| [ttft-lmcache](../ttft-lmcache) | Controlled KV-cache hit-rate sweep |

## Prerequisites

- Python 3.10+ with benchmark deps:

```bash
cd benchmarks/llm-agentx
make install    # uses repo .venv/bin/python3 when present
```

- Running text LLM server: `curl -sS http://127.0.0.1:8000/v1/models`
- ~900 MB disk after `make data`
- Start vLLM **detached** before `make run-parallel`. Interactive
  `docker run -it` in the **same** shell can receive Ctrl+C and stop the
  server.

## Quick start

```bash
cd benchmarks/llm-agentx
make install
make data
export BASE_URL=http://127.0.0.1:8000
export MODEL=Qwen/Qwen2.5-3B-Instruct
make check-server
make run ITERATIONS=5
make run-parallel WORKERS=4 ITERATIONS=10
make report
```

Dry-run against CI fixtures (no server):

```bash
AGENTX_DATA_ROOT=tests/fixtures AGENTX_DRY_RUN=1 ./run-agent.sh
```

Cap requests per trace (useful for smoke tests):

```bash
AGENTX_MAX_REQUESTS=3 make run ITERATIONS=1
```

## Runtime YAML

The benchmark reads checked-in defaults from `../runtime-defaults.yaml`, which
avoids re-exporting the same server, model, corpus, and worker settings for
each run. Put local changes in `../runtime.yaml`, then use the existing Make
targets:

```bash
$EDITOR ../runtime.yaml
make run-parallel
```

Use `RUNTIME_CONFIG_FILE=/path/to/runtime.yaml` to select another file.
Environment variables override checked-in defaults. When an override YAML file
is detected, mapped runtime env vars are ignored so the file wins; command-line
Make variables remain explicit one-off overrides:

```bash
make run AGENTX_MAX_REQUESTS=3
```

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `BASE_URL` | `http://127.0.0.1:8000` | OpenAI API root (no `/v1` suffix) |
| `MODEL` | `Qwen/Qwen2.5-3B-Instruct` | Must match a served model id |
| `AGENTX_DATA_ROOT` | `../../data/cc-traces` | `traces.jsonl` corpus |
| `ITERATIONS` | `1` | Traces per worker run |
| `AGENTX_SEED` | random | Trace selection seed |
| `AGENTX_MAX_REQUESTS` | unset | Cap requests replayed per trace |
| `AGENTX_MAX_CONTEXT` | auto from `/v1/models` | Skip requests when `in+out` exceeds limit |
| `AGENTX_STRICT` | `0` | Fail instead of skip on oversized requests |
| `AGENTX_HONOR_THINK_TIME` | `0` | Sleep trace `think_time` between requests |
| `MAX_TOKENS` | `512` | Cap completion tokens per request |
| `AGENTX_DRY_RUN` | `0` | Skip HTTP; emit fixture metrics |
| `AGENTX_HF_HOME` | `data/cc-traces/.hf-cache` | HF download cache |
| `LLM_AGENTX_BENCH_ROOT` | auto | Override benchmark tree (Slurm) |

## Replay model

For each trace, requests are replayed in order. Main-agent turns (`type`
`s`/`n`) accumulate chat history; `type: subagent` blocks start a fresh
sub-agent conversation for their inner requests. Each HTTP call pads the
prompt to the trace's `in` token count and requests up to `out` completion
tokens (capped by `MAX_TOKENS`).

## Layout

| Path | Role |
|------|------|
| `run-agent.sh` | Serial loop; JSONL on stdout |
| `run-agent-parallel.sh` | Parallel workers |
| `scripts/download-dataset.py` | HF fetch of `traces.jsonl` |
| `scripts/token_fill.py` | o200k_base filler for target ISL |
| `scripts/replay-trace.py` | Trace replay + TTFT metrics |
| `scripts/parse-results.py` | Aggregate worker JSONL |
| `configs/cc-traces-hf.yaml` | Pinned HF dataset |
| `tests/fixtures/traces.jsonl` | CI mini trace |

## References

[cc-traces]: https://huggingface.co/datasets/semianalysisai/cc-traces-weka-with-subagents-052726-256k
[hf-dataset]: https://huggingface.co/datasets/semianalysisai/cc-traces-weka-with-subagents-052726-256k
