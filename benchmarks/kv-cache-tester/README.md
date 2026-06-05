<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# kv-cache-tester benchmark (trace replay)

Engine-agnostic workload using upstream
**[callanjfox/kv-cache-tester][kv-cache-tester]** to replay Claude Code traces
against any OpenAI-compatible vLLM server. Same harness as the
[LMCache MI300X blog][atom-blog].

Part of [rocm-aic](../../README.md). Used with
[recipies/aic-drivenets](../../recipies/aic-drivenets/) and
[recipies/vllm-atom-andy](../../recipies/vllm-atom-andy/).

## Compared to other benchmarks

| Benchmark | Focus |
|-----------|-------|
| [llm-prefill-benchmark](../llm-prefill-benchmark) | Gutenberg long-context prefill |
| [llm-agentx](../llm-agentx) | SemiAnalysis CC agent traces (HF dataset) |
| **kv-cache-tester** | Upstream 739-trace replay (MI300X blog profile) |
| [ttft-lmcache](../ttft-lmcache) | Controlled KV hit-rate sweep |

## Prerequisites

- Host: `python3`, `git`
- Running vLLM server: `curl -sS http://127.0.0.1:8000/v1/models`
- ~1 GB disk for upstream clone + traces (under `data/kv-cache-tester/`)

## Quick start

```bash
make -C recipies/aic-drivenets run-batch    # or any vLLM recipe on :8000
make -C benchmarks/kv-cache-tester install data check-server
make -C benchmarks/kv-cache-tester run
```

Smoke test (single prompt, shorter):

```bash
make -C benchmarks/kv-cache-tester run-smoke BASE_URL=http://127.0.0.1:8000
```

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `BASE_URL` | `http://127.0.0.1:8000` | OpenAI API root (no `/v1` suffix) |
| `KV_CACHE_TESTER_ROOT` | `../../data/kv-cache-tester` | Upstream git clone |
| `KV_CACHE_TESTER_CONFIG` | `configs/trace-replay-blog.yaml` | Profile to run |
| `KV_CACHE_TESTER_SMOKE_CONFIG` | `configs/smoke-single-prompt.yaml` | Smoke profile |

## Profiles

| Config | Script | Use |
|--------|--------|-----|
| `trace-replay-blog.yaml` | `trace_replay_tester.py` | Blog stress (4→32 users, 1200s, 100K context) |
| `smoke-single-prompt.yaml` | `single_prompt_tester.py` | Quick connectivity check |

## Layout

| Path | Role |
|------|------|
| `scripts/install-upstream.py` | `git clone --recursive` at pinned ref |
| `scripts/build_argv.py` | YAML profile → CLI argv |
| `scripts/run-profile.py` | Execute upstream script |
| `scripts/check-server.py` | Probe `/v1/models` |
| `configs/trace-replay-blog.yaml` | MI300X blog defaults |
| `tests/test_build_argv.py` | CI unit test |

## Ansible

Cluster hosts can use the same wrapper via
[`ansible/roles/kv_cache_tester`](../../ansible/roles/kv_cache_tester) with
**`kv_cache_tester_use_benchmark_wrapper: true`** (default).

<!-- References -->
[kv-cache-tester]: https://github.com/callanjfox/kv-cache-tester
[atom-blog]: https://andyluo7.github.io/llm/amd/mi300x/vllm/lmcache/performance/2026/05/22/atom-lmcache-kv-cache-offload-mi300x/
