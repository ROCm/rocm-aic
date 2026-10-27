# AgentX Benchmark

The AgentX benchmark exercises an OpenAI-compatible inference server with
real multi-turn agentic coding traces from the
[SemiAnalysis `cc-traces-weka-062126`](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126)
dataset. Unlike the cliff benchmark (synthetic shared-prefix prompts), AgentX
reflects real agentic workload patterns: variable ISL/OSL per turn, subagent
call-chains, and heavy KV-cache prefix reuse across turns.

Uses the `agentx` CLI (part of the `aiperf` package) — the same tool and
dataset used by SemiAnalysis InferenceX CI.

## Quick Start

```bash
# Build the aic-agentx container (once)
make agentx-build

# Run a single benchmark (requires vLLM to be up via `make up`)
make agentx-run VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# With Prometheus metrics from the inference server (disaggregated: prefill + decode)
make agentx-run VLLM_MODEL=... \
  AGENTX_SERVER_METRICS="http://prefill-host:6100/metrics http://decode-host:6100/metrics"

# Sweep concurrencies 1, 2, 4 overnight
make agentx-sweep VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

## Scripts

| File | Purpose |
| --- | --- |
| `run_agentx.sh` | Single benchmark run via the `aic-agentx` container |
| `overnight_sweep.sh` | Concurrency sweep (1 / 2 / 4 by default) |
| `turn_visualizer.py` | Produce an HTML turn-timeline from a results JSONL |

## Environment Variables

All variables have defaults; the make targets forward the common ones.

| Variable | Default | Description |
| --- | --- | --- |
| `BENCH_ENDPOINT` | `http://localhost:8000` | Inference server URL |
| `AGENTX_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Model ID (must be loaded) |
| `AGENTX_TOKENIZER` | same as `AGENTX_MODEL` | Tokenizer ID (downloaded from HF) |
| `AGENTX_CONCURRENCY` | `2` | Parallel workers |
| `AGENTX_DURATION` | `3600` | Benchmark duration (seconds) |
| `AGENTX_SERVER_METRICS` | _(empty)_ | Space-separated Prometheus `/metrics` URLs passed to `agentx --server-metrics`; useful for multi-node disaggregated setups |
| `AGENTX_OTEL_URL` | _(empty)_ | OTEL collector endpoint; enables OTEL export when set |
| `AGENTX_IMAGE` | `aic-agentx:latest` | Container image ref |
| `HF_TOKEN` | _(empty)_ | HuggingFace token for dataset/tokenizer access |
| `HF_HOME` | `~/.cache/huggingface` | Host HF cache directory (bind-mounted read-only) |

## OTEL / Monitoring

When `AGENTX_OTEL_URL` is set, agentx emits OTEL metrics (TTFT, ITL,
throughput) to the target collector. Point it at the AIC monitoring stack's
Alloy OTLP endpoint, e.g.:

```bash
make agentx-run VLLM_MODEL=... AGENTX_OTEL_URL=http://aic-alloy:4318
```

## Results

Results land in `$(LOG)/agentx/<timestamp>_c<N>/`. Each run produces:

- `run.log` — console log teed from the run
- `aiperf_artifacts/profile_export_aiperf.json` — per-request results
- `aiperf_artifacts/profile_export_aiperf.csv` — tabular summary
- `aiperf_artifacts/profile_export_aiperf_timeslices.{json,csv}` — time-series slices

## Building the Image

The `aic-agentx` image is a thin `python:3.12-slim` container with `aiperf`
(which installs the `agentx` CLI) installed. No ROCm runtime.

```bash
# Via bake (recommended):
docker buildx bake -f docker/docker-bake.hcl agentx --load

# Standalone (from repo root):
DOCKER_BUILDKIT=1 docker build \
  -f docker/agentx/Dockerfile \
  -t aic-agentx:latest \
  .
```

Pin a specific aiperf version with `--build-arg AIPERF_VERSION=0.5.0` or
the `AIPERF_VERSION` bake variable.
