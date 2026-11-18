# AgentX Benchmark

AgentX replays recorded agentic coding sessions against an OpenAI-compatible
server. Each trace is one real multi-turn session: the prompt grows turn by
turn, subagents fan out in parallel, and the recorded pauses between requests
are honoured. Unlike the cliff benchmark (synthetic shared-prefix prompts), it
exercises the KV cache the way an agent does: long prefixes reused across
turns, with new sessions evicting old ones.

The client is [`aiperf`](https://github.com/ai-dynamo/aiperf), run as
`aiperf profile --scenario inferencex-agentx-mvp`, the scenario SemiAnalysis
InferenceX uses. The scenario fixes what makes runs comparable: streaming
(so TTFT and inter-token latency are measured), `ignore_eos`, a per-trajectory
cache-bust marker at the start of the prompt, a warm-up before the measured
phase, replay of the recorded delays (idle gaps
capped at 10 s), and a duration of at least 900 s. `aiperf` refuses a run that
breaks one of these unless `--unsafe-override` is given, and then marks the
result `submission_valid: false`.

The traces come from the public
[`semianalysisai/cc-traces-weka-062126`](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126)
dataset (393 traces, not gated). They hold token counts and block hashes, not
text: `aiperf` synthesizes prompts of the recorded length whose shared blocks
are identical.

## The corpus needs a large context

The traces were recorded against a large-context model. The smallest one peaks
at 44,048 tokens of context (prompt plus output), the median at about 226,000.
`aiperf` drops every trace whose peak exceeds `--max-context-length`, which the
runner always passes (the served model's `max_model_len` unless
`AGENTX_MAX_CONTEXT` is set), and stops if none remain:

| Served context | Traces that fit (of 393) |
| --- | --- |
| 32,768 | 0 |
| 65,536 | 7 |

So the benchmark as published needs a model served with a context of several
hundred thousand tokens. For anything smaller there is the scaled corpus.

## Scaled corpus (not the AgentX benchmark)

`make agentx-corpus` writes a copy of the corpus with every token count
divided by `AGENTX_CORPUS_SCALE` (default 8) to
`~/.cache/aic-agentx/cc-traces-weka-062126-scaled-<N>/`. The request order,
the timing, the subagent fan-out and the prefix-reuse shape are kept; the
context sizes, the KV footprint and the load on the server are not. With a
divisor of 8, 220 of the 393 traces fit a context of 32,768.

**A scaled corpus is an AgentX-shaped workload, not the AgentX benchmark. Its
results are not comparable with published AgentX or InferenceX numbers.** The
runner adds `--unsafe-override` for it, so the result is stamped
`submission_valid: false`, and says so in `run.log`. The same holds for any
local corpus (`AGENTX_CORPUS`) and for a duration under 900 s.

`SCALED-CORPUS.txt` in the corpus directory records the source dataset and
revision, the divisor, and how many traces fit common context sizes. The
output depends only on the dataset revision and the divisor.

## Running it

The stack publishes no host port, so the client runs in a container on the
`aic` docker network and reaches vLLM by container name. The runner resolves
that name itself, checks `/v1/models` from the same network, and runs
`aiperf` as the invoking user.

```bash
# Build the client image (once): aic-agentx, python:3.12-slim + aiperf
make agentx-build

# With the stack up (make up / make up-batch), against a large-context model
make agentx-run

# Small-context model: scaled corpus, generated on first use
make agentx-run AGENTX_CORPUS_SCALE=8

# Print the docker command without running anything
make agentx-run AGENTX_ARGS=--dry-run

# One run per concurrency level
make agentx-sweep AGENTX_CONCURRENCIES="1 2 4"
```

`BENCH_ENDPOINT=http://<container>:8000` benchmarks another server on the
network (the `vram_only` arm, say), or a remote one; a server on the docker
host is `http://host.docker.internal:<port>`. The model defaults to
`VLLM_MODEL`, else to the model the endpoint serves.
[`docs/ENVIRONMENT.md`](../../docs/ENVIRONMENT.md#agentx) lists every
`AGENTX_*` variable; the scripts take the same settings as options when run
directly (see the header of each).

| File | Purpose |
| --- | --- |
| `run_agentx.sh` | One run, from the `aic-agentx` container |
| `overnight_sweep.sh` | One run per concurrency level |
| `scaled_corpus.sh` | Generate (or reuse) a scaled corpus and print its directory |
| `scale_corpus.py` | The scaling itself; runs inside the `aic-agentx` image |
| `turn_visualizer.py` | HTML turn timeline from a run's `profile_export.jsonl` |

## Results

A run writes everything to one directory, `$(LOG)/agentx/<timestamp>_c<N>/`
(a sweep to `$(LOG)/agentx/sweep_<timestamp>/c<N>/`), owned by the invoking
user:

| File | Content |
| --- | --- |
| `run.log` | The settings, the exact `docker run` command, and the `aiperf` console output |
| `profile_export_aiperf.json`, `.csv` | Aggregate metrics: TTFT, inter-token latency, request latency, throughput, token counts; `submission_valid` |
| `profile_export_aiperf_timeslices.json`, `.csv` | The same metrics per `AGENTX_SLICE_DURATION` window |
| `profile_export.jsonl` | One record per request: session, turn index, agent depth, phase, and its metrics |
| `profile_export_console.txt` | The summary tables as printed |
| `server_metrics_export.json`, `.csv` | Prometheus metrics scraped during the run: vLLM's `/metrics` and, on this stack, LMCache's (port 9091) |
| `logs/aiperf.log` | The `aiperf` log |

```bash
python3 benchmarks/agentx/turn_visualizer.py logs/agentx/<run>/profile_export.jsonl
```

writes `profile_export.html` next to it: request latency per turn for each
session, subagents and warm-up turns marked.

Server-side behaviour during a run (KV cache hits, L1/L2 traffic, evictions)
is on the Grafana dashboard, through the stack's own Prometheus. `aiperf` can
also push its client metrics over OTLP (`--otel-url`, through
`AGENTX_EXTRA_ARGS`), but the AIC monitoring stack has no OTLP receiver, so
that needs a collector of your own.

## gfx1201 (16 GB) note

A 16 GB card serves `Qwen/Qwen2.5-3B-Instruct`, whose context is 32,768
tokens, so only the scaled corpus runs there. Start the stack with
`make up-dev` (`--enforce-eager`) and `VLM_GPU_MEMORY_UTILIZATION=0.68`:

- With CUDA graphs (`make up`, `make up-batch`) graph capture takes about
  4 GiB, which leaves 0.9 GiB for the KV cache at 0.74, and vLLM v0.34.0 then
  refuses a 32,768-token context at startup (it needs 1.12 GiB).
- In eager mode 0.74 starts (a KV cache of 152,256 tokens) but leaves no VRAM
  free, and the first 4,096-token prefill chunk of a trace ends the engine
  with `torch.OutOfMemoryError`. 0.68 gives a KV cache of 124,416 tokens
  (4.27 GiB) and about 0.7 GiB of headroom.

Concurrency 1 and 2 ran clean that way (2026-11-18): no preemptions, and the
KV cache never more than 24 % in use.

```bash
make up-dev \
  ROCM_ARCH=gfx1201 \
  VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct \
  AIC_L2_BACKEND=nixl_posix \
  NVME_DATA=/mnt/lmcache-nvme/aic-nvme \
  VLM_LOAD_FORMAT=auto \
  LMCACHE_L1_SIZE_GB=4 \
  VLM_GPU_MEMORY_UTILIZATION=0.68

make agentx-run AGENTX_CORPUS_SCALE=8 AGENTX_CONCURRENCY=1
```

## The image

`aic-agentx` is `python:3.12-slim` with `aiperf` installed and `aiperf` as
its entry point; it has no ROCm runtime. `docker/docker-bake.hcl` pins
`AIPERF_VERSION` to the release these scripts were run against, because the
flags and the record layout change between releases.

```bash
# What make agentx-build runs:
docker buildx bake -f docker/docker-bake.hcl agentx --load

# Raw build with the args bake resolves (the Dockerfile has no defaults):
mapfile -t args < <(docker/scripts/aic-bake-env.sh --build-args agentx)
docker buildx build "${args[@]}" \
  -f docker/agentx/Dockerfile \
  -t aic-agentx:latest \
  .
```
