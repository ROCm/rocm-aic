# Emulation mode — serving without a GPU

AIC ships a **profile-driven emulation mode**: vLLM serves normally, but the
GPU forward pass is replaced by a latency drawn from a profile pack captured
earlier on real hardware. The scheduler, KV-cache accounting, prefix cache,
tokenizer, OpenAI API and metrics are all the stack's own unmodified code — only
the forward pass is faked, so the whole thing runs on a **CPU-only node**.

It is built on [llm-emu](https://github.com/AKafakA/llm-emu) (pinned in
`docker/Dockerfile` as `LLM_EMU_REF`), wired into vLLM by
`patches/vllm/03-llm-emu-emulator-hooks.patch`.

## What it is for

- Reproducing and debugging **scheduling / admission / KV-cache** behavior
  without occupying a GPU (including the KV-cache cliff itself).
- Running the serving stack in CI, on a laptop-class node, or on the dozens of
  CPU-only nodes a cluster usually has idle.
- Exploring "what if the GPU were 2x faster" style questions by editing a pack.

## What it is **not** for

- **Output quality.** The emulator emits a fixed filler token per step, so
  completions are meaningless text (usually an empty string after detokenizing).
  Assert on token counts, never on content.
- **Numbers for a configuration you did not capture.** A pack encodes one GPU,
  one model, and one serving configuration. Change `--max-num-batched-tokens`,
  enable async scheduling, or switch attention backend and the pack is stale.
- **Anything with `tensor-parallel-size > 1`** — only `UniProcExecutor` is
  hooked today.

| Layer | Real serve | Emulation |
| --- | --- | --- |
| HTTP / OpenAI API, metrics | vLLM | vLLM (unmodified) |
| Tokenizer, scheduler, KV-cache accounting, prefix cache | vLLM | vLLM (unmodified) |
| `UniProcExecutor.execute_model()` | worker → GPU kernels | timer future resolving after the profiled step latency |
| Model weights | loaded into VRAM | **never loaded** |
| Sampled tokens | real | fixed filler token id |

Because weights are never loaded, an 8B model costs a few MB of HuggingFace
download (config + tokenizer) and no parameter memory.

Everything is env-gated: with `VLLM_EMULATOR_ENABLE_ORACLE` unset the plugin's
entry point returns `None` and the patched vLLM paths are guarded, so a GPU
serve is unaffected. The plugin is present in both images.

---

## 1. Build

Two images can serve in emulation mode:

```bash
# Emulation-only image: ~10 min, NO GPU kernels compiled at all
make dist-build-emulate        # -> rocm-aic:7.14-emulate

# The normal production image also works (it ships the plugin)
make dist-build                # -> rocm-aic:7.14-latest
```

`make dist-build-emulate` builds the Dockerfile's `emulate` stage with
`--build-arg VLLM_TARGET_DEVICE=empty`:

- the `emulate` stage stops after vLLM + the plugin — no LMCache HIP extension,
  no NIXL/AIS_MT, no hsa-snoop;
- `VLLM_TARGET_DEVICE=empty` is upstream vLLM's no-extension build, so no HIP
  kernels are compiled — minutes instead of hours.

> [!WARNING]
> An `empty` image has no `vllm._C` and cannot run a real forward pass. Never
> deploy `:7.14-emulate` to a GPU node.

The image is tagged separately (`AIC_EMULATE_IMAGE`, default
`rocm-aic:7.14-emulate`) and gets its own tarball, so it can never overwrite the
GPU image. `ROCM_ARCH` still has to be a valid gfx (`AIC_EMULATE_ARCH`, default
`gfx942`) because torch's AMD wheel index picks its device extra from it —
nothing in the image runs on a GPU.

## 2. Run

On a Slurm CPU node, with assertions:

```bash
make emulate-test                        # emulation image
AIC_EMULATE_IMAGE=rocm-aic:7.14-latest \
  AIC_EMULATE_ARCH=gfx942 make emulate-test   # production image
```

`emulate-test` asserts more than "a 200 came back", because the dangerous
failure mode is a silent fall-through to real execution:

1. the container has **no** `/dev/kfd`;
2. `/v1/models` answers and a chat completion returns tokens
   (`usage.completion_tokens > 0`);
3. the log shows `[ExecutorEmulatorHook] Enabled` — the hook engaged;
4. the log shows `[ExecutorHook] step=` — steps came from the pack;
5. the log shows `load_model SKIPPED` — no weights were materialized.

By hand, on any host with docker:

```bash
IMAGE_NAME=rocm-aic:7.14-emulate \
  docker compose -f docker/docker-compose.yml --profile emulate up vllm-emulator
```

Then use it like any vLLM endpoint on `:8000`. Useful env:

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_MODEL` | `Qwen/Qwen3-8B` | model to serve; should match the pack |
| `VLLM_EMULATOR_PROFILE_PACK` | `/opt/llm-emu/profiles/MI300X-Qwen3-8B.json` | pack to replay |
| `EMU_PROFILE_PACK_HOST` | `/tmp` | host dir mounted at `/profiles` for packs not in the image |
| `VLLM_EMULATOR_MODE` | `realtime` | `realtime` sleeps for the predicted latency; `accelerated` resolves instantly |
| `VLLM_EMULATOR_ORACLE_K` | `1` | oracle neighbor count (`auto` for adaptive) |
| `VLLM_EMULATOR_MEMORY` | from pack | override the emulated device memory, bytes |
| `VLLM_EMULATOR_DEBUG` | — | `1` logs every oracle lookup |

> [!NOTE]
> Pass optional `VLLM_EMULATOR_*` variables only when you mean them. An **empty**
> value is not the same as unset: an empty `VLLM_EMULATOR_MEMORY` used to abort
> the plugin's `cuda_mock` import, leaving the emulator half-installed. The
> compose service therefore uses the pass-through form (`- VLLM_EMULATOR_MEMORY`).

## 3. Profile packs

A pack buckets measured step latencies by **(batch total tokens × concurrency)**,
split into prefill and decode, and stores the raw samples per cell. At serve
time the oracle looks up the cell for the batch the scheduler just built and
draws a latency from it.

Packs baked into the image, under `/opt/llm-emu/profiles/`:

| Pack | GPU | Model |
| --- | --- | --- |
| `MI300X-Qwen3-8B.json` | AMD Instinct MI300X (gfx942) | `Qwen/Qwen3-8B` |
| `A40-Q8-Qwen3-8B.json` | NVIDIA A40 (upstream example) | `Qwen/Qwen3-8B` |

Repository copies and their provenance files live in [`profiles/`](../profiles/).

## 4. Generating a pack for a new GPU

Emulation is only as good as its pack, and a pack is specific to
`(GPU, model, serving config)`. To add one — say for MI355X / gfx950:

### 4.1 Capture

```bash
AIC_ROCM_ARCH=gfx950 make dist-build          # image for that arch, if not built
AIC_ROCM_ARCH=gfx950 AIC_CAPTURE_CONSTRAINT=GFX950 \
  AIC_CAPTURE_MODEL=Qwen/Qwen3-8B \
  make profile-capture
```

This runs a **real serve on a real GPU** with `VLLM_EMULATOR_TRACE_STEP_CYCLE=1`
— a passive measurement; the emulator itself is *not* enabled — drives a
`vllm bench serve` sweep, and converts the resulting step trace into a pack.
Everything lands in `AIC_CAPTURE_DIR` (default
`/scratch/$USER/images/profiles/`):

```text
<Model>-<stamp>.json              the pack
<Model>-<stamp>.capture.txt       node, image, serve flags, sweep  (provenance)
step-trace-<Model>-<stamp>.jsonl  raw per-step samples
bench/real-<Model>-*.json         the real-hardware benchmark results
```

Knobs: `AIC_CAPTURE_MODEL`, `AIC_CAPTURE_NODE`, `AIC_CAPTURE_SWEEP`,
`AIC_CAPTURE_MAX_MODEL_LEN`, `AIC_CAPTURE_MAX_BATCHED_TOKENS`,
`AIC_CAPTURE_GPU_UTIL`, `AIC_CAPTURE_DIR`, `AIC_CAPTURE_HF_HOME`.

**Sweep design matters more than anything else.** `AIC_CAPTURE_SWEEP` is a list
of `input_len,output_len,concurrency,num_prompts` points. Rules of thumb:

- **Prefill cells come from prompts**, roughly one sample per prompt, so prefill
  coverage is driven by *how many prompts* you send and *how many distinct input
  lengths* you use. Short outputs (`osl=16`) maximise prompts per second.
- **Decode cells come from output tokens** at each concurrency level; a handful
  of points with `osl=128` across concurrency 1→64 fills them quickly.
- Sweep concurrency across the range you care about — the oracle interpolates,
  it does not extrapolate.
- The upstream A40 reference pack has ~46 k prefill and ~232 k decode samples.
  A first pass with a few hundred prefill samples will model decode well and
  TTFT badly.

> [!IMPORTANT]
> Both capture and validate default to `--no-enable-prefix-caching`
> (`AIC_CAPTURE_EXTRA_ARGS` / `AIC_VALIDATE_EXTRA_ARGS`). With prefix caching on,
> a sweep that revisits an input length serves the second pass out of the cache,
> so the trace mixes cold and warm prefills *and* the real-hardware reference
> numbers stop being comparable to a cold replay — we measured a 6x swing in
> reference TTFT from this alone. Capture the compute cost; let the emulator's
> own (unmodified) prefix cache decide what gets reused at serve time. Whatever
> you change here, change on both sides.

### 4.2 Validate — always

```bash
AIC_VALIDATE_PACK=/scratch/$USER/images/profiles/<pack>.json make emulate-validate
```

Replays the pack on a CPU-only node, re-runs the same benchmark points, and
prints real-vs-emulated deltas. From the shipped MI300X pack:

```text
point                             metric                 real     emulated   delta
Qwen-Qwen3-8B-isl1024-osl128-c1   mean_ttft_ms          53.98        44.79   -17.0%
Qwen-Qwen3-8B-isl1024-osl128-c1   mean_tpot_ms           5.15         5.82   +13.0%
Qwen-Qwen3-8B-isl1024-osl128-c16  mean_ttft_ms         328.64       330.77    +0.6%
Qwen-Qwen3-8B-isl1024-osl128-c16  mean_tpot_ms           8.62        13.42   +55.8%
Qwen-Qwen3-8B-isl4096-osl128-c8   mean_ttft_ms         810.35       488.90   -39.7%
Qwen-Qwen3-8B-isl4096-osl128-c8   mean_tpot_ms          12.00        10.31   -14.1%
```

The upstream paper reports ±1-2% on TPOT/ITL and ~±10% on TTFT for packs built
from hours of ShareGPT traffic (~46 k prefill and ~232 k decode samples). Our
20-minute synthetic sweep gets an order of magnitude fewer samples and lands
within ~15% on TPOT at low concurrency and ~±40% on TTFT, degrading to +50% on
TPOT at concurrency 16. Treat a fresh pack as **directionally useful, not
quantitatively trustworthy**, until you have validated it at the concurrencies
you care about.

Where the remaining error comes from, in rough order:

1. **Sample count.** Thin cells make the oracle's draw noisy. More prompts and
   more distinct input lengths is the first lever;
   `VLLM_EMULATOR_ORACLE_K=auto` (adaptive-K pooling) is the second, worth a few
   percent.
2. **The pack keys on `(scheduled tokens x concurrency)` only.** `sum_kv` — the
   attention depth of the batch — is recorded in the trace but not used for
   bucketing, so a decode step over 4 k-token contexts is modeled the same as
   one over 200-token contexts. This is the main gap for KV-cache-cliff work and
   shows up as TPOT error that grows with concurrency and context length.
3. **The emulator serializes steps** on a virtual GPU timeline, so it cannot
   reproduce whatever CPU/GPU overlap the real engine achieved.

### 4.3 Ship it

Copy the pack and its `.capture.txt` into [`profiles/`](../profiles/) named
`<GPU>-<Model>.json`, then rebuild. Both images `COPY profiles/*.json` into
`/opt/llm-emu/profiles/`, and the build fails if
`AIC_EMULATOR_DEFAULT_PACK` is missing.

---

## 5. Capturing on hardware you do not own

The trace side is just an env var on a normal serve, so a pack can be captured
anywhere the AIC image runs:

```bash
docker run --device /dev/kfd --device /dev/dri --network host \
  -e VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
  -e VLLM_EMULATOR_STEP_TRACE_OUTPUT=/trace/step-trace.jsonl \
  -v /some/host/dir:/trace \
  rocm-aic:7.14-latest --model <model> [normal serve flags]

# ... drive real traffic through it, then:
docker run --rm -v /some/host/dir:/trace --entrypoint python3 rocm-aic:7.14-latest \
  -m vllm_emulator.profile.build_serving_profile_filtered \
  /trace/step-trace.jsonl /trace/pack.json
```

The trace records only `(total_tokens, num_new_reqs, num_decode_seqs, sum_kv,
step_cycle_us)` per step plus a header of GPU/model metadata — no prompts, no
outputs, so a pack is safe to share when the traffic is not.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `AssertionError: ... out of bounds for 0 devices` | the plugin loaded but its torch shims did not — usually an invalid/empty `VLLM_EMULATOR_*` value aborting `cuda_mock` import |
| No `[ExecutorEmulatorHook] Enabled` in the log | pack path wrong or pack failed schema validation; the hook stays **disabled** and vLLM tries a real forward pass |
| Completion content is `""` | expected — filler token, see above |
| `Failed to import from vllm._C` warning | expected in the `empty` build; harmless because no kernel ever runs |
| Emulated TTFT far off, TPOT fine | pack under-samples prefill; widen `AIC_CAPTURE_SWEEP` |
