# Emulation mode — serving without a GPU

AIC ships a **profile-driven emulation mode**: vLLM serves normally, but the
GPU forward pass is replaced by a latency drawn from a profile pack captured
earlier on real hardware. The scheduler, KV-cache accounting, prefix cache,
tokenizer, OpenAI API and metrics are all the stack's own unmodified code — only
the forward pass is faked, so the whole thing runs on a **CPU-only node**.

It is built on [llm-emu](https://github.com/AKafakA/llm-emu) (pinned in
`docker/Dockerfile` as `LLM_EMU_REF`), wired into vLLM by
`patches/vllm/03-llm-emu-emulator-hooks.patch`.

The stack above the forward pass can include the **full LMCache MP recipe** —
standalone `lmcache server` plus `LMCacheMPConnector` — with the KV tensors the
connector registers synthesized in host memory. See [§6](#6-emulating-the-full-lmcache-mp-stack).

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
- **Structured output under async scheduling** — `pending_structured_output_tokens`
  defers sampling to the next step, which the hook's single pending future does
  not model. Everything else about async scheduling works; see the note below.

> [!NOTE]
> Async scheduling **is on by default** in v0.26.0
> (`SchedulerConfig.async_scheduling` defaults to `None` → auto → `True`), so the
> emulator runs the `step_with_batch_queue()` path unless you pass
> `--no-async-scheduling`. There, `model_executed` decides whether
> `sample_tokens()` is called, and upstream answers it with
> `total_num_scheduled_tokens > 0` — which is wrong for a **zero-token** step.
> The emulator still claims those: with a KV connector it has to poll for
> finished transfers, exactly as `GPUModelRunner.execute_model` calls
> `kv_connector_no_forward()` for the same batch. Marked not-executed, the step
> never reaches `sample_tokens()`, the hook's parked future is stranded, and the
> engine dies with `RuntimeError: unexpected error` on the first idle step after
> a request completes. `patches/vllm/03-llm-emu-emulator-hooks.patch` answers
> from the hook's own state instead, so the emulated services match the GPU one
> rather than having to opt out of async scheduling.

| Layer | Real serve | Emulation |
| --- | --- | --- |
| HTTP / OpenAI API, metrics | vLLM | vLLM (unmodified) |
| Tokenizer, scheduler, KV-cache accounting, prefix cache | vLLM | vLLM (unmodified) |
| `UniProcExecutor.execute_model()` | worker → GPU kernels | timer future resolving after the profiled step latency |
| Model weights | loaded into VRAM | **never loaded** |
| Sampled tokens | real | fixed filler token id |
| Paged KV tensors | allocated in VRAM | not allocated — unless a KV connector is attached, see §6 |
| KV connector (LMCache MP, NIXL) | real | real, against host-memory KV buffers (§6) |

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
| `VLLM_EMULATOR_KV_BUFFERS` | `1` | `0` skips the fake KV buffers, making a KV connector inert (§6) |
| `VLLM_EMULATOR_DEBUG` | — | `1` logs every oracle lookup |

> [!NOTE]
> Pass optional `VLLM_EMULATOR_*` variables only when you mean them. An **empty**
> value is not the same as unset: an empty `VLLM_EMULATOR_MEMORY` used to abort
> the plugin's `cuda_mock` import, leaving the emulator half-installed. The
> compose service therefore uses the pass-through form (`- VLLM_EMULATOR_MEMORY`).

### 2.1 In CI

Two GPU-free gates cover this path:

- **AIC Nightly Patch Validation** (`ubuntu-latest`, on every push/PR touching
  `docker/Dockerfile` or `patches/**`) checks that the vLLM, LMCache, NIXL and
  LLM-Emu patches all apply against their pinned upstream refs, byte-compiles
  the patched vLLM files, and validates every pack in `profiles/` through the
  emulator's own loader and oracle — including that the pack compose defaults to
  actually exists.
- **AIC Nightly Emulate Test** (self-hosted → cluster) builds the emulation
  image and serve-tests it on a CPU-only node. It is the only hardware-CI stage
  that needs no GPU, so it does not compete with the smoke/tiny/cliff chain.

`make emulate-mp-test` (§6) is deliberately not in that nightly: it needs the
production image, which needs a GPU-capable build node, so it belongs with the
smoke/tiny chain rather than the GPU-free lane.

## 3. Profile packs

A pack buckets measured step latencies by **(batch total tokens × concurrency ×
KV depth)**, split into prefill and decode, and stores the raw samples per cell.
At serve time the oracle looks up the cell for the batch the scheduler just
built and draws a latency from it.

The third axis, `sum_kv`, is the total KV depth of the batch. It matters more
than it looks: on MI300X/Qwen3-8B the deep-context quartile of a *single*
(tokens, concurrency) cell runs 1.13x-1.45x slower than the shallow quartile,
worst at high concurrency. Packs also carry the 2-axis tables for compatibility;
set `VLLM_EMULATOR_ORACLE_IGNORE_KV=1` to force the old behavior (useful when
comparing two packs). Build with `--kv-bucket-width 0` to omit the 3-axis tables.

A pack is **self-describing**: alongside the latency cells it records the GPU
(name, memory, CUs, and the KV pool vLLM actually measured) and the serving
configuration it was captured under (`max_num_batched_tokens`, `max_num_seqs`,
prefix caching, chunked prefill, `kv_cache_dtype`, async scheduling,
`gpu_memory_utilization`, vLLM version).

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
The trace is written to node-local disk and copied at the end: the tracer writes
from inside the engine loop, so a shared-filesystem stall would be recorded as a
fabricated multi-hundred-millisecond step.
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

Set `VLLM_EMULATOR_TRACE_STEP_CYCLE=1` on the *emulated* service (the compose
`emulate` profile passes it through) to record the replay's own steps. Diffing
that trace against the capture's — step counts, prefill/decode split, per-step
medians — localises a mismatch to scheduling, cost model, or reference noise in
one run. That is how the prefill lookup bug above was found.

```bash
AIC_VALIDATE_PACK=/scratch/$USER/images/profiles/<pack>.json make emulate-validate
```

Replays the pack on a CPU-only node, re-runs the same benchmark points, and
prints real-vs-emulated deltas. From the shipped MI300X pack:

```text
point                             metric                 real     emulated   delta
Qwen-Qwen3-8B-isl1024-osl128-c1   mean_ttft_ms          53.23        45.42   -14.7%
Qwen-Qwen3-8B-isl1024-osl128-c1   mean_tpot_ms           5.11         5.47    +7.0%
Qwen-Qwen3-8B-isl1024-osl128-c1   output_throughput    182.22       172.98    -5.1%
Qwen-Qwen3-8B-isl1024-osl128-c16  mean_ttft_ms         332.17       202.78   -39.0%
Qwen-Qwen3-8B-isl1024-osl128-c16  mean_tpot_ms           8.65        12.13   +40.2%
Qwen-Qwen3-8B-isl1024-osl128-c16  output_throughput   1428.31      1172.75   -17.9%
Qwen-Qwen3-8B-isl4096-osl128-c8   mean_ttft_ms         828.31       757.76    -8.5%
Qwen-Qwen3-8B-isl4096-osl128-c8   mean_tpot_ms          12.04        13.01    +8.1%
Qwen-Qwen3-8B-isl4096-osl128-c8   output_throughput    433.14       423.74    -2.2%
```

The upstream paper reports ±1-2% on TPOT/ITL and ~±10% on TTFT for packs built
from hours of ShareGPT traffic (~46 k prefill and ~232 k decode samples). Our
20-minute synthetic sweep gets an order of magnitude fewer samples and lands
within ~15% on every metric at concurrency 1 and 8, but is still ~40% out on
both TTFT and TPOT at concurrency 16 — it finishes queued prefills too fast and
decodes too slowly there. Treat a fresh pack as **directionally useful, not
quantitatively trustworthy**, until you have validated it at the concurrencies
you care about.

> [!IMPORTANT]
> **Read TPOT first.** Two runs of the *same* benchmark point on the *same*
> MI300X differed by +215% on TTFT and −34% on throughput (cold engine vs warm,
> on a shared node) while TPOT moved 1.9%. So TPOT is the metric worth tuning
> against; TTFT and throughput deltas smaller than about 35% are inside the
> reference's own run-to-run spread. Lead `AIC_CAPTURE_SWEEP` with a throwaway
> warm-up point, and repeat the reference before believing a TTFT delta.

Where the remaining error comes from, in rough order:

1. **Sample count.** Thin cells make the oracle's draw noisy. More prompts and
   more distinct input lengths is the first lever;
   `VLLM_EMULATOR_ORACLE_K=auto` (adaptive-K pooling) is the second, worth a few
   percent.
2. **The emulator serializes steps** on a virtual GPU timeline, so it cannot
   reproduce whatever CPU/GPU overlap the real engine achieved.
3. **Not scheduling.** Diffing an emulated step trace against a real one for the
   same point (see below) showed 400 steps against 400 and 20 prefill-bearing
   steps against 23 — the emulated engine builds the same batches, because it is
   the same scheduler. Per-step decode cost matched to 0.7%. If a replay is off,
   suspect the cost model or the reference, not the scheduler.
4. **Bucket resolution.** `--kv-bucket-width` (default 4096) and
   `--conc-bucket-width` (default 5) trade cell resolution against samples per
   cell. Narrow them only if the capture has the samples to fill them.

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

## 6. Emulating the full LMCache MP stack

Everything above fakes the *compute*. The `emulate-mp` compose profile keeps
that and adds the rest of the shipped stack back: a standalone `lmcache server`,
vLLM talking to it over ZMQ with `LMCacheMPConnector`, KV bytes really moving
into L1/L2 and back — on a node with no `/dev/kfd`. The point is to measure
LMCache/NIXL transfer cost *on top of* the profile-pack compute cost, which is
what makes the emulated cliff resemble the deployed one.

```bash
AIC_ROCM_ARCH=gfx942 make dist-build     # emulate-mp needs LMCache in the image
make emulate-mp-test
```

By hand:

```bash
IMAGE_NAME=rocm-aic:7.14-latest DEVICE_TYPE=cpu \
  VLLM_EMULATOR_MEMORY=$((8 * 1024**3)) \
  EMU_KV_TRANSFER_ARG="--kv-transfer-config '{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://localhost\",\"lmcache.mp.port\":6555,\"lmcache.mp.mp_transfer_mode\":\"lmcache_driven\"}}'" \
  docker compose -f docker/docker-compose.yml --profile emulate-mp \
  up lmcache-emulator vllm-emulator
```

> [!IMPORTANT]
> The connector arg is `EMU_KV_TRANSFER_ARG`, not `KV_TRANSFER_ARG`. The value is
> the identical JSON the GPU stack uses — this really is the same recipe — but
> the Makefile `export`s `KV_TRANSFER_ARG` at file scope (`Makefile:90`), so
> reusing that name attaches the MP connector to the **plain** `emulate` profile
> as well, and `make emulate-test` then hangs for 300 s waiting on an lmcache
> server that profile never starts.

> [!IMPORTANT]
> `emulate-mp` runs on the **production** image, not `:7.14-emulate`. The
> `emulate` Dockerfile stage stops before LMCache, so the connector import fails
> there. The production image ships the emulator plugin too, and the plugin is
> inert without `VLLM_EMULATOR_ENABLE_ORACLE=1`, so one image covers both.

### 6.1 How the VRAM gets faked

A KV connector is handed vLLM's paged KV tensors once, at startup, via
`register_kv_caches()`. Emulation never allocates them — and it cannot, on a
node with no VRAM.

But nothing about those buffers is device-specific *from the connector's point
of view*: they are paged byte buffers whose shape, dtype and byte count vLLM has
already computed and put in the `KVCacheConfig`. So `vllm_emulator` synthesizes
them in **host memory** from that config and registers those
(`patches/llm-emu/04-kv-connector-emulation.patch`). Their contents are zeros;
their volume is real, which is what makes the measured transfer cost real.

LMCache needs no persuading to accept them. Its device detection resolves to
`StubCPUDevice`, and `lmcache/v1/platform/cpu/shm.py` migrates a CPU tensor's
storage into a POSIX SHM segment the mp server can map — the same server-pull
topology as HIP IPC, with `/dev/shm` in place of the GPU. Two consequences:

- both emulated services need `ipc: host`, because Docker gives each container
  its own `/dev/shm`;
- `DEVICE_TYPE=cpu` is mandatory. `cuda_mock` shims `torch.cuda.is_available()`
  to `True` so vLLM's GPU code paths run, and LMCache's `CudaDeviceSpec.
  is_available()` is that same call — auto-detection would pick CUDA and then
  try to export device handles for host memory. Honouring that env var needs
  `patches/lmcache/10-lmcache-cpu-device-type-optin.patch`; upstream gates it on
  `CpuDeviceSpec.is_available()`, which is deliberately `False`.

The connector JSON also pins `"lmcache.mp.mp_transfer_mode":"lmcache_driven"`.
Left on `auto`, `create_transfer_context()` dispatches on `tensor.device.type`
and host-memory KV falls to `engine_driven`, which moves the copy into the vLLM
worker — a working configuration, but one that exercises less of the shipped
stack. Use it as a fallback if the SHM pull path misbehaves.

### 6.2 Sizing

Unlike the plain `emulate` profile, the KV pool here is genuinely allocated.
`torch.zeros` on CPU is lazily backed, so untouched blocks cost no resident
memory — but RSS grows toward the full pool as blocks get written, and a real
MI300X pack records ~100+ GiB of `available_kv_cache_bytes`. **Set
`VLLM_EMULATOR_MEMORY`** to something the node can hold; that also moves the
emulated admission cliff, which is usually what you want to control anyway.
`make emulate-mp-test` defaults to 8 GiB (`AIC_EMULATE_MP_KV_BYTES`).

`AIC_L2_BACKEND` defaults to **`nixl_posix`** for this profile. The usual AIS_MT
NVMe adapter is hipFile P2PDMA and cannot initialise without `/dev/kfd`, but
NIXL's POSIX plugin uses a CPU staging buffer and needs no GPUDirect — so NIXL
itself stays in the path, and with it the telemetry exporter on `:19090`.
`local_disk` (LMCache's native `LocalDiskBackend`) is the NIXL-free fallback and
`none` gives a DRAM-only L1.

### 6.3 Metrics

`make emulate-mp-test` brings up the Prometheus sidecar
(`monitoring/docker-compose.monitoring.yml`, exporters off — a CPU node has no
GPU/HSA/NVMe telemetry worth collecting) and asserts that the `lmcache`,
`nixl` and `vllm` scrape jobs are not just **up** but actually **serving
series**. A target that answers on its port while exporting nothing is the
failure this checks for; both facts land in the job log, and the TSDB is kept
under `logs/<job-id>/prometheus/`.

| Job | Port | Present when |
| --- | --- | --- |
| `vllm` | 8000 | always |
| `lmcache` | 8080 | always — the MP server's own HTTP API. **Not** `--prometheus-port` (9091): v0.5.2 logs `standalone metrics HTTP server disabled; /metrics must be exposed by the caller` and opens no listener there, so the old 9091-only scrape config collected nothing. Both are listed now. |
| `nixl` | 19090 | only with a `nixl_store` L2 — i.e. `AIC_L2_BACKEND=nixl_posix`. With `local_disk` no NIXL agent is created, so the target is legitimately down and the check reports SKIP. |

Set `AIC_EMULATE_MP_MONITORING=0` to skip the sidecar entirely.

### 6.4 What this does and does not model

Real: batch composition, admission, prefix-cache and LMCache hit accounting, the
number and size of chunks crossing ZMQ/SHM, L1 eviction, L2 writes and reads,
and the wall-clock those take.

Not real:

- **The bytes.** Every KV block is zeros. Do not draw conclusions about
  compression, dedup, or anything content-dependent.
- **Store timing.** The connector cycle runs synchronously on the engine thread,
  where vLLM runs it, rather than inside the timer that models emulated compute.
  A store is therefore submitted up to one step earlier than a real run would
  submit it. Fine for hit rate, transfer volume and throughput; wrong for tight
  per-step store-latency attribution.
- **GPU-side copy cost.** There is no `hipMemcpy`, no P2PDMA, no GPUDirect. What
  is measured is the host-side and storage-side cost of the same transfer.

`VLLM_EMULATOR_TRACE_STEP_CYCLE=1` on both a connector-free and an `emulate-mp`
run of the same benchmark point, diffed, isolates the LMCache/NIXL cost: the
step counts should match, and the wall-clock difference is what the connector
added.

---

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `AssertionError: ... out of bounds for 0 devices` | the plugin loaded but its torch shims did not — usually an invalid/empty `VLLM_EMULATOR_*` value aborting `cuda_mock` import |
| No `[ExecutorEmulatorHook] Enabled` in the log | pack path wrong or pack failed schema validation; the hook stays **disabled** and vLLM tries a real forward pass |
| Completion content is `""` | expected — filler token, see above |
| `Failed to import from vllm._C` warning | expected in the `empty` build; harmless because no kernel ever runs |
| Emulated TTFT far off, TPOT fine | pack under-samples prefill; widen `AIC_CAPTURE_SWEEP` |
| `ModuleNotFoundError: lmcache` under `emulate-mp` | you are on `:7.14-emulate`; that stage stops before LMCache — use the production image (§6) |
| `emulate-mp` hangs with requests in `WAITING_FOR_REMOTE_KVS` | the connector's worker side is not reporting completions; check for `[EmulatorKVBuffers] registered` in the log |
| `Creating transfer context (device_type=cuda, ...)` on a CPU node | `DEVICE_TYPE=cpu` was not honoured — LMCache patch 10 missing, or the var is empty rather than `cpu` (§6.1) |
| Emulator RSS climbing to tens of GiB under `emulate-mp` | expected — the fake KV pool is really allocated; cap it with `VLLM_EMULATOR_MEMORY` (§6.2) |
| `FileExistsError: [Errno 17] File exists: '/lmcache_kv_<pid>_0'` | stale POSIX SHM segments from a killed previous run. `ipc: host` puts them in the **host's** `/dev/shm` and LMCache names them after the containerised engine pid, which repeats. `rm -f /dev/shm/lmcache_kv_*` with no emulator container running; `emulate-mp-test` does this at both ends. |
| `ConnectionError: Cannot reach the LMCache MP server` from the **plain** `emulate` profile | something exported `KV_TRANSFER_ARG` (the Makefile does, globally). The emulator reads `EMU_KV_TRANSFER_ARG` instead precisely so this cannot happen — check you are not on an older compose file. |
