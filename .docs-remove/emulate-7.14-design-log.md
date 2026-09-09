# LLM-Emu emulation mode on the ROCm 7.14 AIC stack

Working log for wiring [llm-emu](https://github.com/AKafakA/llm-emu) (commit
`3e87da6`) into the AIC image so vLLM can serve on **CPU-only** nodes, with the
GPU forward pass replaced by a latency draw from an offline profile pack.

Status: **working and verified on the cluster** — the emulation image and the
production GPU image both serve on a CPU-only node with the forward pass faked.
See the [run log](#5-run-log).

---

## 1. What emulation mode actually does

| Layer | Real serve | Emulation |
| --- | --- | --- |
| HTTP / OpenAI API | vLLM | vLLM (unmodified) |
| Tokenizer, scheduler, KV-cache accounting, prefix cache | vLLM | vLLM (unmodified) |
| `UniProcExecutor.execute_model()` | worker → GPU kernels | `threading.Timer` future resolving after the profiled step latency |
| Model weights | loaded to VRAM | **never loaded** (`load_model` / `profile_run` stubbed) |
| Sampled tokens | real | fixed filler token id from the pack |

Consequences worth remembering:

* Output **text is meaningless** — the emulator emits one filler token per step.
  Emulation reproduces *timing and scheduling behavior*, not model output.
* An 8B model costs a few MB of HF download (config + tokenizer) and no
  parameter memory, so a CPU node with modest RAM is enough.
* Activation is entirely env-gated (`VLLM_EMULATOR_ENABLE_ORACLE=1`,
  `VLLM_EMULATOR_EXECUTOR_HOOK=1`). With them unset the plugin's
  `vllm.platform_plugins` entry point returns `None` and the patched vLLM code
  paths are guarded by `self._emulator_enabled`, so a GPU serve is unchanged.

Two directions of travel:

* **Replay** (`VLLM_EMULATOR_ENABLE_ORACLE=1`) — serve from a profile pack.
* **Capture** (`VLLM_EMULATOR_TRACE_STEP_CYCLE=1`) — record one
  `(total_tokens, concurrency, step_us)` sample per step on *real* hardware.
  This is how an MI300X/MI355X pack gets made; the shipped example pack is an
  NVIDIA A40 / Qwen3-8B capture and is **not** an AMD latency model.

---

## 2. Review of commit b353693 — findings

### 2.1 `patches/vllm/03-llm-emu-emulator-hooks.patch` was corrupt and stale — FIXED

`git apply --check` failed outright:

```text
error: corrupt patch at .../03-llm-emu-emulator-hooks.patch:67
```

Hunk line counts did not match their `@@` headers. Beyond that, the context was
v0.18.x/v0.25-era and would not have applied to v0.26.0 even with correct
counts:

* `self.scheduler.schedule()` → v0.26.0 takes an argument
  (`self.scheduler.schedule(self._should_throttle_prefills())`)
* `self.log_iteration_details(...)` → renamed to
  `self.capture_iteration_details(...)` (and now yields `iteration_details`)

Since the Dockerfile hard-fails on a patch that does not apply, `make
dist-build` on b353693 could never have produced an image. The commit was
never built.

**Fix:** regenerated the patch from a real `v0.26.0` checkout (edit the files,
`git diff`), and verified all three vLLM patches apply in sequence on a
pristine tree. Deliberate changes vs the ported version:

* Both `step()` and `step_with_batch_queue()` guards are gated on
  `self._emulator_enabled`, so real serving is bit-identical rather than
  merely "probably equivalent".
* Dropped the `run_busy_loop()` `[LoopTiming]` printf instrumentation and the
  cosmetic `scheduler.py` blank-line hunk — paper-benchmark scaffolding with no
  emulator function, and pure rebase risk on the next vLLM bump.
* Dropped the fine-grained `sched_ms`/`exec_ms`/`sample_ms` extra trace fields:
  `profile/build_serving_profile_filtered.py` only consumes `total_tokens`,
  `num_new_reqs`, `num_decode_seqs`, `sum_kv` and `step_cycle_us`.
* The step-cycle recording moved into a named `_record_emulator_step()` helper
  instead of being inlined in `_process_engine_step()`.
* `ImportError` on the hook/tracer now logs a warning instead of `pass` — a
  silent fallback to real execution is exactly the failure mode this feature
  must not have.

### 2.2 `make dist-build` did build the emulator in — but into the wrong place

The Dockerfile step was correct in substance (clone at a pinned ref, install as
a `vllm.platform_plugins` entry point, `--no-deps` to dodge the upstream
`vllm==0.18.1` pin) and `make dist-build` picks it up automatically, since
dist-build just runs `docker build` on `docker/Dockerfile`.

Improvements made:

* The install now **verifies its own result** — it imports every subpackage and
  asserts the example profile pack landed. `llm-emu`'s `pyproject.toml` lists
  only `packages = ["vllm_emulator"]`, i.e. no subpackages; they survive purely
  by the `package-data = ["**/*.py"]` catch-all. Verified working today
  (`hooks/`, `oracle/`, `profile/`, `profiler/` all install), but it is one
  upstream packaging tweak away from silently shipping a plugin whose hook
  cannot be imported.
* The upstream example pack is copied to a stable path
  (`/opt/llm-emu/profiles/A40-Q8-Qwen3-8B.json`) so `--profile emulate` works
  with no external file.

### 2.3 The compose service pointed at a profile pack that does not exist

`VLLM_EMULATOR_PROFILE_PACK` defaulted to `/profiles/profile.json`, from a
`${EMU_PROFILE_PACK_HOST:-/tmp}` mount — i.e. the default configuration cannot
work. `load_profile_pack()` raises `FileNotFoundError`, the hook catches it,
prints, and stays **disabled**, and vLLM then attempts a real forward pass.

Fixed: default to the pack baked into the image; the `/profiles` mount remains
for real MI300X/MI355X captures.

Also changed on that service: `network_mode: host` (consistent with the other
services, no bridge/NAT dependency), a `--gpu-memory-utilization` flag (the
emulator's `determine_available_memory` reads it to size the KV pool), and a
log mount.

### 2.4 No emulation-only build or test path existed

Addressed in §3.

---

## 3. Emulation-only ("no GFX") build

`docker/Dockerfile` was split into linear stages — no layer contents changed,
so caching for existing builds is unaffected:

```text
base     ROCm base + system deps + uv                (sections 0-2)
vllm     torch + source-built vLLM + llm-emu         (sections 3, 3b)
emulate  emulation-only runtime image                (section 3c)
build    + LMCache, NIXL, hsa-snoop, runtime env     (sections 4-7)
wheels   scratch image with just /wheels             (section 8)
runtime  default target, aliases `build`             (section 9)
```

Two independent savings:

1. `--target emulate` stops after vLLM + the plugin: no LMCache HIP extension,
   no NIXL/AIS_MT, no hsa-snoop.
2. `--build-arg VLLM_TARGET_DEVICE=empty` is upstream vLLM's no-extension build
   (`setup.py`: `if _no_device(): ext_modules = []`) — **no GPU kernels are
   compiled at all**. Safe here only because the emulator intercepts
   `execute_model()` before any kernel would run;
   `Platform.import_kernels()` tolerates a missing `vllm._C` with a warning.
   Never ship an `empty` image to a GPU node.

Driver plumbing:

| Command | Effect |
| --- | --- |
| `make dist-build-emulate` | builds the emulation image on a Slurm CPU node, saves its own tarball |
| `make emulate-test` | serves it on a CPU-only node and asserts the emulator produced the output |
| `make dist-build` | unchanged: full GPU image, now with the plugin baked in |

The emulation image is tagged separately (`AIC_EMULATE_IMAGE`, default
`rocm-aic:7.14-emulate`) so its tarball never overwrites the GPU image's.
`ROCM_ARCH` still has to be a valid gfx (`AIC_EMULATE_ARCH`, default `gfx942`)
because torch's AMD wheel index selects its device extra from it — nothing in
the image runs on a GPU.

`emulate-test` asserts more than "a 200 came back", because the dangerous
failure mode is a silent fall-through to real execution:

1. `/v1/models` comes up and a chat completion returns non-empty content;
2. the log contains `[ExecutorEmulatorHook] Enabled` (hook engaged);
3. the log contains `[ExecutorHook] step=` (steps served from the pack);
4. `load_model SKIPPED` (weights never materialized) — warning only.

---

## 4. Bugs found by actually running it

Everything below was found on the cluster, not by reading code.

### 4.1 `AttributeError: 'float' object has no attribute 'language_model'`

`llm-emu`'s stubbed `gpu_worker.Worker.compile_or_warm_up_model` returns `0.0`.
vLLM v0.26.0's `Executor.initialize_from_config` reads
`max(t.language_model for t in compilation_times)` — it wants a
`CompilationTimes` NamedTuple. Engine core dies during KV-cache init.

Fixed in `patches/llm-emu/01-vllm-026-compat.patch` (returns `CompilationTimes`
when the type exists, else the old float). The Dockerfile now applies
`patches/llm-emu/*.patch` the same way it applies the vLLM/LMCache/NIXL ones.

### 4.2 An empty `VLLM_EMULATOR_MEMORY` silently half-disables the emulator

The first CPU-node run failed with:

```text
AssertionError: DP adjusted local rank 0 is out of bounds for 0 devices.
  (vllm/v1/worker/gpu_worker.py:352, torch.accelerator.device_count() == 0)
```

Root cause chain:

1. compose passed `VLLM_EMULATOR_MEMORY=${VLLM_EMULATOR_MEMORY:-}` → the var is
   present but **empty** in the container;
2. `cuda_mock.py` does `int(os.environ.get("VLLM_EMULATOR_MEMORY", 12*1024**3))`
   at module import → `ValueError` on `""`;
3. `vllm_emulator/__init__.py` imports `cuda_mock` inside
   `try: ... except Exception: pass` → the exception vanishes;
4. the platform plugin still activates and reports device `cuda`, but **no**
   `torch.cuda` shims and **no** model stubs are installed;
5. the real gpu_worker runs and finds 0 devices.

Two fixes, deliberately both: compose no longer materializes the variable when
it is unset (`- VLLM_EMULATOR_MEMORY` pass-through form, not `VAR=${VAR:-}`),
and the patch above makes `cuda_mock` ignore a non-numeric value instead of
aborting its own import.

This is also why `emulate-test` asserts on log markers: a half-installed
emulator does not announce itself.

### 4.3 `timeout 30 compose ...` never ran compose (pre-existing)

`compose` is a shell function in these scripts; `timeout` execs a *program*, so
those calls ran `/usr/bin/compose` from mailcap and wrote

```text
Error: no "compose" mailcap rules found for type "application/octet-stream"
```

into the service log file instead of the service logs. Present in `tiny-test`'s
cleanup as well — fixed in both by spelling out `docker compose -f ...`.

### 4.4 The completion comes back with **empty** `content` — and that is correct

```json
"message":{"role":"assistant","content":""},"finish_reason":"length"
"usage":{"prompt_tokens":15,"total_tokens":31,"completion_tokens":16}
```

16 tokens were generated and went through vLLM's real sampler → detokenizer →
HTTP path. They are 16 copies of the emulator's filler token id `100`, which in
Qwen3's byte-level BPE vocab is the single byte-token `'§'`; a run of those
decodes to nothing printable, so `content` is `""`.

So the test asserts on `usage.completion_tokens > 0`, not on the text. Asserting
on the text would be asserting on nonsense: **the emulator models timing, not
language**. Anything that consumes emulator output (a benchmark harness, a
correctness check) has to be told the same thing.

---

## 5. Run log

| When | What | Result |
| --- | --- | --- |
| build #1 (job 67642825) | `make dist-build-emulate`, node `ctr-smc-s22-025` | **OK**, 10 min wall, 8.8 G tarball. All three vLLM patches applied; `vllm-0.26.0+empty` wheel built in ~3 s (no kernels); subpackage import check passed |
| test #1 (job 67642865) | `make emulate-test` | **FAIL** — §4.2 (0 devices) |
| debug run | manual `docker run` of the same image | got past device init; **FAIL** — §4.1 (`language_model`) |
| build #2 (job 67642939) | rebuild with `patches/llm-emu/` applied | **OK**, ~5 min (torch layer cached) |
| test #2 (job 67642965) | `make emulate-test` | engine **served**; hook, no-weights and per-step checks all passed. Only the "non-empty content" check failed — see §4.4 |
| test #3 (job 67642996) | `make emulate-test`, assertion corrected | **ALL CHECKS PASSED** on `ctr-smc-s22-025` |
| test #4 (job 67643020) | `make emulate-test AIC_EMULATE_TEST_NODE=mlse-alola-b38-ws7` | **ALL CHECKS PASSED** on a second, unrelated CPU node (fresh image load) |
| build #3 (job 67643009) | `AIC_ROCM_ARCH=gfx942 make dist-build` — the **full GPU image** on the restructured Dockerfile | **OK**, ~22 min, 9.9 G tarball: `vllm-0.26.0+rocm`, llm-emu patched + installed, 9 LMCache patches, NIXL AIS_MT, hsa-snoop |
| test #5 | `AIC_EMULATE_IMAGE=rocm-aic:7.14-latest AIC_EMULATE_ARCH=gfx942 make emulate-test` | **ALL CHECKS PASSED** — the *production* image (`vllm-0.26.0+rocm`) also serves in emulation mode with no GPU |
| capture #1 (job 67653552) | `make profile-capture` on `ctr-rack31-mi300x-2` (MI300X) | **OK** — 9-point sweep, 6 600 samples; TPOT ±7.5%, TTFT +68% at c=16 |
| capture #2 (job 67654472) | 28-point sweep, prefix caching left on | **discarded** — warm-cache reference made the comparison meaningless (§6.1) |
| capture #3 (job 67654956) | 28-point sweep, `--no-enable-prefix-caching` | **shipped** — 13 400 samples → `profiles/MI300X-Qwen3-8B.json` |
| validate (jobs 67655003/67655406) | `make emulate-validate`, K=1 and K=auto | TTFT +0.6% at c=16, TPOT +13%/-14% at c=1/c=8, +56% at c=16 (§6.2) |
| rebuild + test (jobs 67655401/…) | emulate image with the pack baked in, then `make emulate-test` | **ALL CHECKS PASSED** — first step now draws 11.7 ms (MI300X) instead of 58.5 ms (A40) |

Representative passing output:

```text
[emulate-test] /dev/kfd inside the container: no
[emulate-test] OK: the emulator container has no GPU device at all
[emulate-test] OK: emulated engine generated 16 tokens (decoded text is meaningless by design -- filler token)
[emulate-test] OK: executor hook active -> [ExecutorEmulatorHook] Enabled: mode=realtime, filler_id=100, stops=[2]
[emulate-test] OK: model weights were never loaded (emulator stub)
[emulate-test] OK: steps served from the profile pack -> [ExecutorHook] step=1 tt=15 reqs=1 new=1 latency=58.5ms
[emulate-test] ALL CHECKS PASSED
```

Engine start-up on a CPU node is ~35 s (no weights, no warm-up, no compile).

---

## 6. The AMD profile pack (MI300X / gfx942)

Captured on `ctr-rack31-mi300x-2` (MI300X, gfx942), Qwen3-8B, with the **full
production image** (`rocm-aic:7.14-latest`, `vllm-0.26.0+rocm`) — a real serve,
real weights, real kernels, `VLLM_EMULATOR_TRACE_STEP_CYCLE=1` passively
recording one sample per engine step. Shipped in the repo as
[`profiles/MI300X-Qwen3-8B.json`](profiles/MI300X-Qwen3-8B.json) and baked into
both images at `/opt/llm-emu/profiles/`; it is now the compose default.

```bash
AIC_ROCM_ARCH=gfx942 AIC_CAPTURE_NODE=<mi300x-node> \
  AIC_CAPTURE_SWEEP="<28 points>" make profile-capture
```

Serve config baked into the numbers (also in `MI300X-Qwen3-8B.capture.txt`):
`--max-model-len 8192 --max-num-batched-tokens 4096 --gpu-memory-utilization
0.85 --attention-backend TRITON_ATTN --no-enable-prefix-caching`,
`VLLM_ROCM_USE_AITER=1`, no async scheduling, `kv-cache-dtype auto`. Sweep: 28
`vllm bench serve` points, ISL ∈ {128, 512, 1024, 2048, 3072, 4096, 6144} ×
concurrency ∈ {1, 4, 8, 16, 32, 64}.

```text
gpu_model : AMD Instinct MI300X   (192 GiB, 304 CUs, arch reported as CC 9.4)
model     : Qwen/Qwen3-8B         (36 layers, 8 KV heads, head_dim 128, block 16)
cells     : 272 (tt x concurrency), 13 400 samples
            prefill 194 cells /  2 835 samples
            decode   83 cells / 10 565 samples
```

Representative measured medians (step-cycle = schedule + forward + output
processing, not just kernel time):

| batch | median step |
| --- | --- |
| decode, 1 token, concurrency 1 | 5.15 ms |
| decode, 8 tokens, concurrency ~7 | 7.07 ms |
| decode, 16 tokens, concurrency ~17 | 7.25 ms |
| decode, 64 tokens, concurrency ~62 | 11.68 ms |
| prefill, 4096-token chunk, concurrency ~12 | 164 ms |

### 6.1 Three captures, and why the first two were wrong

| # | Sweep | Prefill samples | Outcome |
| --- | --- | --- | --- |
| 1 | 9 points, 420 prompts | 182 | decode good (TPOT ±7.5%), TTFT +68% at c=16 — prefill cells too thin |
| 2 | 28 points, ~7 600 prompts | 2 530 | **worse** (TTFT +368%) — see below |
| 3 | 28 points, prefix caching **off** | 2 835 | shipped |

Capture #2 is the interesting failure. Denser sweep, worse result — because
**vLLM's prefix cache was on**. `vllm bench serve --seed 0` generates the same
prompts every run, so once the sweep revisited an input length, the real serve
answered out of the prefix cache: reference TTFT at isl=4096/c8 collapsed from
494 ms (capture #1) to 81 ms, and the emulated replay — starting cold — was
compared against it. Half the "error" was in the reference, not the model.

Capture #3 therefore disables prefix caching on **both** sides
(`AIC_CAPTURE_EXTRA_ARGS` / `AIC_VALIDATE_EXTRA_ARGS` default to
`--no-enable-prefix-caching`) and varies the benchmark seed per point. That
measures the compute cost cleanly and leaves reuse to the emulator's own
unmodified prefix cache at serve time.

### 6.2 Does it reproduce the machine? (`make emulate-validate`)

Replaying the shipped pack on a **CPU-only** node, same benchmark points, both
sides cold:

```text
point                             metric                 real     emulated   delta
Qwen-Qwen3-8B-isl1024-osl128-c1   mean_ttft_ms          53.98        44.79   -17.0%
Qwen-Qwen3-8B-isl1024-osl128-c1   mean_tpot_ms           5.15         5.82   +13.0%
Qwen-Qwen3-8B-isl1024-osl128-c1   output_throughput    180.62       163.20    -9.6%
Qwen-Qwen3-8B-isl1024-osl128-c16  mean_ttft_ms         328.64       330.77    +0.6%
Qwen-Qwen3-8B-isl1024-osl128-c16  mean_tpot_ms           8.62        13.42   +55.8%
Qwen-Qwen3-8B-isl1024-osl128-c16  output_throughput   1436.02      1004.51   -30.0%
Qwen-Qwen3-8B-isl4096-osl128-c8   mean_ttft_ms         810.35       488.90   -39.7%
Qwen-Qwen3-8B-isl4096-osl128-c8   mean_tpot_ms          12.00        10.31   -14.1%
Qwen-Qwen3-8B-isl4096-osl128-c8   output_throughput    437.47       567.74   +29.8%
```

The prefill fix worked where it was aimed: **TTFT at concurrency 16 went from
+68% (capture #1) to +0.6%**. TPOT is within ~15% at concurrency 1 and 8 but
+56% at 16, and throughput is 10-30% out. (§6.3 then halves most of this.)

`VLLM_EMULATOR_ORACLE_K=auto` (adaptive-K pooling, `AIC_VALIDATE_ORACLE_K=auto`)
was tried and moves TPOT at c=16 from +56% to +48% and TTFT to −2.4% — real but
not decisive, so the default stays `K=1`.

What is left, in rough order of size:

1. **Sample count.** 2 835 prefill / 10 565 decode samples against the upstream
   A40 reference pack's 46 k / 232 k. More prompts and more distinct input
   lengths is the first lever.
2. **The pack buckets on `(scheduled tokens × concurrency)` only.** `sum_kv` —
   the attention depth of the batch — is recorded in the trace but not used for
   bucketing, so a decode step over 4 k-token contexts is modeled identically to
   one over 200-token contexts. For AIC specifically (KV-cache cliff work) this
   is the modeling gap that matters most, and it is the likeliest explanation
   for TPOT error that grows with concurrency.
3. **The emulator serializes steps** on a virtual GPU timeline
   (`_gpu_free_time`), so it cannot reproduce the real engine's CPU/GPU overlap.

Verdict: usable for scheduling, admission and cliff-shape work at low
concurrency; **not** a substitute for a real benchmark, and not yet trustworthy
for absolute throughput at high concurrency.

### 6.3 Round two: fixing the trace capture itself

The 2-axis pack topped out around ±20-50%. Three fixes to the capture path, all
in `patches/llm-emu/`:

**`02-trace-capture-fidelity.patch` — record what the emulator needs.**
The trace header captured GPU dimensions and model shape but not the thing
`cuda_mock` explicitly prefers: `available_kv_cache_bytes`, the KV pool vLLM
*measured* on the real GPU. Without it the emulator estimates the pool from
total memory minus guessed weights, and the KV-admission point — the whole
reason to emulate a cache cliff — lands somewhere else. The header now records
it (passed from the patched engine core, which has it after
`_initialize_kv_caches`), plus `num_gpu_blocks`, `gpu_memory_utilization`,
`max_num_batched_tokens`, `max_num_seqs`, prefix caching, chunked prefill,
`kv_cache_dtype`, `async_scheduling`, `eos_token_id` and the vLLM version. A
pack is only valid for the configuration it was captured under, so that
configuration now lives *in* the pack instead of a sidecar file.

That immediately surfaced something the sidecar had wrong: **async scheduling
was on** for every capture (it is vLLM v0.26's default), while the provenance
file I hand-wrote claimed it was off.

Same patch: the tracer keeps one append handle open instead of
open()/write/close every 200 records, and registers `close()` with `atexit`.
The old behavior dropped up to 199 trailing records and, on a shared filesystem,
stalled the engine loop — the stall landing in the *next* step's measurement.
That is where the 128 ms "decode" steps and one 1.2 s sample in a 113-token cell
came from. `profile-capture` now also writes the trace to node-local disk and
copies it at the end.

**`03-kv-aware-oracle.patch` — bucket on KV depth.**
`sum_kv` was already in every trace record and already passed to
`estimate_step_latency_us()`; the builder and oracle both ignored it. Measured
inside a *single* (tokens, concurrency) cell of our own capture:

| cell | low-`sum_kv` quartile | high-`sum_kv` quartile | ratio |
| --- | --- | --- | --- |
| tt=1, conc=2 | 5 021 µs | 5 666 µs | 1.13x |
| tt=16, conc=17 | 6 494 µs | 7 339 µs | 1.13x |
| tt=8, conc=7 | 6 344 µs | 7 989 µs | 1.26x |
| tt=64, conc=62 | 8 233 µs | 11 925 µs | 1.45x |

So a decode step over 4 k-token contexts was modeled identically to one over
200-token contexts, and the error grew exactly where emulation gets used: high
concurrency, deep context. The builder now also emits
`{prefill,decode,step_cycle}_3d_distribution` keyed by `(tt, conc, kv)`
(`--kv-bucket-width`, default 4096) and the oracle prefers them, falling back to
the 2-axis tables for older packs or when `VLLM_EMULATOR_ORACLE_IGNORE_KV=1`.
The 2-axis tables are still emitted, so an unpatched oracle is unaffected.

Back-test, replaying all 13 400 captured steps through the oracle and comparing
each prediction against that step's measured latency:

| lookup | MAE | median abs err | prefill MAE |
| --- | --- | --- | --- |
| 2D `(tt, conc)` | 31.1% | 4.2% | 22.2% |
| 3D `(tt, conc, kv)` | 25.7% | **1.1%** | **12.6%** |

End to end, on the shipped pack:

| point | metric | 2-axis pack | KV-aware pack |
| --- | --- | --- | --- |
| isl1024 c1 | TTFT / TPOT / throughput | −17.0% / +13.0% / −9.6% | **−14.7% / +7.0% / −5.1%** |
| isl1024 c16 | TTFT / TPOT / throughput | +0.6% / +55.8% / −30.0% | −39.0% / **+40.2% / −17.9%** |
| isl4096 c8 | TTFT / TPOT / throughput | −39.7% / −14.1% / +29.8% | **−8.5% / +8.1% / −2.2%** |

Concurrency 1 and 8 are now within 15% on every metric, and c=8 within 8.5%.
Concurrency 16 is still ~40% out on both TTFT and TPOT — and interestingly in
*opposite* directions: the emulator finishes queued prefills too fast and then
decodes too slowly. That points at step composition at that operating point
(how prefill chunks and decodes get mixed into a batch) rather than at the
per-step cost model, which is the next thing to chase.

Mean absolute delta across the nine metrics: 23.3% → 15.9%.

### 6.4 A self-inflicted one worth writing down

Two hours were lost to two mistakes with the same root cause — bash reads a
script lazily, and heredocs expand:

* **Editing `run-build-distribute.sh` while a job driver was running it.** Bash
  resumed reading at a byte offset that no longer meant what it did, and the
  script died with a syntax error in the middle of a build.
* **Backticks in comments inside an unquoted `<<REMOTE` heredoc.** ``# ... `grep
  -q` exits on the first match ...`` executed `grep -q` and `docker logs` *on
  the submitting host* at script-generation time. Harmless here, but only by
  luck. All heredoc backticks are now escaped.

### 6.5 Round three: what the step traces actually say

The concurrency-16 gap was chased with the tool the previous round built: run the
*emulated* replay with `VLLM_EMULATOR_TRACE_STEP_CYCLE=1` (now a pass-through on
the compose service) and diff its step trace against a real capture of the same
single benchmark point. Both traces, isl1024 / osl128 / c=16 / 64 prompts:

| | steps | prefill-bearing | decode-only | median prefill step | median decode step |
| --- | --- | --- | --- | --- | --- |
| real | 400 | 20 | 380 | 96.8 ms | 7.23 ms |
| emulated | 400 | 23 | 377 | **34.3 ms** | 7.28 ms |

Three things fell out of that.

**1. Scheduling fidelity is not the problem.** 400 steps against 400, 20
prefill-bearing against 23. The emulated engine builds the same batches as the
real one — which it should, since it *is* the same scheduler, but now that is
measured rather than assumed.

**2. Decode cost is essentially exact** — 7.28 ms against 7.23 ms, 0.7%.

**3. Prefill cost was 3x too cheap, and it was my bug.** The KV-aware lookup
added in §6.3 used one Euclidean distance over range-normalised
`(tt, conc, kv)`. Since `sum_kv` spans ~75 k and `tt` spans ~4 k, a *kv* mismatch
could outvote a *tt* mismatch — so a query for a 4096-token prefill at
`sum_kv≈17k` was answered by a low-token cell whose KV depth happened to match.
The pack held the right answer all along: cell `(tt=4096, conc=17)` has a median
of 168.7 ms across 558 samples.

Fixed by making the KV axis *refine* rather than *replace*: pick the token
bucket exactly as the 2-axis path does, then concurrency within it, then the
nearest KV depth within that cell, pooling outwards along the KV axis only when
adaptive-K needs more samples. Replaying the real trace's own queries through
the fixed oracle gives a prefill median of 99.7 ms against 96.8 ms measured.

Also fixed: the builder now drops the first 25 records (`--warmup-skip`). The
real trace contains a 2.3-second "prefill" step — engine warm-up — which would
otherwise sit in a bucket the oracle draws from uniformly.

### 6.6 The reference moves more than the emulator does

Then the same benchmark point was run twice on real MI300X hardware, and this
came out:

| metric | sweep run | dedicated run | spread |
| --- | --- | --- | --- |
| mean TTFT | 332.17 ms | 1047.89 ms | **+215%** |
| mean TPOT | 8.65 ms | 8.81 ms | +1.9% |
| output throughput | 1428 tok/s | 944 tok/s | **−34%** |
| duration | 5.74 s | 8.68 s | +51% |

Identical benchmark, identical flags, same GPU model, same image. The dedicated
run paid engine warm-up on its first requests; the sweep run reached that point
warm, 27 points in. These nodes are also shared — 176 of 384 CPUs were busy.

That reframes every accuracy number in this document:

* **TPOT is a trustworthy comparison metric** (1.9% run-to-run). The emulator's
  +37% TPOT at concurrency 16 is therefore real signal and still unexplained.
* **TTFT and throughput are not**, unless the reference is measured under the
  same warm/cold conditions. Our −31% TTFT and −17% throughput at c=16 sit
  *inside* the reference's own spread. Tuning against them would be tuning
  against noise.

So `AIC_CAPTURE_SWEEP` should lead with a throwaway warm-up point (the long
capture now does), and `emulate-validate` deltas should be read as: TPOT is the
verdict, TTFT and throughput are indicative until the reference is repeated.

## 7. How to use it

```bash
# CPU-only emulation image (minutes, no GPU kernels compiled)
make dist-build-emulate                 # -> rocm-aic:7.14-emulate + its own tarball
make emulate-test                       # serve + assert on a Slurm CPU node

# Same thing against the production GPU image
make dist-build                         # unchanged; now ships the plugin too
AIC_EMULATE_IMAGE=rocm-aic:7.14-latest AIC_EMULATE_ARCH=gfx942 make emulate-test

# By hand on any node with docker
IMAGE_NAME=rocm-aic:7.14-emulate \
  docker compose -f docker/docker-compose.yml --profile emulate up vllm-emulator
```

Useful knobs (all optional): `AIC_EMULATE_MODEL`, `AIC_EMULATE_PROFILE_PACK`,
`AIC_EMULATE_TEST_NODE`, `AIC_EMULATE_ARCH`, `AIC_EMULATE_VLLM_DEVICE`
(`empty` → no kernels, `rocm` → a normal single-arch vLLM build in the same
lean image), plus the `VLLM_EMULATOR_*` vars the compose service passes through.

Capture an AMD pack, then replay and score it (§6):

```bash
AIC_ROCM_ARCH=gfx942 AIC_CAPTURE_CONSTRAINT='MARKHAM&GFX942' make profile-capture
AIC_VALIDATE_PACK=/scratch/$USER/images/profiles/<pack>.json make emulate-validate
```

To serve with a real AMD capture instead of the baked-in A40 pack:

```bash
EMU_PROFILE_PACK_HOST=/path/to/packs \
VLLM_EMULATOR_PROFILE_PACK=/profiles/MI300X-Qwen3-8B.json \
  docker compose -f docker/docker-compose.yml --profile emulate up vllm-emulator
```

---

## 8. Caveats and open items

1. **Concurrency 16 is still ~40% out on TTFT and TPOT** (§6.3), in opposite
   directions — prefills finish too fast, decode runs too slow. Everything at
   concurrency 1 and 8 is within 15%. Suspicion is step composition (how the
   scheduler mixes prefill chunks with decodes at that operating point) rather
   than the per-step cost model; the next probe is to compare the emulated and
   real *step histograms* from a traced replay, not just the end metrics.
2. **One pack, one model, one config.** `Qwen/Qwen3-8B` at
   `--max-num-batched-tokens 4096`, prefix caching off, no async scheduling. The
   AIC production compose uses `--async-scheduling` and `--kv-cache-dtype fp8`,
   so a pack matching *that* configuration is a separate capture.
3. **No gfx950 (MI355X) pack yet** — same command with
   `AIC_ROCM_ARCH=gfx950 AIC_CAPTURE_CONSTRAINT=GFX950` (note `GFX942` alone
   also matches MI300A, which is a different machine — pin the node), after
   building a gfx950 image.
4. **Emulator output text is meaningless** (§4.4). Any harness pointed at an
   emulated endpoint must not assert on content.
5. **`VLLM_TARGET_DEVICE=empty` images must never reach a GPU node.** They have
   no kernels: `vllm._C` is absent (logged as a warning at startup) and a real
   forward pass would fail. They are tagged `:7.14-emulate` to keep them
   distinguishable, and `make dist-build` is untouched.
6. **The emulation image is still ~8.8 G compressed**, because it inherits the
   `rocm/dev-ubuntu-24.04:7.14.0-full` base and the full ROCm torch wheels. If
   emulation images get distributed widely, rebasing that stage onto a runtime
   ROCm image (or CPU torch) is where the next order of magnitude is.
7. **CI wiring — done.** `aic-patches.yml` gained vLLM and llm-emu patch jobs
   (the hole that let the corrupt patch through) plus profile-pack validation,
   all on `ubuntu-latest`; `aic-amd-nightly-emulate-test.yml` builds and
   serve-tests the emulation image on the cluster without touching a GPU. The
   new runner script needs `make install-ci-scripts` deployed once.
8. **Only `UniProcExecutor` is hooked** (TP=1). A `tensor-parallel-size > 1`
   emulation run would fall straight through to the real executor; the hook
   would have to be added to `MultiProcExecutor` for that.

