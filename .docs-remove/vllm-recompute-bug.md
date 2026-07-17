# vLLM bug: `kv_load_failure_policy=recompute` crashes the EngineCore under `--async-scheduling`

**TL;DR** — With `kv_load_failure_policy=recompute` **and** `async_scheduling=True`,
a KV-connector load failure crashes the vLLM v1 EngineCore with an
`AssertionError` (`num_output_placeholders >= 0`) → `EngineDeadError`, killing the
server. It surfaced in the cliff `kvd_v2 nvme` arm at **c ≥ 80** (job `67535846`),
where the 16 GB DRAM L1 / NIXL staging pool exhausts and KV loads start failing.
Do **not** use `AIC_KV_LOAD_FAILURE_POLICY=recompute` with async scheduling on
vLLM v0.25.0.

## Symptom

In the nvme arm at c ≥ 80, every request failed instantly (wall ≈ 0.2 s, `ok=0`,
`BW=0`). The container log shows a fatal EngineCore error:

```
EngineCore encountered a fatal error.
Traceback (most recent call last):
  .../vllm/v1/engine/core.py, run_busy_loop -> _process_engine_step -> step_with_batch_queue
  .../vllm/v1/core/sched/scheduler.py:1630, update_from_output -> _update_request_with_output
  .../vllm/v1/core/sched/async_scheduler.py:68, _update_request_with_output
    request.num_output_placeholders -= len(new_token_ids)
    assert request.num_output_placeholders >= 0
AssertionError
-> vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.
```

vLLM then shuts the server down; all subsequent requests get connection errors.

## Root cause: async placeholder accounting is not reconciled on recompute-rewind

**What `num_output_placeholders` is.** With `--async-scheduling`, vLLM schedules the
*next* step before the current step's outputs return, and reserves slots for the
not-yet-returned tokens. In `async_scheduler.py`:

- `_update_after_schedule` **increments**:
  `request.num_output_placeholders += num_sampled_tokens_per_step + cur_num_spec_tokens`
- `_update_request_with_output` **decrements then asserts**:
  `request.num_output_placeholders -= len(new_token_ids)` ; `assert ... >= 0`

So it is a running count of "optimistically-scheduled but not-yet-received tokens";
it must never go negative.

**How recompute breaks it.** With `kv_load_failure_policy=recompute` the scheduler
sets `recompute_kv_load_failures=True`; on a failed KV load it calls
`_handle_invalid_blocks` which (per the code comment) *"adjust[s] their computed
token count to trigger recomputation of the invalid blocks"* — i.e. it **rewinds
`num_computed_tokens`**. That rewind path does **not** reconcile the async
scheduler's `num_output_placeholders`. The request still carries placeholders
reserved before the failure; when its stale in-flight output frame returns, the
`-= len(new_token_ids)` drives `num_output_placeholders` **negative** → the
assertion trips → fatal crash.

**The asymmetry that proves it.** vLLM already reconciles the analogous case for a
force-preempt in `reset_prefix_cache`: it sets `async_tokens_to_discard`, and
`_update_request_with_output` drains those stale frames early ("*The request was
force-preempted in reset_prefix_cache; drop one*"). The **KV-load-failure recompute
path has no equivalent drain/reconcile** — that missing reconciliation is the bug.

## Trigger conditions (all three required)

1. `kv_load_failure_policy=recompute`, and
2. `async_scheduling=True` (the cliff passes `--async-scheduling`), and
3. an actual KV-connector **load failure** — which in our setup only happens at
   high concurrency (c ≥ 80) when the **16 GB DRAM L1 / NIXL staging pool exhausts**.

Below c ≤ 64 no load fails, so the recompute path never runs → clean, fast curve
(that run's c ≤ 64 was the best yet: ~78–84 k tok/s, 0 errors). The bug is latent
and only exposed by high-concurrency + an undersized cache tier.

## vLLM version

`V1 LLM engine (v0.25.0)` ( image). File paths above are from that
version's `vllm/v1/core/sched/{async_scheduler.py,scheduler.py}`.

## Workarounds / fixes

- **Don't pair `recompute` with async scheduling** (current guidance). Leave
  `AIC_KV_LOAD_FAILURE_POLICY` unset (vLLM default `fail`, which degrades to
  per-request 500s instead of crashing).
- **Avoid triggering load failures at all** — the robust path: size the cache tiers
  so KV loads don't fail (large DRAM L1 that holds the working set, and/or pure
  NVMe). See the planned larger-pool run in `overnight-cliff-report.md`.
- **To use `recompute` gracefully**, drop `--async-scheduling` (no placeholders to
  underflow) — untested here; a cheap experiment (would need an `AIC_ASYNC_SCHEDULING`
  knob in `run-cliff.sbatch`).
- **Upstream fix:** on the KV-load-failure recompute rewind, reconcile the async
  scheduler's `num_output_placeholders` (e.g. set `async_tokens_to_discard` for the
  affected request, mirroring the `reset_prefix_cache` path). Worth filing against
  vLLM.

## Evidence

- Crash: `logs/67535846/container-aai-cliff-kvd-vllm.log` (EngineCore traceback at
  18:42:58; `async_scheduler.py:68`).
- Clean-vs-crash cliff data + power: `docs/overnight-cliff-report.md`
  ("Final analysis - job 67535846").
- Source read from vLLM v0.25.0 `async_scheduler.py` / `scheduler.py`.
