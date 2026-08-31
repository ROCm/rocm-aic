# Accuracy gate — differential lm_eval

Answers one question: **does routing KV through DRAM/NVMe change the model's
answers?**

Nothing else in the repo answers it. `tiny-test` proves the stack returns a
non-empty completion; `cliff` measures throughput. Neither would notice an L2
tier that returns subtly wrong KV blocks which still decode to plausible text.

## The two assertions

| Test | Catches |
|---|---|
| `test_tiered_matches_baseline` | KV corruption. Scores a VRAM-only arm and a tiered arm in the same job and asserts `|tiered - baseline| <= DELTA`. |
| `test_tiered_above_floor` | Both arms breaking identically — which the differential cannot see, because the difference stays zero. Asserts `tiered >= expected.json[MODEL] - slack`. |

`test_score_survives_restart` runs only in the Slurm driver's restart phase,
where vLLM is restarted but LMCache keeps its DRAM/NVMe state. The prompts are
replayed against a warm tier, so the blocks should be served from DRAM/NVMe
rather than recomputed; a score change means retrieval is corrupting blocks.
**The score alone cannot tell retrieval from recompute** — a restart that lost
the cache entirely would recompute the same answers and pass. Driver phase 5
closes that hole by asserting on the cache counters; see below.

### Why the differentials are two-sided

Both same-run comparisons (`test_tiered_matches_baseline`,
`test_score_survives_restart`) fail on an unexplained *gain* as well as a loss.
The arms answer identical questions with identical weights, so the only honest
outcome is the same score modulo batching nondeterminism. A tiered arm that
beats its baseline by more than `DELTA` has not got smarter — something about
the comparison changed (a different item count, a silently-skipped arm, a stale
supplied baseline, config drift between phases), and that is a broken gate
rather than a win.

The absolute floor stays one-sided. There `expected` is a coarse published
number, not a same-run measurement, so an overshoot carries no information.

### Why differential rather than a committed golden

An absolute table has to be re-edited on every model, vLLM, or attention-backend
bump. A same-run A/B is immune to that drift and is a sharper signal for the
thing AIC actually risks breaking. The absolute floor is kept only as the
second, independent check.

For the same reason there is no `reference.json` and no md5 manifest of LMCache
pool files. Byte-equality of pool slots across runs is not something LMCache
guarantees — which slot a block lands in depends on allocation order, which
depends on request scheduling — so a committed checksum would be a flake
generator. The driver asserts pool *liveness* instead (files exist, are
non-zero, grew during the run), which is what proves the tiered arm actually
tiered.

## Running it

### Against an existing stack

`lm_eval` is host-side, so you need the venv:

```bash
make venv
```

vLLM has no published ports — it is reachable only on the `aic` bridge
network. Resolve the container's bridge IP:

```bash
IP=$(docker inspect -f '{{(index .NetworkSettings.Networks "aic").IPAddress}}' aic-vllm-gpu0)
curl -fsS "http://$IP:8000/v1/models"      # confirm host -> bridge routing
```

Then, with one arm up, the tiered-only assertions:

```bash
PYTHONNOUSERSITE=1 AIC_ACCURACY_TIERED_URL="http://$IP:8000/v1" \
  .venv/bin/pytest tests/accuracy -v -k "floor or restart"
```

With both arms up, everything:

```bash
PYTHONNOUSERSITE=1 \
  AIC_ACCURACY_BASELINE_URL="http://$BASELINE_IP:8000/v1" \
  AIC_ACCURACY_TIERED_URL="http://$TIERED_IP:8000/v1" \
  .venv/bin/pytest tests/accuracy -v
```

Always invoke `.venv/bin/pytest` by absolute path with `PYTHONNOUSERSITE=1`:
on these boxes a populated `~/.local/lib/python3.*/site-packages` otherwise
shadows the venv.

Any endpoint that is unset or unreachable causes a clean `skip`, not an error —
so the package can be collected on a laptop with no GPU.

### Under the driver

```bash
make accuracy-test        # SPUR, both arms, full split (~65 min)
make accuracy-test-fast   # the same, AIC_ROCM_ARCH pinned to AIC_FAST_ARCH
```

The driver runs five phases: score the VRAM-only arm, score the tiered arm and
run the pytest assertions, check the NVMe pool grew, restart vLLM and re-score,
then verify that re-score was served from the cache rather than recomputed.

There is one gate, not a fast one and a thorough one. `accuracy-test-fast`
differs from `accuracy-test` only in the arch pin, exactly as `tiny-test-fast`
differs from `tiny-test`; every phase asks the full gsm8k split in both. The
cheap variant that used to exist dropped the baseline arm and capped items to
200, which skipped `test_tiered_matches_baseline` and widened the surviving
floor to 0.117 against an expected 0.2022 — a 13-minute job that could only
fail on catastrophe. The cap also bought ~85% of the saving on its own, so the
baseline arm was being sacrificed for about four minutes.

## Skipping is a laptop default, not a CI one

The fixtures in `conftest.py` skip when an endpoint is missing. That is what
lets the package be collected on a machine with no GPU, and lets a developer run
only the tiered-arm assertions against a local stack.

Under CI it is the wrong default: a dead endpoint would report green and the
gate would pass having scored nothing. The driver therefore sets
`AIC_ACCURACY_REQUIRED=1`, which turns every unreachability skip into a failure.
It is opt-in rather than auto-detected from `$CI` so the behaviour is
reproducible by hand off a runner.

The switch governs *reachability only*. Skips that encode a genuine "this
assertion does not apply to this run" stay skips, because failing them would
report a config gap as a regression:

- a model with no entry in `expected.json` — no floor to compare against.
- `AIC_ACCURACY_REFERENCE_SCORE` unset — not a restart-phase run.

## Proving the restart re-score came from cache

Phase 4 asserts the post-restart score matches. On its own that is nearly
vacuous: **if the restart lost the cache entirely, vLLM would recompute every
prompt and produce the very same answers.** Phase 3 does not cover it either —
it shows KV was *written* to the pool during phase 2, not that anything was
*read back* afterwards.

Phase 5 reads two independent counter sources, because each alone has a blind
spot:

| Source | Counter | Answers |
|---|---|---|
| vLLM | `vllm:external_prefix_cache_queries_total` / `_hits_total` | Did the engine ask the tier, and get blocks back? (Not *which* tier.) |
| LMCache | `lmcache_mp_l1_read_chunks_total` | Served from DRAM. |
| LMCache | `lmcache_mp_l2_prefetch_hit_chunks_total` / `_load_completed_chunks_total` | Served from NVMe. |

vLLM's counters are read after the restart, so its process is new and they start
at zero. LMCache is *not* restarted, so its counters carry phase-2 history and
are differenced across the re-score.

Failure conditions:

- **zero external queries** — the connector is not wired up post-restart; the
  re-score recomputed everything and phase 4 proved nothing.
- **queries but zero hits** — LMCache did not survive with usable state; the
  matching score is a recompute.

**Hits entirely from L1 (DRAM) is a warning, not a failure.** DRAM survives a
vLLM restart on its own, so a DRAM-only run leaves NVMe retrieval — the thing
this gate exists to check — untested. It is not fatal because the tier split
depends on the model and item cap, and failing CI on a legitimately DRAM-heavy
split would be wrong. With `LMCACHE_L1_SIZE_GB=1` against a pool that grew to
many GB the working set cannot fit in DRAM, so the warning firing is itself a
signal worth chasing. The counters are logged either way.

Both endpoints are scraped via `docker exec`; neither publishes a port to the
host (the convention across this repo).

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `AIC_ACCURACY_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Model to score. Must match what the arms are serving. |
| `AIC_ACCURACY_DELTA` | `0.02` | Allowed tiered-vs-baseline gap, **two-sided**. |
| `AIC_ACCURACY_REQUIRED` | unset | `1` = an unreachable endpoint fails instead of skipping. Set by the CI driver. |
| `AIC_ACCURACY_CONCURRENT` | `32` | `lm_eval` request concurrency. |
| `AIC_ACCURACY_BASELINE_URL` | unset | VRAM-only arm, e.g. `http://172.18.0.4:8000/v1`. |
| `AIC_ACCURACY_TIERED_URL` | unset | LMCache/NIXL arm. |
| `AIC_ACCURACY_REFERENCE_SCORE` | unset | Pre-restart score; enables the restart assertion. |
| `AIC_ACCURACY_SCORE_OUT` | unset | Write the measured tiered score here as JSON. |

There is deliberately no item-cap knob. Capping a pass forces every tolerance
that compares against it to widen by the binomial spread of the smaller sample
(see below), and each widening moves the gate closer to unfailable. If you need
a short local run while iterating, pass `--limit` to `lm_eval` in `_score()` on
a scratch commit rather than reintroducing the knob — a capped run is a smoke
test of the plumbing, not an accuracy result, and should not be able to
masquerade as one in CI.

## Choosing DELTA

`DELTA = 0.02` is **~6σ** of measured baseline noise. Derived, not guessed.

Three consecutive full-split gsm8k runs against one VRAM-only arm, same model,
same node, same weight load — SPUR job 6223 on `crsuse2-m2m-216`, 2026-08-12,
`Qwen/Qwen2.5-0.5B-Instruct`, 5-shot, `exact_match,strict-match`, all 1319 test
items:

| Run | Score |
|---|---|
| 1 | 0.19864 |
| 2 | 0.20470 |
| 3 | 0.20318 |

```
mean   0.2022
range  0.0061
sigma  0.00316   (sample stdev, n=3)
3-sigma 0.0095
```

So the scorer's run-to-run spread is about a third of `DELTA`. A tiered arm
would have to fall roughly six standard deviations below the baseline to trip
the gate, which is the margin we want: the assertion should fire on corruption,
not on noise.

Two notes for whoever revisits this:

- **The measured σ is much tighter than the binomial floor.** A mean of 0.2022
  over n=1319 has a sampling standard error of 0.0111 — 3.5× the observed σ.
  That is expected and not a contradiction: all three runs score the *same* 1319
  items, so the binomial term is common-mode and cancels. It would dominate the
  moment you compared across different item subsets — so **do not reuse this
  DELTA for a subsampled differential** without re-measuring. This is why the
  item cap is gone rather than merely unused by default: the restart phase used
  to re-score a 200-item prefix against a full-split reference, the binomial
  term did *not* cancel, and the tolerance had to widen to ±0.085 to avoid
  flaking — four times looser than the gate it replaced. Every phase now scores
  the same 1319 items, so `DELTA` applies unmodified everywhere and there are no
  tolerance-scaling helpers left to reason about.
- **The model scores ~0.20, lower than upstream's ~0.41 for `Qwen3-0.6B`.**
  That was the concern that motivated measuring: at a low score, run-to-run
  noise could have been a large fraction of DELTA. It is not. If a future change
  makes the spread comparable to DELTA, prefer moving to `Qwen3-0.6B` over
  widening DELTA — a gate that tolerates 0.05 of drift is barely a gate.

`VLM_MAX_MODEL_LEN=4096` was confirmed adequate in the same run: zero truncation
warnings in the vLLM log across all three passes.

Wall clock: three full-split scoring passes plus one bringup took 3m10s total on
a SPUR GPU node, so a single full-split VRAM-only pass is well under a minute
once the endpoint is up.

## What a full two-arm run looks like

SPUR job 6235, `crsuse2-m2m-006`, 2026-08-12 — the first end-to-end run of the
real driver. All four phases passed.

| Phase | Result | Wall clock |
|---|---|---|
| 1 — `vram_only` score | 0.20394 | ~40 s scoring |
| 2 — `kvd` score + assertions | 0.19257, 3 passed | 48 m 55 s |
| 3 — NVMe pool liveness | grew 15,489,564,672 B across 924 files | instant |
| 4 — restart + re-score | 2 passed | 1 m 07 s |

**The tiered arm scores 0.0114 below the baseline** — a pass, but it uses 57% of
the `DELTA=0.02` allowance and is 3.6σ of same-arm noise, so it is more than
sampling scatter. The most likely cause is benign and structural rather than
corruption: the tiered arm runs at `VLM_GPU_MEMORY_UTILIZATION=0.15` against the
baseline's 0.90, deliberately, to force eviction. That changes batching and
block reuse, and chunked prefill over reassembled KV is not bit-identical to a
single fresh prefill. What argues against corruption specifically: phase 4
re-scored within DELTA after a restart.

Note that this run predates phase 5, so **it did not verify that the
post-restart re-score was actually served from the tier** — a full recompute
would have produced the same result. Treat the phase 4 evidence above as weaker
than it reads until a run with phase 5 counters confirms it.

Worth keeping an eye on. If that gap widens toward DELTA, the diagnosis to run
first is a tiered arm at the *baseline's* memory utilisation — if the gap
disappears, it is the eviction pressure, not the tier.

**The tiered arm is ~70× slower to score** (48m55s vs ~40s). That is the number
the CI `timeout-minutes` is set from, and it is the single thing standing
between this gate and a runtime short enough to gate every PR on. If that
2.2 s/item comes down, everything below stops being a trade-off.

## Why there is no item cap

Capping the split is the obvious way to make this gate cheap, and it does not
work: the tolerances have to widen faster than the runtime falls, so what you
buy in minutes you pay for in a gate that cannot fail.

`FLOOR_RTOL = 0.05` is calibrated for the full 1319-item split, where the
binomial standard error is 0.011 — three sigma is 0.033, comfortably inside it.
Capping the item count inflates that error as `1/sqrt(n)`:

| Items | SE | 3×SE | Flat 0.05 floor |
|---|---|---|---|
| 1319 (full) | 0.011 | 0.033 | fine |
| 500 | 0.018 | 0.054 | marginal |
| 200 | 0.028 | 0.085 | **flakes** |
| 100 | 0.040 | 0.121 | **flakes badly** |

At the 200 items the old fast path used, a flat 0.05 floor would fail a
perfectly healthy run a large fraction of the time — sampling noise alone
exceeds the threshold. The fix at the time was to scale the slack, which turned
a *flaky* gate into a *weak* one: the floor landed at 0.117 against an expected
0.2022, so only catastrophe could trip it. Both options are bad; asking all
1319 questions is what makes the tight thresholds honest.

The same arithmetic governs the restart phase, which is why it is no longer
capped either. `DELTA` survives subsampling only when both numbers score the
*identical* items — a 200-item prefix compared against a full-split reference
does not cancel the binomial term, and needed ±0.085 to stay green.

## Adding a model to `expected.json`

1. Serve it and score it: `AIC_ACCURACY_MODEL=<id> .venv/bin/pytest tests/accuracy -k floor -s`.
   With the model absent from the table the test skips but the score is printed.
2. Add `"<id>": <score>` under `models`, and a line under `sources` saying where
   the number came from.
3. Do not guess a value. An absent model skips cleanly; a wrong floor either
   never fires or fires spuriously, and both are worse than no floor.

## Provenance

`test_accuracy.py` is adapted from vLLM's
`tests/v1/kv_connector/nixl_integration/test_accuracy.py` (Apache-2.0). The
lm_eval invocation and the one-sided per-model tolerance come from there; the
differential oracle does not. Both licence headers are retained in the file.
