# Accuracy gate — differential lm_eval

Answers one question: **does routing KV through DRAM/NVMe change the model's
answers?**

Nothing else in the repo answers it. `tiny-test` proves the stack returns a
non-empty completion; `cliff` measures throughput. Neither would notice an L2
tier that returns subtly wrong KV blocks which still decode to plausible text.

## The two assertions

| Test | Catches |
|---|---|
| `test_tiered_matches_baseline` | KV corruption. Scores a VRAM-only arm and a tiered arm in the same job and asserts `tiered >= baseline - DELTA`. |
| `test_tiered_above_floor` | Both arms breaking identically — which the differential cannot see, because the difference stays zero. Asserts `tiered >= expected.json[MODEL] - slack`. |

`test_score_survives_restart` runs only in the Slurm driver's restart phase,
where vLLM is restarted but LMCache keeps its DRAM/NVMe state. Every prompt is
then a guaranteed cache hit, so the whole score is served from the tier; a drop
means retrieval from NVMe is corrupting blocks.

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
AIC_LOCAL=1 AIC_ACCURACY_LIMIT=20 make accuracy-test   # local, no Slurm
make accuracy-test                                     # SPUR, both arms
make accuracy-test-fast                                # SPUR, PR path
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `AIC_ACCURACY_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Model to score. Must match what the arms are serving. |
| `AIC_ACCURACY_LIMIT` | unset (full split) | Cap gsm8k items. Lowers wall clock and resolution together — the score's standard error grows as `1/sqrt(LIMIT)`. |
| `AIC_ACCURACY_DELTA` | `0.02` | Allowed tiered-below-baseline gap. |
| `AIC_ACCURACY_CONCURRENT` | `32` | `lm_eval` request concurrency. |
| `AIC_ACCURACY_BASELINE_URL` | unset | VRAM-only arm, e.g. `http://172.18.0.4:8000/v1`. |
| `AIC_ACCURACY_TIERED_URL` | unset | LMCache/NIXL arm. |
| `AIC_ACCURACY_REFERENCE_SCORE` | unset | Pre-restart score; enables the restart assertion. |
| `AIC_ACCURACY_SCORE_OUT` | unset | Write the measured tiered score here as JSON. |

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
  items, so the binomial term is common-mode and cancels. It would dominate if
  you ever compared across different item subsets, which is exactly what
  `AIC_ACCURACY_LIMIT` does — so **do not reuse this DELTA for a
  limit-restricted differential** without re-measuring. The restart phase is
  safe because it re-scores the same capped prefix.
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
re-scored within DELTA after a restart, when every block came back from NVMe.

Worth keeping an eye on. If that gap widens toward DELTA, the diagnosis to run
first is a tiered arm at the *baseline's* memory utilisation — if the gap
disappears, it is the eviction pressure, not the tier.

**The tiered arm is ~70× slower to score** (48m55s vs ~40s). That is the number
the CI `timeout-minutes` is set from, and the reason the PR path caps items
rather than running the full split.

## Why the floor widens when you cap the item count

The floor's slack is `max(0.05, 3 * SE(limit))`, not a flat 0.05.

`FLOOR_RTOL = 0.05` is calibrated for the full 1319-item split, where the
binomial standard error is 0.011 — three sigma is 0.033, comfortably inside it.
Capping the item count inflates that error as `1/sqrt(n)`:

| Items | SE | 3×SE | Flat 0.05 floor |
|---|---|---|---|
| 1319 (full) | 0.011 | 0.033 | fine |
| 500 | 0.018 | 0.054 | marginal |
| 200 (fast path) | 0.028 | 0.085 | **flakes** |
| 100 | 0.040 | 0.121 | **flakes badly** |

At the fast path's `LIMIT=200`, a flat 0.05 floor would fail a perfectly healthy
run a large fraction of the time — sampling noise alone exceeds the threshold.
Scaling the slack makes a capped run a *weaker* gate rather than a *flaky* one,
which is the right trade for a PR-path check. The nightly runs the full split,
where the floor stays tight.

This is the same reason `DELTA` must not be reused for a limit-restricted
differential: there the two arms score the *same* capped subset, so the sampling
term cancels — but only if both arms use the identical prefix.

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
