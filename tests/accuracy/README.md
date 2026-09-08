# Accuracy gate — differential lm_eval

Answers one question: **does routing KV through DRAM/NVMe change the model's
answers?**

## What it asserts

| Test | Catches |
|---|---|
| `test_tiered_matches_baseline` | KV corruption. Scores a VRAM-only arm and a tiered arm in the same job and asserts `\|tiered - baseline\| <= DELTA`. |
| `test_tiered_above_floor` | Both arms breaking identically, which the differential cannot see. Asserts `tiered >= expected.json[MODEL] - slack`. |
| `test_score_survives_restart` | Retrieval corruption. Driver-only: vLLM is restarted, LMCache keeps its DRAM/NVMe state, the prompts are replayed. |

Both same-run comparisons to the baseline accuracy are **two-sided** to ensure
that unexpected increased or decreases in accuracy are caught.

## Running it

### Against an existing stack

`lm_eval` is host-side, so you need the venv (`make venv`). vLLM publishes no
ports — it is reachable only on the `aic` bridge:

```bash
IP=$(docker inspect -f '{{(index .NetworkSettings.Networks "aic").IPAddress}}' aic-vllm-gpu0)
curl -fsS "http://$IP:8000/v1/models"      # confirm host -> bridge routing
```

With one arm up, the tiered-only assertions; with both, everything:

```bash
PYTHONNOUSERSITE=1 AIC_ACCURACY_TIERED_URL="http://$IP:8000/v1" \
  .venv/bin/pytest tests/accuracy -v -k "floor or restart"

PYTHONNOUSERSITE=1 \
  AIC_ACCURACY_BASELINE_URL="http://$BASELINE_IP:8000/v1" \
  AIC_ACCURACY_TIERED_URL="http://$TIERED_IP:8000/v1" \
  .venv/bin/pytest tests/accuracy -v
```

Always invoke `.venv/bin/pytest` by absolute path with `PYTHONNOUSERSITE=1`: on
these boxes a populated `~/.local/lib/python3.*/site-packages` otherwise shadows
the venv.

### Under the driver

```bash
make accuracy-test        # SPUR, both arms, full split (~65 min)
make accuracy-test-fast   # the same, AIC_ROCM_ARCH pinned to AIC_FAST_ARCH
```

Five phases: score the VRAM-only arm and confirm it never tiered, score the
tiered arm and run the pytest assertions, check the NVMe pool grew and the
scored pass read back through the tier, restart vLLM and re-score, then verify
that re-score came from cache rather than recompute.

There is one gate, not a fast one and a thorough one — `accuracy-test-fast`
differs only in the arch pin, exactly as `tiny-test-fast` differs from
`tiny-test`.

Each scored phase writes its number to `${AIC_LOG_DIR}` as
`baseline-score.json`, `tiered-score.json` and `restart-score.json`. CI lifts
all three out of the harvested log archive and reports them in the job summary,
and on the `/run-ci-accuracy` path in the PR comment. Nothing is compared across
runs — with no golden file, a run's own numbers are the only record it leaves.

## Skipping is a laptop default, not a CI one

The fixtures in `conftest.py` skip when an endpoint is missing, so the package
can be collected on a machine with no GPU. Under CI that is the wrong default: a
dead endpoint would report green having scored nothing. The driver sets
`AIC_ACCURACY_REQUIRED=1`, which turns every unreachability skip into a failure.
It is opt-in rather than read from `$CI` so the behaviour is reproducible by hand.

The switch governs *reachability only*. Skips that encode "this assertion does
not apply to this run" stay skips — a model absent from `expected.json`, or
`AIC_ACCURACY_REFERENCE_SCORE` unset — because failing them would report a
config gap as a regression.

## What the driver proves beyond the score

The score alone is nearly vacuous in two places, so the driver asserts on
Prometheus counters, scraped container-locally via `docker exec` (neither
endpoint publishes a port).

- **The baseline arm really is AIC-free.** A compose regression that left the
  connector wired would make the gate compare AIC against AIC — invisible in the
  scores. Phase 1 requires no `aic-lmcache` container and
  `vllm:external_prefix_cache_queries_total` absent or zero.
- **The scored pass read back through the tier.** Pool growth proves KV went
  *out*; phase 3 differences `lmcache_mp_lookup_requested_tokens_total` and
  `..._hit_tokens_total` across the scored pass and requires both nonzero.
- **The restart re-score was not a recompute.** If the restart lost the cache
  entirely, vLLM would recompute every prompt and produce the same answers.
  Phase 5 requires nonzero vLLM external queries *and* hits, and rejects hits
  that came entirely from L1 — DRAM survives a vLLM restart on its own, so a
  DRAM-only run leaves NVMe retrieval untested.
  `AIC_ACCURACY_ALLOW_L1_ONLY=1` downgrades that to a warning.

**A failed scrape is not a zero counter.** `_metric_sum` cannot distinguish "the
endpoint did not answer" from "the counter never moved" — both are `0` — so
every gate first checks `_metrics_reachable` and fails with *metrics
unavailable* rather than drawing a conclusion about the product. Both scrapes
are dumped to `${AIC_LOG_DIR}/{lmcache,vllm}-metrics-phase3.prom`; these are
LMCache-internal names and an upgrade is free to rename them.

Degradation counters (`lmcache_mp_l1_allocation_failure_chunks_total`,
`lmcache_mp_event_bus_dropped_events_total`) are warned on, not gated, and
`AIC_ACCURACY_MIN_HIT_PCT` is unset, for the same reason: their normal value
under this gate's starved L1 has not been measured, and a guessed threshold
either never fires or fires spuriously. Every run logs them.

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

## Thresholds

`DELTA = 0.02` is ~6σ of measured baseline noise, not a guess.

**Every phase scores the same 1319 items, and there is deliberately no item-cap
knob.** `DELTA` and `FLOOR_RTOL = 0.05` are computed for the full set of
questions, and would have to be modified to account for smaller sample size
if we ever scored a subset of questions. For a short local run, pass `--limit`
to `lm_eval` in `_score()` on a scratch commit rather than reintroducing the
knob.

**The tiered arm is ~70× slower to score** (48m55s vs ~40s) — that is what
`timeout-minutes` is set from, and the single thing standing between this gate
and a runtime short enough to gate every PR on. This is primarily caused be the
artificially limited VRAM of the device that we set in order to force AIC to be
exercised.

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
