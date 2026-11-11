# Accuracy gate — differential lm_eval

Answers one question: **does routing KV through DRAM/NVMe change the model's
answers?**

## What it asserts

| Test | Catches |
|---|---|
| `test_tiered_matches_baseline` | The connector or the KV store path changing the computation. Scores a VRAM-only arm and a tiered arm in the same job and asserts every answer is byte-identical. |
| `test_tiered_above_floor` | Both arms breaking identically, which the differential cannot see. Asserts `tiered >= expected.json[MODEL] - slack`. |
| `test_answers_survive_restart` | Retrieval corruption. Driver-only: vLLM is restarted, LMCache keeps its DRAM/NVMe state, the prompts are replayed, and every answer must match the baseline arm re-scored with its own prefix cache warm. |

Both same-run comparisons are per question and exact, so a gain is caught as
surely as a loss.

### Why the restart is compared with a cached re-score

When part of a prompt is already cached, vLLM computes only the rest, and how
much it computes decides which kernels run. That alone changes ~57% of the
answers: the baseline arm re-scored with its GPU prefix cache warm differs from
its own first pass on 746 of 1319 questions. So the restart pass, whose
prefixes come from LMCache, is compared with that cached re-score instead of
with a first pass. Both arms run with vLLM's block size equal to LMCache's
256-token chunk, so both caches reuse the same number of tokens per prompt.

The one exception is a prompt whose length is an exact multiple of 256. vLLM's
own cache must then leave the whole last block to compute, while the connector
loads all but the last token, so the two recompute different amounts. Those
prompts (7 of 1319 for the default model) are left out of the restart
comparison. On SPUR (job 282084) the other 1312 matched exactly, and 4 of the 7
differed.

## Running it

### Against an existing stack

The arms publish no host port, so `lm_eval` has to run on the `aic` bridge. The
baseline arm is the `vllm-only` compose service, the tiered arm `vllm`, which
sits in `aic-lmcache`'s network namespace; `aic_bridge_url` names the container
that answers for each:

```bash
source scripts/aic-endpoints.sh
BASELINE_URL="$(aic_bridge_url aic-vllm-only-gpu0 8000)"   # http://aic-vllm-only-gpu0:8000
TIERED_URL="$(aic_bridge_url aic-vllm-gpu0 8000)"          # http://aic-lmcache:8000
AIC_FETCH_IMAGE="${VLLM_IMAGE_REF}" aic_fetch "${TIERED_URL}/v1/models"
```

Run pytest from the vLLM image with the repo mounted, as the driver does
(`.slurm/run-accuracy.sh`, which also builds the `.venv-accuracy` it uses). With
one arm up, the tiered-only assertions; with both, everything:

```bash
docker run --rm --network aic --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e PYTHONNOUSERSITE=1 -e HF_HOME=/hf -v "${HF_HOME}:/hf" \
  -v "$PWD:$PWD" -w "$PWD" \
  -e AIC_ACCURACY_BASELINE_URL="${BASELINE_URL}/v1" \
  -e AIC_ACCURACY_TIERED_URL="${TIERED_URL}/v1" \
  --entrypoint "$PWD/.venv-accuracy/bin/python" "${VLLM_IMAGE_REF}" \
  -m pytest tests/accuracy -v
```

Drop `AIC_ACCURACY_BASELINE_URL` and add `-k floor` for a single tiered arm. `PYTHONNOUSERSITE=1` keeps a populated
`~/.local/lib/python3.*/site-packages` from shadowing the venv.

### Under the driver

```bash
make accuracy-test        # SPUR, both arms, full split (~27 min)
make accuracy-test-fast   # the same, AIC_ROCM_ARCH pinned to AIC_FAST_ARCH
```

Five phases: score the VRAM-only arm twice (cold, then with its prefix cache
warm) and confirm it never tiered, score the tiered arm and run the pytest
assertions, check the NVMe pool grew and LMCache was consulted, restart vLLM and
re-score, then verify that re-score came from cache rather than recompute.

There is one gate, not a fast one and a thorough one — `accuracy-test-fast`
differs only in the arch pin, exactly as `tiny-test-fast` differs from
`tiny-test`.

Each scored phase writes its number to `${AIC_LOG_DIR}` as
`baseline-score.json`, `tiered-score.json` and `restart-score.json`. CI lifts
all three out of the harvested log archive and reports them in the job summary,
and on the `/run-ci-accuracy` path in the PR comment. Nothing is compared across
runs — with no golden file, a run's own numbers are the only record it leaves.

Each pass also writes every question's answer as JSON lines —
`baseline-samples.jsonl`, `baseline-cached-samples.jsonl`,
`tiered-samples.jsonl`, `restart-samples.jsonl` — with the generated text, the
extracted answer, `exact_match`, the `prompt_hash` lm_eval records, and the
prompt's token count. When two arms disagree, these say on which questions and
where the text diverged.

## Skipping is a laptop default, not a CI one

The fixtures in `conftest.py` skip when an endpoint is missing, so the package
can be collected on a machine with no GPU. Under CI that is the wrong default: a
dead endpoint would report green having scored nothing. The driver sets
`AIC_ACCURACY_REQUIRED=1`, which turns every unreachability skip into a failure.
It is opt-in rather than read from `$CI` so the behavior is reproducible by hand.

The switch governs *reachability only*. Skips that encode "this assertion does
not apply to this run" stay skips — a model absent from `expected.json`, or
`AIC_ACCURACY_REFERENCE_SAMPLES` unset — because failing them would report a
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
| `AIC_ACCURACY_BASELINE_SAMPLES` | unset | Baseline answers from an earlier pass, as JSON lines; otherwise the baseline URL is scored. |
| `AIC_ACCURACY_REQUIRED` | unset | `1` = an unreachable endpoint fails instead of skipping. Set by the CI driver. |
| `AIC_ACCURACY_CONCURRENT` | `1` | `lm_eval` request concurrency. Above 1, answers depend on request timing. |
| `AIC_ACCURACY_BASELINE_URL` | unset | VRAM-only arm, e.g. `http://172.18.0.4:8000/v1`. |
| `AIC_ACCURACY_TIERED_URL` | unset | LMCache/NIXL arm. |
| `AIC_ACCURACY_REFERENCE_SAMPLES` | unset | Cached baseline re-score's answers, as JSON lines; enables the restart assertion. |
| `AIC_ACCURACY_BLOCK_SIZE` | `256` | vLLM `--block-size` the arms ran with; the driver sets it to LMCache's chunk size. |
| `AIC_ACCURACY_SCORE_OUT` | unset | Write the measured tiered score here as JSON. |
| `AIC_ACCURACY_SAMPLES_OUT` | unset | Write the tiered arm's per-question answers here as JSON lines. |

## Thresholds

**The same-run comparisons have no tolerance:** every compared answer must be
byte-identical. They used to compare scores within `DELTA = 0.02`, which was
described as ~6σ of baseline noise. That figure came from three runs of one
arm (σ = 0.0032) and left out the √2 for a difference of two arms. Measured
across nightly runs at 32 requests in flight, one arm moved by σ ≈ 0.007 and
the difference by σ ≈ 0.010, so 0.02 was ~2σ and failed healthy runs.

**Every phase scores the same 1319 items, and there is deliberately no item-cap
knob.** `FLOOR_RTOL = 0.05` is computed for the full set of
questions, and would have to be modified to account for smaller sample size
if we ever scored a subset of questions. For a short local run, pass `--limit`
to `lm_eval` in `_score()` on a scratch commit rather than reintroducing the
knob.

**Each scoring pass sends one request at a time**, ~4.5 min per pass on an
MI355X against ~1 min at 32 in flight. Concurrency is what made the arms
disagree: vLLM batches whatever requests have arrived, the batch shape selects
the attention and GEMM kernels, and greedy decoding turns a last-bit difference
into a different answer whenever two tokens are nearly tied. At 32 in flight two
identical baseline servers disagreed on 747 of 1319 answers; at 1 they agreed on
every one.

## Adding a model to `expected.json`

1. Serve it and score it: run the `docker run ... -m pytest` command above with
   `-e AIC_ACCURACY_MODEL=<id>` and `-k floor -s`.
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
