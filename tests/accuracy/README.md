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
| `test_tiered_above_floor` | Both arms breaking identically — which the differential cannot see, because the difference stays zero. Asserts `tiered >= expected.json[MODEL] - 0.05`. |

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

<!-- TASK-3-PLACEHOLDER: replace with the measured spread. -->

**Not yet measured.** `DELTA=0.02` is a placeholder pending the baseline
variance measurement. The procedure: run the VRAM-only arm three times with the
same model and seed, take the standard deviation of the three strict-match
scores, and set `DELTA` to roughly `3σ`. If σ turns out comparable to 0.02 on
`Qwen2.5-0.5B-Instruct` — plausible, since it scores only ~0.3 on gsm8k, where
run-to-run noise is a larger fraction of the score — move to `Qwen3-0.6B`
instead of widening the gate into uselessness.

Record the measured numbers here when they exist. The next person to touch
`DELTA` needs to see what it was derived from.

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
