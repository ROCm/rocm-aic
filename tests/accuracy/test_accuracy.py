# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""AIC KV-integrity accuracy gate — differential lm_eval over two serving arms.

Adapted from vLLM's NIXL integration accuracy test:
https://github.com/vllm-project/vllm/blob/main/tests/v1/kv_connector/nixl_integration/test_accuracy.py
which is Apache-2.0 licensed. That file's structure (lm_eval `simple_evaluate`
against a `local-completions` endpoint, gsm8k / strict-match, a per-model
expected-value table with a one-sided tolerance) is retained; the oracle is not.
The rest of this repository is MIT. Both headers are kept above, deliberately.

What this catches that upstream's version does not
--------------------------------------------------
Upstream asserts a single served endpoint clears a hardcoded per-model score.
That answers "is this model still good?", which drifts with every model, vLLM,
and attention-backend bump. It does not answer the question AIC actually risks
getting wrong: *does routing KV through DRAM/NVMe change the answers?*

So the primary assertion here is a same-run A/B. Two arms are brought up from
the same image, same weights, same seed, in the same job:

    baseline  — plain vLLM, KV stays in VRAM
    tiered    — LMCache + NIXL, KV spills to DRAM/NVMe

``tiered >= baseline - DELTA`` catches KV corruption, and is immune to version
drift because both numbers move together.

That single assertion has one blind spot: if both arms break in the same way,
the difference stays zero. So a second, independent assertion keeps upstream's
idea — ``tiered >= expected.json[MODEL] - FLOOR_RTOL`` — as an absolute floor.
Two failure modes, two assertions.

Configuration is entirely environment-driven so that CI and a laptop run the
same code; see ``README.md`` in this directory.
"""

from __future__ import annotations

import json
import math
import os
import pathlib

import pytest

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MODEL = os.getenv("AIC_ACCURACY_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# None => full gsm8k test split. A small integer makes a PR-path run tractable
# at the cost of resolution: the strict-match score is a mean over LIMIT items,
# so its standard error grows as 1/sqrt(LIMIT).
LIMIT = int(os.getenv("AIC_ACCURACY_LIMIT", "0")) or None

# How far the tiered arm may fall below the baseline arm before we call it
# corruption. Derived from measured run-to-run variance, not chosen a priori —
# see README.md.
DELTA = float(os.getenv("AIC_ACCURACY_DELTA", "0.02"))

# Absolute-floor slack, matching upstream's RTOL. Applies as-is to a full-split
# run; `_floor_slack` widens it when LIMIT makes sampling noise the larger term.
FLOOR_RTOL = 0.05

NUM_CONCURRENT = int(os.getenv("AIC_ACCURACY_CONCURRENT", "32"))

# The two arms cannot be up at once: they share a container name, a port, and a
# GPU. So the driver runs them sequentially and hands the first arm's score to
# the second arm's pytest invocation through this variable. When it is set, the
# baseline is not re-measured (and no baseline endpoint is needed).
_BASELINE_SCORE = os.getenv("AIC_ACCURACY_BASELINE_SCORE")
BASELINE_SCORE = float(_BASELINE_SCORE) if _BASELINE_SCORE else None

# Set by the restart phase of the Slurm driver: the tiered score measured
# before vLLM was restarted. When set, `test_score_survives_restart` compares
# the current tiered score against it.
_REFERENCE_SCORE = os.getenv("AIC_ACCURACY_REFERENCE_SCORE")
REFERENCE_SCORE = float(_REFERENCE_SCORE) if _REFERENCE_SCORE else None

# When set, the measured tiered score is written here as JSON so a later pytest
# invocation (the restart phase) can be handed it via AIC_ACCURACY_REFERENCE_SCORE.
SCORE_OUT = os.getenv("AIC_ACCURACY_SCORE_OUT")

TASK = "gsm8k"
FILTER = "exact_match,strict-match"
NUM_FEWSHOT = 5

_EXPECTED_PATH = pathlib.Path(__file__).parent / "expected.json"


def _expected_values() -> dict[str, float]:
    with _EXPECTED_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)["models"]


def _meets_accuracy_threshold(measured: float, expected: float, rtol: float) -> bool:
    """One-sided: overshooting the expected value is never a failure."""
    return measured >= expected - rtol


def _floor_slack(expected: float, limit: int | None) -> float:
    """Floor tolerance, widened for the sampling noise a small LIMIT introduces.

    FLOOR_RTOL=0.05 is calibrated for the full 1319-item split, where the
    binomial standard error is 0.011 and three sigma (0.033) still fits inside
    it. Capping the item count inflates that error as 1/sqrt(n): at LIMIT=200 it
    is 0.028, so three sigma is 0.085 -- and a fixed 0.05 floor would fail a
    perfectly healthy run about a third of the time.

    So the slack is max(FLOOR_RTOL, 3 * SE(limit)). This deliberately makes a
    capped run a weaker gate rather than a flaky one; the nightly runs the full
    split, where the floor stays tight.
    """
    if not limit:
        return FLOOR_RTOL
    se = math.sqrt(max(expected * (1.0 - expected), 0.0) / limit)
    return max(FLOOR_RTOL, 3.0 * se)


def _score(base_url: str) -> float:
    """Run gsm8k against one endpoint and return its strict-match score."""
    import lm_eval

    model_args = (
        f"model={MODEL},"
        f"base_url={base_url}/completions,"
        f"num_concurrent={NUM_CONCURRENT},"
        "tokenized_requests=False,"
        "trust_remote_code=True"
    )
    results = lm_eval.simple_evaluate(
        model="local-completions",
        model_args=model_args,
        tasks=TASK,
        num_fewshot=NUM_FEWSHOT,
        limit=LIMIT,
    )
    return float(results["results"][TASK][FILTER])


# --------------------------------------------------------------------------
# Fixtures — each arm is scored at most once per session, then shared.
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiered_score(tiered_url: str) -> float:
    score = _score(tiered_url)
    print(f"\n[accuracy] tiered   {TASK} {FILTER} = {score:.4f}  (limit={LIMIT})")
    if SCORE_OUT:
        pathlib.Path(SCORE_OUT).write_text(
            json.dumps({"model": MODEL, "limit": LIMIT, "score": score}) + "\n",
            encoding="utf-8",
        )
    return score


@pytest.fixture(scope="session")
def baseline_score(request: pytest.FixtureRequest) -> float:
    """Baseline score, either handed in by the driver or measured here.

    Prefer the handed-in value: the driver scores the VRAM-only arm first, tears
    it down (the arms share a container name, a port, and the GPU, so they
    cannot coexist), then brings up the tiered arm. Requesting `baseline_url`
    only in the measure-it-here path keeps the sequential driver from needing a
    live baseline endpoint it has already torn down.
    """
    if BASELINE_SCORE is not None:
        print(f"\n[accuracy] baseline {TASK} {FILTER} = {BASELINE_SCORE:.4f}  (supplied)")
        return BASELINE_SCORE
    url = request.getfixturevalue("baseline_url")
    score = _score(url)
    print(f"\n[accuracy] baseline {TASK} {FILTER} = {score:.4f}  (limit={LIMIT})")
    return score


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_threshold_is_one_sided() -> None:
    """Guard the comparison helper itself. Runs anywhere, needs no endpoint."""
    assert _meets_accuracy_threshold(0.72, 0.77, 0.05)
    assert _meets_accuracy_threshold(0.83, 0.77, 0.05)
    assert not _meets_accuracy_threshold(0.71, 0.77, 0.05)


def test_tiered_matches_baseline(tiered_score: float, baseline_score: float) -> None:
    """The differential assertion: tiering KV must not change the answers."""
    assert tiered_score >= baseline_score - DELTA, (
        f"tiered arm regressed against baseline by more than DELTA={DELTA}\n"
        f"  model:    {MODEL}\n"
        f"  task:     {TASK} {FILTER} (limit={LIMIT}, {NUM_FEWSHOT}-shot)\n"
        f"  baseline: {baseline_score:.4f}\n"
        f"  tiered:   {tiered_score:.4f}\n"
        f"  delta:    {tiered_score - baseline_score:+.4f} "
        f"(allowed: >= -{DELTA})"
    )


def test_tiered_above_floor(tiered_score: float) -> None:
    """The absolute floor: catches both arms breaking identically."""
    expected = _expected_values().get(MODEL)
    if expected is None:
        pytest.skip(
            f"no floor for {MODEL} in {_EXPECTED_PATH.name}; "
            "add one to enable this assertion (see README.md). "
            "An unknown model is a config gap, not a regression."
        )
    slack = _floor_slack(expected, LIMIT)
    assert _meets_accuracy_threshold(tiered_score, expected, slack), (
        f"tiered arm fell below the absolute floor\n"
        f"  model:    {MODEL}\n"
        f"  task:     {TASK} {FILTER} (limit={LIMIT}, {NUM_FEWSHOT}-shot)\n"
        f"  expected: {expected:.4f} (floor {expected - slack:.4f}, "
        f"slack {slack:.4f}{'' if slack == FLOOR_RTOL else ' widened for LIMIT'})\n"
        f"  measured: {tiered_score:.4f}"
    )


def test_score_survives_restart(tiered_score: float) -> None:
    """Re-scoring after a vLLM restart must match the pre-restart score.

    Only runs when the driver supplies AIC_ACCURACY_REFERENCE_SCORE. LMCache
    keeps its DRAM/NVMe state across a vLLM restart, so the prompts are
    guaranteed cache hits and every block is served from the tier rather than
    recomputed. A drop here means retrieval from NVMe is corrupting blocks.
    """
    if REFERENCE_SCORE is None:
        pytest.skip("AIC_ACCURACY_REFERENCE_SCORE unset; not a restart-phase run")
    assert tiered_score >= REFERENCE_SCORE - DELTA, (
        f"score dropped after vLLM restart — suspect NVMe retrieval\n"
        f"  model:         {MODEL}\n"
        f"  pre-restart:   {REFERENCE_SCORE:.4f}\n"
        f"  post-restart:  {tiered_score:.4f}\n"
        f"  delta:         {tiered_score - REFERENCE_SCORE:+.4f} "
        f"(allowed: >= -{DELTA})"
    )
