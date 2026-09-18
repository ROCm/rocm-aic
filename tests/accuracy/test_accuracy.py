# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""AIC KV-integrity accuracy gate — differential lm_eval over two serving arms.

Adapted from vLLM's NIXL integration accuracy test:
https://github.com/vllm-project/vllm/blob/main/tests/v1/kv_connector/nixl_integration/test_accuracy.py
@ 1ab2801ddebe31b75dd6022c69113b610bbdc950.

We assert that an A/B accuracy test of a given model produces similar results
when using the following modes:
    baseline  — plain vLLM, KV stays in VRAM
    tiered    — the AIC stack

We additionally track a `FLOOR_RTOL` in `expected.json` to catch regressions
that would impact both paths.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MODEL = os.getenv("AIC_ACCURACY_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# Item cap passed to lm_eval.simple_evaluate(limit=...).  None means the full
# split.  A capped run widens the floor tolerance by 3× the binomial SE so the
# gate remains meaningful without becoming flaky.  The differential (DELTA) is
# left unchanged: both arms sample the same items, so the binomial term cancels.
_LIMIT_ENV = os.getenv("AIC_ACCURACY_LIMIT")
LIMIT: int | None = int(_LIMIT_ENV) if _LIMIT_ENV else None

# If |accuracy[tiered] - accuracy[baseline]| > AIC_ACCURACY_DELTA, fail.  Two-
# sided: the arms answer identical prompts, so an unexplained gain is the same
# evidence of divergence as a loss.
DELTA = float(os.getenv("AIC_ACCURACY_DELTA", "0.02"))

# Slack on the absolute floor, calibrated for the full 1319-item split.  When
# LIMIT is set the floor is widened by 3× the binomial SE so healthy capped runs
# are not rejected by noise while still catching catastrophic regressions.
FLOOR_RTOL = 0.05


def _floor_slack(expected: float) -> float:
    """Return the floor tolerance for the current LIMIT (or full split)."""
    if LIMIT is None:
        return FLOOR_RTOL
    import math
    se = math.sqrt(expected * (1.0 - expected) / LIMIT)
    return max(FLOOR_RTOL, 3.0 * se)


def _delta_slack() -> float:
    """Return the differential tolerance for the current LIMIT (or full split).

    The baseline and tiered arms run as separate vLLM instances, so their
    outputs differ by per-request nondeterminism even on the same prompts.
    For the full split this is negligible (DELTA=0.02 is ~6 sigma); for a
    capped run the expected inter-instance spread is sqrt(2)*SE per arm,
    which must be covered by the tolerance.  We use 4 sigma so healthy
    capped runs pass with high probability.
    """
    if LIMIT is None:
        return DELTA
    import math
    # Approximate score for SE calculation using DELTA midpoint; the exact
    # value is unknown at call time so use the floor from expected.json if
    # available, otherwise a conservative p=0.5 (maximum variance).
    try:
        p = list(_expected_values().values())[0]
    except Exception:
        p = 0.5
    se = math.sqrt(p * (1.0 - p) / LIMIT)
    return max(DELTA, 4.0 * math.sqrt(2.0) * se)

NUM_CONCURRENT = int(os.getenv("AIC_ACCURACY_CONCURRENT", "32"))

_BASELINE_SCORE = os.getenv("AIC_ACCURACY_BASELINE_SCORE")
BASELINE_SCORE = float(_BASELINE_SCORE) if _BASELINE_SCORE else None
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
    """One-sided: overshooting the expected value is never a failure.

    Used only for the absolute floor, where `expected` is a coarse published
    number rather than a same-run measurement, so an overshoot carries no
    information. Same-run comparisons use `_within_delta` instead.
    """
    return measured >= expected - rtol


def _within_delta(measured: float, reference: float, delta: float) -> bool:
    """Two-sided: the reference is a same-run score over identical prompts."""
    return abs(measured - reference) <= delta


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
        # Fixed seeds ensure repeated calls (e.g. pre/post-restart) score the
        # same items in the same order, so the restart delta only reflects model
        # nondeterminism rather than sampling variance.
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
    )
    return float(results["results"][TASK][FILTER])


# --------------------------------------------------------------------------
# Fixtures — each arm is scored at most once per session, then shared.
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiered_score(tiered_url: str) -> float:
    score = _score(tiered_url)
    print(f"\n[accuracy] tiered   {TASK} {FILTER} = {score:.4f}")
    if SCORE_OUT:
        pathlib.Path(SCORE_OUT).write_text(
            json.dumps({"model": MODEL, "score": score}) + "\n",
            encoding="utf-8",
        )
    return score


@pytest.fixture(scope="session")
def baseline_score(request: pytest.FixtureRequest) -> float:
    """Baseline score, either measured by this run, or requested from results from previous run."""
    if BASELINE_SCORE is not None:
        print(
            f"\n[accuracy] baseline {TASK} {FILTER} = {BASELINE_SCORE:.4f}  (supplied)"
        )
        return BASELINE_SCORE
    url = request.getfixturevalue("baseline_url")
    score = _score(url)
    print(f"\n[accuracy] baseline {TASK} {FILTER} = {score:.4f}")
    return score


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_floor_threshold_is_one_sided() -> None:
    """Guard the floor helper. Runs anywhere, needs no endpoint."""
    assert _meets_accuracy_threshold(0.72, 0.77, 0.05)
    assert _meets_accuracy_threshold(0.83, 0.77, 0.05)
    assert not _meets_accuracy_threshold(0.71, 0.77, 0.05)


def test_differential_threshold_is_two_sided() -> None:
    """An unexplained gain fails the differential just as a loss does."""
    assert _within_delta(0.77, 0.77, 0.02)
    assert _within_delta(0.76, 0.77, 0.02)
    assert _within_delta(0.78, 0.77, 0.02)
    assert not _within_delta(0.74, 0.77, 0.02)
    assert not _within_delta(0.80, 0.77, 0.02)


def test_tiered_matches_baseline(tiered_score: float, baseline_score: float) -> None:
    """The differential assertion: tiering KV must not change the answers.

    Two-sided. Both arms answer the same questions with the same weights, so
    the only honest outcome is "the same score, modulo batching nondeterminism".
    A tiered arm that beats the baseline by more than DELTA has not got smarter;
    something about the comparison has changed — a stale supplied baseline, a
    config drift between the phases — and that is a broken gate, not a win.
    """
    tol = _delta_slack()
    assert _within_delta(tiered_score, baseline_score, tol), (
        f"tiered arm diverged from baseline by more than tolerance={tol:.4f}\n"
        f"  model:    {MODEL}\n"
        f"  task:     {TASK} {FILTER} ({NUM_FEWSHOT}-shot"
        + (f", limit={LIMIT}" if LIMIT else "") + ")\n"
        f"  baseline: {baseline_score:.4f}\n"
        f"  tiered:   {tiered_score:.4f}\n"
        f"  delta:    {tiered_score - baseline_score:+.4f} "
        f"(allowed: +/-{tol:.4f})"
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
    slack = _floor_slack(expected)
    assert _meets_accuracy_threshold(tiered_score, expected, slack), (
        f"tiered arm fell below the absolute floor\n"
        f"  model:    {MODEL}\n"
        f"  task:     {TASK} {FILTER} ({NUM_FEWSHOT}-shot"
        + (f", limit={LIMIT}" if LIMIT else "") + ")\n"
        f"  expected: {expected:.4f} (floor {expected - slack:.4f}, "
        f"slack {slack:.4f})\n"
        f"  measured: {tiered_score:.4f}"
    )


def test_score_survives_restart(tiered_score: float) -> None:
    """Re-scoring after a vLLM restart must match the pre-restart score.

    LMCache should preserve it's cached state in DRAM/NVMe across a restart,
    so the repeated prompts will be cache hits and fetched from the cache tier
    rather than being recomputed.

    Two-sided, for the same reason as the baseline differential: the post-restart
    run replays the same prompts against the same weights, so a score that moves
    in either direction means the answers changed.

    Both numbers come from the full split, so DELTA applies unmodified — the
    re-score and the reference asked exactly the same questions.
    """
    if REFERENCE_SCORE is None:
        pytest.skip("AIC_ACCURACY_REFERENCE_SCORE unset; not a restart-phase run")
    tol = _delta_slack()
    assert _within_delta(tiered_score, REFERENCE_SCORE, tol), (
        f"score changed after vLLM restart — suspect NVMe retrieval\n"
        f"  model:         {MODEL}\n"
        f"  pre-restart:   {REFERENCE_SCORE:.4f}\n"
        f"  post-restart:  {tiered_score:.4f}\n"
        f"  delta:         {tiered_score - REFERENCE_SCORE:+.4f} "
        f"(allowed: +/-{tol:.4f})"
    )
