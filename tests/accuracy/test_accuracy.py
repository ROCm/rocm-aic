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

We assert that a given model gives the same answer to every question when
served in the following modes:
    baseline  — plain vLLM, KV stays in VRAM
    tiered    — the AIC stack

We additionally track a `FLOOR_RTOL` in `expected.json` to catch regressions
that would impact both paths.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import NamedTuple

import pytest

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MODEL = os.getenv("AIC_ACCURACY_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# Item cap passed to lm_eval.simple_evaluate(limit=...).  None means the full
# split.  A capped run widens the floor tolerance by 3× the binomial SE so the
# gate remains meaningful without becoming flaky.  The per-question comparisons
# need no adjustment: every pass is asked the same items.
_LIMIT_ENV = os.getenv("AIC_ACCURACY_LIMIT")
LIMIT: int | None = int(_LIMIT_ENV) if _LIMIT_ENV else None

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


# One request in flight.  Decoding is greedy, but which requests vLLM batches
# together depends on when they arrive, and the batch shape picks the attention
# and GEMM kernels, so concurrent requests get slightly different arithmetic and
# a near-tie token can flip.  At 32, two identical servers disagreed on 747 of
# 1319 answers; at 1 they agreed on all of them.
NUM_CONCURRENT = int(os.getenv("AIC_ACCURACY_CONCURRENT", "1"))

# The baseline arm's per-question answers, from an earlier scoring pass.  The arms
# share the GPU, so the driver scores the baseline first and hands its answers on.
BASELINE_SAMPLES = os.getenv("AIC_ACCURACY_BASELINE_SAMPLES")
# The restart pass's reference: the baseline arm re-scored with its GPU prefix
# cache still warm, so its prompts' prefixes were reused rather than recomputed,
# just as the restart pass reuses them from LMCache.  Enables the restart test.
REFERENCE_SAMPLES = os.getenv("AIC_ACCURACY_REFERENCE_SAMPLES")
# vLLM's --block-size, which the driver sets equal to LMCache's chunk size.
BLOCK_SIZE = int(os.getenv("AIC_ACCURACY_BLOCK_SIZE", "256"))

# When set, the measured tiered score is written here as JSON for the CI report.
SCORE_OUT = os.getenv("AIC_ACCURACY_SCORE_OUT")
# When set, the tiered arm's per-question answers are written here as JSON lines.
SAMPLES_OUT = os.getenv("AIC_ACCURACY_SAMPLES_OUT")

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
    information. Same-run comparisons compare answers instead; see _diverging.
    """
    return measured >= expected - rtol


class Arm(NamedTuple):
    """One scoring pass: its score and its answers keyed by gsm8k doc_id."""

    score: float
    samples: dict[int, dict]


def _load_samples(path: str | pathlib.Path) -> dict[int, dict]:
    with open(path, encoding="utf-8") as fh:
        return {s["doc_id"]: s for s in map(json.loads, fh)}


def _score_of(samples: dict[int, dict]) -> float:
    return sum(s["exact_match"] for s in samples.values()) / len(samples)


def _diverging(measured: dict[int, dict], reference: dict[int, dict]) -> list[int]:
    """doc_ids whose generated text differs between two passes.

    Raises if the passes were not asked the same questions, since then a
    difference in the answers would say nothing about the KV path.
    """
    if measured.keys() != reference.keys():
        raise ValueError(
            f"the passes scored different questions: {len(measured)} vs "
            f"{len(reference)} items, {len(measured.keys() ^ reference.keys())} "
            "doc_ids in only one of them"
        )
    prompts = [i for i in reference if measured[i]["prompt_hash"] != reference[i]["prompt_hash"]]
    if prompts:
        raise ValueError(
            f"{len(prompts)} questions got different prompts (first doc_ids: "
            f"{sorted(prompts)[:5]}); check the lm_eval version and few-shot seed"
        )
    return sorted(i for i in reference if measured[i]["resp"] != reference[i]["resp"])


def _describe_divergence(
    ids: list[int], measured: dict[int, dict], reference: dict[int, dict], shown: int = 5
) -> str:
    """Which answers changed, which way their correctness moved, and where."""
    lost = sum(1 for i in ids if reference[i]["exact_match"] and not measured[i]["exact_match"])
    gained = sum(1 for i in ids if measured[i]["exact_match"] and not reference[i]["exact_match"])
    lines = [
        f"{len(ids)} of {len(reference)} answers differ "
        f"({lost} newly wrong, {gained} newly right)"
    ]
    for i in ids[:shown]:
        a, b = reference[i]["resp"], measured[i]["resp"]
        at = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        lines.append(
            f"  doc {i}: diverges at char {at}: "
            f"{a[max(0, at - 20):at + 40]!r} -> {b[max(0, at - 20):at + 40]!r}"
        )
    if len(ids) > shown:
        lines.append(f"  ... and {len(ids) - shown} more")
    return "\n".join(lines)


# Runs in a child process; see _score.  Takes its settings as one JSON argument.
# With samples_out set, writes one JSON line per question: the generated text,
# the extracted answer, and whether it was right.  prompt_hash lets a comparison
# prove two runs were asked the same questions; prompt_tokens is the prompt's
# length as vLLM counts it (same tokenizer, no special tokens added).
_SCORE_SCRIPT = """
import json, sys, lm_eval
from transformers import AutoTokenizer
cfg = json.loads(sys.argv[1])
results = lm_eval.simple_evaluate(
    model="local-completions",
    model_args=(
        f"model={cfg['model']},"
        f"base_url={cfg['url']}/completions,"
        f"num_concurrent={cfg['concurrent']},"
        "tokenized_requests=False,"
        "trust_remote_code=True"
    ),
    tasks=cfg["task"],
    num_fewshot=cfg["fewshot"],
    limit=cfg["limit"],
    random_seed=42,
    numpy_random_seed=42,
    torch_random_seed=42,
    log_samples=bool(cfg["samples_out"]),
)
metric, filter_name = cfg["filter"].split(",")
if cfg["samples_out"]:
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    with open(cfg["samples_out"], "w", encoding="utf-8") as fh:
        for s in results["samples"][cfg["task"]]:
            if s["filter"] != filter_name:
                continue
            fh.write(json.dumps({
                "doc_id": s["doc_id"],
                "prompt_hash": s["prompt_hash"],
                "prompt_tokens": len(tok(s["arguments"][0][0])["input_ids"]),
                "resp": s["resps"][0][0],
                "answer": s["filtered_resps"][0],
                "exact_match": s[metric],
            }) + "\\n")
print(json.dumps({"score": results["results"][cfg["task"]][cfg["filter"]]}))
"""


def _score(base_url: str, samples_out: str | None = None) -> float:
    """Run gsm8k against one endpoint and return its strict-match score.

    With samples_out, also write each question's answer there as JSON lines.

    lm_eval.simple_evaluate() uses asyncio.run() internally; if called more
    than once in the same process, aiohttp's TCPConnector retains a reference
    to the first (now-closed) event loop and fails with addr_infos=[].  Running
    in a subprocess gives each call a clean Python process and event loop.
    """
    import subprocess
    import sys

    cfg = {
        "model": MODEL, "url": base_url, "concurrent": NUM_CONCURRENT,
        "task": TASK, "fewshot": NUM_FEWSHOT, "limit": LIMIT, "filter": FILTER,
        "samples_out": samples_out,
    }
    proc = subprocess.run(
        [sys.executable, "-c", _SCORE_SCRIPT, json.dumps(cfg)],
        capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"lm_eval subprocess failed (rc={proc.returncode}):\n"
            + proc.stderr[-2000:]
        )
    # The subprocess may emit lm_eval INFO lines before the JSON; take the last.
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            data = json.loads(line)
            print(f'\n{{"url": "{base_url}", "model": "{MODEL}", "score": {data["score"]}}}')
            return float(data["score"])
    raise RuntimeError(f"No JSON score in subprocess output:\n{proc.stdout[-2000:]}")


# --------------------------------------------------------------------------
# Fixtures — each arm is scored at most once per session, then shared.
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiered(tiered_url: str, tmp_path_factory: pytest.TempPathFactory) -> Arm:
    samples_out = SAMPLES_OUT or str(tmp_path_factory.mktemp("tiered") / "samples.jsonl")
    score = _score(tiered_url, samples_out)
    print(f"\n[accuracy] tiered   {TASK} {FILTER} = {score:.4f}")
    if SCORE_OUT:
        pathlib.Path(SCORE_OUT).write_text(
            json.dumps({"model": MODEL, "score": score}) + "\n",
            encoding="utf-8",
        )
    return Arm(score, _load_samples(samples_out))


@pytest.fixture(scope="session")
def tiered_score(tiered: Arm) -> float:
    return tiered.score


@pytest.fixture(scope="session")
def baseline(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Arm:
    """Baseline answers, either measured by this run or supplied by the driver."""
    if BASELINE_SAMPLES:
        samples = _load_samples(BASELINE_SAMPLES)
        score = _score_of(samples)
        print(f"\n[accuracy] baseline {TASK} {FILTER} = {score:.4f}  (supplied)")
        return Arm(score, samples)
    url = request.getfixturevalue("baseline_url")
    samples_out = tmp_path_factory.mktemp("baseline") / "samples.jsonl"
    score = _score(url, str(samples_out))
    print(f"\n[accuracy] baseline {TASK} {FILTER} = {score:.4f}")
    return Arm(score, _load_samples(samples_out))


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_floor_threshold_is_one_sided() -> None:
    """Guard the floor helper. Runs anywhere, needs no endpoint."""
    assert _meets_accuracy_threshold(0.72, 0.77, 0.05)
    assert _meets_accuracy_threshold(0.83, 0.77, 0.05)
    assert not _meets_accuracy_threshold(0.71, 0.77, 0.05)


def test_divergence_is_reported_per_question() -> None:
    """Guard the answer comparison. Runs anywhere, needs no endpoint."""
    ref = {
        0: {"prompt_hash": "p0", "resp": "#### 1", "exact_match": 1.0},
        1: {"prompt_hash": "p1", "resp": "#### 2", "exact_match": 0.0},
    }
    same = {i: dict(s) for i, s in ref.items()}
    assert _diverging(same, ref) == []
    changed = {**same, 1: {"prompt_hash": "p1", "resp": "#### 3", "exact_match": 1.0}}
    assert _diverging(changed, ref) == [1]
    assert "1 newly right" in _describe_divergence([1], changed, ref)
    with pytest.raises(ValueError, match="different prompts"):
        _diverging({**same, 0: {**same[0], "prompt_hash": "other"}}, ref)
    with pytest.raises(ValueError, match="different questions"):
        _diverging({0: same[0]}, ref)


def test_tiered_matches_baseline(tiered: Arm, baseline: Arm) -> None:
    """The differential assertion: tiering KV must not change the answers.

    Compared question by question, and exactly.  With one request in flight
    both arms run every prompt through the same kernels, so their generated
    text is byte-identical.  Every prompt in this pass is new, so the tiered arm
    reads no KV back from the tier; what this proves is that wiring in the
    connector and writing KV out to LMCache leave the computation untouched.
    Reading it back is test_answers_survive_restart's job.
    """
    ids = _diverging(tiered.samples, baseline.samples)
    assert not ids, (
        "tiered arm's answers differ from the baseline arm's\n"
        f"  model:    {MODEL}\n"
        f"  task:     {TASK} {FILTER} ({NUM_FEWSHOT}-shot"
        + (f", limit={LIMIT}" if LIMIT else "") + ")\n"
        f"  baseline: {baseline.score:.4f}\n"
        f"  tiered:   {tiered.score:.4f}\n"
        + _describe_divergence(ids, tiered.samples, baseline.samples)
        + (f"\n  AIC_ACCURACY_CONCURRENT={NUM_CONCURRENT}: above 1, answers "
           "depend on request timing" if NUM_CONCURRENT > 1 else "")
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


def _comparable_after_restart(samples: dict[int, dict]) -> dict[int, dict]:
    """The questions whose prompt prefix is reused identically from either cache.

    When a prompt's prefix is cached, vLLM recomputes only the rest, and the
    length of that remainder decides which kernels run.  Both caches reuse
    whole BLOCK_SIZE-token blocks and must leave at least the last token to
    compute, so both reuse floor((n - 1) / B) * B tokens of an n-token prompt --
    except when n is an exact multiple of B.  There vLLM's own cache has to
    drop the whole last block, while the connector may load all but the last
    token, so the two recompute B tokens and 1.  Those prompts (7 of 1319 for
    the default model at B = 256) cannot be compared exactly and are left out.
    """
    return {i: s for i, s in samples.items() if s["prompt_tokens"] % BLOCK_SIZE}


def test_restart_reuse_rule_excludes_block_multiples() -> None:
    """Guard the exclusion rule. Runs anywhere, needs no endpoint."""
    samples = {
        i: {"prompt_tokens": n}
        for i, n in enumerate([BLOCK_SIZE - 1, BLOCK_SIZE, BLOCK_SIZE + 1, 4 * BLOCK_SIZE])
    }
    assert sorted(_comparable_after_restart(samples)) == [0, 2]


def test_answers_survive_restart(tiered: Arm) -> None:
    """KV read back from LMCache must give the answers the GPU's own cache gives.

    vLLM is restarted while LMCache keeps its DRAM/NVMe state, and the same
    prompts are re-scored, so their prefixes come back from the tier.  The
    reference is the baseline arm re-scored with its GPU prefix cache still
    warm: the same prefixes reused, but from VRAM.  The first pass cannot be the
    reference, since it recomputed every prompt in full, and a different amount
    of recomputation is enough to change the arithmetic and so ~57% of the
    answers.  Against the cached re-score the tier must match exactly.
    """
    if REFERENCE_SAMPLES is None:
        pytest.skip("AIC_ACCURACY_REFERENCE_SAMPLES unset; not a restart-phase run")
    reference = _load_samples(REFERENCE_SAMPLES)
    comparable = _comparable_after_restart(reference)
    ids = [i for i in _diverging(tiered.samples, reference) if i in comparable]
    assert not ids, (
        "answers after the vLLM restart differ from the cached baseline re-score "
        "-- suspect KV retrieval from the tier\n"
        f"  model:     {MODEL}\n"
        f"  reference: {_score_of(reference):.4f} (baseline, prefixes from VRAM)\n"
        f"  restart:   {tiered.score:.4f} (tiered, prefixes from LMCache)\n"
        f"  compared:  {len(comparable)} of {len(reference)} questions "
        f"({len(reference) - len(comparable)} prompts are a multiple of {BLOCK_SIZE} "
        "tokens long; see _comparable_after_restart)\n"
        + _describe_divergence(ids, tiered.samples, reference)
    )
