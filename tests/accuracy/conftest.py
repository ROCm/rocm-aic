# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Fixtures for the AIC differential accuracy gate.

Every test in this package needs at least one live OpenAI-compatible endpoint.
By default the fixtures skip, with an explicit reason, when one is missing.
That is what lets a developer run just the tiered-arm assertions against a
local stack, with no baseline arm, and lets the whole package be collected on
a laptop with no GPU at all.

Skipping is the wrong default under CI: a dead endpoint would report green and
the gate would pass without having scored anything. Set AIC_ACCURACY_REQUIRED=1
(the CI driver does) to turn every skip in this file into a failure. The
knob is deliberately opt-in rather than auto-detected from $CI, so the same
behaviour is reproducible by hand off a runner.

`AIC_ACCURACY_REQUIRED` only governs endpoint reachability. Skips that encode a
genuine "this assertion does not apply to this run" — an unknown model with no
floor, or the restart phase's absent reference score — stay skips, since
failing those would be reporting a config gap as a regression.
"""

from __future__ import annotations

import os

import pytest

_PROBE_TIMEOUT_S = 10.0


def _required() -> bool:
    return os.getenv("AIC_ACCURACY_REQUIRED", "").strip() not in ("", "0", "false")


def _unusable(arm: str, reason: str) -> None:
    """Skip, or fail when the caller declared a live endpoint mandatory."""
    message = f"{arm} endpoint unreachable: {reason}"
    if _required():
        pytest.fail(
            f"{message}\n"
            "AIC_ACCURACY_REQUIRED=1, so an unscored arm is a failure rather "
            "than a skip — the gate cannot pass without scoring this endpoint."
        )
    pytest.skip(message)


def _probe(url: str | None) -> str | None:
    """Return a human-readable reason the endpoint is unusable, else None."""
    if not url:
        return "not configured"

    import httpx

    models_url = url.rstrip("/") + "/models"
    try:
        resp = httpx.get(models_url, timeout=_PROBE_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 - any transport error is "dead"
        return f"{models_url}: {type(exc).__name__}: {exc}"
    if resp.status_code != 200:
        return f"{models_url}: HTTP {resp.status_code}"
    return None


@pytest.fixture(scope="session")
def tiered_url() -> str:
    """Base URL of the tiered (LMCache DRAM/NVMe) arm, or skip."""
    url = os.getenv("AIC_ACCURACY_TIERED_URL")
    reason = _probe(url)
    if reason is not None:
        _unusable("tiered", reason)
    assert url is not None
    return url.rstrip("/")


@pytest.fixture(scope="session")
def baseline_url() -> str:
    """Base URL of the VRAM-only baseline arm, or skip.

    An operator can declare the baseline arm absent for this run
    (AIC_ACCURACY_SKIP_BASELINE=1, what `make accuracy-test-fast` does to halve
    the bringups). That is a deliberate loss of the differential, not a broken
    endpoint, so it stays a skip even under AIC_ACCURACY_REQUIRED.
    """
    if os.getenv("AIC_ACCURACY_SKIP_BASELINE", "").strip() not in ("", "0", "false"):
        pytest.skip(
            "AIC_ACCURACY_SKIP_BASELINE=1; this run has no baseline arm, so the "
            "differential does not apply (the floor and restart gates still do)"
        )
    url = os.getenv("AIC_ACCURACY_BASELINE_URL")
    reason = _probe(url)
    if reason is not None:
        _unusable("baseline", reason)
    assert url is not None
    return url.rstrip("/")
