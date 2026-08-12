# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Fixtures for the AIC differential accuracy gate.

Every test in this package needs at least one live OpenAI-compatible endpoint.
Rather than fail when none is configured, the fixtures here skip with an
explicit reason. That is what lets a developer run just the tiered-arm
assertions against a local stack, with no baseline arm, and lets the whole
package be collected on a laptop with no GPU at all.
"""

from __future__ import annotations

import os

import pytest

# How long to wait for /v1/models before declaring an endpoint dead. This is a
# liveness probe, not a readiness wait — the caller (the Slurm driver) is
# responsible for waiting out the weight load before invoking pytest.
_PROBE_TIMEOUT_S = 10.0


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
        pytest.skip(f"tiered endpoint unreachable: {reason}")
    assert url is not None
    return url.rstrip("/")


@pytest.fixture(scope="session")
def baseline_url() -> str:
    """Base URL of the VRAM-only baseline arm, or skip."""
    url = os.getenv("AIC_ACCURACY_BASELINE_URL")
    reason = _probe(url)
    if reason is not None:
        pytest.skip(f"baseline endpoint unreachable: {reason}")
    assert url is not None
    return url.rstrip("/")
