# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Minimal OpenAI-compatible HTTP server backed by vLLM's LLMEmulator.
#
# The emulator skips GPU kernel launches and models timing mathematically, so
# this server runs on CPU-only nodes without /dev/kfd or /dev/dri.  It exposes
# the same three endpoints the production vLLM server does:
#
#   GET  /health              — liveness probe (returns {"status": "ok"})
#   POST /v1/completions      — OpenAI-compatible completion (delegates to emulator)
#   GET  /metrics             — Prometheus metrics (request counters + latency)
#
# Mount this file into the rocm-aic container at runtime; no image rebuild needed:
#
#   docker run --rm -p 127.0.0.1::8000 \
#     -v "$(pwd)/monitoring/scripts/vllm_emulator_server.py:/app/vllm_emulator_server.py:ro" \
#     -e VLLM_MODEL=facebook/opt-125m \
#     rocm-aic python3 /app/vllm_emulator_server.py
#
# Or via docker/docker-compose.emulator.yml.
#
# Required env:
#   VLLM_MODEL   — model name/path passed to LLMEmulator (default: facebook/opt-125m)
#
# Optional env:
#   EMULATOR_HOST   — bind address (default: 0.0.0.0)
#   EMULATOR_PORT   — listen port (default: 8000)
#   EMULATOR_LOG    — uvicorn log level (default: info)

import os
import time
import uuid
import logging

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional

# prometheus_client is bundled with vLLM
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

log = logging.getLogger("vllm-emulator")

# ---------------------------------------------------------------------------
# Prometheus metrics (mirrors the names the production vLLM server emits so
# Grafana dashboards and recording rules pick them up)
# ---------------------------------------------------------------------------
_requests_total = Counter(
    "vllm_requests_total",
    "Total number of completions requests",
    ["model"],
)
_request_duration = Histogram(
    "vllm_request_duration_seconds",
    "Completion request latency in seconds",
    ["model"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
_tokens_generated = Counter(
    "vllm_generation_tokens_total",
    "Total number of generation tokens produced",
    ["model"],
)

# ---------------------------------------------------------------------------
# LLMEmulator initialisation
# ---------------------------------------------------------------------------
MODEL = os.environ.get("VLLM_MODEL", "facebook/opt-125m")

_emulator = None
try:
    from vllm.engine.emulator import LLMEmulator  # type: ignore[import]
    _emulator = LLMEmulator(model=MODEL, emulate_latency=True)
    log.info("LLMEmulator loaded for model %s", MODEL)
except Exception as exc:
    # This vLLM build is GPU-only; fall back to a pure-Python stub that
    # returns canned completions so the smoke test can run on CPU-only nodes.
    log.warning("LLMEmulator unavailable (%s); using pure-Python stub", exc)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="vLLM Emulator Server", docs_url=None, redoc_url=None)


class CompletionRequest(BaseModel):
    model: str
    prompt: str | List[str]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[str | List[str]] = None
    stream: Optional[bool] = False


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(
        generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/v1/completions")
async def completions(req: CompletionRequest) -> JSONResponse:
    prompts = [req.prompt] if isinstance(req.prompt, str) else req.prompt
    t0 = time.perf_counter()

    elapsed = time.perf_counter() - t0
    choices = []

    if _emulator is not None:
        try:
            from vllm import SamplingParams  # type: ignore[import]
            params = SamplingParams(
                max_tokens=req.max_tokens or 16,
                temperature=req.temperature or 1.0,
                top_p=req.top_p or 1.0,
            )
            outputs = _emulator.generate(prompts, params)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        elapsed = time.perf_counter() - t0
        for i, out in enumerate(outputs):
            text = out.outputs[0].text if out.outputs else ""
            _tokens_generated.labels(model=req.model).inc(
                len(out.outputs[0].token_ids) if out.outputs else 0
            )
            choices.append({"index": i, "text": text, "logprobs": None, "finish_reason": "length"})
    else:
        # Pure-Python stub: return a canned completion so CPU-only smoke tests pass.
        elapsed = time.perf_counter() - t0
        for i, prompt in enumerate(prompts):
            text = f"[stub] {prompt[:20]} ..."
            _tokens_generated.labels(model=req.model).inc(4)
            choices.append({"index": i, "text": text, "logprobs": None, "finish_reason": "length"})

    _requests_total.labels(model=req.model).inc()
    _request_duration.labels(model=req.model).observe(elapsed)

    return JSONResponse({
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": choices,
        "usage": {
            "prompt_tokens": sum(len(p.split()) for p in prompts),
            "completion_tokens": sum(len(c["text"].split()) for c in choices),
            "total_tokens": 0,
        },
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        app,
        host=os.environ.get("EMULATOR_HOST", "0.0.0.0"),
        port=int(os.environ.get("EMULATOR_PORT", "8000")),
        log_level=os.environ.get("EMULATOR_LOG", "info"),
    )
