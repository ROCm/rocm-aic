#!/usr/bin/env python3
"""
kv_reset_test.py — external KV cache retrieval test, for any vLLM KV connector.

Stores a few anchor prompts, clears vLLM's GPU prefix cache with
POST /reset_prefix_cache, and sends the anchors again.  With the GPU cache
empty, the only way vLLM can skip their prefill is to restore the blocks from
the connector's store, which vLLM itself counts in
vllm:external_prefix_cache_hits.  Nothing here asks the cache server anything,
so the same test covers LMCache and MORI UMBP (`make kv-reset-test
AIC_KV_BACKEND=mori`); vllm_reset_test.py is the LMCache-specific one that also
checks the L1 and L2 tiers.

Exit codes
----------
  0  blocks were restored from the external cache
  1  test logic error / unexpected failure
  2  no external cache hit after the GPU cache reset

Env overrides (all optional)
-----------------------------
  AIC_TEST_VLLM_URL     vLLM OpenAI-compat base URL (default: http://aic-vllm:8000)
  AIC_TEST_MODEL        Model name served by vLLM  (default: the first one listed)
  AIC_TEST_ANCHORS      Number of anchor prompts   (default: 4)
  AIC_TEST_ANCHOR_REPS  Preamble reps per anchor    (default: 22, ~2090 tok)
  AIC_TEST_MAX_TOKENS   Tokens generated per request (default: 20)
  AIC_TEST_TIMEOUT      Per-request timeout (s)    (default: 120)
  AIC_TEST_SEED         RNG seed for prompt generation (default: 42)
  AIC_TEST_SETTLE_S     Wait after the anchors, for stores to land (default: 5)
"""

import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

VLLM = os.getenv("AIC_TEST_VLLM_URL", "http://aic-vllm:8000").rstrip("/")
MODEL = os.getenv("AIC_TEST_MODEL", "")
N_ANCHOR = int(os.getenv("AIC_TEST_ANCHORS", "4"))
ANCHOR_REPS = int(os.getenv("AIC_TEST_ANCHOR_REPS", "22"))
MAX_TOK = int(os.getenv("AIC_TEST_MAX_TOKENS", "20"))
TIMEOUT = int(os.getenv("AIC_TEST_TIMEOUT", "120"))
SETTLE_S = float(os.getenv("AIC_TEST_SETTLE_S", "5"))

_PREAMBLE = (
    "You are an expert in distributed KV cache systems, GPU memory management, "
    "NVMe storage tiering, Prometheus observability, HIP IPC shared memory, "
    "vLLM prefix caching, ROCm internals and RDMA over Converged Ethernet. The "
    "following is a detailed technical document covering all of these topics. "
) * ANCHOR_REPS

# The bridge is reached directly; a proxy from the environment would not know it.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _request(path, payload=None, timeout=15):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        VLLM + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            return resp.read().decode(errors="replace")
    except (urllib.error.URLError, OSError) as exc:
        log(f"{path}: {exc}", prefix="! ")
        return ""


def _json(path, payload=None, timeout=15):
    try:
        return json.loads(_request(path, payload, timeout))
    except ValueError:
        return None


def counter(name):
    """Sum of a counter over its label sets; vLLM appends _total on export."""
    total = 0.0
    for line in _request("/metrics").splitlines():
        if line.startswith((name + "_total{", name + "_total ", name + "{", name + " ")):
            try:
                total += float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return int(total)


def snap():
    return {
        "gpu_hits": counter("vllm:prefix_cache_hits"),
        "ext_queries": counter("vllm:external_prefix_cache_queries"),
        "ext_hits": counter("vllm:external_prefix_cache_hits"),
        "prompt": counter("vllm:prompt_tokens"),
    }


def send(prompt):
    r = _json("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOK,
        "temperature": 0.0,
    }, timeout=TIMEOUT)
    return r is not None and "choices" in r


def log(msg, prefix=""):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {prefix}{msg}", flush=True)


def delta(after, before):
    return {k: after[k] - before[k] for k in after}


def main():
    global MODEL
    rng = random.Random(int(os.getenv("AIC_TEST_SEED", "42")))

    if not MODEL:
        models = _json("/v1/models") or {}
        MODEL = next((m["id"] for m in models.get("data", [])), "")
    if not MODEL:
        log(f"ERROR: {VLLM} lists no model", prefix="✗ ")
        return 1

    print("=" * 60)
    print("AIC KV Reset Test — external KV cache retrieval")
    print(f"  vLLM:     {VLLM}")
    print(f"  Model:    {MODEL}")
    print(f"  Anchors:  {N_ANCHOR} x {ANCHOR_REPS} preamble reps")
    print("=" * 60)

    if not send("hi"):
        log("ERROR: vLLM not responding or model not loaded", prefix="✗ ")
        return 1
    reset = _json("/reset_prefix_cache", {})
    if not (reset and reset.get("success")):
        log("ERROR: POST /reset_prefix_cache failed — is VLLM_SERVER_DEV_MODE=1 set?", prefix="✗ ")
        return 1

    # The tag leads, so no two anchors share a prefix: every block of an anchor
    # is its own, and after the reset none can come from another anchor's reload.
    anchors = [
        f"[ANCHOR-{i:04d}-{rng.randrange(10**12, 10**13)}] " + _PREAMBLE
        for i in range(N_ANCHOR)
    ]

    base = snap()
    log(f"1. Storing {N_ANCHOR} anchor prompts")
    ok = sum(send(a) for a in anchors)
    stored = delta(snap(), base)
    log(f"   {ok}/{N_ANCHOR} served; prompt tokens={stored['prompt']:,}")
    if ok != N_ANCHOR:
        log("ERROR: an anchor request failed", prefix="✗ ")
        return 1
    time.sleep(SETTLE_S)

    log("2. Resetting the GPU prefix cache")
    reset = _json("/reset_prefix_cache", {})
    if not (reset and reset.get("success")):
        log("ERROR: GPU cache reset failed", prefix="✗ ")
        return 1
    before = snap()

    log(f"3. Re-sending the {N_ANCHOR} anchors — GPU cache empty, expect external hits")
    for i, anchor in enumerate(anchors):
        if not send(anchor):
            log(f"ERROR: anchor {i} failed on replay", prefix="✗ ")
            return 1
        d = delta(snap(), before)
        log(f"   anchor {i:02d}: external hits=+{d['ext_hits']:,} tokens  "
            f"GPU hits=+{d['gpu_hits']:,}")
    d = delta(snap(), before)

    print("-" * 60)
    print(f"  replayed prompt tokens:        {d['prompt']:,}")
    print(f"  external cache queries:        {d['ext_queries']:,} tokens")
    print(f"  restored from external cache:  {d['ext_hits']:,} tokens")
    print(f"  served from GPU prefix cache:  {d['gpu_hits']:,} tokens")
    print("-" * 60)
    if d["ext_hits"] <= 0:
        log("FAIL: nothing was restored from the external cache", prefix="✗ ")
        return 2
    log(f"PASS: {d['ext_hits']:,} of {d['prompt']:,} replayed prompt tokens "
        "came back from the external cache", prefix="✓ ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
