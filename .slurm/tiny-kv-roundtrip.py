#!/usr/bin/env python3
"""tiny-test KV round-trip: prove LMCache really holds vLLM's KV cache.

Run inside the vLLM container (stdlib only):
    docker exec -i aic-vllm-gpuN python3 - <model> < tiny-kv-roundtrip.py

A completion coming back proves nothing about LMCache: when the connector's
KV registration is skipped, vLLM serves from VRAM and LMCache stays inert.  So:

  1. Ask for a fact planted at the start of a prompt several LMCache chunks
     long.  Recalling it is the control: the model can do the task at all.
  2. Drop vLLM's own prefix cache (VLLM_SERVER_DEV_MODE endpoint).
  3. Ask again.  vLLM must now fetch the prefix from LMCache, so
     vllm:external_prefix_cache_hits_total has to grow, and the answer must
     still hold the fact.

The fact sits in the first chunk, so it survives only if the KV LMCache
stored is what vLLM computed and the retrieve wrote it back into vLLM's own
KV cache.  A connector that copies the cache once at registration stores
and returns that stale copy, and the second answer loses the fact.
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = sys.argv[1]
SECRET = "7391-ALPHA"
HITS = "vllm:external_prefix_cache_hits_total"

# ~1.5k tokens: about six 256-token LMCache chunks ahead of the question.
FILLER = "".join(
    f"Log entry {i}: the shipment for warehouse {i * 7 % 23} was checked "
    f"and found complete.\n"
    for i in range(80)
)
PROMPT = (
    f"Remember this: the secret code is {SECRET}.\n\n{FILLER}\n"
    "Question: what is the secret code mentioned at the start?\n"
    "Answer: The secret code is"
)


def post(path, body=None):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=120).read().decode()


def complete():
    # The prompt spans several chunks, so LMCache stores it; with KV
    # registration skipped that store kills vLLM's engine and this gets a 500.
    try:
        out = json.loads(post("/v1/completions", {
            "model": MODEL, "prompt": PROMPT, "max_tokens": 16, "temperature": 0,
        }))
    except urllib.error.HTTPError as e:
        sys.exit(f"[kv-roundtrip] FAIL: completion returned HTTP {e.code}; "
                 "see the vLLM log")
    return out["choices"][0]["text"]


def hits():
    body = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    total = None
    for line in body.splitlines():
        if line.startswith(HITS) and line[len(HITS)] in "{ ":
            total = (total or 0.0) + float(line.rsplit(" ", 1)[1])
    if total is None:
        sys.exit(f"[kv-roundtrip] FAIL: {HITS} missing from /metrics")
    return total


before = hits()
first = complete()
print(f"[kv-roundtrip] first answer : {first!r}")
if SECRET not in first:
    sys.exit("[kv-roundtrip] FAIL: the model could not recall the code even "
             "with the prompt computed in VRAM; the check is inconclusive")

# LMCache stores asynchronously after the request finishes.
time.sleep(5)
post("/reset_prefix_cache")
second = complete()
gained = hits() - before
print(f"[kv-roundtrip] second answer: {second!r}")
print(f"[kv-roundtrip] external prefix-cache hits: +{gained:.0f} tokens")

if gained <= 0:
    sys.exit("[kv-roundtrip] FAIL: no external prefix-cache hits; LMCache "
             "never served the prefix (KV registration skipped?)")
if SECRET not in second:
    sys.exit("[kv-roundtrip] FAIL: LMCache served the prefix but the answer "
             "lost the code; the KV it returned is not what vLLM computed")
print("[kv-roundtrip] OK: LMCache returned vLLM's own KV")
