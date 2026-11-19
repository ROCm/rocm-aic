# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""Shared pieces for the small synthetic KV-cache drivers in this directory.

Everything here talks to the stack by container name from the aic bridge
(run the scripts in aic-client).  AIC_STRESS_HOST names the container that owns
vLLM's network namespace (default aic-lmcache: vLLM on :8000, LMCache on :9091).
"""
import json
import os
import random
import time
import urllib.request

HOST = os.environ.get("AIC_STRESS_HOST", "aic-lmcache")
BASE = f"http://{HOST}:8000"
HSA_SNOOP = os.environ.get("AIC_HSA_SNOOP_URL", "http://aic-hsa-snoop:9488")
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
         "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()
L1_HIT = "lmcache_mp_lookup_hit_l1_tokens_total"
L2_HIT = "lmcache_mp_lookup_hit_l2_tokens_total"


def model():
    with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def prompt(tag, n_words):
    """Deterministic text for <tag>: the same tag always gives the same prompt."""
    r = random.Random(tag)
    return f"Document {tag}. " + " ".join(r.choice(WORDS) + str(r.randrange(1000)) for _ in range(n_words))


def complete(model_id, text, max_tokens=4):
    """POST one completion; return (usage, seconds)."""
    body = json.dumps({"model": model_id, "prompt": text, "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(BASE + "/v1/completions", body, {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        usage = json.load(r)["usage"]
    return usage, time.time() - t


def scrape(url, names):
    """Sum every series of each metric name in a Prometheus exposition."""
    out = dict.fromkeys(names, 0.0)
    with urllib.request.urlopen(url, timeout=30) as r:
        for line in r.read().decode().splitlines():
            for n in names:
                if line.startswith(n + "{") or line.startswith(n + " "):
                    out[n] += float(line.rsplit(" ", 1)[1])
    return out


def lmcache(names):
    return scrape(f"http://{HOST}:9091/metrics", names)


def vllm(names):
    return scrape(BASE + "/metrics", names)


def tier_hits():
    m = lmcache([L1_HIT, L2_HIT])
    return int(m[L1_HIT]), int(m[L2_HIT])
