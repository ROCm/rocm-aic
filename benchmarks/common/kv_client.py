# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""Shared pieces for the small synthetic KV-cache drivers (replay_hits.py,
fixed_load.py, agent-simple/stress.py).

Everything here talks to the stack by container name from the aic bridge; run
the drivers in aic-client, through `aic-bench`.

    AIC_BENCH_ENDPOINT             vLLM base URL, no /v1 (default http://aic-lmcache:8000)
    AIC_BENCH_MODEL                served model id (default: the first one the endpoint lists)
    AIC_BENCH_LMCACHE_METRICS_URL  LMCache exporter (default: the endpoint's host, port 9091)
    AIC_BENCH_NIXL_METRICS_URL     NIXL exporter (default: the endpoint's host, port 19090)
    AIC_HSA_SNOOP_URL              hsa-snoop exporter (default http://aic-hsa-snoop:9488)
"""
import json
import os
import random
import time
import urllib.request
from urllib.parse import urlsplit

ENDPOINT = (os.environ.get("AIC_BENCH_ENDPOINT") or "http://aic-lmcache:8000").rstrip("/")


def _peer(env, port):
    """A service in the endpoint's network namespace, unless <env> names it."""
    host = urlsplit(ENDPOINT).hostname
    if not host:
        raise SystemExit(f"AIC_BENCH_ENDPOINT {ENDPOINT!r} is not a URL such as http://aic-lmcache:8000")
    return os.environ.get(env) or (f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}")


LMCACHE_METRICS = _peer("AIC_BENCH_LMCACHE_METRICS_URL", 9091)
NIXL_METRICS = _peer("AIC_BENCH_NIXL_METRICS_URL", 19090)
HSA_SNOOP = os.environ.get("AIC_HSA_SNOOP_URL", "http://aic-hsa-snoop:9488")
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
         "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()
L1_HIT = "lmcache_mp_lookup_hit_l1_tokens_total"
L2_HIT = "lmcache_mp_lookup_hit_l2_tokens_total"


def model():
    if os.environ.get("AIC_BENCH_MODEL"):
        return os.environ["AIC_BENCH_MODEL"]
    with urllib.request.urlopen(ENDPOINT + "/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def prompt(tag, n_words):
    """Deterministic text for <tag>: the same tag always gives the same prompt."""
    r = random.Random(tag)
    return f"Document {tag}. " + " ".join(r.choice(WORDS) + str(r.randrange(1000)) for _ in range(n_words))


def complete(model_id, text, max_tokens=4):
    """POST one completion; return (usage, seconds)."""
    body = json.dumps({"model": model_id, "prompt": text, "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(ENDPOINT + "/v1/completions", body, {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        usage = json.load(r)["usage"]
    return usage, time.time() - t


def _samples(url):
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=30) as r:
        for line in r.read().decode().splitlines():
            if line and not line.startswith("#"):
                series, _, value = line.rpartition(" ")
                yield series, float(value)


def scrape(url, names):
    """Sum every series of each metric name in the exporter at <url>."""
    out = dict.fromkeys(names, 0.0)
    for series, value in _samples(url):
        name = series.split("{", 1)[0]
        if name in out:
            out[name] += value
    return out


def by_label(url, name, label):
    """Sum one metric's series per value of <label>."""
    out = {}
    for series, value in _samples(url):
        if series.startswith(name + "{") and f'{label}="' in series:
            key = series.split(f'{label}="', 1)[1].split('"', 1)[0]
            out[key] = out.get(key, 0.0) + value
    return out


def lmcache(names):
    return scrape(LMCACHE_METRICS, names)


def vllm(names):
    return scrape(ENDPOINT, names)


def nixl(names):
    return scrape(NIXL_METRICS, names)


def tier_hits():
    m = lmcache([L1_HIT, L2_HIT])
    return int(m[L1_HIT]), int(m[L2_HIT])
