#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""Concurrent mixed-size replay against vLLM to draw GPU, LMCache L1 and L2 hits.

Synthetic cache-mechanics load, not a realistic workload: deterministic
pseudo-random documents are sent once (warm-up), then revisited at random by
N concurrent clients, 20% with a fresh suffix.  The tier breakdown comes from
counter deltas on the vLLM, LMCache and NIXL exporters, so anything else using
the endpoint during a run is counted too.

Runs in aic-client on the aic bridge (run_stress.sh, or `make agent-simple-run`):

    stress.py <run-id> <documents> <concurrency> <seconds> [<max-words>]

AIC_STRESS_HOST names the container that owns the vLLM network namespace
(default aic-lmcache: vLLM on :8000, LMCache on :9091, NIXL on :19090).
"""
import json
import math
import os
import random
import sys
import threading
import time
import urllib.request

HOST = os.environ.get("AIC_STRESS_HOST", "aic-lmcache")
BASE = f"http://{HOST}:8000"
MODEL = json.load(urllib.request.urlopen(BASE + "/v1/models"))["data"][0]["id"]
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu".split()
RUN, NDOCS, CONC, SECONDS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rng = random.Random(RUN)
# log-uniform sizes, about 250 to 6000 tokens (4.58 tokens per word)
MAXW = int(sys.argv[5]) if len(sys.argv) > 5 else 1300
SIZES = [int(math.exp(rng.uniform(math.log(55), math.log(MAXW)))) for _ in range(NDOCS)]

def text(tag, n):
    r = random.Random(tag)
    return f"Document {tag}. " + " ".join(r.choice(WORDS) + str(r.randrange(1000)) for _ in range(n))

def metric(port, names):
    out = dict.fromkeys(names, 0.0)
    for line in urllib.request.urlopen(f"http://{HOST}:{port}/metrics", timeout=30).read().decode().splitlines():
        for n in names:
            if line.startswith(n + "{") or line.startswith(n + " "):
                out[n] += float(line.rsplit(" ", 1)[1])
    return out
LS = ["lmcache_mp_lookup_hit_l1_tokens_total", "lmcache_mp_lookup_hit_l2_tokens_total",
      "lmcache_mp_num_finished_stores_total", "lmcache_mp_num_finished_retrieves_total",
      "lmcache_mp_l1_evicted_chunks_total"]
def snap():
    v = {}
    for line in urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode().splitlines():
        if line.startswith("vllm:prompt_tokens_by_source_total{"):
            src = line.split('source="')[1].split('"')[0]; v[src] = v.get(src, 0) + float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:request_success_total{"):
            fr = line.split('finished_reason="')[1].split('"')[0]; v["fin_" + fr] = float(line.rsplit(" ", 1)[1])
    l = metric(9091, LS)
    n = metric(19090, ["agent_errors_total"])
    return dict(gpu=v.get("local_cache_hit", 0), ext=v.get("external_kv_transfer", 0), computed=v.get("local_compute", 0),
                l1=l[LS[0]], l2=l[LS[1]], stores=l[LS[2]], retrieves=l[LS[3]], l1_evicted=l[LS[4]],
                nixl_err=n["agent_errors_total"], err=v.get("fin_error", 0) + v.get("fin_abort", 0))

lock = threading.Lock(); stats = dict(ok=0, fail=0, lat=[]); stop = False; fails = {}
def send(doc, suffix=None, max_tokens=16):
    p = text(f"{RUN}-D{doc}", SIZES[doc]) + (" " + text(suffix, 60) if suffix else "")
    body = json.dumps({"model": MODEL, "prompt": p, "max_tokens": max_tokens, "temperature": 0}).encode()
    t = time.time()
    try:
        urllib.request.urlopen(urllib.request.Request(BASE + "/v1/completions", body, {"Content-Type": "application/json"}), timeout=600).read()
        with lock: stats["ok"] += 1; stats["lat"].append(time.time() - t)
    except Exception as e:  # noqa: BLE001
        with lock:
            stats["fail"] += 1; k = type(e).__name__ + ":" + str(getattr(e, "code", ""))[:6]; fails[k] = fails.get(k, 0) + 1

def phase(name, jobs):
    it = iter(jobs)
    def worker():
        while not stop:
            with lock:
                j = next(it, None)
            if j is None: return
            send(*j)
    ths = [threading.Thread(target=worker) for _ in range(CONC)]
    [t.start() for t in ths]; return ths

def show(label, a, b, dt):
    d = {k: b[k] - a[k] for k in a}
    tot = d["gpu"] + d["ext"] + d["computed"]
    pct = lambda x: f"{100*x/tot:5.1f}%" if tot else "  n/a"
    print(f"{label}: {int(tot):>8} prompt tok in {dt:5.1f}s | GPU {pct(d['gpu'])} | L1 {pct(d['l1'])} | L2 {pct(d['l2'])} | computed {pct(d['computed'])}"
          f" | stores +{int(d['stores'])} retrieves +{int(d['retrieves'])} L1evict +{int(d['l1_evicted'])} | nixlErr +{int(d['nixl_err'])} reqErr +{int(d['err'])}"
          f" | ok={stats['ok']} fail={stats['fail']}", flush=True)
    return d

total_tok = int(sum(SIZES) * 4.58)
print(f"model={MODEL} docs={NDOCS} (~{total_tok} tokens, sizes {int(min(SIZES)*4.58)}..{int(max(SIZES)*4.58)}) concurrency={CONC} stress={SECONDS}s", flush=True)
s0 = snap(); t0 = time.time()
ths = phase("warm", [(d, None, 8) for d in range(NDOCS)]); [t.join() for t in ths]
time.sleep(35)  # lazy offload flush window
s1 = snap(); show("warm  ", s0, s1, time.time() - t0)

def jobs():
    r = random.Random(RUN + "-stress"); i = 0
    while True:
        i += 1
        d = r.randrange(NDOCS)
        yield (d, f"{RUN}-S{i}" if r.random() < 0.2 else None, r.choice((8, 16, 32, 64)))
ths = phase("stress", jobs()); prev, tp, tstart = s1, time.time(), time.time()
while time.time() - tstart < SECONDS:
    time.sleep(30); cur = snap(); show(f"t+{int(time.time()-tstart):>3}s", prev, cur, time.time() - tp); prev, tp = cur, time.time()
stop = True; [t.join() for t in ths]
time.sleep(5); s2 = snap()
print("-" * 40); d = show("STRESS", s1, s2, time.time() - tstart)
lat = sorted(stats["lat"])
print(f"requests ok={stats['ok']} failed={stats['fail']} {fails or ''} latency p50={lat[len(lat)//2]:.2f}s p95={lat[int(len(lat)*.95)]:.2f}s max={lat[-1]:.2f}s")
print(f"hit tokens: GPU={int(d['gpu'])} L1={int(d['l1'])} L2={int(d['l2'])} vLLM-external={int(d['ext'])} (L1+L2={int(d['l1']+d['l2'])}) computed={int(d['computed'])}")
