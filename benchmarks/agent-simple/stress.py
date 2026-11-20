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

    aic-bench agent-simple <run-id> <documents> <concurrency> <seconds> [<max-words>]

The endpoint and the exporters come from common/kv_client.py (AIC_BENCH_ENDPOINT).
"""
import argparse
import math
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import kv_client as kv  # noqa: E402

TOKENS_PER_WORD = 4.58
LMCACHE = [kv.L1_HIT, kv.L2_HIT, "lmcache_mp_num_finished_stores_total",
           "lmcache_mp_num_finished_retrieves_total", "lmcache_mp_l1_evicted_chunks_total"]
NIXL_ERRORS = "agent_errors_total"


def snap():
    source = kv.by_label(kv.ENDPOINT, "vllm:prompt_tokens_by_source_total", "source")
    finished = kv.by_label(kv.ENDPOINT, "vllm:request_success_total", "finished_reason")
    lm = kv.lmcache(LMCACHE)
    return dict(gpu=source.get("local_cache_hit", 0), ext=source.get("external_kv_transfer", 0),
                computed=source.get("local_compute", 0),
                l1=lm[LMCACHE[0]], l2=lm[LMCACHE[1]], stores=lm[LMCACHE[2]], retrieves=lm[LMCACHE[3]],
                l1_evicted=lm[LMCACHE[4]], nixl_err=kv.nixl([NIXL_ERRORS])[NIXL_ERRORS],
                err=finished.get("error", 0) + finished.get("abort", 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", help="tag that makes this run's documents unique")
    ap.add_argument("documents", type=int)
    ap.add_argument("concurrency", type=int)
    ap.add_argument("seconds", type=int, help="length of the stress phase")
    ap.add_argument("max_words", type=int, nargs="?", default=1300,
                    help="largest document, in words (default 1300, about 6000 tokens)")
    ap.add_argument("--flush-wait", type=float, default=35,
                    help="seconds to wait after the warm-up for lazy offload (default 35)")
    ap.add_argument("--report-every", type=float, default=30,
                    help="seconds between progress lines (default 30)")
    args = ap.parse_args()
    run, ndocs = args.run_id, args.documents
    model = kv.model()
    rng = random.Random(run)
    # log-uniform sizes, from about 250 tokens up to max_words
    sizes = [int(math.exp(rng.uniform(math.log(55), math.log(args.max_words)))) for _ in range(ndocs)]

    lock = threading.Lock()
    stats = dict(ok=0, fail=0, lat=[])
    fails = {}
    stop = threading.Event()

    def send(doc, suffix, max_tokens):
        text = kv.prompt(f"{run}-D{doc}", sizes[doc]) + (" " + kv.prompt(suffix, 60) if suffix else "")
        try:
            _, seconds = kv.complete(model, text, max_tokens)
            with lock:
                stats["ok"] += 1
                stats["lat"].append(seconds)
        except Exception as e:  # noqa: BLE001
            key = type(e).__name__ + ":" + str(getattr(e, "code", ""))[:6]
            with lock:
                stats["fail"] += 1
                fails[key] = fails.get(key, 0) + 1

    def phase(jobs):
        it = iter(jobs)

        def worker():
            while not stop.is_set():
                with lock:
                    job = next(it, None)
                if job is None:
                    return
                send(*job)

        threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
        for t in threads:
            t.start()
        return threads

    def show(label, a, b, dt):
        d = {k: b[k] - a[k] for k in a}
        tot = d["gpu"] + d["ext"] + d["computed"]

        def pct(x):
            return f"{100 * x / tot:5.1f}%" if tot else "  n/a"

        print(f"{label}: {int(tot):>8} prompt tok in {dt:5.1f}s | GPU {pct(d['gpu'])} | L1 {pct(d['l1'])}"
              f" | L2 {pct(d['l2'])} | computed {pct(d['computed'])}"
              f" | stores +{int(d['stores'])} retrieves +{int(d['retrieves'])} L1evict +{int(d['l1_evicted'])}"
              f" | nixlErr +{int(d['nixl_err'])} reqErr +{int(d['err'])}"
              f" | ok={stats['ok']} fail={stats['fail']}", flush=True)
        return d

    def stress_jobs():
        r = random.Random(run + "-stress")
        i = 0
        while True:
            i += 1
            doc = r.randrange(ndocs)
            yield doc, f"{run}-S{i}" if r.random() < 0.2 else None, r.choice((8, 16, 32, 64))

    print(f"model={model} docs={ndocs} (~{int(sum(sizes) * TOKENS_PER_WORD)} tokens, sizes "
          f"{int(min(sizes) * TOKENS_PER_WORD)}..{int(max(sizes) * TOKENS_PER_WORD)}) "
          f"concurrency={args.concurrency} stress={args.seconds}s", flush=True)
    s0, t0 = snap(), time.time()
    for t in phase([(d, None, 8) for d in range(ndocs)]):
        t.join()
    time.sleep(args.flush_wait)
    s1 = snap()
    show("warm  ", s0, s1, time.time() - t0)

    threads = phase(stress_jobs())
    prev, tp, tstart = s1, time.time(), time.time()
    while (left := tstart + args.seconds - time.time()) > 0:
        time.sleep(min(args.report_every, left))
        cur = snap()
        show(f"t+{int(time.time() - tstart):>3}s", prev, cur, time.time() - tp)
        prev, tp = cur, time.time()
    stop.set()
    for t in threads:
        t.join()
    time.sleep(min(5, args.report_every))
    s2 = snap()
    print("-" * 40)
    d = show("STRESS", s1, s2, time.time() - tstart)
    lat = sorted(stats["lat"])
    latency = (f"latency p50={lat[len(lat) // 2]:.2f}s p95={lat[int(len(lat) * .95)]:.2f}s max={lat[-1]:.2f}s"
               if lat else "no request succeeded")
    print(f"requests ok={stats['ok']} failed={stats['fail']} {fails or ''} {latency}")
    print(f"hit tokens: GPU={int(d['gpu'])} L1={int(d['l1'])} L2={int(d['l2'])} vLLM-external={int(d['ext'])}"
          f" (L1+L2={int(d['l1'] + d['l2'])}) computed={int(d['computed'])}")
    return 0 if stats["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
