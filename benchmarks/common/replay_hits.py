#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""Push prompts out of the GPU prefix cache, then replay them to draw LMCache hits.

Sends a few target prompts, then enough filler prompts to evict them from the
GPU cache, waits for lazy offload, and replays a band of early fillers and the
targets, printing the L1 / L2 hit tokens each replay drew.  With --reset-gpu it
then clears only the GPU prefix cache (needs VLLM_SERVER_DEV_MODE=1) and replays
the same prompts again; they were just loaded into L1, so these are L1 hits.

    aic-bench replay-hits <run-id> <fillers> <words> <replay-from> <replay-to> [<pace-s>] [--reset-gpu]
    aic-bench replay-hits r1 90 440 12 32 3 --reset-gpu     # 2,000-token prompts, paced

Keep prompts well under the L1 size: a prompt whose KV does not fit in L1 is
dropped at store time and can never be hit.  A paced fill gives L1 eviction
time to keep up.  --reset-gpu also drops any pending lazy-offload stores.
"""
import argparse
import time
import urllib.request

import kv_client as kv

V = ["vllm:external_prefix_cache_hits_total", "vllm:external_prefix_cache_queries_total"]
L = [kv.L1_HIT, kv.L2_HIT, "lmcache_mp_num_finished_retrieves_total", "lmcache_mp_num_finished_stores_total"]


def short(counters):
    return {k.split(":")[-1].replace("lmcache_mp_", ""): int(v) for k, v in counters.items()}


def snap(label):
    s = {**kv.vllm(V), **kv.lmcache(L)}
    print(label, short(s), flush=True)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", help="tag that makes this run's prompts unique")
    ap.add_argument("fillers", type=int, help="filler prompts sent to evict the targets")
    ap.add_argument("words", type=int, help="words per prompt")
    ap.add_argument("replay_from", type=int, help="first filler to replay")
    ap.add_argument("replay_to", type=int, help="one past the last filler to replay")
    ap.add_argument("pace", type=float, nargs="?", default=0.0, help="seconds between fillers (default 0)")
    ap.add_argument("--reset-gpu", action="store_true", help="clear the GPU prefix cache and replay again")
    args = ap.parse_args()
    run, words = args.run_id, args.words
    model = kv.model()

    def send(tag, show=False):
        before = kv.tier_hits() if show else None
        usage, dt = kv.complete(model, kv.prompt(tag, words))
        if show:
            after = kv.tier_hits()
            print(f"  {tag:<10} prompt_tokens={usage['prompt_tokens']} {dt:.2f}s "
                  f"L1+{after[0] - before[0]} L2+{after[1] - before[1]}", flush=True)

    targets = [f"{run}-T{i}" for i in range(4)]
    fillers = [f"{run}-F{i}" for i in range(args.fillers)]
    snap("before:")
    print("targets (first pass)")
    for t in targets:
        send(t)
    print(f"{args.fillers} fillers")
    for f in fillers:
        send(f)
        time.sleep(args.pace)
    print("waiting 40 s for lazy offload")
    time.sleep(40)
    send(f"{run}-flush")
    time.sleep(5)
    mid = snap("after fill:")
    replay = fillers[args.replay_from:args.replay_to] + targets
    print(f"replay fillers {args.replay_from}..{args.replay_to - 1} and the targets")
    for t in replay:
        send(t, True)
    time.sleep(35)
    after = snap("after replay:")
    print("delta over replay:", short({k: after[k] - mid[k] for k in after}))

    if args.reset_gpu:
        req = urllib.request.Request(kv.ENDPOINT + "/reset_prefix_cache", b"", method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            print("reset_prefix_cache (GPU only) ->", r.status)
        time.sleep(2)
        for t in replay[-8:]:
            send(t, True)
        time.sleep(32)
        snap("after GPU reset replay:")


if __name__ == "__main__":
    main()
