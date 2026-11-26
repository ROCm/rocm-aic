#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""A fixed, repeatable workload for comparing hsa-snoop builds and settings.

Sends N fresh prompts one at a time (64 generated tokens each, so 64 engine
steps per request) and reports how many kernel launches and durations hsa-snoop
counted for them, per engine step.  Run it once per configuration; the workload
is the same every time apart from the run id, which keeps the prompts uncached.

Launch counts are not comparable with runs recorded before hsa-snoop 1.8.0,
which stopped counting vendor packets in the launch counter.

    aic-bench fixed-load <run-id> <requests> [<words>]
    aic-bench fixed-load base 12
"""
import argparse
import time

import kv_client as kv

# Per-GPU totals, not the pid-labelled counters: hsa-snoop drops a process's
# series --series-retain-sec (300 s) after its last queue goes away, so a bare
# sum() over those can fall between the two scrapes.  hsa_kernel_duration_
# seconds_count has no pid label and is safe.
HSA = ["hsa_gpu_kernel_launches_total", "hsa_kernel_duration_seconds_count", "hsa_gpu_dispatches_missed_total"]
STEPS = "vllm:iteration_tokens_total_count"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", help="tag that makes this run's prompts unique")
    ap.add_argument("requests", type=int)
    ap.add_argument("words", type=int, nargs="?", default=440, help="words per prompt (default 440)")
    args = ap.parse_args()
    model = kv.model()

    a, s0, t = kv.scrape(kv.HSA_SNOOP, HSA), kv.vllm([STEPS])[STEPS], time.time()
    for i in range(args.requests):
        kv.complete(model, kv.prompt(f"{args.run_id}-{i}", args.words), max_tokens=64)
    elapsed = time.time() - t
    time.sleep(3)
    b, s1 = kv.scrape(kv.HSA_SNOOP, HSA), kv.vllm([STEPS])[STEPS]
    steps = int(s1 - s0)
    d = {k: int(b[k] - a[k]) for k in HSA}
    print(f"{args.run_id}: {args.requests} requests in {elapsed:.1f}s engine_steps={steps} launches={d[HSA[0]]} "
          f"durations={d[HSA[1]]} missed={d[HSA[2]]} launches/step={d[HSA[0]] / max(steps, 1):.1f}")


if __name__ == "__main__":
    main()
