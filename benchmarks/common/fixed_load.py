#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""A fixed, repeatable workload for comparing hsa-snoop builds and settings.

Sends N fresh prompts one at a time (64 generated tokens each, so 64 engine
steps per request) and reports how many kernel launches and durations hsa-snoop
counted for them, per engine step.  Run it once per configuration; the workload
is the same every time apart from the run id, which keeps the prompts uncached.

    fixed_load.py <run-id> <requests> [<words>]
    fixed_load.py base 12
"""
import sys
import time

import kv_client as kv

tag, n = sys.argv[1], int(sys.argv[2])
words = int(sys.argv[3]) if len(sys.argv) > 3 else 440
MODEL = kv.model()
HSA = ["hsa_kernel_launches_total", "hsa_kernel_duration_seconds_count", "hsa_dispatches_missed_total"]
STEPS = "vllm:iteration_tokens_total_count"


def hsa():
    return kv.scrape(kv.HSA_SNOOP + "/metrics", HSA)


a, s0, t = hsa(), kv.vllm([STEPS])[STEPS], time.time()
for i in range(n):
    kv.complete(MODEL, kv.prompt(f"{tag}-{i}", words), max_tokens=64)
elapsed = time.time() - t
time.sleep(3)
b, s1 = hsa(), kv.vllm([STEPS])[STEPS]
steps = int(s1 - s0)
d = {k: int(b[k] - a[k]) for k in HSA}
print(f"{tag}: {n} requests in {elapsed:.1f}s engine_steps={steps} launches={d[HSA[0]]} "
      f"durations={d[HSA[1]]} missed={d[HSA[2]]} launches/step={d[HSA[0]] / max(steps, 1):.1f}")
