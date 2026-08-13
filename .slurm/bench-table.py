#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""Render the RESULT lines emitted by .slurm/bench-serve.sh as a markdown table.

One RESULT line per configuration, one row per (configuration, concurrency).
Reading the raw lines by eye does not scale past about three arms.

Usage:  .slurm/bench-table.py logs/bench/results-*.txt
"""
import re
import sys

ROW = ("| {name} | {tp} | {conn} | {c} | {out} | {perGPU} | {perUser} | "
       "{ttft_p50} | {ttft_p95} | {tpot_p50} | {e2e_p50} | {e2e_p95} |")
HEAD = ("| run | TP | conn | c | out tok/s | tok/s/GPU | tok/s/user | "
        "TTFT p50 | TTFT p95 | TPOT p50 | E2E p50 | E2E p95 |\n"
        "|-----|----|------|---|-----------|-----------|------------|"
        "----------|----------|----------|---------|---------|")


def parse(line):
    """RESULT: k=v ... sweep=c8[k=v,...] c32[...] status=ok"""
    sweep = ""
    m = re.search(r"\bsweep=(.*?)\s+status=", line)
    if m:
        sweep = m.group(1)
        line = line[:m.start()] + line[m.end() - len("status="):]
    top = dict(re.findall(r"(\w+)=([^\s]+)", line.replace("RESULT:", "")))
    points = []
    for c, body in re.findall(r"c(\d+)\[([^\]]*)\]", sweep):
        d = dict(kv.split("=", 1) for kv in body.split(",") if "=" in kv)
        d["c"] = c
        points.append(d)
    return top, points


def main(paths):
    print(HEAD)
    for path in paths:
        for line in open(path):
            if not line.startswith("RESULT:"):
                continue
            top, points = parse(line)
            if not points:
                print(f"| {top.get('name','?')} | | | | "
                      f"**{top.get('status','?')}** | | | | | | | |")
                continue
            for p in points:
                print(ROW.format(name=top.get("name", "?"), tp=top.get("tp", "?"),
                                 conn=top.get("conn", "?"), **p))


if __name__ == "__main__":
    main(sys.argv[1:] or ["-"])
