#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Correctness check: greedy decoding, same prompts → vanilla vs kvd arm
# MUST produce identical token streams. Any divergence = KV restoration
# error in the chunked-fusion / hipFile load path (e.g. truncated chunk
# silently DMA'd into VRAM, or staging-buffer race).
#
# Usage:
#   python3 correctness_check.py \
#     --vanilla http://<vanilla-host>:8000 \
#     --kvd http://127.0.0.1:8000 \
#     --model openai/gpt-oss-120b \
#     --c 16 --iters 3 --max-tokens 64 \
#     --isl 60000 --shared-prefix-tokens 60000
#
# To stress the kvd load path: pick c high enough that vLLM L1 evicts
# (working set > GPU prefix-cache budget). For 60k-token prompts at
# block_size=64, that's roughly c=32+ on a 80 GiB MI300X with default
# kv-cache-memory-bytes.

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from typing import Any

import httpx


_SHARED_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog. " * 200
).strip()


def _approx_words_for_tokens(n: int) -> int:
    return max(1, int(n * 0.75))


def _build_per_client_prefix(cid: int, n_tokens: int) -> str:
    rng = random.Random(cid * 7919 + 13)
    vocab = _SHARED_PARAGRAPH.split()
    out: list[str] = []
    target = _approx_words_for_tokens(n_tokens)
    while len(out) < target:
        rng.shuffle(vocab)
        out.extend(vocab)
    return " ".join(out[:target])


def build_prompt(cid: int, n_tokens: int) -> str:
    return _build_per_client_prefix(cid, n_tokens) + "\n\nQ: What is the next word?\nA:"


async def one(client: httpx.AsyncClient, endpoint: str, model: str,
              prompt: str, max_tokens: int) -> dict[str, Any]:
    r = await client.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model, "prompt": prompt,
            "max_tokens": max_tokens, "temperature": 0, "top_p": 1.0,
            "seed": 0,
        },
        timeout=600.0,
    )
    r.raise_for_status()
    d = r.json()
    return {"text": d["choices"][0]["text"]}


async def run_iter(endpoint: str, model: str, c: int, prefix_tokens: int,
                   max_tokens: int) -> list[dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        coros = [
            one(client, endpoint, model,
                build_prompt(cid=i, n_tokens=prefix_tokens), max_tokens)
            for i in range(c)
        ]
        return await asyncio.gather(*coros)


def compare(va: list[dict], kvd: list[dict]) -> tuple[int, int, list[str]]:
    match = 0
    diffs: list[str] = []
    for i, (a, b) in enumerate(zip(va, kvd)):
        if a["text"] == b["text"]:
            match += 1
        elif len(diffs) < 3:
            pos = next(
                (j for j in range(min(len(a["text"]), len(b["text"])))
                 if a["text"][j] != b["text"][j]),
                min(len(a["text"]), len(b["text"])),
            )
            diffs.append(
                f"  req{i}: diverge@char{pos}: "
                f"vanilla={a['text'][pos:pos+40]!r} kvd={b['text'][pos:pos+40]!r}"
            )
    return match, len(va), diffs


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla", required=True, help="vanilla arm endpoint (no kvd)")
    ap.add_argument("--kvd", required=True, help="kvd arm endpoint")
    ap.add_argument("--model", required=True)
    ap.add_argument("--c", type=int, default=16)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--isl", type=int, default=60000)
    ap.add_argument("--shared-prefix-tokens", type=int, default=60000)
    args = ap.parse_args()

    print(f"=== correctness check ===")
    print(f"  vanilla : {args.vanilla}")
    print(f"  kvd     : {args.kvd}")
    print(f"  c={args.c}  iters={args.iters}  max_tokens={args.max_tokens}")
    print(f"  prefix={args.shared_prefix_tokens} tok / req")
    print()

    any_diverged = False
    for it in range(args.iters):
        print(f"--- iter {it} ---", flush=True)
        va  = await run_iter(args.vanilla, args.model, args.c,
                             args.shared_prefix_tokens, args.max_tokens)
        kvd = await run_iter(args.kvd,     args.model, args.c,
                             args.shared_prefix_tokens, args.max_tokens)
        match, total, diffs = compare(va, kvd)
        rate = 100.0 * match / max(1, total)
        marker = "✓" if match == total else "✗"
        print(f"  {marker} {match}/{total} match ({rate:.1f}%)")
        if diffs:
            any_diverged = True
            for d in diffs:
                print(d)

    sys.exit(1 if any_diverged else 0)


if __name__ == "__main__":
    asyncio.run(main())
