#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Measure Time-To-First-Token (TTFT) against a vLLM + LMCache server.

Sends a single long-context chat completion request in streaming mode
and records the wall-clock time from request submission to the first
SSE content token.  Results are appended to a JSON-lines file so
multiple invocations (warmup, sweep points) accumulate in one place.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from transformers import AutoTokenizer


def build_prompt(corpus_path: Path, context_tokens: int,
                 seed: int, tokenizer_name: str) -> tuple[str, int]:
    """Return (prompt_text, actual_token_count).

    A deterministic excerpt is selected from the corpus using *seed*
    to choose the character offset.  The excerpt is tokenized and
    truncated to *context_tokens* tokens, then decoded back to text
    so the exact token boundary is respected.
    """
    raw = corpus_path.read_text(encoding="utf-8", errors="replace")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                              trust_remote_code=True)

    rng = random.Random(seed)
    max_offset = max(0, len(raw) - context_tokens * 6)
    offset = rng.randint(0, max_offset) if max_offset > 0 else 0

    excerpt = raw[offset:]
    token_ids = tokenizer.encode(excerpt, add_special_tokens=False)
    token_ids = token_ids[:context_tokens]
    context_text = tokenizer.decode(token_ids,
                                    skip_special_tokens=True)

    question = "Summarize the above text in two sentences."
    prompt = f"{context_text}\n\n{question}"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return prompt, len(prompt_ids)


def measure_ttft(client: OpenAI, model: str,
                 prompt: str) -> tuple[float, str]:
    """Send a streaming chat completion and return (ttft_seconds, reply).

    TTFT is the wall-clock delta between the moment the request is
    submitted and the first non-empty content chunk arriving.
    """
    start = time.perf_counter()
    ttft: float | None = None
    fragments: list[str] = []

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=64,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta is not None:
            if ttft is None:
                ttft = time.perf_counter() - start
            fragments.append(delta)

    if ttft is None:
        raise RuntimeError("Server returned no content tokens")

    return ttft, "".join(fragments)


def detect_model(client: OpenAI) -> str:
    models = client.models.list()
    if not models.data:
        raise RuntimeError("No models available on the server")
    return models.data[0].id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TTFT benchmark for vLLM + LMCache")
    parser.add_argument("--server-url", default="http://localhost:8000",
                        help="vLLM OpenAI-compatible endpoint")
    parser.add_argument("--model", default=None,
                        help="Model name (auto-detected if omitted)")
    parser.add_argument("--corpus-file",
                        default="/app/configs/books.txt",
                        help="Path to the text corpus")
    parser.add_argument("--context-tokens", type=int, default=10000,
                        help="Number of context tokens to send")
    parser.add_argument("--seed", type=int, default=42,
                        help="PRNG seed for corpus excerpt selection")
    parser.add_argument("--output", default="/app/results.jsonl",
                        help="JSON-lines output file (appended)")
    parser.add_argument("--tag", default="untagged",
                        help="Label for this measurement point")
    args = parser.parse_args()

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "dummy-key"),
        base_url=f"{args.server_url}/v1",
    )

    model = args.model or detect_model(client)
    tokenizer_name = model

    print(f"[bench_ttft] tag={args.tag}  model={model}  "
          f"context_tokens={args.context_tokens}  seed={args.seed}")

    prompt, actual_tokens = build_prompt(
        Path(args.corpus_file), args.context_tokens,
        args.seed, tokenizer_name)

    print(f"[bench_ttft] actual prompt tokens: {actual_tokens}")

    ttft_s, reply = measure_ttft(client, model, prompt)
    ttft_ms = ttft_s * 1000.0

    print(f"[bench_ttft] TTFT = {ttft_ms:.1f} ms")
    print(f"[bench_ttft] reply preview: {reply[:120]}...")

    record = {
        "tag": args.tag,
        "model": model,
        "context_tokens": actual_tokens,
        "seed": args.seed,
        "ttft_ms": round(ttft_ms, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    print(f"[bench_ttft] result appended to {out_path}")


if __name__ == "__main__":
    main()
