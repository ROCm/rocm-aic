#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Measure Time-To-First-Token (TTFT) against a llama-server instance.

Sends a single long-context chat completion request in streaming mode
and records the wall-clock time from request submission to the first
SSE content token.  Optionally saves or restores a llama-server slot
before/after the measurement via the /slots REST API.

Results are appended to a JSON-lines file so multiple invocations
(cold, warm) accumulate in one place.
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
from urllib.request import Request, urlopen
from urllib.error import URLError

from openai import OpenAI


def build_prompt(corpus_path: Path, context_chars: int,
                 seed: int) -> str:
    """Return a deterministic excerpt from the corpus.

    llama-server tokenises internally, so we work in characters
    rather than token IDs.  *context_chars* is an approximate
    target; the actual token count depends on the model's
    vocabulary.
    """
    raw = corpus_path.read_text(encoding="utf-8", errors="replace")

    rng = random.Random(seed)
    max_offset = max(0, len(raw) - context_chars)
    offset = rng.randint(0, max_offset) if max_offset > 0 else 0

    excerpt = raw[offset:offset + context_chars]
    question = "Summarize the above text in two sentences."
    return f"{excerpt}\n\n{question}"


def measure_ttft(client: OpenAI, model: str,
                 prompt: str) -> tuple[float, str]:
    """Send a streaming chat completion and return (ttft_seconds, reply)."""
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


def slot_action(server_url: str, slot_id: int,
                action: str, filename: str) -> None:
    """POST /slots/{slot_id}?action={save|restore} to llama-server."""
    url = f"{server_url}/slots/{slot_id}?action={action}"
    body = json.dumps({"filename": filename}).encode()
    req = Request(url, data=body,
                  headers={"Content-Type": "application/json"},
                  method="POST")
    try:
        with urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            print(f"[bench_ttft] slot {action}: {result}")
    except URLError as exc:
        print(f"[bench_ttft] WARNING: slot {action} failed: {exc}",
              file=sys.stderr)
        raise


def detect_model(client: OpenAI) -> str:
    models = client.models.list()
    if not models.data:
        raise RuntimeError("No models available on the server")
    return models.data[0].id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TTFT benchmark for llama-server")
    parser.add_argument("--server-url", default="http://localhost:8080",
                        help="llama-server endpoint")
    parser.add_argument("--model", default=None,
                        help="Model name (auto-detected if omitted)")
    parser.add_argument("--corpus-file", default="/app/corpus.txt",
                        help="Path to the text corpus")
    parser.add_argument("--context-chars", type=int, default=40000,
                        help="Approximate context size in characters")
    parser.add_argument("--seed", type=int, default=42,
                        help="PRNG seed for corpus excerpt selection")
    parser.add_argument("--output", default="/app/results.jsonl",
                        help="JSON-lines output file (appended)")
    parser.add_argument("--tag", default="untagged",
                        help="Label for this measurement point")
    parser.add_argument("--save-slot", default=None, metavar="FILENAME",
                        help="After measurement, save slot 0 to FILENAME")
    parser.add_argument("--restore-slot", default=None, metavar="FILENAME",
                        help="Before measurement, restore slot 0 from FILENAME")
    parser.add_argument("--slot-id", type=int, default=0,
                        help="Slot ID to save/restore (default: 0)")
    args = parser.parse_args()

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "dummy-key"),
        base_url=f"{args.server_url}/v1",
    )

    model = args.model or detect_model(client)

    print(f"[bench_ttft] tag={args.tag}  model={model}  "
          f"context_chars={args.context_chars}  seed={args.seed}")

    if args.restore_slot:
        print(f"[bench_ttft] restoring slot {args.slot_id} "
              f"from {args.restore_slot}")
        slot_action(args.server_url, args.slot_id,
                    "restore", args.restore_slot)

    prompt = build_prompt(Path(args.corpus_file),
                          args.context_chars, args.seed)

    print(f"[bench_ttft] prompt length: {len(prompt)} chars")

    ttft_s, reply = measure_ttft(client, model, prompt)
    ttft_ms = ttft_s * 1000.0

    print(f"[bench_ttft] TTFT = {ttft_ms:.1f} ms")
    print(f"[bench_ttft] reply preview: {reply[:120]}...")

    if args.save_slot:
        print(f"[bench_ttft] saving slot {args.slot_id} "
              f"to {args.save_slot}")
        slot_action(args.server_url, args.slot_id,
                    "save", args.save_slot)

    record = {
        "tag": args.tag,
        "model": model,
        "context_chars": len(prompt),
        "seed": args.seed,
        "ttft_ms": round(ttft_ms, 2),
        "restored_from": args.restore_slot,
        "saved_to": args.save_slot,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    print(f"[bench_ttft] result appended to {out_path}")


if __name__ == "__main__":
    main()
