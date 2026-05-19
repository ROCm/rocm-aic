#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Emit fixed-length word windows from a Project Gutenberg book at random offsets."""

from __future__ import annotations

import argparse
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

GUTENBERG_START = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
GUTENBERG_END = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*",
    re.IGNORECASE | re.DOTALL,
)


def chunk_label(chunk_words: int) -> str:
    if chunk_words >= 1000 and chunk_words % 1000 == 0:
        return f"{chunk_words // 1000}k"
    return str(chunk_words)


def strip_gutenberg_boilerplate(text: str) -> str:
    m = GUTENBERG_START.search(text)
    if m:
        text = text[m.end() :]
    m = GUTENBERG_END.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


def fetch_gutenberg_text(pg_id: int, timeout: float = 120.0) -> str:
    urls = [
        f"https://www.gutenberg.org/cache/epub/{pg_id}/pg{pg_id}.txt",
        f"https://www.gutenberg.org/files/{pg_id}/{pg_id}-0.txt",
    ]
    last_err: Exception | None = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vllm-radeon/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            return strip_gutenberg_boilerplate(raw)
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
    raise RuntimeError(f"failed to download PG#{pg_id}: {last_err}") from last_err


def load_text(input_path: Path | None, pg_id: int | None) -> str:
    if input_path is not None:
        return strip_gutenberg_boilerplate(
            input_path.read_text(encoding="utf-8", errors="replace")
        )
    if pg_id is None:
        raise ValueError("provide input_path or --pg-id")
    return fetch_gutenberg_text(pg_id)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        help="Local UTF-8 text (Gutenberg boilerplate stripped if present).",
    )
    src.add_argument(
        "--pg-id",
        type=int,
        metavar="N",
        help="Project Gutenberg ebook id (downloads plain text).",
    )
    p.add_argument(
        "--slug",
        required=True,
        help="Short name for output files, e.g. war-and-peace (hyphenated).",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for <slug>-<chunk>.<offset>.txt files.",
    )
    p.add_argument(
        "--chunk-words",
        type=int,
        default=10_000,
        help="Words per chunk (default: 10000).",
    )
    p.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of chunks (default: 100).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible offsets (default: non-deterministic).",
    )
    args = p.parse_args()

    slug = args.slug.strip()
    if not slug or "/" in slug or "\\" in slug:
        print("error: --slug must be a single path-safe token", file=sys.stderr)
        return 1

    try:
        text = load_text(args.input_path, args.pg_id)
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    words = text.split()
    n = len(words)
    chunk = args.chunk_words
    if n < chunk:
        print(f"error: need at least {chunk} words, got {n}", file=sys.stderr)
        return 1

    max_start = n - chunk
    population = max_start + 1
    if args.count > population:
        print(
            f"error: cannot pick {args.count} unique offsets from {population} starts",
            file=sys.stderr,
        )
        return 1

    label = chunk_label(chunk)
    prefix = f"{slug}-{label}"

    rng = random.Random(args.seed)
    offsets = rng.sample(range(population), args.count)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for offset in offsets:
        body = " ".join(words[offset : offset + chunk])
        out = args.output_dir / f"{prefix}.{offset}.txt"
        out.write_text(body + "\n", encoding="utf-8")

    print(f"wrote {args.count} files under {args.output_dir} (prefix {prefix}.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
