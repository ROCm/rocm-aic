#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Build chunked context fixtures and questions for a Gutenberg book library."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def chunk_label(chunk_words: int) -> str:
    if chunk_words >= 1000 and chunk_words % 1000 == 0:
        return f"{chunk_words // 1000}k"
    return str(chunk_words)


def load_manifest(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    books = data.get("books")
    if not isinstance(books, list) or not books:
        raise ValueError(f"{path}: 'books' must be a non-empty array")
    return data


def book_dir_complete(
    book_dir: Path,
    slug: str,
    chunk_label_str: str,
    chunk_count: int,
    question_count: int,
) -> bool:
    chunks = list(book_dir.glob(f"{slug}-{chunk_label_str}.*.txt"))
    if len(chunks) < chunk_count:
        return False
    questions_path = book_dir / f"{slug}.questions.json"
    if not questions_path.is_file():
        return False
    try:
        payload = json.loads(questions_path.read_text(encoding="utf-8"))
        q = payload.get("questions")
        if not isinstance(q, list) or len(q) < question_count:
            return False
    except (json.JSONDecodeError, OSError):
        return False
    return True


def run_step(cmd: list[str], label: str) -> None:
    print(f"  {label}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def build_book(
    *,
    recipe_root: Path,
    python: str,
    data_root: Path,
    book: dict[str, object],
    chunk_words: int,
    chunk_count: int,
    question_count: int,
    skip_existing: bool,
) -> str:
    slug = str(book["slug"]).strip()
    pg_id = int(book["pg_id"])
    title = str(book["title"])
    author = str(book.get("author", "the author"))
    label = chunk_label(chunk_words)
    book_dir = data_root / slug

    if skip_existing and book_dir_complete(
        book_dir, slug, label, chunk_count, question_count
    ):
        return "skipped"

    book_dir.mkdir(parents=True, exist_ok=True)
    split_script = recipe_root / "scripts" / "split-gutenberg-random-chunks.py"
    questions_script = recipe_root / "scripts" / "gen-questions-json.py"

    run_step(
        [
            python,
            str(split_script),
            "--pg-id",
            str(pg_id),
            "--slug",
            slug,
            "-o",
            str(book_dir),
            "--chunk-words",
            str(chunk_words),
            "--count",
            str(chunk_count),
        ],
        "split",
    )
    run_step(
        [
            python,
            str(questions_script),
            "--slug",
            slug,
            "--title",
            title,
            "--author",
            author,
            "--pg-id",
            str(pg_id),
            "--count",
            str(question_count),
        ],
        "questions",
    )

    if not book_dir_complete(book_dir, slug, label, chunk_count, question_count):
        raise RuntimeError(f"incomplete output under {book_dir}")

    return "built"


def main() -> int:
    recipe_root = Path(__file__).resolve().parents[1]
    default_manifest = recipe_root / "configs" / "gutenberg-library.json"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest,
        help=f"Library manifest JSON (default: {default_manifest.name}).",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=recipe_root / "data",
        help="Output root directory (default: data/).",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter for child scripts.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip books whose output directory is already complete.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N books from the manifest.",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=1.0,
        metavar="SEC",
        help="Seconds to sleep between books (default: 1.0).",
    )
    p.add_argument(
        "--book",
        metavar="SLUG",
        default=None,
        help="Build only the book with this slug.",
    )
    p.add_argument(
        "--allow-failures",
        action="store_true",
        help="Exit 0 even if some books fail.",
    )
    args = p.parse_args()

    try:
        manifest = load_manifest(args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    chunk_words = int(manifest.get("chunk_words", 10_000))
    chunk_count = int(manifest.get("chunk_count", 100))
    question_count = int(manifest.get("question_count", 100))
    books_raw = manifest["books"]
    assert isinstance(books_raw, list)

    books: list[dict[str, object]] = []
    for entry in books_raw:
        if not isinstance(entry, dict):
            print("error: each book entry must be an object", file=sys.stderr)
            return 1
        for key in ("pg_id", "slug", "title"):
            if key not in entry:
                print(f"error: book entry missing {key!r}: {entry}", file=sys.stderr)
                return 1
        books.append(entry)

    if args.book is not None:
        books = [b for b in books if str(b["slug"]) == args.book]
        if not books:
            print(f"error: slug not in manifest: {args.book}", file=sys.stderr)
            return 1

    if args.limit is not None:
        if args.limit < 1:
            print("error: --limit must be >= 1", file=sys.stderr)
            return 1
        books = books[: args.limit]

    args.data_root.mkdir(parents=True, exist_ok=True)

    built = skipped = failed = 0
    failures: list[str] = []

    for i, book in enumerate(books, start=1):
        slug = str(book["slug"])
        print(f"[{i}/{len(books)}] {slug} (PG#{book['pg_id']})", flush=True)
        try:
            result = build_book(
                recipe_root=recipe_root,
                python=args.python,
                data_root=args.data_root,
                book=book,
                chunk_words=chunk_words,
                chunk_count=chunk_count,
                question_count=question_count,
                skip_existing=args.skip_existing,
            )
            if result == "skipped":
                skipped += 1
                print(f"  skip: already complete", flush=True)
            else:
                built += 1
                print(f"  ok: {args.data_root / slug}", flush=True)
        except (subprocess.CalledProcessError, RuntimeError, OSError) as e:
            failed += 1
            msg = f"{slug}: {e}"
            failures.append(msg)
            print(f"  FAIL: {msg}", file=sys.stderr)

        if args.delay > 0 and i < len(books):
            time.sleep(args.delay)

    print(
        f"summary: built={built} skipped={skipped} failed={failed} "
        f"total={len(books)}",
        flush=True,
    )
    if failures:
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)

    if failed and not args.allow_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
