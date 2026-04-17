#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Download the benchmark corpus from Project Gutenberg and
# concatenate into configs/books.txt.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${SCRIPT_DIR}/../configs/books.txt"

URLS=(
  "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"
  "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"
)

if [ -f "$OUT" ]; then
  echo "corpus already exists: $OUT"
  exit 0
fi

echo "downloading benchmark corpus ..."
: > "$OUT"
for url in "${URLS[@]}"; do
  echo "  $url"
  curl -fsSL "$url" >> "$OUT"
done
echo "wrote $(wc -l < "$OUT") lines to $OUT"
