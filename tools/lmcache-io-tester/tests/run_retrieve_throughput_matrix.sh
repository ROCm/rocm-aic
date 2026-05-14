#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# A/B matrix for retrieve-only throughput (manual / lab).
# Run from tools/lmcache-io-tester with venv active or set PY.
#
# Environment:
#   LMCACHE_IO_MATRIX_ROOT       Base directory (default: mktemp under /tmp)
#   LMCACHE_IO_MATRIX_OPS_STORE  Store ops (default: 400)
#   LMCACHE_IO_MATRIX_OPS_RETR   Retrieve ops per case (default: 200)
#   PY                           Python (default: python3)

set -euo pipefail

ROOT="${LMCACHE_IO_MATRIX_ROOT:-}"
if [[ -z "$ROOT" ]]; then
  ROOT="$(mktemp -d /tmp/lmcache-io-matrix.XXXXXX)"
fi
OPS_STORE="${LMCACHE_IO_MATRIX_OPS_STORE:-400}"
OPS_RETR="${LMCACHE_IO_MATRIX_OPS_RETRIEVE:-200}"
PY="${PY:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG="$TOOL_ROOT/configs/lmcache-config.yml"

run_store() {
  local d="$1"
  shift
  rm -rf "$d"
  mkdir -p "$d"
  ( cd "$TOOL_ROOT" && "$PY" -m src.lmcache-sim run \
      --storage-type filesystem \
      --storage-path "$d" \
      --device cpu \
      --config "$CFG" \
      --pattern store-only \
      --num-operations "$OPS_STORE" "$@" )
}

run_retrieve_json() {
  local name="$1"
  local dir="$2"
  shift 2
  echo "=== $name ===" >&2
  ( cd "$TOOL_ROOT" && "$PY" -m src.lmcache-sim run \
      --storage-type filesystem \
      --storage-path "$dir" \
      --device cpu \
      --config "$CFG" \
      --pattern retrieve-only \
      --num-operations "$OPS_RETR" \
      --output-format json "$@" ) > "$ROOT/${name}.json"
  echo >&2
}

echo "Matrix root: $ROOT"
echo "Store ops: $OPS_STORE  Retrieve ops per case: $OPS_RETR"
echo

D1="$ROOT/case_default"
run_store "$D1"
run_retrieve_json "default_kv_float16" "$D1"

D2="$ROOT/case_odirect"
run_store "$D2"
run_retrieve_json "fs_odirect" "$D2" --fs-odirect

D3="$ROOT/case_chunk128"
run_store "$D3" --chunk-size 128 --kv-shape "2,2,128,4,16"
run_retrieve_json "chunk128_kv_match" "$D3" --chunk-size 128 --kv-shape "2,2,128,4,16"

echo "Summary (throughput_ops_per_sec):"
for f in "$ROOT"/*.json; do
  [[ -f "$f" ]] || continue
  tp="$("$PY" -c "
import json, sys
from pathlib import Path
raw = Path(sys.argv[1]).read_text(encoding='utf-8')
i = raw.find('{')
if i < 0:
    sys.stderr.write('no json in file: ' + sys.argv[1] + chr(10))
    sys.exit(1)
obj, _ = json.JSONDecoder().raw_decode(raw[i:])
print(obj['throughput_ops_per_sec'])
" "$f")"
  echo "$(basename "$f" .json): $tp"
done
