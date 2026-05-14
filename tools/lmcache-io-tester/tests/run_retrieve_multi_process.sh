#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Aggregate retrieve-only IOPS using N independent processes. Default: each
# worker gets a copy of the populated cache (disjoint --storage-path). For
# filesystem backends LMCache supports read-only retrieve from the same tree
# across processes (one engine per process); you may point all workers at
# $GOLD instead of cp -a to stress one on-disk cache. Thread safety of a
# single LMCacheEngine is not assumed.
#
# Usage (from tools/lmcache-io-tester, venv active):
#   tests/run_retrieve_multi_process.sh
#
# Environment:
#   LMCACHE_IO_MP_WORKERS   Number of parallel workers (default: 4)
#   LMCACHE_IO_MP_OPS_STORE  Store ops per golden dir (default: 200)
#   LMCACHE_IO_MP_OPS_RETR   Retrieve ops per worker (default: 100)
#   PY                       Python (default: python3)

set -euo pipefail

WORKERS="${LMCACHE_IO_MP_WORKERS:-4}"
OPS_STORE="${LMCACHE_IO_MP_OPS_STORE:-200}"
OPS_RETR="${LMCACHE_IO_MP_OPS_RETRIEVE:-100}"
PY="${PY:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG="$TOOL_ROOT/configs/lmcache-config.yml"
ROOT="$(mktemp -d /tmp/lmcache-io-mp.XXXXXX)"
GOLD="$ROOT/golden"

mkdir -p "$GOLD"
( cd "$TOOL_ROOT" && "$PY" -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path "$GOLD" \
    --device cpu \
    --config "$CFG" \
    --pattern store-only \
    --num-operations "$OPS_STORE" )

pids=()
for ((i=0; i<WORKERS; i++)); do
  WDIR="$ROOT/w$i"
  rm -rf "$WDIR"
  cp -a "$GOLD" "$WDIR"
  (
    cd "$TOOL_ROOT" && "$PY" -m src.lmcache-sim run \
      --storage-type filesystem \
      --storage-path "$WDIR" \
      --device cpu \
      --config "$CFG" \
      --pattern retrieve-only \
      --num-operations "$OPS_RETR" \
      --output-format json >"$ROOT/out$i.json"
  ) &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "Per-worker throughput_ops_per_sec (retrieve-only):"
sum=0
for ((i=0; i<WORKERS; i++)); do
  tp="$("$PY" -c "
import json, sys
from pathlib import Path
raw = Path(sys.argv[1]).read_text(encoding='utf-8')
i = raw.find('{')
if i < 0:
    sys.exit(1)
obj, _ = json.JSONDecoder().raw_decode(raw[i:])
print(obj['throughput_ops_per_sec'])
" "$ROOT/out$i.json")"
  echo "  worker $i: $tp"
  sum="$("$PY" -c "print(float(sys.argv[1]) + float(sys.argv[2]))" "$sum" "$tp")"
done
echo "Sum of worker throughputs (aggregate RPS across processes): $sum"
echo "Artifacts under: $ROOT"
