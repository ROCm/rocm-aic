#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Run verify against multiple backends. Filesystem always runs locally.
# Optional: export REDIS_URL=redis://127.0.0.1:6379 for redis row.
# S3 requires credentials and bucket via env / extra-config (not auto-run).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY=(python -m src.lmcache-sim)

run_verify() {
  local label="$1"
  shift
  echo "=== ${label} ==="
  if "${PY[@]}" verify "$@"; then
    echo "OK ${label}"
  else
    echo "FAIL ${label}" >&2
    return 1
  fi
}

FAILURES=0

TMP="${TMPDIR:-/tmp}/lmcache-io-matrix-$$"
mkdir -p "$TMP"
cleanup() {
  rm -rf "$TMP"
}
trap cleanup EXIT

run_verify "filesystem" \
  --storage-type filesystem \
  --storage-path "$TMP/fs" || FAILURES=$((FAILURES + 1))

if [[ -n "${REDIS_URL:-}" ]]; then
  run_verify "redis" \
    --storage-type redis \
    --remote-url "$REDIS_URL" \
    --probe-remote || FAILURES=$((FAILURES + 1))
else
  echo "=== redis (skipped; set REDIS_URL to enable) ==="
fi

if [[ "${RUN_S3_VERIFY:-}" == "1" ]] && \
   [[ -n "${S3_BUCKET_URL:-}" ]] && \
   [[ -n "${S3_REGION:-}" ]]; then
  run_verify "s3" \
    --storage-type s3 \
    --remote-url "$S3_BUCKET_URL" \
    --s3-region "$S3_REGION" || FAILURES=$((FAILURES + 1))
else
  echo "=== s3 (skipped; set RUN_S3_VERIFY=1 S3_BUCKET_URL s3://... " \
       "S3_REGION to enable) ==="
fi

if [[ "${FAILURES}" -ne 0 ]]; then
  echo "${FAILURES} backend(s) failed" >&2
  exit 1
fi
echo "All matrix steps passed."
