#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Run retrieve-only against an existing LMCache sidecar under DATA_DIR
# (populate first with store-only / lookup-only as needed). Default 120 s.

set -euo pipefail

DATA_DIR="/mnt/rocm-icms-cache/stebates/lmcache-io-tester/data"
DURATION_SEC="${RETRIEVE_DURATION_SEC:-120}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
PER_OP_DIR="${DATA_DIR%/}/per-op-logs/${RUN_ID}"
mkdir -p "${PER_OP_DIR}"

python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path "${DATA_DIR}" \
    --pattern retrieve-only \
    --duration "${DURATION_SEC}" \
    --per-op-log "${PER_OP_DIR}/retrieve-only.jsonl"

echo "Per-op JSONL: ${PER_OP_DIR}/retrieve-only.jsonl"
