#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

set -euo pipefail

DATA_DIR="/mnt/rocm-icms-cache/stebates/lmcache-io-tester/data"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
PER_OP_DIR="${DATA_DIR%/}/per-op-logs/${RUN_ID}"
mkdir -p "${PER_OP_DIR}"

python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path "${DATA_DIR}" \
    --pattern store-only \
    --num-operations 65536 \
    --per-op-log "${PER_OP_DIR}/01-fs-store.jsonl"

sleep 3

python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path "${DATA_DIR}" \
    --pattern lookup-only \
    --duration 60 \
    --per-op-log "${PER_OP_DIR}/02-fs-lookup.jsonl"

sleep 3

python -m src.lmcache-sim run \
    --storage-type filesystem \
    --storage-path "${DATA_DIR}" \
    --pattern retrieve-only \
    --duration 120 \
    --per-op-log "${PER_OP_DIR}/03-fs-retrieve.jsonl"

sleep 3

python -m src.lmcache-sim run \
    --storage-type local-disk \
    --storage-path "${DATA_DIR}" \
    --pattern store-only \
    --num-operations 16384 \
    --per-op-log "${PER_OP_DIR}/04-local-store.jsonl"

sleep 3

python -m src.lmcache-sim run \
    --storage-type local-disk \
    --storage-path "${DATA_DIR}" \
    --pattern retrieve-only \
    --duration 120 \
    --per-op-log "${PER_OP_DIR}/05-local-retrieve.jsonl"

echo "Per-op JSONL under: ${PER_OP_DIR}"
