#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Overnight AgentX concurrency sweep.
#
# Runs run_agentx.sh at each concurrency level in AGENTX_CONCURRENCIES,
# collecting results in separate timestamped subdirectories.  The inference
# server and monitoring stack must already be running.
#
# Usage:
#   nohup ./benchmarks/agentx/overnight_sweep.sh > /tmp/agentx-sweep.log 2>&1 &
#
# Or via make:
#   make agentx-sweep VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
#
# Environment: inherits everything run_agentx.sh reads (BENCH_ENDPOINT,
# AGENTX_MODEL, AGENTX_DURATION, AGENTX_MAX_CONTEXT, AGENTX_OTEL_URL,
# AGENTX_IMAGE, HF_TOKEN, HF_HOME, LOG).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${AGENTX_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DURATION="${AGENTX_DURATION:-900}"
MAX_CONTEXT="${AGENTX_MAX_CONTEXT:-65536}"

# Space-separated concurrency levels to sweep
AGENTX_CONCURRENCIES="${AGENTX_CONCURRENCIES:-1 2 4}"

echo "=== AgentX Overnight Sweep — $(date) ==="
echo "Model:         ${MODEL}"
echo "Duration:      ${DURATION}s"
echo "Max context:   ${MAX_CONTEXT} tokens"
echo "Concurrencies: ${AGENTX_CONCURRENCIES}"
echo ""

# shellcheck disable=SC2086  # word-splitting of AGENTX_CONCURRENCIES is intentional
for C in ${AGENTX_CONCURRENCIES}; do
  echo "────────────────────────────────────────"
  echo "concurrency=${C}  $(date)"
  echo ""

  AGENTX_CONCURRENCY="${C}" \
  AGENTX_MODEL="${MODEL}" \
  AGENTX_DURATION="${DURATION}" \
  AGENTX_MAX_CONTEXT="${MAX_CONTEXT}" \
    bash "${SCRIPT_DIR}/run_agentx.sh"

  echo ""
  echo "Completed concurrency=${C} at $(date)"
  # Brief pause to let the inference server KV cache settle between runs
  sleep 30
done

echo ""
echo "=== Sweep complete — $(date) ==="
