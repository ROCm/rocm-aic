#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# AgentX concurrency sweep: one run_agentx.sh run per concurrency level in
# AGENTX_CONCURRENCIES, each in its own subdirectory of one sweep directory
# (<LOG>/agentx/sweep_<timestamp>/c<N>/).  The stack must already be running.
#
# Usage:
#   nohup ./benchmarks/agentx/overnight_sweep.sh > /tmp/agentx-sweep.log 2>&1 &
#
# Or via make:
#   make agentx-sweep
#
# Options are run_agentx.sh's and are passed to every run (--dry-run prints the
# commands).  The sweep sets --concurrency and --output-dir itself.
#
# Environment:
#   AGENTX_CONCURRENCIES  Space-separated concurrency levels (default: 1 2 4)
#   LOG                   Log root (default: <repo>/logs)
#   everything run_agentx.sh reads: the endpoint, model, duration, context
#   limit, dataset or corpus, seed and image apply to every level.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONCURRENCIES="${AGENTX_CONCURRENCIES:-1 2 4}"
SWEEP_DIR="${LOG:-${REPO_ROOT}/logs}/agentx/sweep_$(date +%Y%m%dT%H%M%S)"
# Lets the server's KV cache settle between levels.
PAUSE_S=30

DRY_RUN="false"
for _arg in "$@"; do
  [[ "${_arg}" != "--dry-run" ]] || DRY_RUN="true"
done

echo "=== AgentX sweep — $(date) ==="
echo "Concurrencies: ${CONCURRENCIES}"
echo "Sweep dir:     ${SWEEP_DIR}"
echo ""

_first="true"
_failed=()
# shellcheck disable=SC2086  # a space-separated list, split on purpose
for C in ${CONCURRENCIES}; do
  if [[ "${_first}" != "true" && "${DRY_RUN}" != "true" ]]; then
    sleep "${PAUSE_S}"
  fi
  _first="false"
  echo "────────────────────────────────────────"
  echo "concurrency=${C}  $(date)"
  echo ""

  # A failed level does not end the sweep: the remaining ones still run.
  if bash "${SCRIPT_DIR}/run_agentx.sh" --concurrency "${C}" --output-dir "${SWEEP_DIR}/c${C}" "$@"; then
    echo "Completed concurrency=${C} at $(date)"
  else
    echo "FAILED concurrency=${C} at $(date)" >&2
    _failed+=("${C}")
  fi
  echo ""
done

echo "=== Sweep complete — $(date) ==="
if (( ${#_failed[@]} )); then
  echo "Failed concurrency levels: ${_failed[*]}" >&2
  exit 1
fi
echo "Results: ${SWEEP_DIR}"
