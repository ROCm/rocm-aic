#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Launch N parallel run-agent.sh workers with distinct AGENTX_SEED values.
#
# Usage:
#   ./run-agent-parallel.sh [WORKERS]
#
# Wrapper env: WORKERS, BASE_SEED, OUTPUT_DIR, STAGGER_SEC, ITERATIONS
#
# SIGINT/SIGTERM stops benchmark workers only (does not propagate to the
# shell's other jobs). Start vLLM detached (e.g. make run-batch) rather than
# interactive docker run -it in the same shell session.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_agent="${here}/run-agent.sh"

if [[ ! -x "${run_agent}" ]]; then
	echo "error: missing or non-executable ${run_agent}" >&2
	exit 1
fi

WORKERS="${1:-${WORKERS:-4}}"
BASE_SEED="${BASE_SEED:-$RANDOM}"
OUTPUT_DIR="${OUTPUT_DIR:-${here}/logs/run-agent-parallel}"
STAGGER_SEC="${STAGGER_SEC:-0}"
ITERATIONS="${ITERATIONS:-1}"

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
	echo "error: WORKERS must be a positive integer (got ${WORKERS})" >&2
	exit 1
fi
if ! [[ "${BASE_SEED}" =~ ^[0-9]+$ ]]; then
	echo "error: BASE_SEED must be a non-negative integer (got ${BASE_SEED})" >&2
	exit 1
fi

mkdir -p "${OUTPUT_DIR}"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="${OUTPUT_DIR}/${stamp}"
mkdir -p "${run_dir}"

echo "run-agent-parallel: workers=${WORKERS} base_seed=${BASE_SEED} iterations=${ITERATIONS}" >&2
echo "run-agent-parallel: output=${run_dir}" >&2

failed=0
pids=()
_interrupted=0

_stop_workers() {
	local w pid
	for ((w = 0; w < ${#pids[@]}; w++)); do
		pid="${pids[w]}"
		if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
			kill -TERM "${pid}" 2>/dev/null || true
		fi
	done
}

_on_interrupt() {
	_interrupted=1
	echo "run-agent-parallel: interrupted — stopping workers (vLLM/docker untouched)" >&2
	_stop_workers
}

trap '_on_interrupt' INT TERM

for ((w = 0; w < WORKERS; w++)); do
	seed=$((BASE_SEED + w))
	out_jsonl="${run_dir}/worker-${w}.jsonl"
	out_log="${run_dir}/worker-${w}.log"

	(
		export AGENTX_SEED="${seed}"
		export RUN_AGENT_WORKER="${w}"
		exec bash "${run_agent}"
	) >"${out_jsonl}" 2>"${out_log}" &
	pid=$!
	pids+=("${pid}")
	echo "run-agent-parallel: started worker=${w} pid=${pid} seed=${seed}" >&2

	if [[ "${w}" -lt $((WORKERS - 1)) && "${STAGGER_SEC}" != "0" ]]; then
		sleep "${STAGGER_SEC}"
	fi
done

for ((w = 0; w < WORKERS; w++)); do
	pid="${pids[w]}"
	if wait "${pid}"; then
		echo "run-agent-parallel: worker=${w} pid=${pid} ok" >&2
	else
		rc=$?
		if [[ "${_interrupted}" -eq 1 && "${rc}" -gt 128 ]]; then
			echo "run-agent-parallel: worker=${w} pid=${pid} stopped" >&2
		else
			echo "run-agent-parallel: worker=${w} pid=${pid} FAILED (see ${run_dir}/worker-${w}.log)" >&2
			failed=$((failed + 1))
		fi
	fi
done

trap - INT TERM

echo "run-agent-parallel: done failed=${failed}/${WORKERS} dir=${run_dir}" >&2
if [[ "${_interrupted}" -eq 1 ]]; then
	exit 130
fi
if [[ "${failed}" -gt 0 ]]; then
	exit 1
fi
