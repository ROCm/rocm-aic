#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Serial CC trace replay loop. Prints one JSON object per request to stdout.
# Env: BASE_URL, MODEL, AGENTX_DATA_ROOT, ITERATIONS, AGENTX_SEED,
#      AGENTX_MAX_REQUESTS, MAX_TOKENS, AGENTX_DRY_RUN, AGENTX_HONOR_THINK_TIME,
#      RUN_AGENT_WORKER, RUN_AGENT_REPLAY (override replay-trace.py path).

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./benchmarks/llm-agentx/lib/bench-root.sh
source "${here}/lib/bench-root.sh"
BENCH_ROOT="${LLM_AGENTX_BENCH_ROOT:-${here}}"
REPLAY="${RUN_AGENT_REPLAY:-${BENCH_ROOT}/scripts/replay-trace.py}"
_default_python() {
	local repo_root
	repo_root="$(cd "${BENCH_ROOT}/../.." && pwd)"
	if [[ -x "${repo_root}/.venv/bin/python3" ]]; then
		printf '%s/.venv/bin/python3' "${repo_root}"
		return 0
	fi
	printf 'python3'
}
PYTHON="${PYTHON:-$(_default_python)}"

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
_default_data_root() {
	local repo_root
	repo_root="$(cd "${BENCH_ROOT}/../.." && pwd)"
	if [[ -d "${repo_root}/data/cc-traces" ]]; then
		printf '%s/data/cc-traces' "${repo_root}"
	else
		printf '%s/tests/fixtures' "${BENCH_ROOT}"
	fi
}
AGENTX_DATA_ROOT="${AGENTX_DATA_ROOT:-$(_default_data_root)}"
ITERATIONS="${ITERATIONS:-1}"
MAX_TOKENS="${MAX_TOKENS:-512}"
AGENTX_DRY_RUN="${AGENTX_DRY_RUN:-0}"
AGENTX_HONOR_THINK_TIME="${AGENTX_HONOR_THINK_TIME:-0}"
AGENTX_STRICT="${AGENTX_STRICT:-0}"
RUN_AGENT_WORKER="${RUN_AGENT_WORKER:-}"

if ! [[ "${ITERATIONS}" =~ ^[1-9][0-9]*$ ]]; then
	echo "error: ITERATIONS must be a positive integer (got ${ITERATIONS})" >&2
	exit 1
fi

if [[ -n "${AGENTX_SEED+x}" && -n "${AGENTX_SEED}" ]]; then
	if ! [[ "${AGENTX_SEED}" =~ ^[0-9]+$ ]]; then
		echo "error: AGENTX_SEED must be a non-negative integer (got ${AGENTX_SEED})" >&2
		exit 1
	fi
	SEED="${AGENTX_SEED}"
	_seed_log=" seed=${AGENTX_SEED}"
else
	SEED="${RANDOM}"
	_seed_log=" seed=${SEED} (random)"
fi

_worker_log=""
if [[ -n "${RUN_AGENT_WORKER}" ]]; then
	_worker_log=" worker=${RUN_AGENT_WORKER}"
fi

_replay_opts=(
	--url "${BASE_URL}"
	--data-root "${AGENTX_DATA_ROOT}"
	--model "${MODEL}"
	--max-tokens "${MAX_TOKENS}"
	--count 1
)

if [[ -n "${AGENTX_MAX_REQUESTS:-}" ]]; then
	_replay_opts+=(--max-requests "${AGENTX_MAX_REQUESTS}")
fi
if [[ "${AGENTX_DRY_RUN}" == "1" ]]; then
	_replay_opts+=(--dry-run --skip-health-check)
fi
if [[ "${AGENTX_HONOR_THINK_TIME}" == "1" ]]; then
	_replay_opts+=(--honor-think-time)
fi
if [[ -n "${AGENTX_MAX_CONTEXT:-}" ]]; then
	_replay_opts+=(--max-context "${AGENTX_MAX_CONTEXT}")
fi
if [[ "${AGENTX_STRICT}" == "1" ]]; then
	_replay_opts+=(--strict)
fi

echo "run-agent: iterations=${ITERATIONS}${_worker_log}${_seed_log} data=${AGENTX_DATA_ROOT}" >&2

for ((i = 1; i <= ITERATIONS; i++)); do
	iter_seed=$((SEED + i - 1))
	echo "run-agent: iteration=${i}/${ITERATIONS} seed=${iter_seed}" >&2
	tmp="$(mktemp)"
	"${PYTHON}" "${REPLAY}" \
		"${_replay_opts[@]}" \
		--seed "${iter_seed}" \
		-o "${tmp}"

	while IFS= read -r line; do
		[[ -z "${line}" ]] && continue
		if [[ -n "${RUN_AGENT_WORKER}" ]]; then
			"${PYTHON}" -c "
import json, sys
row = json.loads(sys.argv[1])
row['run_agent_iteration'] = int(sys.argv[2])
row['run_agent_seed'] = int(sys.argv[3])
row['run_agent_worker'] = int(sys.argv[4])
print(json.dumps(row, ensure_ascii=False))
" "${line}" "${i}" "${iter_seed}" "${RUN_AGENT_WORKER}"
		else
			"${PYTHON}" -c "
import json, sys
row = json.loads(sys.argv[1])
row['run_agent_iteration'] = int(sys.argv[2])
row['run_agent_seed'] = int(sys.argv[3])
print(json.dumps(row, ensure_ascii=False))
" "${line}" "${i}" "${iter_seed}"
		fi
	done <"${tmp}"
	rm -f "${tmp}"
done
