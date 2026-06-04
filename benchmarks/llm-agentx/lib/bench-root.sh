#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Resolve LLM_AGENTX_BENCH_ROOT (benchmarks/llm-agentx).
_llm_agentx_bench_root() {
	if [[ -n "${LLM_AGENTX_BENCH_ROOT:-}" ]]; then
		printf '%s' "${LLM_AGENTX_BENCH_ROOT}"
		return 0
	fi
	local lib_dir repo
	lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	repo="$(cd "${lib_dir}/../.." && pwd)"
	printf '%s/benchmarks/llm-agentx' "${repo}"
}
