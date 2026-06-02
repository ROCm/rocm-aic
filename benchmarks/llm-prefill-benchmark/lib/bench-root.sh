#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Resolve LLM_PREFILL_BENCH_ROOT (benchmarks/llm-prefill-benchmark).
_llm_prefill_bench_root() {
	if [[ -n "${LLM_PREFILL_BENCH_ROOT:-}" ]]; then
		printf '%s' "${LLM_PREFILL_BENCH_ROOT}"
		return 0
	fi
	local here repo
	here="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
	repo="$(cd "${here}/../../.." && pwd)"
	printf '%s/benchmarks/llm-prefill-benchmark' "${repo}"
}
