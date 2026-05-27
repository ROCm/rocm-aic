#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Shared benchmark path helpers for recipe Slurm jobs.

_recipe_llm_prefill_bench_root() {
	if [[ -n "${LLM_PREFILL_BENCH_ROOT:-}" ]]; then
		printf '%s' "${LLM_PREFILL_BENCH_ROOT}"
		return 0
	fi
	if [[ -n "${REPO_DIR:-}" ]]; then
		printf '%s/benchmarks/llm-prefill-benchmark' "${REPO_DIR}"
		return 0
	fi
	printf '%s' "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/benchmarks/llm-prefill-benchmark"
}

_recipe_export_bench_root() {
	export LLM_PREFILL_BENCH_ROOT="$(_recipe_llm_prefill_bench_root)"
}
