# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

REPO_ROOT := $(abspath $(CURDIR)/../..)
BENCH_ROOT := $(CURDIR)
VENV_PYTHON := $(REPO_ROOT)/.venv/bin/python3
ifeq ($(wildcard $(VENV_PYTHON)),)
PYTHON ?= python3
else
PYTHON ?= $(VENV_PYTHON)
endif

KV_CACHE_TESTER_ROOT ?= $(REPO_ROOT)/data/kv-cache-tester
KV_CACHE_TESTER_CONFIG ?= $(BENCH_ROOT)/configs/trace-replay-blog.yaml
KV_CACHE_TESTER_SMOKE_CONFIG ?= $(BENCH_ROOT)/configs/smoke-single-prompt.yaml
BASE_URL ?= http://127.0.0.1:8000

.PHONY: help install data check-server run run-smoke test

.DEFAULT_GOAL := help

help:
	@echo "kv-cache-tester — trace replay stimulus (OpenAI API)"
	@echo ""
	@echo "  make install        wrapper venv + clone upstream kv-cache-tester"
	@echo "  make data           ensure upstream checkout and traces/ submodule"
	@echo "  make check-server   probe BASE_URL/v1/models"
	@echo "  make run            trace_replay_tester.py (blog profile)"
	@echo "  make run-smoke      single_prompt_tester.py (quick validation)"
	@echo "  make test           unit tests (argv builder)"
	@echo ""
	@echo "Env: BASE_URL (default $(BASE_URL))"
	@echo "     KV_CACHE_TESTER_ROOT (default $(KV_CACHE_TESTER_ROOT))"
	@echo "     KV_CACHE_TESTER_CONFIG (default trace-replay-blog.yaml)"

install: data
	@$(PYTHON) -m pip install -r "$(BENCH_ROOT)/requirements.txt"
	@set -e; \
	upstream_venv="$(KV_CACHE_TESTER_ROOT)/.venv"; \
	if [ ! -x "$$upstream_venv/bin/python" ]; then \
		$(PYTHON) -m venv "$$upstream_venv"; \
	fi; \
	"$$upstream_venv/bin/pip" install -r "$(KV_CACHE_TESTER_ROOT)/requirements.txt"

data:
	@$(PYTHON) "$(BENCH_ROOT)/scripts/install-upstream.py" \
		--config "$(KV_CACHE_TESTER_CONFIG)" \
		--dest "$(KV_CACHE_TESTER_ROOT)"

check-server:
	@$(PYTHON) "$(BENCH_ROOT)/scripts/check-server.py" --url "$(BASE_URL)"

run: install check-server
	@$(PYTHON) "$(BENCH_ROOT)/scripts/run-profile.py" \
		--config "$(KV_CACHE_TESTER_CONFIG)" \
		--upstream-root "$(KV_CACHE_TESTER_ROOT)" \
		--api-endpoint "$(BASE_URL)" \
		--bench-root "$(BENCH_ROOT)"

run-smoke: install check-server
	@$(PYTHON) "$(BENCH_ROOT)/scripts/run-profile.py" \
		--config "$(KV_CACHE_TESTER_SMOKE_CONFIG)" \
		--upstream-root "$(KV_CACHE_TESTER_ROOT)" \
		--api-endpoint "$(BASE_URL)" \
		--bench-root "$(BENCH_ROOT)"

test:
	@$(PYTHON) -m unittest discover -s "$(BENCH_ROOT)/tests" -p 'test_*.py'
