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

AGENTX_DATA_ROOT ?= $(REPO_ROOT)/data/cc-traces
HF_CONFIG ?= $(BENCH_ROOT)/configs/cc-traces-hf.yaml
REQ_FILE := $(BENCH_ROOT)/requirements.txt

BASE_URL ?= http://127.0.0.1:8000
MODEL ?= Qwen/Qwen2.5-3B-Instruct
WORKERS ?= 4
ITERATIONS ?= 1
BASE_SEED ?= 42
AGENTX_MAX_REQUESTS ?=
MAX_TOKENS ?= 512
AGENTX_HONOR_THINK_TIME ?= 0
AGENTX_MAX_CONTEXT ?=
AGENTX_STRICT ?= 0

.PHONY: help install check-deps check-data check-server data run run-parallel report

.DEFAULT_GOAL := help

help:
	@echo "llm-agentx — SemiAnalysis CC agent trace replay (OpenAI API)"
	@echo ""
	@echo "  make install        pip install -r requirements.txt (uses $(PYTHON))"
	@echo "  make check-server   verify BASE_URL and MODEL"
	@echo "  make data           download CC traces from Hugging Face"
	@echo "  make run            serial run-agent.sh (ITERATIONS=$(ITERATIONS))"
	@echo "  make run-parallel   run-agent-parallel.sh (WORKERS=$(WORKERS))"
	@echo "  make report         aggregate latest parallel run logs"
	@echo ""
	@echo "Python: $(PYTHON)"
	@echo "Env: BASE_URL, MODEL, AGENTX_DATA_ROOT, ITERATIONS, WORKERS,"
	@echo "     AGENTX_MAX_REQUESTS, AGENTX_MAX_CONTEXT, AGENTX_STRICT,"
	@echo "     AGENTX_HONOR_THINK_TIME, MAX_TOKENS"

install:
	@$(PYTHON) -m pip install -r "$(REQ_FILE)"

check-deps:
	@set -e; \
	command -v $(PYTHON) >/dev/null 2>&1 || { \
		echo "ERROR: $(PYTHON) not found (create .venv: python3 -m venv $(REPO_ROOT)/.venv)" >&2; \
		exit 1; \
	}; \
	$(PYTHON) -c "import huggingface_hub, yaml, tiktoken" 2>/dev/null || { \
		echo "ERROR: missing Python deps for llm-agentx" >&2; \
		echo "  fix: make -C $(BENCH_ROOT) install" >&2; \
		exit 1; \
	}

check-data:
	@test -f "$(AGENTX_DATA_ROOT)/traces.jsonl" || { \
		echo "ERROR: missing $(AGENTX_DATA_ROOT)/traces.jsonl — run 'make data' first" >&2; \
		exit 1; \
	}

check-server: check-deps
	@$(PYTHON) "$(BENCH_ROOT)/scripts/replay-trace.py" \
		--url "$(BASE_URL)" --data-root "$(AGENTX_DATA_ROOT)" \
		--model "$(MODEL)" --check-only

data: check-deps
	@set -e; \
	echo "make data: -> $(AGENTX_DATA_ROOT) (python=$(PYTHON))"; \
	$(PYTHON) "$(BENCH_ROOT)/scripts/download-dataset.py" \
		--config $(HF_CONFIG) --output $(AGENTX_DATA_ROOT)

run: check-deps check-data
	@set -e; \
	"$(PYTHON)" "$(BENCH_ROOT)/scripts/replay-trace.py" \
		--url "$(BASE_URL)" --data-root "$(AGENTX_DATA_ROOT)" \
		--model "$(MODEL)" --check-only; \
	ts=$$(date -u +%Y%m%dT%H%M%SZ); \
	out="$(BENCH_ROOT)/logs/run-agent/$$ts/run.jsonl"; \
	mkdir -p "$$(dirname "$$out")"; \
	PYTHON="$(PYTHON)" BASE_URL="$(BASE_URL)" MODEL="$(MODEL)" \
		AGENTX_DATA_ROOT="$(AGENTX_DATA_ROOT)" \
		ITERATIONS="$(ITERATIONS)" 		AGENTX_MAX_REQUESTS="$(AGENTX_MAX_REQUESTS)" \
		AGENTX_MAX_CONTEXT="$(AGENTX_MAX_CONTEXT)" AGENTX_STRICT="$(AGENTX_STRICT)" \
		AGENTX_HONOR_THINK_TIME="$(AGENTX_HONOR_THINK_TIME)" MAX_TOKENS="$(MAX_TOKENS)" \
		bash "$(BENCH_ROOT)/run-agent.sh" >"$$out"; \
	echo "wrote $$out"

run-parallel: check-deps check-data
	@PYTHON="$(PYTHON)" BASE_URL="$(BASE_URL)" MODEL="$(MODEL)" \
		AGENTX_DATA_ROOT="$(AGENTX_DATA_ROOT)" \
		WORKERS="$(WORKERS)" ITERATIONS="$(ITERATIONS)" BASE_SEED="$(BASE_SEED)" \
		AGENTX_MAX_REQUESTS="$(AGENTX_MAX_REQUESTS)" \
		AGENTX_HONOR_THINK_TIME="$(AGENTX_HONOR_THINK_TIME)" \
		MAX_TOKENS="$(MAX_TOKENS)" \
		bash "$(BENCH_ROOT)/run-agent-parallel.sh" "$(WORKERS)"

report: check-deps
	@set -e; \
	latest="$$(ls -td "$(BENCH_ROOT)/logs/run-agent-parallel"/*/ 2>/dev/null | head -1)"; \
	if [ -z "$$latest" ]; then \
		echo "error: no run-agent-parallel logs under $(BENCH_ROOT)/logs/" >&2; \
		exit 1; \
	fi; \
	echo "report: $$latest"; \
	$(PYTHON) "$(BENCH_ROOT)/scripts/parse-results.py" "$$latest" \
		-o "$(BENCH_ROOT)/reports/$$(basename "$$latest").json"
