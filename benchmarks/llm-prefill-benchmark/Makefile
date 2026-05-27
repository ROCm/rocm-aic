# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

REPO_ROOT := $(abspath $(CURDIR)/../..)
BENCH_ROOT := $(CURDIR)
PYTHON ?= python3

BOOK_SLUG ?= war-and-peace
BOOK_PG_ID ?= 2600
BOOK_TITLE ?= War and Peace
BOOK_AUTHOR ?= Leo Tolstoy
BOOK_CHUNK_WORDS ?= 10000
BOOK_CHUNK_COUNT ?= 100
BOOK_DATA_DIR ?= $(BENCH_ROOT)/data/$(BOOK_SLUG)
BOOK_DATA_ROOT ?= $(BENCH_ROOT)/data
GUTENBERG_MANIFEST ?= $(BENCH_ROOT)/configs/gutenberg-library.json
DATA_ALL_LIMIT ?=
DATA_ALL_BOOK ?=
DATA_ALL_DELAY ?= 1.0
DATA_ALL_SKIP_EXISTING ?= 1

BASE_URL ?= http://127.0.0.1:8000
MODEL ?= openai/gpt-oss-120b
WORKERS ?= 4
ITERATIONS ?= 1
BASE_SEED ?= 42

.PHONY: help data data-all run run-parallel

.DEFAULT_GOAL := help

help:
	@echo "llm-prefill-benchmark — Gutenberg long-context prefill (engine-agnostic)"
	@echo ""
	@echo "  make data           download one PG book, chunk, write questions JSON"
	@echo "  make data-all       build library from configs/gutenberg-library.json"
	@echo "  make run            serial run-long.sh (ITERATIONS=$(ITERATIONS))"
	@echo "  make run-parallel   run-long-parallel.sh (WORKERS=$(WORKERS))"
	@echo ""
	@echo "Env: BASE_URL, MODEL, BOOK_DATA_ROOT, BOOK_SLUG, ITERATIONS, WORKERS"

data:
	@set -e; \
	command -v $(PYTHON) >/dev/null 2>&1 || { echo "ERROR: $(PYTHON) not found" >&2; exit 1; }; \
	echo "make data: BOOK_SLUG=$(BOOK_SLUG) PG#$(BOOK_PG_ID) -> $(BOOK_DATA_DIR)"; \
	$(PYTHON) "$(BENCH_ROOT)/scripts/split-gutenberg-random-chunks.py" \
		--pg-id "$(BOOK_PG_ID)" \
		--slug "$(BOOK_SLUG)" \
		-o "$(BOOK_DATA_DIR)" \
		--chunk-words "$(BOOK_CHUNK_WORDS)" \
		--count "$(BOOK_CHUNK_COUNT)"; \
	$(PYTHON) "$(BENCH_ROOT)/scripts/gen-questions-json.py" \
		--slug "$(BOOK_SLUG)" \
		--title "$(BOOK_TITLE)" \
		--author "$(BOOK_AUTHOR)" \
		--pg-id "$(BOOK_PG_ID)" \
		-o "$(BOOK_DATA_DIR)/$(BOOK_SLUG).questions.json"

data-all:
	@set -e; \
	command -v $(PYTHON) >/dev/null 2>&1 || { echo "ERROR: $(PYTHON) not found" >&2; exit 1; }; \
	_opts="--manifest $(GUTENBERG_MANIFEST) --data-root $(BOOK_DATA_ROOT) --delay $(DATA_ALL_DELAY)"; \
	if [ "$(DATA_ALL_SKIP_EXISTING)" = "1" ]; then _opts="$$_opts --skip-existing"; fi; \
	if [ -n "$(DATA_ALL_BOOK)" ]; then _opts="$$_opts --book $(DATA_ALL_BOOK)"; fi; \
	if [ -n "$(DATA_ALL_LIMIT)" ]; then _opts="$$_opts --limit $(DATA_ALL_LIMIT)"; fi; \
	echo "make data-all: manifest=$(GUTENBERG_MANIFEST) -> $(BOOK_DATA_ROOT)"; \
	$(PYTHON) "$(BENCH_ROOT)/scripts/build-gutenberg-library.py" $$_opts

run:
	@BASE_URL="$(BASE_URL)" MODEL="$(MODEL)" BOOK_DATA_ROOT="$(BOOK_DATA_ROOT)" \
		ITERATIONS="$(ITERATIONS)" bash "$(BENCH_ROOT)/run-long.sh"

run-parallel:
	@BASE_URL="$(BASE_URL)" MODEL="$(MODEL)" BOOK_DATA_ROOT="$(BOOK_DATA_ROOT)" \
		WORKERS="$(WORKERS)" ITERATIONS="$(ITERATIONS)" BASE_SEED="$(BASE_SEED)" \
		bash "$(BENCH_ROOT)/run-long-parallel.sh" "$(WORKERS)"
