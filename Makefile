# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

REPO_ROOT := $(abspath $(CURDIR))
BENCH_DIR := $(REPO_ROOT)/benchmarks/llm-prefill-benchmark
BOOK_DATA_ROOT ?= $(REPO_ROOT)/data/gutenberg

.PHONY: help data data-all gutenberg-data gutenberg-data-all \
	grafana-normalize grafana-check

.DEFAULT_GOAL := help

help:
	@echo "rocm-aic — repo-root targets"
	@echo ""
	@echo "  make data              download one Gutenberg book -> data/gutenberg/"
	@echo "  make data-all          build full Gutenberg library -> data/gutenberg/"
	@echo "  make grafana-normalize strip volatile fields from grafana/*.json"
	@echo "  make grafana-check     CI: assert dashboards are normalized"
	@echo ""
	@echo "Overrides: BOOK_DATA_ROOT, BOOK_SLUG, BOOK_PG_ID, DATA_ALL_LIMIT, ..."

data gutenberg-data:
	@$(MAKE) -C "$(BENCH_DIR)" data BOOK_DATA_ROOT="$(BOOK_DATA_ROOT)"

data-all gutenberg-data-all:
	@$(MAKE) -C "$(BENCH_DIR)" data-all BOOK_DATA_ROOT="$(BOOK_DATA_ROOT)"

grafana-normalize:
	@python3 "$(REPO_ROOT)/grafana/scripts/normalize-dashboard.py"

grafana-check:
	@python3 "$(REPO_ROOT)/grafana/scripts/normalize-dashboard.py" --check
