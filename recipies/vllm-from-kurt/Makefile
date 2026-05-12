# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

REPO_ROOT := $(abspath $(CURDIR)/../..)
DOCKERFILE := Dockerfile
IMAGE_NAME ?= vllm-kurt
HADOLINT_IMG ?= hadolint/hadolint:latest

SHELL_SCRIPTS := vllm-container \
	scripts/vllm-benchmark \
	scripts/fio-benchmark \
	scripts/vllm-server-hipfile \
	scripts/vllm-server-native-disk

BUILD_ARGS := --build-arg ROCM_ARCH=$(ROCM_ARCH)
ifneq ($(strip $(LMCACHE_SHA1)),)
BUILD_ARGS += --build-arg LMCACHE_SHA1=$(LMCACHE_SHA1)
endif

.PHONY: help lint lint-hadolint lint-shell test build vllm-kurt

.DEFAULT_GOAL := help

help:
	@echo "vllm-from-kurt Makefile"
	@echo ""
	@echo "  make test          Hadolint (Dockerfile) + shellcheck (host scripts)"
	@echo "  make lint          same as make test"
	@echo "  make lint-hadolint Dockerfile only (runs hadolint in Docker)"
	@echo "  make lint-shell    shellcheck on vllm-container and scripts/*"
	@echo "  make build         docker build -t $(IMAGE_NAME) (requires ROCM_ARCH)"
	@echo "  make vllm-kurt     alias for build"
	@echo ""
	@echo "Set ROCM_ARCH for build (e.g. export ROCM_ARCH=gfx942)."
	@echo "Optional: LMCACHE_SHA1=<git-sha> for an extra --build-arg."

lint: lint-hadolint lint-shell

test: lint

lint-hadolint:
	docker run --rm \
		-v "$(REPO_ROOT):/w:ro" \
		-w /w \
		$(HADOLINT_IMG) \
		hadolint --config .hadolint.yaml recipies/vllm-from-kurt/$(DOCKERFILE)

lint-shell:
	@command -v shellcheck >/dev/null 2>&1 || { \
		echo "ERROR: shellcheck not found (e.g. apt install shellcheck)" >&2; \
		exit 1; \
	}
	shellcheck $(SHELL_SCRIPTS)

build vllm-kurt:
	@test -n "$(ROCM_ARCH)" || { \
		echo "ERROR: ROCM_ARCH is not set (e.g. export ROCM_ARCH=gfx942)" >&2; \
		exit 1; \
	}
	docker build . -t $(IMAGE_NAME) $(BUILD_ARGS)
