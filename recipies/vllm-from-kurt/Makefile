# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

REPO_ROOT := $(abspath $(CURDIR)/../..)
DOCKERFILE := Dockerfile
IMAGE_NAME ?= vllm-kurt
HADOLINT_IMG ?= hadolint/hadolint:latest
# Container CLI for build and hadolint (override if only Podman is installed).
DOCKER ?= docker

SHELL_SCRIPTS := vllm-container \
	scripts/vllm-benchmark \
	scripts/fio-benchmark \
	scripts/vllm-server-hipfile \
	scripts/vllm-server-native-disk

# First gfx ISA from rocm_agent_enumerator (empty if ROCm not installed / no GPU).
_ROCM_ARCH_DETECTED := $(shell rocm_agent_enumerator 2>/dev/null | \
	grep -E '^gfx' | head -1)
# Honor non-empty ROCM_ARCH from the command line or environment; otherwise
# use local auto-detection.
ROCM_ARCH := $(if $(strip $(ROCM_ARCH)),$(strip $(ROCM_ARCH)),$(_ROCM_ARCH_DETECTED))

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
	@echo "  make build         $(DOCKER) build -t $(IMAGE_NAME)"
	@echo "  make vllm-kurt     alias for build"
	@echo ""
	@echo "ROCM_ARCH defaults to the first gfx ISA from rocm_agent_enumerator."
	@echo "Override when needed: make build ROCM_ARCH=gfx942"
	@echo "If docker is not on PATH: make build DOCKER=podman (or install Docker)."
	@echo "Optional: LMCACHE_SHA1=<git-sha> for an extra --build-arg."

lint: lint-hadolint lint-shell

test: lint

lint-hadolint:
	@command -v $(DOCKER) >/dev/null 2>&1 || { \
		echo "ERROR: '$(DOCKER)' not found (install Docker, or e.g. DOCKER=podman)" >&2; \
		exit 1; \
	}
	$(DOCKER) run --rm \
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
	@command -v $(DOCKER) >/dev/null 2>&1 || { \
		echo "ERROR: '$(DOCKER)' not found (install Docker Engine, or use DOCKER=podman)" >&2; \
		exit 1; \
	}
	@test -n "$(ROCM_ARCH)" || { \
		echo "ERROR: ROCM_ARCH is empty (install ROCm / rocm_agent_enumerator," >&2; \
		echo "       or set e.g. ROCM_ARCH=gfx942)" >&2; \
		exit 1; \
	}
	$(DOCKER) build . -t $(IMAGE_NAME) $(BUILD_ARGS)
