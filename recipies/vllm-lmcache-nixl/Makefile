# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

REPO_ROOT := $(abspath $(CURDIR)/../..)
BENCH_DIR := $(REPO_ROOT)/benchmarks/llm-prefill-benchmark
include $(abspath $(CURDIR)/../common/nixl/defaults.mk)
IMAGE_NAME ?= vllm-lmcache-nixl
GPU ?= 0
# **`docker run --name`** / **`--hostname`** (default **`$(IMAGE_NAME)-gpu$(GPU)`**, e.g. **vllm-lmcache-nixl-gpu0**).
CONTAINER_NAME ?= $(IMAGE_NAME)-gpu$(GPU)
DATA ?= /mnt/lmcache-nvme
LOG ?= $(CURDIR)/logs
HF_HOME ?= $(HOME)/.cache/huggingface
HF_TOKEN_FILE ?=
CONTAINER_HF_HOME ?= /hf
CONTAINER_DATA_DIR ?= /data
CONTAINER_LOG_DIR ?= /var/log/vllm-lmcache-nixl
TZ ?= America/Edmonton
VLN_LMCACHE_IO ?= nixl-posix
VLN_LMCACHE_LOG_LEVEL ?=
VLN_NIXL_BUFFER_SIZE ?=
VLN_NIXL_BUFFER_DEVICE ?=
VLN_NIXL_POOL_SIZE ?=
# NIXL opens one FD per pool slot at init; raise above VLN_NIXL_POOL_SIZE.
VLN_DOCKER_NOFILE ?= 1048576
VLN_MAX_MODEL_LEN ?=
VLN_GPU_MEMORY_UTILIZATION ?=
VLN_MAX_NUM_BATCHED_TOKENS ?=
VLN_LMCACHE_ENABLE_CHUNK_STATISTICS ?=
VLN_LMCACHE_CHUNK_STATISTICS_STRATEGY ?=
VLN_LMCACHE_ENABLE_KV_EVENTS ?=
VLN_ENABLE_MFU_METRICS ?=
VLN_ENFORCE_EAGER ?=
VLN_PYTORCH_ALLOC_CONF ?=
# hipFile AIS stats for ais-stats (0=off, 1=basic, 2=detailed, 3=max). AIS modes only.
VLH_HIPFILE_STATS_LEVEL ?=
NIXL_PLUGIN_DIR ?= /opt/nixl/lib/x86_64-linux-gnu/plugins
# Enables vLLM dev HTTP APIs (e.g. POST /reset_prefix_cache). Set to 0 to disable.
VLLM_SERVER_DEV_MODE ?= 1
# Served model (vllm-server + Slurm benchmarks). Unset → configs/vllm-lmcache-nixl.yaml default.
VLLM_MODEL ?=
ARGS ?=
EXTRA_DOCKER_RUN_FLAGS ?=
DOCKER_RUN_ATTACH ?= -it
RECIPE_RUNTIME_LOADER := $(REPO_ROOT)/recipies/common/scripts/load-recipe-runtime.sh
_recipe_runtime_file_present = $(or $(RUNTIME_CONFIG_FILE),$(RECIPE_RUNTIME_FILE),$(RECIPE_RUNTIME_CONFIG),$(wildcard $(REPO_ROOT)/recipies/runtime.yaml),$(wildcard $(CURDIR)/runtime.yaml))
_recipe_runtime_env_origins = $(if $(_recipe_runtime_file_present),command line,environment command line environment override)
_recipe_runtime_env = $(foreach var,$(1),$(if $(filter $(_recipe_runtime_env_origins),$(origin $(var))),$(var)="$(value $(var))"))
_recipe_runtime_vars = RUNTIME_CONFIG_FILE RECIPE_RUNTIME_FILE RECIPE_RUNTIME_CONFIG \
	IMAGE_NAME GPU CONTAINER_NAME DATA LOG HF_HOME HF_TOKEN HF_TOKEN_FILE \
	CONTAINER_HF_HOME CONTAINER_DATA_DIR CONTAINER_LOG_DIR TZ ARGS \
	EXTRA_DOCKER_RUN_FLAGS ROCM_ARCH PYTORCH_ALLOC_CONF \
	VLLM_PYTORCH_ALLOC_CONF VLLM_MODEL VLLM_SERVER_DEV_MODE \
	VLLM_GPU_MEMORY_UTILIZATION VLLM_ENFORCE_EAGER VLLM_ENABLE_MFU_METRICS \
	VLN_LMCACHE_IO VLN_LMCACHE_LOG_LEVEL VLN_NIXL_BUFFER_SIZE \
	VLN_NIXL_BUFFER_DEVICE VLN_NIXL_POOL_SIZE VLN_DOCKER_NOFILE \
	VLN_MAX_MODEL_LEN VLN_GPU_MEMORY_UTILIZATION \
	VLN_MAX_NUM_BATCHED_TOKENS VLN_LMCACHE_ENABLE_CHUNK_STATISTICS \
	VLN_LMCACHE_CHUNK_STATISTICS_STRATEGY VLN_LMCACHE_ENABLE_KV_EVENTS \
	VLN_ENABLE_MFU_METRICS VLN_ENFORCE_EAGER VLN_PYTORCH_ALLOC_CONF \
	VLH_HIPFILE_STATS_LEVEL NIXL_PLUGIN_DIR

# Gutenberg long-context fixtures (gitignored under data/; not LMCache DATA mount)
BOOK_SLUG ?= war-and-peace
BOOK_PG_ID ?= 2600
BOOK_TITLE ?= War and Peace
BOOK_AUTHOR ?= Leo Tolstoy
BOOK_CHUNK_WORDS ?= 10000
BOOK_CHUNK_COUNT ?= 100
BOOK_DATA_ROOT ?= $(REPO_ROOT)/data/gutenberg
BOOK_DATA_DIR ?= $(BOOK_DATA_ROOT)/$(BOOK_SLUG)
GUTENBERG_MANIFEST ?= $(BENCH_DIR)/configs/gutenberg-library.json
DATA_ALL_LIMIT ?=
DATA_ALL_BOOK ?=
DATA_ALL_DELAY ?= 1.0
DATA_ALL_SKIP_EXISTING ?= 1
PYTHON ?= python3

_ROCM_ARCH_DETECTED := $(shell rocm_agent_enumerator 2>/dev/null | \
	grep -E '^gfx' | head -1)
ROCM_ARCH := $(if $(strip $(ROCM_ARCH)),$(strip $(ROCM_ARCH)),$(_ROCM_ARCH_DETECTED))

BUILD_ARGS := --build-arg ROCM_ARCH=$(ROCM_ARCH)
ifneq ($(strip $(LMCACHE_SHA1)),)
BUILD_ARGS += --build-arg LMCACHE_SHA1=$(LMCACHE_SHA1)
endif
ifneq ($(strip $(HIPFILE_SHA)),)
BUILD_ARGS += --build-arg HIPFILE_SHA=$(HIPFILE_SHA)
endif
ifneq ($(strip $(NIXL_REF)),)
BUILD_ARGS += --build-arg NIXL_REF=$(NIXL_REF)
endif
ifneq ($(strip $(NIXL_SHA)),)
BUILD_ARGS += --build-arg NIXL_SHA=$(NIXL_SHA)
endif
ifneq ($(strip $(NIXL_GIT_URL)),)
BUILD_ARGS += --build-arg NIXL_GIT_URL=$(NIXL_GIT_URL)
endif

.PHONY: help build vllm-lmcache-nixl run run-batch data data-all

.DEFAULT_GOAL := help

help:
	@echo "vllm-lmcache-nixl Makefile"
	@echo ""
	@echo "  make build         docker build -t $(IMAGE_NAME)"
	@echo "  make vllm-lmcache-nixl    alias for build"
	@echo "  make run            docker run -it ... bind-mounts repo configs/ + scripts/"
	@echo "                      + LOG (server.txt); DATA = LMCache KV only"
	@echo "                      default CONTAINER_NAME=$(CONTAINER_NAME)"
	@echo "  make run-batch      docker run -d (non-interactive; for Slurm sbatch)"
	@echo "  make data           download PG book, chunk, and write questions JSON"
	@echo "                      (BOOK_SLUG=$(BOOK_SLUG) BOOK_PG_ID=$(BOOK_PG_ID))"
	@echo "  make data-all       build 100-book library from configs/gutenberg-library.json"
	@echo "                      (DATA_ALL_LIMIT=N smoke test; ~500MB-1GB+ under data/)"
	@echo ""
	@echo ""
	@echo "make run: requires HF_TOKEN or HF_TOKEN_FILE (shell env, Makefile"
	@echo "  HF_TOKEN_FILE=..., or make run HF_TOKEN_FILE=/path). See README."
	@echo "  Examples: make run"
	@echo "            make run HF_TOKEN_FILE=/path/to/token"
	@echo ""
	@echo "make data: writes $(BOOK_DATA_DIR)/ (override BOOK_SLUG, BOOK_PG_ID,"
	@echo "  BOOK_TITLE, BOOK_AUTHOR, BOOK_CHUNK_WORDS, BOOK_CHUNK_COUNT)."
	@echo "  Example: make data BOOK_SLUG=pride-and-prejudice BOOK_PG_ID=1342 \\"
	@echo "    BOOK_TITLE='Pride and Prejudice' BOOK_AUTHOR='Jane Austen'"
	@echo ""
	@echo "make data-all: manifest=$(GUTENBERG_MANIFEST) -> $(BOOK_DATA_ROOT)/"
	@echo "  DATA_ALL_LIMIT=3 smoke test; DATA_ALL_BOOK=adam-bede one book;"
	@echo "  DATA_ALL_SKIP_EXISTING=0 force rebuild."
	@echo "  ./run-long.sh uses library mode (all books) when BOOK_SLUG is unset."
	@echo ""

data-all:
	@$(MAKE) -C "$(BENCH_DIR)" data-all \
		BOOK_DATA_ROOT="$(BOOK_DATA_ROOT)" \
		GUTENBERG_MANIFEST="$(GUTENBERG_MANIFEST)" \
		DATA_ALL_LIMIT="$(DATA_ALL_LIMIT)" \
		DATA_ALL_BOOK="$(DATA_ALL_BOOK)" \
		DATA_ALL_DELAY="$(DATA_ALL_DELAY)" \
		DATA_ALL_SKIP_EXISTING="$(DATA_ALL_SKIP_EXISTING)"

data:
	@$(MAKE) -C "$(BENCH_DIR)" data \
		BOOK_SLUG="$(BOOK_SLUG)" \
		BOOK_PG_ID="$(BOOK_PG_ID)" \
		BOOK_TITLE="$(BOOK_TITLE)" \
		BOOK_AUTHOR="$(BOOK_AUTHOR)" \
		BOOK_CHUNK_WORDS="$(BOOK_CHUNK_WORDS)" \
		BOOK_CHUNK_COUNT="$(BOOK_CHUNK_COUNT)" \
		BOOK_DATA_DIR="$(BOOK_DATA_DIR)"

run:
	@set -e; \
	if [ -x "$(RECIPE_RUNTIME_LOADER)" ]; then \
		runtime_exports="$$("$(RECIPE_RUNTIME_LOADER)" vllm-lmcache-nixl "$(CURDIR)")"; \
		if [ -n "$$runtime_exports" ]; then eval "$$runtime_exports"; fi; \
	fi; \
	env $(call _recipe_runtime_env,$(_recipe_runtime_vars)) \
		$(MAKE) -f "$(CURDIR)/Makefile" _docker_run DOCKER_RUN_ATTACH=-it

run-batch:
	@set -e; \
	if [ -x "$(RECIPE_RUNTIME_LOADER)" ]; then \
		runtime_exports="$$("$(RECIPE_RUNTIME_LOADER)" vllm-lmcache-nixl "$(CURDIR)")"; \
		if [ -n "$$runtime_exports" ]; then eval "$$runtime_exports"; fi; \
	fi; \
	env $(call _recipe_runtime_env,$(_recipe_runtime_vars)) \
		$(MAKE) -f "$(CURDIR)/Makefile" _docker_run DOCKER_RUN_ATTACH=-d

_docker_run:
	@set -e; \
	HF_TOKEN_FILE_MAKE="$(HF_TOKEN_FILE)"; \
	if [ -n "$$HF_TOKEN_FILE_MAKE" ]; then export HF_TOKEN_FILE="$$HF_TOKEN_FILE_MAKE"; fi; \
	command -v docker >/dev/null 2>&1 || { \
		echo "ERROR: docker not found (install Docker Engine)" >&2; \
		exit 1; \
	}; \
	if [ -z "$$HF_TOKEN" ] && [ -n "$$HF_TOKEN_FILE" ] && [ -r "$$HF_TOKEN_FILE" ]; then \
		export HF_TOKEN="$$(tr -d '\r\n' < "$$HF_TOKEN_FILE")"; \
	fi; \
	if [ -z "$$HF_TOKEN" ]; then \
		echo "ERROR: set HF_TOKEN or a readable HF_TOKEN_FILE" >&2; \
		exit 1; \
	fi; \
	mkdir -p "$(DATA)" "$(LOG)" "$(HF_HOME)" \
		"$(HF_HOME)/hub" "$(HF_HOME)/datasets" "$(HF_HOME)/vllm" \
		"$(HF_HOME)/vllm_config" "$(HF_HOME)/torch" "$(HF_HOME)/torch_inductor"; \
	docker rm -f "$(CONTAINER_NAME)" >/dev/null 2>&1 || true; \
	docker run $(DOCKER_RUN_ATTACH) --rm \
		--name="$(CONTAINER_NAME)" \
		--hostname="$(CONTAINER_NAME)" \
		--ipc=host \
		--network=host \
		--cap-add=CAP_SYS_ADMIN \
		--cap-add=SYS_PTRACE \
		--security-opt seccomp=unconfined \
		--ulimit nofile=$(VLN_DOCKER_NOFILE):$(VLN_DOCKER_NOFILE) \
		--device=/dev/kfd \
		--device=/dev/dri \
		--env HF_HOME="$(CONTAINER_HF_HOME)" \
		--env HF_HUB_CACHE="$(CONTAINER_HF_HOME)/hub" \
		--env HF_DATASETS_CACHE="$(CONTAINER_HF_HOME)/datasets" \
		--env VLLM_CACHE_ROOT="$(CONTAINER_HF_HOME)/vllm" \
		--env VLLM_CONFIG_ROOT="$(CONTAINER_HF_HOME)/vllm_config" \
		--env TORCH_HOME="$(CONTAINER_HF_HOME)/torch" \
		--env TORCHINDUCTOR_CACHE_DIR="$(CONTAINER_HF_HOME)/torch_inductor" \
		--env VLN_CONTAINER_DATA_DIR="$(CONTAINER_DATA_DIR)" \
		--env VLN_SERVER_LOG_DIR="$(CONTAINER_LOG_DIR)" \
		--env VLN_LMCACHE_IO="$(VLN_LMCACHE_IO)" \
		--env VLN_LMCACHE_LOG_LEVEL="$(VLN_LMCACHE_LOG_LEVEL)" \
		--env VLN_NIXL_BUFFER_SIZE="$(VLN_NIXL_BUFFER_SIZE)" \
		$$(if [ -n "$(VLN_NIXL_BUFFER_DEVICE)" ]; then printf '%s' "--env VLN_NIXL_BUFFER_DEVICE=$(VLN_NIXL_BUFFER_DEVICE)"; fi) \
		$$(if [ -n "$(VLN_NIXL_POOL_SIZE)" ]; then printf '%s' "--env VLN_NIXL_POOL_SIZE=$(VLN_NIXL_POOL_SIZE)"; fi) \
		$$(if [ -n "$(VLN_LMCACHE_ENABLE_CHUNK_STATISTICS)" ]; then printf '%s' "--env VLN_LMCACHE_ENABLE_CHUNK_STATISTICS=$(VLN_LMCACHE_ENABLE_CHUNK_STATISTICS)"; fi) \
		$$(if [ -n "$(VLN_LMCACHE_CHUNK_STATISTICS_STRATEGY)" ]; then printf '%s' "--env VLN_LMCACHE_CHUNK_STATISTICS_STRATEGY=$(VLN_LMCACHE_CHUNK_STATISTICS_STRATEGY)"; fi) \
		$$(if [ -n "$(VLN_LMCACHE_ENABLE_KV_EVENTS)" ]; then printf '%s' "--env VLN_LMCACHE_ENABLE_KV_EVENTS=$(VLN_LMCACHE_ENABLE_KV_EVENTS)"; fi) \
		$$(if [ -n "$(VLN_MAX_MODEL_LEN)" ]; then printf '%s' "--env VLN_MAX_MODEL_LEN=$(VLN_MAX_MODEL_LEN)"; fi) \
		$$(if [ -n "$(VLN_GPU_MEMORY_UTILIZATION)" ]; then printf '%s' "--env VLN_GPU_MEMORY_UTILIZATION=$(VLN_GPU_MEMORY_UTILIZATION)"; fi) \
		$$(if [ -n "$(VLN_MAX_NUM_BATCHED_TOKENS)" ]; then printf '%s' "--env VLN_MAX_NUM_BATCHED_TOKENS=$(VLN_MAX_NUM_BATCHED_TOKENS)"; fi) \
		$$(if [ -n "$(VLN_ENABLE_MFU_METRICS)" ]; then printf '%s' "--env VLN_ENABLE_MFU_METRICS=$(VLN_ENABLE_MFU_METRICS)"; fi) \
		$$(if [ -n "$(VLN_ENFORCE_EAGER)" ]; then printf '%s' "--env VLN_ENFORCE_EAGER=$(VLN_ENFORCE_EAGER)"; fi) \
		$$(if [ -n "$(VLN_PYTORCH_ALLOC_CONF)" ]; then printf '%s' "--env VLN_PYTORCH_ALLOC_CONF=$(VLN_PYTORCH_ALLOC_CONF)"; fi) \
		$$(if [ -n "$(VLH_HIPFILE_STATS_LEVEL)" ]; then printf '%s' "--env VLH_HIPFILE_STATS_LEVEL=$(VLH_HIPFILE_STATS_LEVEL)"; fi) \
		--env NIXL_PLUGIN_DIR="$(NIXL_PLUGIN_DIR)" \
		--env VLLM_SERVER_DEV_MODE="$(VLLM_SERVER_DEV_MODE)" \
		$$(if [ -n "$(VLLM_MODEL)" ]; then printf '%s' "--env VLLM_MODEL=$(VLLM_MODEL)"; fi) \
		--env HF_TOKEN="$$HF_TOKEN" \
		--env ROCR_VISIBLE_DEVICES="$(GPU)" \
		--env TZ="$(TZ)" \
		--env PYTORCH_ALLOC_CONF="$${PYTORCH_ALLOC_CONF:-expandable_segments:True}" \
		-v "$(HF_HOME):$(CONTAINER_HF_HOME)" \
		-v "$(DATA):$(CONTAINER_DATA_DIR)" \
		-v "$(LOG):$(CONTAINER_LOG_DIR)" \
		-v "$(CURDIR)/configs:/app/configs:ro" \
		-v "$(CURDIR)/scripts:/app/scripts:ro" \
		-v /boot/config-$$(uname -r):/boot/config-$$(uname -r):ro \
		$(EXTRA_DOCKER_RUN_FLAGS) \
		$(IMAGE_NAME) $(ARGS); \
	if [ "$(DOCKER_RUN_ATTACH)" = "-d" ]; then \
		echo "Started container $(CONTAINER_NAME)"; \
	fi

build vllm-lmcache-nixl:
	@command -v docker >/dev/null 2>&1 || { \
		echo "ERROR: docker not found (install Docker Engine)" >&2; \
		exit 1; \
	}
	@test -n "$(ROCM_ARCH)" || { \
		echo "ERROR: ROCM_ARCH is empty (install ROCm / rocm_agent_enumerator," >&2; \
		echo "       or set e.g. ROCM_ARCH=gfx1201 for RX 9070 XT)" >&2; \
		exit 1; \
	}
	docker build -f "$(CURDIR)/Dockerfile" -t $(IMAGE_NAME) $(BUILD_ARGS) "$(REPO_ROOT)"
