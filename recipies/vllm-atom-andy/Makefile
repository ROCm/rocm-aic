# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

REPO_ROOT := $(abspath $(CURDIR)/../..)
IMAGE_NAME ?= vllm-atom-andy
GPU ?= 0
CONTAINER_NAME ?= $(IMAGE_NAME)-gpu$(GPU)
DATA ?= /mnt/lmcache-nvme
LOG ?= $(CURDIR)/logs
HF_HOME ?= $(HOME)/.cache/huggingface
HF_TOKEN_FILE ?=
CONTAINER_HF_HOME ?= /hf
CONTAINER_DATA_DIR ?= /data
CONTAINER_LOG_DIR ?= /var/log/vllm-atom-andy
TZ ?= America/Edmonton
VAA_LMCACHE_TIER ?= cpu
VAA_LMCACHE_LOG_LEVEL ?=
VAA_TENSOR_PARALLEL_SIZE ?=
VAA_GPU_MEMORY_UTILIZATION ?=
VAA_MAX_MODEL_LEN ?=
VAA_MAX_NUM_BATCHED_TOKENS ?=
VAA_LMCACHE_MAX_LOCAL_CPU_SIZE ?=
VAA_MODEL_PROFILE ?=
VAA_ENABLE_MFU_METRICS ?=
VAA_LMCACHE_ENABLE_CHUNK_STATISTICS ?=
VAA_LMCACHE_CHUNK_STATISTICS_STRATEGY ?=
VLLM_SERVER_DEV_MODE ?= 1
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
	VLLM_MODEL VLLM_SERVER_DEV_MODE VLLM_GPU_MEMORY_UTILIZATION \
	VAA_LMCACHE_TIER VAA_LMCACHE_LOG_LEVEL VAA_TENSOR_PARALLEL_SIZE \
	VAA_GPU_MEMORY_UTILIZATION VAA_MAX_MODEL_LEN \
	VAA_MAX_NUM_BATCHED_TOKENS VAA_LMCACHE_MAX_LOCAL_CPU_SIZE \
	VAA_MODEL_PROFILE VAA_ENABLE_MFU_METRICS \
	VAA_LMCACHE_ENABLE_CHUNK_STATISTICS \
	VAA_LMCACHE_CHUNK_STATISTICS_STRATEGY

# rocm/atom-dev:vllm-v0.19.0-nightly_20260522 (Andy blog era; not vllm-latest)
ATOM_DEV_DIGEST ?= sha256:c7e019b4872efe85557097731e863207189aaaa80ce8071c534565b20d03a509

_ROCM_ARCH_DETECTED := $(shell rocm_agent_enumerator 2>/dev/null | \
	grep -E '^gfx' | head -1)
ROCM_ARCH := $(if $(strip $(ROCM_ARCH)),$(strip $(ROCM_ARCH)),$(_ROCM_ARCH_DETECTED))

BUILD_ARGS := --build-arg ROCM_ARCH=$(ROCM_ARCH)
ifneq ($(strip $(ATOM_DEV_DIGEST)),)
BUILD_ARGS += --build-arg ATOM_DEV_DIGEST=$(ATOM_DEV_DIGEST)
endif
ifneq ($(strip $(LMCACHE_SHA)),)
BUILD_ARGS += --build-arg LMCACHE_SHA=$(LMCACHE_SHA)
endif

.PHONY: help build vllm-atom-andy run run-batch verify

.DEFAULT_GOAL := help

help:
	@echo "vllm-atom-andy Makefile (ATOM + LMCache, Andy Luo blog)"
	@echo ""
	@echo "  make build         docker build -t $(IMAGE_NAME)"
	@echo "                      base: rocm/atom-dev:vllm-v0.19.0-nightly_20260522"
	@echo "                      (digest-pinned; override ATOM_DEV_DIGEST=sha256:...)"
	@echo "  make verify        confirm LMCache HIP c_ops backend (.so)"
	@echo "  make run            docker run -it ... bind-mounts configs/ + scripts/"
	@echo "                      + LOG (server.txt); DATA = LMCache NVMe tier"
	@echo "                      default CONTAINER_NAME=$(CONTAINER_NAME)"
	@echo "  make run-batch      docker run -d (non-interactive)"
	@echo ""
	@echo "VAA_LMCACHE_TIER: hbm | cpu (default) | cpu-nvme"
	@echo "  hbm       ATOM HBM-only (no LMCache connector)"
	@echo "  cpu       HBM L1 + CPU DRAM L2"
	@echo "  cpu-nvme  HBM L1 + CPU L2 + NVMe L3 (needs DATA mount)"
	@echo ""
	@echo "make run: requires HF_TOKEN or HF_TOKEN_FILE. See README for blog overrides"
	@echo "  (TP=2, MiniMax-M2.5, GMU=0.78, max context 100K)."
	@echo ""

run:
	@set -e; \
	if [ -x "$(RECIPE_RUNTIME_LOADER)" ]; then \
		if ! runtime_exports="$$("$(RECIPE_RUNTIME_LOADER)" vllm-atom-andy "$(CURDIR)")"; then \
			exit 1; \
		fi; \
		if [ -n "$$runtime_exports" ]; then eval "$$runtime_exports"; fi; \
	fi; \
	env $(call _recipe_runtime_env,$(_recipe_runtime_vars)) \
		$(MAKE) -f "$(CURDIR)/Makefile" _docker_run DOCKER_RUN_ATTACH=-it

run-batch:
	@set -e; \
	if [ -x "$(RECIPE_RUNTIME_LOADER)" ]; then \
		if ! runtime_exports="$$("$(RECIPE_RUNTIME_LOADER)" vllm-atom-andy "$(CURDIR)")"; then \
			exit 1; \
		fi; \
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
		--group-add video \
		--cap-add=SYS_PTRACE \
		--security-opt seccomp=unconfined \
		--device=/dev/kfd \
		--device=/dev/dri \
		--env HF_HOME="$(CONTAINER_HF_HOME)" \
		--env HF_HUB_CACHE="$(CONTAINER_HF_HOME)/hub" \
		--env HF_DATASETS_CACHE="$(CONTAINER_HF_HOME)/datasets" \
		--env VLLM_CACHE_ROOT="$(CONTAINER_HF_HOME)/vllm" \
		--env VLLM_CONFIG_ROOT="$(CONTAINER_HF_HOME)/vllm_config" \
		--env TORCH_HOME="$(CONTAINER_HF_HOME)/torch" \
		--env TORCHINDUCTOR_CACHE_DIR="$(CONTAINER_HF_HOME)/torch_inductor" \
		--env VAA_CONTAINER_DATA_DIR="$(CONTAINER_DATA_DIR)" \
		--env VAA_SERVER_LOG_DIR="$(CONTAINER_LOG_DIR)" \
		--env VAA_LMCACHE_TIER="$(VAA_LMCACHE_TIER)" \
		--env VAA_LMCACHE_LOG_LEVEL="$(VAA_LMCACHE_LOG_LEVEL)" \
		--env VLLM_SERVER_DEV_MODE="$(VLLM_SERVER_DEV_MODE)" \
		$$(if [ -n "$(VAA_TENSOR_PARALLEL_SIZE)" ]; then printf '%s' "--env VAA_TENSOR_PARALLEL_SIZE=$(VAA_TENSOR_PARALLEL_SIZE)"; fi) \
		$$(if [ -n "$(VAA_GPU_MEMORY_UTILIZATION)" ]; then printf '%s' "--env VAA_GPU_MEMORY_UTILIZATION=$(VAA_GPU_MEMORY_UTILIZATION)"; fi) \
		$$(if [ -n "$(VAA_MAX_MODEL_LEN)" ]; then printf '%s' "--env VAA_MAX_MODEL_LEN=$(VAA_MAX_MODEL_LEN)"; fi) \
		$$(if [ -n "$(VAA_MAX_NUM_BATCHED_TOKENS)" ]; then printf '%s' "--env VAA_MAX_NUM_BATCHED_TOKENS=$(VAA_MAX_NUM_BATCHED_TOKENS)"; fi) \
		$$(if [ -n "$(VAA_LMCACHE_MAX_LOCAL_CPU_SIZE)" ]; then printf '%s' "--env VAA_LMCACHE_MAX_LOCAL_CPU_SIZE=$(VAA_LMCACHE_MAX_LOCAL_CPU_SIZE)"; fi) \
		$$(if [ -n "$(VAA_MODEL_PROFILE)" ]; then printf '%s' "--env VAA_MODEL_PROFILE=$(VAA_MODEL_PROFILE)"; fi) \
		$$(if [ -n "$(VAA_ENABLE_MFU_METRICS)" ]; then printf '%s' "--env VAA_ENABLE_MFU_METRICS=$(VAA_ENABLE_MFU_METRICS)"; fi) \
		$$(if [ -n "$(VAA_LMCACHE_ENABLE_CHUNK_STATISTICS)" ]; then printf '%s' "--env VAA_LMCACHE_ENABLE_CHUNK_STATISTICS=$(VAA_LMCACHE_ENABLE_CHUNK_STATISTICS)"; fi) \
		$$(if [ -n "$(VAA_LMCACHE_CHUNK_STATISTICS_STRATEGY)" ]; then printf '%s' "--env VAA_LMCACHE_CHUNK_STATISTICS_STRATEGY=$(VAA_LMCACHE_CHUNK_STATISTICS_STRATEGY)"; fi) \
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

build vllm-atom-andy:
	@command -v docker >/dev/null 2>&1 || { \
		echo "ERROR: docker not found (install Docker Engine)" >&2; \
		exit 1; \
	}
	@test -n "$(ROCM_ARCH)" || { \
		echo "ERROR: ROCM_ARCH is empty (install ROCm / rocm_agent_enumerator," >&2; \
		echo "       or set e.g. ROCM_ARCH=gfx942 for MI300X)" >&2; \
		exit 1; \
	}
	docker build -f "$(CURDIR)/Dockerfile" -t $(IMAGE_NAME) $(BUILD_ARGS) "$(REPO_ROOT)"

verify:
	@command -v docker >/dev/null 2>&1 || { \
		echo "ERROR: docker not found (install Docker Engine)" >&2; \
		exit 1; \
	}
	docker run --rm $(IMAGE_NAME) python3 /app/scripts/verify-lmcache-hip.py
