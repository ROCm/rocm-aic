# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

REPO_ROOT := $(abspath $(CURDIR)/../..)
IMAGE_NAME ?= aic-drivenets
GPU ?= 0
CONTAINER_NAME ?= $(IMAGE_NAME)-gpu$(GPU)
DATA ?= /data
LOG ?= $(CURDIR)/logs
HF_HOME ?= $(HOME)/.cache/huggingface
HF_TOKEN_FILE ?=
CONTAINER_HF_HOME ?= /hf
CONTAINER_DATA_DIR ?= /data
CONTAINER_LOG_DIR ?= /var/log/aic-drivenets
TZ ?= America/Edmonton
ADE_LMCACHE_TIER ?= cpu-nvme
ADE_LMCACHE_LOG_LEVEL ?=
ADE_TENSOR_PARALLEL_SIZE ?=
ADE_GPU_MEMORY_UTILIZATION ?=
ADE_MAX_MODEL_LEN ?=
ADE_MAX_NUM_BATCHED_TOKENS ?=
ADE_LMCACHE_MAX_LOCAL_CPU_SIZE ?=
ADE_LMCACHE_MAX_LOCAL_DISK_SIZE ?=
VLLM_SERVER_DEV_MODE ?= 1
VLLM_MODEL ?=
ARGS ?=
EXTRA_DOCKER_RUN_FLAGS ?=
DOCKER_RUN_ATTACH ?= -it

VLLM_BASE_TAG ?= rocm7.13.0_gfx950-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1

_ROCM_ARCH_DETECTED := $(shell rocm_agent_enumerator 2>/dev/null | \
	grep -E '^gfx' | head -1)
ROCM_ARCH := $(if $(strip $(ROCM_ARCH)),$(strip $(ROCM_ARCH)),$(_ROCM_ARCH_DETECTED))
ifeq ($(strip $(ROCM_ARCH)),)
ROCM_ARCH := gfx950
endif

BUILD_ARGS := --build-arg ROCM_ARCH=$(ROCM_ARCH) --build-arg VLLM_BASE_TAG=$(VLLM_BASE_TAG)
ifneq ($(strip $(LMCACHE_SHA)),)
BUILD_ARGS += --build-arg LMCACHE_SHA=$(LMCACHE_SHA)
endif

.PHONY: help build aic-drivenets run run-batch verify

.DEFAULT_GOAL := help

help:
	@echo "aic-drivenets Makefile (gfx950 vLLM + LMCache, DriveNets run flags)"
	@echo ""
	@echo "  make build         docker build -t $(IMAGE_NAME)"
	@echo "                      base: rocm/vllm:$(VLLM_BASE_TAG)"
	@echo "  make verify        confirm LMCache HIP c_ops backend (.so)"
	@echo "  make run            docker run -it ... bind-mounts configs/ + scripts/"
	@echo "                      + LOG (server.txt); DATA = LMCache NVMe tier"
	@echo "                      default CONTAINER_NAME=$(CONTAINER_NAME)"
	@echo "  make run-batch      docker run -d (non-interactive)"
	@echo ""
	@echo "ADE_LMCACHE_TIER: hbm | cpu | cpu-nvme (default)"
	@echo "  hbm       HBM-only (no LMCache connector)"
	@echo "  cpu       HBM L1 + CPU DRAM L2"
	@echo "  cpu-nvme  HBM L1 + CPU L2 + NVMe L3 (needs DATA mount)"
	@echo ""
	@echo "Benchmark stimulus: make -C benchmarks/kv-cache-tester install run"
	@echo ""
	@echo "make run: requires HF_TOKEN or HF_TOKEN_FILE. See README."
	@echo ""

run:
	@$(MAKE) -f "$(CURDIR)/Makefile" _docker_run DOCKER_RUN_ATTACH=-it

run-batch:
	@$(MAKE) -f "$(CURDIR)/Makefile" _docker_run DOCKER_RUN_ATTACH=-d

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
		--device=/dev/kfd \
		--device=/dev/dri \
		--network=host \
		--ipc=host \
		--group-add 44 \
		--group-add 993 \
		--cap-add=SYS_PTRACE \
		--cap-add=SYS_NICE \
		--shm-size=64g \
		--env HF_HOME="$(CONTAINER_HF_HOME)" \
		--env HF_HUB_CACHE="$(CONTAINER_HF_HOME)/hub" \
		--env HF_DATASETS_CACHE="$(CONTAINER_HF_HOME)/datasets" \
		--env VLLM_CACHE_ROOT="$(CONTAINER_HF_HOME)/vllm" \
		--env VLLM_CONFIG_ROOT="$(CONTAINER_HF_HOME)/vllm_config" \
		--env TORCH_HOME="$(CONTAINER_HF_HOME)/torch" \
		--env TORCHINDUCTOR_CACHE_DIR="$(CONTAINER_HF_HOME)/torch_inductor" \
		--env ADE_CONTAINER_DATA_DIR="$(CONTAINER_DATA_DIR)" \
		--env ADE_SERVER_LOG_DIR="$(CONTAINER_LOG_DIR)" \
		--env ADE_LMCACHE_TIER="$(ADE_LMCACHE_TIER)" \
		--env ADE_LMCACHE_LOG_LEVEL="$(ADE_LMCACHE_LOG_LEVEL)" \
		--env VLLM_SERVER_DEV_MODE="$(VLLM_SERVER_DEV_MODE)" \
		$$(if [ -n "$(ADE_TENSOR_PARALLEL_SIZE)" ]; then printf '%s' "--env ADE_TENSOR_PARALLEL_SIZE=$(ADE_TENSOR_PARALLEL_SIZE)"; fi) \
		$$(if [ -n "$(ADE_GPU_MEMORY_UTILIZATION)" ]; then printf '%s' "--env ADE_GPU_MEMORY_UTILIZATION=$(ADE_GPU_MEMORY_UTILIZATION)"; fi) \
		$$(if [ -n "$(ADE_MAX_MODEL_LEN)" ]; then printf '%s' "--env ADE_MAX_MODEL_LEN=$(ADE_MAX_MODEL_LEN)"; fi) \
		$$(if [ -n "$(ADE_MAX_NUM_BATCHED_TOKENS)" ]; then printf '%s' "--env ADE_MAX_NUM_BATCHED_TOKENS=$(ADE_MAX_NUM_BATCHED_TOKENS)"; fi) \
		$$(if [ -n "$(ADE_LMCACHE_MAX_LOCAL_CPU_SIZE)" ]; then printf '%s' "--env ADE_LMCACHE_MAX_LOCAL_CPU_SIZE=$(ADE_LMCACHE_MAX_LOCAL_CPU_SIZE)"; fi) \
		$$(if [ -n "$(ADE_LMCACHE_MAX_LOCAL_DISK_SIZE)" ]; then printf '%s' "--env ADE_LMCACHE_MAX_LOCAL_DISK_SIZE=$(ADE_LMCACHE_MAX_LOCAL_DISK_SIZE)"; fi) \
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

build aic-drivenets:
	@command -v docker >/dev/null 2>&1 || { \
		echo "ERROR: docker not found (install Docker Engine)" >&2; \
		exit 1; \
	}
	@test -n "$(ROCM_ARCH)" || { \
		echo "ERROR: ROCM_ARCH is empty (install ROCm / rocm_agent_enumerator," >&2; \
		echo "       or set e.g. ROCM_ARCH=gfx950)" >&2; \
		exit 1; \
	}
	docker build -f "$(CURDIR)/Dockerfile" -t $(IMAGE_NAME) $(BUILD_ARGS) "$(REPO_ROOT)"

verify:
	@command -v docker >/dev/null 2>&1 || { \
		echo "ERROR: docker not found (install Docker Engine)" >&2; \
		exit 1; \
	}
	docker run --rm $(IMAGE_NAME) python3 /app/scripts/verify-lmcache-hip.py
