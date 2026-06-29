# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

REPO_ROOT := $(abspath $(CURDIR)/../..)
include $(abspath $(CURDIR)/../common/nixl/defaults.mk)

# Override NIXL source: sbates130272/nixl @ feat/amd-ais-mt (adds AIS_MT plugin).
# These override the defaults.mk values before they are exported as build args.
NIXL_GIT_URL := https://github.com/sbates130272/nixl.git
NIXL_SHA     := 9d146424117a7cd022bdbaec8cbd7551f1c08add

# ---- GPU -------------------------------------------------------------------
GPU ?= 0

# ---- Host storage paths ----------------------------------------------------
NVME_DATA ?= /mnt/lmcache-nvme
NFS_DATA  ?= /mnt/lmcache-nfs

# ---- Log / HuggingFace -----------------------------------------------------
LOG     ?= $(CURDIR)/logs
HF_HOME ?= $(HOME)/.cache/huggingface
HF_TOKEN_FILE ?=

# ---- LMCache server --------------------------------------------------------
LMCACHE_PORT       ?= 6555
LMCACHE_L1_SIZE_GB ?= 20
LMCACHE_NVME_POOL      ?= 4096
LMCACHE_NVME_SLOT_SIZE ?= 268435456
LMCACHE_NFS_POOL       ?= 1024

# ---- vLLM knobs ------------------------------------------------------------
VLLM_MODEL                 ?=
TENSOR_PARALLEL_SIZE        ?= 1
VLM_GPU_MEMORY_UTILIZATION  ?=
VLM_MAX_MODEL_LEN           ?=
VLM_MAX_NUM_BATCHED_TOKENS  ?=

# ---- ROCm arch (auto-detected if not set) ----------------------------------
_ROCM_ARCH_DETECTED := $(shell rocm_agent_enumerator 2>/dev/null | grep -E '^gfx' | head -1)
ROCM_ARCH := $(if $(strip $(ROCM_ARCH)),$(strip $(ROCM_ARCH)),$(_ROCM_ARCH_DETECTED))

# Export everything compose needs from the environment.
export ROCM_ARCH GPU GDS_SLAB_DATA LOG HF_HOME
export LMCACHE_PORT LMCACHE_L1_SIZE_GB LMCACHE_NVME_POOL LMCACHE_NVME_SLOT_SIZE LMCACHE_NFS_POOL
export NVME_DATA NFS_DATA
export VLLM_MODEL TENSOR_PARALLEL_SIZE
export VLM_GPU_MEMORY_UTILIZATION VLM_MAX_MODEL_LEN VLM_MAX_NUM_BATCHED_TOKENS
export NIXL_GIT_URL NIXL_SHA

# Prefer the 'docker compose' plugin (Docker 23+); fall back to standalone docker-compose.
_COMPOSE_BIN := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")
COMPOSE        := DOCKER_BUILDKIT=1 $(_COMPOSE_BIN) -f "$(CURDIR)/docker-compose.yml"
COMPOSE_GDS_L1 := $(COMPOSE) -f "$(CURDIR)/docker-compose.gds-l1.yml"

.PHONY: help build up up-batch up-gds-l1 up-gds-l1-batch down logs logs-lmcache logs-vllm \
        ps shell-lmcache shell-vllm restart-vllm restart-lmcache _check_hf_token _prep_dirs

.DEFAULT_GOAL := help

help:
	@echo "vllm-lmcache-mp — vLLM + standalone LMCache server (MP/ZMQ mode)"
	@echo ""
	@echo "Targets:"
	@echo "  make build           Build the shared image"
	@echo "  make up              Start both containers (foreground, DRAM L1 + AIS_MT L2)"
	@echo "  make up-batch        Start both containers (background, DRAM L1 + AIS_MT L2)"
	@echo "  make up-gds-l1       Start with hipFile GDS NVMe slab as L1, no L2 (foreground)"
	@echo "  make up-gds-l1-batch Start with hipFile GDS NVMe slab as L1, no L2 (background)"
	@echo "  make down            Stop and remove both containers"
	@echo "  make logs            Follow logs from both containers"
	@echo "  make logs-lmcache    Follow lmcache container logs only"
	@echo "  make logs-vllm       Follow vllm container logs only"
	@echo "  make ps              Show container status"
	@echo "  make shell-lmcache   Exec bash into lmcache container"
	@echo "  make shell-vllm      Exec bash into vllm container"
	@echo "  make restart-vllm    Restart vllm container only (lmcache keeps running)"
	@echo "  make restart-lmcache Restart lmcache container only"
	@echo ""
	@echo "Required env:"
	@echo "  HF_TOKEN       HuggingFace access token"
	@echo "  ROCM_ARCH      GPU arch (detected: $(ROCM_ARCH))"
	@echo ""
	@echo "Key vars (current):"
	@echo "  GPU=$(GPU)  NVME_DATA=$(NVME_DATA)  NFS_DATA=$(NFS_DATA)"
	@echo "  LMCACHE_PORT=$(LMCACHE_PORT)  LMCACHE_L1_SIZE_GB=$(LMCACHE_L1_SIZE_GB) GiB"
	@echo "  LMCACHE_NVME_POOL=$(LMCACHE_NVME_POOL)  LMCACHE_NFS_POOL=$(LMCACHE_NFS_POOL)"
	@echo ""
	@echo "Examples:"
	@echo "  make build"
	@echo "  make up HF_TOKEN=hf_... NVME_DATA=/mnt/nvme NFS_DATA=/mnt/nfs"
	@echo "  make up-batch LMCACHE_L1_SIZE_GB=40 VLLM_MODEL=Qwen/Qwen3-8B"
	@echo "  make restart-vllm VLLM_MODEL=Qwen/Qwen3-14B"
	@echo ""

_check_hf_token:
	@if [ -z "$$HF_TOKEN" ] && [ -n "$(HF_TOKEN_FILE)" ] && [ -r "$(HF_TOKEN_FILE)" ]; then \
		export HF_TOKEN="$$(tr -d '\r\n' < "$(HF_TOKEN_FILE)")"; \
	fi; \
	if [ -z "$$HF_TOKEN" ]; then \
		echo "ERROR: set HF_TOKEN or HF_TOKEN_FILE" >&2; exit 1; \
	fi

_prep_dirs:
	@mkdir -p "$(NVME_DATA)" "$(NFS_DATA)" \
		"$(LOG)/lmcache" "$(LOG)/vllm" \
		"$(HF_HOME)/hub" "$(HF_HOME)/datasets" "$(HF_HOME)/vllm" \
		"$(HF_HOME)/vllm_config" "$(HF_HOME)/torch" "$(HF_HOME)/torch_inductor"

build:
	@test -n "$(ROCM_ARCH)" || { \
		echo "ERROR: ROCM_ARCH empty (install ROCm or set ROCM_ARCH=gfxNNNN)" >&2; exit 1; }
	cd "$(REPO_ROOT)" && $(COMPOSE) build

up: _check_hf_token _prep_dirs
	$(COMPOSE) up

up-batch: _check_hf_token _prep_dirs
	$(COMPOSE) up -d
	@echo "Started. Use 'make logs' to follow or 'make down' to stop."

up-gds-l1: _check_hf_token _prep_dirs
	$(COMPOSE_GDS_L1) up

up-gds-l1-batch: _check_hf_token _prep_dirs
	$(COMPOSE_GDS_L1) up -d
	@echo "Started (GDS L1 mode). Use 'make logs' to follow or 'make down' to stop."

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

logs-lmcache:
	$(COMPOSE) logs -f lmcache

logs-vllm:
	$(COMPOSE) logs -f vllm

ps:
	$(COMPOSE) ps

shell-lmcache:
	docker exec -it vllm-lmcache-mp-lmcache bash -l

shell-vllm:
	docker exec -it vllm-lmcache-mp-vllm-gpu$(GPU) bash -l

# Restart vllm only — lmcache stays up, warm KV state preserved.
restart-vllm:
	$(COMPOSE) restart vllm

# Restart lmcache only (vllm will reconnect over ZMQ).
restart-lmcache:
	$(COMPOSE) restart lmcache
