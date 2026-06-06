# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# vllm-optimus-kvd recipe Makefile. Mirrors recipies/vllm-lmcache-hipfile/
# layout (build / run / cliff / shell targets, ROCM_ARCH, GPU pinning).

REPO_ROOT     := $(abspath $(CURDIR)/../..)
IMAGE_NAME    ?= vllm-optimus-kvd
GPU           ?= 0
CONTAINER_NAME?= $(IMAGE_NAME)-gpu$(GPU)

# Mount points (host → container).
DATA          ?= /mnt/optimus-nvme       # kvd long+short SSD tiers live here
LOG           ?= $(CURDIR)/logs
HF_HOME       ?= $(HOME)/.cache/huggingface
HF_TOKEN_FILE ?=

CONTAINER_HF_HOME    ?= /hf
CONTAINER_DATA_DIR   ?= /data
CONTAINER_LOG_DIR    ?= /var/log/vllm-optimus-kvd

# vLLM serve knobs (consumed by /app/scripts/vllm-server).
VLLM_MODEL                  ?= openai/gpt-oss-120b
TENSOR_PARALLEL_SIZE        ?= 1
MAX_MODEL_LEN               ?= 131072
GPU_MEMORY_UTILIZATION      ?= 0.90
KV_CACHE_MEMORY_BYTES       ?= 85899345920  # 80 GiB; align to your card
PORT                        ?= 8000

# Optimus connector knobs (env into the engine process — see
# rocserve/engine/vllm/kvd_connector.py for all knobs).
ROCSERVE_KVD_LAYERWISE_LOAD  ?= parallel        # off|stepped|prefetch|parallel
ROCSERVE_KVD_GPU_DIRECT      ?= auto            # 0|1|auto (ais-check)
ROCSERVE_KVD_LOAD_WORKERS    ?= 8
ROCSERVE_KVD_CHUNK_TARGET_MIB?= 128             # P2PDMA sweet spot
ROCSERVE_KVD_LONG_BYTES      ?= 549755813888    # 512 GiB
ROCSERVE_KVD_SHORT_BYTES     ?= 68719476736     #  64 GiB
# C3 crash-safe save publish; default off (fast). Enable on hosts
# without UPS that need clean restart recovery.
ROCSERVE_KVD_FSYNC_SAVE      ?=

# Cliff bench knobs.
CLIFF_CONCURRENCIES ?= 16,32,48,64
CLIFF_ISL           ?= 60000
CLIFF_SHARED_PREFIX ?= 60000   # = ISL → 0 suffix → full-cache test
CLIFF_ITERS         ?= 3
CLIFF_MAX_TOKENS    ?= 64      # >1 so we can verify output correctness too

TZ                  ?= America/Edmonton
ARGS                ?=
EXTRA_DOCKER_RUN_FLAGS ?=
DOCKER_RUN_ATTACH   ?= -it

_ROCM_ARCH_DETECTED := $(shell rocm_agent_enumerator 2>/dev/null | \
	grep -E '^gfx' | head -1)
ROCM_ARCH := $(if $(strip $(ROCM_ARCH)),$(strip $(ROCM_ARCH)),$(_ROCM_ARCH_DETECTED))

.DEFAULT_GOAL := help

help:
	@echo "vllm-optimus-kvd Makefile"
	@echo ""
	@echo "  make build         docker build -t $(IMAGE_NAME) (needs ROCM_ARCH=...)"
	@echo "  make run           docker run -it (foreground)"
	@echo "  make run-batch     docker run -d (background; e.g. for Slurm)"
	@echo "  make cliff         run cliff sweep against running container"
	@echo "  make correctness   run correctness check (greedy vanilla-vs-kvd)"
	@echo "  make shell         exec bash in the running container"
	@echo "  make stop          docker rm -f $(CONTAINER_NAME)"
	@echo ""
	@echo "Env:"
	@echo "  HF_TOKEN, HF_TOKEN_FILE   HuggingFace gated model auth"
	@echo "  ROCM_ARCH                 GPU arch (auto-detected via rocm_agent_enumerator)"
	@echo "  VLLM_MODEL=$(VLLM_MODEL)"
	@echo "  TENSOR_PARALLEL_SIZE=$(TENSOR_PARALLEL_SIZE)"
	@echo "  KV_CACHE_MEMORY_BYTES=$(KV_CACHE_MEMORY_BYTES)"
	@echo "  DATA=$(DATA)              kvd long+short tier root (must exist + be fast)"
	@echo "  LOG=$(LOG)"
	@echo ""
	@echo "Cliff bench:"
	@echo "  CLIFF_CONCURRENCIES=$(CLIFF_CONCURRENCIES)"
	@echo "  CLIFF_ISL=$(CLIFF_ISL)  CLIFF_SHARED_PREFIX=$(CLIFF_SHARED_PREFIX)"
	@echo "  CLIFF_ITERS=$(CLIFF_ITERS)  CLIFF_MAX_TOKENS=$(CLIFF_MAX_TOKENS)"

# Build target — recipe-local, but Dockerfile COPYs from REPO_ROOT
# so build context must be the rocm-aic repo root.
#
# Optimus is a private repo, so the clone inside the image needs a
# GitHub token. Set GH_TOKEN in the shell (or autoload via
# `gh auth token` if you have gh installed and logged in). The token
# is passed via BuildKit `--secret` (NOT baked into the image; not
# visible in `docker history`).
build:
	@test -n "$(ROCM_ARCH)" || { echo "ERROR: ROCM_ARCH not set / detected"; exit 1; }
	@command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found"; exit 1; }
	@if [ -z "$$GH_TOKEN" ]; then \
		if command -v gh >/dev/null 2>&1; then \
			GH_TOKEN_AUTO="$$(gh auth token 2>/dev/null)"; \
			if [ -n "$$GH_TOKEN_AUTO" ]; then \
				export GH_TOKEN="$$GH_TOKEN_AUTO"; \
				echo "Using gh auth token for AMD-AGI/Optimus clone"; \
			fi; \
		fi; \
	fi; \
	if [ -z "$$GH_TOKEN" ]; then \
		echo "ERROR: GH_TOKEN not set. Either:"; \
		echo "  1) export GH_TOKEN=<your-token-with-AMD-AGI-Optimus-read>"; \
		echo "  2) install gh (https://cli.github.com) and 'gh auth login'"; \
		exit 1; \
	fi; \
	cd $(REPO_ROOT) && \
	DOCKER_BUILDKIT=1 docker build \
		--build-arg ROCM_ARCH=$(ROCM_ARCH) \
		--build-arg OPTIMUS_GIT_REF=$(if $(OPTIMUS_GIT_REF),$(OPTIMUS_GIT_REF),main) \
		--secret id=gh_token,env=GH_TOKEN \
		-f recipies/vllm-optimus-gpu-direct/Dockerfile \
		-t $(IMAGE_NAME) \
		.

run:
	@$(MAKE) -f "$(CURDIR)/Makefile" _docker_run DOCKER_RUN_ATTACH=-it

run-batch:
	@$(MAKE) -f "$(CURDIR)/Makefile" _docker_run DOCKER_RUN_ATTACH=-d

_docker_run:
	@set -e; \
	command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found"; exit 1; }; \
	if [ -z "$$HF_TOKEN" ] && [ -n "$(HF_TOKEN_FILE)" ] && [ -r "$(HF_TOKEN_FILE)" ]; then \
		export HF_TOKEN="$$(tr -d '\r\n' < '$(HF_TOKEN_FILE)')"; \
	fi; \
	if [ -z "$$HF_TOKEN" ]; then \
		echo "ERROR: set HF_TOKEN or HF_TOKEN_FILE" >&2; \
		exit 1; \
	fi; \
	mkdir -p "$(DATA)" "$(LOG)" "$(HF_HOME)"; \
	docker rm -f "$(CONTAINER_NAME)" >/dev/null 2>&1 || true; \
	docker run $(DOCKER_RUN_ATTACH) --rm \
		--name="$(CONTAINER_NAME)" \
		--hostname="$(CONTAINER_NAME)" \
		--ipc=host --network=host \
		--cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
		--security-opt seccomp=unconfined \
		--device=/dev/kfd --device=/dev/dri --group-add=video \
		-v "$(CURDIR)/configs":/app/configs \
		-v "$(CURDIR)/scripts":/app/scripts \
		-v "$(CURDIR)/tests":/app/tests \
		-v "$(DATA)":$(CONTAINER_DATA_DIR) \
		-v "$(LOG)":$(CONTAINER_LOG_DIR) \
		-v "$(HF_HOME)":$(CONTAINER_HF_HOME) \
		--env HIP_VISIBLE_DEVICES="$(GPU)" \
		--env HF_TOKEN="$$HF_TOKEN" \
		--env HF_HOME=$(CONTAINER_HF_HOME) \
		--env HF_HUB_CACHE=$(CONTAINER_HF_HOME)/hub \
		--env TZ=$(TZ) \
		--env VLLM_MODEL=$(VLLM_MODEL) \
		--env TENSOR_PARALLEL_SIZE=$(TENSOR_PARALLEL_SIZE) \
		--env MAX_MODEL_LEN=$(MAX_MODEL_LEN) \
		--env GPU_MEMORY_UTILIZATION=$(GPU_MEMORY_UTILIZATION) \
		--env KV_CACHE_MEMORY_BYTES=$(KV_CACHE_MEMORY_BYTES) \
		--env PORT=$(PORT) \
		--env CONTAINER_DATA_DIR=$(CONTAINER_DATA_DIR) \
		--env CONTAINER_LOG_DIR=$(CONTAINER_LOG_DIR) \
		--env ROCSERVE_KVD_LAYERWISE_LOAD=$(ROCSERVE_KVD_LAYERWISE_LOAD) \
		--env ROCSERVE_KVD_GPU_DIRECT=$(ROCSERVE_KVD_GPU_DIRECT) \
		--env ROCSERVE_KVD_LOAD_WORKERS=$(ROCSERVE_KVD_LOAD_WORKERS) \
		--env ROCSERVE_KVD_CHUNK_TARGET_MIB=$(ROCSERVE_KVD_CHUNK_TARGET_MIB) \
		--env ROCSERVE_KVD_LONG_BYTES=$(ROCSERVE_KVD_LONG_BYTES) \
		--env ROCSERVE_KVD_SHORT_BYTES=$(ROCSERVE_KVD_SHORT_BYTES) \
		--env ROCSERVE_KVD_FSYNC_SAVE=$(ROCSERVE_KVD_FSYNC_SAVE) \
		--env ARGS="$(ARGS)" \
		$(EXTRA_DOCKER_RUN_FLAGS) \
		$(IMAGE_NAME)

cliff:
	@docker exec $(if $(shell tty -s && echo yes),-it,-i) "$(CONTAINER_NAME)" \
		python3 /app/tests/run_cliff.py \
			--endpoint http://127.0.0.1:$(PORT) \
			--model $(VLLM_MODEL) \
			--concurrencies $(CLIFF_CONCURRENCIES) \
			--isl $(CLIFF_ISL) \
			--shared-prefix-tokens $(CLIFF_SHARED_PREFIX) \
			--iters $(CLIFF_ITERS) \
			--max-tokens $(CLIFF_MAX_TOKENS) \
			--out /var/log/vllm-optimus-kvd/cliff-$$(date +%Y%m%d-%H%M%S).csv

# Greedy correctness check: same prompts to vanilla (no kvd) vs kvd
# arms, compare outputs token-by-token. KV restoration error reveals
# itself as deterministic-greedy divergence on iter 2+ where kvd
# load fires.
correctness:
	@test -n "$(VANILLA_ENDPOINT)" || { \
		echo "ERROR: set VANILLA_ENDPOINT=http://<host>:<port> for the vanilla arm"; \
		exit 1; \
	}
	@docker exec $(if $(shell tty -s && echo yes),-it,-i) "$(CONTAINER_NAME)" \
		python3 /app/tests/correctness_check.py \
			--vanilla $(VANILLA_ENDPOINT) \
			--kvd http://127.0.0.1:$(PORT) \
			--model $(VLLM_MODEL) \
			--c 16 --iters 3 --max-tokens 64 \
			--isl 60000 --shared-prefix-tokens 60000

shell:
	@docker exec -it "$(CONTAINER_NAME)" bash

stop:
	@docker rm -f "$(CONTAINER_NAME)" || true

.PHONY: help build run run-batch _docker_run cliff correctness shell stop
