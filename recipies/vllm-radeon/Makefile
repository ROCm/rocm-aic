# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

IMAGE_NAME ?= vllm-radeon
GPU ?= 0
# **`docker run --name`** / **`--hostname`** (default **`$(IMAGE_NAME)-gpu$(GPU)`**, e.g. **vllm-radeon-gpu0**).
CONTAINER_NAME ?= $(IMAGE_NAME)-gpu$(GPU)
DATA ?= /mnt/lmcache-nvme
LOG ?= $(CURDIR)/logs
HF_HOME ?= $(HOME)/.cache/huggingface
HF_TOKEN_FILE ?=
CONTAINER_HF_HOME ?= /hf
CONTAINER_DATA_DIR ?= /data
CONTAINER_LOG_DIR ?= /var/log/vllm-radeon
TZ ?= America/Edmonton
RADEON_LMCACHE_IO ?= hipfile
RADEON_LMCACHE_LOG_LEVEL ?=
RADEON_LMCACHE_GDS_BUFFER_SIZE ?=
# Enables vLLM dev HTTP APIs (e.g. POST /reset_prefix_cache). Set to 0 to disable.
VLLM_SERVER_DEV_MODE ?= 1
# Served model (vllm-server + Slurm benchmarks). Unset → configs/vllm-radeon.yaml default.
VLLM_MODEL ?=
ARGS ?=
EXTRA_DOCKER_RUN_FLAGS ?=
DOCKER_RUN_ATTACH ?= -it

# Gutenberg long-context fixtures (gitignored under data/; not LMCache DATA mount)
BOOK_SLUG ?= war-and-peace
BOOK_PG_ID ?= 2600
BOOK_TITLE ?= War and Peace
BOOK_AUTHOR ?= Leo Tolstoy
BOOK_CHUNK_WORDS ?= 10000
BOOK_CHUNK_COUNT ?= 100
BOOK_DATA_DIR ?= $(CURDIR)/data/$(BOOK_SLUG)
BOOK_DATA_ROOT ?= $(CURDIR)/data
GUTENBERG_MANIFEST ?= $(CURDIR)/configs/gutenberg-library.json
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

.PHONY: help build vllm-radeon run run-batch data data-all

.DEFAULT_GOAL := help

help:
	@echo "vllm-radeon Makefile"
	@echo ""
	@echo "  make build         docker build -t $(IMAGE_NAME)"
	@echo "  make vllm-radeon    alias for build"
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
	@set -e; \
	command -v $(PYTHON) >/dev/null 2>&1 || { \
		echo "ERROR: $(PYTHON) not found" >&2; exit 1; \
	}; \
	_opts="--manifest $(GUTENBERG_MANIFEST) --data-root $(BOOK_DATA_ROOT) --delay $(DATA_ALL_DELAY)"; \
	if [ "$(DATA_ALL_SKIP_EXISTING)" = "1" ]; then _opts="$$_opts --skip-existing"; fi; \
	if [ -n "$(DATA_ALL_BOOK)" ]; then _opts="$$_opts --book $(DATA_ALL_BOOK)"; fi; \
	if [ -n "$(DATA_ALL_LIMIT)" ]; then _opts="$$_opts --limit $(DATA_ALL_LIMIT)"; fi; \
	echo "make data-all: manifest=$(GUTENBERG_MANIFEST) -> $(BOOK_DATA_ROOT)"; \
	$(PYTHON) "$(CURDIR)/scripts/build-gutenberg-library.py" $$_opts

data:
	@set -e; \
	command -v $(PYTHON) >/dev/null 2>&1 || { \
		echo "ERROR: $(PYTHON) not found" >&2; exit 1; \
	}; \
	echo "make data: BOOK_SLUG=$(BOOK_SLUG) PG#$(BOOK_PG_ID) -> $(BOOK_DATA_DIR)"; \
	$(PYTHON) "$(CURDIR)/scripts/split-gutenberg-random-chunks.py" \
		--pg-id "$(BOOK_PG_ID)" \
		--slug "$(BOOK_SLUG)" \
		-o "$(BOOK_DATA_DIR)" \
		--chunk-words "$(BOOK_CHUNK_WORDS)" \
		--count "$(BOOK_CHUNK_COUNT)"; \
	$(PYTHON) "$(CURDIR)/scripts/gen-questions-json.py" \
		--slug "$(BOOK_SLUG)" \
		--title "$(BOOK_TITLE)" \
		--author "$(BOOK_AUTHOR)" \
		--pg-id "$(BOOK_PG_ID)" \
		-o "$(BOOK_DATA_DIR)/$(BOOK_SLUG).questions.json"

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
		--ipc=host \
		--network=host \
		--cap-add=CAP_SYS_ADMIN \
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
		--env RADEON_CONTAINER_DATA_DIR="$(CONTAINER_DATA_DIR)" \
		--env RADEON_SERVER_LOG_DIR="$(CONTAINER_LOG_DIR)" \
		--env RADEON_LMCACHE_IO="$(RADEON_LMCACHE_IO)" \
		--env RADEON_LMCACHE_LOG_LEVEL="$(RADEON_LMCACHE_LOG_LEVEL)" \
		--env RADEON_LMCACHE_GDS_BUFFER_SIZE="$(RADEON_LMCACHE_GDS_BUFFER_SIZE)" \
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

build vllm-radeon:
	@command -v docker >/dev/null 2>&1 || { \
		echo "ERROR: docker not found (install Docker Engine)" >&2; \
		exit 1; \
	}
	@test -n "$(ROCM_ARCH)" || { \
		echo "ERROR: ROCM_ARCH is empty (install ROCm / rocm_agent_enumerator," >&2; \
		echo "       or set e.g. ROCM_ARCH=gfx1201 for RX 9070 XT)" >&2; \
		exit 1; \
	}
	docker build . -t $(IMAGE_NAME) $(BUILD_ARGS)
