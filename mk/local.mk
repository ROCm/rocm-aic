# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Local-dev targets: build, stack lifecycle, shells, logs, venv, reset test.

.PHONY: ensure-compose build up up-batch up-dev up-gds-l1 up-gds-l1-batch \
        down logs logs-lmcache logs-vllm ps shell-lmcache shell-vllm \
        restart-vllm restart-lmcache venv vllm-reset-test

ensure-compose:
	@if docker compose version >/dev/null 2>&1; then \
		echo "docker compose present: $$(docker compose version --short 2>/dev/null)"; \
	else \
		echo "docker compose plugin missing; installing $(COMPOSE_PLUGIN_VERSION) -> ~/.docker/cli-plugins"; \
		mkdir -p "$$HOME/.docker/cli-plugins" && \
		arch="$$(uname -m)" && \
		curl -fsSL "https://github.com/docker/compose/releases/download/$(COMPOSE_PLUGIN_VERSION)/docker-compose-linux-$$arch" \
			-o "$$HOME/.docker/cli-plugins/docker-compose" && \
		chmod +x "$$HOME/.docker/cli-plugins/docker-compose" && \
		docker compose version >/dev/null 2>&1 || { \
			echo "ERROR: docker compose still unavailable after install" >&2; exit 1; }; \
		echo "installed: $$(docker compose version --short 2>/dev/null)"; \
	fi

# Three-stage build: aic-base (torch + torchvision), then aic-vllm and
# aic-lmcache, which both sit on top of it.  Each Dockerfile declares a fallback
# `base` stage (FROM $ROCM_BASE_IMAGE); supplying a `base` build-context is what
# makes vllm/lmcache inherit the torch-bearing aic-base instead.  Without it they
# build green against the raw ROCm image and ship without torch.
#
# The docker-container buildx driver is isolated from the host daemon, so a
# docker-image:// build-context cannot see a locally built aic-base.  Export the
# base in OCI layout and hand it over as oci-layout:// — the same approach
# .slurm/run-build-distribute.sh uses for the distributed build.
BUILD_PROGRESS ?= auto

_BUILD_ARGS := \
	--build-arg ROCM_ARCH="$(ROCM_ARCH)" \
	--build-arg BUILD_JOBS="$(BUILD_JOBS)" \
	--build-arg AIC_UCX_FAST="$(AIC_UCX_FAST)" \
	$(foreach _v,$(_FRAMEWORK_ARGS),$(if $(value $(_v)),--build-arg $(_v)="$(value $(_v))")) \
	$(if $(TLS_CERT),--secret id=tls_cert$(comma)src=$(TLS_CERT),)

# Layer cache is opt-in: empty AIC_CACHE_DIR means no --cache-from/--cache-to.
# ignore-error=true so a cache-export failure never fails the build.
_CACHE_DIR  = $(if $(strip $(AIC_CACHE_DIR)),$(AIC_CACHE_DIR)/$(_ARCH_SLUG),)
_CACHE_ARGS = $(if $(strip $(AIC_CACHE_DIR)),\
	--cache-from type=local$(comma)src="$(_CACHE_DIR)" \
	--cache-to type=local$(comma)dest="$(_CACHE_DIR)"$(comma)mode=$(AIC_CACHE_MODE)$(comma)ignore-error=true,)

# Prefix for every buildx invocation: see AIC_BUILD_NO_PROXY in the top Makefile.
# Appends to an existing NO_PROXY rather than replacing it.
_NO_PROXY_ENV = $(if $(strip $(AIC_BUILD_NO_PROXY)),\
	NO_PROXY="$${NO_PROXY:+$${NO_PROXY}$(comma)}$(AIC_BUILD_NO_PROXY)" \
	no_proxy="$${no_proxy:+$${no_proxy}$(comma)}$(AIC_BUILD_NO_PROXY)",)

# Base image OCI layout staging area.  Multi-GB — keep it off a small /tmp tmpfs
# by pointing TMPDIR at a real filesystem.
_OCI_TAR = $${TMPDIR:-/tmp}/aic-base-oci.$(IMAGE_TAG).tar
_OCI_DIR = $${TMPDIR:-/tmp}/aic-base-oci.$(IMAGE_TAG).d

build:
	@test -n "$(ROCM_ARCH)" || { \
		echo "ERROR: ROCM_ARCH empty (install ROCm or set ROCM_ARCH=gfxNNNN)" >&2; exit 1; }
	@if ! docker buildx inspect $(AIC_BUILDX_BUILDER) >/dev/null 2>&1; then \
		echo "Creating buildx builder $(AIC_BUILDX_BUILDER) (docker-container driver)..."; \
		docker buildx create --name $(AIC_BUILDX_BUILDER) --driver docker-container \
			--driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=-1 \
			--driver-opt env.BUILDKIT_STEP_LOG_MAX_SPEED=10485760 \
			--bootstrap; \
	fi
	@if [ -n "$(strip $(AIC_CACHE_DIR))" ]; then \
		mkdir -p "$(_CACHE_DIR)"; \
		echo "Layer cache: $(_CACHE_DIR) (mode $(AIC_CACHE_MODE))"; \
	else \
		echo "Layer cache: disabled (set AIC_CACHE_DIR to enable)"; \
	fi
	@rm -rf "$(_OCI_TAR)" "$(_OCI_DIR)"
	@echo "--- build [1/4]: aic-base:$(IMAGE_TAG) (ROCM_ARCH=$(ROCM_ARCH)) ---"
	$(_NO_PROXY_ENV) docker buildx build \
		--builder $(AIC_BUILDX_BUILDER) \
		--progress=$(BUILD_PROGRESS) \
		--load \
		--output type=oci$(comma)dest="$(_OCI_TAR)" \
		$(_BUILD_ARGS) \
		$(_CACHE_ARGS) \
		-f "$(REPO_ROOT)/docker/base/Dockerfile" \
		-t "aic-base:$(IMAGE_TAG)" \
		"$(REPO_ROOT)"
	@mkdir -p "$(_OCI_DIR)"
	@tar -xf "$(_OCI_TAR)" -C "$(_OCI_DIR)"
	@rm -f "$(_OCI_TAR)"
	@echo "--- build [2/4]: $(VLLM_IMAGE_REF) ---"
	$(_NO_PROXY_ENV) docker buildx build \
		--builder $(AIC_BUILDX_BUILDER) \
		--progress=$(BUILD_PROGRESS) \
		--load \
		$(_BUILD_ARGS) \
		$(_CACHE_ARGS) \
		--build-context base="oci-layout://$(_OCI_DIR)" \
		-f "$(REPO_ROOT)/docker/vllm/Dockerfile" \
		-t "$(VLLM_IMAGE_REF)" \
		"$(REPO_ROOT)"
	@docker tag "$(VLLM_IMAGE_REF)" "$(VLLM_IMAGE_NAME):latest"
	@echo "Built $(VLLM_IMAGE_REF) (also tagged $(VLLM_IMAGE_NAME):latest)"
	@echo "--- build [3/4]: $(LMCACHE_IMAGE_REF) ---"
	$(_NO_PROXY_ENV) docker buildx build \
		--builder $(AIC_BUILDX_BUILDER) \
		--progress=$(BUILD_PROGRESS) \
		--load \
		$(_BUILD_ARGS) \
		$(_CACHE_ARGS) \
		--build-context base="oci-layout://$(_OCI_DIR)" \
		-f "$(REPO_ROOT)/docker/lmcache/Dockerfile" \
		-t "$(LMCACHE_IMAGE_REF)" \
		"$(REPO_ROOT)"
	@docker tag "$(LMCACHE_IMAGE_REF)" "$(LMCACHE_IMAGE_NAME):latest"
	@echo "Built $(LMCACHE_IMAGE_REF) (also tagged $(LMCACHE_IMAGE_NAME):latest)"
	@rm -rf "$(_OCI_DIR)"
	@echo "--- build [4/4]: $(HSA_SNOOP_IMAGE_REF) ---"
	$(_NO_PROXY_ENV) docker buildx build \
		--builder $(AIC_BUILDX_BUILDER) \
		--progress=$(BUILD_PROGRESS) \
		--load \
		$(_BUILD_ARGS) \
		$(_CACHE_ARGS) \
		-f "$(REPO_ROOT)/docker/hsa-snoop/Dockerfile" \
		-t "$(HSA_SNOOP_IMAGE_REF)" \
		"$(REPO_ROOT)"
	@docker tag "$(HSA_SNOOP_IMAGE_REF)" "$(HSA_SNOOP_IMAGE_NAME):latest"
	@echo "Built $(HSA_SNOOP_IMAGE_REF) (also tagged $(HSA_SNOOP_IMAGE_NAME):latest)"

up: ensure-compose check-hf-token prep-dirs
	@mkdir -p "$(AIC_METRICS_DIR)"
	PROM_UID="$$(id -u)" PROM_GID="$$(id -g)" \
	    $(COMPOSE_CACHE) --profile monitoring up

up-batch: ensure-compose check-hf-token prep-dirs
	@mkdir -p "$(AIC_METRICS_DIR)"
	PROM_UID="$$(id -u)" PROM_GID="$$(id -g)" \
	    $(COMPOSE_CACHE) --profile monitoring up -d
	@echo "Started. Use 'make logs' to follow or 'make down' to stop."

up-dev: ensure-compose check-hf-token prep-dirs
	@mkdir -p "$(AIC_METRICS_DIR)"
	PROM_UID="$$(id -u)" PROM_GID="$$(id -g)" \
	    VLLM_EXTRA_ARGS="--enforce-eager $${VLLM_EXTRA_ARGS}" \
	    $(COMPOSE_CACHE) --profile monitoring up -d
	@echo "Started in dev mode (enforce-eager, no CUDA graphs). Use 'make logs' to follow."

up-gds-l1: ensure-compose check-hf-token check-gds-slab prep-dirs
	@mkdir -p "$(AIC_METRICS_DIR)"
	PROM_UID="$$(id -u)" PROM_GID="$$(id -g)" GDS_MODE=1 $(COMPOSE_CACHE) --profile monitoring up

up-gds-l1-batch: ensure-compose check-hf-token check-gds-slab prep-dirs
	@mkdir -p "$(AIC_METRICS_DIR)"
	PROM_UID="$$(id -u)" PROM_GID="$$(id -g)" GDS_MODE=1 $(COMPOSE_CACHE) --profile monitoring up -d
	@echo "Started (GDS L1 mode + monitoring). Use 'make logs' to follow or 'make down' to stop."
	@echo "Started (GDS L1 mode). Use 'make logs' to follow or 'make down' to stop."

down:
	$(COMPOSE_CACHE) --profile monitoring down

logs:
	$(COMPOSE_CACHE) logs -f

logs-lmcache:
	$(COMPOSE_CACHE) logs -f lmcache

logs-vllm:
	$(COMPOSE_CACHE) logs -f vllm

ps:
	$(COMPOSE_CACHE) ps

shell-lmcache:
	docker exec -it aic-lmcache bash -l

shell-vllm:
	docker exec -it aic-vllm-gpu$(GPU) bash -l

restart-vllm:
	$(COMPOSE_CACHE) restart vllm

restart-lmcache:
	$(COMPOSE_CACHE) restart lmcache

venv:
	@if [ ! -d "$(REPO_ROOT)/.venv" ]; then \
		python3 -m venv "$(REPO_ROOT)/.venv"; \
	fi
	"$(REPO_ROOT)/.venv/bin/pip" install --upgrade pip
	"$(REPO_ROOT)/.venv/bin/pip" install -e "$(CURDIR)[dev]"
	@echo "venv ready at $(REPO_ROOT)/.venv"
	@echo "Activate: source $(REPO_ROOT)/.venv/bin/activate"

vllm-reset-test: check-hf-token prep-dirs
	@echo "Starting LMCache L1+L2 retrieval test (L1=$(LMCACHE_L1_SIZE_GB)GiB) + NIXL POSIX L2..."
	@mkdir -p "$(AIC_METRICS_DIR)"
	PROM_UID="$$(id -u)" PROM_GID="$$(id -g)" \
	    LMCACHE_L1_SIZE_GB=$(LMCACHE_L1_SIZE_GB) \
	    VLLM_EXTRA_ARGS="--enforce-eager $${VLLM_EXTRA_ARGS}" \
	    AIC_L2_BACKEND=$(AIC_L2_BACKEND) \
	    $(COMPOSE_CACHE) --profile monitoring up -d
	@echo "Waiting for vLLM to be healthy..."
	@for i in $$(seq 1 $${VLM_READY_RETRIES:-60}); do \
	    r=$$(docker exec aic-client curl -s -o /dev/null -w '%{http_code}' \
	        http://aic-vllm-gpu0:8000/health 2>/dev/null); \
	    [ "$$r" = "200" ] && echo "vLLM healthy after $${i}s" && break; \
	    [ "$$i" = "$${VLM_READY_RETRIES:-60}" ] && echo "ERROR: vLLM not healthy after vllm_reset_test timeout" >&2 && exit 1; \
	    sleep 5; \
	done
	@echo "Clearing vLLM GPU prefix cache..."
	@docker exec aic-client curl -s -X POST \
	    http://aic-vllm-gpu0:8000/reset_prefix_cache \
	    -H 'Content-Type: application/json' -d '{}' | grep -q '"success":true' \
	    || { echo "ERROR: vLLM cache reset failed" >&2; exit 1; }
	@echo "Clearing LMCache L1 DRAM cache..."
	@docker exec aic-client curl -s -X POST \
	    http://aic-lmcache:8080/cache/clear \
	    -H 'Content-Type: application/json' \
	    -d '{"tier":"l1","force":true}' | grep -q '"status":"ok"' \
	    || { echo "ERROR: LMCache L1 clear failed" >&2; exit 1; }
	@echo "Resetting LMCache Prometheus counters..."
	@docker exec aic-client curl -s -X POST \
	    http://aic-lmcache:8080/metrics/reset > /dev/null
	$(PYTHON) "$(CURDIR)/benchmarks/vllm_reset_test.py"
	@echo "Test complete. Run 'make down' to stop the stack."
