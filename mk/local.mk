# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Local-dev targets: build, stack lifecycle, shells, logs, venv, reset test.

.PHONY: ensure-compose build build-local build-cached up up-batch up-dev up-gds-l1 up-gds-l1-batch \
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

build: ensure-compose
	@test -n "$(ROCM_ARCH)" || { \
		echo "ERROR: ROCM_ARCH empty (install ROCm or set ROCM_ARCH=gfxNNNN)" >&2; exit 1; }
	cd "$(REPO_ROOT)" && $(COMPOSE_CACHE) build \
		$(if $(TLS_CERT),--secret id=tls_cert$(comma)src=$(TLS_CERT),)
	@docker tag "$(IMAGE_REF)" "$(IMAGE_NAME):latest"
	@echo "Built $(IMAGE_REF) (also tagged $(IMAGE_NAME):latest)"

# Two-stage local build: aic-base (torch + torchvision) then aic-vllm.
# Use this on nodes without Slurm when docker compose build fails because the
# vllm Dockerfile requires aic-base as a --build-context.
_BUILD_LOCAL_ARGS := \
	--build-arg ROCM_ARCH="$(ROCM_ARCH)" \
	--build-arg BUILD_JOBS="$(BUILD_JOBS)" \
	--build-arg AIC_UCX_FAST="$(AIC_UCX_FAST)" \
	$(foreach _v,$(_FRAMEWORK_VERSION_ARGS),$(if $(value $(_v)),--build-arg $(_v)="$(value $(_v))")) \
	$(if $(TLS_CERT),--secret id=tls_cert$(comma)src=$(TLS_CERT),)

build-local:
	@test -n "$(ROCM_ARCH)" || { \
		echo "ERROR: ROCM_ARCH empty (install ROCm or set ROCM_ARCH=gfxNNNN)" >&2; exit 1; }
	@echo "--- build-local [1/2]: aic-base:$(IMAGE_TAG) (ROCM_ARCH=$(ROCM_ARCH)) ---"
	DOCKER_BUILDKIT=1 docker build \
		--progress=plain \
		$(_BUILD_LOCAL_ARGS) \
		-f "$(REPO_ROOT)/docker/base/Dockerfile" \
		-t "aic-base:$(IMAGE_TAG)" \
		"$(REPO_ROOT)"
	@echo "--- build-local [2/2]: $(VLLM_IMAGE_REF) ---"
	DOCKER_BUILDKIT=1 docker build \
		--progress=plain \
		$(_BUILD_LOCAL_ARGS) \
		--build-context base="docker-image://aic-base:$(IMAGE_TAG)" \
		-f "$(REPO_ROOT)/docker/vllm/Dockerfile" \
		-t "$(VLLM_IMAGE_REF)" \
		"$(REPO_ROOT)"
	@docker tag "$(VLLM_IMAGE_REF)" "$(VLLM_IMAGE_NAME):latest"
	@echo "Built $(VLLM_IMAGE_REF) (also tagged $(VLLM_IMAGE_NAME):latest)"

build-cached:
	@test -n "$(ROCM_ARCH)" || { \
		echo "ERROR: ROCM_ARCH empty (install ROCm or set ROCM_ARCH=gfxNNNN)" >&2; exit 1; }
	@if ! docker buildx inspect $(AIC_LOCAL_BUILDER) >/dev/null 2>&1; then \
		echo "Creating buildx builder $(AIC_LOCAL_BUILDER) (docker-container driver)..."; \
		docker buildx create --name $(AIC_LOCAL_BUILDER) --driver docker-container --bootstrap; \
	fi
	@mkdir -p "$(AIC_LOCAL_CACHE_DIR)"
	@echo "Cache dir: $(AIC_LOCAL_CACHE_DIR)"
	$(_FRAMEWORK_VERSION_ENV) DOCKER_BUILDKIT=1 \
	docker buildx build \
		--builder $(AIC_LOCAL_BUILDER) \
		--progress=plain \
		--load \
		--build-arg ROCM_ARCH="$(ROCM_ARCH)" \
		--build-arg BUILD_JOBS="$(BUILD_JOBS)" \
		--build-arg AIC_UCX_FAST="$(AIC_UCX_FAST)" \
		$(if $(TLS_CERT),--secret id=tls_cert$(comma)src=$(TLS_CERT),) \
		--cache-from type=local,src="$(AIC_LOCAL_CACHE_DIR)" \
		--cache-to   type=local,dest="$(AIC_LOCAL_CACHE_DIR)",mode=max \
		-f "$(REPO_ROOT)/docker/lmcache/Dockerfile" \
		-t "$(IMAGE_REF)" \
		-t "$(IMAGE_NAME):latest" \
		"$(REPO_ROOT)"
	@echo "Built $(IMAGE_REF) (also tagged $(IMAGE_NAME):latest)"
	@echo "Cache stored in $(AIC_LOCAL_CACHE_DIR)"

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
	GDS_MODE=1 $(COMPOSE_CACHE) up

up-gds-l1-batch: ensure-compose check-hf-token check-gds-slab prep-dirs
	GDS_MODE=1 $(COMPOSE_CACHE) up -d
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
	@for i in $$(seq 1 60); do \
	    r=$$(docker exec aic-client curl -s -o /dev/null -w '%{http_code}' \
	        http://aic-vllm-gpu0:8000/health 2>/dev/null); \
	    [ "$$r" = "200" ] && echo "vLLM healthy after $${i}s" && break; \
	    [ "$$i" = "60" ] && echo "ERROR: vLLM not healthy after 300s" >&2 && exit 1; \
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
