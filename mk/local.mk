# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Local-dev targets: build, stack lifecycle, shells, logs, venv, reset test.

.PHONY: ensure-compose build up up-batch up-dev up-gds-l1 up-gds-l1-batch \
        down logs logs-lmcache logs-vllm ps shell-lmcache shell-vllm \
        restart-vllm restart-lmcache venv vllm-reset-test bench-serve

# ---- Agentic serving benchmark -----------------------------------------------
BENCH_DURATION   ?= 60     # minutes
BENCH_TOKENS     ?= 256    # max output tokens per turn
BENCH_POOL_SIZE  ?= 0      # 0 = full dataset; N = repeat within N convs (generates cache hits)
BENCH_SESSIONS   ?= 4      # concurrent simulated user sessions
BENCH_THINK_MIN  ?= 1.0    # min think-time between turns (seconds)
BENCH_THINK_MAX  ?= 5.0    # max think-time between turns (seconds)
BENCH_SEED       ?= 42     # random seed — change to select a different conversation pool

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

# Build aic-base, then aic-vllm + aic-lmcache (which sit on top of it) and the
# aic-hsa-snoop sidecar, in one `docker buildx bake` pass.  The build graph lives
# in docker/docker-bake.hcl: bake's `base = "target:base"` build-context wires the
# in-session aic-base into vllm/lmcache directly, so the old per-stage buildx
# calls + OCI-layout export/untar handoff (needed to cross the docker-container
# builder's daemon isolation) are gone.  Version pins, the image tag and the
# cache/secret wiring all live in the HCL; the Makefile only forwards env.
BUILD_PROGRESS ?= auto

# Which bake group to build, and where the result goes.  The defaults are the
# local-developer case: the whole stack, loaded into the local docker daemon.
# CI overrides them to push straight to a registry instead -- `--load` through
# the docker-container driver round-trips every layer through the daemon, which
# on a GitHub runner is both slow and the thing that fills the disk.
#   make build BAKE_TARGETS=vllm BAKE_OUTPUT_FLAG=--push
BAKE_TARGETS     ?= default
BAKE_OUTPUT_FLAG ?= --load

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
	@if [ -n "$(strip $(AIC_CACHE_REF))" ]; then \
		echo "Layer cache: registry $(AIC_CACHE_REF) (mode $(AIC_CACHE_MODE))"; \
	elif [ -n "$(strip $(AIC_CACHE_DIR))" ]; then \
		mkdir -p "$(AIC_CACHE_DIR)/$(_ARCH_SLUG)"; \
		echo "Layer cache: $(AIC_CACHE_DIR)/$(_ARCH_SLUG) (mode $(AIC_CACHE_MODE))"; \
	else \
		echo "Layer cache: disabled (set AIC_CACHE_REF or AIC_CACHE_DIR to enable)"; \
	fi
	@echo "--- build: aic-base/vllm/lmcache/hsa-snoop:$(IMAGE_TAG) + nvme/rdma-exporter:local (ROCM_ARCH=$(ROCM_ARCH)) ---"
	BUILDX_BAKE_ENTITLEMENTS_FS=0 \
	    ROCM_ARCH="$(ROCM_ARCH)" BUILD_JOBS="$(BUILD_JOBS)" AIC_UCX_FAST="$(AIC_UCX_FAST)" \
	    AIC_VERSION="$(AIC_VERSION)" \
	    AIC_CACHE_REF="$(AIC_CACHE_REF)" \
	    AIC_CACHE_DIR="$(AIC_CACHE_DIR)" AIC_CACHE_MODE="$(AIC_CACHE_MODE)" \
	    $(if $(strip $(TLS_CERT)),TLS_CERT="$(TLS_CERT)",) \
	    docker buildx bake \
		--builder $(AIC_BUILDX_BUILDER) \
		--progress=$(BUILD_PROGRESS) \
		$(BAKE_OUTPUT_FLAG) \
		-f "$(REPO_ROOT)/docker/docker-bake.hcl" \
		$(BAKE_TARGETS)
	@echo "Built aic-base/aic-vllm/aic-lmcache/aic-hsa-snoop:$(IMAGE_TAG) (vllm/lmcache/hsa-snoop also tagged :latest)"
	@echo "Built aic-nvme-exporter:local, aic-rdma-exporter:local"

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
	    r=$$(curl -s -o /dev/null -w '%{http_code}' \
	        http://localhost:$${VLLM_HOST_PORT:-8000}/health 2>/dev/null); \
	    [ "$$r" = "200" ] && echo "vLLM healthy after $${i}s" && break; \
	    [ "$$i" = "$${VLM_READY_RETRIES:-60}" ] && echo "ERROR: vLLM not healthy after vllm_reset_test timeout" >&2 && exit 1; \
	    sleep 5; \
	done
	@echo "Clearing vLLM GPU prefix cache..."
	@curl -s -X POST \
	    http://localhost:8000/reset_prefix_cache \
	    -H 'Content-Type: application/json' -d '{}' | grep -q '"success":true' \
	    || { echo "ERROR: vLLM cache reset failed" >&2; exit 1; }
	@echo "Clearing LMCache L1 DRAM cache..."
	@curl -s -X POST \
	    http://localhost:8080/cache/clear \
	    -H 'Content-Type: application/json' \
	    -d '{"tier":"l1","force":true}' | grep -q '"status":"ok"' \
	    || { echo "ERROR: LMCache L1 clear failed" >&2; exit 1; }
	@echo "Resetting LMCache Prometheus counters..."
	@curl -s -X POST \
	    http://localhost:8080/metrics/reset > /dev/null
	$(PYTHON) "$(CURDIR)/benchmarks/vllm_reset_test.py"
	@echo "Test complete. Run 'make down' to stop the stack."

bench-serve: check-hf-token prep-dirs
	@echo "Starting AIC multi-turn serving benchmark ($(BENCH_DURATION)min, $(BENCH_SESSIONS) sessions)..."
	@echo "  Model:       $(VLLM_MODEL)"
	@echo "  Sessions:    $(BENCH_SESSIONS)  Max tokens/turn: $(BENCH_TOKENS)"
	@echo "  Think-time:  $(BENCH_THINK_MIN)–$(BENCH_THINK_MAX)s  Pool: $(BENCH_POOL_SIZE)"
	@echo "  Results:     /tmp/aic-bench-results/"
	@curl -sf http://localhost:$${VLLM_HOST_PORT:-8000}/health >/dev/null || { \
	    echo "ERROR: vLLM not healthy at http://localhost:$${VLLM_HOST_PORT:-8000} — run 'make up' first" >&2; exit 1; }
	$(PYTHON) "$(REPO_ROOT)/benchmarks/aic_bench_serve.py" \
	    --base-url http://localhost:8000 \
	    --model "$(VLLM_MODEL)" \
	    --duration $(BENCH_DURATION) \
	    --sessions $(BENCH_SESSIONS) \
	    --max-tokens $(BENCH_TOKENS) \
	    --think-min $(BENCH_THINK_MIN) \
	    --think-max $(BENCH_THINK_MAX) \
	    $(if $(filter-out 0,$(BENCH_POOL_SIZE)),--pool-size $(BENCH_POOL_SIZE),) \
	    --seed $(BENCH_SEED) \
	    --output /tmp/aic-bench-results
	@echo "Benchmark complete. Run 'make down' to stop the stack."
