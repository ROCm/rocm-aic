# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Distributed / Slurm targets: build, test, emulate, profile, accuracy, cliff.

# _user_or(VAR, default): use the user-supplied value when the var came from the
# command line or environment; fall back to `default` when it came from the
# Makefile's own ?= default (i.e. origin == "file").
_user_or = $(if $(filter file,$(origin $(1))),$(2),$($(1)))

# _SPUR_SBATCH(script, extra-args): build the sbatch invocation for a given
# .slurm script, switching between SPUR and standard Slurm.
ifeq ($(AIC_SPUR_CLUSTER),1)
_SPUR_SBATCH = $(_CLIFF_STRIP) SPUR_CONTROLLER_ADDR=$(AIC_SPUR_CONTROLLER) sbatch \
    $(_SPUR_COMMON_ARGS) $(2) $(1) 2>&1 | \
    tee /dev/stderr | grep -oE '[0-9]+$$' | tail -1
else
_SPUR_SBATCH = $(_CLIFF_STRIP) sbatch --parsable \
    $(2) $(1)
endif

# ---- Cliff GFX / constraint selection ----------------------------------------
AIC_CLIFF_GFX  ?=
AIC_CLIFF_GPUS ?= 1
ifeq ($(strip $(AIC_CLIFF_CONSTRAINT)),)
ifneq ($(strip $(AIC_CLIFF_GFX)),)
AIC_CLIFF_CONSTRAINT := $(shell echo '$(AIC_CLIFF_GFX)' | tr '[:lower:]' '[:upper:]')
else
AIC_CLIFF_CONSTRAINT := GFX942&NVME
endif
endif

ifeq ($(AIC_SPUR_CLUSTER),1)
_CLIFF_SPUR_CTL   := SPUR_CONTROLLER_ADDR=$(AIC_SPUR_CONTROLLER)
_SPUR_COMMON_ARGS := --partition=amd-spur --constraint= --gres= --gpus=$(AIC_CLIFF_GPUS) \
    $(if $(AIC_CLIFF_NODE),--nodelist=$(AIC_CLIFF_NODE),)
_CLIFF_SUBMIT     = $(_CLIFF_SPUR_CTL) $(_CLIFF_STRIP) sbatch \
    $(_SPUR_COMMON_ARGS) $(1) .slurm/run-cliff.sbatch 2>&1 | \
    tee /dev/stderr | grep -oE '[0-9]+$$' | tail -1
_KVBENCH_SBATCH_ARGS := --partition=amd-spur --constraint= \
    $(if $(AIC_KVBENCH_NODE),--nodelist=$(AIC_KVBENCH_NODE),)
_KVBENCH_SUBMIT := SPUR_CONTROLLER_ADDR=$(AIC_SPUR_CONTROLLER) $(_CLIFF_STRIP) sbatch \
    $(_KVBENCH_SBATCH_ARGS) $(1) .slurm/run-kvbench-cliff.sbatch 2>&1 | \
    tee /dev/stderr | grep -oE '[0-9]+$$' | tail -1
else
_CLIFF_SBATCH_ARGS := --constraint='$(AIC_CLIFF_CONSTRAINT)' \
    $(if $(AIC_CLIFF_NODE),--nodelist=$(AIC_CLIFF_NODE),)
_CLIFF_SUBMIT     = $(_CLIFF_STRIP) sbatch --parsable \
    $(_CLIFF_SBATCH_ARGS) $(1) .slurm/run-cliff.sbatch
ifdef AIC_KVBENCH_NODE
_KVBENCH_NODE_ARG := --nodelist=$(AIC_KVBENCH_NODE)
endif
ifdef AIC_KVBENCH_CONSTRAINT
_KVBENCH_CONSTRAINT_ARG := --constraint='$(AIC_KVBENCH_CONSTRAINT)'
endif
_KVBENCH_SUBMIT := $(_CLIFF_STRIP) sbatch --parsable \
    $(_KVBENCH_NODE_ARG) $(_KVBENCH_CONSTRAINT_ARG) $(1) .slurm/run-kvbench-cliff.sbatch
endif

# ---- dist-build family -------------------------------------------------------
# All nine targets are thin wrappers over run-build-distribute.sh.
# dist-build-base/vllm/lmcache also require the corresponding Slurm subcommands
# (build-base, build-vllm, build-lmcache) which are implemented in the script.

.PHONY: dist-build dist-build-fast dist-build-emulate \
        dist-build-base dist-build-vllm dist-build-lmcache dist-build-parallel \
        dist-build-exporters dist-build-monitoring dist-push \
        smoke-test smoke-test-fast tiny-test tiny-test-fast \
        emulate-test emulate-mp-test emulate-validate profile-capture \
        accuracy-test accuracy-test-fast accuracy-test-very-fast \
        install-ci-scripts prometheus-dump \
        cliff-kvbench-submit cliff-submit cliff-kvd cliff-spur-l2 cliff-spur-l2-debug \
        cliff-short cliff-long-64k cliff-long-128k

dist-build:
	"$(DIST)" build-split
	@[ "$(AIC_BUILD_EXPORTERS)" = "0" ] || "$(DIST)" build-exporters \
	    || echo "WARNING: fabric-exporter build failed (optional; main image is built). Retry on a node with Docker Hub access, or set AIC_BUILD_EXPORTERS=0."

dist-build-fast:
	@$(MAKE) --no-print-directory dist-build \
	    AIC_ROCM_ARCH='$(AIC_FAST_ARCH)' AIC_BUILD_EXPORTERS=0 AIC_UCX_FAST=1

dist-build-emulate:
	"$(DIST)" build-emulate

dist-build-base:
	"$(DIST)" build-base

dist-build-vllm:
	"$(DIST)" build-vllm

dist-build-lmcache:
	"$(DIST)" build-lmcache

dist-build-parallel:
	@$(MAKE) --no-print-directory dist-build-base
	@$(MAKE) --no-print-directory -j2 dist-build-vllm dist-build-lmcache

dist-build-exporters:
	"$(DIST)" build-exporters

dist-build-monitoring:
	@set -e; \
	for img in \
	    "prom/prometheus:v3.14.0" \
	    "rocm/device-metrics-exporter:v1.5.2" \
	; do \
	    tag="$$(printf '%s' "$$img" | tr '/:' '--').tar.zst"; \
	    dest="$(AIC_IMAGE_DIR)/$$tag"; \
	    if [ ! -f "$$dest" ]; then \
	        echo "Pulling $$img and saving to $$dest ..."; \
	        docker pull "$$img"; \
	        docker save "$$img" | zstd -T0 -q > "$$dest"; \
	        echo "  saved $$(du -h "$$dest" | cut -f1) -> $$dest"; \
	    else \
	        echo "  $$dest already present (use AIC_FORCE_LOAD=1 to refresh)"; \
	    fi; \
	done

dist-push:
	"$(DIST)" push

smoke-test:
	"$(DIST)" test

smoke-test-fast:
	@$(MAKE) --no-print-directory smoke-test \
	    AIC_ROCM_ARCH='$(AIC_FAST_ARCH)' AIC_SMOKE_EXPORTERS=0

tiny-test:
	"$(DIST)" tiny-test

tiny-test-fast:
	@$(MAKE) --no-print-directory tiny-test \
	    AIC_ROCM_ARCH='$(AIC_FAST_ARCH)'

profile-capture:
	"$(DIST)" profile-capture

emulate-validate:
	"$(DIST)" emulate-validate

emulate-test:
	"$(DIST)" emulate-test

emulate-mp-test:
	"$(DIST)" emulate-mp-test

accuracy-test: check-hf-token
	"$(DIST)" accuracy-test

accuracy-test-fast:
	@$(MAKE) --no-print-directory accuracy-test \
	    AIC_ROCM_ARCH='$(AIC_FAST_ARCH)'

accuracy-test-very-fast:
	@$(MAKE) --no-print-directory accuracy-test \
	    AIC_ROCM_ARCH='$(AIC_FAST_ARCH)' \
	    AIC_ACCURACY_LIMIT=100 \
	    AIC_ACCURACY_TIME=01:00:00

install-ci-scripts:
	@set -e; \
	src="$(AIC_CI_SCRIPT_DIR)"; dst="$(AIC_CI_LIB_DIR)"; \
	ls "$$src"/*.sh >/dev/null 2>&1 || { echo "ERROR: no runner scripts under $$src" >&2; exit 1; }; \
	if [ -w "$$(dirname "$$dst")" ] || [ -w "$$dst" ]; then SUDO=; else SUDO="sudo"; \
		echo "$$dst not writable; using sudo"; fi; \
	$$SUDO install -d -m 0755 "$$dst"; \
	for f in "$$src"/*.sh; do \
		$$SUDO install -m 0755 "$$f" "$$dst/$$(basename "$$f")"; \
		echo "installed $$(basename "$$f") -> $$dst/"; \
	done; \
	echo "CI runner scripts deployed to $$dst"

prometheus-dump:
	"$(DIST)" prometheus-dump

# ---- Cliff submit targets ----------------------------------------------------

cliff-kvbench-submit:
	@cd "$(CURDIR)" && jobid=$$($(call _KVBENCH_SUBMIT,\
	    $(if $(AIC_KVBENCH_TIME),--time=$(AIC_KVBENCH_TIME),))) && \
	    echo "submitted kvbench cliff job $$jobid" && \
	    echo "log: $(CURDIR)/logs/$$jobid/kvbench-cliff.out"

cliff-submit:
	@cd "$(CURDIR)" && jobid=$$($(call _CLIFF_SUBMIT,\
	    $(if $(AIC_CLIFF_TIME),--time=$(AIC_CLIFF_TIME),))) && \
	    echo "submitted cliff job $$jobid" && \
	    echo "log: $(CURDIR)/logs/$$jobid/cliff.out"

cliff-kvd:
	@cd "$(CURDIR)" && jobid=$$( \
	    BENCH_PREFIX_MODE=shared \
	    BENCH_CONCUR="$${BENCH_CONCUR:-1,8,32,64,128,250}" \
	    BENCH_ITERS="$${BENCH_ITERS:-2}" \
	    $(call _CLIFF_SUBMIT,--job-name=aic-cliff-kvd \
	    --time=$(if $(AIC_CLIFF_TIME),$(AIC_CLIFF_TIME),02:00:00))) && \
	    echo "submitted cliff-kvd job $$jobid (shared prefix, c=1,8,32,64,128,250, 2 iters)" && \
	    echo "log: $(CURDIR)/logs/$$jobid/cliff.out"

cliff-spur-l2:
	@cd "$(CURDIR)" && jobid=$$( \
	    BENCH_PREFIX_MODE=per_client \
	    BENCH_CONCUR="$${BENCH_CONCUR:-32}" \
	    BENCH_ITERS="$${BENCH_ITERS:-2}" \
	    VLLM_MODEL="$${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}" \
	    VLM_GPU_MEMORY_UTILIZATION="$(call _user_or,VLM_GPU_MEMORY_UTILIZATION,0.40)" \
	    AIC_LOCAL_CPU=true \
	    LMCACHE_MAX_LOCAL_CPU_SIZE="$(call _user_or,LMCACHE_MAX_LOCAL_CPU_SIZE,8)" \
	    AIC_CLIFF_ARMS="$${AIC_CLIFF_ARMS:-nvme}" \
	    $(call _CLIFF_SUBMIT,--job-name=aic-cliff-spur-l2 \
	    --time=$(if $(AIC_CLIFF_TIME),$(AIC_CLIFF_TIME),03:00:00))) && \
	    echo "submitted cliff-spur-l2 job $$jobid" && \
	    echo "  util=0.40, DRAM L1=8GB, POSIX NVMe L2, per_client, Qwen2.5-3B, c=32, 2 iters, nvme only" && \
	    echo "log: $(CURDIR)/logs/$$jobid/cliff.out"

cliff-spur-l2-debug:
	@cd "$(CURDIR)" && jobid=$$( \
	    BENCH_PREFIX_MODE=per_client \
	    BENCH_CONCUR="$${BENCH_CONCUR:-1}" \
	    BENCH_ITERS="$${BENCH_ITERS:-1}" \
	    VLLM_MODEL="$${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}" \
	    VLM_GPU_MEMORY_UTILIZATION="$(call _user_or,VLM_GPU_MEMORY_UTILIZATION,0.40)" \
	    AIC_LOCAL_CPU=true \
	    LMCACHE_MAX_LOCAL_CPU_SIZE="$(call _user_or,LMCACHE_MAX_LOCAL_CPU_SIZE,0.1)" \
	    AIC_CLIFF_ARMS="$${AIC_CLIFF_ARMS:-nvme}" \
	    VLLM_LOGGING_LEVEL=DEBUG \
	    $(call _CLIFF_SUBMIT,--job-name=aic-cliff-l2-debug \
	    --time=$(if $(AIC_CLIFF_TIME),$(AIC_CLIFF_TIME),01:00:00))) && \
	    echo "submitted cliff-spur-l2-debug job $$jobid" && \
	    echo "  util=0.40, DRAM L1=0.1GB (forces L2 hits at c=1), per_client, c=1, 1 iter, DEBUG logging" && \
	    echo "log: $(CURDIR)/logs/$$jobid/cliff.out" && \
	    echo "  check container-aic-vllm.log for 'vLLM hit is' to confirm ext_hit"

cliff-short:
	@cd "$(CURDIR)" && jobid=$$(BENCH_CONCUR="$${BENCH_CONCUR:-1}" BENCH_ITERS="$${BENCH_ITERS:-1}" \
	    $(call _CLIFF_SUBMIT,--job-name=aic-cliff-short)) && \
	    echo "submitted cliff-short job $$jobid (BENCH_CONCUR=$${BENCH_CONCUR:-1} BENCH_ITERS=$${BENCH_ITERS:-1})" && \
	    echo "log: $(CURDIR)/logs/$$jobid/cliff.out"

cliff-long-64k:
	@cd "$(CURDIR)" && jobid=$$( \
	    VLLM_MODEL="$${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}" \
	    VLM_GPU_MEMORY_UTILIZATION="$(call _user_or,VLM_GPU_MEMORY_UTILIZATION,0.12)" \
	    VLM_YARN_FACTOR="$${VLM_YARN_FACTOR:-2.0}" VLM_MAX_MODEL_LEN="$(call _user_or,VLM_MAX_MODEL_LEN,65536)" \
	    BENCH_ISL="$${BENCH_ISL:-64000}" BENCH_SHARED_TOK="$${BENCH_SHARED_TOK:-60000}" \
	    BENCH_PREFIX_MODE="$${BENCH_PREFIX_MODE:-per_client}" BENCH_ITERS="$${BENCH_ITERS:-2}" \
	    AIC_LOCAL_CPU="$${AIC_LOCAL_CPU:-true}" LMCACHE_MAX_LOCAL_CPU_SIZE="$${LMCACHE_MAX_LOCAL_CPU_SIZE:-64}" \
	    LMCACHE_NVME_POOL="$(call _user_or,LMCACHE_NVME_POOL,262144)" AIC_NIXL_BUFFER_SIZE="$${AIC_NIXL_BUFFER_SIZE:-8589934592}" \
	    LMCACHE_L1_SIZE_GB="$(call _user_or,LMCACHE_L1_SIZE_GB,320)" \
	    $(call _CLIFF_SUBMIT,--job-name=aic-cliff-long64k \
	    --time=$(if $(AIC_CLIFF_TIME),$(AIC_CLIFF_TIME),16:00:00))) && \
	    echo "submitted cliff-long-64k job $$jobid (ISL=64000, YaRN x2 -> 65536, DRAM L1=64G, NVMe pool=262144, all 3 arms)" && \
	    echo "log: $(CURDIR)/logs/$$jobid/cliff.out"

cliff-long-128k:
	@cd "$(CURDIR)" && jobid=$$( \
	    VLLM_MODEL="$${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}" \
	    VLM_GPU_MEMORY_UTILIZATION="$(call _user_or,VLM_GPU_MEMORY_UTILIZATION,0.12)" \
	    VLM_YARN_FACTOR="$${VLM_YARN_FACTOR:-4.0}" VLM_MAX_MODEL_LEN="$(call _user_or,VLM_MAX_MODEL_LEN,131072)" \
	    BENCH_ISL="$${BENCH_ISL:-128000}" BENCH_SHARED_TOK="$${BENCH_SHARED_TOK:-126000}" \
	    BENCH_PREFIX_MODE="$${BENCH_PREFIX_MODE:-per_client}" BENCH_ITERS="$${BENCH_ITERS:-1}" \
	    AIC_LOCAL_CPU="$${AIC_LOCAL_CPU:-true}" LMCACHE_MAX_LOCAL_CPU_SIZE="$${LMCACHE_MAX_LOCAL_CPU_SIZE:-64}" \
	    LMCACHE_NVME_POOL="$(call _user_or,LMCACHE_NVME_POOL,524288)" AIC_NIXL_BUFFER_SIZE="$${AIC_NIXL_BUFFER_SIZE:-8589934592}" \
	    LMCACHE_L1_SIZE_GB="$(call _user_or,LMCACHE_L1_SIZE_GB,640)" \
	    $(call _CLIFF_SUBMIT,--job-name=aic-cliff-long128k \
	    --time=$(if $(AIC_CLIFF_TIME),$(AIC_CLIFF_TIME),24:00:00))) && \
	    echo "submitted cliff-long-128k job $$jobid (ISL=128000, YaRN x4 -> 131072, DRAM L1=64G, NVMe pool=524288, all 3 arms)" && \
	    echo "log: $(CURDIR)/logs/$$jobid/cliff.out"
