# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Distributed / Slurm targets: build, test, emulate, profile, accuracy, cliff.

# The cliff runs through `run-build-distribute.sh cliff`, which submits
# .slurm/run-cliff.sbatch via _sbatch_run (partition, GPU request, node excludes,
# logs/<job-id>/cliff.{out,exit}) and BLOCKS until the job ends, so the recipe's
# exit status is the job's.  That script owns the job-sizing defaults
# (AIC_CLIFF_GPUS, AIC_CLIFF_CONSTRAINT and its AIC_CLIFF_GFX shorthand, ...);
# a value the user set reaches it through the environment.
#
# The cliff targets below are presets over run-cliff.sbatch's own defaults, and
# a preset yields to a value the user set (command line, environment or
# .env.<cluster>).  Knobs make holds no default for are written
# `$(or $(VAR),preset)`.  BENCH_CONCUR and BENCH_ITERS cannot be: make has
# non-empty defaults for the local `make cliff` (never exported, so the job does
# not see them), and `$(or)` would split a comma list.  Those two keep the shell
# form `$${VAR:-preset}`, which sees exactly the user-set values.
_CLIFF_RUN = $(_CLIFF_STRIP) "$(DIST)" cliff

ifeq ($(AIC_SPUR_CLUSTER),1)
# GPUs reserved for a SPUR kvbench submit.  The KVBench cliff drives no GPU,
# but SPUR refuses any submission that carries no GPU request at all.
_KVBENCH_SBATCH_ARGS := --partition=$(AIC_SPUR_PARTITION) --constraint= --gpus=1 \
    $(if $(AIC_KVBENCH_NODE),--nodelist=$(AIC_KVBENCH_NODE),)
_KVBENCH_SUBMIT := $(_CLIFF_STRIP) sbatch \
    $(_KVBENCH_SBATCH_ARGS) $(1) .slurm/run-kvbench-cliff.sbatch 2>&1 | \
    tee /dev/stderr | grep -oE '[0-9]+$$' | tail -1
else
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
# Thin wrappers over run-build-distribute.sh.

.PHONY: dist-build dist-build-fast dist-build-emulate \
        dist-wheels \
        dist-build-exporters dist-build-monitoring dist-push \
        smoke-test smoke-test-fast tiny-test tiny-test-fast \
        emulate-test emulate-mp-test emulate-validate profile-capture \
        accuracy-test accuracy-test-fast accuracy-test-very-fast \
        install-ci-scripts prometheus-dump \
        cliff-kvbench-submit cliff-submit cliff-kvd cliff-spur-l2 \
        cliff-short cliff-short-fast

dist-build:
	$(DIST_RUN) bake

dist-build-fast:
	@$(MAKE) --no-print-directory dist-build \
	    AIC_ROCM_ARCH='$(AIC_FAST_ARCH)' AIC_UCX_FAST=1

# Export the nightly-release pip wheels on a Slurm node.  `make wheels` runs the
# same bake group locally; this one exists because the PyTorch + vLLM source
# builds behind it do not fit in a GitHub-hosted runner's 6h job limit.
dist-wheels:
	$(DIST_RUN) wheels

dist-build-emulate:
	$(DIST_RUN) build-emulate

dist-build-exporters:
	$(DIST_RUN) build-exporters

dist-build-monitoring:
	@set -e; \
	_default="$(CURDIR)/docker/scripts/aic-compose-default.sh"; \
	_prom="$$("$$_default" AIC_PROM_IMAGE)"; \
	_amdgpu="$$("$$_default" AIC_AMDGPU_EXPORTER_IMAGE)"; \
	for img in "$$_prom" "$$_amdgpu"; do \
	    tag="$$(printf '%s' "$$img" | tr '/:' '--').tar.zst"; \
	    dest="$(AIC_IMAGE_DIR)/$$tag"; \
	    if [ ! -f "$$dest" ] || [ "$${AIC_FORCE_LOAD:-0}" = 1 ]; then \
	        echo "Pulling $$img and saving to $$dest ..."; \
	        docker pull "$$img"; \
	        docker save "$$img" | zstd -T0 -q > "$$dest"; \
	        echo "  saved $$(du -h "$$dest" | cut -f1) -> $$dest"; \
	    else \
	        echo "  $$dest already present (use AIC_FORCE_LOAD=1 to refresh)"; \
	    fi; \
	done

dist-push:
	$(DIST_RUN) push

smoke-test:
	$(DIST_RUN) test

smoke-test-fast:
	@$(MAKE) --no-print-directory smoke-test \
	    AIC_ROCM_ARCH='$(AIC_FAST_ARCH)' AIC_SMOKE_EXPORTERS=0

tiny-test:
	$(DIST_RUN) tiny-test

tiny-test-fast:
	@$(MAKE) --no-print-directory tiny-test \
	    AIC_ROCM_ARCH='$(AIC_FAST_ARCH)'

profile-capture:
	$(DIST_RUN) profile-capture

emulate-validate:
	$(DIST_RUN) emulate-validate

emulate-test:
	$(DIST_RUN) emulate-test

emulate-mp-test:
	$(DIST_RUN) emulate-mp-test

accuracy-test: check-hf-token
	$(DIST_RUN) accuracy-test

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
	rev=$$(git -C "$$src" rev-parse HEAD) || rev=unknown; \
	git -C "$$src" diff --quiet HEAD -- . || rev="$$rev-dirty"; \
	echo "$$rev" | $$SUDO tee "$$dst/VERSION" >/dev/null; \
	echo "recorded $$rev -> $$dst/VERSION"; \
	echo "CI runner scripts deployed to $$dst"

prometheus-dump:
	$(DIST_RUN) prometheus-dump

# ---- Cliff submit targets ----------------------------------------------------

cliff-kvbench-submit:
	@cd "$(CURDIR)" && jobid=$$($(call _KVBENCH_SUBMIT,\
	    $(if $(AIC_KVBENCH_TIME),--time=$(AIC_KVBENCH_TIME),))) && \
	    echo "submitted kvbench cliff job $$jobid" && \
	    echo "log: $(CURDIR)/logs/$$jobid/kvbench-cliff.out"

# The nightly sweep.  Sized so the VRAM cliff lands inside c=1..250 for the default
# gpt-oss-120b on a 288 GB MI355X: util 0.40 leaves vLLM ~1.6M KV tokens, so
# vram_only falls off between c=80 and c=128 (at 0.90 it held 8.3M and never fell).
# Each 20K-token request stages ~472 MiB through LMCache's L1, also for L2 hits, so
# the kvd tiers must cover c=250 or they miss with it: 160 GB of DRAM in front of
# the nvme arm's NVMe L2, and a 256 GB slab for gds (prefixes plus the suffixes of
# each wave, with no L2 behind it).  Jobs 279409, 279482 and 279683.
# 4h: the full sweep ran 2h23m on a whole node (job 279891, image pull included),
# mostly vram_only recomputing every prefix past its cliff.
cliff-submit:
	@cd "$(CURDIR)" && \
	    AIC_CLIFF_TIME="$(or $(AIC_CLIFF_TIME),04:00:00)" \
	    VLM_GPU_MEMORY_UTILIZATION="$(or $(VLM_GPU_MEMORY_UTILIZATION),0.40)" \
	    AIC_LOCAL_CPU="$(or $(AIC_LOCAL_CPU),true)" \
	    LMCACHE_MAX_LOCAL_CPU_SIZE="$(or $(LMCACHE_MAX_LOCAL_CPU_SIZE),160)" \
	    LMCACHE_L1_SIZE_GB="$(or $(LMCACHE_L1_SIZE_GB),256)" \
	    $(_CLIFF_RUN)

cliff-kvd:
	@echo "cliff-kvd: shared prefix, c=1,8,32,64,128,250, 2 iters"
	@cd "$(CURDIR)" && \
	    BENCH_PREFIX_MODE=shared \
	    BENCH_CONCUR="$${BENCH_CONCUR:-1,8,32,64,128,250}" \
	    BENCH_ITERS="$${BENCH_ITERS:-2}" \
	    AIC_CLIFF_JOB_NAME=aic-cliff-kvd \
	    AIC_CLIFF_TIME="$(or $(AIC_CLIFF_TIME),02:00:00)" \
	    $(_CLIFF_RUN)

cliff-spur-l2:
	@echo "cliff-spur-l2: util=0.40, DRAM L1=8GB, POSIX NVMe L2, per_client, Qwen2.5-3B, c=32, 2 iters, nvme only"
	@cd "$(CURDIR)" && \
	    BENCH_PREFIX_MODE=per_client \
	    BENCH_CONCUR="$${BENCH_CONCUR:-32}" \
	    BENCH_ITERS="$${BENCH_ITERS:-2}" \
	    VLLM_MODEL="$(or $(VLLM_MODEL),Qwen/Qwen2.5-3B-Instruct)" \
	    VLM_GPU_MEMORY_UTILIZATION="$(or $(VLM_GPU_MEMORY_UTILIZATION),0.40)" \
	    AIC_LOCAL_CPU=true \
	    LMCACHE_MAX_LOCAL_CPU_SIZE="$(or $(LMCACHE_MAX_LOCAL_CPU_SIZE),8)" \
	    AIC_CLIFF_ARMS="$(or $(AIC_CLIFF_ARMS),nvme)" \
	    AIC_CLIFF_JOB_NAME=aic-cliff-spur-l2 \
	    AIC_CLIFF_TIME="$(or $(AIC_CLIFF_TIME),03:00:00)" \
	    $(_CLIFF_RUN)

cliff-short:
	@echo "cliff-short: BENCH_CONCUR=$${BENCH_CONCUR:-1} BENCH_ITERS=$${BENCH_ITERS:-1}"
	@cd "$(CURDIR)" && \
	    BENCH_CONCUR="$${BENCH_CONCUR:-1}" BENCH_ITERS="$${BENCH_ITERS:-1}" \
	    AIC_CLIFF_JOB_NAME=aic-cliff-short \
	    $(_CLIFF_RUN)

# The cliff loads the exact <name>-<tag>-<arch> tarballs, so after dist-build-fast
# (single arch) it must resolve the same arch tag; see accuracy-test-fast.
cliff-short-fast:
	@$(MAKE) --no-print-directory cliff-short \
	    AIC_ROCM_ARCH='$(AIC_FAST_ARCH)'

