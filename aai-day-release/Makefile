# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# This tree is self-contained: the build context and all sources live here, so
# "repo root" is this directory (no dependency on any parent checkout).
REPO_ROOT := $(CURDIR)
include $(CURDIR)/scripts/nixl/defaults.mk

# ai-dynamo/nixl upstream main (2026-07-10); AIS_MT added via patches/nixl/.
NIXL_GIT_URL := https://github.com/ai-dynamo/nixl.git
NIXL_SHA     := 644facf0eb3de14ec63c1d2831238f63cd03c0e0

IMAGE_NAME ?= rocm-aic-aai-day

# ---- GPU -------------------------------------------------------------------
GPU ?= 0

# ---- Host storage paths ----------------------------------------------------
NVME_DATA     ?= /mnt/lmcache-nvme
NFS_DATA      ?= /mnt/lmcache-nfs
GDS_SLAB_DATA ?=

# ---- Log / HuggingFace -----------------------------------------------------
LOG           ?= $(CURDIR)/logs
HF_HOME       ?= $(HOME)/.cache/huggingface
HF_TOKEN_FILE ?=

# ---- LMCache server --------------------------------------------------------
LMCACHE_PORT           ?= 6555
LMCACHE_L1_SIZE_GB     ?= 20
LMCACHE_NVME_POOL      ?= 4096
LMCACHE_NVME_SLOT_SIZE ?= 268435456
LMCACHE_NFS_POOL       ?= 1024

# ---- vLLM knobs ------------------------------------------------------------
VLLM_MODEL                  ?=
TENSOR_PARALLEL_SIZE        ?= 1
VLM_GPU_MEMORY_UTILIZATION  ?=
VLM_MAX_MODEL_LEN           ?=
VLM_MAX_NUM_BATCHED_TOKENS  ?=

# ---- Benchmark knobs -------------------------------------------------------
BENCH_ARM         ?= kvd_v2
BENCH_ISL         ?= 20000
BENCH_SHARED_TOK  ?= 18000
BENCH_CONCUR      ?= 1,2,4,8,16,32,48,64,80,100,128,160,200,250
BENCH_ITERS       ?= 3
BENCH_ENDPOINT    ?= http://localhost:8000
BENCH_MODEL       ?= $(VLLM_MODEL)
# Non-Slurm runs have no job id, so they mirror the sbatch "manual" fallback and
# land under logs/manual/ -- keeping the tree root free of results/ and plots/.
BENCH_LOGDIR      := logs/manual
BENCH_OUT         := $(BENCH_LOGDIR)/results/cliff-$(BENCH_ARM)-$(shell date +%Y%m%d-%H%M%S).csv

# ---- ROCm arch (auto-detected if not set) ----------------------------------
_ROCM_ARCH_DETECTED := $(shell rocm_agent_enumerator 2>/dev/null | grep -E '^gfx' | head -1)
ROCM_ARCH := $(if $(strip $(ROCM_ARCH)),$(strip $(ROCM_ARCH)),$(_ROCM_ARCH_DETECTED))

export ROCM_ARCH GPU GDS_SLAB_DATA LOG HF_HOME IMAGE_NAME
export LMCACHE_PORT LMCACHE_L1_SIZE_GB LMCACHE_NVME_POOL LMCACHE_NVME_SLOT_SIZE LMCACHE_NFS_POOL
export NVME_DATA NFS_DATA
export VLLM_MODEL TENSOR_PARALLEL_SIZE
export VLM_GPU_MEMORY_UTILIZATION VLM_MAX_MODEL_LEN VLM_MAX_NUM_BATCHED_TOKENS
export NIXL_GIT_URL NIXL_SHA

comma := ,
_COMPOSE_BIN := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")
COMPOSE      := DOCKER_BUILDKIT=1 $(_COMPOSE_BIN) -f "$(CURDIR)/docker-compose.yml"

# ---- Metrics capture (Prometheus sidecar) ----------------------------------
# AAI_METRICS_DIR: Prometheus TSDB dir (bind-mount an NFS path here to explore
# a run afterward).  AAI_EXPORTERS=1 also launches the containerized node + AMD
# GPU exporters (for nodes without the host-installed exporter services).
AAI_METRICS_DIR ?= $(CURDIR)/logs/prometheus
AAI_EXPORTERS   ?= 0
MON_COMPOSE     := $(_COMPOSE_BIN) -f "$(CURDIR)/monitoring/docker-compose.monitoring.yml"
_MON_PROFILE    := $(if $(filter 1,$(AAI_EXPORTERS)),--profile exporters,)
export AAI_METRICS_DIR

# ---- Fabric exporters (nvme_exporter / rdma_exporter) ----------------------
# No published upstream images; we build them from monitoring/*/Dockerfile so the
# `exporters-fabric` compose profile and the .slurm docker-run fallback (nodes
# without the compose plugin) can containerize them.  Versions match the batesste
# host services for Grafana parity; override to bump.
NVME_EXPORTER_IMAGE   ?= aai-day-nvme-exporter:local
RDMA_EXPORTER_IMAGE   ?= aai-day-rdma-exporter:local
NVME_EXPORTER_VERSION ?= 3.0.0
RDMA_EXPORTER_VERSION ?= 0.3.0

PYTHON := $(if $(wildcard $(REPO_ROOT)/.venv/bin/python3),$(REPO_ROOT)/.venv/bin/python3,python3)

# ---- Distribute / cliff (Slurm) --------------------------------------------
# The dist-* / cliff-* targets shell out to the .slurm scripts; the AAI_* knobs
# they read pass straight through the environment (e.g. make dist-build
# AAI_ROCM_ARCH=gfx942, make cliff-submit AAI_CLIFF_NODE=ctr-s95-mi300x-3
# AAI_CLIFF_ARMS=nvme).  AAI_CACHE_DIR is the shared BuildKit file cache on
# /scratch so a failed build resumes from the last good layer on any node
# (set AAI_CACHE_DIR= to disable); AAI_BUILD_EXPORTERS=0 skips the fabric images.
DIST := $(CURDIR)/.slurm/run-build-distribute.sh
export AAI_CACHE_DIR ?= /scratch/$(USER)/images/buildcache

.PHONY: help build up up-batch up-gds-l1 up-gds-l1-batch down logs logs-lmcache logs-vllm \
        ps shell-lmcache shell-vllm restart-vllm restart-lmcache cliff plot venv \
        monitoring-up monitoring-down monitoring-logs monitoring-build-exporters \
        dist-build dist-push smoke-test cliff-submit cliff-short \
        _check_hf_token _prep_dirs _check_gds_slab

.DEFAULT_GOAL := help

help:
	@echo "rocm-aic aai-day-release — AMD Infinity Context inference stack + benchmarks"
	@echo ""
	@echo "Stack targets:"
	@echo "  make build             Build the shared image ($(IMAGE_NAME))"
	@echo "  make up                Start lmcache + vllm (foreground, DRAM L1 + AIS_MT/NFS L2)"
	@echo "  make up-batch          Start lmcache + vllm (background)"
	@echo "  make up-gds-l1         Start with hipFile GDS NVMe slab as L1 (foreground)"
	@echo "  make up-gds-l1-batch   Start with hipFile GDS NVMe slab as L1 (background)"
	@echo "  make down              Stop and remove both containers"
	@echo "  make logs              Follow logs from both containers"
	@echo "  make logs-lmcache      lmcache container logs only"
	@echo "  make logs-vllm         vllm container logs only"
	@echo "  make ps                Container status"
	@echo "  make shell-lmcache     Exec bash into lmcache container"
	@echo "  make shell-vllm        Exec bash into vllm container"
	@echo "  make restart-vllm      Restart vllm only (lmcache + warm KV preserved)"
	@echo "  make restart-lmcache   Restart lmcache only"
	@echo ""
	@echo "Benchmark targets:"
	@echo "  make venv              Create/update repo-root .venv with bench+plot deps"
	@echo "  make cliff             Run KV-cache cliff benchmark, write CSV to $(BENCH_LOGDIR)/results/"
	@echo "  make plot              Generate cliff PNG charts from $(BENCH_LOGDIR)/results/ CSVs"
	@echo ""
	@echo "Distribute / cliff targets (Slurm; wrap .slurm/ scripts + sbatch):"
	@echo "  make dist-build        Build image (+ fabric exporters) on a Slurm build node, save tarballs"
	@echo "  make dist-push         Tag + push the built image (needs AAI_PUSH_REF)"
	@echo "  make smoke-test        Load + smoke-test the image on a GPU+NVMe node"
	@echo "  make cliff-submit      sbatch the full 3-arm cliff sweep -> logs/<job-id>/"
	@echo "  make cliff-short       sbatch a 1-point cliff (concur=1, 1 iter) to smoke-test the flow"
	@echo "    Chain like the old run-this.sh:  make dist-build dist-push smoke-test"
	@echo "    Pin a node: AAI_CLIFF_NODE=<node>   Narrow arms: AAI_CLIFF_ARMS=nvme (vram,nvme,gds)"
	@echo "    Override sweep/model via env: BENCH_CONCUR=1,8,64 VLLM_MODEL=... make cliff-submit"
	@echo "    AAI_CACHE_DIR=$(AAI_CACHE_DIR)  (shared BuildKit cache; set empty to disable)"
	@echo ""
	@echo "Metrics targets (Prometheus sidecar; scrapes vLLM/LMCache/exporters):"
	@echo "  make monitoring-up     Start Prometheus, TSDB -> AAI_METRICS_DIR"
	@echo "  make monitoring-down   Stop the metrics sidecar (TSDB retained)"
	@echo "  make monitoring-logs   Follow Prometheus logs"
	@echo "  make monitoring-build-exporters  Build nvme_exporter + rdma_exporter images"
	@echo "    AAI_METRICS_DIR=$(AAI_METRICS_DIR)"
	@echo "    AAI_EXPORTERS=$(AAI_EXPORTERS)  (1 = also launch node + AMD GPU exporters)"
	@echo ""
	@echo "Required env:"
	@echo "  HF_TOKEN       HuggingFace access token"
	@echo "  ROCM_ARCH      GPU arch (detected: $(ROCM_ARCH))"
	@echo ""
	@echo "Optional build env:"
	@echo "  TLS_CERT       Path to corporate CA cert (e.g. Zscaler); passed as a"
	@echo "                 BuildKit secret — never baked into the image."
	@echo "                 Example: make build TLS_CERT=/etc/ssl/certs/zscaler-ca.crt"
	@echo ""
	@echo "Key storage vars (current):"
	@echo "  NVME_DATA=$(NVME_DATA)  NFS_DATA=$(NFS_DATA)  GDS_SLAB_DATA=$(GDS_SLAB_DATA)"
	@echo ""
	@echo "Key LMCache vars (current):"
	@echo "  LMCACHE_PORT=$(LMCACHE_PORT)  LMCACHE_L1_SIZE_GB=$(LMCACHE_L1_SIZE_GB) GiB"
	@echo "  LMCACHE_NVME_POOL=$(LMCACHE_NVME_POOL)  LMCACHE_NFS_POOL=$(LMCACHE_NFS_POOL)"
	@echo ""
	@echo "Examples:"
	@echo "  make build"
	@echo "  make up HF_TOKEN=hf_... NVME_DATA=/mnt/nvme NFS_DATA=/mnt/nfs"
	@echo "  make up-gds-l1 GDS_SLAB_DATA=/mnt/nvme HF_TOKEN=hf_..."
	@echo "  make cliff BENCH_ARM=vram_only BENCH_ENDPOINT=http://localhost:8000"
	@echo "  make plot"
	@echo ""

_check_hf_token:
	@if [ -z "$$HF_TOKEN" ] && [ -n "$(HF_TOKEN_FILE)" ] && [ -r "$(HF_TOKEN_FILE)" ]; then \
		export HF_TOKEN="$$(tr -d '\r\n' < "$(HF_TOKEN_FILE)")"; \
	fi; \
	if [ -z "$$HF_TOKEN" ]; then \
		echo "ERROR: set HF_TOKEN or HF_TOKEN_FILE" >&2; exit 1; \
	fi

_check_gds_slab:
	@if [ -z "$(GDS_SLAB_DATA)" ]; then \
		echo "ERROR: GDS_SLAB_DATA must be set for GDS L1 mode" >&2; exit 1; \
	fi

_prep_dirs:
	@mkdir -p "$(NVME_DATA)" "$(NFS_DATA)" \
		"$(LOG)/lmcache" "$(LOG)/vllm" \
		"$(HF_HOME)/hub" "$(HF_HOME)/datasets" "$(HF_HOME)/vllm" \
		"$(HF_HOME)/vllm_config" "$(HF_HOME)/torch" "$(HF_HOME)/torch_inductor" \
		"$(BENCH_LOGDIR)/results" "$(BENCH_LOGDIR)/plots"

build:
	@test -n "$(ROCM_ARCH)" || { \
		echo "ERROR: ROCM_ARCH empty (install ROCm or set ROCM_ARCH=gfxNNNN)" >&2; exit 1; }
	cd "$(REPO_ROOT)" && $(COMPOSE) build \
		$(if $(TLS_CERT),--secret id=tls_cert$(comma)src=$(TLS_CERT),)

up: _check_hf_token _prep_dirs
	$(COMPOSE) up

up-batch: _check_hf_token _prep_dirs
	$(COMPOSE) up -d
	@echo "Started. Use 'make logs' to follow or 'make down' to stop."

up-gds-l1: _check_hf_token _check_gds_slab _prep_dirs
	GDS_MODE=1 $(COMPOSE) up

up-gds-l1-batch: _check_hf_token _check_gds_slab _prep_dirs
	GDS_MODE=1 $(COMPOSE) up -d
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
	docker exec -it aai-day-lmcache bash -l

shell-vllm:
	docker exec -it aai-day-vllm-gpu$(GPU) bash -l

restart-vllm:
	$(COMPOSE) restart vllm

restart-lmcache:
	$(COMPOSE) restart lmcache

venv:
	@if [ ! -d "$(REPO_ROOT)/.venv" ]; then \
		python3 -m venv "$(REPO_ROOT)/.venv"; \
	fi
	"$(REPO_ROOT)/.venv/bin/pip" install --upgrade pip
	"$(REPO_ROOT)/.venv/bin/pip" install -e "$(CURDIR)[dev]"
	@echo "venv ready at $(REPO_ROOT)/.venv"
	@echo "Activate: source $(REPO_ROOT)/.venv/bin/activate"

cliff: _prep_dirs
	@test -n "$(BENCH_MODEL)" || { \
		echo "ERROR: set BENCH_MODEL or VLLM_MODEL to the served model name" >&2; exit 1; }
	$(PYTHON) "$(CURDIR)/benchmarks/run_cliff.py" \
		--endpoint "$(BENCH_ENDPOINT)" \
		--model "$(BENCH_MODEL)" \
		--arm "$(BENCH_ARM)" \
		--isl "$(BENCH_ISL)" \
		--shared-prefix-tokens "$(BENCH_SHARED_TOK)" \
		--concurrencies "$(BENCH_CONCUR)" \
		--iters "$(BENCH_ITERS)" \
		--warmup-iters 1 \
		--out "$(BENCH_OUT)"
	@echo "Results written to $(BENCH_OUT)"

plot: _prep_dirs
	$(PYTHON) "$(CURDIR)/benchmarks/plot_cliff.py" \
		--input "$(BENCH_LOGDIR)/results/" \
		--output-dir "$(BENCH_LOGDIR)/plots/"
	@echo "Charts written to $(BENCH_LOGDIR)/plots/"

monitoring-up:
	@mkdir -p "$(AAI_METRICS_DIR)"
	PROM_UID="$$(id -u)" PROM_GID="$$(id -g)" \
		$(MON_COMPOSE) $(_MON_PROFILE) up -d
	@echo "Prometheus up on :9090  (TSDB -> $(AAI_METRICS_DIR))"

monitoring-down:
	$(MON_COMPOSE) $(_MON_PROFILE) down

monitoring-logs:
	$(MON_COMPOSE) logs -f prometheus

# Build the two fabric-exporter images (plain `docker build`, so it works on
# nodes without the compose plugin).  Tag/version overridable via the vars above.
monitoring-build-exporters:
	DOCKER_BUILDKIT=1 docker build \
		--build-arg NVME_EXPORTER_VERSION=$(NVME_EXPORTER_VERSION) \
		-t "$(NVME_EXPORTER_IMAGE)" "$(CURDIR)/monitoring/nvme-exporter"
	DOCKER_BUILDKIT=1 docker build \
		--build-arg RDMA_EXPORTER_VERSION=$(RDMA_EXPORTER_VERSION) \
		-t "$(RDMA_EXPORTER_IMAGE)" "$(CURDIR)/monitoring/rdma-exporter"
	@echo "Built $(NVME_EXPORTER_IMAGE) and $(RDMA_EXPORTER_IMAGE)."
	@echo "Run them via:  AAI_EXPORTERS=1 with --profile exporters-fabric, or set"
	@echo "AAI_NVME_EXPORTER_IMAGE / AAI_RDMA_EXPORTER_IMAGE for the .slurm docker-run path."

# ---- Distribute / cliff (Slurm) --------------------------------------------
# Thin wrappers over .slurm/run-build-distribute.sh (build/push/test on a Slurm
# node) and `sbatch .slurm/run-cliff.sbatch` (the full cliff sweep).  These
# replace the former ./run-this.sh driver; chain them like the old one-shot, e.g.
# `make dist-build dist-push smoke-test` (make runs goals left-to-right).

dist-build:                    # Build image (+ fabric exporters) on a Slurm build node, save tarballs
	"$(DIST)" build
	@[ "$(AAI_BUILD_EXPORTERS)" = "0" ] || "$(DIST)" build-exporters

dist-push:                     # Tag + push the built image to a registry (needs AAI_PUSH_REF)
	"$(DIST)" push

smoke-test:                    # Load + smoke-test the image on a GPU+NVMe node
	"$(DIST)" test

# Submit the full 3-arm cliff sweep (vram_only + kvd_v2 nvme + kvd_v2 gds).  Pin
# a node with AAI_CLIFF_NODE, narrow arms with AAI_CLIFF_ARMS=nvme (etc).  The job
# creates logs/<job-id>/ itself and redirects its output there.
cliff-submit:
	@cd "$(CURDIR)" && jobid=$$(sbatch --parsable \
	    $(if $(AAI_CLIFF_NODE),--nodelist=$(AAI_CLIFF_NODE),) \
	    .slurm/run-cliff.sbatch) && \
	    echo "submitted cliff job $$jobid" && \
	    echo "log: $(CURDIR)/logs/$$jobid/cliff.out"

# Fast setup check: a single concurrency point, one timed iteration, all 3 arms.
# Respects user overrides of BENCH_CONCUR / BENCH_ITERS.
cliff-short:
	@cd "$(CURDIR)" && jobid=$$(BENCH_CONCUR="$${BENCH_CONCUR:-1}" BENCH_ITERS="$${BENCH_ITERS:-1}" \
	    sbatch --parsable --job-name=aai-day-cliff-short \
	    $(if $(AAI_CLIFF_NODE),--nodelist=$(AAI_CLIFF_NODE),) \
	    .slurm/run-cliff.sbatch) && \
	    echo "submitted cliff-short job $$jobid (BENCH_CONCUR=$${BENCH_CONCUR:-1} BENCH_ITERS=$${BENCH_ITERS:-1})" && \
	    echo "log: $(CURDIR)/logs/$$jobid/cliff.out"
