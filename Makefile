# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# This tree is self-contained: the build context and all sources live here, so
# "repo root" is this directory (no dependency on any parent checkout).
REPO_ROOT := $(CURDIR)

# ai-dynamo/nixl upstream main (2026-07-10); AIS_MT added via patches/nixl/.
NIXL_GIT_URL := https://github.com/ai-dynamo/nixl.git
NIXL_SHA     := 644facf0eb3de14ec63c1d2831238f63cd03c0e0

IMAGE_NAME ?= rocm-aic

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

# ---- Build parallelism -----------------------------------------------------
# Caps parallel compile jobs in the image build, Empty = use all cores ($(nproc)).
BUILD_JOBS ?=

export ROCM_ARCH GPU GDS_SLAB_DATA LOG HF_HOME IMAGE_NAME BUILD_JOBS
export LMCACHE_PORT LMCACHE_L1_SIZE_GB LMCACHE_NVME_POOL LMCACHE_NVME_SLOT_SIZE LMCACHE_NFS_POOL
export NVME_DATA NFS_DATA
export VLLM_MODEL TENSOR_PARALLEL_SIZE
export VLM_GPU_MEMORY_UTILIZATION VLM_MAX_MODEL_LEN VLM_MAX_NUM_BATCHED_TOKENS
export NIXL_GIT_URL NIXL_SHA

comma := ,
_COMPOSE_BIN := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")
COMPOSE      := DOCKER_BUILDKIT=1 $(_COMPOSE_BIN) -f "$(CURDIR)/docker/docker-compose.yml"

# ---- Metrics capture (Prometheus sidecar) ----------------------------------
# AIC_METRICS_DIR: Prometheus TSDB dir (bind-mount an NFS path here to explore
# a run afterward).  AIC_EXPORTERS=1 also launches the containerized node + AMD
# GPU exporters (for nodes without the host-installed exporter services).
AIC_METRICS_DIR ?= $(CURDIR)/logs/prometheus
AIC_EXPORTERS   ?= 0
MON_COMPOSE     := $(_COMPOSE_BIN) -f "$(CURDIR)/monitoring/docker-compose.monitoring.yml"
_MON_PROFILE    := $(if $(filter 1,$(AIC_EXPORTERS)),--profile exporters,)
export AIC_METRICS_DIR

# ---- Fabric exporters (nvme_exporter / rdma_exporter) ----------------------
# No published upstream images; we build them from monitoring/*/Dockerfile so the
# `exporters-fabric` compose profile and the .slurm docker-run fallback (nodes
# without the compose plugin) can containerize them.  Versions match the batesste
# host services for Grafana parity; override to bump.
NVME_EXPORTER_IMAGE   ?= aic-nvme-exporter:local
RDMA_EXPORTER_IMAGE   ?= aic-rdma-exporter:local
NVME_EXPORTER_VERSION ?= 3.0.0
RDMA_EXPORTER_VERSION ?= 0.3.0

PYTHON := $(if $(wildcard $(REPO_ROOT)/.venv/bin/python3),$(REPO_ROOT)/.venv/bin/python3,python3)

# ---- Distribute / cliff (Slurm) --------------------------------------------
# The dist-* / cliff-* targets shell out to the .slurm scripts; the AIC_* knobs
# they read pass straight through the environment (e.g. make dist-build
# AIC_ROCM_ARCH=gfx942, make cliff-submit AIC_CLIFF_NODE=<node-name>
# AIC_CLIFF_ARMS=nvme).  AIC_CACHE_DIR is the shared BuildKit file cache on
# /scratch so a failed build resumes from the last good layer on any node
# (set AIC_CACHE_DIR= to disable); AIC_BUILD_EXPORTERS=0 skips the fabric images.
DIST := $(CURDIR)/.slurm/run-build-distribute.sh

# ---- SPUR cluster overrides ------------------------------------------------
# When AIC_SPUR_CLUSTER=1, default storage paths to AIC_SHARED_NFS (the NFS
# volume shared across all SPUR compute nodes) instead of /scratch (not present
# on this cluster), and wire up the controller address via AIC_SPUR_CONTROLLER
# (set SPUR_CONTROLLER_ADDR in your environment, or pass AIC_SPUR_CONTROLLER=).
AIC_SPUR_CLUSTER ?= 0
AIC_SHARED_NFS ?=
ifeq ($(AIC_SPUR_CLUSTER),1)
export AIC_SPUR_CLUSTER
export AIC_SPUR_CONTROLLER  ?= $(SPUR_CONTROLLER_ADDR)
export AIC_IMAGE_DIR        ?= $(AIC_SHARED_NFS)/$(USER)/images
export AIC_CACHE_DIR        ?= $(AIC_SHARED_NFS)/$(USER)/images/buildcache
# SPUR nodes have 8x NVMe drives combined into a single LVM at /mnt/m2m_nobackup.
# Use override (not ?=) so these win over the top-level ?= defaults set earlier.
# HF_HOME points to AIC_SHARED_NFS since /scratch does not exist on this cluster.
override export NVME_DATA     := /mnt/m2m_nobackup/aic-cliff/nvme
override export GDS_SLAB_DATA := /mnt/m2m_nobackup/aic-cliff/slab
override export HF_HOME       := $(AIC_SHARED_NFS)/$(USER)/hf
else
export AIC_CACHE_DIR        ?= /scratch/$(USER)/images/buildcache
endif

# The cliff sbatch has its own node-appropriate defaults for the HuggingFace
# cache (staged on /scratch), the LMCache storage tiers (node-local /tmp), and
# the Prometheus TSDB (per-job logs/<job-id>/prometheus).  But the Makefile
# export'd HF_HOME / NVME_DATA / NFS_DATA / GDS_SLAB_DATA / AIC_METRICS_DIR with
# compose-oriented defaults (/mnt/..., ~/.cache, logs/prometheus), which
# sbatch --export=ALL would otherwise push into the job and clobber those
# defaults -- breaking cliff runs launched via make (the old run-this.sh never
# exported them).  So for cliff submits, strip ONLY the ones that came from the
# Makefile's own defaults ($(origin ...) = "file"); a value the user set on the
# command line or in their environment is kept and still flows through.
_CLIFF_STRIP := env \
    $(if $(filter 0,$(AIC_SPUR_CLUSTER)),$(if $(filter file,$(origin HF_HOME)),-u HF_HOME)) \
    $(if $(filter 0,$(AIC_SPUR_CLUSTER)),$(if $(filter file,$(origin NVME_DATA)),-u NVME_DATA)) \
    $(if $(filter file,$(origin NFS_DATA)),-u NFS_DATA) \
    $(if $(filter 0,$(AIC_SPUR_CLUSTER)),$(if $(filter file,$(origin GDS_SLAB_DATA)),-u GDS_SLAB_DATA)) \
    $(if $(filter file,$(origin AIC_METRICS_DIR)),-u AIC_METRICS_DIR)

# ---- Export tarball --------------------------------------------------------
# `make export` packages the sources needed to run this tooling into a single
# self-contained tarball.  It captures the current WORKING TREE -- tracked files
# with any local edits, plus new-but-not-yet-committed files -- while honoring
# .gitignore, so logs/, __pycache__/, *.pyc and prior tarballs stay out.  The
# default filename stamps today's date + the HEAD short rev (with a -dirty suffix
# when the tree has uncommitted changes).  Override the whole path with
# EXPORT_TARBALL=... or just the top-level dir inside the tarball via
# EXPORT_PREFIX=...
EXPORT_PREFIX  ?= aic-release
_GIT_SHORT_REV := $(shell git -C "$(CURDIR)" rev-parse --short HEAD 2>/dev/null || echo nogit)
_GIT_DIRTY     := $(if $(shell git -C "$(CURDIR)" status --porcelain -- . 2>/dev/null),-dirty,)
_GEN_DATE      := $(shell date +%Y%m%d)
EXPORT_TARBALL ?= $(CURDIR)/$(EXPORT_PREFIX)-$(_GEN_DATE)-$(_GIT_SHORT_REV)$(_GIT_DIRTY).tar.gz

.PHONY: help build up up-batch up-gds-l1 up-gds-l1-batch down logs logs-lmcache logs-vllm \
        ps shell-lmcache shell-vllm restart-vllm restart-lmcache cliff plot venv \
        monitoring-up monitoring-down monitoring-logs monitoring-build-exporters \
        dist-build dist-build-exporters dist-push smoke-test cliff-submit cliff-short \
        cliff-long-64k cliff-long-128k \
        export _check_hf_token _prep_dirs _check_gds_slab

.DEFAULT_GOAL := help

help:
	@echo "rocm-aic aic-release — AMD Infinity Context inference stack + benchmarks"
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
	@echo "  (dist-build/dist-build-exporters/smoke-test submit via sbatch and log to logs/<job-id>/)"
	@echo "  make dist-build        Build image (+ fabric exporters) on a Slurm build node, save tarballs"
	@echo "  make dist-build-exporters  Build ONLY the nvme/rdma exporter images (no main rebuild)"
	@echo "  make dist-push         Tag + push the built image (needs AIC_PUSH_REF)"
	@echo "  make smoke-test        Load + smoke-test the image on a GPU+NVMe node"
	@echo "                         (also sanity-checks exporters + writes a Prometheus TSDB"
	@echo "                          to logs/<job-id>/prometheus; AIC_SMOKE_EXPORTERS=0 skips)"
	@echo "  make cliff-submit      sbatch the full 3-arm cliff sweep -> logs/<job-id>/"
	@echo "  make cliff-short       sbatch a 1-point cliff (concur=1, 1 iter) to smoke-test the flow"
	@echo "  make cliff-long-64k    sbatch a 64k-ISL YaRN(x2) 3-arm sweep (pools sized for the working set)"
	@echo "  make cliff-long-128k   sbatch a 128k-ISL YaRN(x4) 3-arm sweep (extreme; big DRAM/slab pools)"
	@echo "    Chain like the old run-this.sh:  make dist-build dist-push smoke-test"
	@echo "    Pin a node: AIC_CLIFF_NODE=<node>   Narrow arms: AIC_CLIFF_ARMS=nvme (vram,nvme,gds)"
	@echo "    Target another GFX: AIC_CLIFF_GFX=gfx950 (or AIC_CLIFF_CONSTRAINT=<site>&GFX90A)"
	@echo "      non-gfx942 nodes: no local NVMe (nvme/gds arms fall back to /tmp); the model"
	@echo "      auto-selects by GPU arch (big CDNA=gpt-oss-120b, else Qwen2.5-3B); image is multi-arch"
	@echo "    Override sweep/model via env: BENCH_CONCUR=1,8,64 VLLM_MODEL=... make cliff-submit"
	@echo "    AIC_CACHE_DIR=$(AIC_CACHE_DIR)  (shared BuildKit cache; set empty to disable)"
	@echo ""
	@echo "Export target:"
	@echo "  make export            Tarball the working-tree sources (tracked + local edits)"
	@echo "    Default: $(notdir $(EXPORT_TARBALL))"
	@echo "    Override: make export EXPORT_TARBALL=/path/to/foo.tar.gz"
	@echo ""
	@echo "Metrics targets (Prometheus sidecar; scrapes vLLM/LMCache/exporters):"
	@echo "  make monitoring-up     Start Prometheus, TSDB -> AIC_METRICS_DIR"
	@echo "  make monitoring-down   Stop the metrics sidecar (TSDB retained)"
	@echo "  make monitoring-logs   Follow Prometheus logs"
	@echo "  make monitoring-build-exporters  Build nvme_exporter + rdma_exporter images"
	@echo "    AIC_METRICS_DIR=$(AIC_METRICS_DIR)"
	@echo "    AIC_EXPORTERS=$(AIC_EXPORTERS)  (1 = also launch node + AMD GPU exporters)"
	@echo ""
	@echo "Required env:"
	@echo "  HF_TOKEN       HuggingFace access token"
	@echo "  ROCM_ARCH      GPU arch (detected: $(ROCM_ARCH))"
	@echo ""
	@echo "Optional build env:"
	@echo "  TLS_CERT       Path to corporate CA cert (e.g. Zscaler); passed as a"
	@echo "                 BuildKit secret — never baked into the image."
	@echo "                 Example: make build TLS_CERT=/etc/ssl/certs/zscaler-ca.crt"
	@echo "  BUILD_JOBS     Cap parallel compile jobs (default: all cores)."
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
	@echo "  make build BUILD_JOBS=3          # cap parallelism on low-RAM hosts"
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
	docker exec -it aic-lmcache bash -l

shell-vllm:
	docker exec -it aic-vllm-gpu$(GPU) bash -l

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
	@mkdir -p "$(AIC_METRICS_DIR)"
	PROM_UID="$$(id -u)" PROM_GID="$$(id -g)" \
		$(MON_COMPOSE) $(_MON_PROFILE) up -d
	@echo "Prometheus up on :9090  (TSDB -> $(AIC_METRICS_DIR))"

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
	@echo "Run them via:  AIC_EXPORTERS=1 with --profile exporters-fabric, or set"
	@echo "AIC_NVME_EXPORTER_IMAGE / AIC_RDMA_EXPORTER_IMAGE for the .slurm docker-run path."

# ---- Distribute / cliff (Slurm) --------------------------------------------
# Thin wrappers over .slurm/run-build-distribute.sh (build/push/test on a Slurm
# node) and `sbatch .slurm/run-cliff.sbatch` (the full cliff sweep).  These
# replace the former ./run-this.sh driver; chain them like the old one-shot, e.g.
# `make dist-build dist-push smoke-test` (make runs goals left-to-right).

dist-build:                    # Build image (+ fabric exporters) on a Slurm build node, save tarballs
	"$(DIST)" build
	@# The fabric exporters are optional (bare-node fallback) and their Dockerfile
	@# pulls debian:12-slim from Docker Hub, so the build node needs registry egress.
	@# Keep a failure here non-fatal: the main image (the artifact that matters) is
	@# already built.  Set AIC_BUILD_EXPORTERS=0 to skip the step entirely.
	@[ "$(AIC_BUILD_EXPORTERS)" = "0" ] || "$(DIST)" build-exporters \
	    || echo "WARNING: fabric-exporter build failed (optional; main image is built). Retry on a node with Docker Hub access, or set AIC_BUILD_EXPORTERS=0."

dist-build-exporters:          # Build ONLY the fabric exporters (no main-image rebuild)
	@# Rebuild just the nvme/rdma exporter images -- e.g. after `make dist-build`
	@# succeeded for the main image but the exporter step failed for lack of Docker
	@# Hub egress.  Pin an egress-capable node with AIC_BUILD_NODE=<node>, or build
	@# on the current host with AIC_BUILD_LOCAL=1.
	"$(DIST)" build-exporters

dist-push:                     # Tag + push the built image to a registry (needs AIC_PUSH_REF)
	"$(DIST)" push

smoke-test:                    # Load + smoke-test the image on a GPU+NVMe node
	"$(DIST)" test

# Submit the full 3-arm cliff sweep (vram_only + kvd_v2 nvme + kvd_v2 gds).  Pin
# a node with AIC_CLIFF_NODE, narrow arms with AIC_CLIFF_ARMS=nvme (etc), and
# override the job wall-time with AIC_CLIFF_TIME=HH:MM:SS (e.g. an overnight full
# sweep).  The job creates logs/<job-id>/ itself and redirects its output there.
# _CLIFF_SBATCH_ARGS: partition + constraint overrides passed on the sbatch
# command line (takes precedence over #SBATCH directives in run-cliff.sbatch).
# On SPUR, override to amd-spur with no constraint and no --gres (no GPU GRES
# configured); on standard Slurm we pass $(AIC_CLIFF_CONSTRAINT) (below).
#
# ---- cliff GFX / constraint selection ----
# By default the cliff job runs on a gfx942 node with local NVMe -- the
# validated tiered-cache path.  To target another GFX arch, set
#   AIC_CLIFF_GFX=gfx950        -> expands to constraint "GFX950"
# (the &NVME requirement is dropped, since only gfx942 nodes advertise NVME),
# or pass a full Slurm constraint expression directly via
#   AIC_CLIFF_CONSTRAINT=GFX90A
# AIC_CLIFF_CONSTRAINT wins if both are set; either overrides the #SBATCH
# --constraint line baked into run-cliff.sbatch.  Caveats for non-gfx942 nodes:
#   * no local NVMe -> the nvme/gds arms fall back to root-disk /tmp
#     (AIC_NVME_AUTO case 4): slower and less representative, but they run.
#   * gpt-oss-120b will NOT fit on small-VRAM parts (gfx1100/1151/1201; tight on
#     gfx90a), so the job auto-selects the model from the node's detected GPU
#     arch (big CDNA gfx942/gfx950 -> gpt-oss-120b, everything else -> a small
#     model); see select_default_model in .slurm/run-cliff.sbatch.  Override with
#     VLLM_MODEL=<pre-staged model> (offline HF_HOME) or the AIC_MODEL_BIG/
#     AIC_MODEL_SMALL tier knobs.
#   * the loaded image must contain kernels for the target arch.  This is
#     already the case: `make dist-build` is multi-arch by default (AIC_ROCM_ARCH
#     defaults to gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;
#     gfx1201 -- see .slurm/run-build-distribute.sh).  RDNA parts have no
#     NVMe-DMA hardware, so the gds arm is CDNA-only there.
AIC_CLIFF_GFX ?=
ifeq ($(strip $(AIC_CLIFF_CONSTRAINT)),)
ifneq ($(strip $(AIC_CLIFF_GFX)),)
AIC_CLIFF_CONSTRAINT := $(shell echo '$(AIC_CLIFF_GFX)' | tr '[:lower:]' '[:upper:]')
else
AIC_CLIFF_CONSTRAINT := GFX942&NVME
endif
endif
ifeq ($(AIC_SPUR_CLUSTER),1)
_CLIFF_SPUR_CTL  := SPUR_CONTROLLER_ADDR=$(AIC_SPUR_CONTROLLER)
_CLIFF_SBATCH_ARGS := --partition=amd-spur --constraint= \
    $(if $(AIC_CLIFF_NODE),--nodelist=$(AIC_CLIFF_NODE),)
# SPUR sbatch does not support --parsable; parse job id from "Submitted batch job N"
_CLIFF_SUBMIT     = $(_CLIFF_SPUR_CTL) $(_CLIFF_STRIP) sbatch \
    $(_CLIFF_SBATCH_ARGS) $(1) .slurm/run-cliff.sbatch 2>&1 | \
    tee /dev/stderr | grep -oE '[0-9]+$$' | tail -1
else
# NB: single-quote the constraint -- it contains '&' (a shell metacharacter) that
# would otherwise background the sbatch call in the recipe subshell.
_CLIFF_SBATCH_ARGS := --constraint='$(AIC_CLIFF_CONSTRAINT)' \
    $(if $(AIC_CLIFF_NODE),--nodelist=$(AIC_CLIFF_NODE),)
_CLIFF_SUBMIT     = $(_CLIFF_STRIP) sbatch --parsable \
    $(_CLIFF_SBATCH_ARGS) $(1) .slurm/run-cliff.sbatch
endif

cliff-submit:
	@cd "$(CURDIR)" && jobid=$$($(call _CLIFF_SUBMIT,\
	    $(if $(AIC_CLIFF_TIME),--time=$(AIC_CLIFF_TIME),))) && \
	    echo "submitted cliff job $$jobid" && \
	    echo "log: $(CURDIR)/logs/$$jobid/cliff.out"

# Fast setup check: a single concurrency point, one timed iteration, all 3 arms.
# Respects user overrides of BENCH_CONCUR / BENCH_ITERS.
cliff-short:
	@cd "$(CURDIR)" && jobid=$$(BENCH_CONCUR="$${BENCH_CONCUR:-1}" BENCH_ITERS="$${BENCH_ITERS:-1}" \
	    $(call _CLIFF_SUBMIT,--job-name=aic-cliff-short)) && \
	    echo "submitted cliff-short job $$jobid (BENCH_CONCUR=$${BENCH_CONCUR:-1} BENCH_ITERS=$${BENCH_ITERS:-1})" && \
	    echo "log: $(CURDIR)/logs/$$jobid/cliff.out"

# Long-ISL sweeps with YaRN RoPE scaling (all 3 arms).  Sizing philosophy: a
# MODEST, portable DRAM L1 (64 GB -- realistic on real servers) in front of a
# BIG NVMe L2, so the bulk of the working set lives on NVMe (the representative
# tiered config).  LMCACHE_NVME_POOL is a SLOT COUNT (~4.5 MiB/slot for
# Qwen2.5-3B, one FD per slot, cap <= container nofile 1048576); slots are
# lazily sized on disk, so a large pool caps capacity without pre-consuming it
# -- real disk use = the working set.  Working set ~= concurrency x shared-prefix
# KV: 64k/60k ~= 1.05 GiB/client (~263 GB @ c=250); 128k/126k ~= 2.16 GiB/client
# (~540 GB @ c=250) -- both fit the spare NVMe.  Qwen2.5 is YaRN-trained to 128k.
# Every knob is overridable, e.g. LMCACHE_MAX_LOCAL_CPU_SIZE=128 make cliff-long-64k.
#
# NOTE on the origin-guarded pool/slab sizing below: LMCACHE_NVME_POOL and
# LMCACHE_L1_SIZE_GB are BOTH Makefile-defaulted (?=) AND exported (line ~61), so
# they arrive in the recipe shell already SET to the small compose defaults
# (4096 / 20).  A plain $${VAR:-262144} therefore never fires -- the var is set,
# so it silently kept 4096 / 20, which caused the c>=32 cliff collapse (pool was
# 18 GiB not 1.15 TiB; gds slab 20 GB not 320 GB) in jobs 67536798/67537066.
# Use $(origin ...)=file to mean "came from the Makefile default, not the user":
# a Makefile default -> the long-ISL value; a real user override (command line /
# environment) is kept.  Mirrors the _CLIFF_STRIP idiom above.
cliff-long-64k:                # sbatch a 64k-ISL YaRN(x2 -> 65536) 3-arm sweep
	@cd "$(CURDIR)" && jobid=$$( \
	    VLLM_MODEL="$${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}" \
	    VLM_GPU_MEMORY_UTILIZATION="$${VLM_GPU_MEMORY_UTILIZATION:-0.12}" \
	    VLM_YARN_FACTOR="$${VLM_YARN_FACTOR:-2.0}" VLM_MAX_MODEL_LEN="$${VLM_MAX_MODEL_LEN:-65536}" \
	    BENCH_ISL="$${BENCH_ISL:-64000}" BENCH_SHARED_TOK="$${BENCH_SHARED_TOK:-60000}" \
	    BENCH_PREFIX_MODE="$${BENCH_PREFIX_MODE:-per_client}" BENCH_ITERS="$${BENCH_ITERS:-2}" \
	    AIC_LOCAL_CPU="$${AIC_LOCAL_CPU:-true}" LMCACHE_MAX_LOCAL_CPU_SIZE="$${LMCACHE_MAX_LOCAL_CPU_SIZE:-64}" \
	    LMCACHE_NVME_POOL="$(if $(filter file,$(origin LMCACHE_NVME_POOL)),262144,$(LMCACHE_NVME_POOL))" AIC_NIXL_BUFFER_SIZE="$${AIC_NIXL_BUFFER_SIZE:-8589934592}" \
	    LMCACHE_L1_SIZE_GB="$(if $(filter file,$(origin LMCACHE_L1_SIZE_GB)),320,$(LMCACHE_L1_SIZE_GB))" \
	    $(call _CLIFF_SUBMIT,--job-name=aic-cliff-long64k \
	    --time=$(if $(AIC_CLIFF_TIME),$(AIC_CLIFF_TIME),16:00:00))) && \
	    echo "submitted cliff-long-64k job $$jobid (ISL=64000, YaRN x2 -> 65536, DRAM L1=64G, NVMe pool=262144, all 3 arms)" && \
	    echo "log: $(CURDIR)/logs/$$jobid/cliff.out"

cliff-long-128k:               # sbatch a 128k-ISL YaRN(x4 -> 131072) 3-arm sweep (extreme)
	@cd "$(CURDIR)" && jobid=$$( \
	    VLLM_MODEL="$${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}" \
	    VLM_GPU_MEMORY_UTILIZATION="$${VLM_GPU_MEMORY_UTILIZATION:-0.12}" \
	    VLM_YARN_FACTOR="$${VLM_YARN_FACTOR:-4.0}" VLM_MAX_MODEL_LEN="$${VLM_MAX_MODEL_LEN:-131072}" \
	    BENCH_ISL="$${BENCH_ISL:-128000}" BENCH_SHARED_TOK="$${BENCH_SHARED_TOK:-126000}" \
	    BENCH_PREFIX_MODE="$${BENCH_PREFIX_MODE:-per_client}" BENCH_ITERS="$${BENCH_ITERS:-1}" \
	    AIC_LOCAL_CPU="$${AIC_LOCAL_CPU:-true}" LMCACHE_MAX_LOCAL_CPU_SIZE="$${LMCACHE_MAX_LOCAL_CPU_SIZE:-64}" \
	    LMCACHE_NVME_POOL="$(if $(filter file,$(origin LMCACHE_NVME_POOL)),524288,$(LMCACHE_NVME_POOL))" AIC_NIXL_BUFFER_SIZE="$${AIC_NIXL_BUFFER_SIZE:-8589934592}" \
	    LMCACHE_L1_SIZE_GB="$(if $(filter file,$(origin LMCACHE_L1_SIZE_GB)),640,$(LMCACHE_L1_SIZE_GB))" \
	    $(call _CLIFF_SUBMIT,--job-name=aic-cliff-long128k \
	    --time=$(if $(AIC_CLIFF_TIME),$(AIC_CLIFF_TIME),24:00:00))) && \
	    echo "submitted cliff-long-128k job $$jobid (ISL=128000, YaRN x4 -> 131072, DRAM L1=64G, NVMe pool=524288, all 3 arms)" && \
	    echo "log: $(CURDIR)/logs/$$jobid/cliff.out"

# ---- Export tarball --------------------------------------------------------
export:                        # Package the working-tree sources into a dated, rev-stamped tarball
	@git -C "$(CURDIR)" rev-parse HEAD >/dev/null 2>&1 || { \
		echo "ERROR: not a git checkout; cannot enumerate sources" >&2; exit 1; }
	@# ls-files (cached + others, minus .gitignore'd) gives the working set as
	@# NUL-separated relative paths; tar reads their on-disk content so local
	@# edits are included, --transform renames the top dir to EXPORT_PREFIX, and
	@# --ignore-failed-read tolerates index-only entries (e.g. a tracked file
	@# deleted on disk) instead of aborting.
	@cd "$(CURDIR)" && git ls-files -z --cached --others --exclude-standard \
		| tar --null --no-recursion --ignore-failed-read --owner=0 --group=0 \
			--transform='s|^|$(EXPORT_PREFIX)/|' \
			-czf "$(EXPORT_TARBALL)" -T -
	@echo "Wrote $(EXPORT_TARBALL)"
