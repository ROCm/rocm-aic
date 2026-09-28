# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# This tree is self-contained: the build context and all sources live here, so
# "repo root" is this directory (no dependency on any parent checkout).
REPO_ROOT := $(CURDIR)

# ai-dynamo/nixl v1.4.1 release; AIS_MT added via patches/nixl/.
NIXL_GIT_URL := https://github.com/ai-dynamo/nixl.git
NIXL_SHA     := v1.4.1

IMAGE_NAME       ?= rocm-aic
VLLM_IMAGE_NAME  ?= aic-vllm
LMCACHE_IMAGE_NAME ?= aic-lmcache
override AIC_VERSION := $(strip $(file <$(REPO_ROOT)/VERSION))

# Detect ROCM_ARCH early (before _IMAGE_TAG) so the tag script can decide
# whether to include FlashAttention (CDNA-only).  Command-line overrides win.
_ROCM_ARCH_DETECTED := $(shell rocm_agent_enumerator 2>/dev/null | grep -E '^gfx' | head -1)
ROCM_ARCH := $(if $(strip $(ROCM_ARCH)),$(strip $(ROCM_ARCH)),$(_ROCM_ARCH_DETECTED))

# PyTorch, vLLM, and all deps are source-built; PYTORCH_BRANCH, VLLM_REF, and
# LLM_EMU_REF are the key version knobs.  hipFile ships in the ROCm base image.
_FRAMEWORK_VERSION_ARGS := AIC_VERSION ROCM_VERSION PYTORCH_BRANCH VLLM_REF LLM_EMU_REF AITER_REF FLASH_ATTN_REF LMCACHE_REF NIXL_REF HSA_SNOOP_REF
_single_quote := '
_shell_quote = '$(subst $(_single_quote),'"'"',$(1))'
_FRAMEWORK_VERSION_ENV := $(foreach _arg,$(_FRAMEWORK_VERSION_ARGS),$(if $(filter undefined,$(origin $(_arg))),,$(_arg)=$(call _shell_quote,$(value $(_arg)))))

_IMAGE_TAG := $(shell ROCM_ARCH=$(ROCM_ARCH) $(_FRAMEWORK_VERSION_ENV) $(REPO_ROOT)/docker/scripts/aic-image-tag.sh 2>/dev/null)
IMAGE_TAG  ?= $(if $(_IMAGE_TAG),$(_IMAGE_TAG),latest)
IMAGE_REF          := $(IMAGE_NAME):$(IMAGE_TAG)
VLLM_IMAGE_REF     := $(VLLM_IMAGE_NAME):$(IMAGE_TAG)
LMCACHE_IMAGE_REF  := $(LMCACHE_IMAGE_NAME):$(IMAGE_TAG)

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
VLM_BLOCK_SIZE              ?=

# ---- Benchmark knobs -------------------------------------------------------
BENCH_ARM         ?= kvd_v2
BENCH_ISL         ?= 20000
BENCH_SHARED_TOK  ?= 18000
BENCH_CONCUR      ?= 1,2,4,8,16,32,48,64,80,100,128,160,200,250
BENCH_ITERS       ?= 3
BENCH_ENDPOINT    ?= http://localhost:8000
BENCH_MODEL       ?= $(VLLM_MODEL)

# ---- KVBench container -----------------------------------------------------
KVBENCH_GIT_URL           ?= https://github.com/wvaske/llm-kv-passthrough.git
KVBENCH_REF               ?= 177cf4e7c46ece8e03a701f89b60cd3a329f45fd
KVBENCH_IMAGE_NAME        ?= aic-kvbench
KVBENCH_IMAGE_TAG         ?= 177cf4e
KVBENCH_IMAGE_REF         ?= $(KVBENCH_IMAGE_NAME):$(KVBENCH_IMAGE_TAG)
KVBENCH_MODEL             ?= llama-3.1-8b
KVBENCH_GPU_PROFILE       ?= H100_SXM
KVBENCH_PORT              ?= 8000
KVBENCH_HOST_PORT         ?= 18000
KVBENCH_CHUNK_SIZE        ?= 256
KVBENCH_LOCAL_CPU_SIZE_GB ?= 0.25
KVBENCH_READY_S           ?= 120

# Non-Slurm runs land under logs/manual/ — keeps the tree root free of results/ and plots/.
BENCH_LOGDIR      := logs/manual
BENCH_OUT         := $(BENCH_LOGDIR)/results/cliff-$(BENCH_ARM)-$(shell date +%Y%m%d-%H%M%S).csv

# ---- Build parallelism -----------------------------------------------------
BUILD_JOBS ?=

# ---- Local buildx cache ----------------------------------------------------
AIC_LOCAL_BUILDER   ?= aic-local
AIC_LOCAL_CACHE_DIR ?= $(HOME)/.cache/rocm-aic-buildx

export AIC_VERSION ROCM_ARCH GPU GDS_SLAB_DATA LOG HF_HOME HF_TOKEN IMAGE_NAME IMAGE_REF IMAGE_TAG BUILD_JOBS
export VLLM_IMAGE_NAME VLLM_IMAGE_REF LMCACHE_IMAGE_NAME LMCACHE_IMAGE_REF
export LMCACHE_PORT LMCACHE_L1_SIZE_GB LMCACHE_NVME_POOL LMCACHE_NVME_SLOT_SIZE LMCACHE_NFS_POOL
export NVME_DATA NFS_DATA
export VLLM_MODEL TENSOR_PARALLEL_SIZE
export VLM_GPU_MEMORY_UTILIZATION VLM_MAX_MODEL_LEN VLM_MAX_NUM_BATCHED_TOKENS VLM_BLOCK_SIZE
export NIXL_GIT_URL NIXL_SHA
export KVBENCH_GIT_URL KVBENCH_REF KVBENCH_IMAGE_REF KVBENCH_MODEL KVBENCH_GPU_PROFILE
export KVBENCH_PORT KVBENCH_HOST_PORT KVBENCH_CHUNK_SIZE KVBENCH_LOCAL_CPU_SIZE_GB

# Compose passes a declared build argument through only when it is present in
# its environment.  Do not export empty attention-backend overrides: an empty
# value would override the pinned Dockerfile default and produce an invalid
# AITER release URL.  A non-empty command-line or environment override remains
# available for validated version-pair updates.
ifneq ($(strip $(AITER_REF)),)
export AITER_REF
endif
ifneq ($(strip $(FLASH_ATTN_REF)),)
export FLASH_ATTN_REF
endif

comma := ,
_COMPOSE_BIN := docker compose
COMPOSE      := $(_FRAMEWORK_VERSION_ENV) DOCKER_BUILDKIT=1 $(_COMPOSE_BIN) -f "$(CURDIR)/docker/docker-compose.yml"
COMPOSE_CACHE := $(COMPOSE) --profile cache

COMPOSE_PLUGIN_VERSION ?= v5.5.1

# vLLM --kv-transfer-config for the MP connector.  Leave KV_TRANSFER_ARG empty
# for a plain (baseline) vLLM.
_MP_CONNECTOR_JSON := {"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://aic-lmcache","lmcache.mp.port":$(LMCACHE_PORT)}}
KV_TRANSFER_ARG    ?= --kv-transfer-config '$(_MP_CONNECTOR_JSON)'
export KV_TRANSFER_ARG

# ---- Metrics capture (Prometheus sidecar) ----------------------------------
AIC_METRICS_DIR  ?= $(CURDIR)/logs/prometheus
AIC_EXPORTERS    ?= 0
AIC_GRAFANA_PORT ?= 3000
AIC_GRAFANA_IMAGE ?= grafana/grafana:13.2.2
MON_COMPOSE     := $(_COMPOSE_BIN) -f "$(CURDIR)/docker/docker-compose.yml"
_MON_PROFILE    := --profile monitoring-base $(if $(filter 1,$(AIC_EXPORTERS)),--profile exporters,)
export AIC_METRICS_DIR AIC_GRAFANA_PORT AIC_GRAFANA_IMAGE

# ---- Fabric exporters (nvme_exporter / rdma_exporter) ----------------------
NVME_EXPORTER_IMAGE   ?= aic-nvme-exporter:local
RDMA_EXPORTER_IMAGE   ?= aic-rdma-exporter:local
NVME_EXPORTER_VERSION ?= 3.0.0
RDMA_EXPORTER_VERSION ?= 0.7.3

PYTHON := $(if $(wildcard $(REPO_ROOT)/.venv/bin/python3),$(REPO_ROOT)/.venv/bin/python3,python3)

# ---- Distribute / cliff (Slurm) --------------------------------------------
DIST := $(CURDIR)/.slurm/run-build-distribute.sh

# ---- Self-hosted CI runner scripts -----------------------------------------
AIC_CI_LIB_DIR    ?= /usr/local/lib/aic-ci
AIC_CI_SCRIPT_DIR := $(CURDIR)/.github/scripts/runners

AIC_FAST_ARCH ?= gfx950

# ---- accuracy-test ---------------------------------------------------------
export AIC_ACCURACY_MODEL AIC_ACCURACY_DELTA
export AIC_ACCURACY_TIME AIC_ACCURACY_CPUS AIC_ACCURACY_MEM
export AIC_ACCURACY_READY_TIMEOUT
export AIC_ACCURACY_LIMIT

# ---- SPUR cluster overrides ------------------------------------------------
AIC_SPUR_CLUSTER ?= 0
AIC_SHARED_NFS ?=
ifeq ($(AIC_SPUR_CLUSTER),1)
export AIC_SPUR_CLUSTER
export AIC_SPUR_CONTROLLER  ?= $(SPUR_CONTROLLER_ADDR)
export AIC_IMAGE_DIR        ?= $(AIC_SHARED_NFS)/rocm-aic/images
export AIC_CACHE_DIR        ?= $(AIC_SHARED_NFS)/$(USER)/buildcache
override export NVME_DATA     := /mnt/m2m_nobackup/aic-cliff/nvme
override export GDS_SLAB_DATA := /mnt/m2m_nobackup/aic-cliff/slab
override export HF_HOME       := $(AIC_SHARED_NFS)/huggingface
override export ROCM_ARCH     := gfx950
export AIC_L2_BACKEND         ?= nixl_posix
export LMCACHE_NIXL_POSIX_SLOT_SIZE ?= 33554432
export LMCACHE_NIXL_POSIX_POOL      ?= 32768
override export AIC_EXPORTERS            := safe
override export AIC_HSA_SNOOP_PID_MODE  := container:aic-lmcache
export LMCACHE_MAX_GPU_WORKERS      ?= 1
else
export AIC_CACHE_DIR        ?= /scratch/$(USER)/images/buildcache
ifneq ($(wildcard /scratch/models/hub),)
ifeq ($(origin HF_HOME),file)
override export HF_HOME     := /scratch/models
_HF_HOME_IS_DEFAULT := 1
endif
endif
endif
_HF_HOME_IS_DEFAULT ?= $(if $(filter file,$(origin HF_HOME)),1,)

# Strip Makefile-defaulted path vars before cliff sbatch so they don't clobber
# the job's node-appropriate defaults.  User-set values (command-line /
# environment) are preserved.
_CLIFF_STRIP := env \
    $(if $(filter 0,$(AIC_SPUR_CLUSTER)),$(if $(_HF_HOME_IS_DEFAULT),-u HF_HOME)) \
    $(if $(filter 0,$(AIC_SPUR_CLUSTER)),$(if $(filter file,$(origin NVME_DATA)),-u NVME_DATA)) \
    $(if $(filter file,$(origin NFS_DATA)),-u NFS_DATA) \
    $(if $(filter 0,$(AIC_SPUR_CLUSTER)),$(if $(filter file,$(origin GDS_SLAB_DATA)),-u GDS_SLAB_DATA)) \
    $(if $(filter file,$(origin AIC_METRICS_DIR)),-u AIC_METRICS_DIR)

# ---- Export tarball --------------------------------------------------------
EXPORT_PREFIX  ?= aic-release
_GIT_SHORT_REV := $(shell git -C "$(CURDIR)" rev-parse --short HEAD 2>/dev/null || echo nogit)
_GIT_DIRTY     := $(if $(shell git -C "$(CURDIR)" status --porcelain -- . 2>/dev/null),-dirty,)
_GEN_DATE      := $(shell date +%Y%m%d)
EXPORT_TARBALL ?= $(CURDIR)/$(EXPORT_PREFIX)-$(_GEN_DATE)-$(_GIT_SHORT_REV)$(_GIT_DIRTY).tar.gz

# ---- PROM_DUMP knobs (consumed by mk/dist.mk prometheus-dump) ---------------
PROM_DUMP_OUT  ?=
PROM_DUMP_WAIT ?= 15

# ---- Sub-makefiles ----------------------------------------------------------
include mk/infra.mk
include mk/local.mk
include mk/bench.mk
include mk/dist.mk

.PHONY: help
.DEFAULT_GOAL := help

help:
	@echo "rocm-aic aic-release — AMD Infinity Context inference stack + benchmarks"
	@echo ""
	@echo "Stack targets:"
	@echo "  make ensure-compose    Install the docker compose v2 plugin if missing (user-local)"
	@echo "  make build             Build the shared image ($(IMAGE_REF))"
	@echo "  make build-local       Two-stage local build without Slurm: aic-base then aic-vllm"
	@echo "                         Use when docker compose build fails (vllm Dockerfile needs aic-base)"
	@echo "  make build-cached      Like build but uses buildx with a local layer cache"
	@echo "                         (build_pytorch survives docker system prune)"
	@echo "                         AIC_LOCAL_CACHE_DIR=$(AIC_LOCAL_CACHE_DIR)"
	@echo "  make up                Start lmcache + vllm (foreground, DRAM L1 + AIS_MT/NFS L2)"
	@echo "  make up-batch          Start lmcache + vllm (background)"
	@echo "  make up-dev            Start in dev mode: --enforce-eager skips CUDA graph capture (~60s faster, ~10% slower inference)"
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
	@echo "  make stress-grafana    Sustained KVD stress loop for Grafana: ISL=1024, c=1/2/4/8,"
	@echo "                         5 iters per pass, repeats until ctrl-c — drives L1/L2 panels"
	@echo "  make vllm-reset-test   Verify LMCache L1+L2 retrieval: small L1 (1GiB), NIXL POSIX L2,"
	@echo "                         flood to overflow, POST /reset_prefix_cache, confirm L1+L2 hits"
	@echo "  make cliff             Run KV-cache cliff benchmark, write CSV to $(BENCH_LOGDIR)/results/"
	@echo "  make kvbench-build     Build the pinned KVBench image ($(KVBENCH_IMAGE_REF))"
	@echo "  make kvbench-up        Start KVBench + client on the internal compose network"
	@echo "  make kvbench-logs      Follow KVBench logs"
	@echo "  make kvbench-down      Stop the KVBench compose profile"
	@echo "  make cliff-kvbench-local  Start KVBench + client, wait for readiness, then run the kvbench cliff arm"
	@echo "  make plot              Generate cliff PNG charts from $(BENCH_LOGDIR)/results/ CSVs"
	@echo ""
	@echo "Distribute / cliff targets (Slurm; wrap .slurm/ scripts + sbatch):"
	@echo "  (dist-build/dist-build-exporters/smoke-test submit via sbatch and log to logs/<job-id>/)"
	@echo "  make dist-build        Build image (+ fabric exporters) on a Slurm build node, save tarballs"
	@echo "  make dist-build-fast   Single-arch dev build (AIC_FAST_ARCH=$(AIC_FAST_ARCH), no exporters) -- faster iteration"
	@echo "  make dist-build-emulate  Build the CPU-only emulation image (no GPU kernels compiled)"
	@echo "  make dist-build-base   Build aic-base image (PyTorch + torchvision)"
	@echo "  make dist-build-vllm   Build aic-vllm image (requires aic-base)"
	@echo "  make dist-build-lmcache  Build aic-lmcache image (requires aic-base)"
	@echo "  make dist-build-parallel  Build base, then vllm + lmcache in parallel"
	@echo "  make dist-build-exporters  Build ONLY the nvme/rdma exporter images (no main rebuild)"
	@echo "  make dist-build-monitoring Pull + save Prometheus/amdgpu-exporter to AIC_IMAGE_DIR"
	@echo "  make dist-push         Tag + push the built image (needs AIC_PUSH_REF)"
	@echo "  make smoke-test        Load + smoke-test the image on a GPU+NVMe node"
	@echo "  make smoke-test-fast   Smoke-test the single-arch dev image (AIC_FAST_ARCH=$(AIC_FAST_ARCH))"
	@echo "  make tiny-test         End-to-end serve check (MP stack + tiny model, one completion)"
	@echo "  make tiny-test-fast    Fast variant of tiny-test"
	@echo "  make test-rocjitsu-local  Boot a QEMU VM with emulated gfx1250 (rocjitsu), build+load the gfx1250 image, run a test completion (requires /dev/kvm)"
	@echo "  make test-emulate-local  Local emulate test (no SLURM): bring up vllm-emulator, assert completion + hook"
	@echo "  make stress-emulate-local  Start emulator + Prometheus, run sustained sweep, print /metrics"
	@echo "  make capture-profile-local  Local profile capture (requires /dev/kfd): real GPU serve + sweep -> pack"
	@echo "  make emulate-test      Serve check of the emulation image on a CPU-only node (no GPU)"
	@echo "  make emulate-mp-test   Emulation + the full LMCache MP recipe on a CPU-only node"
	@echo "  make profile-capture   Capture an AMD profile pack from a REAL GPU serve (gfx942/gfx950)"
	@echo "  make emulate-validate  Replay a captured pack on CPU and diff vs the real-hardware run"
	@echo "  make accuracy-test     KV-integrity gate: differential lm_eval, two arms"
	@echo "  make accuracy-test-fast"
	@echo "                         The same gate, AIC_ROCM_ARCH pinned to AIC_FAST_ARCH"
	@echo "  make accuracy-test-very-fast"
	@echo "                         100-item cap per arm (~25 min); floor widened by 3x SE"
	@echo "  make install-ci-scripts  Deploy .github/scripts/runners/*.sh to $(AIC_CI_LIB_DIR) (sudo if needed)"
	@echo "  make cliff-kvbench-submit  sbatch a CPU-only KVBench cliff run (compose kvbench + client)"
	@echo "  make cliff-submit      sbatch the full 3-arm cliff sweep -> logs/<job-id>/"
	@echo "  make cliff-kvd         sbatch focused KVD cliff: shared prefix, sparse c ladder (1,8,32,64,128,250)"
	@echo "  make cliff-spur-l2     sbatch SPUR-tuned L2 cliff: per_client prefix, util=0.40, 8GB DRAM L1, c=1/8/32 (vram+nvme)"
	@echo "  make cliff-spur-l2-debug  Like cliff-spur-l2 but c=1 only, tiny L1=0.1GB, DEBUG logging (diagnose ext_hit=0)"
	@echo "  make cliff-short       sbatch a 1-point cliff (concur=1, 1 iter) to smoke-test the flow"
	@echo "  make cliff-long-64k    sbatch a 64k-ISL YaRN(x2) 3-arm sweep (pools sized for the working set)"
	@echo "  make cliff-long-128k   sbatch a 128k-ISL YaRN(x4) 3-arm sweep (extreme; big DRAM/slab pools)"
	@echo "    Chain like the old run-this.sh:  make dist-build dist-push smoke-test"
	@echo "    Pin a node: AIC_CLIFF_NODE=<node>   Narrow arms: AIC_CLIFF_ARMS=nvme (vram,nvme,gds)"
	@echo "    Target another GFX: AIC_CLIFF_GFX=gfx950 (or AIC_CLIFF_CONSTRAINT=<site>&GFX90A)"
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
	@echo "  make prometheus-dump   Submit SPUR job: full GPU stack → scrape all /metrics →"
	@echo "                         Markdown reference doc on shared NFS (requires built image)"
	@echo "    PROM_DUMP_OUT=$(if $(PROM_DUMP_OUT),$(PROM_DUMP_OUT),<AIC_IMAGE_DIR>/../prometheus-dump.md)"
	@echo "    AIC_METRICS_DIR=$(AIC_METRICS_DIR)"
	@echo "    AIC_EXPORTERS=$(AIC_EXPORTERS)  (1 = also launch node + AMD GPU exporters)"
	@echo "    AIC_GRAFANA_PORT=$(AIC_GRAFANA_PORT)   Grafana host port (default: 3000)"
	@echo "    AIC_GRAFANA_IMAGE=$(AIC_GRAFANA_IMAGE)"
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
	@echo "  make build-cached ROCM_ARCH=gfx1201  # persistent buildx cache"
	@echo "  make build BUILD_JOBS=3          # cap parallelism on low-RAM hosts"
	@echo "  make up HF_TOKEN=hf_... NVME_DATA=/mnt/nvme NFS_DATA=/mnt/nfs"
	@echo "  make up-gds-l1 GDS_SLAB_DATA=/mnt/nvme HF_TOKEN=hf_..."
	@echo "  make cliff BENCH_ARM=vram_only BENCH_ENDPOINT=http://localhost:8000"
	@echo "  make cliff BENCH_ARM=kvbench BENCH_ENDPOINT=http://localhost:8000"
	@echo "  make cliff-kvbench-local KVBENCH_MODEL=llama-3.1-8b BENCH_CONCUR=1,2"
	@echo "  make plot"
	@echo ""
