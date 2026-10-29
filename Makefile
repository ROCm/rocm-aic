# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# This tree is self-contained: the build context and all sources live here, so
# "repo root" is this directory (no dependency on any parent checkout).
REPO_ROOT := $(CURDIR)

# ---- Per-cluster environment file -------------------------------------------
# A repo-root `.env.<cluster>` carries the settings that differ per site --
# partition, constraint, image dir, cache dir, arch -- so they stop living on
# the make command line.  Plain KEY=value lines, unquoted (make takes quotes
# literally); `.slurm/run-build-distribute.sh` reads the same file.
#
# AIC_CLUSTER picks the file and is auto-detected from the hostname: the first
# dash-separated token with a matching file (ctr2-alola-login-04 -> .env.alola).
# Precedence here is `make VAR=...` > .env.<cluster> > the defaults below; in
# the Slurm script an exported variable also beats the file.  Keep these files
# untracked -- they describe one site and are where private paths end up.
AIC_CLUSTER  ?= $(firstword $(foreach _t,$(SLURM_CLUSTER_NAME) \
                    $(subst -, ,$(shell hostname -s 2>/dev/null)),\
                    $(if $(wildcard $(REPO_ROOT)/.env.$(_t)),$(_t))))
AIC_ENV_FILE ?= $(if $(AIC_CLUSTER),$(REPO_ROOT)/.env.$(AIC_CLUSTER))
# Keys the file sets, so the site-specific defaults further down can tell
# "the user chose this" from "the Makefile defaulted it".
_AIC_ENV_KEYS := $(if $(wildcard $(AIC_ENV_FILE)),$(shell sed -n \
    's/^[[:space:]]*\(export[[:space:]][[:space:]]*\)\{0,1\}\([A-Za-z_][A-Za-z0-9_]*\)[[:space:]]*=.*/\2/p' \
    $(AIC_ENV_FILE)))
-include $(AIC_ENV_FILE)
$(foreach _k,$(_AIC_ENV_KEYS),$(eval export $(_k)))
export AIC_CLUSTER AIC_ENV_FILE

# "file" when VAR still holds a Makefile default, empty once the user has set
# it (command line, environment, or .env.<cluster>).  `$(origin VAR)` alone
# cannot tell the last case apart from a default -- an included makefile is
# also origin "file" -- so the env file's keys are subtracted explicitly.
_is_makefile_default = $(if $(filter $(1),$(_AIC_ENV_KEYS)),,$(filter file,$(origin $(1))))

IMAGE_NAME       ?= rocm-aic
VLLM_IMAGE_NAME  ?= aic-vllm
LMCACHE_IMAGE_NAME ?= aic-lmcache
override AIC_VERSION := $(strip $(file <$(REPO_ROOT)/VERSION))

# Detect ROCM_ARCH early (before _IMAGE_TAG) so the tag script can decide
# whether to include FlashAttention (CDNA-only).  Command-line overrides win.
_ROCM_ARCH_DETECTED := $(shell rocm_agent_enumerator 2>/dev/null | grep -E '^gfx' | head -1)
ROCM_ARCH := $(if $(strip $(ROCM_ARCH)),$(strip $(ROCM_ARCH)),$(_ROCM_ARCH_DETECTED))

# ROCM_VERSION has a single source of truth: the ARG in docker/base/Dockerfile.
# It is read from there and forwarded to all three image builds, so the vllm and
# lmcache images cannot drift from the base they are built on.  Must be defined
# before _FRAMEWORK_VERSION_ENV below, which bakes it into the image tag.
# Simply-expanded (:=) on purpose: _FRAMEWORK_VERSION_ENV uses
# $(value ROCM_VERSION), which would otherwise hand on the literal $(shell ...)
# text instead of the version string.
ifeq ($(origin ROCM_VERSION),undefined)
ROCM_VERSION := $(shell sed -n 's/^ARG ROCM_VERSION=//p' $(REPO_ROOT)/docker/base/Dockerfile | head -1)
endif

# LMCACHE_REF likewise: docker/lmcache/Dockerfile owns the pin (it is what the
# image tag and the CI pin check read), and it is forwarded to the vllm build
# too.  The vllm image installs the lmcache client that provides
# LMCacheMPConnector, and the MP wire protocol has no version negotiation, so
# client and server must come from the same release.  Also feeds _LMCACHE_VER
# in mk/bench.mk.  Simply-expanded for the same reason as ROCM_VERSION.
ifeq ($(origin LMCACHE_REF),undefined)
LMCACHE_REF := $(shell sed -n 's/^ARG LMCACHE_REF=//p' $(REPO_ROOT)/docker/lmcache/Dockerfile | head -1)
endif

# PyTorch, vLLM, and all deps are source-built; PYTORCH_REF, VLLM_REF, and
# LLM_EMU_REF are the key version knobs.  hipFile ships in the ROCm base image.
_FRAMEWORK_VERSION_ARGS := AIC_VERSION ROCM_VERSION PYTORCH_REF PYTORCH_SERIES TORCHVISION_REF VLLM_REF LLM_EMU_REF AITER_REF FLASH_ATTN_REF LMCACHE_REF NIXL_REF HSA_SNOOP_REF
# Repository URLs for the same frameworks.  Forwarded exactly like the version
# pins so a build can be pointed at a fork without editing a Dockerfile; the
# Dockerfile ARGs remain the source of truth for the defaults.  These do not
# affect the image tag, so they stay out of _FRAMEWORK_VERSION_ENV.
_FRAMEWORK_URL_ARGS := PYTORCH_GIT_URL TORCHVISION_GIT_URL VLLM_GIT_URL LLM_EMU_GIT_URL \
                       AITER_GIT_URL FLASH_ATTN_GIT_URL LMCACHE_GIT_URL NIXL_GIT_URL HSA_SNOOP_GIT_URL
_FRAMEWORK_ARGS := $(_FRAMEWORK_VERSION_ARGS) $(_FRAMEWORK_URL_ARGS)

_single_quote := '
_shell_quote = '$(subst $(_single_quote),'"'"',$(1))'
_FRAMEWORK_VERSION_ENV := $(foreach _arg,$(_FRAMEWORK_VERSION_ARGS),$(if $(filter undefined,$(origin $(_arg))),,$(_arg)=$(call _shell_quote,$(value $(_arg)))))

# ROCM_ARCH is shell-quoted like the framework args: an unquoted ';'-list ends
# the command mid-assignment, the tag script never runs, and every multi-arch
# image silently collides on :latest.  stderr stays visible so a genuine
# tagging failure is not mistaken for "latest".
_IMAGE_TAG := $(shell ROCM_ARCH=$(call _shell_quote,$(ROCM_ARCH)) $(_FRAMEWORK_VERSION_ENV) $(REPO_ROOT)/docker/scripts/aic-image-tag.sh)
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
# lmcache --max-gpu-workers: parallel STORE/RETRIEVE worker threads.  1 serializes
# all clients onto one thread (AffinityThreadPool warns), so under concurrent load
# stores back up behind L1->L2 eviction and trip vLLM's health check into degraded
# mode.  Default 4 to sustain bench-serve / multi-client load; drop to 1 for a
# single-client run to save GPU memory.
LMCACHE_MAX_GPU_WORKERS ?= 4

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
# Empty = resolve the running vLLM's published URL (see `make endpoints`).
BENCH_ENDPOINT    ?=
BENCH_MODEL       ?= $(VLLM_MODEL)

# Host-side URL of a published container port: $(call aic_url,<container>,<port>).
# Prints http://addr:port, or a message and a non-zero status.  Host ports are
# ephemeral, so resolve after `up` and again after any restart.
aic_url = bash "$(CURDIR)/scripts/aic-endpoints.sh" url $(1) $(2)

# Readiness probe from inside the container: $(call aic_probe,<container>,<port>,<path>).
# Exits non-zero until the server answers; needs no host port.
aic_probe = docker exec $(1) python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$(2)$(3)', timeout=3)" >/dev/null 2>&1

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

# ---- BuildKit builder + layer cache ----------------------------------------
# One builder name and one cache dir cover both the local `build` target and the
# distributed build in .slurm/run-build-distribute.sh.  AIC_CACHE_DIR is set
# further down (SPUR block); empty means "no layer cache".
export AIC_BUILDX_BUILDER ?= aic-local
export AIC_CACHE_MODE     ?= max

# Per-arch cache subdir slug; mirrors _arch_tag() in run-build-distribute.sh.
# Recursive (=) on purpose: the SPUR block below may override ROCM_ARCH.
empty :=
space := $(empty) $(empty)
_ARCH_SLUG = $(subst $(comma),-,$(subst ;,-,$(subst $(space),-,$(ROCM_ARCH))))

export AIC_VERSION ROCM_ARCH GPU GDS_SLAB_DATA LOG HF_HOME HF_TOKEN IMAGE_NAME IMAGE_REF IMAGE_TAG BUILD_JOBS
export VLLM_IMAGE_NAME VLLM_IMAGE_REF LMCACHE_IMAGE_NAME LMCACHE_IMAGE_REF
export LMCACHE_PORT LMCACHE_L1_SIZE_GB LMCACHE_NVME_POOL LMCACHE_NVME_SLOT_SIZE LMCACHE_NFS_POOL
export LMCACHE_MAX_GPU_WORKERS
export NVME_DATA NFS_DATA
export VLLM_MODEL TENSOR_PARALLEL_SIZE
export VLM_GPU_MEMORY_UTILIZATION VLM_MAX_MODEL_LEN VLM_MAX_NUM_BATCHED_TOKENS VLM_BLOCK_SIZE
export KVBENCH_GIT_URL KVBENCH_REF KVBENCH_IMAGE_REF KVBENCH_MODEL KVBENCH_GPU_PROFILE
export KVBENCH_PORT KVBENCH_HOST_PORT KVBENCH_CHUNK_SIZE KVBENCH_LOCAL_CPU_SIZE_GB

# Export every framework pin the caller actually overrode, so that
# .slurm/run-build-distribute.sh and docker/scripts/aic-image-tag.sh — both of
# which test `[[ -v NAME ]]` — see it.  Never export an empty value: an empty
# build arg overrides the pinned Dockerfile default rather than falling back to
# it (an empty AITER_REF, for example, produces an invalid AITER release URL).
# The Dockerfile ARGs remain the single source of truth for the pins themselves;
# this only forwards deliberate overrides.
$(foreach _v,$(_FRAMEWORK_ARGS),$(if $(strip $(value $(_v))),$(eval export $(_v))))

comma := ,
_COMPOSE_BIN := docker compose
COMPOSE      := $(_FRAMEWORK_VERSION_ENV) DOCKER_BUILDKIT=1 $(_COMPOSE_BIN) -f "$(CURDIR)/docker/docker-compose.yml"
COMPOSE_CACHE := $(COMPOSE) --profile cache

COMPOSE_PLUGIN_VERSION ?= v5.5.1

# vLLM --kv-transfer-config for the MP connector.  Leave KV_TRANSFER_ARG empty
# for a plain (baseline) vLLM.
_MP_CONNECTOR_JSON := {"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":$(LMCACHE_PORT),"lmcache.mp.lazy_offload":true,"lmcache.mp.lazy_offload_policy":"EVICTION_AWARE","lmcache.mp.lazy_offload_max_deferral_seconds":30.0,"lmcache.mp.save_decode_cache":true}}
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

# ---- Fabric exporters (nvme_exporter / rdma_exporter) + hsa-snoop sidecar --
NVME_EXPORTER_IMAGE   ?= aic-nvme-exporter:local
RDMA_EXPORTER_IMAGE   ?= aic-rdma-exporter:local
HSA_SNOOP_IMAGE_NAME  ?= aic-hsa-snoop
HSA_SNOOP_IMAGE_REF   := $(HSA_SNOOP_IMAGE_NAME):$(IMAGE_TAG)
HSA_SNOOP_IMAGE       ?= $(HSA_SNOOP_IMAGE_REF)
export HSA_SNOOP_IMAGE_NAME HSA_SNOOP_IMAGE_REF HSA_SNOOP_IMAGE
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
export AIC_IMAGE_DIR        ?= $(AIC_SHARED_NFS)/rocm-aic/images
export AIC_CACHE_DIR        ?= $(AIC_SHARED_NFS)/$(USER)/buildcache
override export NVME_DATA     := /mnt/m2m_nobackup/aic-cliff/nvme
override export GDS_SLAB_DATA := /mnt/m2m_nobackup/aic-cliff/slab
override export HF_HOME       := $(AIC_SHARED_NFS)/huggingface
override export ROCM_ARCH     := gfx950
export AIC_L2_BACKEND         ?= nixl_posix
export LMCACHE_NIXL_POSIX_SLOT_SIZE    ?= 33554432
export LMCACHE_NIXL_POSIX_POOL         ?= 32768
export LMCACHE_NIXL_DYNAMIC_MAX_GB     ?= 100
export LMCACHE_NIXL_DYNAMIC_DIRECT_IO ?= false
override export AIC_EXPORTERS            := safe
override export AIC_HSA_SNOOP_PID_MODE  := container:aic-lmcache
else
# Empty by default off-SPUR: a workstation has no /scratch, and an unwritable
# cache dir is worse than none.  Set AIC_CACHE_DIR explicitly to enable the
# BuildKit layer cache (e.g. a non-SPUR cluster with shared /scratch).
export AIC_CACHE_DIR        ?=
ifneq ($(wildcard /scratch/models/hub),)
ifeq ($(call _is_makefile_default,HF_HOME),file)
override export HF_HOME     := /scratch/models
_HF_HOME_IS_DEFAULT := 1
endif
endif
endif
_HF_HOME_IS_DEFAULT ?= $(if $(call _is_makefile_default,HF_HOME),1,)

# Strip Makefile-defaulted path vars before cliff sbatch so they don't clobber
# the job's node-appropriate defaults.  User-set values (command-line /
# environment / .env.<cluster>) are preserved.
_CLIFF_STRIP := env \
    $(if $(filter 0,$(AIC_SPUR_CLUSTER)),$(if $(_HF_HOME_IS_DEFAULT),-u HF_HOME)) \
    $(if $(filter 0,$(AIC_SPUR_CLUSTER)),$(if $(call _is_makefile_default,NVME_DATA),-u NVME_DATA)) \
    $(if $(call _is_makefile_default,NFS_DATA),-u NFS_DATA) \
    $(if $(filter 0,$(AIC_SPUR_CLUSTER)),$(if $(call _is_makefile_default,GDS_SLAB_DATA),-u GDS_SLAB_DATA)) \
    $(if $(call _is_makefile_default,AIC_METRICS_DIR),-u AIC_METRICS_DIR)

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
	@echo "  make build             Three-stage buildx build, no Slurm needed:"
	@echo "                           1. aic-base:$(IMAGE_TAG) (torch + torchvision)"
	@echo "                           2. $(VLLM_IMAGE_REF)"
	@echo "                           3. $(LMCACHE_IMAGE_REF)"
	@echo "                         Stages 2-3 inherit aic-base via an OCI build-context."
	@echo "                         AIC_CACHE_DIR=$(if $(strip $(AIC_CACHE_DIR)),$(AIC_CACHE_DIR),<empty: no layer cache>)"
	@echo "                           Set it to keep BuildKit layers across docker system prune."
	@echo "                           Cached per arch under <dir>/$(_ARCH_SLUG)/; mode=$(AIC_CACHE_MODE)."
	@echo "                         AIC_BUILDX_BUILDER=$(AIC_BUILDX_BUILDER)"
	@echo "                         TMPDIR holds a multi-GB base OCI layout — keep it off tmpfs."
	@echo "  make wheels            Export the nightly-release pip wheels to WHEELS_DIR=$(WHEELS_DIR):"
	@echo "                           vllm + amd_aiter (+ flash_attn on CDNA), lmcache, and the"
	@echo "                           single common nixl_rocm wheel built once in aic-base."
	@echo "                         Builds no images; see docs/PIP_WHEELS.md."
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
	@echo "  make endpoints         Print the URL of every running service (host ports are ephemeral)"
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
	@echo "  (dist-build/smoke-test submit via sbatch and log to logs/<job-id>/)"
	@echo "  make dist-build        Build base, then vllm+lmcache in parallel (docker buildx bake) + fabric"
	@echo "                         exporters on a Slurm build node, save tarballs"
	@echo "  make dist-wheels       Like 'make wheels' but on a Slurm node (what the nightly wheels CI runs)"
	@echo "  make dist-build-fast   Single-arch dev build (AIC_FAST_ARCH=$(AIC_FAST_ARCH)) -- faster iteration"
	@echo "  make dist-build-emulate  Build the CPU-only emulation image (no GPU kernels compiled)"
	@echo "  make dist-build-base   Build aic-base image (PyTorch + torchvision)"
	@echo "  make dist-build-vllm   Build aic-vllm image (requires AIC_BUILD_CONTEXT_BASE and a"
	@echo "                         'lmcache-wheels' build context -- see docker/vllm/Dockerfile)"
	@echo "  make dist-build-lmcache  Build aic-lmcache image (requires aic-base)"
	@echo "  make dist-build-exporters  Build ONLY the nvme/rdma exporter images (no main rebuild)"
	@echo "  make dist-build-monitoring Pull + save Prometheus/amdgpu-exporter to AIC_IMAGE_DIR"
	@echo "  make dist-push         Tag + push the built images (needs AIC_VLLM_PUSH_REF + AIC_LMCACHE_PUSH_REF)"
	@echo "  make smoke-test        Load + smoke-test the image on a GPU+NVMe node"
	@echo "  make smoke-test-fast   Smoke-test the single-arch dev image (AIC_FAST_ARCH=$(AIC_FAST_ARCH))"
	@echo "  make tiny-test         End-to-end serve check (MP stack + tiny model, one completion)"
	@echo "  make tiny-test-fast    Fast variant of tiny-test"
	@echo "  make test-rocjitsu-local  Boot a QEMU VM with emulated gfx1250 (rocjitsu), build+load the gfx1250 image, run a test completion (requires /dev/kvm)"
	@echo "                         Reuses matching gfx1250 images if present; RJ_FORCE_BUILD=1 rebuilds."
	@echo "                         Set AIC_CACHE_DIR to make a rebuild incremental."
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
	@echo "    AIC_CACHE_DIR=$(if $(strip $(AIC_CACHE_DIR)),$(AIC_CACHE_DIR),<empty: no cache>)  (shared BuildKit cache; empty disables)"
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
	@echo "  LMCACHE_MAX_GPU_WORKERS=$(LMCACHE_MAX_GPU_WORKERS)  (parallel store/retrieve threads; raise under concurrent load)"
	@echo "  LMCACHE_NVME_POOL=$(LMCACHE_NVME_POOL)  LMCACHE_NFS_POOL=$(LMCACHE_NFS_POOL)"
	@echo ""
	@echo "Examples:"
	@echo "  make build"
	@echo "  make build ROCM_ARCH=gfx1201 AIC_CACHE_DIR=~/.cache/rocm-aic-buildx  # persistent layer cache"
	@echo "  make build BUILD_JOBS=3          # cap parallelism on low-RAM hosts"
	@echo "  make up HF_TOKEN=hf_... NVME_DATA=/mnt/nvme NFS_DATA=/mnt/nfs"
	@echo "  make up-gds-l1 GDS_SLAB_DATA=/mnt/nvme HF_TOKEN=hf_..."
	@echo "  make cliff BENCH_ARM=vram_only                  # endpoint resolved from the running vLLM"
	@echo "  make cliff BENCH_ARM=kvbench BENCH_ENDPOINT=<url from 'make endpoints'>"
	@echo "  make cliff-kvbench-local KVBENCH_MODEL=llama-3.1-8b BENCH_CONCUR=1,2"
	@echo "  make plot"
	@echo ""
