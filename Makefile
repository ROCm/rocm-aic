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
# mode.  Default 4 for multi-client load; drop to 1 for a single-client run to
# save GPU memory.
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
# Empty = the running tiered vLLM, by name on the aic bridge (`make cliff` runs
# in aic-client); e.g. http://aic-vllm-only-gpu0:8000 for the vram_only arm.
BENCH_ENDPOINT    ?=
BENCH_MODEL       ?= $(VLLM_MODEL)

# Readiness probe from inside the container: $(call aic_probe,<container>,<port>,<path>).
# Exits non-zero until the server answers; needs no host port.
aic_probe = bash "$(CURDIR)/scripts/aic-endpoints.sh" probe $(1) $(2) $(3)

# URL of a container port as seen from the aic bridge, where containerized
# clients (aic-client, aic_fetch) run: $(call aic_bridge_url,<container>,<port>).
aic_bridge_url = bash "$(CURDIR)/scripts/aic-endpoints.sh" bridge-url $(1) $(2)

# One request from a throwaway container on the aic bridge, printing the body:
# $(aic_fetch) <url> [<json-body>].  Not a $(call): a JSON body has commas.
aic_fetch = bash "$(CURDIR)/scripts/aic-endpoints.sh" fetch

# ---- KVBench container -----------------------------------------------------
KVBENCH_GIT_URL           ?= https://github.com/wvaske/llm-kv-passthrough.git
KVBENCH_REF               ?= 177cf4e7c46ece8e03a701f89b60cd3a329f45fd
KVBENCH_IMAGE_NAME        ?= aic-kvbench
KVBENCH_IMAGE_TAG         ?= 177cf4e
KVBENCH_IMAGE_REF         ?= $(KVBENCH_IMAGE_NAME):$(KVBENCH_IMAGE_TAG)
KVBENCH_MODEL             ?= llama-3.1-8b
KVBENCH_GPU_PROFILE       ?= H100_SXM
KVBENCH_PORT              ?= 8000
KVBENCH_CHUNK_SIZE        ?= 256
KVBENCH_LOCAL_CPU_SIZE_GB ?= 0.25
KVBENCH_READY_S           ?= 120

# Non-Slurm runs land under logs/manual/ — keeps the tree root free of results/ and plots/.
# ?= and absolute, matching LOG above.  As a relative `:=` this silently
# ignored BENCH_LOGDIR from the environment and resolved against make's CWD,
# so prep-dirs tried to mkdir <cwd>/logs/manual/{results,plots} -- which fails
# wherever the tree is not writable, e.g. the 9p-mounted repo inside the
# rocjitsu guest.  A command-line override worked; an exported one did not.
BENCH_LOGDIR      ?= $(CURDIR)/logs/manual
BENCH_OUT         := $(BENCH_LOGDIR)/results/cliff-$(BENCH_ARM)-$(shell date +%Y%m%d-%H%M%S).csv

# ---- Build parallelism -----------------------------------------------------
BUILD_JOBS ?=

# ---- BuildKit builder + layer cache ----------------------------------------
# One builder name and one cache dir cover both the local `build` target and the
# distributed build in .slurm/run-build-distribute.sh.  AIC_CACHE_DIR is set
# further down (SPUR block); empty means "no layer cache".
export AIC_BUILDX_BUILDER ?= aic-local
export AIC_CACHE_MODE     ?= max
# Registry layer cache.  Takes precedence over AIC_CACHE_DIR in the bake HCL and
# in run-build-distribute.sh, which both already honoured it -- `make build` did
# not forward it, so CI had no way to share layers between runners without a
# shared filesystem.  Empty means "fall back to AIC_CACHE_DIR".
export AIC_CACHE_REF      ?=

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
export KVBENCH_PORT KVBENCH_CHUNK_SIZE KVBENCH_LOCAL_CPU_SIZE_GB

# Export every framework pin the caller actually overrode, so that
# .slurm/run-build-distribute.sh and docker/scripts/aic-image-tag.sh — both of
# which test `[[ -v NAME ]]` — see it.  Never export an empty value: an empty
# build arg overrides the pinned Dockerfile default rather than falling back to
# it (an empty AITER_REF, for example, produces an invalid AITER release URL).
# The Dockerfile ARGs remain the single source of truth for the pins themselves;
# this only forwards deliberate overrides.
$(foreach _v,$(_FRAMEWORK_ARGS),$(if $(strip $(value $(_v))),$(eval export $(_v))))

comma := ,
_AIC_B := \033[1m
_AIC_X := \033[36m
_AIC_G := \033[32m
_AIC_D := \033[90m
_AIC_R := \033[0m
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
AIC_GRAFANA_IMAGE ?= grafana/grafana:13.2.2
export AIC_METRICS_DIR AIC_GRAFANA_IMAGE

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
export AIC_SPUR_PARTITION   ?= amd-spur-v2
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

.PHONY: help list-vars
.DEFAULT_GOAL := help

help:
	@printf '$(_AIC_B)rocm-aic$(_AIC_R) — AMD Infinity Context inference stack + benchmarks  v$(AIC_VERSION)\n'
	@printf '\n$(_AIC_B)$(_AIC_X)BUILD (local)$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-32s$(_AIC_R) %s\n' \
	    'make build' 'Build aic-base → aic-vllm + aic-lmcache  (docker buildx bake)' \
	    'make wheels' 'Export pip wheels (nixl, vllm, amd_aiter, lmcache) → WHEELS_DIR' \
	    'make venv' 'Create/update .venv with bench+plot deps' \
	    'make ensure-compose' 'Install docker compose v2 plugin if missing'
	@printf '\n$(_AIC_B)$(_AIC_X)BUILD (Slurm/Spur)$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-32s$(_AIC_R) %s\n' \
	    'make dist-build' 'Build aic-base then vllm + lmcache in parallel, save tarballs' \
	    'make dist-build-fast' 'Single-arch (AIC_FAST_ARCH=$(AIC_FAST_ARCH)) dev build' \
	    'make dist-build-emulate' 'Build CPU-only emulation image (no GPU kernels compiled)' \
	    'make dist-build-exporters' 'Build only nvme/rdma exporter images (no main rebuild)' \
	    'make dist-build-monitoring' 'Pull + save Prometheus/amdgpu-exporter to AIC_IMAGE_DIR' \
	    'make dist-wheels' 'Export pip wheels on a Slurm node (nightly wheels CI)' \
	    'make dist-push' 'Tag + push images (needs AIC_VLLM_PUSH_REF + AIC_LMCACHE_PUSH_REF)'
	@printf '\n$(_AIC_B)$(_AIC_X)STACK (local)$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-32s$(_AIC_R) %s\n' \
	    'make up' 'Start lmcache + vllm + monitoring (foreground, DRAM L1)' \
	    'make up-batch' 'Start lmcache + vllm + monitoring (background)' \
	    'make up-dev' 'Start in --enforce-eager mode (~60 s faster, ~10% slower inference)' \
	    'make up-gds-l1' 'Start with hipFile GDS NVMe slab as L1 (foreground)' \
	    'make up-gds-l1-batch' 'Start with hipFile GDS NVMe slab as L1 (background)' \
	    'make down' 'Stop and remove all containers' \
	    'make logs' 'Follow logs from all containers' \
	    'make logs-lmcache' 'Follow lmcache logs only' \
	    'make logs-vllm' 'Follow vllm logs only' \
	    'make ps' 'Show container status' \
	    'make endpoints' 'Print host URL of every published service (Grafana only)' \
	    'make shell-lmcache' 'Exec bash into aic-lmcache container' \
	    'make shell-vllm' 'Exec bash into aic-vllm-gpu$(GPU) container' \
	    'make restart-vllm' 'Restart vllm only (lmcache + warm KV cache preserved)' \
	    'make restart-lmcache' 'Restart lmcache'
	@printf '\n$(_AIC_B)$(_AIC_X)BENCHMARK (local)$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-32s$(_AIC_R) %s\n' \
	    'make vllm-reset-test' 'Verify LMCache L1+L2 KV retrieval end-to-end' \
	    'make cliff' 'Run KV-cache cliff benchmark, write CSV to BENCH_LOGDIR/results/' \
	    'make plot' 'Generate cliff PNG charts from BENCH_LOGDIR/results/ CSVs' \
	    'make kvbench-up' 'Start KVBench + client on internal compose network' \
	    'make kvbench-down' 'Stop the KVBench compose profile' \
	    'make cliff-kvbench-local' 'Start KVBench, wait for readiness, run kvbench cliff arm'
	@printf '\n$(_AIC_B)$(_AIC_X)EMULATE & PROFILE (local)$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-32s$(_AIC_R) %s\n' \
	    'make test-emulate-local' 'Bring up vllm-emulator, assert completion + hook active' \
	    'make capture-profile-local' 'Capture profile pack from real GPU serve (requires /dev/kfd)' \
	    'make test-rocjitsu-local' 'Boot QEMU VM with emulated gfx1250, run vllm-reset-test'
	@printf '\n$(_AIC_B)$(_AIC_X)TEST (Slurm)$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-32s$(_AIC_R) %s\n' \
	    'make smoke-test' 'Load + smoke-test image on a GPU+NVMe Slurm node' \
	    'make smoke-test-fast' 'Smoke-test single-arch (AIC_FAST_ARCH=$(AIC_FAST_ARCH)) image' \
	    'make tiny-test' 'End-to-end serve check (MP stack, tiny model, one completion)' \
	    'make tiny-test-fast' 'tiny-test pinned to AIC_FAST_ARCH' \
	    'make emulate-test' 'Serve check on a CPU-only Slurm node (emulation image)' \
	    'make emulate-mp-test' 'Emulation + full LMCache MP connector recipe on CPU node' \
	    'make emulate-validate' 'Replay captured pack on CPU, diff vs real-hardware run' \
	    'make profile-capture' 'Capture profile pack from real GPU serve on a Slurm node' \
	    'make accuracy-test' 'KV-integrity gate: differential lm_eval, two arms' \
	    'make accuracy-test-fast' 'accuracy-test pinned to AIC_FAST_ARCH' \
	    'make accuracy-test-very-fast' '100-item cap per arm (~25 min); floor widened 3× SE' \
	    'make prometheus-dump' 'Submit SPUR job: scrape all /metrics → Markdown reference doc'
	@printf '\n$(_AIC_B)$(_AIC_X)CLIFF SWEEPS (Slurm)$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-32s$(_AIC_R) %s\n' \
	    'make cliff-submit' 'Full 3-arm sweep (waits for job): vram+nvme+gds, c=1..250' \
	    'make cliff-kvbench-submit' 'sbatch CPU-only KVBench cliff run' \
	    'make cliff-kvd' 'Focused KVD cliff: shared prefix, sparse c=1,8,32,64,128,250' \
	    'make cliff-spur-l2' 'SPUR L2 cliff: per_client, util=0.40, 8 GB DRAM L1, nvme only' \
	    'make cliff-short' '1-point cliff (c=1, 1 iter) to smoke-test the job flow' \
	    'make cliff-short-fast' 'cliff-short on the single-arch (AIC_FAST_ARCH) image'
	@printf '\n$(_AIC_B)$(_AIC_X)UTILITIES$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-32s$(_AIC_R) %s\n' \
	    'make export' 'Tarball working-tree sources (tracked + local edits)' \
	    'make install-ci-scripts' 'Deploy .github/scripts/runners/*.sh to AIC_CI_LIB_DIR' \
	    'make list-vars' 'Show current values of all configurable environment variables'
	@printf '\n$(_AIC_B)$(_AIC_X)EXAMPLES$(_AIC_R)\n'
	@printf '  %s\n' \
	    'make build ROCM_ARCH=gfx1201 AIC_CACHE_DIR=~/.cache/aic-buildx' \
	    'make build BUILD_JOBS=3   # cap parallelism on low-RAM hosts' \
	    'make up VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct HF_TOKEN=hf_... NVME_DATA=/mnt/nvme' \
	    'make up-gds-l1 GDS_SLAB_DATA=/mnt/nvme HF_TOKEN=hf_...' \
	    'make cliff BENCH_ARM=vram_only BENCH_MODEL=Qwen/Qwen2.5-7B-Instruct' \
	    'make cliff-kvbench-local KVBENCH_MODEL=llama-3.1-8b BENCH_CONCUR=1,2' \
	    'make dist-build dist-push smoke-test   # the nightly build chain'
	@printf '\n'

list-vars:
	@printf '$(_AIC_B)rocm-aic environment variables$(_AIC_R)\n'
	@printf '$(_AIC_D)Override any of these on the make command line or via .env.<cluster>$(_AIC_R)\n'
	@printf '\n$(_AIC_B)$(_AIC_X)IMAGE / BUILD$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'VLLM_IMAGE_NAME' '$(VLLM_IMAGE_NAME)' \
	    'LMCACHE_IMAGE_NAME' '$(LMCACHE_IMAGE_NAME)' \
	    'VLLM_IMAGE_REF' '$(VLLM_IMAGE_REF)' \
	    'LMCACHE_IMAGE_REF' '$(LMCACHE_IMAGE_REF)' \
	    'ROCM_ARCH' '$(or $(ROCM_ARCH),<not detected>)' \
	    'ROCM_VERSION' '$(ROCM_VERSION)' \
	    'AIC_VERSION' '$(AIC_VERSION)' \
	    'BUILD_JOBS' '$(or $(BUILD_JOBS),<nproc>)' \
	    'BUILD_PROGRESS' '$(BUILD_PROGRESS)' \
	    'BAKE_TARGETS' '$(BAKE_TARGETS)' \
	    'BAKE_OUTPUT_FLAG' '$(BAKE_OUTPUT_FLAG)' \
	    'AIC_FAST_ARCH' '$(AIC_FAST_ARCH)' \
	    'WHEELS_DIR' '$(WHEELS_DIR)'
	@printf '\n$(_AIC_B)$(_AIC_X)LAYER CACHE$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'AIC_BUILDX_BUILDER' '$(AIC_BUILDX_BUILDER)' \
	    'AIC_CACHE_MODE' '$(AIC_CACHE_MODE)' \
	    'AIC_CACHE_REF' '$(or $(AIC_CACHE_REF),<empty>)' \
	    'AIC_CACHE_DIR' '$(or $(AIC_CACHE_DIR),<empty: no local cache>)'
	@printf '\n$(_AIC_B)$(_AIC_X)CLUSTER / SITE$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'AIC_CLUSTER' '$(or $(AIC_CLUSTER),<not detected>)' \
	    'AIC_ENV_FILE' '$(or $(AIC_ENV_FILE),<none>)' \
	    'AIC_SPUR_CLUSTER' '$(AIC_SPUR_CLUSTER)' \
	    'AIC_SPUR_PARTITION' '$(or $(AIC_SPUR_PARTITION),<not set>)' \
	    'AIC_SHARED_NFS' '$(or $(AIC_SHARED_NFS),<not set>)' \
	    'AIC_IMAGE_DIR' '$(or $(AIC_IMAGE_DIR),<not set>)'
	@printf '\n$(_AIC_B)$(_AIC_X)STORAGE$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'NVME_DATA' '$(NVME_DATA)' \
	    'NFS_DATA' '$(NFS_DATA)' \
	    'GDS_SLAB_DATA' '$(or $(GDS_SLAB_DATA),<empty: GDS disabled>)' \
	    'GPU' '$(GPU)'
	@printf '\n$(_AIC_B)$(_AIC_X)HUGGING FACE$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'HF_HOME' '$(HF_HOME)' \
	    'HF_TOKEN_FILE' '$(or $(HF_TOKEN_FILE),<not set>)' \
	    'HF_TOKEN' '$(if $(HF_TOKEN),<set>,<not set>)'
	@printf '\n$(_AIC_B)$(_AIC_X)VLLM$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'VLLM_MODEL' '$(or $(VLLM_MODEL),<not set>)' \
	    'VLLM_REF' '$(or $(VLLM_REF),<bake.hcl default>)' \
	    'TENSOR_PARALLEL_SIZE' '$(TENSOR_PARALLEL_SIZE)' \
	    'VLM_GPU_MEMORY_UTILIZATION' '$(or $(VLM_GPU_MEMORY_UTILIZATION),<vLLM default>)' \
	    'VLM_MAX_MODEL_LEN' '$(or $(VLM_MAX_MODEL_LEN),<vLLM default>)' \
	    'VLM_MAX_NUM_BATCHED_TOKENS' '$(or $(VLM_MAX_NUM_BATCHED_TOKENS),<vLLM default>)' \
	    'VLM_BLOCK_SIZE' '$(or $(VLM_BLOCK_SIZE),<vLLM default>)'
	@printf '\n$(_AIC_B)$(_AIC_X)LMCACHE$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'LMCACHE_REF' '$(LMCACHE_REF)' \
	    'LMCACHE_PORT' '$(LMCACHE_PORT)' \
	    'LMCACHE_L1_SIZE_GB' '$(LMCACHE_L1_SIZE_GB) GiB' \
	    'LMCACHE_MAX_GPU_WORKERS' '$(LMCACHE_MAX_GPU_WORKERS)' \
	    'LMCACHE_NVME_POOL' '$(LMCACHE_NVME_POOL)' \
	    'LMCACHE_NVME_SLOT_SIZE' '$(LMCACHE_NVME_SLOT_SIZE)' \
	    'LMCACHE_NFS_POOL' '$(LMCACHE_NFS_POOL)' \
	    'AIC_L2_BACKEND' '$(or $(AIC_L2_BACKEND),<not set>)'
	@printf '\n$(_AIC_B)$(_AIC_X)BENCHMARK (cliff)$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'BENCH_ARM' '$(BENCH_ARM)' \
	    'BENCH_ISL' '$(BENCH_ISL)' \
	    'BENCH_SHARED_TOK' '$(BENCH_SHARED_TOK)' \
	    'BENCH_CONCUR' '$(BENCH_CONCUR)' \
	    'BENCH_ITERS' '$(BENCH_ITERS)' \
	    'BENCH_ENDPOINT' '$(or $(BENCH_ENDPOINT),<running vLLM via aic bridge>)' \
	    'BENCH_MODEL' '$(or $(BENCH_MODEL),$(or $(VLLM_MODEL),<not set>))' \
	    'BENCH_LOGDIR' '$(BENCH_LOGDIR)'
	@printf '\n$(_AIC_B)$(_AIC_X)EMULATE & PROFILE$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'AIC_EMULATE_MODEL' '$(AIC_EMULATE_MODEL)' \
	    'AIC_EMULATE_READY_S' '$(AIC_EMULATE_READY_S) s' \
	    'AIC_CAPTURE_MODEL' '$(AIC_CAPTURE_MODEL)' \
	    'AIC_CAPTURE_DIR' '$(AIC_CAPTURE_DIR)' \
	    'AIC_CAPTURE_GPU' '$(AIC_CAPTURE_GPU)' \
	    'AIC_CAPTURE_GPU_UTIL' '$(AIC_CAPTURE_GPU_UTIL)'
	@printf '\n$(_AIC_B)$(_AIC_X)ROCJITSU VM (test-rocjitsu-local)$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'RJ_ROCJITSU_ARCH' '$(RJ_ROCJITSU_ARCH)' \
	    'RJ_MODEL' '$(RJ_MODEL)' \
	    'RJ_IMAGE_REF' '$(RJ_IMAGE_REF)' \
	    'RJ_IMAGE_PREFIX' '$(or $(RJ_IMAGE_PREFIX),<empty: build locally>)' \
	    'RJ_FORCE_BUILD' '$(RJ_FORCE_BUILD)' \
	    'RJ_VM_VCPUS / RJ_VM_MEM_MB' '$(RJ_VM_VCPUS) / $(RJ_VM_MEM_MB) MB' \
	    'RJ_WORK_DIR' '$(RJ_WORK_DIR)' \
	    'RJ_PROBE / RJ_PROBE_ONLY' '$(RJ_PROBE) / $(RJ_PROBE_ONLY)'
	@printf '\n$(_AIC_B)$(_AIC_X)KVBENCH$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'KVBENCH_IMAGE_REF' '$(KVBENCH_IMAGE_REF)' \
	    'KVBENCH_MODEL' '$(KVBENCH_MODEL)' \
	    'KVBENCH_PORT' '$(KVBENCH_PORT)' \
	    'KVBENCH_CHUNK_SIZE' '$(KVBENCH_CHUNK_SIZE)' \
	    'KVBENCH_LOCAL_CPU_SIZE_GB' '$(KVBENCH_LOCAL_CPU_SIZE_GB) GB' \
	    'KVBENCH_READY_S' '$(KVBENCH_READY_S) s'
	@printf '\n$(_AIC_B)$(_AIC_X)MONITORING$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'AIC_METRICS_DIR' '$(AIC_METRICS_DIR)' \
	    'AIC_EXPORTERS' '$(AIC_EXPORTERS)' \
	    'AIC_GRAFANA_IMAGE' '$(AIC_GRAFANA_IMAGE)'
	@printf '\n$(_AIC_B)$(_AIC_X)CLIFF (Slurm)$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'AIC_CLIFF_GFX' '$(or $(AIC_CLIFF_GFX),<empty>)' \
	    'AIC_CLIFF_CONSTRAINT' '$(AIC_CLIFF_CONSTRAINT)' \
	    'AIC_CLIFF_GPUS' '$(AIC_CLIFF_GPUS)' \
	    'AIC_CLIFF_NODE' '$(or $(AIC_CLIFF_NODE),<any>)' \
	    'AIC_CLIFF_TIME' '$(or $(AIC_CLIFF_TIME),<target default>)' \
	    'AIC_CLIFF_ARMS' '$(or $(AIC_CLIFF_ARMS),all)'
	@printf '\n$(_AIC_B)$(_AIC_X)LOGGING / EXPORT$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'LOG' '$(LOG)' \
	    'PROM_DUMP_OUT' '$(or $(PROM_DUMP_OUT),<AIC_IMAGE_DIR>/../prometheus-dump.md)' \
	    'PROM_DUMP_WAIT' '$(PROM_DUMP_WAIT) s' \
	    'EXPORT_PREFIX' '$(EXPORT_PREFIX)' \
	    'EXPORT_TARBALL' '$(notdir $(EXPORT_TARBALL))' \
	    'AIC_CI_LIB_DIR' '$(AIC_CI_LIB_DIR)'
	@printf '\n'
