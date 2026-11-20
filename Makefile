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

override AIC_VERSION := $(strip $(file <$(REPO_ROOT)/VERSION))

# ROCM_ARCH is the arch of this host's stack: the user's value, else the probed
# GPU, and always gfx950 on SPUR.  It is settled here, ahead of the bake query
# below, because the image tag carries it.  (The arch set a Slurm build compiles
# for is a separate variable, AIC_ROCM_ARCH.)
AIC_SPUR_CLUSTER ?= 0
_ROCM_ARCH_DETECTED := $(shell rocm_agent_enumerator 2>/dev/null | grep -E '^gfx' | head -1)
ROCM_ARCH := $(if $(strip $(ROCM_ARCH)),$(strip $(ROCM_ARCH)),$(_ROCM_ARCH_DETECTED))
ifeq ($(AIC_SPUR_CLUSTER),1)
_SPUR_ARCH_FORCED := $(if $(strip $(ROCM_ARCH)),,1)
override ROCM_ARCH := gfx950
endif

# PyTorch, vLLM, and all deps are source-built; PYTORCH_REF, VLLM_REF, and
# LLM_EMU_REF are the key version knobs.  hipFile ships in the ROCm base image.
_FRAMEWORK_VERSION_ARGS := AIC_VERSION ROCM_VERSION PYTORCH_REF PYTORCH_SERIES TORCHVISION_REF VLLM_REF LLM_EMU_REF AITER_REF FLASH_ATTN_REF LMCACHE_REF NIXL_REF HSA_SNOOP_REF
# Only the pins that shape the image tag are listed: make hands them to its own
# bake query.  Every other bake variable (the *_GIT_URLs, CLR_REF, ...)
# reaches bake and the Slurm script through the environment, as any variable set
# on the command line, in the environment or in .env.<cluster> does.

empty :=
space := $(empty) $(empty)
define _newline


endef
_single_quote := '
_shell_quote = '$(subst $(_single_quote),'"'"',$(1))'
# The listed variables make already holds, as shell assignments.  Passed
# explicitly because command-line variables only reach $(shell) on make >= 4.4.
_shell_env = $(foreach _arg,$(1),$(if $(filter undefined,$(origin $(_arg))),,$(_arg)=$(call _shell_quote,$(value $(_arg)))))
_framework_version_env = $(call _shell_env,$(_FRAMEWORK_VERSION_ARGS))

# docker/docker-bake.hcl owns every pin, the tag format and the image names.
# One `bake --print` per make run (via aic-bake-env.sh) resolves what make
# itself consumes, from the same inputs a later `make build` hands to bake, so
# the refs exported to compose are the strings bake tags.  The stack refs are
# assigned with `:=` (an inherited environment value is a parent make's, for
# possibly another arch); the rest keep `?=`.  Inputs are shell-quoted: an
# unquoted ';'-list in ROCM_ARCH would end the command mid-assignment.  stderr
# stays visible so a genuine failure is not mistaken for "latest".
_BAKE_INPUTS      := ROCM_ARCH IMAGE_TAG AIC_IMAGE_PREFIX KVBENCH_REF KVBENCH_IMAGE_TAG $(_FRAMEWORK_VERSION_ARGS)
_BAKE_KEYS_STACK  := VLLM_IMAGE_REF LMCACHE_IMAGE_REF HSA_SNOOP_IMAGE_REF
_BAKE_KEYS        := IMAGE_TAG ROCM_VERSION LMCACHE_REF KVBENCH_GIT_URL KVBENCH_REF KVBENCH_IMAGE_REF CLIENT_IMAGE_REF
_IMAGE_TAG_ORIGIN := $(origin IMAGE_TAG)
_AIC_BAKE_MK := $(shell $(call _shell_env,$(_BAKE_INPUTS)) \
    $(REPO_ROOT)/docker/scripts/aic-bake-env.sh --make | sed -n -E \
    -e 's/^($(subst $(space),|,$(_BAKE_KEYS_STACK))) \?= (.*)$$/\1 := \2@NL@/p' \
    -e 's/^($(subst $(space),|,$(_BAKE_KEYS))) \?= .*$$/&@NL@/p')
$(eval $(subst @NL@ ,$(_newline),$(_AIC_BAKE_MK) ))
# Without docker buildx (a Slurm login node submitting `make dist-build`) the
# helper prints nothing: the image refs stay unset so compose falls back to its
# own :latest defaults.  The Slurm build jobs resolve their pins from bake on
# the build node.
IMAGE_TAG ?= latest
_FRAMEWORK_VERSION_ENV := $(_framework_version_env)
# Without docker buildx IMAGE_TAG cannot shape the compose refs (bake owns the
# image names), so a user's IMAGE_TAG would silently start the :latest images.
ifeq ($(MAKELEVEL)$(filter undefined,$(_IMAGE_TAG_ORIGIN))$(strip $(VLLM_IMAGE_REF)),0)
ifneq ($(filter-out help list-vars dist-% cliff-%,$(or $(MAKECMDGOALS),help)),)
$(warning IMAGE_TAG=$(IMAGE_TAG) does not select the compose images here (docker buildx bake unavailable); pass VLLM_IMAGE_REF, LMCACHE_IMAGE_REF and HSA_SNOOP_IMAGE instead)
endif
endif
# The Slurm scripts name images and artifacts by the tag, and derive it
# themselves (no arch suffix) when run directly from a login node.  Hand them
# that same tag rather than one carrying the SPUR arch make forced above, so
# both entry points find each other's artifacts.  Settled by the top-level make;
# a sub-make inherits IMAGE_TAG and ROCM_ARCH and could not tell.
ifeq ($(MAKELEVEL),0)
_AIC_SLURM_IMAGE_TAG := $(if $(and $(_SPUR_ARCH_FORCED),$(filter undefined,$(_IMAGE_TAG_ORIGIN))),$(patsubst %-gfx950,%,$(IMAGE_TAG)))
else ifeq ($(origin IMAGE_TAG),command line)
_AIC_SLURM_IMAGE_TAG :=
endif
export _AIC_SLURM_IMAGE_TAG
_SLURM_TAG_ENV := $(if $(_AIC_SLURM_IMAGE_TAG),IMAGE_TAG=$(call _shell_quote,$(_AIC_SLURM_IMAGE_TAG)))

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

# ---- Container-runtime knobs -----------------------------------------------
# docker/docker-compose.yml owns the defaults of every knob left empty here.
# Make gives them none and does not re-export them: a value set on the command
# line, in the environment or in .env.<cluster> reaches compose and the Slurm
# scripts through the environment, and an unset one leaves compose's default in
# force.  The few with a value are the ones a recipe expands itself, and
# AIC_CONTAINER_USER, exported because compose's own default is root.
# RENDER_GID is /dev/kfd's group, which a non-root container needs; render's
# gid varies by host, and empty (no /dev/kfd) leaves compose's default.

export AIC_CONTAINER_USER ?= $(shell id -u):$(shell id -g)
export RENDER_GID ?= $(shell stat -c %g /dev/kfd 2>/dev/null)

# Grafana's host port: the one aic-grafana already holds, else the lowest free
# port from 3000, so the URL is the same after every down/up.  Left to compose,
# Docker picks a new ephemeral port on each recreate.  Empty (3000-3009 all
# taken) falls back to that.
ifeq ($(origin AIC_GRAFANA_PORT),undefined)
export AIC_GRAFANA_PORT := $(shell bash "$(CURDIR)/scripts/aic-endpoints.sh" pick-port aic-grafana 3000 3000 3009 2>/dev/null)
endif

# ---- LMCache server --------------------------------------------------------
LMCACHE_PORT            ?= 6555
LMCACHE_L1_SIZE_GB      ?=
LMCACHE_NVME_POOL       ?=
LMCACHE_NVME_SLOT_SIZE  ?=
LMCACHE_NFS_POOL        ?=
LMCACHE_MAX_GPU_WORKERS ?=

# ---- vLLM knobs ------------------------------------------------------------
VLLM_MODEL                  ?=
TENSOR_PARALLEL_SIZE        ?=
VLM_GPU_MEMORY_UTILIZATION  ?=
VLM_MAX_MODEL_LEN           ?=
VLM_MAX_NUM_BATCHED_TOKENS  ?=
VLM_BLOCK_SIZE              ?=
# vllm-reset-test: health polls (5 s apart) before giving up, and how much of
# the vLLM log to print when the container dies first.
VLM_READY_RETRIES           ?= 60
VLM_FAIL_LOG_LINES          ?= 250

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
# KVBENCH_GIT_URL, KVBENCH_REF and KVBENCH_IMAGE_REF come from bake (above).
KVBENCH_MODEL             ?= llama-3.1-8b
KVBENCH_GPU_PROFILE       ?=
KVBENCH_PORT              ?= 8000
KVBENCH_CHUNK_SIZE        ?=
KVBENCH_LOCAL_CPU_SIZE_GB ?=
KVBENCH_READY_S           ?= 120

# Non-Slurm runs land under logs/manual/ — keeps the tree root free of results/ and plots/.
# ?= and absolute, matching LOG above.  As a relative `:=` this silently
# ignored BENCH_LOGDIR from the environment and resolved against make's CWD,
# so prep-dirs tried to mkdir <cwd>/logs/manual/{results,plots} -- which fails
# wherever the tree is not writable, e.g. the 9p-mounted repo inside the
# rocjitsu guest.  A command-line override worked; an exported one did not.
BENCH_LOGDIR      ?= $(CURDIR)/logs/manual
BENCH_OUT         := $(BENCH_LOGDIR)/results/cliff-$(BENCH_ARM)-$(shell date +%Y%m%d-%H%M%S).csv

# ---- BuildKit builder + layer cache ----------------------------------------
# One builder name covers both the local `build` target and the distributed
# build in .slurm/run-build-distribute.sh.
#
# docker/docker-bake.hcl owns the defaults of the build knobs BUILD_JOBS,
# AIC_UCX_FAST, AIC_CACHE_MODE, AIC_CACHE_REF, AIC_CACHE_DIR and TLS_CERT, so
# make gives them none: a value set on the command line, in the environment or
# in .env.<cluster> reaches bake and the Slurm script through the environment,
# and an unset one leaves each to its own default.  (The one exception is the
# SPUR site default for AIC_CACHE_DIR further down.)
export AIC_BUILDX_BUILDER ?= aic-local

# AIC_TLS_CERT is the deprecated name of TLS_CERT; accepted for one release.
ifeq ($(strip $(TLS_CERT)),)
ifneq ($(strip $(AIC_TLS_CERT)),)
export TLS_CERT := $(AIC_TLS_CERT)
endif
endif

# Per-arch cache subdir slug; mirrors _arch_tag() in run-build-distribute.sh.
_ARCH_SLUG = $(subst $(comma),-,$(subst ;,-,$(subst $(space),-,$(ROCM_ARCH))))

export AIC_VERSION ROCM_ARCH GPU GDS_SLAB_DATA LOG HF_HOME HF_TOKEN IMAGE_TAG
export LMCACHE_PORT
export NVME_DATA NFS_DATA
export VLLM_MODEL
export VLM_GPU_MEMORY_UTILIZATION VLM_MAX_MODEL_LEN VLM_MAX_NUM_BATCHED_TOKENS VLM_BLOCK_SIZE
export KVBENCH_MODEL KVBENCH_PORT

comma := ,
_AIC_B := \033[1m
_AIC_X := \033[36m
_AIC_G := \033[32m
_AIC_D := \033[90m
_AIC_R := \033[0m
_COMPOSE_BIN := docker compose
COMPOSE      := $(_FRAMEWORK_VERSION_ENV) DOCKER_BUILDKIT=1 $(_COMPOSE_BIN) -f "$(CURDIR)/docker/docker-compose.yml"
COMPOSE_CACHE := $(COMPOSE) --profile cache

# vLLM --kv-transfer-config for the MP connector.  Leave KV_TRANSFER_ARG empty
# for a plain (baseline) vLLM.
# lmcache.mp.use_vmm_api=false selects legacy HIP IPC.  Left unset, LMCache
# >= 0.5.8 asks vLLM for fd-exported KV memory, which vLLM >= 0.34 refuses on
# ROCm (and with --num-gpu-blocks-override) at engine-config time.
_MP_CONNECTOR_JSON := {"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":$(LMCACHE_PORT),"lmcache.mp.use_vmm_api":false,"lmcache.mp.lazy_offload":true,"lmcache.mp.lazy_offload_policy":"EVICTION_AWARE","lmcache.mp.lazy_offload_max_deferral_seconds":30.0,"lmcache.mp.save_decode_cache":true}}
KV_TRANSFER_ARG    ?= --kv-transfer-config '$(_MP_CONNECTOR_JSON)'
export KV_TRANSFER_ARG

# ---- Metrics capture (Prometheus sidecar) ----------------------------------
# AIC_EXPORTERS belongs to monitoring/monitoring-lib.sh (the Slurm jobs); the
# local targets pick their exporters with compose profiles.  The Grafana image
# pin is compose's.
AIC_METRICS_DIR  ?= $(CURDIR)/logs/prometheus
export AIC_METRICS_DIR

# ---- Image refs for compose ------------------------------------------------
# HSA_SNOOP_IMAGE is the name compose reads; bake's tag is its default.
HSA_SNOOP_IMAGE       ?= $(HSA_SNOOP_IMAGE_REF)
AIC_CLIENT_IMAGE      ?= $(CLIENT_IMAGE_REF)
# What bake resolved, exported only when it did: an empty KVBENCH_REF in the
# environment would replace bake's own default in a Slurm-side build.
$(foreach _v,VLLM_IMAGE_REF LMCACHE_IMAGE_REF HSA_SNOOP_IMAGE KVBENCH_IMAGE_REF KVBENCH_GIT_URL KVBENCH_REF AIC_CLIENT_IMAGE,\
    $(if $(strip $($(_v))),$(eval export $(_v))))

PYTHON := $(if $(wildcard $(REPO_ROOT)/.venv/bin/python3),$(REPO_ROOT)/.venv/bin/python3,python3)

# ---- Distribute / cliff (Slurm) --------------------------------------------
DIST := $(CURDIR)/.slurm/run-build-distribute.sh
DIST_RUN = $(_SLURM_TAG_ENV) "$(DIST)"

# ---- Self-hosted CI runner scripts -----------------------------------------
AIC_CI_LIB_DIR    ?= /usr/local/lib/aic-ci
AIC_CI_SCRIPT_DIR := $(CURDIR)/.github/scripts/runners

AIC_FAST_ARCH ?= gfx950

# ---- accuracy-test ---------------------------------------------------------
export AIC_ACCURACY_MODEL
export AIC_ACCURACY_TIME AIC_ACCURACY_CPUS AIC_ACCURACY_MEM
export AIC_ACCURACY_READY_TIMEOUT
export AIC_ACCURACY_LIMIT

# ---- SPUR site defaults ----------------------------------------------------
# Tracked, because every SPUR user needs them, but defaults all the same: a
# value from the command line, the environment or .env.<cluster> wins.  The
# three paths make already defaulted above take the _is_makefile_default guard
# (`?=` is a no-op on a variable that is already defined).
ifeq ($(AIC_SPUR_CLUSTER),1)
AIC_SHARED_NFS ?= /shared_nfs
export AIC_SPUR_CLUSTER
export AIC_SPUR_PARTITION   ?= amd-spur-v2
export AIC_IMAGE_DIR        ?= $(AIC_SHARED_NFS)/rocm-aic/images
export AIC_CACHE_DIR        ?= $(AIC_SHARED_NFS)/$(USER)/buildcache
ifeq ($(call _is_makefile_default,NVME_DATA),file)
export NVME_DATA := /mnt/m2m_nobackup/aic-cliff/nvme
endif
ifeq ($(call _is_makefile_default,GDS_SLAB_DATA),file)
export GDS_SLAB_DATA := /mnt/m2m_nobackup/aic-cliff/slab
endif
ifeq ($(call _is_makefile_default,HF_HOME),file)
export HF_HOME := $(AIC_SHARED_NFS)/huggingface
endif
export AIC_L2_BACKEND       ?= nixl_posix
# spur-authz refuses pid:host, which the full exporter fleet and hsa-snoop's
# host mode both need.
export AIC_EXPORTERS          ?= safe
export AIC_HSA_SNOOP_PID_MODE ?= container:aic-lmcache
else
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
# environment / .env.<cluster>) are preserved, and so are the SPUR site
# defaults above, which are the node-appropriate paths there.
#
# A recursive $(MAKE) (the *-fast targets) receives the parent's defaults as
# environment values, so the top-level make settles the list and hands it down;
# a sub-make keeps a name only while it still holds that inherited value.
ifeq ($(MAKELEVEL),0)
_AIC_DEFAULTED := $(strip $(if $(_HF_HOME_IS_DEFAULT),HF_HOME) \
    $(foreach v,NVME_DATA NFS_DATA GDS_SLAB_DATA AIC_METRICS_DIR,$(if $(call _is_makefile_default,$(v)),$(v))))
else
_AIC_DEFAULTED := $(strip $(foreach v,$(_AIC_DEFAULTED),$(if $(filter environment,$(origin $(v))),$(v))))
endif
export _AIC_DEFAULTED
_CLIFF_STRIP := env \
    $(if $(filter 0,$(AIC_SPUR_CLUSTER)),$(if $(filter HF_HOME,$(_AIC_DEFAULTED)),-u HF_HOME)) \
    $(if $(filter 0,$(AIC_SPUR_CLUSTER)),$(if $(filter NVME_DATA,$(_AIC_DEFAULTED)),-u NVME_DATA)) \
    $(if $(filter NFS_DATA,$(_AIC_DEFAULTED)),-u NFS_DATA) \
    $(if $(filter 0,$(AIC_SPUR_CLUSTER)),$(if $(filter GDS_SLAB_DATA,$(_AIC_DEFAULTED)),-u GDS_SLAB_DATA)) \
    $(if $(filter AIC_METRICS_DIR,$(_AIC_DEFAULTED)),-u AIC_METRICS_DIR) \
    $(_SLURM_TAG_ENV)

# ---- Export tarball --------------------------------------------------------
EXPORT_PREFIX  ?= aic-release
_GIT_SHORT_REV := $(shell git -C "$(CURDIR)" rev-parse --short HEAD 2>/dev/null || echo nogit)
_GIT_DIRTY     := $(if $(shell git -C "$(CURDIR)" status --porcelain -- . 2>/dev/null),-dirty,)
_GEN_DATE      := $(shell date +%Y%m%d)
EXPORT_TARBALL ?= $(CURDIR)/$(EXPORT_PREFIX)-$(_GEN_DATE)-$(_GIT_SHORT_REV)$(_GIT_DIRTY).tar.gz

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
	    'make client-build' 'Build the aic-client image (every benchmark client, aic-bench)' \
	    'make bench-list' 'List the aic-bench subcommands' \
	    'make bench BENCH=<subcommand>' 'Run aic-bench <subcommand> with BENCH_ARGS in aic-client' \
	    'make vllm-reset-test' 'Verify LMCache L1+L2 KV retrieval end-to-end' \
	    'make cliff' 'Run KV-cache cliff benchmark, write CSV to BENCH_LOGDIR/results/' \
	    'make plot' 'Generate cliff PNG charts from BENCH_LOGDIR/results/ CSVs' \
	    'make kvbench-up' 'Start KVBench + client on internal compose network' \
	    'make kvbench-down' 'Stop the KVBench compose profile' \
	    'make cliff-kvbench-local' 'Start KVBench, wait for readiness, run kvbench cliff arm' \
	    'make agent-simple-run' 'Run the synthetic cache-tier stress (AGENT_SIMPLE_ARGS)' \
	    'make agent-kv-nvme-sweep' 'agent-kv L2 hit TTFT against injected NVMe read delay (AGENT_KV_SWEEP_ARGS)' \
	    'make agentx-run' 'Run the AgentX agentic-trace benchmark against the running stack' \
	    'make agentx-sweep' 'agentx-run at each concurrency in AGENTX_CONCURRENCIES' \
	    'make agentx-corpus' 'Generate a token-scaled trace corpus (NOT the AgentX benchmark)'
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
	    'make list-vars' 'Show the values make would use (reference: docs/ENVIRONMENT.md)' \
	    'make env-docs' 'Regenerate the variable tables in docs/ENVIRONMENT.md'
	@printf '\n$(_AIC_B)$(_AIC_X)EXAMPLES$(_AIC_R)\n'
	@printf '  %s\n' \
	    'make build ROCM_ARCH=gfx1201 AIC_CACHE_DIR=~/.cache/aic-buildx' \
	    'make build BUILD_JOBS=3   # cap parallelism on low-RAM hosts' \
	    'make up VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct HF_TOKEN=hf_... NVME_DATA=/mnt/nvme' \
	    'make up-gds-l1 GDS_SLAB_DATA=/mnt/nvme HF_TOKEN=hf_...' \
	    'make cliff BENCH_ARM=vram_only BENCH_MODEL=Qwen/Qwen2.5-7B-Instruct' \
	    'make cliff-kvbench-local KVBENCH_MODEL=llama-3.1-8b BENCH_CONCUR=1,2' \
	    'make agentx-run AGENTX_CORPUS_SCALE=8 AGENTX_CONCURRENCY=1   # 32k-context model' \
	    'make dist-build dist-push smoke-test   # the nightly build chain'
	@printf '\n'

# The value make holds, else the default compose would apply (read from compose,
# so list-vars cannot disagree with it).
_or_compose_default = $(or $($(1)),<compose default: $(or $(shell "$(REPO_ROOT)/docker/scripts/aic-compose-default.sh" $(1) 2>/dev/null),?)>)

list-vars:
	@printf '$(_AIC_B)rocm-aic environment variables$(_AIC_R)\n'
	@printf '$(_AIC_D)Override any of these on the make command line or via .env.<cluster>$(_AIC_R)\n'
	@printf '$(_AIC_D)Defaults, owners and descriptions: docs/ENVIRONMENT.md$(_AIC_R)\n'
	@printf '\n$(_AIC_B)$(_AIC_X)IMAGE / BUILD$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'IMAGE_TAG' '$(IMAGE_TAG)' \
	    'VLLM_IMAGE_REF' '$(call _or_compose_default,VLLM_IMAGE_REF)' \
	    'LMCACHE_IMAGE_REF' '$(call _or_compose_default,LMCACHE_IMAGE_REF)' \
	    'HSA_SNOOP_IMAGE' '$(call _or_compose_default,HSA_SNOOP_IMAGE)' \
	    'ROCM_ARCH' '$(or $(ROCM_ARCH),<not detected>)' \
	    'ROCM_VERSION' '$(ROCM_VERSION)' \
	    'AIC_VERSION' '$(AIC_VERSION)' \
	    'BUILD_JOBS' '$(or $(BUILD_JOBS),<bake default: nproc>)' \
	    'BUILD_PROGRESS' '$(BUILD_PROGRESS)' \
	    'BAKE_TARGETS' '$(BAKE_TARGETS)' \
	    'BAKE_OUTPUT_FLAG' '$(BAKE_OUTPUT_FLAG)' \
	    'AIC_FAST_ARCH' '$(AIC_FAST_ARCH)' \
	    'WHEELS_DIR' '$(WHEELS_DIR)'
	@printf '\n$(_AIC_B)$(_AIC_X)LAYER CACHE$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'AIC_BUILDX_BUILDER' '$(AIC_BUILDX_BUILDER)' \
	    'AIC_CACHE_MODE' '$(or $(AIC_CACHE_MODE),<bake default>)' \
	    'AIC_CACHE_REF' '$(or $(AIC_CACHE_REF),<empty>)' \
	    'AIC_CACHE_DIR' '$(or $(AIC_CACHE_DIR),<empty: no local cache>)' \
	    'TLS_CERT' '$(or $(TLS_CERT),<none>)'
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
	    'VLLM_MODEL' '$(call _or_compose_default,VLLM_MODEL)' \
	    'VLLM_REF' '$(or $(VLLM_REF),<bake default>)' \
	    'TENSOR_PARALLEL_SIZE' '$(call _or_compose_default,TENSOR_PARALLEL_SIZE)' \
	    'VLM_GPU_MEMORY_UTILIZATION' '$(call _or_compose_default,VLM_GPU_MEMORY_UTILIZATION)' \
	    'VLM_MAX_MODEL_LEN' '$(call _or_compose_default,VLM_MAX_MODEL_LEN)' \
	    'VLM_MAX_NUM_BATCHED_TOKENS' '$(call _or_compose_default,VLM_MAX_NUM_BATCHED_TOKENS)' \
	    'VLM_BLOCK_SIZE' '$(call _or_compose_default,VLM_BLOCK_SIZE)' \
	    'VLM_READY_RETRIES' '$(VLM_READY_RETRIES) x 5 s' \
	    'VLM_FAIL_LOG_LINES' '$(VLM_FAIL_LOG_LINES)'
	@printf '\n$(_AIC_B)$(_AIC_X)LMCACHE$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'LMCACHE_REF' '$(LMCACHE_REF)' \
	    'LMCACHE_PORT' '$(LMCACHE_PORT)' \
	    'LMCACHE_L1_SIZE_GB' '$(call _or_compose_default,LMCACHE_L1_SIZE_GB) GiB' \
	    'LMCACHE_MAX_GPU_WORKERS' '$(call _or_compose_default,LMCACHE_MAX_GPU_WORKERS)' \
	    'LMCACHE_NVME_POOL' '$(call _or_compose_default,LMCACHE_NVME_POOL)' \
	    'LMCACHE_NVME_SLOT_SIZE' '$(call _or_compose_default,LMCACHE_NVME_SLOT_SIZE)' \
	    'LMCACHE_NFS_POOL' '$(call _or_compose_default,LMCACHE_NFS_POOL)' \
	    'AIC_L2_BACKEND' '$(call _or_compose_default,AIC_L2_BACKEND)'
	@printf '\n$(_AIC_B)$(_AIC_X)BENCHMARK (client image, aic-bench)$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'AIC_CLIENT_IMAGE' '$(call _or_compose_default,AIC_CLIENT_IMAGE)' \
	    'AIPERF_VERSION' '$(or $(AIPERF_VERSION),<bake default>)' \
	    'AGENT_KV_REF' '$(or $(AGENT_KV_REF),<bake default>)' \
	    'BENCH' '$(BENCH)' \
	    'BENCH_ARGS' '$(or $(BENCH_ARGS),<none>)' \
	    'AGENT_SIMPLE_ARGS' '$(AGENT_SIMPLE_ARGS)'
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
	@printf '\n$(_AIC_B)$(_AIC_X)BENCHMARK (AgentX)$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'AGENTX_MODEL' '$(or $(AGENTX_MODEL),$(VLLM_MODEL),<the model the endpoint serves>)' \
	    'AGENTX_CONCURRENCY' '$(or $(AGENTX_CONCURRENCY),<run_agentx.sh default>)' \
	    'AGENTX_CONCURRENCIES' '$(or $(AGENTX_CONCURRENCIES),<overnight_sweep.sh default>)' \
	    'AGENTX_DURATION' '$(if $(AGENTX_DURATION),$(AGENTX_DURATION) s,<run_agentx.sh default>)' \
	    'AGENTX_MAX_CONTEXT' '$(or $(AGENTX_MAX_CONTEXT),<max_model_len of the served model>)' \
	    'AGENTX_DATASET' '$(or $(AGENTX_DATASET),<run_agentx.sh default>)' \
	    'AGENTX_CORPUS' '$(or $(AGENTX_CORPUS),<empty: the public dataset>)' \
	    'AGENTX_CORPUS_SCALE' '$(or $(AGENTX_CORPUS_SCALE),<empty: unscaled>)' \
	    'AGENTX_CORPUS_CACHE' '$(or $(AGENTX_CORPUS_CACHE),<scaled_corpus.sh default>)' \
	    'AGENTX_SERVER_METRICS' '$(or $(AGENTX_SERVER_METRICS),<LMCache metrics via aic bridge>)' \
	    'AGENTX_ARGS' '$(or $(AGENTX_ARGS),<none>)'
	@printf '\n$(_AIC_B)$(_AIC_X)EMULATE & PROFILE (local targets; the Slurm jobs have their own defaults)$(_AIC_R)\n'
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
	    'KVBENCH_IMAGE_REF' '$(call _or_compose_default,KVBENCH_IMAGE_REF)' \
	    'KVBENCH_MODEL' '$(KVBENCH_MODEL)' \
	    'KVBENCH_PORT' '$(KVBENCH_PORT)' \
	    'KVBENCH_GPU_PROFILE' '$(call _or_compose_default,KVBENCH_GPU_PROFILE)' \
	    'KVBENCH_CHUNK_SIZE' '$(call _or_compose_default,KVBENCH_CHUNK_SIZE)' \
	    'KVBENCH_LOCAL_CPU_SIZE_GB' '$(call _or_compose_default,KVBENCH_LOCAL_CPU_SIZE_GB) GB' \
	    'KVBENCH_READY_S' '$(KVBENCH_READY_S) s'
	@printf '\n$(_AIC_B)$(_AIC_X)MONITORING$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'AIC_METRICS_DIR' '$(AIC_METRICS_DIR)' \
	    'AIC_EXPORTERS' '$(or $(AIC_EXPORTERS),<monitoring-lib.sh default>)' \
	    'AIC_GRAFANA_IMAGE' '$(call _or_compose_default,AIC_GRAFANA_IMAGE)'
	@printf '\n$(_AIC_B)$(_AIC_X)CLIFF (Slurm)$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'AIC_CLIFF_GFX' '$(or $(AIC_CLIFF_GFX),<empty>)' \
	    'AIC_CLIFF_CONSTRAINT' '$(or $(AIC_CLIFF_CONSTRAINT),<run-build-distribute.sh default>)' \
	    'AIC_CLIFF_GPUS' '$(or $(AIC_CLIFF_GPUS),<run-build-distribute.sh default>)' \
	    'AIC_CLIFF_NODE' '$(or $(AIC_CLIFF_NODE),<any>)' \
	    'AIC_CLIFF_TIME' '$(or $(AIC_CLIFF_TIME),<target default>)' \
	    'AIC_CLIFF_ARMS' '$(or $(AIC_CLIFF_ARMS),<target default>)'
	@printf '\n$(_AIC_B)$(_AIC_X)LOGGING / EXPORT$(_AIC_R)\n'
	@printf '  $(_AIC_G)%-38s$(_AIC_R) %s\n' \
	    'LOG' '$(LOG)' \
	    'PROM_DUMP_OUT' '$(or $(PROM_DUMP_OUT),<run-build-distribute.sh default>)' \
	    'PROM_DUMP_WAIT' '$(if $(PROM_DUMP_WAIT),$(PROM_DUMP_WAIT) s,<run-build-distribute.sh default>)' \
	    'EXPORT_PREFIX' '$(EXPORT_PREFIX)' \
	    'EXPORT_TARBALL' '$(notdir $(EXPORT_TARBALL))' \
	    'AIC_CI_LIB_DIR' '$(AIC_CI_LIB_DIR)'
	@printf '\n'
