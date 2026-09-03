# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# This tree is self-contained: the build context and all sources live here, so
# "repo root" is this directory (no dependency on any parent checkout).
REPO_ROOT := $(CURDIR)

# ai-dynamo/nixl v1.3.2 release; AIS_MT added via patches/nixl/.
NIXL_GIT_URL := https://github.com/ai-dynamo/nixl.git
NIXL_SHA     := v1.3.2

IMAGE_NAME ?= rocm-aic
override AIC_VERSION := $(strip $(file <$(REPO_ROOT)/VERSION))

# vLLM is source-built from VLLM_REF, so there is no VLLM_VERSION/VLLM_ROCM_VARIANT
# wheel pin, and hipFile now ships in the ROCm base image (pinned by ROCM_VERSION).
_FRAMEWORK_VERSION_ARGS := AIC_VERSION ROCM_VERSION VLLM_REF LMCACHE_REF NIXL_REF HSA_SNOOP_REF
_single_quote := '
_shell_quote = '$(subst $(_single_quote),'"'"',$(1))'
_FRAMEWORK_VERSION_ENV := $(foreach _arg,$(_FRAMEWORK_VERSION_ARGS),$(if $(filter undefined,$(origin $(_arg))),,$(_arg)=$(call _shell_quote,$(value $(_arg)))))

_IMAGE_TAG := $(shell $(_FRAMEWORK_VERSION_ENV) $(REPO_ROOT)/docker/scripts/aic-image-tag.sh 2>/dev/null)
IMAGE_TAG  ?= $(if $(_IMAGE_TAG),$(_IMAGE_TAG),latest)
IMAGE_REF  := $(IMAGE_NAME):$(IMAGE_TAG)

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

export AIC_VERSION ROCM_ARCH GPU GDS_SLAB_DATA LOG HF_HOME HF_TOKEN IMAGE_NAME IMAGE_REF IMAGE_TAG BUILD_JOBS
export LMCACHE_PORT LMCACHE_L1_SIZE_GB LMCACHE_NVME_POOL LMCACHE_NVME_SLOT_SIZE LMCACHE_NFS_POOL
export NVME_DATA NFS_DATA
export VLLM_MODEL TENSOR_PARALLEL_SIZE
export VLM_GPU_MEMORY_UTILIZATION VLM_MAX_MODEL_LEN VLM_MAX_NUM_BATCHED_TOKENS VLM_BLOCK_SIZE
export NIXL_GIT_URL NIXL_SHA

comma := ,
# The whole stack is `docker compose` (v2) only -- the docker-compose v1 standalone
# and the old docker-run sidecar fallbacks are gone.  `make ensure-compose` installs
# the v2 plugin into ~/.docker/cli-plugins (shared $HOME) on nodes that lack it.
_COMPOSE_BIN := docker compose
COMPOSE      := $(_FRAMEWORK_VERSION_ENV) DOCKER_BUILDKIT=1 $(_COMPOSE_BIN) -f "$(CURDIR)/docker/docker-compose.yml"
# The lmcache service lives behind the `cache` profile; the interactive stack and
# the cliff kvd arms enable it, the plain vram baseline does not.
COMPOSE_CACHE := $(COMPOSE) --profile cache

# docker compose v2 plugin (installed by `ensure-compose` when missing).  Override
# the version to bump.  Downloaded from the docker/compose GitHub releases.
COMPOSE_PLUGIN_VERSION ?= v2.40.0

# vLLM --kv-transfer-config for the MP connector (interactive `make up`).  The JSON
# is wrapped in single quotes so compose's shlex splitting preserves the inner
# double quotes; leave KV_TRANSFER_ARG empty for a plain (baseline) vLLM.
_MP_CONNECTOR_JSON := {"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://aic-lmcache","lmcache.mp.port":$(LMCACHE_PORT)}}
KV_TRANSFER_ARG    ?= --kv-transfer-config '$(_MP_CONNECTOR_JSON)'
export KV_TRANSFER_ARG

# ---- Metrics capture (Prometheus sidecar) ----------------------------------
# AIC_METRICS_DIR: Prometheus TSDB dir (bind-mount an NFS path here to explore
# a run afterward).  AIC_EXPORTERS=1 also launches the containerized node + AMD
# GPU exporters (for nodes without the host-installed exporter services).
AIC_METRICS_DIR  ?= $(CURDIR)/logs/prometheus
AIC_EXPORTERS    ?= 0
AIC_GRAFANA_PORT ?= 3000
AIC_GRAFANA_IMAGE ?= grafana/grafana:13.1.3
MON_COMPOSE     := $(_COMPOSE_BIN) -f "$(CURDIR)/monitoring/docker-compose.monitoring.yml"
_MON_PROFILE    := $(if $(filter 1,$(AIC_EXPORTERS)),--profile exporters,)
export AIC_METRICS_DIR AIC_GRAFANA_PORT AIC_GRAFANA_IMAGE

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

# ---- Self-hosted CI runner scripts -----------------------------------------
# The hardware-CI workflows call helper scripts from AIC_CI_LIB_DIR on the
# self-hosted runner (spur-dist-build.sh / spur-smoke-test.sh / spur-tiny-test.sh
# / spur-cliff.sh / spur-emulate-test.sh -- the last of which needs no GPU at
# either end).  `make install-ci-scripts` deploys the source copies from
# .github/scripts/runners there.  Scripts invoked directly from checked-out
# workflows live separately under .github/scripts/workflows.  Writing under
# /usr/local usually needs root, so the target uses sudo when needed.
AIC_CI_LIB_DIR    ?= /usr/local/lib/aic-ci
AIC_CI_SCRIPT_DIR := $(CURDIR)/.github/scripts/runners

AIC_FAST_ARCH ?= gfx950

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
export AIC_IMAGE_DIR        ?= $(AIC_SHARED_NFS)/rocm-aic/images
# Per-user BuildKit cache on the shared NFS volume: $HOME is small and quota'd
# on SPUR, and a cache shared across users is not writable by all of them.
# Cache export is best-effort, so a user's concurrent builds still share it.
export AIC_CACHE_DIR        ?= $(AIC_SHARED_NFS)/$(USER)/buildcache
# SPUR nodes have 8x NVMe drives combined into a single LVM at /mnt/m2m_nobackup.
# Use override (not ?=) so these win over the top-level ?= defaults set earlier.
# HF_HOME points to the cluster-wide model cache since /scratch does not exist
# on this cluster.
override export NVME_DATA     := /mnt/m2m_nobackup/aic-cliff/nvme
override export GDS_SLAB_DATA := /mnt/m2m_nobackup/aic-cliff/slab
override export HF_HOME       := $(AIC_SHARED_NFS)/huggingface
# SPUR nodes use XFS LVM (not bare NVMe); hipFile P2PDMA (AIS_MT backend) fails at
# register_memory because GPUDirect storage is not available on this filesystem.
# Default the nvme arm to the NIXL first-class POSIX plugin (cpu staging buffer),
# which avoids hipFileBufRegister entirely.  A user override still takes precedence.
# SPUR nodes are MI355X (gfx950); pin the arch for the compose build args.  The image
# ref itself is left to the version-derived tag (docker/scripts/aic-image-tag.sh) so a
# caller-supplied AIC_IMAGE_NAME -- notably CI's per-SHA rocm-aic-ci-<short> -- is
# honoured.  Arch selection is carried by the tarball's _arch_tag suffix, not the image
# tag, so cliff still resolves the gfx950 tarball via its glob.
override export ROCM_ARCH     := gfx950
export AIC_L2_BACKEND         ?= nixl_posix
# Right-size the POSIX slot file: must be >= KV chunk size for the target model.
#   gpt-oss-120b fp8: 256 tok × 8 KV heads × 64 dim × 2 × 36 layers × 1B = 9 MiB
#   Qwen2.5-3B fp8:  256 tok × 8 KV heads × 128 dim × 2 × 36 layers × 1B = 18 MiB
#   Default 32 MiB covers both; override for other models.
#   (vs the 256 MiB AIS_MT default, which pre-allocates a full staging buffer —
#    28× too large for POSIX where only actual KV bytes are written per slot).
# Also enlarge the pool: 18K-tok prefix = 71 chunks/client; at c=250 need ~18K slots;
#   32K at 32 MiB = 1 TiB on disk (lazy, fine for the 30 TB XFS LVM).
# Slot size: must be >= the KV chunk size for the target model.
#   gpt-oss-120b fp8: 256 tok × 8 heads × 64 dim × 2 × 36 layers × 1B = 9 MiB → 16 MiB OK
#   Qwen2.5-3B fp8:  256 tok × 8 heads × 128 dim × 2 × 36 layers × 1B = 18 MiB → 32 MiB needed
# Default 32 MiB covers both; override per model if needed.
export LMCACHE_NIXL_POSIX_SLOT_SIZE ?= 33554432
export LMCACHE_NIXL_POSIX_POOL      ?= 32768
# SPUR authz plugin blocks --pid=host, so node-exporter, hsa-snoop, and nvme-exporter
# (which all need --pid=host) cannot start.  Use "safe" mode: amdgpu-exporter and
# rdma-exporter do NOT need --pid=host and work fine on SPUR.
# Set AIC_EXPORTERS=1 for the full fleet on nodes that allow --pid=host.
# Set AIC_EXPORTERS=0 for Prometheus-only (no exporters at all).
override export AIC_EXPORTERS            := safe
# hsa-snoop: use container PID namespace instead of host to avoid spur-authz block.
# vLLM joins lmcache's PID ns (pid:service:lmcache), so aic-lmcache sees both processes.
override export AIC_HSA_SNOOP_PID_MODE  := container:aic-lmcache
# Reduce POSIX GPU workers: at high concurrency (c=32) the NIXL POSIX read path hits
# NIXL_ERR_INVALID_PARAM in makeXferReq when too many concurrent handles are in flight.
# 1 worker serialises STORE/RETRIEVE so the NIXL descriptor pool is never exhausted.
# The write cost at c=1/8 is still acceptable; L2 reads work reliably at 1 worker.
export LMCACHE_MAX_GPU_WORKERS      ?= 1
else
export AIC_CACHE_DIR        ?= /scratch/$(USER)/images/buildcache
# Alola mounts /scratch from BeeGFS, so unlike SPUR it IS shared across compute
# nodes -- and /scratch/models is the cluster-wide HuggingFace cache everyone
# already populates (its hub/ holds the weights).  /home is the wrong home for
# multi-GB checkpoints there: 87% full at 100T, and NFS rather than parallel.
# Keyed on the hub layout so no extra flag is needed, and applied only while
# HF_HOME still holds the file-scope default above -- an environment variable or
# `make HF_HOME=...` keeps winning.
ifneq ($(wildcard /scratch/models/hub),)
ifeq ($(origin HF_HOME),file)
override export HF_HOME     := /scratch/models
# The override changes $(origin HF_HOME) from "file" to "override", which would
# stop _CLIFF_STRIP below from unsetting it -- silently pushing the shared cache
# into cliff jobs that have their own node-appropriate staging.  Remember that
# this value is still just a Makefile default so the strip keeps applying.
_HF_HOME_IS_DEFAULT := 1
endif
endif
endif
_HF_HOME_IS_DEFAULT ?= $(if $(filter file,$(origin HF_HOME)),1,)

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
    $(if $(filter 0,$(AIC_SPUR_CLUSTER)),$(if $(_HF_HOME_IS_DEFAULT),-u HF_HOME)) \
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

.PHONY: help ensure-compose build up up-batch up-dev up-monitoring down-monitoring up-gds-l1 up-gds-l1-batch down logs logs-lmcache logs-vllm \
        ps shell-lmcache shell-vllm restart-vllm restart-lmcache cliff plot venv vllm-reset-test stress-grafana \
        monitoring-up monitoring-down monitoring-logs monitoring-build-exporters \
        dist-build dist-build-fast dist-build-emulate dist-build-exporters dist-build-monitoring dist-push \
        smoke-test smoke-test-fast tiny-test tiny-test-fast \
        emulate-test emulate-mp-test emulate-validate test-emulate-local stress-emulate-local capture-profile-local profile-capture \
        install-ci-scripts cliff-submit cliff-short \
        cliff-kvd cliff-spur-l2 cliff-spur-l2-debug cliff-long-64k cliff-long-128k \
        export _check_hf_token _prep_dirs _check_gds_slab

.DEFAULT_GOAL := help

help:
	@echo "rocm-aic aic-release — AMD Infinity Context inference stack + benchmarks"
	@echo ""
	@echo "Stack targets:"
	@echo "  make ensure-compose    Install the docker compose v2 plugin if missing (user-local)"
	@echo "  make build             Build the shared image ($(IMAGE_REF))"
	@echo "  make up                Start lmcache + vllm (foreground, DRAM L1 + AIS_MT/NFS L2)"
	@echo "  make up-batch          Start lmcache + vllm (background)"
	@echo "  make up-dev            Start in dev mode: --enforce-eager skips CUDA graph capture (~60s faster, ~10% slower inference)"
	@echo "  make up-monitoring     Start lmcache + vllm + prometheus + exporters (background)"
	@echo "  make down-monitoring   Stop full stack including monitoring profile"
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
	@echo "  make plot              Generate cliff PNG charts from $(BENCH_LOGDIR)/results/ CSVs"
	@echo ""
	@echo "Distribute / cliff targets (Slurm; wrap .slurm/ scripts + sbatch):"
	@echo "  (dist-build/dist-build-exporters/smoke-test submit via sbatch and log to logs/<job-id>/)"
	@echo "  make dist-build        Build image (+ fabric exporters) on a Slurm build node, save tarballs"
	@echo "  make dist-build-fast   Single-arch dev build (AIC_FAST_ARCH=$(AIC_FAST_ARCH), no exporters) -- faster iteration"
	@echo "  make dist-build-emulate  Build the CPU-only emulation image (no GPU kernels compiled)"
	@echo "  make dist-build-exporters  Build ONLY the nvme/rdma exporter images (no main rebuild)"
	@echo "  make dist-build-monitoring Pull + save Prometheus/amdgpu-exporter to AIC_IMAGE_DIR"
	@echo "  make dist-push         Tag + push the built image (needs AIC_PUSH_REF)"
	@echo "  make smoke-test        Load + smoke-test the image on a GPU+NVMe node"
	@echo "                         (also sanity-checks exporters + writes a Prometheus TSDB"
	@echo "                          to logs/<job-id>/prometheus; AIC_SMOKE_EXPORTERS=0 skips)"
	@echo "  make smoke-test-fast   Smoke-test the single-arch dev image (AIC_FAST_ARCH=$(AIC_FAST_ARCH))"
	@echo "  make tiny-test         End-to-end serve check (MP stack + tiny model, one completion)"
	@echo "  make tiny-test-fast    Fast variant of tiny-test"
	@echo "  make test-emulate-local  Local emulate test (no SLURM): bring up vllm-emulator, assert completion + hook"
	@echo "  make stress-emulate-local  Start emulator + Prometheus, run sustained sweep, print /metrics"
	@echo "  make capture-profile-local  Local profile capture (requires /dev/kfd): real GPU serve + sweep -> pack"
	@echo "  make emulate-test      Serve check of the emulation image on a CPU-only node (no GPU)"
	@echo "  make emulate-mp-test   Emulation + the full LMCache MP recipe on a CPU-only node"
	@echo "  make profile-capture   Capture an AMD profile pack from a REAL GPU serve (gfx942/gfx950)"
	@echo "  make emulate-validate  Replay a captured pack on CPU and diff vs the real-hardware run"
	@echo "  make install-ci-scripts  Deploy .github/scripts/runners/*.sh to $(AIC_CI_LIB_DIR) (sudo if needed)"
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

# Ensure the `docker compose` (v2) plugin is available.  Docker checks
# $HOME/.docker/cli-plugins before the system dir, and $HOME is shared across the
# Slurm/SPUR nodes, so a user-local install fixes every node without root.  No-op
# when compose is already present; idempotent.
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

build: ensure-compose monitoring-build-exporters
	@test -n "$(ROCM_ARCH)" || { \
		echo "ERROR: ROCM_ARCH empty (install ROCm or set ROCM_ARCH=gfxNNNN)" >&2; exit 1; }
	cd "$(REPO_ROOT)" && $(COMPOSE_CACHE) build \
		$(if $(TLS_CERT),--secret id=tls_cert$(comma)src=$(TLS_CERT),)
	@docker tag "$(IMAGE_REF)" "$(IMAGE_NAME):latest"
	@echo "Built $(IMAGE_REF) (also tagged $(IMAGE_NAME):latest)"

up: ensure-compose _check_hf_token _prep_dirs
	@mkdir -p "$(AIC_METRICS_DIR)"
	PROM_UID="$$(id -u)" PROM_GID="$$(id -g)" \
	    $(COMPOSE_CACHE) --profile monitoring up

up-batch: ensure-compose _check_hf_token _prep_dirs
	@mkdir -p "$(AIC_METRICS_DIR)"
	PROM_UID="$$(id -u)" PROM_GID="$$(id -g)" \
	    $(COMPOSE_CACHE) --profile monitoring up -d
	@echo "Started. Use 'make logs' to follow or 'make down' to stop."

up-dev: ensure-compose _check_hf_token _prep_dirs  # Fast startup: --enforce-eager skips CUDA graph capture (~60s faster)
	@mkdir -p "$(AIC_METRICS_DIR)"
	PROM_UID="$$(id -u)" PROM_GID="$$(id -g)" \
	    VLLM_EXTRA_ARGS="--enforce-eager $${VLLM_EXTRA_ARGS}" \
	    $(COMPOSE_CACHE) --profile monitoring up -d
	@echo "Started in dev mode (enforce-eager, no CUDA graphs). Use 'make logs' to follow."

# up-monitoring kept as an alias for up-batch for backward compatibility.
up-monitoring: up-batch

down-monitoring:                                           # Stop full stack including monitoring profile
	$(COMPOSE_CACHE) --profile monitoring down

up-gds-l1: ensure-compose _check_hf_token _check_gds_slab _prep_dirs
	GDS_MODE=1 $(COMPOSE_CACHE) up

up-gds-l1-batch: ensure-compose _check_hf_token _check_gds_slab _prep_dirs
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

vllm-reset-test: _check_hf_token _prep_dirs
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

# Sustained Grafana stress target: drives L1→L2 spill and L2 reads continuously
# so all Grafana panels show live activity.  Sized for the running stack:
#   ISL=1024 shared-prefix=896 → ~14 chunks/request, small enough for the
#   Qwen/small model, large enough to build up L1 pressure at c=8.
#   Flood mode: 10 iters × c=8 = 80 request-batches per pass, repeated in a
#   shell loop so the panels never go idle.  Ctrl-C to stop.
# Watch: L1 usage gauge, NIXL TX/RX, ISL/OSL row, TTFT, kernel launch rate.
stress-grafana: _prep_dirs
	@test -n "$(BENCH_MODEL)" || { \
		echo "ERROR: set BENCH_MODEL or VLLM_MODEL to the served model name" >&2; exit 1; }
	@docker inspect aic-client > /dev/null 2>&1 || { \
		echo "ERROR: aic-client container not running — start with 'make up'" >&2; exit 1; }
	@echo "Stressing stack for Grafana — watch http://localhost:3000  (ctrl-c to stop)"
	@echo "  endpoint: http://aic-vllm-gpu0:8000 (via aic-client)"
	@while true; do \
		docker exec aic-client \
			python3 -u benchmarks/run_cliff.py \
			--endpoint http://aic-vllm-gpu0:8000 \
			--model "$(BENCH_MODEL)" \
			--arm kvd_v2 \
			--isl 1024 \
			--shared-prefix-tokens 896 \
			--concurrencies 1,2,4,8 \
			--iters 5 \
			--warmup-iters 1 \
			--post-warmup-sleep-s 2 \
			--out /logs/manual/results/stress-$$(date +%Y%m%d-%H%M%S).csv; \
		echo "--- pass complete, restarting in 3s ---"; \
		sleep 3; \
	done

plot: _prep_dirs
	$(PYTHON) "$(CURDIR)/benchmarks/plot_cliff.py" \
		--input "$(BENCH_LOGDIR)/results/" \
		--output-dir "$(BENCH_LOGDIR)/plots/"
	@echo "Charts written to $(BENCH_LOGDIR)/plots/"

monitoring-up: ensure-compose
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

dist-build-fast:               # Single-arch (AIC_FAST_ARCH) dev build -- fast edit-build loop
	@# A cut-down version of dist-build.
	@$(MAKE) --no-print-directory dist-build \
	    AIC_ROCM_ARCH='$(AIC_FAST_ARCH)' AIC_BUILD_EXPORTERS=0 AIC_UCX_FAST=1
dist-build-emulate:            # Build the CPU-only emulation image on a Slurm build node
	@# Dockerfile `emulate` stage + VLLM_TARGET_DEVICE=empty: vLLM + the llm-emu
	@# plugin with NO GPU kernels compiled (no HIP kernels, no LMCache HIP ext, no
	@# NIXL, no hsa-snoop).  Tagged separately (AIC_EMULATE_IMAGE, default
	@# $(IMAGE_NAME):7.14-emulate) with its own tarball, so `make dist-build`'s
	@# GPU image is untouched.  Pair with `make emulate-test`.
	"$(DIST)" build-emulate

dist-build-exporters:          # Build ONLY the fabric exporters (no main-image rebuild)
	@# Rebuild just the nvme/rdma exporter images -- e.g. after `make dist-build`
	@# succeeded for the main image but the exporter step failed for lack of Docker
	@# Hub egress.  Pin an egress-capable node with AIC_BUILD_NODE=<node>, or build
	@# on the current host with AIC_BUILD_LOCAL=1.
	"$(DIST)" build-exporters

dist-build-monitoring:         # Pull + save monitoring sidecar images to AIC_IMAGE_DIR
	@# Saves prom/prometheus and rocm/device-metrics-exporter to the shared NFS image
	@# dir so cliff nodes can load them without Docker Hub egress.  Run once; the
	@# load_image_if_needed call in run-cliff.sbatch picks them up automatically.
	@set -e; \
	for img in \
	    "prom/prometheus:v3.13.2" \
	    "rocm/device-metrics-exporter:v1.5.1" \
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

dist-push:                     # Tag + push the built image to a registry (needs AIC_PUSH_REF)
	"$(DIST)" push

smoke-test:                    # Load + smoke-test the image on a GPU+NVMe node
	"$(DIST)" test

smoke-test-fast:               # Smoke-test the single-arch (AIC_FAST_ARCH) dev image
	@# A cut-down version of smoke-test.
	@# Must pin the SAME AIC_ROCM_ARCH as dist-build-fast.
	@$(MAKE) --no-print-directory smoke-test \
	    AIC_ROCM_ARCH='$(AIC_FAST_ARCH)' AIC_SMOKE_EXPORTERS=0

tiny-test:                     # End-to-end serve check: MP stack + a tiny model, one real completion
	@# Stages Qwen/Qwen2.5-0.5B-Instruct, brings up the compose MP stack (nvme arm),
	@# waits for the endpoint, and asserts one non-empty chat completion.  Fast
	@# functional gate that exercises the connector path a smoke-test cannot.
	"$(DIST)" tiny-test

tiny-test-fast:
	@# Must pin the SAME AIC_ROCM_ARCH as dist-build-fast and smoke-test-fast.
	@$(MAKE) --no-print-directory tiny-test \
	    AIC_ROCM_ARCH='$(AIC_FAST_ARCH)'

# Local emulate test — no SLURM required.  Works with the full production image
# (build + runtime target) or the CPU-only emulate target; both have the llm-emu
# plugin installed.  Uses whatever IMAGE_REF resolves to.
AIC_EMULATE_MODEL     ?= Qwen/Qwen3-8B
AIC_EMULATE_READY_S   ?= 120

test-emulate-local: ensure-compose _prep_dirs  ## Spin up emulate profile, assert completion + hook active, tear down
	@echo "=== test-emulate-local: IMAGE_REF=$(IMAGE_REF) model=$(AIC_EMULATE_MODEL) ==="
	@VLLM_MODEL="$(AIC_EMULATE_MODEL)" IMAGE_REF="$(IMAGE_REF)" $(COMPOSE) --profile emulate up -d vllm-emulator
	@echo "Waiting up to $(AIC_EMULATE_READY_S)s for /health (engine fully ready) ..."
	@_ready=0; \
	for _i in $$(seq 1 $$(($(AIC_EMULATE_READY_S)/5))); do \
	    if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then _ready=1; break; fi; \
	    if [ -z "$$(docker ps -q -f name=aic-vllm-emulator)" ]; then \
	        echo "FAIL: emulator container exited during startup" >&2; \
	        $(COMPOSE) --profile emulate logs --tail 60 vllm-emulator; \
	        $(COMPOSE) --profile emulate down --remove-orphans >/dev/null 2>&1; \
	        exit 1; \
	    fi; \
	    sleep 5; \
	done; \
	if [ "$$_ready" != "1" ]; then \
	    echo "FAIL: endpoint not ready after $(AIC_EMULATE_READY_S)s" >&2; \
	    $(COMPOSE) --profile emulate logs --tail 80 vllm-emulator; \
	    $(COMPOSE) --profile emulate down --remove-orphans >/dev/null 2>&1; \
	    exit 1; \
	fi; \
	sleep 5; \
	echo "Endpoint ready — sending completion ..."; \
	resp=$$(curl -sS http://localhost:8000/v1/completions \
	    -H 'Content-Type: application/json' \
	    -d '{"model":"$(AIC_EMULATE_MODEL)","prompt":"Hello","max_tokens":16,"temperature":0}' 2>&1); \
	echo "Response: $$resp"; \
	rc=0; \
	tokens=$$(printf '%s' "$$resp" | grep -oE '"completion_tokens"[[:space:]]*:[[:space:]]*[0-9]+' | grep -oE '[0-9]+$$'); \
	if [ -n "$$tokens" ] && [ "$$tokens" -gt 0 ]; then \
	    echo "OK: emulated engine generated $$tokens tokens"; \
	else \
	    echo "FAIL: no tokens generated" >&2; rc=1; \
	fi; \
	logfile=$(LOG)/emulate-check.log; \
	mkdir -p "$(LOG)"; \
	$(COMPOSE) --profile emulate logs --no-color --no-log-prefix vllm-emulator > "$$logfile" 2>&1 || true; \
	if grep -q '\[ExecutorEmulatorHook\] Enabled' "$$logfile"; then \
	    echo "OK: executor hook active"; \
	else \
	    echo "FAIL: executor hook never activated (is VLLM_EMULATOR_ENABLE_ORACLE=1?)" >&2; rc=1; \
	fi; \
	if grep -q '\[ExecutorHook\] step=' "$$logfile"; then \
	    echo "OK: steps served from the profile pack"; \
	else \
	    echo "FAIL: no emulated steps recorded in logs" >&2; rc=1; \
	fi; \
	$(COMPOSE) --profile emulate down --remove-orphans >/dev/null 2>&1; \
	[ "$$rc" -eq 0 ] && echo "=== test-emulate-local PASSED ===" || { echo "=== test-emulate-local FAILED ===" >&2; exit 1; }

# Local profile capture — no SLURM.  Requires /dev/kfd and /dev/dri on the host.
# Runs a real GPU serve with VLLM_EMULATOR_TRACE_STEP_CYCLE=1, drives a sweep,
# then builds + validates the pack.  The pack lands in AIC_CAPTURE_DIR.
#
# gfx1201 (RDNA4, ~16GB VRAM) defaults: Qwen2.5-3B-Instruct, small sweep.
# For a larger GPU (gfx942/gfx950) override AIC_CAPTURE_MODEL and AIC_CAPTURE_SWEEP.
AIC_CAPTURE_MODEL     ?= Qwen/Qwen2.5-3B-Instruct
AIC_CAPTURE_HF_HOME   ?= $(HF_HOME)
AIC_CAPTURE_DIR       ?= $(CURDIR)/profiles/captures
AIC_CAPTURE_GPU       ?= $(GPU)
AIC_CAPTURE_GPU_UTIL  ?= 0.85
AIC_CAPTURE_MAX_MODEL_LEN     ?= 4096
AIC_CAPTURE_MAX_BATCHED_TOKENS ?= 2048
# Shorter sweep than the MI300X one: gfx1201 is slower so high-concurrency
# points take much longer.  Covers the token-count and concurrency axes the
# oracle buckets on without running for hours.
AIC_CAPTURE_SWEEP ?= 128,16,1,12 128,16,8,64 512,64,1,8 512,64,4,32 \
                     1024,64,1,8 1024,64,4,32 1024,64,16,64 \
                     2048,64,1,6 2048,64,4,24 4096,64,1,4 4096,64,4,16
AIC_CAPTURE_WARMUP_SKIP ?= 5

AIC_EMULATE_STRESS_CONCUR ?= 1,4,8,16
AIC_EMULATE_STRESS_ISL    ?= 512
AIC_EMULATE_STRESS_OSL    ?= 128
AIC_EMULATE_STRESS_ITERS  ?= 5

stress-emulate-local: ensure-compose _prep_dirs  ## Start emulator + Prometheus, run a sustained cliff sweep, show /metrics
	@echo "=== stress-emulate-local: $(IMAGE_REF) model=$(AIC_EMULATE_MODEL) ==="
	@mkdir -p "$(AIC_METRICS_DIR)"
	IMAGE_REF="$(IMAGE_REF)" VLLM_MODEL="$(AIC_EMULATE_MODEL)" \
	    PROM_UID="$$(id -u)" PROM_GID="$$(id -g)" \
	    $(COMPOSE) --profile emulate --profile monitoring up -d vllm-emulator prometheus
	@echo "Waiting for emulator endpoint ..."
	@for _i in $$(seq 1 24); do \
	    curl -fsS http://localhost:8000/health >/dev/null 2>&1 && break; \
	    sleep 5; \
	done
	@echo "Running sweep: ISL=$(AIC_EMULATE_STRESS_ISL) OSL=$(AIC_EMULATE_STRESS_OSL) c=$(AIC_EMULATE_STRESS_CONCUR) x$(AIC_EMULATE_STRESS_ITERS)"
	docker exec aic-vllm-emulator vllm bench serve \
	    --host localhost --port 8000 \
	    --model "$(AIC_EMULATE_MODEL)" \
	    --dataset-name random \
	    --random-input-len "$(AIC_EMULATE_STRESS_ISL)" \
	    --random-output-len "$(AIC_EMULATE_STRESS_OSL)" \
	    --num-prompts $$(($(AIC_EMULATE_STRESS_ITERS) * 64)) \
	    --max-concurrency "$$(echo $(AIC_EMULATE_STRESS_CONCUR) | tr ',' '\n' | sort -rn | head -1)" \
	    --percentile-metrics ttft,tpot,itl,e2el \
	    --ignore-eos 2>&1 | tail -30
	@echo ""
	@echo "=== Key vLLM /metrics ==="
	@curl -s http://localhost:8000/metrics | \
	    grep -E "^vllm:(num_requests|e2e_request_latency|request_prompt_tokens|request_generation_tokens|gpu_cache_usage|request_success|request_failure)" | \
	    grep -v "^#" | sort | head -30
	@echo ""
	@echo "Prometheus at http://localhost:9090 — stack left running. Use 'make down' to stop."

capture-profile-local: _prep_dirs  ## Capture a gfx1201 profile pack locally (requires /dev/kfd)
	@test -e /dev/kfd || { echo "ERROR: /dev/kfd not found — GPU not accessible here"; exit 1; }
	@test -n "$(ROCM_ARCH)" || { echo "ERROR: ROCM_ARCH empty" >&2; exit 1; }
	@mkdir -p "$(AIC_CAPTURE_DIR)/bench"
	@stamp=$$(date +%Y%m%d-%H%M%S); \
	model_tag=$$(echo '$(AIC_CAPTURE_MODEL)' | tr '/' '-'); \
	trace_file="step-trace-$${model_tag}-$${stamp}.jsonl"; \
	pack_name="$${model_tag}-$${stamp}.json"; \
	echo "=== capture-profile-local: IMAGE_REF=$(IMAGE_REF) model=$(AIC_CAPTURE_MODEL) arch=$(ROCM_ARCH) ==="; \
	echo "Trace -> $(AIC_CAPTURE_DIR)/$${trace_file}"; \
	docker run -d --name aic-vllm-capture \
	    --device /dev/kfd --device /dev/dri \
	    --network host --ipc host \
	    --cap-add CAP_SYS_ADMIN --cap-add SYS_PTRACE \
	    --security-opt seccomp=unconfined \
	    -v "$(AIC_CAPTURE_HF_HOME):/hf" \
	    -v "$(AIC_CAPTURE_DIR):/trace" \
	    -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub \
	    -e HF_TOKEN="$${HF_TOKEN:-}" -e HF_HUB_OFFLINE=0 \
	    -e ROCR_VISIBLE_DEVICES="$(AIC_CAPTURE_GPU)" \
	    -e VLLM_ROCM_USE_AITER=1 \
	    -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:False \
	    -e PYTHONUNBUFFERED=1 \
	    -e VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
	    -e "VLLM_EMULATOR_STEP_TRACE_OUTPUT=/trace/$${trace_file}" \
	    "$(IMAGE_REF)" \
	    --model "$(AIC_CAPTURE_MODEL)" \
	    --host 0.0.0.0 --port 8000 \
	    --max-model-len "$(AIC_CAPTURE_MAX_MODEL_LEN)" \
	    --max-num-batched-tokens "$(AIC_CAPTURE_MAX_BATCHED_TOKENS)" \
	    --gpu-memory-utilization "$(AIC_CAPTURE_GPU_UTIL)" \
	    --no-enable-prefix-caching \
	    --attention-backend TRITON_ATTN \
	    --disable-access-log-for-endpoints "/health,/metrics,/v1/models" \
	    >/dev/null || { echo "FAIL: docker run failed" >&2; exit 1; }; \
	echo "Waiting for endpoint (weights download may take a few minutes) ..."; \
	_ready=0; for _i in $$(seq 1 60); do \
	    curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1 && { _ready=1; break; }; \
	    [ -z "$$(docker ps -q -f name=aic-vllm-capture)" ] && \
	        { docker logs --tail 40 aic-vllm-capture >&2; docker rm -f aic-vllm-capture >/dev/null 2>&1; exit 1; }; \
	    sleep 10; \
	done; \
	[ "$$_ready" != "1" ] && { echo "FAIL: endpoint never ready" >&2; docker rm -f aic-vllm-capture >/dev/null 2>&1; exit 1; }; \
	echo "Endpoint ready — running sweep ..."; \
	rc=0; _seed=0; \
	for point in $(AIC_CAPTURE_SWEEP); do \
	    _seed=$$((_seed+1)); \
	    isl=$$(echo "$$point" | cut -d, -f1); \
	    osl=$$(echo "$$point" | cut -d, -f2); \
	    conc=$$(echo "$$point" | cut -d, -f3); \
	    np=$$(echo "$$point" | cut -d, -f4); \
	    echo "--- isl=$$isl osl=$$osl c=$$conc n=$$np ---"; \
	    docker exec aic-vllm-capture vllm bench serve \
	        --host localhost --port 8000 \
	        --model "$(AIC_CAPTURE_MODEL)" \
	        --dataset-name random \
	        --random-input-len "$$isl" --random-output-len "$$osl" \
	        --num-prompts "$$np" --max-concurrency "$$conc" \
	        --ignore-eos --seed "$$_seed" \
	        --percentile-metrics ttft,tpot,itl,e2el \
	        --save-result --result-dir /trace/bench \
	        --result-filename "real-$${model_tag}-isl$${isl}-osl$${osl}-c$${conc}.json" \
	        2>&1 | sed 's/^/  [bench] /' || rc=1; \
	done; \
	echo "Sweep done (rc=$$rc); stopping server ..."; \
	docker stop -t 60 aic-vllm-capture >/dev/null 2>&1 || true; \
	sleep 3; \
	docker rm -f aic-vllm-capture >/dev/null 2>&1 || true; \
	[ -s "$(AIC_CAPTURE_DIR)/$$trace_file" ] || { echo "FAIL: trace not written" >&2; exit 1; }; \
	echo "Building profile pack ..."; \
	docker run --rm \
	    -v "$(AIC_CAPTURE_DIR):/trace" \
	    --entrypoint python3 "$(IMAGE_REF)" \
	    -m vllm_emulator.profile.build_serving_profile_filtered \
	    "/trace/$$trace_file" "/trace/$$pack_name" \
	    --warmup-skip "$(AIC_CAPTURE_WARMUP_SKIP)" \
	    2>&1 | sed 's/^/  [pack] /'; \
	echo "Pack: $(AIC_CAPTURE_DIR)/$$pack_name"; \
	echo "Copy it to profiles/ and add a .capture.txt sibling to use it in CI."

profile-capture:               # Capture an AMD profile pack from a REAL GPU serve
	@# Runs the full image on a GPU node with VLLM_EMULATOR_TRACE_STEP_CYCLE=1 (real
	@# weights, real kernels), drives a vllm bench serve sweep over input-length x
	@# concurrency, and turns the step trace into a profile pack the emulator can
	@# replay.  The pack, the raw trace and the real-hardware benchmark JSONs land in
	@# AIC_CAPTURE_DIR.  AIC_ROCM_ARCH must match the built image tarball, e.g.
	@#   AIC_ROCM_ARCH=gfx942 AIC_CAPTURE_NODE=<mi300x-node> make profile-capture
	"$(DIST)" profile-capture

emulate-validate:              # Replay a captured pack and diff against real hardware
	@# Runs the capture's benchmark points against the emulator on a CPU node and
	@# prints real-vs-emulated TTFT / TPOT / throughput deltas.  Needs the pack from
	@# `make profile-capture`:
	@#   AIC_VALIDATE_PACK=/scratch/$(USER)/images/profiles/<pack>.json make emulate-validate
	"$(DIST)" emulate-validate

emulate-test:                  # Serve check of the emulation image on a CPU-ONLY node
	@# Brings up the compose `emulate` profile (no GPU, no weights loaded), asserts
	@# a non-empty completion, and asserts the llm-emu executor hook -- not a real
	@# forward pass -- produced it.  Needs `make dist-build-emulate` first.
	"$(DIST)" emulate-test

emulate-mp-test:               # Emulation + the FULL LMCache MP recipe, still no GPU
	@# The compose `emulate-mp` profile: standalone lmcache server + vLLM with
	@# LMCacheMPConnector, on a CPU-only node.  The KV tensors the connector
	@# registers are synthesized in host memory (zeros, but vLLM's own shape and
	@# byte count), so LMCache/NIXL transfer cost is measured on top of the
	@# profile-pack compute cost.  Asserts the PATH, not just a 200: buffers
	@# registered, LMCache on the CPU SHM lmcache-driven route, bytes stored,
	@# nothing left waiting on a remote KV load.
	@#
	@# Uses the PRODUCTION image -- the `emulate` stage stops before LMCache --
	@# so run `make dist-build` first, not `dist-build-emulate`:
	@#   AIC_ROCM_ARCH=gfx942 make dist-build emulate-mp-test
	"$(DIST)" emulate-mp-test

install-ci-scripts:            # Deploy .github/scripts/runners/*.sh to the runner's AIC_CI_LIB_DIR
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
# GPUs reserved for a SPUR cliff submit. raise it only alongside a matching
# tensor-parallel config.
AIC_CLIFF_GPUS ?= 1
ifeq ($(strip $(AIC_CLIFF_CONSTRAINT)),)
ifneq ($(strip $(AIC_CLIFF_GFX)),)
AIC_CLIFF_CONSTRAINT := $(shell echo '$(AIC_CLIFF_GFX)' | tr '[:lower:]' '[:upper:]')
else
AIC_CLIFF_CONSTRAINT := GFX942&NVME
endif
endif
ifeq ($(AIC_SPUR_CLUSTER),1)
_CLIFF_SPUR_CTL  := SPUR_CONTROLLER_ADDR=$(AIC_SPUR_CONTROLLER)
_CLIFF_SBATCH_ARGS := --partition=amd-spur --constraint= --gpus=$(AIC_CLIFF_GPUS) \
    $(if $(AIC_CLIFF_NODE),--nodelist=$(AIC_CLIFF_NODE),)
# SPUR sbatch does not support --parsable or --no-requeue; parse job id from "Submitted batch job N"
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

# Focused KVD cliff: shared prefix mode + sparse concurrency ladder.
# Rationale: per_client mode never exercises L2 read-back (VRAM L1 holds
# all per-client prefixes without evicting to L2).  shared mode makes every
# second+ client hit the same cached prefix, so the vram/kvd performance
# difference is visible.  The concurrency ladder skips the dense plateau
# (c=48–160 all plateau at ~145K tok/s) and captures the ramp + cliff edge:
#   1 (cold), 8, 32 (near-peak), 64, 128 (peak), 250 (beyond peak).
# With 2 iters and shared prefix the sweep runs in ~30–45 min per arm.
cliff-kvd:
	@cd "$(CURDIR)" && jobid=$$( \
	    BENCH_PREFIX_MODE=shared \
	    BENCH_CONCUR="$${BENCH_CONCUR:-1,8,32,64,128,250}" \
	    BENCH_ITERS="$${BENCH_ITERS:-2}" \
	    $(call _CLIFF_SUBMIT,--job-name=aic-cliff-kvd \
	    --time=$(if $(AIC_CLIFF_TIME),$(AIC_CLIFF_TIME),02:00:00))) && \
	    echo "submitted cliff-kvd job $$jobid (shared prefix, c=1,8,32,64,128,250, 2 iters)" && \
	    echo "log: $(CURDIR)/logs/$$jobid/cliff.out"

# SPUR-tuned L2 cliff: exercises real KV eviction through the DRAM L1 into the POSIX NVMe L2.
#
# Why per_client prefix:
#   With shared prefix, all c clients share ~630 MiB of KV chunks, which fits 13× in the
#   8 GB DRAM L1 regardless of c — ext_hit stays 0% at every concurrency (confirmed job 11301).
#   per_client gives each client a unique stable prefix; the combined working set is c × 630 MiB,
#   which overflows the 8 GB DRAM L1 at c≥13 and forces real NVMe L2 reads (ext_hit > 0%).
#
# Tier sizing:
#   VLM_GPU_MEMORY_UTILIZATION=0.40   VRAM KV cache ~17 GB; evicts at c≈7 per_client
#   AIC_LOCAL_CPU=true / DRAM L1=8 GB absorbs VRAM evictions; spills to NVMe at c≥13
#   POSIX NVMe L2 (nixl_posix, 16 MiB slots, pool=32768) on /mnt/m2m_nobackup XFS LVM
#
# Single concurrency point: c=32.
#   c=32 overflows the 8 GB DRAM L1 (32 × 630 MiB ≈ 20 GB) → ext_hit > 0% expected.
#   vram arm skipped — we have clean vram_only data from previous runs.
#   post-warmup-sleep scaled to 300s so the POSIX write queue drains before
#   timed iters begin (workers=1 serialises writes; c=32 warmup takes ~450s).
#   BENCH_ITERS=2 for speed; iters are identical once cache is primed.
# Arms: nvme only — vram arm skipped (data already collected), gds skipped (SPUR XFS).
cliff-spur-l2:
	@cd "$(CURDIR)" && jobid=$$( \
	    BENCH_PREFIX_MODE=per_client \
	    BENCH_CONCUR="$${BENCH_CONCUR:-32}" \
	    BENCH_ITERS="$${BENCH_ITERS:-2}" \
	    VLLM_MODEL="$${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}" \
	    VLM_GPU_MEMORY_UTILIZATION="$(if $(filter command line environment override,$(origin VLM_GPU_MEMORY_UTILIZATION)),$(VLM_GPU_MEMORY_UTILIZATION),0.40)" \
	    AIC_LOCAL_CPU=true \
	    LMCACHE_MAX_LOCAL_CPU_SIZE="$(if $(filter command line environment override,$(origin LMCACHE_MAX_LOCAL_CPU_SIZE)),$(LMCACHE_MAX_LOCAL_CPU_SIZE),8)" \
	    AIC_CLIFF_ARMS="$${AIC_CLIFF_ARMS:-nvme}" \
	    $(call _CLIFF_SUBMIT,--job-name=aic-cliff-spur-l2 \
	    --time=$(if $(AIC_CLIFF_TIME),$(AIC_CLIFF_TIME),03:00:00))) && \
	    echo "submitted cliff-spur-l2 job $$jobid" && \
	    echo "  util=0.40, DRAM L1=8GB, POSIX NVMe L2, per_client, Qwen2.5-3B, c=32, 2 iters, nvme only" && \
	    echo "log: $(CURDIR)/logs/$$jobid/cliff.out"

cliff-spur-l2-debug:         # Tiny L1 (0.1GB) + DEBUG vLLM logging to diagnose ext_hit=0
	@# c=1, 1 timed iter, 0.1GB DRAM L1 so a single client's ~630MiB prefix
	@# immediately overflows into L2.  VLLM_LOGGING_LEVEL=DEBUG enables the
	@# "vLLM hit is: N" log line in lmcache_mp_connector.py; search
	@# container-aic-vllm.log for "vLLM hit is" to confirm ext_hit fires.
	@cd "$(CURDIR)" && jobid=$$( \
	    BENCH_PREFIX_MODE=per_client \
	    BENCH_CONCUR="$${BENCH_CONCUR:-1}" \
	    BENCH_ITERS="$${BENCH_ITERS:-1}" \
	    VLLM_MODEL="$${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}" \
	    VLM_GPU_MEMORY_UTILIZATION="$(if $(filter command line environment override,$(origin VLM_GPU_MEMORY_UTILIZATION)),$(VLM_GPU_MEMORY_UTILIZATION),0.40)" \
	    AIC_LOCAL_CPU=true \
	    LMCACHE_MAX_LOCAL_CPU_SIZE="$(if $(filter command line environment override,$(origin LMCACHE_MAX_LOCAL_CPU_SIZE)),$(LMCACHE_MAX_LOCAL_CPU_SIZE),0.1)" \
	    AIC_CLIFF_ARMS="$${AIC_CLIFF_ARMS:-nvme}" \
	    VLLM_LOGGING_LEVEL=DEBUG \
	    $(call _CLIFF_SUBMIT,--job-name=aic-cliff-l2-debug \
	    --time=$(if $(AIC_CLIFF_TIME),$(AIC_CLIFF_TIME),01:00:00))) && \
	    echo "submitted cliff-spur-l2-debug job $$jobid" && \
	    echo "  util=0.40, DRAM L1=0.1GB (forces L2 hits at c=1), per_client, c=1, 1 iter, DEBUG logging" && \
	    echo "log: $(CURDIR)/logs/$$jobid/cliff.out" && \
	    echo "  check container-aic-vllm.log for 'vLLM hit is' to confirm ext_hit"


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
	    VLM_GPU_MEMORY_UTILIZATION="$(if $(filter command line environment override,$(origin VLM_GPU_MEMORY_UTILIZATION)),$(VLM_GPU_MEMORY_UTILIZATION),0.12)" \
	    VLM_YARN_FACTOR="$${VLM_YARN_FACTOR:-2.0}" VLM_MAX_MODEL_LEN="$(if $(filter command line environment override,$(origin VLM_MAX_MODEL_LEN)),$(VLM_MAX_MODEL_LEN),65536)" \
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
	    VLM_GPU_MEMORY_UTILIZATION="$(if $(filter command line environment override,$(origin VLM_GPU_MEMORY_UTILIZATION)),$(VLM_GPU_MEMORY_UTILIZATION),0.12)" \
	    VLM_YARN_FACTOR="$${VLM_YARN_FACTOR:-4.0}" VLM_MAX_MODEL_LEN="$(if $(filter command line environment override,$(origin VLM_MAX_MODEL_LEN)),$(VLM_MAX_MODEL_LEN),131072)" \
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
