# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Single source of truth for building the AIC stack images.
#
# `docker buildx bake` builds aic-base and its dependents (aic-vllm, aic-lmcache)
# in ONE BuildKit session: the `base = "target:base"` build-context wires the
# locally-built base into the downstream images directly, so the OCI-layout
# export/untar/rm dance the old hand-rolled `docker buildx build` chain needed
# (to cross the docker-container builder's daemon isolation) is gone.
#
# Version pins live here as `variable` defaults and are the authoritative pins;
# every value is overridable by a same-named environment variable, so
# `VLLM_REF=v0.29.1 docker buildx bake vllm` points the build at a different ref
# without editing this file.  The Dockerfiles keep their own `ARG X=<default>`
# lines purely as standalone fallbacks for a plain `docker build`.
#
# Paths are resolved relative to this file's directory (docker/); `context = ".."`
# makes every build's context the repo root, matching the old `"$(REPO_ROOT)"`.
#
# Quick reference:
#   docker buildx bake -f docker/docker-bake.hcl default --print   # see resolved config
#   ROCM_ARCH=gfx1201 docker buildx bake -f docker/docker-bake.hcl --load default
#   docker buildx bake -f docker/docker-bake.hcl base --print \
#       | jq -r '.target.base.tags[0]'                              # the image tag

# ---- Version pins (authoritative; override via env) --------------------------

# Keep in sync with the repo-root VERSION file (the Makefile exports AIC_VERSION
# from it, so make-driven builds always use VERSION; this default covers a
# standalone `docker buildx bake`).
variable "AIC_VERSION"   { default = "0.1.0" }

variable "ROCM_VERSION"      { default = "7.14.1" }
variable "PYTORCH_REF"       { default = "ec5f3b1effb14f0e36b66bab4db1c9b49b278f82" }
variable "PYTORCH_SERIES"    { default = "2.13" }
variable "TORCHVISION_REF"   { default = "v0.28.0" }
variable "VLLM_REF"          { default = "v0.29.0" }
variable "VLLM_TARGET_DEVICE" { default = "rocm" }
variable "LLM_EMU_REF"       { default = "b71c602" }
variable "AITER_REF"         { default = "v0.1.22.post1" }
variable "FLASH_ATTN_REF"    { default = "v2.8.3.post1" }
variable "LMCACHE_REF"       { default = "v0.5.6" }
variable "NIXL_REF"          { default = "v1.4.1" }
variable "HSA_SNOOP_REF"     { default = "v1.2.0" }

# Repository URLs — forwarded like the pins so a build can be pointed at a fork.
variable "PYTORCH_GIT_URL"     { default = "https://github.com/ROCm/pytorch.git" }
variable "TORCHVISION_GIT_URL" { default = "https://github.com/pytorch/vision.git" }
variable "VLLM_GIT_URL"        { default = "https://github.com/vllm-project/vllm.git" }
variable "LLM_EMU_GIT_URL"     { default = "https://github.com/AKafakA/llm-emu.git" }
variable "AITER_GIT_URL"       { default = "https://github.com/ROCm/aiter.git" }
variable "FLASH_ATTN_GIT_URL"  { default = "https://github.com/Dao-AILab/flash-attention.git" }
variable "LMCACHE_GIT_URL"     { default = "https://github.com/LMCache/LMCache.git" }
variable "NIXL_GIT_URL"        { default = "https://github.com/ai-dynamo/nixl.git" }
variable "HSA_SNOOP_GIT_URL"   { default = "https://github.com/sbates130272/hsa-snoop.git" }

# Fabric exporter + kvbench pins.
variable "NVME_EXPORTER_VERSION" { default = "3.0.0" }
variable "RDMA_EXPORTER_VERSION" { default = "0.7.3" }
variable "KVBENCH_GIT_URL"       { default = "https://github.com/wvaske/llm-kv-passthrough.git" }
variable "KVBENCH_REF"           { default = "177cf4e7c46ece8e03a701f89b60cd3a329f45fd" }
variable "KVBENCH_IMAGE_TAG"     { default = "177cf4e" }

# ---- Build knobs (override via env) ------------------------------------------

# ';'-, ','- or space-separated gfx list.  Empty means "let the Dockerfile ARG
# default / caller decide" and omits the arch suffix from the image tag.
variable "ROCM_ARCH"      { default = "" }
variable "BUILD_JOBS"     { default = "" }       # empty = nproc
variable "AIC_UCX_FAST"   { default = "" }

# Layer cache: a registry ref (AIC_CACHE_REF) takes precedence over a local dir
# (AIC_CACHE_DIR); with both empty there is no --cache-from/--cache-to at all.
# (AIC_CACHE_INSECURE / self-signed-registry TLS is a buildkitd-config concern
# set when the builder is created, not here.)
variable "AIC_CACHE_REF"  { default = "" }
variable "AIC_CACHE_DIR"  { default = "" }
variable "AIC_CACHE_MODE" { default = "max" }

# Corporate CA cert injected as a BuildKit secret (id=tls_cert).  Empty = none.
variable "TLS_CERT"       { default = "" }

# Where the `wheels` group writes.  Each wheel target gets its OWN subdir: bake
# does not namespace a shared local dest, so three targets pointed at one
# directory would overwrite each other's export.  `make wheels` flattens them.
variable "WHEEL_DEST"     { default = "./wheels-out" }

# ---- Tag derivation (mirrors docker/scripts/aic-image-tag.sh) ----------------

# FlashAttention is CDNA-only (gfx9xx): include it when every arch token is gfx9*
# (an empty arch list is treated as CDNA, matching the script's conservative
# default of including fa when the arch is unknown).
function "cdna_only" {
  params = [arch]
  result = length([for a in regexall("[^;, ]+", arch) : a if substr(a, 0, 4) != "gfx9"]) == 0
}

# Tag arch slug: strip the gfx prefix from each token and join with '-'
# (e.g. "gfx942;gfx950" -> "942-950"); emitted once as "gfx<slug>".
function "tag_arch_slug" {
  params = [arch]
  result = join("-", [for a in regexall("[^;, ]+", arch) : trimprefix(a, "gfx")])
}

# Cache-dir arch slug: separators -> '-' but keep the gfx prefix
# (e.g. "gfx942;gfx950" -> "gfx942-gfx950"); mirrors _ARCH_SLUG in the Makefile.
function "cache_arch_slug" {
  params = [arch]
  result = join("-", regexall("[^;, ]+", arch))
}

# The version portion of the tag (arch-independent apart from fa inclusion).
variable "_TAG_BASE" {
  default = "${AIC_VERSION}-rocm${ROCM_VERSION}-pytorch${PYTORCH_SERIES}-vllm${trimprefix(VLLM_REF, "v")}-aiter${trimprefix(AITER_REF, "v")}${cdna_only(ROCM_ARCH) ? "-fa${trimprefix(FLASH_ATTN_REF, "v")}" : ""}-lmcache${trimprefix(LMCACHE_REF, "v")}-nixl${trimprefix(NIXL_REF, "v")}-hsasnoop${trimprefix(HSA_SNOOP_REF, "v")}"
}

# Full image tag.  The arch suffix is dropped when ROCM_ARCH is empty, and
# collapses to gfxm<sha1-8> when a long multi-arch list would overflow Docker's
# 128-character tag limit.
variable "IMAGE_TAG" {
  default = ROCM_ARCH == "" ? _TAG_BASE : (
    strlen("${_TAG_BASE}-gfx${tag_arch_slug(ROCM_ARCH)}") > 128
      ? "${_TAG_BASE}-gfxm${substr(sha1(ROCM_ARCH), 0, 8)}"
      : "${_TAG_BASE}-gfx${tag_arch_slug(ROCM_ARCH)}"
  )
}

# ---- Shared helpers ----------------------------------------------------------

# Layer cache sources/exports.  Registry ref wins over a per-arch local dir;
# with neither set the cache is disabled.  Mirrors the backend precedence in
# .slurm/run-build-distribute.sh.
#
# `name` gives every target its OWN local cache subdir
# (<dir>/<arch>/<target>).  A single bake call builds all targets concurrently,
# and vllm/lmcache share base's layers; pointing them at one shared dir makes
# their cache *exports* race and fail with "ref ... locked ... unavailable".
# Per-target dirs remove the contention (at the cost of storing base's layers
# once per dependent).  Registry refs are content-addressed and handle
# concurrent writers, so they stay on a single ref.
function "cache_from" {
  params = [arch, name]
  result = AIC_CACHE_REF != "" ? ["type=registry,ref=${AIC_CACHE_REF}"] : (
           AIC_CACHE_DIR != "" ? ["type=local,src=${AIC_CACHE_DIR}/${cache_arch_slug(arch)}/${name}"] : [])
}
function "cache_to" {
  params = [arch, name]
  result = AIC_CACHE_REF != "" ? ["type=registry,ref=${AIC_CACHE_REF},mode=${AIC_CACHE_MODE}"] : (
           AIC_CACHE_DIR != "" ? ["type=local,dest=${AIC_CACHE_DIR}/${cache_arch_slug(arch)}/${name},mode=${AIC_CACHE_MODE},ignore-error=true"] : [])
}

# Corporate CA secret, only when TLS_CERT is set.
function "tls_secret" {
  params = []
  result = TLS_CERT == "" ? [] : ["id=tls_cert,src=${TLS_CERT}"]
}

# ---- Groups ------------------------------------------------------------------

# `make build` bakes this group: the full local stack including the fabric
# exporters, so a fresh build can run the monitored compose stack
# (`make up` / `vllm-reset-test`) without a separate exporter build.  kvbench is
# a standalone benchmarking tool and stays out (see the `all` group).
group "default" {
  targets = ["base", "vllm", "lmcache", "hsa-snoop", "nvme-exporter", "rdma-exporter"]
}

group "exporters" {
  targets = ["nvme-exporter", "rdma-exporter"]
}

group "all" {
  targets = ["base", "vllm", "lmcache", "hsa-snoop", "nvme-exporter", "rdma-exporter", "kvbench"]
}

# Everything the nightly wheels release publishes.  Exports .whl files to
# WHEEL_DEST instead of producing images; see `make wheels`.
#
# NIXL appears once: it is built in aic-base and inherited by both aic-vllm and
# aic-lmcache, so there is a single nixl_rocm wheel for the whole stack.
group "wheels" {
  targets = ["nixl-wheels", "vllm-wheels", "lmcache-wheels-export"]
}

# ---- Targets -----------------------------------------------------------------

target "base" {
  context    = "."
  dockerfile = "docker/base/Dockerfile"
  tags       = ["aic-base:${IMAGE_TAG}"]
  args = {
    ROCM_ARCH           = ROCM_ARCH
    ROCM_VERSION        = ROCM_VERSION
    BUILD_JOBS          = BUILD_JOBS
    AIC_UCX_FAST        = AIC_UCX_FAST
    PYTORCH_GIT_URL     = PYTORCH_GIT_URL
    PYTORCH_REF         = PYTORCH_REF
    PYTORCH_SERIES      = PYTORCH_SERIES
    TORCHVISION_GIT_URL = TORCHVISION_GIT_URL
    TORCHVISION_REF     = TORCHVISION_REF
    NIXL_GIT_URL        = NIXL_GIT_URL
    NIXL_REF            = NIXL_REF
    # Busts the arch-specific ROCm BLAS install layer on every build.
    BLAS_CACHE_BUST     = timestamp()
  }
  cache-from = cache_from(ROCM_ARCH, "base")
  cache-to   = cache_to(ROCM_ARCH, "base")
  secret     = tls_secret()
}

target "vllm" {
  context    = "."
  dockerfile = "docker/vllm/Dockerfile"
  # The default (runtime) stage installs aic-lmcache's wheel from the
  # lmcache-wheels context and stops without one.
  contexts   = { base = "target:base", "lmcache-wheels" = "target:lmcache-wheels" }
  tags       = ["aic-vllm:${IMAGE_TAG}", "aic-vllm:latest"]
  args = {
    ROCM_ARCH          = ROCM_ARCH
    ROCM_VERSION       = ROCM_VERSION
    BUILD_JOBS         = BUILD_JOBS
    AIC_VERSION        = AIC_VERSION
    VLLM_GIT_URL       = VLLM_GIT_URL
    VLLM_REF           = VLLM_REF
    VLLM_TARGET_DEVICE = VLLM_TARGET_DEVICE
    AITER_GIT_URL      = AITER_GIT_URL
    AITER_REF          = AITER_REF
    FLASH_ATTN_GIT_URL = FLASH_ATTN_GIT_URL
    FLASH_ATTN_REF     = FLASH_ATTN_REF
    LLM_EMU_GIT_URL    = LLM_EMU_GIT_URL
    LLM_EMU_REF        = LLM_EMU_REF
    LMCACHE_REF        = LMCACHE_REF
  }
  cache-from = cache_from(ROCM_ARCH, "vllm")
  cache-to   = cache_to(ROCM_ARCH, "vllm")
  secret     = tls_secret()
}

target "lmcache" {
  context    = "."
  dockerfile = "docker/lmcache/Dockerfile"
  contexts   = { base = "target:base" }
  tags       = ["aic-lmcache:${IMAGE_TAG}", "aic-lmcache:latest"]
  args = {
    ROCM_ARCH       = ROCM_ARCH
    ROCM_VERSION    = ROCM_VERSION
    BUILD_JOBS      = BUILD_JOBS
    AIC_VERSION     = AIC_VERSION
    LMCACHE_GIT_URL = LMCACHE_GIT_URL
    LMCACHE_REF     = LMCACHE_REF
    NIXL_GIT_URL    = NIXL_GIT_URL
    NIXL_REF        = NIXL_REF
    AITER_REF       = AITER_REF
    FLASH_ATTN_REF  = FLASH_ATTN_REF
    HSA_SNOOP_REF   = HSA_SNOOP_REF
  }
  cache-from = cache_from(ROCM_ARCH, "lmcache")
  cache-to   = cache_to(ROCM_ARCH, "lmcache")
  secret     = tls_secret()
}

# aic-lmcache's /wheels, for aic-vllm.  Reads the lmcache cache but does not
# write it: that dir keeps a single ref, and it should stay lmcache's full chain.
target "lmcache-wheels" {
  inherits = ["lmcache"]
  target   = "wheels"
  tags     = []
  cache-to = []
}

# ---- Wheel export targets ----------------------------------------------------
# Each carries its own `output`, so they are driven WITHOUT --load/--push:
#   WHEEL_DEST=/tmp/w docker buildx bake -f docker/docker-bake.hcl wheels
#
# `cache-to = []` on every one of them: a wheel target shares its whole layer
# chain with the image target it inherits from, and two concurrent writers on
# one local cache dir fail with "ref ... locked ... unavailable".  They read the
# image targets' cache and leave writing it to those targets.

# The common nixl_rocm wheel, built once in aic-base.
target "nixl-wheels" {
  inherits = ["base"]
  target   = "nixl-wheels"
  tags     = []
  output   = ["type=local,dest=${WHEEL_DEST}/nixl"]
  cache-to = []
}

# vllm + amd_aiter (+ flash_attn on CDNA-only arch sets).  It inherits vllm's
# `lmcache-wheels` context too -- `inherits` merges context maps rather than
# replacing them -- but that costs nothing: the group builds
# lmcache-wheels-export from the same chain anyway, so BuildKit dedupes it.
target "vllm-wheels" {
  inherits = ["vllm"]
  target   = "wheels"
  tags     = []
  output   = ["type=local,dest=${WHEEL_DEST}/vllm"]
  cache-to = []
}

# The patched lmcache wheel.  Distinct from the `lmcache-wheels` target above,
# which must stay export-free so it can serve as aic-vllm's in-session context.
target "lmcache-wheels-export" {
  inherits = ["lmcache-wheels"]
  output   = ["type=local,dest=${WHEEL_DEST}/lmcache"]
}

target "hsa-snoop" {
  context    = "."
  dockerfile = "docker/hsa-snoop/Dockerfile"
  tags       = ["aic-hsa-snoop:${IMAGE_TAG}", "aic-hsa-snoop:latest"]
  args = {
    ROCM_VERSION     = ROCM_VERSION
    BUILD_JOBS       = BUILD_JOBS
    HSA_SNOOP_GIT_URL = HSA_SNOOP_GIT_URL
    HSA_SNOOP_REF    = HSA_SNOOP_REF
  }
  cache-from = cache_from(ROCM_ARCH, "hsa-snoop")
  cache-to   = cache_to(ROCM_ARCH, "hsa-snoop")
  secret     = tls_secret()
}

# CPU-only emulation image: vLLM scheduler/HTTP stack with the forward pass
# replaced by a profile-pack latency draw.  Tagged separately so it never
# overwrites the real GPU aic-vllm image.
target "emulate" {
  inherits = ["vllm"]
  target   = "emulate"
  tags     = ["aic-vllm-emulate:${IMAGE_TAG}"]
  args = {
    VLLM_TARGET_DEVICE = "empty"
  }
}

target "nvme-exporter" {
  context    = "."
  dockerfile = "docker/nvme-exporter/Dockerfile"
  tags       = ["aic-nvme-exporter:local"]
  args = {
    NVME_EXPORTER_VERSION = NVME_EXPORTER_VERSION
  }
}

target "rdma-exporter" {
  context    = "."
  dockerfile = "docker/rdma-exporter/Dockerfile"
  tags       = ["aic-rdma-exporter:local"]
  args = {
    RDMA_EXPORTER_VERSION = RDMA_EXPORTER_VERSION
  }
}

target "kvbench" {
  context    = "."
  dockerfile = "docker/kvbench/Dockerfile"
  tags       = ["aic-kvbench:${KVBENCH_IMAGE_TAG}"]
  args = {
    KVBENCH_GIT_URL = KVBENCH_GIT_URL
    KVBENCH_REF     = KVBENCH_REF
  }
}
