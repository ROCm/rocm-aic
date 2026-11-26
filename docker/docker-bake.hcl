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
# Every build input (version pins, git URLs, base images, tool versions, image
# tags, cache settings) is a `variable` default here; each is overridable by a
# same-named environment variable, so
# `VLLM_REF=v0.29.1 docker buildx bake vllm` points the build at a different ref
# without editing this file.  Everything outside bake (make, Slurm scripts, CI,
# the README sync) reads the resolved values through
# docker/scripts/aic-bake-env.sh, which wraps `docker buildx bake --print`.
#
# The Dockerfiles hold no copy: an `ARG` this file supplies has no default
# there, and each consuming stage stops the build when one arrives unset, so a
# Dockerfile is built through bake or with bake's resolved args:
#   mapfile -t args < <(docker/scripts/aic-bake-env.sh --build-args <target>)
#   docker buildx build "${args[@]}" -f docker/<dir>/Dockerfile ...
# tests/test_aic_bake_env.py fails when a default reappears.  The exceptions
# are docker/kvbench/Dockerfile and docker/client/Dockerfile, which compose
# also builds; the same test keeps their defaults equal to the values here.
#
# Run bake from the repo root: every target uses `context = "."`, which bake
# resolves against the invoking directory, not against this file.
#
# Quick reference:
#   docker buildx bake -f docker/docker-bake.hcl default --print   # see resolved config
#   ROCM_ARCH=gfx1201 docker buildx bake -f docker/docker-bake.hcl --load default
#   docker/scripts/aic-bake-env.sh --env                            # resolved pins, tag, image refs

# ---- Version pins (authoritative; override via env) --------------------------

# The repo-root VERSION file owns this; make and aic-bake-env.sh export it from
# there.  Bake cannot read a file, so this literal is the fallback for a
# standalone `docker buildx bake` and tests/test_aic_bake_env.py keeps it equal.
variable "AIC_VERSION"   { default = "0.1.0" }

variable "ROCM_VERSION"      { default = "10.2.0" }
# ROCm/pytorch publishes no tag for the 2.15 series, so a SHA is the only
# reproducible ref.  abecaa8 is the head of release/2.15 as of 2026-11-17.
variable "PYTORCH_REF"       { default = "abecaa87238d2d43d2641a22d89a51de185779b2" }
# Display only: the torch series in the image tag, README and release notes (a
# bare SHA says nothing about which torch this is), and the CPU torch the
# emulate builds install.  Bump alongside PYTORCH_REF.
variable "PYTORCH_SERIES"    { default = "2.15" }
variable "TORCHVISION_REF"   { default = "v0.29.2" }
variable "VLLM_REF"          { default = "v0.34.0" }
variable "VLLM_TARGET_DEVICE" { default = "rocm" }
variable "LLM_EMU_REF"       { default = "b71c602" }
variable "AITER_REF"         { default = "v0.1.29.post1" }
variable "FLASH_ATTN_REF"    { default = "0e60e394" }
variable "LMCACHE_REF"       { default = "v0.5.8" }
variable "NIXL_REF"          { default = "v1.5.2" }
# MORI supplies UMBP: vLLM's UMBPStoreConnector, umbp_standalone_server and
# umbp_master.  The ref follows vLLM's Dockerfile.rocm_base (v0.34.0).
variable "MORI_REF"          { default = "v1.2.3.post1" }
# cupy-rocm-7-0 on both ends of the HIP IPC path (aic-lmcache and aic-vllm).
variable "CUPY_VERSION"      { default = "14.2.0" }
# UCX is NIXL's transport.  v1.22.0 is the latest stable release; upstream NIXL
# v1.5.2 CI builds against the pre-release v1.23.x branch.
variable "UCX_REF"           { default = "v1.22.0" }
# Unreleased commit on feat/cpu-deload: v1.6.0 stalls its ring poller under
# --mem-snoop and cannot read the queue rings of a ROCm 10.2 process.  Move to
# the tag once cut.
variable "HSA_SNOOP_REF"     { default = "aba6a306" }

# Base images.  ROCM_BASE_IMAGE is the full ROCm toolchain the source builds
# run on; ROCM_RUNTIME_IMAGE is what the shipped images start from.
variable "ROCM_BASE_IMAGE"    { default = "rocm/dev-ubuntu-26.04:${ROCM_VERSION}-full" }
variable "ROCM_RUNTIME_IMAGE" { default = "ubuntu:26.04" }

# Source of the patched libamdhip64 built into aic-base.  A literal, not derived
# from ROCM_VERSION: the 10.2 branch is release/therock-10.2, with no patch level.
variable "CLR_REF"            { default = "release/therock-10.2" }
# Triton is source-built for aic-base; the ref follows vLLM's Dockerfile.rocm_base
# (v0.34.0; branch head 669b31ac when pinned).
variable "TRITON_REF"         { default = "release/internal/3.8.x" }

# Repository URLs — forwarded like the pins so a build can be pointed at a fork.
variable "PYTORCH_GIT_URL"     { default = "https://github.com/ROCm/pytorch.git" }
variable "TORCHVISION_GIT_URL" { default = "https://github.com/pytorch/vision.git" }
variable "VLLM_GIT_URL"        { default = "https://github.com/vllm-project/vllm.git" }
variable "LLM_EMU_GIT_URL"     { default = "https://github.com/AKafakA/llm-emu.git" }
variable "AITER_GIT_URL"       { default = "https://github.com/ROCm/aiter.git" }
variable "FLASH_ATTN_GIT_URL"  { default = "https://github.com/Dao-AILab/flash-attention.git" }
variable "LMCACHE_GIT_URL"     { default = "https://github.com/LMCache/LMCache.git" }
variable "NIXL_GIT_URL"        { default = "https://github.com/ai-dynamo/nixl.git" }
variable "MORI_GIT_URL"        { default = "https://github.com/ROCm/mori.git" }
variable "UCX_GIT_URL"         { default = "https://github.com/openucx/ucx.git" }
variable "HSA_SNOOP_GIT_URL"   { default = "https://github.com/sbates130272/hsa-snoop.git" }
variable "CLR_GIT_URL"         { default = "https://github.com/ROCm/rocm-systems.git" }
variable "TRITON_GIT_URL"      { default = "https://github.com/ROCm/triton.git" }

# Fabric exporter + kvbench pins.
variable "NVME_EXPORTER_VERSION" { default = "3.0.0" }
variable "RDMA_EXPORTER_VERSION" { default = "0.7.5" }
variable "KVBENCH_GIT_URL"       { default = "https://github.com/wvaske/llm-kv-passthrough.git" }
variable "KVBENCH_REF"           { default = "177cf4e7c46ece8e03a701f89b60cd3a329f45fd" }
variable "KVBENCH_IMAGE_TAG"     { default = substr(KVBENCH_REF, 0, 7) }
# PyPI version, not a git ref: kvbench's [lmcache] extra is `~=0.5.1` and 0.5.7
# removed lmcache.v1.cache_engine, so this does not follow LMCACHE_REF.
variable "KVBENCH_LMCACHE_VERSION" { default = "0.5.6" }

# Benchmark client image (no ROCm dependency).  aiperf is pinned to the release
# benchmarks/agentx/ was run against: its flags and record layout move.
variable "AIPERF_VERSION"        { default = "0.13.0" }
# A commit: agent-kv has no tags.
variable "AGENT_KV_GIT_URL"      { default = "https://github.com/sbates130272/agent-kv" }
variable "AGENT_KV_REF"          { default = "19fcc86be6e267839889567ce5d20dd0fed6bbaf" }

# ---- Build knobs (override via env) ------------------------------------------

# ';'-, ','- or space-separated gfx list.  Empty omits the arch suffix from the
# image tag and reaches the Dockerfiles as an empty ROCM_ARCH (they carry no
# default), so set it for any real build.
variable "ROCM_ARCH"      { default = "" }
variable "BUILD_JOBS"     { default = "" }       # empty = nproc
variable "AIC_UCX_FAST"   { default = "" }

# 1 compiles AITER's kernels into the vllm image at build time instead of on
# first use; only valid for a single-arch build on a node with that GPU.
variable "AITER_PREBUILD_KERNELS" { default = "0" }

# Cache key of the ROCm runtime-package layer in aic-base.  Change it (e.g.
# BLAS_CACHE_BUST=$(date +%s)) to reinstall those packages; every image built
# on aic-base is then re-assembled too.
variable "BLAS_CACHE_BUST" { default = "" }

# Layer cache: a registry ref (AIC_CACHE_REF) takes precedence over a GitHub
# Actions cache scope prefix (AIC_CACHE_GHA_SCOPE), which takes precedence over a
# local dir (AIC_CACHE_DIR); with all three empty there is no
# --cache-from/--cache-to at all.
#
# AIC_CACHE_GHA_SCOPE exists for small GitHub-hosted builds only.  Do NOT point
# the wheels or image builds at it: GitHub caps the Actions cache at 10 GB per
# repo, far under this stack's layer set, which is why the nightly wheels
# workflow builds on Slurm against AIC_CACHE_DIR instead.
# (AIC_CACHE_INSECURE / self-signed-registry TLS is a buildkitd-config concern
# set when the builder is created, not here.)
variable "AIC_CACHE_REF"       { default = "" }
variable "AIC_CACHE_GHA_SCOPE" { default = "" }
variable "AIC_CACHE_DIR"       { default = "" }
variable "AIC_CACHE_MODE"      { default = "max" }

# Corporate CA cert injected as a BuildKit secret (id=tls_cert).  Empty = none.
variable "TLS_CERT"       { default = "" }

# Where the `wheels` group writes the pip wheels: one subdirectory per wheel
# target, relative to the directory bake runs in.  Each target gets its OWN
# subdir because bake does not namespace a shared local dest -- three targets
# pointed at one directory would overwrite each other's export.  `make wheels`
# flattens them back into one directory.
variable "WHEELS_DIR"     { default = "wheels-out" }

# Registry prefix for every image tag, e.g. "ghcr.io/rocm/rocm-aic/".  Empty
# (the default) keeps the bare local names a developer expects from `--load`.
# `--push` needs registry-qualified tags on EVERY tag of a target, so when this
# is set it replaces the bare names rather than adding to them -- one
# unqualified tag in the list fails the whole push.
# Must include the trailing '/'.
variable "AIC_IMAGE_PREFIX" { default = "" }

# Optional short alias pushed alongside the full version tag, e.g. "gfx1250".
# The version tag is 100+ characters and is derived from a dozen pins, so a
# consumer that just wants "the current gfx1250 image" cannot name it without
# recomputing the tag.  CI pushes both and pulls the alias.
variable "AIC_IMAGE_EXTRA_TAG" { default = "" }

# Tag list for an image: the version tag, plus the alias when one is set.
# `latest` is only useful locally and would clobber across arches in a shared
# registry, so it is dropped as soon as a prefix is in play.
function "tags" {
  params = [name, latest]
  result = (AIC_IMAGE_PREFIX == ""
    ? concat(["${name}:${IMAGE_TAG}"], (latest ? ["${name}:latest"] : []))
    : concat(["${AIC_IMAGE_PREFIX}${name}:${IMAGE_TAG}"],
             (AIC_IMAGE_EXTRA_TAG == "" ? [] : ["${AIC_IMAGE_PREFIX}${name}:${AIC_IMAGE_EXTRA_TAG}"])))
}

# Same, for images whose tag is a fixed string rather than the version tag
# (the exporters' ":local", kvbench's pinned tag, the client's AIC version,
# hsa-snoop's own ref).
function "fixed_tags" {
  params = [name, tag]
  result = (AIC_IMAGE_PREFIX == ""
    ? ["${name}:${tag}"]
    : concat(["${AIC_IMAGE_PREFIX}${name}:${tag}"],
             (AIC_IMAGE_EXTRA_TAG == "" ? [] : ["${AIC_IMAGE_PREFIX}${name}:${AIC_IMAGE_EXTRA_TAG}"])))
}

# ---- Tag derivation (docker/scripts/aic-image-tag.sh prints the result) ------

# FlashAttention is CDNA-only (gfx9xx): include it when at least one arch token
# is gfx9*, in which case docker/vllm/Dockerfile compiles it for just those (an
# empty arch list is treated as CDNA: include fa when the arch is unknown).
function "has_gfx9" {
  params = [arch]
  result = length(regexall("[^;, ]+", arch)) == 0 || length([for a in regexall("[^;, ]+", arch) : a if substr(a, 0, 4) == "gfx9"]) > 0
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

# The ROCM_ARCH build-arg every Dockerfile receives: the same list re-joined with
# ';' (e.g. "gfx942,gfx950" -> "gfx942;gfx950").  The Dockerfiles hand it to
# PYTORCH_ROCM_ARCH (which torch's extension builder splits on ';' and spaces
# only), to vLLM's CMake (';' only) and to the FlashAttention CDNA check (';'
# only), so a ','-separated list accepted above would otherwise reach them as one
# bogus arch.  A ';'-separated list passes through unchanged.
function "rocm_arch_list" {
  params = [arch]
  result = join(";", regexall("[^;, ]+", arch))
}

# The version portion of the tag (arch-independent apart from fa inclusion).
# hsa-snoop is not part of it: that image shares nothing with these, so its tag
# is its own ref and a bump of either side leaves the other's tag alone.
variable "_TAG_BASE" {
  default = "${AIC_VERSION}-rocm${ROCM_VERSION}-pytorch${PYTORCH_SERIES}-vllm${trimprefix(VLLM_REF, "v")}-aiter${trimprefix(AITER_REF, "v")}${has_gfx9(ROCM_ARCH) ? "-fa${trimprefix(FLASH_ATTN_REF, "v")}" : ""}-lmcache${trimprefix(LMCACHE_REF, "v")}-nixl${trimprefix(NIXL_REF, "v")}"
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
# once per dependent).  A registry stores blobs once however many refs use
# them, but a ref still names ONE cache manifest, so concurrent writers to one
# ref would overwrite each other's: it gets the same per-arch, per-target split
# (<ref>-<arch>-<target>), as does a GitHub Actions cache scope
# (<scope>-<arch>-<target>), which also keeps two arch groups from sharing
# entries.  ignore-error: a cache-service hiccup must not fail an hours-long build.
function "cache_from" {
  params = [arch, name]
  result = AIC_CACHE_REF != "" ? ["type=registry,ref=${AIC_CACHE_REF}-${cache_arch_slug(arch)}-${name}"] : (
           AIC_CACHE_GHA_SCOPE != "" ? ["type=gha,scope=${AIC_CACHE_GHA_SCOPE}-${cache_arch_slug(arch)}-${name}"] : (
           AIC_CACHE_DIR != "" ? ["type=local,src=${AIC_CACHE_DIR}/${cache_arch_slug(arch)}/${name}"] : []))
}
function "cache_to" {
  params = [arch, name]
  result = AIC_CACHE_REF != "" ? ["type=registry,ref=${AIC_CACHE_REF}-${cache_arch_slug(arch)}-${name},mode=${AIC_CACHE_MODE},ignore-error=true"] : (
           AIC_CACHE_GHA_SCOPE != "" ? ["type=gha,scope=${AIC_CACHE_GHA_SCOPE}-${cache_arch_slug(arch)}-${name},mode=${AIC_CACHE_MODE},ignore-error=true"] : (
           AIC_CACHE_DIR != "" ? ["type=local,dest=${AIC_CACHE_DIR}/${cache_arch_slug(arch)}/${name},mode=${AIC_CACHE_MODE},ignore-error=true"] : []))
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
# WHEELS_DIR instead of producing images; see `make wheels`.
#
# NIXL appears once: it is built in aic-base and inherited by both aic-vllm and
# aic-lmcache, so there is a single nixl_rocm wheel for the whole stack.
group "wheels" {
  targets = ["wheels-base", "wheels-lmcache", "wheels-vllm"]
}

# CPU-only emulate wheel set: vllm (empty device) + vllm_emulator.
# No ROCm dependency; builds on python:3.12-bookworm for use on any x86_64 host.
# Drive with: WHEELS_DIR=/tmp/emu docker buildx bake -f docker/docker-bake.hcl emulate-wheels
group "emulate-wheels" {
  targets = ["wheels-emulate"]
}

# ---- Targets -----------------------------------------------------------------

target "base" {
  context    = "."
  dockerfile = "docker/base/Dockerfile"
  tags       = tags("aic-base", false)
  args = {
    ROCM_ARCH           = rocm_arch_list(ROCM_ARCH)
    ROCM_VERSION        = ROCM_VERSION
    ROCM_BASE_IMAGE     = ROCM_BASE_IMAGE
    ROCM_RUNTIME_IMAGE  = ROCM_RUNTIME_IMAGE
    BUILD_JOBS          = BUILD_JOBS
    AIC_UCX_FAST        = AIC_UCX_FAST
    PYTORCH_GIT_URL     = PYTORCH_GIT_URL
    PYTORCH_REF         = PYTORCH_REF
    PYTORCH_SERIES      = PYTORCH_SERIES
    TORCHVISION_GIT_URL = TORCHVISION_GIT_URL
    TORCHVISION_REF     = TORCHVISION_REF
    NIXL_GIT_URL        = NIXL_GIT_URL
    NIXL_REF            = NIXL_REF
    UCX_GIT_URL         = UCX_GIT_URL
    UCX_REF             = UCX_REF
    CLR_GIT_URL         = CLR_GIT_URL
    CLR_REF             = CLR_REF
    TRITON_GIT_URL      = TRITON_GIT_URL
    TRITON_REF          = TRITON_REF
    BLAS_CACHE_BUST     = BLAS_CACHE_BUST
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
  contexts   = { base = "target:base", "lmcache-wheels" = "target:lmcache-wheels", "base-wheels":  "target:base-wheels" }
  tags       = tags("aic-vllm", true)
  args = {
    ROCM_ARCH          = rocm_arch_list(ROCM_ARCH)
    ROCM_VERSION       = ROCM_VERSION
    ROCM_BASE_IMAGE    = ROCM_BASE_IMAGE
    BUILD_JOBS         = BUILD_JOBS
    AIC_VERSION        = AIC_VERSION
    VLLM_GIT_URL       = VLLM_GIT_URL
    VLLM_REF           = VLLM_REF
    VLLM_TARGET_DEVICE = VLLM_TARGET_DEVICE
    AITER_GIT_URL      = AITER_GIT_URL
    AITER_REF          = AITER_REF
    AITER_PREBUILD_KERNELS = AITER_PREBUILD_KERNELS
    FLASH_ATTN_GIT_URL = FLASH_ATTN_GIT_URL
    FLASH_ATTN_REF     = FLASH_ATTN_REF
    LLM_EMU_GIT_URL    = LLM_EMU_GIT_URL
    LLM_EMU_REF        = LLM_EMU_REF
    LMCACHE_REF        = LMCACHE_REF
    MORI_GIT_URL       = MORI_GIT_URL
    MORI_REF           = MORI_REF
    CUPY_VERSION       = CUPY_VERSION
    PYTORCH_SERIES     = PYTORCH_SERIES
  }
  cache-from = cache_from(ROCM_ARCH, "vllm")
  cache-to   = cache_to(ROCM_ARCH, "vllm")
  secret     = tls_secret()
}

target "lmcache" {
  context    = "."
  dockerfile = "docker/lmcache/Dockerfile"
  contexts   = { base = "target:base", "base-wheels":  "target:base-wheels" }
  tags       = tags("aic-lmcache", true)
  args = {
    ROCM_ARCH       = rocm_arch_list(ROCM_ARCH)
    ROCM_VERSION    = ROCM_VERSION
    ROCM_BASE_IMAGE = ROCM_BASE_IMAGE
    BUILD_JOBS      = BUILD_JOBS
    AIC_VERSION     = AIC_VERSION
    LMCACHE_GIT_URL = LMCACHE_GIT_URL
    LMCACHE_REF     = LMCACHE_REF
    CUPY_VERSION    = CUPY_VERSION
    NIXL_REF        = NIXL_REF
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

target "base-wheels" {
  inherits = ["base"]
  target   = "wheels"
  tags     = []
  cache-to = []
}

# ---- Published wheel exports (the `wheels` group) ----------------------------
# Each exports one Dockerfile's `wheels` stage to ${WHEELS_DIR}/<name>; none of
# those stages is part of a runtime image.  They carry their own `output`, so the
# group is driven WITHOUT --load/--push:
#   WHEELS_DIR=/tmp/w docker buildx bake -f docker/docker-bake.hcl wheels
#
# wheels-lmcache is lmcache-wheels with an output: lmcache-wheels itself stays
# output-less because aic-vllm consumes it as a build context.
#
# Cache: wheels-lmcache/wheels-vllm read their parent image's cache as well as
# their own (a host that already built the images reuses it) and write their own,
# because the `wheels` group never builds aic-lmcache/aic-vllm.  wheels-base
# writes none: its stages are a subset of aic-base's, which the group always
# builds (as the `base` context of the other two) and caches under "base".
target "wheels-base" {
  inherits = ["base-wheels"]
  output   = ["type=local,dest=${WHEELS_DIR}/base"]
}

target "wheels-lmcache" {
  inherits   = ["lmcache-wheels"]
  output     = ["type=local,dest=${WHEELS_DIR}/lmcache"]
  cache-from = distinct(concat(cache_from(ROCM_ARCH, "lmcache"), cache_from(ROCM_ARCH, "wheels-lmcache")))
  cache-to   = cache_to(ROCM_ARCH, "wheels-lmcache")
}

target "wheels-vllm" {
  inherits   = ["vllm"]
  target     = "wheels"
  tags       = []
  output     = ["type=local,dest=${WHEELS_DIR}/vllm"]
  cache-from = distinct(concat(cache_from(ROCM_ARCH, "vllm"), cache_from(ROCM_ARCH, "wheels-vllm")))
  cache-to   = cache_to(ROCM_ARCH, "wheels-vllm")
}

target "wheels-emulate" {
  context    = "."
  dockerfile = "docker/wheels-emulate/Dockerfile"
  target     = "wheels"
  tags       = []
  output     = ["type=local,dest=${WHEELS_DIR}/emulate"]
  args = {
    VLLM_GIT_URL    = VLLM_GIT_URL
    VLLM_REF        = VLLM_REF
    LLM_EMU_GIT_URL = LLM_EMU_GIT_URL
    LLM_EMU_REF     = LLM_EMU_REF
    PYTORCH_SERIES  = PYTORCH_SERIES
    BUILD_JOBS      = BUILD_JOBS
  }
  cache-from = cache_from("emulate", "wheels")
  cache-to   = cache_to("emulate", "wheels")
  secret     = tls_secret()
}

target "hsa-snoop" {
  context    = "."
  dockerfile = "docker/hsa-snoop/Dockerfile"
  tags       = concat(fixed_tags("aic-hsa-snoop", trimprefix(HSA_SNOOP_REF, "v")),
                      AIC_IMAGE_PREFIX == "" ? ["aic-hsa-snoop:latest"] : [])
  args = {
    ROCM_VERSION       = ROCM_VERSION
    ROCM_BUILD_IMAGE   = ROCM_BASE_IMAGE
    ROCM_RUNTIME_IMAGE = ROCM_RUNTIME_IMAGE
    BUILD_JOBS         = BUILD_JOBS
    HSA_SNOOP_GIT_URL  = HSA_SNOOP_GIT_URL
    HSA_SNOOP_REF      = HSA_SNOOP_REF
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
  tags     = tags("aic-vllm-emulate", false)
  args = {
    VLLM_TARGET_DEVICE = "empty"
  }
}

target "nvme-exporter" {
  context    = "."
  dockerfile = "docker/nvme-exporter/Dockerfile"
  tags       = fixed_tags("aic-nvme-exporter", "local")
  args = {
    NVME_EXPORTER_VERSION = NVME_EXPORTER_VERSION
  }
  cache-from = cache_from("noarch", "nvme-exporter")
  cache-to   = cache_to("noarch", "nvme-exporter")
}

target "rdma-exporter" {
  context    = "."
  dockerfile = "docker/rdma-exporter/Dockerfile"
  tags       = fixed_tags("aic-rdma-exporter", "local")
  args = {
    RDMA_EXPORTER_VERSION = RDMA_EXPORTER_VERSION
  }
  cache-from = cache_from("noarch", "rdma-exporter")
  cache-to   = cache_to("noarch", "rdma-exporter")
}

target "kvbench" {
  context    = "."
  dockerfile = "docker/kvbench/Dockerfile"
  tags       = fixed_tags("aic-kvbench", KVBENCH_IMAGE_TAG)
  args = {
    KVBENCH_GIT_URL         = KVBENCH_GIT_URL
    KVBENCH_REF             = KVBENCH_REF
    KVBENCH_LMCACHE_VERSION = KVBENCH_LMCACHE_VERSION
  }
  cache-from = cache_from("noarch", "kvbench")
  cache-to   = cache_to("noarch", "kvbench")
}

# Benchmark client: the benchmarks/ scripts, agent-kv and aiperf behind
# `aic-bench`.  No ROCm dependency.
# Build: docker buildx bake -f docker/docker-bake.hcl client --load
# Other aiperf release: AIPERF_VERSION=<version> docker buildx bake -f docker/docker-bake.hcl client --load
target "client" {
  context    = "."
  dockerfile = "docker/client/Dockerfile"
  # :latest only locally, as in tags().
  tags       = concat(fixed_tags("aic-client", AIC_VERSION),
                      AIC_IMAGE_PREFIX == "" ? ["aic-client:latest"] : [])
  args = {
    AIPERF_VERSION   = AIPERF_VERSION
    AGENT_KV_GIT_URL = AGENT_KV_GIT_URL
    AGENT_KV_REF     = AGENT_KV_REF
  }
  cache-from = cache_from("noarch", "client")
  cache-to   = cache_to("noarch", "client")
}
