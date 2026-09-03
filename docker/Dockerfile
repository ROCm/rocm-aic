# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# AIC: Ubuntu 24.04 + ROCm 7.14 inference stack.
#
#   Base image: rocm/dev-ubuntu-24.04:${ROCM_VERSION}-full  (Ubuntu 24.04, Python 3.12)
#   vLLM:       built from source at ${VLLM_REF} (torch 2.13.0+rocm7.2 from pytorch.org)
#   LMCache:    LMCache/LMCache ${LMCACHE_REF} + AMD patches from patches/lmcache/
#   hipFile:    included in the ROCm 7.14 base image (GA since 7.14) — no source build needed
#   NIXL:       ai-dynamo/nixl ${NIXL_REF} + nixl-rocm-ais-mt.patch (ROCm/HIP + AIS_MT)
#   llm-emu:    profile-driven emulation plugin (inert unless VLLM_EMULATOR_ENABLE_ORACLE=1)
#
# Two compose services share this single image:
#   lmcache — standalone lmcache-server (MP mode)
#               L1:  GPU/CPU DRAM (--l1-size-gb)  OR  hipFile NVMe slab (GDS L1 mode)
#               L2a: nixl_store AIS_MT → local NVMe (hipFile P2PDMA)
#               L2b: nixl_store POSIX  → NFS-over-RDMA
#   vllm    — vllm serve with LMCacheMPConnector (ZMQ to lmcache)
#
# Build (from the repo root — the build context is this tree):
#   DOCKER_BUILDKIT=1 docker build \
#     --build-arg AIC_VERSION="$(<VERSION)" \
#     --build-arg ROCM_ARCH=gfx942 \
#     -f Dockerfile \
#     -t rocm-aic .
#
#   # Behind AMD Zscaler proxy — pass cert as a BuildKit secret (never baked into image):
#   DOCKER_BUILDKIT=1 docker build \
#     --build-arg AIC_VERSION="$(<VERSION)" \
#     --build-arg ROCM_ARCH=gfx942 \
#     --secret id=tls_cert,src=/path/to/corp-ca.crt \
#     -f Dockerfile \
#     -t rocm-aic .
#
#   # Emulation-only image (CPU nodes; no GPU kernels compiled at all):
#   DOCKER_BUILDKIT=1 docker build \
#     --target emulate \
#     --build-arg ROCM_ARCH=gfx942 \
#     --build-arg VLLM_TARGET_DEVICE=empty \
#     -f Dockerfile \
#     -t rocm-aic:emulate .
#
# Orchestration:
#   make build                                  # no proxy
#   make build TLS_CERT=/path/to/corp-ca.crt   # behind proxy
#   make dist-build-emulate                     # emulation-only image on a Slurm CPU node
#   make emulate-test                           # serve + assert a completion on a CPU node
#
# Patches: every *.patch under patches/vllm/, patches/lmcache/, patches/nixl/
# and patches/llm-emu/ is applied to its upstream checkout during the build, in
# lexical (filename) order -- hence the NN- prefixes.  Drop a new NN-*.patch
# into the directory to include it.
#
# Wheels: the build also produces pip-installable wheels for vLLM (source-built),
# the patched LMCache, and ROCm NIXL under /wheels.  Extract them with the `wheels` stage:
#   docker build --build-arg AIC_VERSION="$(<VERSION)" \
#     --target wheels --output type=local,dest=./wheels .
# (the default target remains the full runtime image -- see stages 8/9 below).

# Stages (all strictly linear, so layer caching is unaffected by the split):
#
#   base      ROCm base image + system deps + uv                (sections 0-2)
#   vllm      torch + source-built vLLM + llm-emu plugin        (sections 3, 3b)
#   emulate   emulation-only runtime image  (--target emulate)  (section 3c)
#   build     + LMCache, NIXL, hsa-snoop, runtime env           (sections 4-7)
#   wheels    scratch image holding just /wheels                (section 8)
#   runtime   default target, aliases `build`                   (section 9)
#
# `emulate` is the CPU-only image: it stops after vLLM + the llm-emu plugin, so
# nothing GPU-specific (LMCache's HIP extension, NIXL AIS_MT, hsa-snoop) is
# compiled.  Combined with --build-arg VLLM_TARGET_DEVICE=empty it compiles no
# GPU kernels at all -- see section 3.

ARG ROCM_VERSION=7.14.0
# ROCm 7.14 publishes the hipFile-bearing variant as `-full` (not `-complete`).
ARG ROCM_BASE_IMAGE=rocm/dev-ubuntu-24.04:${ROCM_VERSION}-full
# ROCM_BASE_IMAGE has a tagged default but remains overridable by callers.
# hadolint ignore=DL3006
FROM ${ROCM_BASE_IMAGE} AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ROCM_VERSION
ARG ROCM_BASE_IMAGE

# ----- 0. Corporate CA (optional, Zscaler/AMD) --------------------------------
# Pass a CA cert at build time via BuildKit secret to reach HTTPS through a
# corporate proxy (e.g. AMD Zscaler).  Omit entirely for builds outside the proxy.
#
#   docker build --secret id=tls_cert,src=/path/to/corp-ca.crt ...
#
# The secret is mounted read-only into this single RUN layer and is never
# written into the image filesystem or any build cache entry.
RUN --mount=type=secret,id=tls_cert,required=false \
    if [ -f /run/secrets/tls_cert ] && [ -s /run/secrets/tls_cert ]; then \
        cp /run/secrets/tls_cert /usr/local/share/ca-certificates/corp-ca.crt && \
        update-ca-certificates; \
    fi

# ----- 1. System dependencies ------------------------------------------------
# apt's package lists + downloaded debs live in BuildKit cache mounts, so they
# are NOT baked into the image layer (nothing to rm afterward) yet are reused by
# later rebuilds.  The base image ships /etc/apt/apt.conf.d/docker-clean, which
# purges the archive cache after each install -- drop it here so the cache mount
# actually persists.  `sharing=locked` serializes concurrent builds on the mount.
#
# libboost-dev (headers only) rather than libboost-all-dev: none of our C++
# components (NIXL, hsa-snoop) declare a boost dependency, and the full
# metapackage drags in libboost-mpi-dev -> openmpi-bin/-dev + gfortran + automake,
# which emit a raft of update-alternatives man-page warnings and bloat the image.
# Headers stay available for any transitive #include.
#
# libopenmpi3t64 (openmpi RUNTIME libs) is required on its own: ROCm torch wheels
# have a hard NEEDED on libmpi_cxx.so.40, loaded via torch._load_global_deps() on
# `import torch`.  We add just the runtime lib (not openmpi-bin/-dev), so torch
# imports cleanly without registering the mpirun/mpicc alternatives that caused
# warnings before.
#
# ccache is used to accelerate repeated vLLM HIP/C++ compilations across rebuilds.
# libsqlite3-dev, libfmt-dev, libmsgpack-dev are required by vLLM's C++ extensions.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libboost-dev \
        libdrm-dev \
        libfmt-dev \
        libibverbs-dev \
        ibverbs-providers \
        libmount-dev \
        libmsgpack-dev \
        libopenmpi3t64 \
        libsqlite3-dev \
        ninja-build \
        nvme-cli \
        pkg-config \
        python3-dev \
        bpftrace \
    && update-ca-certificates

# Silence git's detached-HEAD advisory.  Every source below is cloned then
# checked out at a pinned SHA (detached HEAD by design), which otherwise prints a
# multi-line advice block per clone.  Written to /root/.gitconfig, so it applies
# to every later git in this stage (LMCache, the NIXL clone script, hsa-snoop).
RUN git config --global advice.detachedHead false

# Ubuntu 24.04 marks the system interpreter PEP-668 "externally managed", which
# makes both `uv pip install --system` and plain `python3 -m pip install` refuse
# to touch /usr.  This is a single-purpose container built entirely on the system
# Python, so drop the marker to allow system-wide installs.
RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

# Every pip/uv install below runs as root by design (single-purpose container on
# system Python).  PIP_ROOT_USER_ACTION silences the "Running pip as the 'root'
# user" advisory at build time (also persists to the final image).  UV_LINK_MODE
# is set to copy because uv's download cache lives on a BuildKit cache mount (a
# separate filesystem from /usr), so uv cannot hardlink installs across it and
# would otherwise warn "Failed to hardlink files; falling back to full copy".
ENV PIP_ROOT_USER_ACTION=ignore \
    UV_LINK_MODE=copy

# ----- 2. uv (fast Python package manager) + pip bootstrap ------------------
# Pin UV_VERSION so the install URL is deterministic and the layer caches
# across builds.  Bump when upgrading uv.
ARG UV_VERSION=0.7.20
RUN curl -fsSL "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv && \
    ln -s /root/.local/bin/uvx /usr/local/bin/uvx

# =============================================================================
# vLLM stage — torch + source-built vLLM + the llm-emu emulation plugin.
# Everything an emulation-only image needs; the full stack continues from here.
# ARGs do not cross stage boundaries, so the ones used below are re-declared.
# =============================================================================
FROM base AS vllm

ARG ROCM_ARCH

# vLLM: built from source at this tag.  Check https://github.com/vllm-project/vllm/tags
# for the latest release; bump VLLM_REF and rebuild when updating.
ARG VLLM_GIT_URL=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=v0.27.1

# vLLM build target.  `rocm` (default) compiles the HIP kernels for ROCM_ARCH --
# the dominant cost of this whole Dockerfile.  `empty` is upstream vLLM's
# no-extension build (setup.py: `if _no_device(): ext_modules = []`): the Python
# stack is installed with zero kernels compiled.  That is only usable for
# emulation (`--target emulate`), where vllm_emulator intercepts execute_model()
# before any kernel would run -- do NOT ship an `empty` image to a GPU node.
ARG VLLM_TARGET_DEVICE=rocm

# ----- 3. vLLM (built from source at pinned tag) ---------------------------------
# Install torch 2.13.0+rocm7.2 from the upstream pytorch.org wheel index.
#
# NOTE: torch 2.12.0+rocm7.14.0 from repo.amd.com ships Triton 3.7.1 which
# triggers hipErrorInvalidImage on SPUR gfx950 nodes (amdgpu/KFD 6.14.14 host
# driver).  The pytorch.org rocm7.2 wheel ships a Triton version compatible with
# vLLM 0.27.1 and the current SPUR KFD.  Revert to repo.amd.com once the SPUR
# nodes are upgraded to a KFD that supports ROCm 7.14 compiled kernels.
#
# Version alignment (download.pytorch.org/whl/rocm7.2):
#   torch 2.13.0+rocm7.2  ->  torchvision 0.28.0+rocm7.2
# Wheels are pre-downloaded to /opt/pip-cache/wheels on the build host and
# passed in via --build-context pip-wheels=<dir> to avoid the 6 GB download.
# The context is mounted read-only and never baked into the image layer.
# If the build-context is not provided, falls back to the index URL.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=from=pip-wheels,source=.,target=/opt/pip-cache/wheels,ro \
    if [ -f /opt/pip-cache/wheels/torch-2.13.0+rocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl ]; then \
        pip3 install \
            --find-links /opt/pip-cache/wheels \
            --prefer-binary \
            --index-url https://download.pytorch.org/whl/rocm7.2 \
            "torch==2.13.0+rocm7.2" \
            "torchvision==0.28.0+rocm7.2"; \
    else \
        pip3 install \
            --timeout 300 \
            --retries 5 \
            "torch==2.13.0+rocm7.2" \
            "torchvision==0.28.0+rocm7.2" \
            --index-url https://download.pytorch.org/whl/rocm7.2; \
    fi

# Validate ROCM_ARCH here — after the torch download so a per-arch build
# doesn't bust the 6 GB wheel cache.  First use of ROCM_ARCH is in the vLLM
# build step below.
RUN test -n "$ROCM_ARCH" || { echo "ERROR: ROCM_ARCH is not set"; exit 1; }

RUN git clone --depth 1 --branch "${VLLM_REF}" "${VLLM_GIT_URL}" /app/vllm

COPY patches/ /app/patches/

# Patch setup.py so _is_hip() short-circuits on VLLM_TARGET_DEVICE=rocm at
# build time, without probing torch.version.hip before it is installed.
RUN sed -i 's/def _is_hip() -> bool:/def _is_hip() -> bool:\n    if VLLM_TARGET_DEVICE == "rocm":\n        return True/' /app/vllm/setup.py && \
    grep -A2 '^def _is_hip() -> bool:$' /app/vllm/setup.py | \
        grep -Fxq '    if VLLM_TARGET_DEVICE == "rocm":'

# Apply every patch under patches/vllm/ in lexical order.
RUN cd /app/vllm && \
    shopt -s nullglob && \
    patches=(/app/patches/vllm/*.patch) && \
    if [ "${#patches[@]}" -gt 0 ]; then \
        for patch in "${patches[@]}"; do \
            if git apply "$patch"; then \
                echo "Applied: $patch"; \
            else \
                echo "ERROR: vLLM patch does not apply cleanly: $patch" >&2; \
                echo "The vLLM pin (${VLLM_REF}) likely drifted; rebase this patch against it." >&2; \
                exit 1; \
            fi; \
        done; \
    fi

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    mkdir -p /wheels && \
    cd /app/vllm && \
    printf 'torch==2.13.0+rocm7.2\ntorchvision==0.28.0+rocm7.2\n' > /tmp/torch-constraints.txt && \
    pip3 install -r requirements/rocm.txt \
        --constraint /tmp/torch-constraints.txt && \
    VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE}" \
        PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        MAX_JOBS="$(nproc)" \
        CCACHE_DIR=/root/.cache/ccache \
        VLLM_VERSION_OVERRIDE="${VLLM_REF#v}+${VLLM_TARGET_DEVICE}" \
        pip3 wheel --no-build-isolation --no-deps --no-cache-dir -w /wheels . && \
    pip3 install --no-cache-dir /wheels/vllm-*.whl

# ----- 3b. vllm_emulator (llm-emu profile-driven emulation plugin) -----------
# Clone llm-emu at the pinned commit and install vllm_emulator as a
# vllm.platform_plugins entry point.  --no-deps skips the pinned vllm==0.18.1
# dependency from the upstream pyproject.toml; we already have the version above.
#
# The plugin is inert unless VLLM_EMULATOR_ENABLE_ORACLE=1 is set at runtime
# (its platform_plugins entry point returns None), so it is installed in BOTH
# the emulation image and the full GPU image -- a GPU serve is unaffected.
#
# The upstream repo's example profile pack is copied to a stable path so
# `--profile emulate` works with no external file; it is an NVIDIA A40 /
# Qwen3-8B capture, i.e. good enough to prove the plumbing but NOT an AMD
# latency model.  Capture a real one with VLLM_EMULATOR_TRACE_STEP_CYCLE=1 on
# an MI300X/MI355X (see patches/vllm/04-llm-emu-emulator-hooks.patch).
ARG LLM_EMU_GIT_URL=https://github.com/AKafakA/llm-emu.git
ARG LLM_EMU_REF=3e87da6
# The upstream example pack (NVIDIA A40) and the AIC default (AMD MI300X, from
# profiles/ -- see the COPY in the emulate/build stages below).  Both are in the
# image; the compose `emulate` profile defaults to the AMD one.
ENV AIC_EMULATOR_EXAMPLE_PACK=/opt/llm-emu/profiles/A40-Q8-Qwen3-8B.json \
    AIC_EMULATOR_DEFAULT_PACK=/opt/llm-emu/profiles/MI300X-Qwen3-8B.json
RUN git clone "${LLM_EMU_GIT_URL}" /app/llm-emu && \
    git -C /app/llm-emu checkout "${LLM_EMU_REF}" && \
    shopt -s nullglob && \
    for patch in /app/patches/llm-emu/*.patch; do \
        if git -C /app/llm-emu apply "$patch"; then \
            echo "Applied: $patch"; \
        else \
            echo "ERROR: llm-emu patch does not apply cleanly: $patch" >&2; \
            echo "The llm-emu pin (${LLM_EMU_REF}) likely drifted; rebase this patch against it." >&2; \
            exit 1; \
        fi; \
    done && \
    pip3 install --no-cache-dir --no-build-isolation --no-deps /app/llm-emu/ && \
    python3 -c "import vllm_emulator.hooks.executor_hook, vllm_emulator.oracle, \
vllm_emulator.profile.loader, vllm_emulator.profiler.trace_profiler; \
print('vllm_emulator subpackages import OK')" && \
    mkdir -p /opt/llm-emu/profiles && \
    cp /app/llm-emu/example_profiling_data/*.json /opt/llm-emu/profiles/ && \
    test -r "${AIC_EMULATOR_EXAMPLE_PACK}"

# =============================================================================
# ----- 3c. emulate — emulation-only runtime image (--target emulate) ---------
# CPU-only: no LMCache HIP extension, no NIXL, no hsa-snoop, and (with
# --build-arg VLLM_TARGET_DEVICE=empty) no compiled GPU kernels at all.  Runs
# the stock vLLM OpenAI server; vllm_emulator replaces the forward pass with a
# profile-pack latency draw.  Brought up by the `emulate` compose profile.
# =============================================================================
FROM vllm AS emulate
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PYTHONPATH=/opt/rocm/share/amd_smi
# AMD profile packs captured with `make profile-capture` (see profiles/README.md).
# This COPY is deliberately repeated at the end of the `build` stage rather than
# shared in `vllm`: a new pack is a frequent, cheap change, and putting it in the
# shared stage would invalidate the LMCache/NIXL/hsa-snoop layers every time.
COPY profiles/*.json /opt/llm-emu/profiles/
RUN test -r "${AIC_EMULATOR_DEFAULT_PACK}" || \
    { echo "ERROR: default profile pack missing: ${AIC_EMULATOR_DEFAULT_PACK}" >&2; exit 1; }
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

# =============================================================================
# Full stack — LMCache, NIXL, hsa-snoop on top of the vLLM stage.
# =============================================================================
FROM vllm AS build

ARG ROCM_ARCH

# Parallel compile jobs for the C++/HIP source builds below (LMCache, hsa-snoop).
# Defaults to $(nproc).
ARG BUILD_JOBS=

# Pinned component revisions — bump together when validating a new combo.
# ai-dynamo/nixl release tag + nixl-rocm-ais-mt.patch (ROCm/HIP + AIS_MT).
# The patch is AIS_MT-only (it no longer relocates upstream GDS_MT), so it applies
# cleanly on both v1.3.0 and v1.3.1 -- bump NIXL_REF and re-run the patch --check.
ARG NIXL_GIT_URL=https://github.com/ai-dynamo/nixl.git
ARG NIXL_REF=v1.3.2

# hsa-snoop: HSA AQL queue/dispatch snooper + Prometheus exporter (includes AIS metrics).
ARG HSA_SNOOP_GIT_URL=https://github.com/sbates130272/hsa-snoop.git
ARG HSA_SNOOP_REF=v1.0.0

# LMCache: upstream repo, pinned to the v0.5.4 release tag.
# All 14 AMD patches in patches/lmcache/ apply cleanly against this tag.
# Note: LMCache v0.5.2+ is required for vLLM >= 0.26.0.
ARG LMCACHE_GIT_URL=https://github.com/LMCache/LMCache.git
ARG LMCACHE_REF=v0.5.4

# Optional: enable UCX fast-path for NIXL (default off).
ARG AIC_UCX_FAST=

# ----- 4. LMCache (upstream release + AMD patches) ---------------------------
# Shallow-clone the pinned release tag directly (no history/other branches).
RUN git clone --depth 1 --branch "${LMCACHE_REF}" "${LMCACHE_GIT_URL}" /app/LMCache

# Apply every patch under patches/lmcache/ in lexical (filename) order -- the
# NN- prefixes fix the sequence.  Drop a new NN-*.patch in to include it.
RUN cd /app/LMCache && \
    shopt -s nullglob && \
    patches=(/app/patches/lmcache/*.patch) && \
    if [ "${#patches[@]}" -eq 0 ]; then \
        echo "ERROR: no patches found in /app/patches/lmcache/" >&2; exit 1; \
    fi && \
    for patch in "${patches[@]}"; do \
        if git apply "$patch"; then \
            echo "Applied: $patch"; \
        else \
            echo "ERROR: LMCache patch does not apply cleanly: $patch" >&2; \
            echo "The LMCache pin (${LMCACHE_REF}) likely drifted; rebase this patch against it." >&2; \
            exit 1; \
        fi; \
    done

# Build the LMCache wheel (with the AMD patches + HIP extension) and install THAT,
# instead of `pip install -e .`, so the wheel we publish is byte-for-byte what the
# image runs.  The wheel lands in /wheels (extracted by the `wheels` stage below).
#
# ROCM_TARGET_LST pins the arch that hipcc auto-detects for the host-side .cpp
# files (which torch compiles without an explicit --offload-arch): on a GPU-less
# build host, rocm_agent_enumerator can report a phantom arch (e.g. gfx1105) that
# this ROCm's clang rejects, breaking the compile.  GPU kernels still get the full
# multi-arch set from PYTORCH_ROCM_ARCH via torch, so a single valid arch here
# (the first of the list) is enough to satisfy the host compiles.
# MAX_JOBS / CMAKE_BUILD_PARALLEL_LEVEL parallelize the HIP extension compile
# (torch's build_ext is single-threaded otherwise -- the dominant build step).
# The pip build cache is a BuildKit cache mount, so repeated builds reuse it.
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p /wheels && \
    printf '%s\n' "${ROCM_ARCH%%;*}" > /etc/rocm_target.lst && \
    cd /app/LMCache && \
    PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        CXX=hipcc \
        BUILD_WITH_HIP=1 \
        ROCM_TARGET_LST=/etc/rocm_target.lst \
        MAX_JOBS="${BUILD_JOBS:-$(nproc)}" \
        CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS:-$(nproc)}" \
        python3 -m pip wheel --no-build-isolation --no-deps -w /wheels . && \
    uv pip install --system --no-cache /wheels/lmcache-*.whl

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "matplotlib>=3.8,<4"

# Remove any NIXL wheels that upstream packages may have pulled — we build from source below.
# Also purge stray CUDA-only deps (cupy-cuda12x, cufile-python, cuda-pathfinder): if they
# remain, vLLM/LMCache can try to import non-existent CUDA .so files at runtime on ROCm.
RUN uv pip uninstall --system \
        nixl nixl-cu12 nixl-cu13 \
        cupy-cuda12x cufile-python cuda-pathfinder 2>/dev/null || true

# ----- 5. hipFile — provided by the ROCm 7.14 base image (GA since 7.14) ------
# No source build needed.  hipFile headers, libraries, Python bindings, ais-stats,
# and ais-check are all pre-installed in rocm/dev-ubuntu-24.04:${ROCM_VERSION}-full.

# ----- 6. NIXL (ai-dynamo/nixl upstream + nixl-rocm-ais-mt.patch) ------------
# Clone upstream ai-dynamo/nixl, then apply nixl-rocm-ais-mt.patch to add ROCm/HIP
# detection and the AIS_MT plugin tree.
COPY docker/scripts /tmp/nixl-scripts
COPY patches/nixl/ /tmp/nixl-patches/
# Apply every patch under patches/nixl/ (lexical order) to the fresh checkout.
RUN --mount=type=cache,target=/root/.cache/uv \
    chmod +x /tmp/nixl-scripts/*.sh && \
    NIXL_GIT_URL="${NIXL_GIT_URL}" NIXL_REF="${NIXL_REF}" NIXL_REQUIRE_ROCM=1 \
        NIXL_DEST=/tmp/nixl /tmp/nixl-scripts/clone-nixl.sh && \
    shopt -s nullglob && \
    nixl_patches=(/tmp/nixl-patches/*.patch) && \
    if [ "${#nixl_patches[@]}" -eq 0 ]; then \
        echo "ERROR: no patches found in /tmp/nixl-patches/" >&2; exit 1; \
    fi && \
    for patch in "${nixl_patches[@]}"; do \
        if git -C /tmp/nixl apply --verbose "$patch"; then \
            echo "[nixl] applied $(basename "$patch")"; \
        else \
            echo "ERROR: nixl patch does not apply cleanly: $patch" >&2; exit 1; \
        fi; \
    done && \
    NIXL_SRC=/tmp/nixl \
        AIS_PATH=/opt/rocm ROCM_PATH=/opt/rocm \
        NIXL_INSTALL_PREFIX=/opt/nixl \
        NIXL_BUILD_WHEEL=1 NIXL_WHEEL_DIR=/wheels \
        BUILD_JOBS="${BUILD_JOBS}" \
        AIC_UCX_FAST="${AIC_UCX_FAST}" \
        UV_SYSTEM_CERTS=1 \
        /tmp/nixl-scripts/build-nixl.sh

# ----- 6b. hsa-snoop (HSA AQL queue snooper + Prometheus exporter) -----------
# GPU dispatch/queue telemetry on :9488 (/metrics), including AIS (AMD Infinity
# Storage) P2P metrics (ais_rx_ops_total, ais_tx_bytes_total, etc.) added in
# v1.0.0.  The snooper binary is host-only C++ (x86-64) and ARCH-INDEPENDENT --
# it parses HSA AQL queues and AIS ioctls from userspace, so one binary runs on
# every gfx target we ship.  We disable the optional examples/
# (CMAKE_DISABLE_FIND_PACKAGE_hip=ON): they compile HIP device code
# (examples/vecwork.hip) for a single auto-detected arch -- and on this CPU-only
# build node detection falls back to gfx1100, which neither matches our
# multi-arch set nor is needed by the exporter.
# Built WITH the Prometheus exporter (-DHSA_SNOOP_PROMETHEUS=ON fetches
# prometheus-cpp via CMake); the build is rejected if the resulting binary lacks
# --prometheus.  At runtime it must run as root in the host PID namespace,
# privileged (tracefs + pagemap + process_vm_readv), to see the vLLM/LMCache GPU
# processes -- see monitoring/docker-compose.monitoring.yml.
RUN git clone --depth 1 --branch "${HSA_SNOOP_REF}" "${HSA_SNOOP_GIT_URL}" /app/hsa-snoop && \
    cmake -S /app/hsa-snoop -B /app/hsa-snoop/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DHSA_SNOOP_PROMETHEUS=ON \
        -DCMAKE_DISABLE_FIND_PACKAGE_hip=ON && \
    cmake --build /app/hsa-snoop/build --parallel "${BUILD_JOBS:-$(nproc)}" && \
    cmake --install /app/hsa-snoop/build --prefix /usr/local && \
    test -x /usr/local/bin/hsa-snoop && \
    { /usr/local/bin/hsa-snoop --help 2>&1 || true; } | grep -q -- '--prometheus' \
        || { echo "ERROR: hsa-snoop built without Prometheus mode (--prometheus absent)" >&2; exit 1; } && \
    echo "PASS: hsa-snoop v${HSA_SNOOP_REF} with Prometheus exporter (incl. AIS metrics) installed"

# ----- 7. Runtime environment ------------------------------------------------
# Vars from upstream vLLM Dockerfile.rocm added here:
#   HSA_ENABLE_IPC_MODE_LEGACY=1  — legacy HIP IPC for cross-container KV registration (LMCache MP)
#   SAFETENSORS_FAST_GPU=1        — GPU-accelerated weight loading, faster model startup
#   HIP_FORCE_DEV_KERNARG=1       — kernel arg handling fix on MI300+/gfx942/gfx950
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    NIXL_PLUGIN_DIR=/opt/nixl/lib/x86_64-linux-gnu/plugins \
    PYTHONPATH=/opt/rocm/share/amd_smi:/opt/nixl/lib/python3/dist-packages \
    LD_LIBRARY_PATH=/opt/nixl/lib/x86_64-linux-gnu:/opt/nixl/lib:/opt/rocnixl-ucx/lib:/opt/rocm/lib \
    HSA_ENABLE_IPC_MODE_LEGACY=1 \
    SAFETENSORS_FAST_GPU=1 \
    HIP_FORCE_DEV_KERNARG=1

# AMD profile packs (see the identical COPY in the `emulate` stage for why this
# is repeated rather than shared).  The GPU image can serve in emulation mode
# too -- useful for reproducing a scheduling issue without occupying a GPU.
COPY profiles/*.json /opt/llm-emu/profiles/
RUN test -r "${AIC_EMULATOR_DEFAULT_PACK}" || \
    { echo "ERROR: default profile pack missing: ${AIC_EMULATOR_DEFAULT_PACK}" >&2; exit 1; }

# Provide image metadata. AIC_VERSION is declared and used here — after all
# expensive build steps — so version bumps only invalidate this single cheap
# layer rather than busting the torch/vLLM/LMCache/NIXL compilation cache.
ARG AIC_VERSION
ARG ROCM_BASE_IMAGE
RUN test -n "$AIC_VERSION" || { echo "ERROR: AIC_VERSION is not set (read it from VERSION)"; exit 1; }
LABEL org.opencontainers.image.title="rocm-aic" \
      org.opencontainers.image.description="AMD Infinity Context: ROCm vLLM + LMCache tiered-cache inference stack" \
      org.opencontainers.image.version="${AIC_VERSION}" \
      org.opencontainers.image.base.name="${ROCM_BASE_IMAGE}" \
      ai.amd.aic.version="${AIC_VERSION}" \
      ai.amd.aic.rocm="${ROCM_VERSION}" \
      ai.amd.aic.vllm="${VLLM_REF}" \
      ai.amd.aic.llm-emu="${LLM_EMU_REF}" \
      ai.amd.aic.lmcache="${LMCACHE_REF}" \
      ai.amd.aic.nixl="${NIXL_REF}" \
      ai.amd.aic.hipfile="rocm-${ROCM_VERSION}" \
      ai.amd.aic.hsa-snoop="${HSA_SNOOP_REF}"

# vLLM OpenAI-compatible server is the default entrypoint.
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

# ----- 8. Wheel export stage -------------------------------------------------
# A scratch image whose only content is the pip wheels built above (source-built
# vLLM + patched LMCache + ROCm NIXL).  Extract them WITHOUT building/loading
# the full image:
#   docker build --build-arg AIC_VERSION="$(<VERSION)" \
#     --target wheels --output type=local,dest=./wheels .
# The nightly-wheels GitHub workflow uses exactly this to publish the wheels.
FROM scratch AS wheels
COPY --from=build /wheels /

# ----- 9. Runtime image (default target) -------------------------------------
# Aliases the `build` stage so the DEFAULT build target (no --target) is the full
# runtime image, exactly as before -- Makefile/compose/run-build-distribute.sh
# and the existing Docker CI all keep getting the runtime image unchanged.  Must
# be the LAST stage in this file so BuildKit treats it as the default.
FROM build AS runtime
