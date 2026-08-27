# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# AIC: Ubuntu 24.04 + ROCm 7.14 inference stack.
#
#   Base image: rocm/dev-ubuntu-24.04:${ROCM_VERSION}-full  (Ubuntu 24.04, Python 3.12)
#   vLLM:       built from source at ${VLLM_REF} (torch 2.12.0+git6bbd260 from wheels.vllm.ai)
#   AITER:      ROCm/aiter ${AITER_REF}, installed from its matching ROCm wheel
#   FlashAttn:  Dao-AILab/flash-attention ${FLASH_ATTN_REF}, built for ROCm
#   LMCache:    LMCache/LMCache ${LMCACHE_REF} + AMD patches from patches/lmcache/
#   hipFile:    included in the ROCm 7.14 base image (GA since 7.14) — no source build needed
#   NIXL:       ai-dynamo/nixl ${NIXL_REF} + nixl-rocm-ais-mt.patch (ROCm/HIP + AIS_MT)
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
# Orchestration: make up
#   make build                                  # no proxy
#   make build TLS_CERT=/path/to/corp-ca.crt   # behind proxy
#
# Patches: every *.patch under patches/vllm/, patches/lmcache/, and patches/nixl/ is applied to
# its upstream checkout during the build, in lexical (filename) order -- hence
# the NN- prefixes.  Drop a new NN-*.patch into the directory to include it.
#
# Wheels: the build also produces pip-installable wheels for vLLM (source-built),
# AITER (upstream prebuilt), FlashAttention, the patched LMCache, and ROCm NIXL under /wheels.  Extract
# them with the `wheels` stage:
#   docker build --build-arg AIC_VERSION="$(<VERSION)" \
#     --target wheels --output type=local,dest=./wheels .
# (the default target remains the full runtime image -- see stages 9/10 below).

# The main build stage is named `build`; two trailing stages at the end of this
# file expose the pip wheels (`wheels`) and the final runtime image (`runtime`,
# the default target -- identical to `build`, so existing `docker build` callers
# are unaffected).  Publishing wheels: `docker build --target wheels
# --output type=local,dest=<dir>` extracts /wheels without loading the image.
ARG ROCM_VERSION=7.14.0
# ROCm 7.14 publishes the hipFile-bearing variant as `-full` (not `-complete`).
ARG ROCM_BASE_IMAGE=rocm/dev-ubuntu-24.04:${ROCM_VERSION}-full
# ROCM_BASE_IMAGE has a tagged default but remains overridable by callers.
# hadolint ignore=DL3006
FROM ${ROCM_BASE_IMAGE} AS build

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ROCM_VERSION
ARG ROCM_BASE_IMAGE
ARG ROCM_ARCH

# Parallel compile jobs for the C++/HIP source builds below (LMCache, hsa-snoop).
# Defaults to $(nproc).
ARG BUILD_JOBS=

# vLLM: built from source at this tag.  Check https://github.com/vllm-project/vllm/tags
# for the latest release; bump VLLM_REF and rebuild when updating.
ARG VLLM_GIT_URL=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=v0.28.0

# ROCm attention backends.  These revisions are the pair validated in vLLM's
# ROCm base image; keep them pinned and update together after revalidation.
ARG AITER_REF=v0.1.19
# The official AITER v0.1.19 release publishes CPython 3.12 wheels for ROCm
# 7.2 and gfx942/gfx950.  Keep this aligned with the ROCm torch runtime above.
ARG AITER_WHEEL_ROCM=7.2
ARG FLASH_ATTN_GIT_URL=https://github.com/Dao-AILab/flash-attention.git
ARG FLASH_ATTN_REF=0e60e394

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

# ----- 3. vLLM (built from source at pinned tag) ---------------------------------
# Install the vLLM ROCm 7.2.3 wheel set.  Keep TorchVision paired with the
# PyTorch build; its Triton dependency is resolved from the same wheel index.
#
# Version alignment (wheels.vllm.ai/rocm):
#   torch 2.12.0+git6bbd260  ->  torchvision 0.27.1+df56172
# Wheels are pre-downloaded to /opt/pip-cache/wheels on the build host and
# passed in via --build-context pip-wheels=<dir> to avoid the 1.35 GB download.
# The context is mounted read-only and never baked into the image layer.
# If the build-context is not provided, falls back to the index URL.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=from=pip-wheels,source=.,target=/opt/pip-cache/wheels,ro \
    if [ -f /opt/pip-cache/wheels/torch-2.12.0+git6bbd260-cp312-cp312-manylinux_2_39_x86_64.whl ]; then \
        pip3 install \
            --find-links /opt/pip-cache/wheels \
            --prefer-binary \
            --extra-index-url https://wheels.vllm.ai/rocm/ \
            "torch==2.12.0+git6bbd260" \
            "torchvision==0.27.1+df56172"; \
    else \
        pip3 install \
            --timeout 300 \
            --retries 5 \
            "torch==2.12.0+git6bbd260" \
            "torchvision==0.27.1+df56172" \
            --extra-index-url https://wheels.vllm.ai/rocm/; \
    fi

# Validate ROCM_ARCH here — after the torch download so a per-arch build
# doesn't bust the 1.35 GB wheel cache.  First use of ROCM_ARCH is in the vLLM
# build step below.
RUN test -n "$ROCM_ARCH" || { echo "ERROR: ROCM_ARCH is not set"; exit 1; }

RUN git clone --depth 1 --branch "${VLLM_REF}" "${VLLM_GIT_URL}" /app/vllm

# Copy only vLLM patches here — adding a LMCache or NIXL patch won't bust
# the vLLM build cache.  Each patch directory is copied immediately before
# its own build step for the same reason.
COPY patches/vllm/ /app/patches/vllm/

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
    printf 'torch==2.12.0+git6bbd260\ntorchvision==0.27.1+df56172\n' > /tmp/torch-constraints.txt && \
    pip3 install -r requirements/rocm.txt \
        --constraint /tmp/torch-constraints.txt && \
    VLLM_TARGET_DEVICE=rocm \
        PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        MAX_JOBS="$(nproc)" \
        CCACHE_DIR=/root/.cache/ccache \
        VLLM_VERSION_OVERRIDE="${VLLM_REF#v}+rocm" \
        pip3 wheel --no-build-isolation --no-deps --no-cache-dir -w /wheels . && \
    pip3 install --no-cache-dir --no-deps /wheels/vllm-*.whl && \
    python3 -c 'import torch; assert torch.version.hip, "Expected a ROCm torch build"'

# ----- 4. ROCm attention backends (AITer + FlashAttention) ------------------
# Install these for every image, rather than enabling an attention backend for a
# specific model.  vLLM can then select ROCM_AITER_FA or FLASH_ATTN whenever a
# loaded model and its runtime configuration support it.
#
# AITER's source wheel does not include its required `module_aiter_core` unless
# every JIT kernel is eagerly prebuilt.  That takes more than an hour and is
# incompatible with this image's MI355-stable Torch runtime.  Use AITER's
# matching upstream binary release instead.  The release artifact contains the
# gfx942 and gfx950 kernels and leaves only model-specific work for runtime.
# AITER imports rocminfo to detect the installed GPU, so its actual import is
# checked in the GPU-enabled smoke test after the image build rather than here.
#
# AITER_REF must name a release that publishes this CPython/ROCm wheel format.
# It is generated from the ref so a deliberate, validated override cannot
# silently install a different AITER revision.
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p /wheels && \
    aiter_version="${AITER_REF#v}" && \
    aiter_wheel="/wheels/amd_aiter-${aiter_version}+rocm${AITER_WHEEL_ROCM}.manylinux.2.28-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" && \
    curl --fail --location --retry 5 --retry-all-errors \
        "https://github.com/ROCm/aiter/releases/download/${AITER_REF}/amd_aiter-${aiter_version}%2Brocm${AITER_WHEEL_ROCM}.manylinux.2.28-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" \
        --output "${aiter_wheel}" && \
    pip3 install --no-cache-dir \
        "pandas==3.0.3" \
        "psutil==7.2.2" \
        "pyyaml==6.0.3" \
        "einops==0.8.2" \
        "pybind11==3.0.4" \
        "ninja==1.13.0" \
        "flydsl==0.2.4" && \
    pip3 install --no-cache-dir --no-deps "${aiter_wheel}" && \
    python3 -c 'import importlib.util, pathlib; spec = importlib.util.find_spec("aiter"); assert spec and spec.submodule_search_locations; package_dir = pathlib.Path(next(iter(spec.submodule_search_locations))); assert any(package_dir.glob("jit/module_aiter_core*.so")), "AITer core module is missing"; print("AITer installed:", package_dir)'

# FlashAttention has no ROCm wheel matching this image's pinned torch runtime,
# so build the vLLM-validated revision from source for the selected architecture.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    git clone --recurse-submodules "${FLASH_ATTN_GIT_URL}" /app/flash-attention && \
    cd /app/flash-attention && \
    git checkout "${FLASH_ATTN_REF}" && \
    git submodule update --init --recursive && \
    GPU_ARCHS="${ROCM_ARCH}" \
        PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
        MAX_JOBS="${BUILD_JOBS:-$(nproc)}" \
        CCACHE_DIR=/root/.cache/ccache \
        python3 setup.py bdist_wheel --dist-dir=/wheels && \
    pip3 install --no-cache-dir --no-deps /wheels/flash_attn-*.whl && \
    python3 -c 'import flash_attn; print("FlashAttention installed:", flash_attn.__file__)' && \
    { \
        echo "AITER_REF=${AITER_REF}"; \
        echo "FLASH_ATTN_REF=${FLASH_ATTN_REF}"; \
        python3 -c 'import importlib.util; spec = importlib.util.find_spec("aiter"); assert spec and spec.submodule_search_locations; print("AITER_PACKAGE=" + next(iter(spec.submodule_search_locations)))'; \
        python3 -c 'import flash_attn; print("FLASH_ATTN_PACKAGE=" + flash_attn.__file__)'; \
    } > /etc/rocm-aic-attention-backends.env

# ----- 5. LMCache (upstream release + AMD patches) ---------------------------
# Shallow-clone the pinned release tag directly (no history/other branches).
RUN git clone --depth 1 --branch "${LMCACHE_REF}" "${LMCACHE_GIT_URL}" /app/LMCache

# Copy LMCache patches just before use — isolated from vLLM and NIXL patches
# so adding/changing a LMCache patch only invalidates the LMCache build steps.
COPY patches/lmcache/ /app/patches/lmcache/

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

# ----- 6. hipFile — provided by the ROCm 7.14 base image (GA since 7.14) ------
# No source build needed.  hipFile headers, libraries, Python bindings, ais-stats,
# and ais-check are all pre-installed in rocm/dev-ubuntu-24.04:${ROCM_VERSION}-full.

# ----- 7. NIXL (ai-dynamo/nixl upstream + nixl-rocm-ais-mt.patch) ------------
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

# ----- 7b. hsa-snoop (HSA AQL queue snooper + Prometheus exporter) -----------
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

# ----- 8. Runtime environment ------------------------------------------------
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

# Provide image metadata. AIC_VERSION is declared and used here — after all
# expensive build steps — so version bumps only invalidate this single cheap
# layer rather than busting the torch/vLLM/LMCache/NIXL compilation cache.
ARG AIC_VERSION
RUN test -n "$AIC_VERSION" || { echo "ERROR: AIC_VERSION is not set (read it from VERSION)"; exit 1; }
LABEL org.opencontainers.image.title="rocm-aic" \
      org.opencontainers.image.description="AMD Infinity Context: ROCm vLLM + LMCache tiered-cache inference stack" \
      org.opencontainers.image.version="${AIC_VERSION}" \
      org.opencontainers.image.base.name="${ROCM_BASE_IMAGE}" \
      ai.amd.aic.version="${AIC_VERSION}" \
      ai.amd.aic.rocm="${ROCM_VERSION}" \
      ai.amd.aic.vllm="${VLLM_REF}" \
      ai.amd.aic.aiter="${AITER_REF}" \
      ai.amd.aic.flash-attention="${FLASH_ATTN_REF}" \
      ai.amd.aic.lmcache="${LMCACHE_REF}" \
      ai.amd.aic.nixl="${NIXL_REF}" \
      ai.amd.aic.hipfile="rocm-${ROCM_VERSION}" \
      ai.amd.aic.hsa-snoop="${HSA_SNOOP_REF}"

# vLLM OpenAI-compatible server is the default entrypoint.
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

# ----- 9. Wheel export stage -------------------------------------------------
# A scratch image whose only content is the pip wheels built above (source-built
# vLLM, AITER (prebuilt), FlashAttention, patched LMCache, and ROCm NIXL).  Extract them
# WITHOUT building/loading the full image:
#   docker build --build-arg AIC_VERSION="$(<VERSION)" \
#     --target wheels --output type=local,dest=./wheels .
# The nightly-wheels GitHub workflow uses exactly this to publish the wheels.
FROM scratch AS wheels
COPY --from=build /wheels /

# ----- 10. Runtime image (default target) ------------------------------------
# Aliases the `build` stage so the DEFAULT build target (no --target) is the full
# runtime image, exactly as before -- Makefile/compose/run-build-distribute.sh
# and the existing Docker CI all keep getting the runtime image unchanged.  Must
# be the LAST stage in this file so BuildKit treats it as the default.
FROM build AS runtime
