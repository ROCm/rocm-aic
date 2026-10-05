# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# AIC base image: Ubuntu 24.04 + ROCm 7.14.1 + PyTorch (source) + torchvision (source)
# + NIXL (KV-transfer transport).
#
# This is the shared foundation consumed by docker/vllm/Dockerfile and
# docker/lmcache/Dockerfile via --build-context base=docker-image://aic-base:TAG.
# Both images bind-mount the wheel artifacts from the build_pytorch and
# build_torchvision stages at build time (--mount=type=bind,from=build_pytorch,...).
#
# Stages:
#   pip-wheels        scratch stub for --build-context pip-wheels override
#   base              ROCm-full base + system deps + uv               (sections 0-2)
#   build_pytorch     PyTorch 2.13 built from source                  (section 2b)
#   build_torchvision torchvision built from source                   (section 2c)
#   build_nixl        NIXL built from source (uses ROCm headers)      (section 2d)
#   runtime           Ubuntu 24.04 + 13 ROCm runtime pkgs + torch + NIXL (section 3)
#
# Build (from repo root):
#   DOCKER_BUILDKIT=1 docker build \
#     --build-arg ROCM_ARCH=gfx950 \
#     -f docker/base/Dockerfile \
#     -t aic-base:$(docker/scripts/aic-image-tag.sh) \
#     .
#
# Or via make:
#   make dist-build-base ROCM_ARCH=gfx950

ARG ROCM_VERSION=7.14.1
# Build base: full ROCm toolchain (compiler, headers, math libs) for source builds.
ARG ROCM_BASE_IMAGE=rocm/dev-ubuntu-24.04:${ROCM_VERSION}-full
# Runtime base: plain Ubuntu 24.04.  The 13 ROCm runtime packages needed by
# torch are installed explicitly below — ~2 GB installed vs ~18 GB for -full.
ARG ROCM_RUNTIME_IMAGE=ubuntu:24.04

# Empty default for the `pip-wheels` build context.  A caller that passes
# --build-context pip-wheels=<dir> overrides this stage exactly as before.
FROM scratch AS pip-wheels

# ROCM_BASE_IMAGE has a tagged default but remains overridable by callers.
# hadolint ignore=DL3006
FROM ${ROCM_BASE_IMAGE} AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ROCM_VERSION
ARG ROCM_BASE_IMAGE
ARG ROCM_ARCH

# ----- 0. Corporate CA (optional, Zscaler/AMD) --------------------------------
# Pass a CA cert at build time via BuildKit secret to reach HTTPS through a
# corporate proxy.  Omit entirely for builds outside the proxy.
#
#   docker build --secret id=tls_cert,src=/path/to/corp-ca.crt ...
#
RUN --mount=type=secret,id=tls_cert,required=false \
    if [ -f /run/secrets/tls_cert ] && [ -s /run/secrets/tls_cert ]; then \
        cp /run/secrets/tls_cert /usr/local/share/ca-certificates/corp-ca.crt && \
        update-ca-certificates; \
    fi

# ----- 1. System dependencies ------------------------------------------------
# hadolint ignore=SC2015
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
    && update-ca-certificates && \
    # libdrm-dev pulls in old libhsa-runtime64-1/libhsakmt1 (ROCm 5.7 era) which
    # break hipIpcGetMemHandle.  Remove them so the ROCm 7.14 runtime is the only one.
    DEBIAN_FRONTEND=noninteractive apt-get remove -y --purge \
        libhsa-runtime64-1 libhsakmt1 2>/dev/null || true

RUN git config --global advice.detachedHead false

RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

ENV PIP_ROOT_USER_ACTION=ignore \
    UV_LINK_MODE=copy \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# ----- 2. uv (fast Python package manager) -----------------------------------
ARG UV_VERSION=0.12.15
RUN curl -fsSL "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv && \
    ln -s /root/.local/bin/uvx /usr/local/bin/uvx

# =============================================================================
# PyTorch source build — avoids ROCm 7.2 wheel artifacts on a ROCm 7.14 host.
# =============================================================================
FROM base AS build_pytorch

ARG ROCM_ARCH
ARG PYTORCH_GIT_URL=https://github.com/ROCm/pytorch.git
# Immutable commit pin.  ROCm/pytorch publishes no tag for the 2.13 series, so
# a SHA is the only reproducible ref available; it is cloned then checked out,
# so a branch or tag would work here too.  Pinned to ec5f3b1 rather than the
# current head of release/2.13 (8e0af7fa) because a later commit updated
# third_party/composable_kernel in a way that breaks gfx1250 multi-arch builds
# (CK_BUFFER_RESOURCE_3RD_DWORD undeclared).  See patches/pytorch/01-ck-gfx1250-*.
ARG PYTORCH_REF=ec5f3b1effb14f0e36b66bab4db1c9b49b278f82
# Display only: the human-readable torch series carried in the image tag and
# the README, since a bare SHA says nothing about which torch this is.  Bump
# alongside PYTORCH_REF whenever the series changes.
ARG PYTORCH_SERIES=2.13

# torch 2.13 emits float->__bf16 libcalls absent from gcc-11; build with gcc-13.
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc-13 g++-13 pkg-config liblzma-dev libnuma-dev python3.12-venv && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-13

RUN git clone "${PYTORCH_GIT_URL}" /app/pytorch && \
    git -C /app/pytorch checkout "${PYTORCH_REF}"

RUN --mount=type=cache,target=/root/.cache/uv \
    cd /app/pytorch && \
    uv pip install --system -r requirements.txt && \
    git submodule update --init --recursive

COPY patches/ /app/patches/

# Apply patches to pytorch and its submodules before the build.
# Uses `patch -p1` (not `git apply`) so patches to submodule files work without
# git submodule boundary issues.
# hadolint ignore=SC2061
RUN if compgen -G "/app/patches/pytorch/*.patch" > /dev/null; then \
        for p in /app/patches/pytorch/*.patch; do \
            echo "Applying pytorch patch: $p"; \
            patch -p1 -d /app/pytorch < "$p" || { \
                echo "ERROR: patch does not apply cleanly: $p" >&2; exit 1; }; \
        done; \
    fi

ENV PYTORCH_ROCM_ARCH=${ROCM_ARCH}

# Cap compile parallelism.  Without this the torch_cpu/ATen compile fans out to
# every core; the heavier translation units peak well above 1 GB each, so on a
# host with less RAM than ~2 GB per core this exhausts memory and can hard-hang
# the machine.  Empty means "all cores", matching docker/lmcache/Dockerfile.
ARG BUILD_JOBS=

# hadolint ignore=SC2086
RUN --mount=type=cache,target=/root/.cache/ccache \
    cd /app/pytorch && \
    python3 tools/amd_build/build_amd.py && \
    CCACHE_DIR=/root/.cache/ccache \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    MAX_JOBS="${BUILD_JOBS:-$(nproc)}" \
    CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS:-$(nproc)}" \
    CMAKE_PREFIX_PATH=$(python3 -c 'import sys; print(sys.prefix)') \
    PYTORCH_ROCM_ARCH=$(echo "${PYTORCH_ROCM_ARCH}" | tr ',' ' ') \
        python3 setup.py bdist_wheel --dist-dir=/app/pytorch/dist && \
    find /app/pytorch -mindepth 1 -maxdepth 1 ! -name 'dist' -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# torchvision source build — must compile against our source-built PyTorch.
# Pin TORCHVISION_REF to the release matching PyTorch 2.13 (v0.28.x series).
# NOTE: v0.29.0 requires torch::stable::permute which is not present in
# ROCm/pytorch@release/2.13; stay on v0.28.0 until the ROCm fork advances.
# =============================================================================
FROM build_pytorch AS build_torchvision

ARG ROCM_ARCH
# Re-declared: ARG does not carry across a FROM.
ARG BUILD_JOBS=
ARG TORCHVISION_GIT_URL=https://github.com/pytorch/vision.git
# Immutable tag.  Shallow clone, so this must be a branch or tag, not a SHA.
ARG TORCHVISION_REF=v0.28.0

RUN git clone --depth 1 --branch "${TORCHVISION_REF}" "${TORCHVISION_GIT_URL}" /app/vision

# hadolint ignore=SC2086,DL3008
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libnuma-dev && \
    pip3 install setuptools wheel packaging /app/pytorch/dist/*.whl && \
    cd /app/vision && \
    USE_ROCM=1 USE_CUDA=0 \
    PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
    CCACHE_DIR=/root/.cache/ccache \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    MAX_JOBS="${BUILD_JOBS:-$(nproc)}" \
    CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS:-$(nproc)}" \
        python3 setup.py bdist_wheel --dist-dir=/app/vision/dist

# =============================================================================
# build_nixl — NIXL source build using the ROCm-full toolchain.
# Output (/opt/nixl, /opt/rocnixl-ucx) is copied to the slim runtime stage so
# the runtime image never needs cmake, clang, or ROCm headers.
# =============================================================================
FROM base AS build_nixl

ARG NIXL_GIT_URL=https://github.com/ai-dynamo/nixl.git
ARG NIXL_REF=v1.4.1
ARG BUILD_JOBS=
ARG AIC_UCX_FAST=

COPY docker/scripts /tmp/nixl-scripts
COPY patches/nixl/ /tmp/nixl-patches/
# hadolint ignore=SC3054,SC3044
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
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
        NIXL_BUILD_WHEEL=1 NIXL_WHEEL_DIR=/tmp/nixl-wheels \
        BUILD_JOBS="${BUILD_JOBS}" \
        AIC_UCX_FAST="${AIC_UCX_FAST}" \
        UV_SYSTEM_CERTS=1 \
        /tmp/nixl-scripts/build-nixl.sh && \
    rm -rf /tmp/nixl /tmp/nixl-patches /tmp/nixl-scripts /tmp/nixl-wheels

# =============================================================================
# runtime — Ubuntu 24.04 + 13 ROCm runtime packages + torch + NIXL.
#
# Starts FROM ubuntu:24.04 (not the -full ROCm image) and installs only the
# ROCm runtime libraries that torch actually dlopens at process start.
# Build toolchain, headers, and ~17 GB of unreferenced -full content are
# excluded.  The AMD apt signing key is copied from the build-time base stage
# (which already has it from the -full image) so no network key fetch is needed.
#
# ROCm runtime packages installed (~2 GB total):
#   amdrocm-runtime     HIP + HSA + comgr + hipRTC        (47 MB)
#   amdrocm-base        rocm-core + profiler-register      (5 MB)
#   amdrocm-profiler-base  profiler SDK + rocTX + AQL      (122 MB)
#   amdrocm-dnn-host    MIOpen                             (463 MB)
#   amdrocm-blas-host   rocBLAS + hipBLAS + hipBLASLt + rocRoller  (273 MB)
#   amdrocm-fft-host    hipFFT + rocFFT                    (13 MB)
#   amdrocm-rand-host   hipRAND + rocRAND                  (52 MB)
#   amdrocm-rccl-host   RCCL                               (4 MB)
#   amdrocm-solver-host hipSOLVER + rocSOLVER              (29 MB)
#   amdrocm-sparse-host hipSPARSE + rocSPARSE              (74 MB)
#   amdrocm-amdsmi      AMD SMI lib + Python bindings      (36 MB)
#   amdrocm-llvm        libLLVM + libclang (JIT needed)    (839 MB)
#   amdrocm-math-common rocm-openblas + host-math          (97 MB)
#
# Downstream Dockerfiles (docker/vllm, docker/lmcache) use this stage via
# --build-context base=oci-layout://... so they get a fully working torch env.
# =============================================================================
# hadolint ignore=DL3006
FROM ${ROCM_RUNTIME_IMAGE} AS runtime

ARG ROCM_VERSION
ARG ROCM_BASE_IMAGE
ARG ROCM_RUNTIME_IMAGE

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Corporate CA cert (needed for pip at runtime if behind a corporate proxy).
RUN --mount=type=secret,id=tls_cert,required=false \
    if [ -f /run/secrets/tls_cert ] && [ -s /run/secrets/tls_cert ]; then \
        cp /run/secrets/tls_cert /usr/local/share/ca-certificates/corp-ca.crt && \
        update-ca-certificates; \
    fi

# AMD ROCm apt repo — key copied from the build-time base stage (already
# fetched and verified there from the -full image).  Version-specific package
# names are computed from ROCM_VERSION (e.g. 7.14.1 → "7.14" suffix).
# hadolint ignore=DL3008
COPY --from=base /etc/apt/keyrings/amdrocm.gpg /etc/apt/keyrings/amdrocm.gpg
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] \
https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404 stable main" \
        > /etc/apt/sources.list.d/amdrocm.list && \
    apt-get update && \
    _mm=$(echo "${ROCM_VERSION}" | cut -d. -f1-2) && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        ibverbs-providers \
        libibverbs-dev \
        libopenmpi3t64 \
        nvme-cli \
        pkg-config \
        bpftrace \
        "amdrocm-runtime${_mm}" \
        "amdrocm-base${_mm}" \
        "amdrocm-profiler-base${_mm}" \
        "amdrocm-dnn-host${_mm}" \
        "amdrocm-blas-host${_mm}" \
        "amdrocm-fft-host${_mm}" \
        "amdrocm-rand-host${_mm}" \
        "amdrocm-rccl-host${_mm}" \
        "amdrocm-solver-host${_mm}" \
        "amdrocm-sparse-host${_mm}" \
        "amdrocm-amdsmi${_mm}" \
        "amdrocm-llvm${_mm}" \
        "amdrocm-math-common${_mm}" \
    && update-ca-certificates

RUN git config --global advice.detachedHead false
RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

# uv — used by downstream vllm/lmcache Dockerfiles for pip operations.
COPY --from=build_pytorch /root/.local/bin/uv  /usr/local/bin/uv
COPY --from=build_pytorch /root/.local/bin/uvx /usr/local/bin/uvx

# Installed Python packages: torch + all requirements (from build_pytorch),
# plus setuptools/wheel/packaging (from build_torchvision).
# Does NOT include the PyTorch or torchvision source trees.
COPY --from=build_torchvision /usr/local/lib/python3.12/dist-packages/ \
                               /usr/local/lib/python3.12/dist-packages/
COPY --from=build_torchvision /usr/local/bin/ /usr/local/bin/

ENV PIP_ROOT_USER_ACTION=ignore \
    UV_LINK_MODE=copy \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# torchvision wheel (built in build_torchvision but not yet installed there).
COPY --from=build_torchvision /app/vision/dist/ /tmp/vision/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir /tmp/vision/*.whl && \
    rm -rf /tmp/vision && \
    find /usr/local/lib/python3 -name '__pycache__' -type d \
         -exec rm -rf {} + 2>/dev/null; true

# NIXL native libs and Python bindings — compiled in build_nixl against the
# ROCm-full toolchain; only the output directories are needed at runtime.
COPY --from=build_nixl /opt/nixl/ /opt/nixl/
COPY --from=build_nixl /opt/rocnixl-ucx/ /opt/rocnixl-ucx/

# Guard: upstream pip deps may pull CUDA-only nixl variants that would shadow
# the source-built install from build_nixl.  Remove them if present.
RUN uv pip uninstall --system \
        nixl nixl-cu12 nixl-cu13 \
        cupy-cuda12x cufile-python cuda-pathfinder 2>/dev/null || true

ENV NIXL_PLUGIN_DIR=/opt/nixl/lib/x86_64-linux-gnu/plugins \
    PYTHONPATH=/opt/rocm/share/amd_smi:/opt/nixl/lib/python3/dist-packages \
    LD_LIBRARY_PATH=/opt/nixl/lib/x86_64-linux-gnu:/opt/nixl/lib:/opt/rocnixl-ucx/lib:/opt/rocm/lib \
    HSA_ENABLE_IPC_MODE_LEGACY=1
