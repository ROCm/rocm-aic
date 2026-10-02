# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# AIC base image: Ubuntu 24.04 + ROCm 7.14.1 + PyTorch (source) + torchvision (source).
#
# This is the shared foundation consumed by docker/vllm/Dockerfile and
# docker/lmcache/Dockerfile via --build-context base=docker-image://aic-base:TAG.
# Both images bind-mount the wheel artifacts from the build_pytorch and
# build_torchvision stages at build time (--mount=type=bind,from=build_pytorch,...).
#
# Stages:
#   pip-wheels        scratch stub for --build-context pip-wheels override
#   base              ROCm base + system deps + uv                    (sections 0-2)
#   build_pytorch     PyTorch 2.13 built from source                  (section 2b)
#   build_torchvision torchvision built from source                   (section 2c)
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
# ROCm 7.14 publishes the hipFile-bearing variant as `-full` (not `-complete`).
ARG ROCM_BASE_IMAGE=rocm/dev-ubuntu-24.04:${ROCM_VERSION}-full

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
# so a branch or tag would work here too.  Head of release/2.13 as of
# 2026-10-02 — the built wheel self-reports 2.13.0+git8e0af7f.
ARG PYTORCH_REF=8e0af7fa4388411493540afa152777537bbbfc1c
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
        python3 setup.py bdist_wheel --dist-dir=/app/pytorch/dist && \
    rm -rf /app/pytorch/build /app/pytorch/.git && \
    find /app/pytorch -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null; \
    find /app/pytorch -name '*.o' -maxdepth 8 -delete 2>/dev/null; true

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
# runtime — default target: pytorch + torchvision installed system-wide.
# Downstream Dockerfiles (docker/vllm, docker/lmcache) use this stage via
# --build-context base=docker/base so they get a fully working torch env
# without bind-mounting intermediate wheel directories.
# =============================================================================
FROM build_torchvision AS runtime

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir /app/vision/dist/*.whl && \
    rm -rf /app/pytorch /app/vision && \
    find /usr/lib/python3 /usr/local/lib/python3 -name '__pycache__' -type d \
         -exec rm -rf {} + 2>/dev/null; true
