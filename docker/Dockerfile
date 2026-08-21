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
#   Mooncake:   kvcache-ai/Mooncake built from source at the pinned ${MOONCAKE_REF}
#               in its own stage; the staged SDK is what LMCache's optional
#               Mooncake L2 backend compiles and links against, and the ROCm
#               wheel from upstream's own build script is installed alongside.
#               Packaging only: no runtime service, option or default changes.
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
# Patches: every *.patch under patches/lmcache/ and patches/nixl/ is applied to
# its upstream checkout during the build, in lexical (filename) order -- hence
# the NN- prefixes.  Drop a new NN-*.patch into the directory to include it.
#
# Wheels: the build also produces pip-installable wheels for vLLM (source-built),
# the patched LMCache, source-built Mooncake (ROCm) and ROCm NIXL under /wheels.
# Extract them with the `wheels` stage:
#   docker build --build-arg AIC_VERSION="$(<VERSION)" \
#     --target wheels --output type=local,dest=./wheels .
# (the default target remains the full runtime image -- see stages 8/9 below).

# The main build stage is named `build`; two trailing stages at the end of this
# file expose the pip wheels (`wheels`) and the final runtime image (`runtime`,
# the default target -- identical to `build`, so existing `docker build` callers
# are unaffected).  Publishing wheels: `docker build --target wheels
# --output type=local,dest=<dir>` extracts /wheels without loading the image.
ARG ROCM_VERSION=7.14.0
# ROCm 7.14 publishes the hipFile-bearing variant as `-full` (not `-complete`).
ARG ROCM_BASE_IMAGE=rocm/dev-ubuntu-24.04:${ROCM_VERSION}-full

# Mooncake: source-built at an exact commit (the first revision carrying the
# upstream ROCm/HIP wheel + CI recipe).  Bump only with evidence: the build
# below follows that revision's own ROCm recipe.
ARG MOONCAKE_GIT_URL=https://github.com/kvcache-ai/Mooncake.git
ARG MOONCAKE_REF=e164c96b3c184b34f7979c47c176c2e51a926f66
ARG MOONCAKE_FETCH_REF=refs/pull/3184/head
ARG GO_VERSION=1.25.9
ARG GO_AMD64_SHA256=00859d7bd6defe8bf84d9db9e57b9a4467b2887c18cd93ae7460e713db774bc1
# Staged SDK (headers + shared libraries + tools) consumed by the LMCache
# Mooncake extension.  Deliberately NOT a default compiler/linker search path:
# see the NIXL isolation probe in stage 6.
ARG MOONCAKE_PREFIX=/opt/mooncake-sdk

# An optional named context with this name overrides the empty default stage.
FROM scratch AS pip-wheels

# ----- Stage M. Mooncake (source build at the pinned commit) -----------------
# Independent builder on the same ROCm base.  Only ${MOONCAKE_PREFIX} and the
# built wheel cross into the main stage, so this stage's system-wide side
# effects (Go toolchain, yalantinglibs in /usr/local, the CMake install of the
# mooncake python package into site-packages) never reach the runtime image.
#
# hadolint ignore=DL3006
FROM ${ROCM_BASE_IMAGE} AS mooncake-build

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG MOONCAKE_GIT_URL
ARG MOONCAKE_REF
ARG MOONCAKE_FETCH_REF
ARG MOONCAKE_PREFIX
ARG GO_VERSION
ARG GO_AMD64_SHA256
ARG BUILD_JOBS=
# The image's system interpreter.  Upstream's wheel script invokes
# `python${MOONCAKE_PYTHON_VERSION}` by name.
ARG MOONCAKE_PYTHON_VERSION=3.12

# Corporate CA, same contract as the main stage: mounted for this layer only.
RUN --mount=type=secret,id=tls_cert,required=false \
    if [ -f /run/secrets/tls_cert ] && [ -s /run/secrets/tls_cert ]; then \
        cp /run/secrets/tls_cert /usr/local/share/ca-certificates/corp-ca.crt && \
        update-ca-certificates; \
    fi

# git must exist before the clone, and pip must be importable before upstream's
# wheel script bootstraps its build frontend.  Everything else is installed by
# Mooncake's own dependencies.sh below.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 marks the system interpreter PEP-668 "externally managed"; this
# throwaway builder installs its build frontend into it.
RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED
ENV PIP_ROOT_USER_ACTION=ignore

RUN git config --global advice.detachedHead false

# Mooncake's dependency helper downloads Go from several mirrors without
# verifying the archive. Install its expected version from go.dev first and
# verify the official checksum; dependencies.sh then detects it and skips its
# own downloader.
RUN curl -fsSLo /tmp/go.tar.gz \
        "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" && \
    echo "${GO_AMD64_SHA256}  /tmp/go.tar.gz" | sha256sum -c - && \
    rm -rf /usr/local/go && \
    tar -C /usr/local -xzf /tmp/go.tar.gz && \
    rm -f /tmp/go.tar.gz && \
    test "$(/usr/local/go/bin/go version | awk '{print $3}')" = "go${GO_VERSION}"

ENV PATH=/usr/local/go/bin:/opt/rocm/bin:${PATH}

# Pin by full commit SHA and verify it: a moved tag or a rewritten branch must
# fail the build rather than silently ship a different Mooncake.
RUN git init /src/Mooncake && \
    git -C /src/Mooncake remote add origin "${MOONCAKE_GIT_URL}" && \
    git -C /src/Mooncake fetch --filter=blob:none --depth 1 \
        origin "${MOONCAKE_FETCH_REF}" && \
    git -C /src/Mooncake checkout --detach FETCH_HEAD && \
    if [ "$(git -C /src/Mooncake rev-parse HEAD)" != "${MOONCAKE_REF}" ]; then \
        echo "ERROR: Mooncake checkout is not ${MOONCAKE_REF}" >&2; \
        exit 1; \
    fi

# Upstream's own dependency installer: system packages, git submodules
# (pybind11, yalantinglibs) and the yalantinglibs install into /usr/local.
# Kept whole on purpose - it is the recipe the pinned revision builds against.
RUN cd /src/Mooncake && bash dependencies.sh -y

# Pinned build frontend for the wheel step.  Upstream's build_wheel.sh upgrades
# pip/build/setuptools/wheel/auditwheel and then builds in an isolated env;
# PIP_CONSTRAINT applies to both, so the frontend is pinned instead of floating.
# Transitive dependencies of these packages still resolve at build time.
COPY docker/constraints/mooncake-wheel-build.txt /etc/mooncake-wheel-build.txt
ENV PIP_CONSTRAINT=/etc/mooncake-wheel-build.txt

# Configure/build with the pinned revision's ROCm recipe (.github/workflows/
# ci_rocm.yml): USE_HIP with hipify-perl from /opt/rocm/bin on PATH, no CUDA,
# no EP, HTTP + etcd metadata, no unit tests.  Two deviations, both required
# here and both deliberate:
#   BUILD_SHARED_LIBS=ON  - upstream CI only produces the python wheel, whose
#       engine.so/store.so are self-contained.  A third-party consumer
#       (LMCache's Mooncake extension) has to link libmooncake_store.so, and
#       the install rules for the store/transfer-engine/common libraries are
#       guarded on this option.
#   CMAKE_INSTALL_PREFIX  - the SDK is staged outside the default prefixes.
# The Go etcd wrapper fetches modules from the network; upstream CI retries the
# build for exactly that reason, so mirror the retry rather than fail a long
# build on a transient fetch.
RUN --mount=type=cache,target=/root/.cache/go-build,sharing=locked \
    export PATH="/opt/rocm/bin:/usr/local/go/bin:${PATH}" && \
    python_bin="$(command -v "python${MOONCAKE_PYTHON_VERSION}")" && \
    cmake -G Ninja -S /src/Mooncake -B /src/Mooncake/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_HIP=ON \
        -DUSE_CUDA=OFF \
        -DWITH_EP=OFF \
        -DUSE_HTTP=ON \
        -DUSE_ETCD=ON \
        -DSTORE_USE_ETCD=ON \
        -DWITH_STORE_RUST=OFF \
        -DBUILD_UNIT_TESTS=OFF \
        -DBUILD_BENCHMARK=OFF \
        -DBUILD_EXAMPLES=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DENABLE_SCCACHE=OFF \
        -DCMAKE_INSTALL_PREFIX="${MOONCAKE_PREFIX}" \
        -DPython3_EXECUTABLE="${python_bin}" && \
    attempt=0 && \
    until cmake --build /src/Mooncake/build -j "${BUILD_JOBS:-$(nproc)}"; do \
        attempt=$((attempt + 1)); \
        if [ "${attempt}" -ge 3 ]; then \
            echo "ERROR: Mooncake build failed after ${attempt} attempts" >&2; exit 1; \
        fi; \
        echo "Mooncake build attempt ${attempt} failed, retrying"; sleep 15; \
    done && \
    cmake --install /src/Mooncake/build

# Stage the SDK.  `cmake --install` carries the shared libraries, the
# transfer-engine public headers and the master/client binaries, but NOT the
# store headers, the vendored cachelib tree, the generated headers or
# yalantinglibs - all of which a consumer of the store client compiles against.
# Copy them explicitly, then assert the result: a silently incomplete SDK would
# only surface as a confusing compile error one stage later.
RUN set -eu; \
    src=/src/Mooncake; \
    include="${MOONCAKE_PREFIX}/include"; \
    mkdir -p "${include}" "${MOONCAKE_PREFIX}/lib"; \
    cp -a /usr/local/include/. "${include}/"; \
    cp -a "${src}/mooncake-transfer-engine/include/." "${include}/"; \
    cp -a "${src}/mooncake-common/include/." "${include}/"; \
    cp -a "${src}/mooncake-store/include/." "${include}/"; \
    cp -a "${src}/build/mooncake-store/include/." "${include}/"; \
    for lib in \
        "${src}/build/mooncake-store/src/libmooncake_store.so" \
        "${src}/build/mooncake-common/src/libmooncake_common.so" \
        "${src}/build/mooncake-common/libasio.so" \
        "${src}/build/mooncake-common/etcd/libetcd_wrapper.so" \
        "${src}/build/mooncake-transfer-engine/src/libtransfer_engine.so"; do \
        if [ -f "${lib}" ]; then \
            install -Dm0755 "${lib}" "${MOONCAKE_PREFIX}/lib/$(basename "${lib}")"; \
        fi; \
    done; \
    for required in \
        lib/libmooncake_store.so \
        lib/libtransfer_engine.so \
        include/real_client.h \
        include/transfer_engine.h \
        include/cachelib_memory_allocator/Slab.h \
        include/ylt/struct_pack.hpp; do \
        [ -e "${MOONCAKE_PREFIX}/${required}" ] || \
            { echo "ERROR: staged SDK is missing ${required}" >&2; exit 1; }; \
    done; \
    echo "PASS: Mooncake SDK staged at ${MOONCAKE_PREFIX}"

# Official ROCm/HIP wheel, built by upstream's own script (HIP_BUILD=1 renames
# the distribution and auditwheel excludes the ROCm runtime, which is bound at
# load time).  `pip` is what the script probes for.
RUN set -eu; \
    command -v pip >/dev/null || ln -s "$(command -v pip3)" /usr/local/bin/pip; \
    python${MOONCAKE_PYTHON_VERSION} -m pip install \
        --ignore-installed \
        --constraint /etc/mooncake-wheel-build.txt \
        pip build setuptools wheel auditwheel; \
    cd /src/Mooncake; \
    export PATH="/opt/rocm/bin:/usr/local/go/bin:${PATH}"; \
    export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"; \
    HIP_BUILD=1 \
        PYTHON_VERSION="${MOONCAKE_PYTHON_VERSION}" \
        OUTPUT_DIR=dist-rocm \
        ./scripts/build_wheel.sh; \
    mkdir -p /wheels; \
    cp mooncake-wheel/dist-rocm/*.whl /wheels/; \
    ls -1 /wheels

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
ARG VLLM_REF=v0.26.0

# Pinned component revisions — bump together when validating a new combo.
# ai-dynamo/nixl release tag + nixl-rocm-ais-mt.patch (ROCm/HIP + AIS_MT).
# The patch is AIS_MT-only (it no longer relocates upstream GDS_MT), so it applies
# cleanly on both v1.3.0 and v1.3.1 -- bump NIXL_REF and re-run the patch --check.
ARG NIXL_GIT_URL=https://github.com/ai-dynamo/nixl.git
ARG NIXL_REF=v1.3.2

# hsa-snoop: HSA AQL queue/dispatch snooper + Prometheus exporter (includes AIS metrics).
ARG HSA_SNOOP_GIT_URL=https://github.com/sbates130272/hsa-snoop.git
ARG HSA_SNOOP_REF=v1.0.0

# LMCache: upstream repo, pinned to the v0.5.3 release tag.
# All 16 AMD patches in patches/lmcache/ apply cleanly against this tag.
# Note: LMCache v0.5.2+ is required for vLLM >= 0.26.0.
ARG LMCACHE_GIT_URL=https://github.com/LMCache/LMCache.git
ARG LMCACHE_REF=v0.5.3

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
# Install torch 2.13.0+rocm7.2 from the upstream pytorch.org wheel index.
#
# UNBLOCK (2026-08-03): torch 2.12.0+rocm7.14.0 from repo.amd.com ships
# Triton 3.7.1 which triggers hipErrorInvalidImage on SPUR gfx950 nodes
# (amdgpu/KFD 6.14.14 host driver).  The pytorch.org rocm7.2 wheel ships
# a Triton version compatible with vLLM 0.26.0 and the current SPUR KFD.
# Revert to repo.amd.com once the SPUR nodes are upgraded to a KFD that
# supports ROCm 7.14 compiled kernels.
#
# Version alignment (download.pytorch.org/whl/rocm7.2):
#   torch 2.13.0+rocm7.2  ->  torchvision 0.28.0+rocm7.2
# Wheels are pre-downloaded to /opt/pip-cache/wheels on the build host and
# passed in via --build-context pip-wheels=<dir> to avoid the 6 GB download.
# The context is mounted read-only and never baked into the image layer.
# If the build-context is not provided, falls back to the index URL.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=from=pip-wheels,source=.,target=/opt/pip-cache/wheels,ro \
    torch_wheel=/opt/pip-cache/wheels/torch-2.13.0+rocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl; \
    if [ -f "${torch_wheel}" ]; then \
        pip3 install \
            "${torch_wheel}" \
            "torchvision==0.28.0+rocm7.2" \
            --index-url https://download.pytorch.org/whl/rocm7.2; \
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

# Copy only vLLM patches here — adding a LMCache or NIXL patch won't bust
# the vLLM build cache.  Each patch directory is copied immediately before
# its own build step for the same reason.
COPY patches/vllm/ /app/patches/vllm/

# Patch setup.py so _is_hip() short-circuits on VLLM_TARGET_DEVICE=rocm at
# build time, without probing torch.version.hip before it is installed.
RUN sed -i 's/def _is_hip() -> bool:/def _is_hip() -> bool:\n    if VLLM_TARGET_DEVICE == "rocm":\n        return True/' /app/vllm/setup.py

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
    VLLM_TARGET_DEVICE=rocm \
        PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        MAX_JOBS="$(nproc)" \
        CCACHE_DIR=/root/.cache/ccache \
        VLLM_VERSION_OVERRIDE="${VLLM_REF#v}+rocm" \
        pip3 wheel --no-build-isolation --no-deps --no-cache-dir -w /wheels . && \
    pip3 install --no-cache-dir /wheels/vllm-*.whl

# ----- 3b. Mooncake SDK + ROCm wheel (from the mooncake-build stage) ---------
# Declared here, after the expensive torch/vLLM layers: re-pinning Mooncake
# must not invalidate them (same reasoning as AIC_VERSION at the end).
ARG MOONCAKE_GIT_URL
ARG MOONCAKE_REF
ARG MOONCAKE_PREFIX

# Runtime and header dependencies of the Mooncake store client, which the
# LMCache Mooncake extension compiles and links against in stage 4.  This is
# the subset of Mooncake's own dependency list that its store library links or
# that its public headers include (glog/gflags, jsoncpp, yaml-cpp, curl for the
# HTTP metadata client, zstd, xxhash, io_uring, numa); boost, fmt, msgpack and
# ibverbs are already installed above.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libcurl4-openssl-dev \
        libgoogle-glog-dev \
        libjsoncpp-dev \
        libnuma-dev \
        liburing-dev \
        libxxhash-dev \
        libyaml-cpp-dev \
        libzstd-dev \
        patchelf

COPY --from=mooncake-build ${MOONCAKE_PREFIX} ${MOONCAKE_PREFIX}
COPY --from=mooncake-build /wheels/ /wheels/

# Install the source-built ROCm wheel with its declared dependencies, keeping
# torch pinned so the resolver cannot pull a different build in.
RUN --mount=type=cache,target=/root/.cache/uv \
    printf 'torch==2.13.0+rocm7.2\ntorchvision==0.28.0+rocm7.2\n' > /tmp/mooncake-constraints.txt && \
    uv pip install --system --no-cache \
        --constraint /tmp/mooncake-constraints.txt \
        /wheels/mooncake_transfer_engine_rocm-*.whl

# ----- 4. LMCache (upstream release + AMD patches) ---------------------------
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

# Build ONE LMCache wheel carrying both native extensions (the HIP c_ops and
# the host-only Mooncake L2 backend) and install THAT, instead of
# `pip install -e .`, so the wheel we publish is byte-for-byte what the image
# runs.  The wheel lands in /wheels (extracted by the `wheels` stage below).
#
# Compilers: LMCache collects every extension into a single build_ext, so one
# global CXX used to have to serve both the HIP sources and the host-only ones.
# 15-lmcache-per-extension-cxx.patch makes the compiler a per-extension
# property: the base compiler is g++, and only lmcache.c_ops is switched to
# LMCACHE_ROCM_CXX (hipcc) for its own compile and link.  Do not reintroduce a
# global CXX=hipcc: it would hand hipcc the Mooncake and common C++ sources.
#
# ROCM_TARGET_LST pins the arch that hipcc auto-detects for the host-side .cpp
# files (which torch compiles without an explicit --offload-arch): on a GPU-less
# build host, rocm_agent_enumerator can report a phantom arch (e.g. gfx1105) that
# this ROCm's clang rejects, breaking the compile.  GPU kernels still get the full
# multi-arch set from PYTORCH_ROCM_ARCH via torch, so a single valid arch here
# (the first of the list) is enough to satisfy the host compiles.
# MAX_JOBS / CMAKE_BUILD_PARALLEL_LEVEL parallelize the HIP extension compile
# (torch's build_ext is single-threaded otherwise -- the dominant build step);
# they parallelize work INSIDE one extension, which the compiler switch allows.
#
# MOONCAKE_INCLUDE_DIR lists the same four include roots Mooncake's own CMake
# puts on the store library's compile line (the SDK root, plus the vendored
# cachelib allocator and its two header roots); MOONCAKE_LIB_DIR provides the
# build-time link directory. MOONCAKE_RUNTIME_LIB_DIR is deliberately separate:
# the published LMCache wheel resolves libmooncake_store from the companion
# Mooncake ROCm wheel via paths relative to site-packages, never from the
# image-only SDK path.
# hadolint ignore=SC2016
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p /wheels && \
    printf '%s\n' "${ROCM_ARCH%%;*}" > /etc/rocm_target.lst && \
    cd /app/LMCache && \
    PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        CXX=/usr/bin/g++ \
        LMCACHE_ROCM_CXX=/opt/rocm/bin/hipcc \
        BUILD_WITH_HIP=1 \
        BUILD_WITH_MOONCAKE=1 \
        MOONCAKE_INCLUDE_DIR="${MOONCAKE_PREFIX}/include;${MOONCAKE_PREFIX}/include/cachelib_memory_allocator;${MOONCAKE_PREFIX}/include/cachelib_memory_allocator/include;${MOONCAKE_PREFIX}/include/cachelib_memory_allocator/fake_include" \
        MOONCAKE_LIB_DIR="${MOONCAKE_PREFIX}/lib" \
        MOONCAKE_RUNTIME_LIB_DIR='$ORIGIN/../mooncake;$ORIGIN/../mooncake_transfer_engine_rocm.libs' \
        ROCM_TARGET_LST=/etc/rocm_target.lst \
        MAX_JOBS="${BUILD_JOBS:-$(nproc)}" \
        CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS:-$(nproc)}" \
        python3 -m pip wheel --no-build-isolation --no-deps -w /wheels .

# Fail-closed gates. Archive checks run BEFORE install, so missing extensions
# or tools cannot be papered over by stale site-packages; runtime gates then
# re-prove the exact installed artifacts.
COPY docker/scripts/mooncake-gates.sh /usr/local/bin/aic-mooncake-gates.sh
# Arrays are bash, and SHELL above is bash: the count assertions below make
# "exactly one wheel" a build failure instead of a silent multi-arg call.
# hadolint ignore=SC3054
RUN chmod +x /usr/local/bin/aic-mooncake-gates.sh && \
    shopt -s nullglob && \
    lmcache_wheels=(/wheels/lmcache-*.whl) && \
    mooncake_wheels=(/wheels/mooncake_transfer_engine_rocm-*.whl) && \
    [ "${#lmcache_wheels[@]}" -eq 1 ] && \
    [ "${#mooncake_wheels[@]}" -eq 1 ] && \
    /usr/local/bin/aic-mooncake-gates.sh wheels \
        "${lmcache_wheels[0]}" "${mooncake_wheels[0]}"

# Resolve LMCache's declared dependencies while keeping the existing ROCm
# torch pairing, then reinstall only the canonical local artifact. Passing
# --reinstall to the resolving command would replace ROCm torch with PyPI's
# CUDA stack because LMCache intentionally leaves its torch requirement open.
RUN --mount=type=cache,target=/root/.cache/uv \
    printf 'torch==2.13.0+rocm7.2\ntorchvision==0.28.0+rocm7.2\n' > /tmp/lmcache-constraints.txt && \
    uv pip install --system --no-cache \
        --constraint /tmp/lmcache-constraints.txt \
        /wheels/lmcache-*.whl && \
    uv pip install --system --no-cache --reinstall --no-deps \
        /wheels/lmcache-*.whl

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
# NIXL has a separate preview Mooncake plugin. This image deliberately owns
# Mooncake through LMCache, so build-nixl.sh disables that plugin explicitly;
# the post-build check below verifies it stayed absent.

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
        /tmp/nixl-scripts/build-nixl.sh && \
    mooncake_plugin="$(find /opt/nixl /tmp/nixl/build -type f \
        -iname '*mooncake*.so*' 2>/dev/null || true)" && \
    if [ -n "${mooncake_plugin}" ]; then \
        echo "ERROR: NIXL built or installed a Mooncake plugin: ${mooncake_plugin}" >&2; \
        exit 1; \
    fi && \
    echo "PASS: NIXL built no Mooncake plugin; its plugin set is unchanged"

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
#
# No Mooncake entry belongs on this path.  The LMCache extension reaches
# libmooncake_store and its dependencies through the wheel-relative RUNPATH,
# proven by the clean-install gate with only Torch and ROCm exposed.  Because
# LD_LIBRARY_PATH precedes that RUNPATH, naming either the build-only SDK or the
# wheel's own private directories here would let an LMCache installed in a venv
# or an alternate target bind to this image copy instead of its own companion
# wheel, mixing two glog builds.
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    NIXL_PLUGIN_DIR=/opt/nixl/lib/x86_64-linux-gnu/plugins \
    PYTHONPATH=/opt/rocm/share/amd_smi:/opt/nixl/lib/python3/dist-packages \
    LD_LIBRARY_PATH=/opt/nixl/lib/x86_64-linux-gnu:/opt/nixl/lib:/opt/rocnixl-ucx/lib:/opt/rocm/lib \
    HSA_ENABLE_IPC_MODE_LEGACY=1 \
    SAFETENSORS_FAST_GPU=1 \
    HIP_FORCE_DEV_KERNARG=1

# Prove the installed artifacts in the final package and loader environment,
# after the intentional CUDA-package cleanup and the source NIXL install.
RUN MOONCAKE_PREFIX="${MOONCAKE_PREFIX}" \
    /usr/local/bin/aic-mooncake-gates.sh runtime

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
      ai.amd.aic.lmcache="${LMCACHE_REF}" \
      ai.amd.aic.nixl="${NIXL_REF}" \
      ai.amd.aic.hipfile="rocm-${ROCM_VERSION}" \
      ai.amd.aic.hsa-snoop="${HSA_SNOOP_REF}" \
      ai.amd.aic.mooncake="${MOONCAKE_REF}" \
      ai.amd.aic.mooncake.source="${MOONCAKE_GIT_URL}"

# vLLM OpenAI-compatible server is the default entrypoint.
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

# ----- 8. Wheel export stage -------------------------------------------------
# A scratch image whose only content is the pip wheels built above (source-built
# vLLM + patched LMCache + Mooncake ROCm + ROCm NIXL).  Extract them WITHOUT building/loading
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
