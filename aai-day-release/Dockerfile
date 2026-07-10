# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# aai-day-release: Ubuntu 24.04 + ROCm 7.2.2 inference stack.
#
#   Base image: rocm/dev-ubuntu-24.04:7.2.2-complete  (Ubuntu 24.04, ROCm 7.2.2, Python 3.12)
#   vLLM:       installed from pre-built ROCm wheel (bundles its own torch/triton/flash-attn)
#   LMCache:    LMCache/LMCache upstream main + 6 AMD patches
#   hipFile:    ROCm/rocm-systems (GDS/P2PDMA NVMe transfers)
#   NIXL:       sbates130272/nixl @ feat/amd-ais-mt (AIS_MT plugin)
#
# Two compose services share this single image:
#   lmcache — standalone lmcache-server (MP mode)
#               L1:  GPU/CPU DRAM (--l1-size-gb)  OR  hipFile NVMe slab (GDS L1 mode)
#               L2a: nixl_store AIS_MT → local NVMe (hipFile P2PDMA)
#               L2b: nixl_store POSIX  → NFS-over-RDMA
#   vllm    — vllm serve with LMCacheMPConnector (ZMQ to lmcache)
#
# Build (from repo root):
#   DOCKER_BUILDKIT=1 docker build \
#     --build-arg ROCM_ARCH=gfx942 \
#     -f aai-day-release/Dockerfile \
#     -t rocm-aic-aai-day .
#
#   # Behind AMD Zscaler proxy — pass cert as a BuildKit secret (never baked into image):
#   DOCKER_BUILDKIT=1 docker build \
#     --build-arg ROCM_ARCH=gfx942 \
#     --secret id=tls_cert,src=/path/to/corp-ca.crt \
#     -f aai-day-release/Dockerfile \
#     -t rocm-aic-aai-day .
#
# Orchestration: make -C aai-day-release up
#   make -C aai-day-release build                                  # no proxy
#   make -C aai-day-release build TLS_CERT=/path/to/corp-ca.crt   # behind proxy
#
# LMCache patches applied (in order):
#   lmcache-pr-3008-cache-salt.patch      — vLLM cache_salt → lmcache.tag.cachesalt
#   lmcache-controller-log.patch          — demote noisy controller-absent warning
#   lmcache-chunk-statistics-hash.patch   — pass config into ChunkedTokenDatabase
#   lmcache-nixl-ais-backend.patch        — AIS/AIS_MT in nixl_storage_backend
#   lmcache-hipfile-mp.patch              — hipFile driver ordering fix + Prometheus gauges
#   lmcache-nixl-ais-mt-l2-adapter.patch  — AIS_MT in nixl_store_l2_adapter + GDS L1 compat
# NOTE: lmcache-sha256-cbor-int.patch intentionally omitted — merged upstream (PR #2999).

FROM rocm/dev-ubuntu-24.04:7.2.2-complete

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ROCM_ARCH
RUN test -n "$ROCM_ARCH" || (echo "ERROR: ROCM_ARCH is not set" && exit 1)

# vLLM wheel version on wheels.vllm.ai.  rocm722 = ROCm 7.2.x, Python 3.12.
ARG VLLM_VERSION=0.23.0
ARG VLLM_ROCM_VARIANT=rocm722

# Pinned component revisions — bump together when validating a new combo.
ARG HIPFILE_SHA=adc1a2d2f8e886e93c5e55e674dbff6e05db2dbe
ARG NIXL_GIT_URL=https://github.com/sbates130272/nixl.git
ARG NIXL_SHA=9d146424117a7cd022bdbaec8cbd7551f1c08add
ARG NIXL_ENABLE_AIS=1

# LMCache: upstream repo.  Patches below add AMD-specific features.
ARG LMCACHE_GIT_URL=https://github.com/LMCache/LMCache.git
ARG LMCACHE_GIT_REF=main
# Pin to a validated commit; leave empty to track branch head.
ARG LMCACHE_SHA=

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
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        curl \
        git \
        libboost-all-dev \
        libibverbs-dev \
        ibverbs-providers \
        libmount-dev \
        ninja-build \
        pkg-config \
        python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# ----- 2. uv (fast Python package manager) + pip bootstrap ------------------
RUN curl -fsSL https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv && \
    ln -s /root/.local/bin/uvx /usr/local/bin/uvx

# ----- 3. vLLM (pre-built ROCm wheel — bundles its own torch/triton/flash-attn)
# vllm 0.23.0+rocm722 requires Python 3.12; rocm/dev-ubuntu-24.04 ships 3.12 by default.
# The wheel comes from AMD's wheel server; --extra-index-url lets uv find the +rocm722 variant.
RUN uv pip install --system \
        "vllm==${VLLM_VERSION}+${VLLM_ROCM_VARIANT}" \
        --extra-index-url "https://wheels.vllm.ai/rocm/${VLLM_VERSION}/${VLLM_ROCM_VARIANT}"

# ----- 4. LMCache (upstream main + AMD patches) ------------------------------
RUN git clone --branch "${LMCACHE_GIT_REF}" "${LMCACHE_GIT_URL}" /app/LMCache && \
    if [ -n "${LMCACHE_SHA}" ]; then \
        git -C /app/LMCache checkout "${LMCACHE_SHA}"; \
    fi

COPY aai-day-release/patches/ /app/patches/

RUN cd /app/LMCache && \
    for patch in \
        /app/patches/lmcache-pr-3008-cache-salt.patch \
        /app/patches/lmcache-controller-log.patch \
        /app/patches/lmcache-chunk-statistics-hash.patch \
        /app/patches/lmcache-nixl-ais-backend.patch \
        /app/patches/lmcache-hipfile-mp.patch \
        /app/patches/lmcache-nixl-ais-mt-l2-adapter.patch; do \
        if git apply --check "$patch" 2>/dev/null; then \
            git apply "$patch" && echo "Applied: $patch"; \
        else \
            echo "Skipped (already merged or conflict): $patch"; \
        fi; \
    done

RUN cd /app/LMCache && \
    PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        CXX=hipcc \
        BUILD_WITH_HIP=1 \
        uv pip install --system --no-build-isolation --no-cache -e .

RUN uv pip install --system --no-cache "matplotlib>=3.8,<4"

# Remove any NIXL wheels that upstream packages may have pulled — we build from source below.
RUN uv pip uninstall --system nixl nixl-cu12 nixl-cu13 2>/dev/null || true

# ----- 5. ROCm/rocm-systems hipFile (DEB + Python binding) ------------------
RUN git clone --no-checkout --filter=blob:none \
        https://github.com/ROCm/rocm-systems.git /app/rocm-systems && \
    cd /app/rocm-systems && \
    git sparse-checkout set --cone projects/hipfile && \
    git checkout "${HIPFILE_SHA}" && \
    cd projects/hipfile && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_HIP_PLATFORM=amd -DAIS_INSTALL_TOOLS=ON \
          -DAMDGPU_TARGETS="${ROCM_ARCH}" \
          -DGPU_TARGETS="${ROCM_ARCH}" \
          -DCMAKE_HIP_ARCHITECTURES="${ROCM_ARCH}" .. && \
    cmake --build . -j"$(nproc)" && \
    cpack -G DEB && \
    dpkg -i hipfile*deb && \
    cd ../python && \
    uv pip install --system --no-cache -e . \
        -Ccmake.define.HIPFILE_INCLUDE_DIR=../include \
        -Ccmake.define.HIP_INCLUDE_DIR=/opt/rocm/include && \
    test -x /app/rocm-systems/projects/hipfile/build/tools/ais-stats/ais-stats && \
    ln -sf /app/rocm-systems/projects/hipfile/build/tools/ais-stats/ais-stats \
        /app/ais-stats && \
    ln -sf /app/ais-stats /usr/local/bin/ais-stats

# ----- 6. NIXL (sbates130272/nixl @ feat/amd-ais-mt + AIS overlay) ----------
COPY recipies/common/nixl /tmp/nixl-common
RUN chmod +x /tmp/nixl-common/*.sh /tmp/nixl-common/apply-ais-overlay.sh && \
    NIXL_GIT_URL="${NIXL_GIT_URL}" NIXL_SHA="${NIXL_SHA}" NIXL_REQUIRE_ROCM=1 \
        NIXL_DEST=/tmp/nixl /tmp/nixl-common/clone-nixl.sh && \
    NIXL_SRC=/tmp/nixl NIXL_ENABLE_AIS="${NIXL_ENABLE_AIS}" \
        AIS_PATH=/opt/rocm ROCM_PATH=/opt/rocm \
        NIXL_INSTALL_PREFIX=/opt/nixl \
        UV_NATIVE_TLS=1 \
        /tmp/nixl-common/build-nixl.sh

# ----- 7. Runtime environment ------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    NIXL_PLUGIN_DIR=/opt/nixl/lib/x86_64-linux-gnu/plugins \
    PYTHONPATH=/opt/nixl/lib/python3/dist-packages \
    LD_LIBRARY_PATH=/opt/nixl/lib/x86_64-linux-gnu:/opt/nixl/lib:/opt/rocnixl-ucx/lib:/opt/rocm/lib

# vLLM OpenAI-compatible server is the default entrypoint.
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
