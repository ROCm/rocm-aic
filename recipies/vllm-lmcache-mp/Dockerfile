# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# vllm-lmcache-mp: single image used by both docker compose services.
#
#   lmcache service: docker compose runs  lmcache server --host 0.0.0.0 ...
#   vllm    service: docker compose runs  vllm serve ... --kv-transfer-config
#
# Storage (owned by the lmcache container):
#   L1: GPU / CPU memory managed by lmcache-server (--l1-size-gb)
#   L2a (NVMe): nixl_store AIS_MT adapter  → bind-mounted local NVMe (hipFile P2PDMA)
#   L2b (NFS):  nixl_store POSIX  adapter  → bind-mounted NFS-over-RDMA mount
#
# Build (from repo root):
#   DOCKER_BUILDKIT=1 docker build \
#     --build-arg ROCM_ARCH=gfx942 \
#     -f recipies/vllm-lmcache-mp/Dockerfile \
#     -t vllm-lmcache-mp .
#
# Orchestration: make -C recipies/vllm-lmcache-mp up

FROM vllm/vllm-openai-rocm:v0.23.0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ROCM_ARCH
RUN test -n "$ROCM_ARCH" || (echo "ERROR: ROCM_ARCH is not set" && exit 1)

# Pinned component revisions — bump together when validating a new combo.
ARG HIPFILE_SHA=adc1a2d2f8e886e93c5e55e674dbff6e05db2dbe
ARG NIXL_GIT_URL=https://github.com/sbates130272/nixl.git
ARG NIXL_SHA=9d146424117a7cd022bdbaec8cbd7551f1c08add
ARG NIXL_ENABLE_AIS=1

# LMCache: Ivan's public fork with hipFile MP support.
# The single commit on top of upstream (301f04a) adds --gds-l1-backend hipfile.
ARG LMCACHE_GIT_URL=https://github.com/amd-ivaganev/LMCache.git
ARG LMCACHE_GIT_REF=hipfile-for-mp
# Pin SHA for reproducible builds; leave empty to track branch head.
ARG LMCACHE_SHA=301f04a114ac1089d51807cac5ccc549e6afc71a

# ----- 1. Build dependencies -----------------------------------------------
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        git \
        libboost-all-dev \
        libmount-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip

# ----- 2. LMCache (upstream @ main + AMD patches) ---------------------------
RUN git clone --branch "${LMCACHE_GIT_REF}" "${LMCACHE_GIT_URL}" /app/LMCache && \
    if [ -n "${LMCACHE_SHA}" ]; then \
        git -C /app/LMCache checkout "${LMCACHE_SHA}"; \
    fi

# Apply patches in order; --check first so already-merged ones are skipped.
COPY recipies/vllm-lmcache-hipfile/patches/lmcache-pr-3008-cache-salt.patch \
     recipies/vllm-lmcache-hipfile/patches/lmcache-sha256-cbor-int.patch \
     recipies/vllm-lmcache-hipfile/patches/lmcache-controller-log.patch \
     recipies/vllm-lmcache-hipfile/patches/lmcache-chunk-statistics-hash.patch \
     recipies/vllm-lmcache-nixl/patches/lmcache-nixl-ais-backend.patch \
     recipies/vllm-lmcache-mp/patches/lmcache-hipfile-mp.patch \
     recipies/vllm-lmcache-mp/patches/lmcache-nixl-ais-mt-l2-adapter.patch \
     /app/patches/

RUN cd /app/LMCache && \
    for patch in \
        /app/patches/lmcache-pr-3008-cache-salt.patch \
        /app/patches/lmcache-sha256-cbor-int.patch \
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
    PYTORCH_ROCM_ARCH=${ROCM_ARCH} \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        CXX=hipcc \
        BUILD_WITH_HIP=1 \
        python3 -m pip install --no-build-isolation --no-cache-dir -e .

# Fix argparse %-format bug in long_doc_qa benchmark helper.
RUN python3 <<'PY'
from pathlib import Path
p = Path("/app/LMCache/benchmarks/long_doc_qa/long_doc_qa.py")
if p.exists():
    text = p.read_text(encoding="utf-8")
    needle = "before averaging. Example: 0.1 drops bottom 10% and top 10%."
    if needle in text:
        p.write_text(text.replace(needle, needle.replace("%", "%%"), 1), encoding="utf-8")
PY

RUN python3 -m pip install --no-cache-dir "matplotlib>=3.8,<4"
RUN pip uninstall -y nixl nixl-cu12 nixl-cu13 2>/dev/null || true

# ----- 3. ROCm/rocm-systems hipFile (DEB + Python binding) -----------------
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
    pip install --no-cache-dir -e . \
        -Ccmake.define.HIPFILE_INCLUDE_DIR=../include \
        -Ccmake.define.HIP_INCLUDE_DIR=/opt/rocm/include && \
    test -x /app/rocm-systems/projects/hipfile/build/tools/ais-stats/ais-stats && \
    ln -sf /app/rocm-systems/projects/hipfile/build/tools/ais-stats/ais-stats \
        /app/ais-stats && \
    ln -sf /app/ais-stats /usr/local/bin/ais-stats

# ----- 4. NIXL (andyluo7/nixl @ amd-support + AIS overlay) ----------------
COPY recipies/common/nixl /tmp/nixl-common
RUN chmod +x /tmp/nixl-common/*.sh /tmp/nixl-common/apply-ais-overlay.sh && \
    NIXL_GIT_URL="${NIXL_GIT_URL}" NIXL_SHA="${NIXL_SHA}" NIXL_REQUIRE_ROCM=1 \
        NIXL_DEST=/tmp/nixl /tmp/nixl-common/clone-nixl.sh && \
    NIXL_SRC=/tmp/nixl NIXL_ENABLE_AIS="${NIXL_ENABLE_AIS}" \
        AIS_PATH=/opt/rocm ROCM_PATH=/opt/rocm \
        NIXL_INSTALL_PREFIX=/opt/nixl \
        /tmp/nixl-common/build-nixl.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    NIXL_PLUGIN_DIR=/opt/nixl/lib/x86_64-linux-gnu/plugins \
    PYTHONPATH=/opt/nixl/lib/python3/dist-packages \
    LD_LIBRARY_PATH=/opt/nixl/lib/x86_64-linux-gnu:/opt/nixl/lib:/opt/rocnixl-ucx/lib:/opt/rocm/lib
