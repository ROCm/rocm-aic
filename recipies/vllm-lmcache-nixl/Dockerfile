# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

FROM vllm/vllm-openai-rocm:v0.19.0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ROCM_ARCH
RUN test -n "$ROCM_ARCH" || (echo "ERROR: ROCM_ARCH is not set" && exit 1)

ARG LMCACHE_SHA=92fe433ac8e21baf33d9a1b5dda835e1e56c53dd
ARG LMCACHE_CACHE_SALT_PATCH=lmcache-pr-3008-cache-salt.patch
ARG LMCACHE_NIXL_AIS_PATCH=lmcache-nixl-ais-backend.patch
ARG HIPFILE_SHA=adc1a2d2f8e886e93c5e55e674dbff6e05db2dbe
ARG NIXL_GIT_URL=https://github.com/andyluo7/nixl.git
ARG NIXL_SHA=f72aad2cf4da0dff5d710dfcaa8666defa114d78
ARG NIXL_ENABLE_AIS=1

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

COPY recipies/vllm-lmcache-hipfile/patches/lmcache-pr-3008-cache-salt.patch \
    recipies/vllm-lmcache-hipfile/patches/lmcache-gds-eviction-log.patch \
    recipies/vllm-lmcache-hipfile/patches/lmcache-sha256-cbor-int.patch \
    recipies/vllm-lmcache-hipfile/patches/lmcache-controller-log.patch \
    recipies/vllm-lmcache-hipfile/patches/lmcache-chunk-statistics-hash.patch \
    /app/patches/
COPY recipies/vllm-lmcache-nixl/patches/ /app/patches-nixl/

RUN cd /app && \
    git clone https://github.com/riley-dixon/LMCache.git && \
    cd LMCache && \
    git checkout "${LMCACHE_SHA}" && \
    git apply "/app/patches/${LMCACHE_CACHE_SALT_PATCH}" && \
    git apply /app/patches/lmcache-gds-eviction-log.patch && \
    git apply /app/patches/lmcache-sha256-cbor-int.patch && \
    git apply /app/patches/lmcache-controller-log.patch && \
    git apply /app/patches/lmcache-chunk-statistics-hash.patch && \
    git apply "/app/patches-nixl/${LMCACHE_NIXL_AIS_PATCH}" && \
    PYTORCH_ROCM_ARCH=${ROCM_ARCH} \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        CXX=hipcc \
        BUILD_WITH_HIP=1 \
        python3 -m pip install --no-build-isolation --no-cache-dir -e .

RUN python3 <<'PY'
from pathlib import Path

p = Path("/app/LMCache/benchmarks/long_doc_qa/long_doc_qa.py")
text = p.read_text(encoding="utf-8")
needle = "before averaging. Example: 0.1 drops bottom 10% and top 10%."
if needle not in text:
    raise SystemExit(f"Expected substring not in {p} (LMCache revision changed?)")
p.write_text(text.replace(needle, needle.replace("%", "%%"), 1), encoding="utf-8")
PY

RUN python3 -m pip install --no-cache-dir "matplotlib>=3.8,<4"
RUN pip uninstall -y nixl nixl-cu12 nixl-cu13 2>/dev/null || true

# hipFile for NIXL AIS plugin.
RUN git clone --no-checkout --filter=blob:none \
        https://github.com/ROCm/rocm-systems.git /app/rocm-systems && \
    cd /app/rocm-systems && \
    git sparse-checkout set --cone projects/hipfile && \
    git checkout "${HIPFILE_SHA}" && \
    cd projects/hipfile && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_HIP_PLATFORM=amd .. && \
    cmake --build . -j"$(nproc)" && \
    cpack -G DEB && \
    dpkg -i hipfile*deb && \
    cd ../python && \
    pip install --no-cache-dir -e . \
        -Ccmake.define.HIPFILE_INCLUDE_DIR=../include \
        -Ccmake.define.HIP_INCLUDE_DIR=/opt/rocm/include && \
    python3 -c "from hipfile import Buffer; from hipfile._hipfile import hipFileBufRegister" && \
    test -x /app/rocm-systems/projects/hipfile/build/tools/ais-stats/ais-stats && \
    ln -sf /app/rocm-systems/projects/hipfile/build/tools/ais-stats/ais-stats \
        /app/ais-stats && \
    ln -sf /app/ais-stats /usr/local/bin/ais-stats

COPY recipies/common/nixl /tmp/nixl-common
RUN chmod +x /tmp/nixl-common/*.sh /tmp/nixl-common/apply-ais-overlay.sh && \
    NIXL_GIT_URL="${NIXL_GIT_URL}" NIXL_SHA="${NIXL_SHA}" NIXL_REQUIRE_ROCM=1 \
        NIXL_DEST=/tmp/nixl /tmp/nixl-common/clone-nixl.sh && \
    NIXL_SRC=/tmp/nixl NIXL_ENABLE_AIS="${NIXL_ENABLE_AIS}" \
        AIS_PATH=/opt/rocm ROCM_PATH=/opt/rocm \
        NIXL_INSTALL_PREFIX=/opt/nixl \
        /tmp/nixl-common/build-nixl.sh

RUN mkdir -p /app/configs /app/scripts

COPY recipies/vllm-lmcache-nixl/scripts/ /app/scripts/
COPY recipies/vllm-lmcache-nixl/configs/ /app/configs/
COPY benchmarks/llm-prefill-benchmark/scripts/test-aic.py /app/scripts/

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    NIXL_PLUGIN_DIR=/opt/nixl/lib/x86_64-linux-gnu/plugins \
    PYTHONPATH=/opt/nixl/lib/python3/dist-packages \
    LD_LIBRARY_PATH=/opt/nixl/lib/x86_64-linux-gnu:/opt/nixl/lib:/opt/rocnixl-ucx/lib:/opt/rocm/lib

ENTRYPOINT ["python3", "/app/scripts/vllm-server"]
