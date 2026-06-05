# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

# gfx950 dGPU vLLM + LMCache (DriveNets / rocm/vllm 7.13).
ARG VLLM_BASE_TAG=rocm7.13.0_gfx950-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1
FROM rocm/vllm:${VLLM_BASE_TAG}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ROCM_ARCH
RUN test -n "$ROCM_ARCH" || (echo "ERROR: ROCM_ARCH is not set" && exit 1)

ARG LMCACHE_SHA=92fe433ac8e21baf33d9a1b5dda835e1e56c53dd
ARG LMCACHE_CACHE_SALT_PATCH=lmcache-pr-3008-cache-salt.patch

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

COPY recipies/vllm-lmcache-hipfile/patches/lmcache-pr-3008-cache-salt.patch \
    recipies/aic-drivenets/requirements-lmcache-runtime.txt \
    /app/patches/

RUN cd /app && \
    git clone https://github.com/riley-dixon/LMCache.git && \
    cd LMCache && \
    git checkout "${LMCACHE_SHA}" && \
    git apply "/app/patches/${LMCACHE_CACHE_SALT_PATCH}" && \
    PYTORCH_ROCM_ARCH=${ROCM_ARCH} \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        CXX=hipcc \
        BUILD_WITH_HIP=1 \
        python3 -m pip install --no-build-isolation --no-cache-dir -e . --no-deps && \
    python3 -m pip install --no-cache-dir -r /app/patches/requirements-lmcache-runtime.txt && \
    python3 -m pip uninstall -y \
        nixl nixl-cu12 nixl-cu13 cupy-cuda12x cufile-python cuda-pathfinder \
        2>/dev/null || true

RUN python3 <<'PY'
from pathlib import Path

p = Path("/app/LMCache/benchmarks/long_doc_qa/long_doc_qa.py")
if p.is_file():
    text = p.read_text(encoding="utf-8")
    needle = "before averaging. Example: 0.1 drops bottom 10% and top 10%."
    if needle in text:
        p.write_text(text.replace(needle, needle.replace("%", "%%"), 1), encoding="utf-8")
PY

RUN mkdir -p /app/configs /app/scripts

COPY recipies/aic-drivenets/scripts/ /app/scripts/
COPY recipies/aic-drivenets/configs/ /app/configs/

RUN python3 /app/scripts/verify-lmcache-hip.py

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    VLLM_FLOAT32_MATMUL_PRECISION=high \
    LD_LIBRARY_PATH=/opt/rocm/lib \
    PATH=/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

ENTRYPOINT ["python3", "/app/scripts/vllm-server"]
