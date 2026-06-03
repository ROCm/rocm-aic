# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

# ATOM + vLLM + ROCm 7.2.x (Andy Luo ATOM+LMCache blog, May 2026).
# Pinned base — do not use vllm-latest (moving tag).
# Tag:  rocm/atom-dev:vllm-v0.19.0-nightly_20260522 (published 2026-05-22)
# Digest from Docker Hub API 2026-06-02; override: make build ATOM_DEV_DIGEST=sha256:...
ARG ATOM_DEV_DIGEST=sha256:c7e019b4872efe85557097731e863207189aaaa80ce8071c534565b20d03a509
FROM rocm/atom-dev@${ATOM_DEV_DIGEST}

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

# Use atom-dev venv explicitly; do not `pip install --upgrade pip` (can break pins).

COPY recipies/vllm-lmcache-hipfile/patches/lmcache-pr-3008-cache-salt.patch \
    recipies/vllm-atom-andy/requirements-lmcache-runtime.txt \
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
        /opt/venv/bin/pip install --no-build-isolation --no-cache-dir -e . --no-deps && \
    /opt/venv/bin/pip install --no-cache-dir -r /app/patches/requirements-lmcache-runtime.txt

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

COPY recipies/vllm-atom-andy/scripts/ /app/scripts/
COPY recipies/vllm-atom-andy/configs/ /app/configs/

# Guard: LMCache must not upgrade atom-dev's ROCm PyTorch; c_ops .so must exist.
RUN /opt/venv/bin/python /app/scripts/verify-lmcache-hip.py

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib:/opt/rocm/lib \
    PATH=/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

ENTRYPOINT ["python3", "/app/scripts/vllm-server"]
