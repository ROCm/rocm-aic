# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

FROM vllm/vllm-openai-rocm:v0.19.0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ROCM_ARCH
RUN test -n "$ROCM_ARCH" || (echo "ERROR: ROCM_ARCH is not set" && exit 1)

ARG LMCACHE_SHA=92fe433ac8e21baf33d9a1b5dda835e1e56c53dd
# Backport LMCache/LMCache#3008 (vLLM cache_salt in LMCache V1 cache identity).
ARG LMCACHE_CACHE_SALT_PATCH=lmcache-pr-3008-cache-salt.patch
ARG HIPFILE_SHA=adc1a2d2f8e886e93c5e55e674dbff6e05db2dbe

# Build deps for LMCache, rocm-systems hipFile (DEB + Python), and fio.
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

COPY patches/ /app/patches/

RUN cd /app && \
    git clone https://github.com/riley-dixon/LMCache.git && \
    cd LMCache && \
    git checkout "${LMCACHE_SHA}" && \
    git apply "/app/patches/${LMCACHE_CACHE_SALT_PATCH}" && \
    PYTORCH_ROCM_ARCH=${ROCM_ARCH} \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        CXX=hipcc \
        BUILD_WITH_HIP=1 \
        python3 -m pip install --no-build-isolation --no-cache-dir -e .

# argparse expands ``help=`` with %-formatting; upstream ``--trim-fraction`` text
# uses ``10%`` and breaks ``--help`` (``ValueError: incomplete format``).
RUN python3 <<'PY'
from pathlib import Path

p = Path("/app/LMCache/benchmarks/long_doc_qa/long_doc_qa.py")
text = p.read_text(encoding="utf-8")
needle = "before averaging. Example: 0.1 drops bottom 10% and top 10%."
if needle not in text:
    raise SystemExit(f"Expected substring not in {p} (LMCache revision changed?)")
p.write_text(text.replace(needle, needle.replace("%", "%%"), 1), encoding="utf-8")
PY

# **long_doc_qa --visualize** (LMCache benchmark) needs matplotlib in the image.
RUN python3 -m pip install --no-cache-dir "matplotlib>=3.8,<4"

# hipFile (rocm-systems sparse) and fio with libhipfile.
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
    cd /app && \
    git clone https://github.com/ROCm/fio.git && \
    cd fio && \
    git checkout origin/hipFile && \
    HIPINC=/app/rocm-systems/projects/hipfile/include && \
    HIPLIB=/app/rocm-systems/projects/hipfile/build/src/amd_detail && \
    CFLAGS="-I/opt/rocm/include -I${HIPINC}" \
        LDFLAGS="-L/opt/rocm/lib -L${HIPLIB} -Wl,-rpath,${HIPLIB}" \
        ./configure --enable-libhipfile && \
    make -j"$(nproc)"

RUN mkdir -p /app/configs /app/scripts

COPY scripts/ /app/scripts/
COPY configs/ /app/configs/

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0

ENTRYPOINT ["python3", "/app/scripts/vllm-server"]
