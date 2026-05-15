FROM vllm/vllm-openai-rocm:v0.19.0

# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

ARG ROCM_ARCH
RUN test -n "$ROCM_ARCH" || (echo "ERROR: ROCM_ARCH is not set" && exit 1)

# LMCache pin aligned with **origin/dev/kumcmill/vllm**: **HipFileMemoryAllocator**
# uses **``from hipfile import Buffer``** (rocm-systems layout), not **``hipfile.bindings``**.
ARG LMCACHE_SHA=92fe433ac8e21baf33d9a1b5dda835e1e56c53dd
ARG HIPFILE_SHA=adc1a2d2f8e886e93c5e55e674dbff6e05db2dbe
ARG FIO_SHA=ee6ea2ebcc266f766c69df9d7990f1255e9f59f9

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        git \
        libboost-program-options-dev \
        libmount-dev \
        ninja-build \
        pkg-config \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# LMCache (Riley Dixon fork)
RUN cd /app && \
    git clone https://github.com/riley-dixon/LMCache.git && \
    cd LMCache && \
    git checkout "${LMCACHE_SHA}" && \
    PYTORCH_ROCM_ARCH=${ROCM_ARCH} \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        CXX=hipcc \
        BUILD_WITH_HIP=1 \
        python3 -m pip install --no-cache-dir --no-build-isolation -e .

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

# hipFile: sparse checkout of rocm-systems (projects/hipfile only)
RUN cd /app && \
    git clone --filter=blob:none --no-checkout \
        https://github.com/ROCm/rocm-systems.git && \
    cd rocm-systems && \
    git sparse-checkout init --cone && \
    git sparse-checkout set projects/hipfile && \
    git checkout "${HIPFILE_SHA}" && \
    mkdir -p build && \
    cd build && \
    cmake -G Ninja \
        -DGPU_TARGETS=${ROCM_ARCH} \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_HIP_PLATFORM=amd \
        ../projects/hipfile && \
    cmake --build . -j"$(nproc)" && \
    cmake --install . && \
    cd ../projects/hipfile/python && \
    python3 -m pip install --no-cache-dir --no-build-isolation -e . \
        -Ccmake.define.HIPFILE_INCLUDE_DIR=/app/rocm-systems/projects/hipfile/include \
        -Ccmake.define.HIP_INCLUDE_DIR=/opt/rocm/include

# fio with libhipfile (ROCm fork)
RUN cd /app && \
    git clone https://github.com/ROCm/fio.git && \
    cd fio && \
    git checkout "${FIO_SHA}" && \
    ./configure --prefix=/usr/local --enable-libhipfile && \
    make -j"$(nproc)" && \
    make install

COPY scripts/* /app/

RUN echo 'set -o vi' >> /root/.bashrc

ENTRYPOINT ["/bin/bash"]
CMD []
