# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# vllm-optimus-kvd: ROCm vLLM + AMD-AGI Optimus kvd connector with
# chunked-fusion v2 + hipFile GPU-direct (PR #49 / branch
# feat/vllm-kvd-chunked-fusion-v2-mergemain-phase2).
#
# Image layers, top-down:
#   1. Base: vllm/vllm-openai-rocm:v0.21.0
#   2. Build deps (cmake, ninja, libboost, libmount) for hipFile
#   3. ROCm/rocm-systems hipFile (DEB + Python binding)
#   4. fio with libhipfile (for storage-tier BW benches)
#   5. Async patch over hipFile python binding (Optimus deploy/docker/patches/)
#   6. Optimus source  → /app/optimus
#   7. ENTRYPOINT = scripts/vllm-server (starts kvd daemon + vllm serve)
#
# Build:
#   make build ROCM_ARCH=gfx942        # MI300X/MI325X/MI355X
#   make build ROCM_ARCH=gfx1201       # RX 9070 XT
#
# Run + cliff bench:
#   make run HF_TOKEN=<...>            # start engine
#   make cliff                         # in another terminal: cliff sweep

FROM vllm/vllm-openai-rocm:v0.21.0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ROCM_ARCH
RUN test -n "$ROCM_ARCH" || (echo "ERROR: ROCM_ARCH is not set" && exit 1)

# Pinned upstream revs — bump together when validating a new combo.
ARG HIPFILE_SHA=adc1a2d2f8e886e93c5e55e674dbff6e05db2dbe
ARG OPTIMUS_GIT_URL=https://github.com/AMD-AGI/Optimus.git

# gpu-direct storage work is now merged into Optimus main
ARG OPTIMUS_GIT_REF=main

# ----- 1. Build deps ---------------------------------------------------
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

# ----- 2. rocm-systems hipFile (DEB + Python binding) ------------------
# Sparse clone keeps the layer small (rocm-systems is a monorepo).
RUN git clone --no-checkout --filter=blob:none \
        https://github.com/ROCm/rocm-systems.git /app/rocm-systems && \
    cd /app/rocm-systems && \
    git sparse-checkout set --cone projects/hipfile && \
    git checkout "${HIPFILE_SHA}" && \
    cd projects/hipfile && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_HIP_PLATFORM=amd -DAIS_INSTALL_TOOLS=ON .. && \
    cmake --build . -j"$(nproc)" && \
    cpack -G DEB && \
    dpkg -i hipfile*deb && \
    cd ../python && \
    pip install --no-cache-dir -e . \
        -Ccmake.define.HIPFILE_INCLUDE_DIR=../include \
        -Ccmake.define.HIP_INCLUDE_DIR=/opt/rocm/include && \
    test -x /app/rocm-systems/projects/hipfile/build/tools/ais-stats/ais-stats && \
    ln -sf /app/rocm-systems/projects/hipfile/build/tools/ais-stats/ais-stats \
        /app/ais-stats
# NOTE: ais-check is not a built target on this hipfile rev (only
# ais-stats is). P2PDMA detection at runtime falls back to inferring
# from ais-stats output. The hipfile python binding import smoke-test
#   python3 -c "from hipfile import Buffer; from hipfile._hipfile import hipFileBufRegister"
# requires GPU access (dlopen of libhipfile.so → ROCm runtime → /dev/kfd).
# Docker build typically has no GPU access, so this check is run at
# ENTRYPOINT instead.

# ----- 3. fio with libhipfile (for storage-tier BW benches) ------------
RUN cd /app && \
    git clone https://github.com/ROCm/fio.git && \
    cd fio && \
    git checkout origin/hipFile && \
    HIPINC=/app/rocm-systems/projects/hipfile/include && \
    HIPLIB=/app/rocm-systems/projects/hipfile/build/src/amd_detail && \
    CFLAGS="-I/opt/rocm/include -I${HIPINC}" \
        LDFLAGS="-L/opt/rocm/lib -L${HIPLIB} -Wl,-rpath,${HIPLIB}" \
        ./configure --enable-libhipfile && \
    make -j"$(nproc)"

# ----- 4. Optimus source (PR #49 branch) -------------------------------
# Shallow clone the PR branch; the recipe uses the connector source
# directly (no install needed — Optimus's vllm KV connector loads via
# --kv-connector-module-path = "rocserve.engine.vllm.kvd_connector").
#
# AMD-AGI/Optimus is currently a private repo, so the clone needs a
# GitHub token. We use a BuildKit secret mount (no token baked into
# the image, no token in `docker history`). Pass it via:
#   DOCKER_BUILDKIT=1 docker build --secret id=gh_token,env=GH_TOKEN ...
# (the Makefile does this for `make build`). Use a fine-grained PAT
# with `Contents: Read` on AMD-AGI/Optimus.
RUN --mount=type=secret,id=gh_token,target=/run/secrets/gh_token \
    if [ -s /run/secrets/gh_token ]; then \
        TOKEN="$(cat /run/secrets/gh_token)"; \
        git clone --depth 1 --branch "${OPTIMUS_GIT_REF}" \
            "https://oauth2:${TOKEN}@${OPTIMUS_GIT_URL#https://}" /app/optimus; \
    else \
        echo "ERROR: GH_TOKEN secret not provided. Set GH_TOKEN env and"; \
        echo "       pass --secret id=gh_token,env=GH_TOKEN to docker build,"; \
        echo "       or use 'make build' which wires it up automatically."; \
        exit 1; \
    fi

# ----- 5. Async patch over hipFile python binding ----------------------
# Optimus deploy/docker/patches/hipfile_async/ adds the async API
# (cuFileReadAsync / cuFileWriteAsync via heap-allocated _AsyncIOHandle).
# Without this, Optimus's GPU-direct load path can't issue async I/O
# and silently falls back to sync mmap+H2D.
RUN bash /app/optimus/deploy/docker/patches/hipfile_async/patch_hipfile_async.sh

# ----- 6. Optimus runtime deps -----------------------------------------
# Optimus's vllm KV connector + bench harness need:
#   - httpx for the bench client
#   - pyyaml for our config parsing
RUN python3 -m pip install --no-cache-dir httpx pyyaml

# ----- 7. Recipe overlay (scripts, configs, tests) ---------------------
RUN mkdir -p /app/configs /app/scripts

COPY recipies/vllm-optimus-gpu-direct/scripts/ /app/scripts/
COPY recipies/vllm-optimus-gpu-direct/configs/ /app/configs/
COPY recipies/vllm-optimus-gpu-direct/tests/   /app/tests/

# PYTHONHASHSEED=0 is required for cross-restart KV reuse — vLLM v1's
# `NONE_HASH = os.urandom(32)` salts every block hash per process,
# so without this the connector gets 0 cross-restart cache hits.
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PYTHONPATH=/app/optimus

ENTRYPOINT ["bash", "/app/scripts/vllm-server"]
