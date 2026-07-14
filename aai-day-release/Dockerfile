# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# aai-day-release: Ubuntu 24.04 + ROCm 7.2.2 inference stack.
#
#   Base image: rocm/dev-ubuntu-24.04:7.2.4-complete  (Ubuntu 24.04, ROCm 7.2.4, Python 3.12)
#   vLLM:       installed from pre-built ROCm wheel (bundles its own torch/triton/flash-attn)
#   LMCache:    LMCache/LMCache upstream dev + 7 AMD patches
#   hipFile:    ROCm/rocm-systems (GDS/P2PDMA NVMe transfers)
#   NIXL:       ai-dynamo/nixl upstream main + nixl-rocm-ais-mt.patch (ROCm/HIP + AIS_MT)
#
# Two compose services share this single image:
#   lmcache — standalone lmcache-server (MP mode)
#               L1:  GPU/CPU DRAM (--l1-size-gb)  OR  hipFile NVMe slab (GDS L1 mode)
#               L2a: nixl_store AIS_MT → local NVMe (hipFile P2PDMA)
#               L2b: nixl_store POSIX  → NFS-over-RDMA
#   vllm    — vllm serve with LMCacheMPConnector (ZMQ to lmcache)
#
# Build (from this directory, aai-day-release/ — the build context is this tree):
#   DOCKER_BUILDKIT=1 docker build \
#     --build-arg ROCM_ARCH=gfx942 \
#     -f Dockerfile \
#     -t rocm-aic-aai-day .
#
#   # Behind AMD Zscaler proxy — pass cert as a BuildKit secret (never baked into image):
#   DOCKER_BUILDKIT=1 docker build \
#     --build-arg ROCM_ARCH=gfx942 \
#     --secret id=tls_cert,src=/path/to/corp-ca.crt \
#     -f Dockerfile \
#     -t rocm-aic-aai-day .
#
# Orchestration: make -C aai-day-release up
#   make -C aai-day-release build                                  # no proxy
#   make -C aai-day-release build TLS_CERT=/path/to/corp-ca.crt   # behind proxy
#
# Patches: every *.patch under patches/lmcache/ and patches/nixl/ is applied to
# its upstream checkout during the build, in lexical (filename) order -- hence
# the NN- prefixes.  Drop a new NN-*.patch into the directory to include it.

FROM rocm/dev-ubuntu-24.04:7.2.4-complete

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ROCM_ARCH
RUN test -n "$ROCM_ARCH" || (echo "ERROR: ROCM_ARCH is not set" && exit 1)

# vLLM wheel version on wheels.vllm.ai.  rocm723 = ROCm 7.2.3, Python 3.12 --
# the only ROCm variant published for 0.25.0 (ABI-compatible with the 7.2.4 base).
# Check https://wheels.vllm.ai/rocm/<version>/ for the variant tag on version bumps.
ARG VLLM_VERSION=0.25.0
ARG VLLM_ROCM_VARIANT=rocm723

# Pinned component revisions — bump together when validating a new combo.
# ROCm/rocm-systems develop HEAD (2026-07-13); default branch is 'develop'.
ARG HIPFILE_SHA=6901b670c3e186f9c7052d6fc9748f0286536bdc
# ai-dynamo/nixl upstream main (2026-07-10) + nixl-rocm-ais-mt.patch (ROCm/HIP + AIS_MT).
ARG NIXL_GIT_URL=https://github.com/ai-dynamo/nixl.git
ARG NIXL_SHA=644facf0eb3de14ec63c1d2831238f63cd03c0e0

# hsa-snoop: HSA AQL queue/dispatch snooper + Prometheus exporter (main HEAD 2026-07-13).
ARG HSA_SNOOP_GIT_URL=https://github.com/sbates130272/hsa-snoop.git
ARG HSA_SNOOP_SHA=3bb046763f03e429c4fd983b36346b4727fc3744

# LMCache: upstream repo, pinned to a dev commit above the v0.5.1 tag
# (dev HEAD as of 2026-07-13; upstream's default branch is 'dev', not 'main').
# All 7 AMD patches in patches/lmcache/ apply cleanly against this SHA.
ARG LMCACHE_GIT_URL=https://github.com/LMCache/LMCache.git
ARG LMCACHE_SHA=21b3341943342991534d30249c093d0ddc390bdf

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

# Ubuntu 24.04 marks the system interpreter PEP-668 "externally managed", which
# makes both `uv pip install --system` and plain `python3 -m pip install` refuse
# to touch /usr.  This is a single-purpose container built entirely on the system
# Python, so drop the marker to allow system-wide installs.
RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

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

# ----- 4. LMCache (upstream dev + AMD patches) -------------------------------
# Clone the default branch (fetches all branches + tags), then check out the
# pinned SHA -- avoids depending on a specific branch name that may not exist.
RUN git clone "${LMCACHE_GIT_URL}" /app/LMCache && \
    git -C /app/LMCache checkout "${LMCACHE_SHA}"

COPY patches/ /app/patches/

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
            echo "The LMCache pin (${LMCACHE_SHA}) likely drifted; rebase this patch against it." >&2; \
            exit 1; \
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
    ln -sf /app/ais-stats /usr/local/bin/ais-stats && \
    # ais-check is a source-tree Python script (checks HIP-runtime + amdgpu AIS
    # support at run time); no build/DEB install rule, so symlink it onto PATH.
    test -f /app/rocm-systems/projects/hipfile/tools/ais-check/ais-check && \
    install -m0755 /app/rocm-systems/projects/hipfile/tools/ais-check/ais-check \
        /usr/local/bin/ais-check

# ----- 6. NIXL (ai-dynamo/nixl upstream + nixl-rocm-ais-mt.patch) ------------
# Clone upstream ai-dynamo/nixl, then apply nixl-rocm-ais-mt.patch to add ROCm/HIP
# detection and the AIS_MT plugin tree.  The patched checkout is a native-HIP
# tree, so build-nixl.sh's _NATIVE_HIP path builds AIS_MT with AIS_PATH=/opt/rocm.
COPY common/nixl /tmp/nixl-common
COPY patches/nixl/ /tmp/nixl-patches/
# Apply every patch under patches/nixl/ (lexical order) to the fresh checkout.
RUN chmod +x /tmp/nixl-common/*.sh && \
    NIXL_GIT_URL="${NIXL_GIT_URL}" NIXL_SHA="${NIXL_SHA}" NIXL_REQUIRE_ROCM=1 \
        NIXL_DEST=/tmp/nixl /tmp/nixl-common/clone-nixl.sh && \
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
        UV_NATIVE_TLS=1 \
        /tmp/nixl-common/build-nixl.sh

# ----- 6b. hsa-snoop (HSA AQL queue snooper + Prometheus exporter) -----------
# GPU dispatch/queue telemetry on :9488 (/metrics).  The snooper binary is
# host-only C++ (x86-64) and ARCH-INDEPENDENT -- it parses HSA AQL queues from
# userspace, so one binary runs on every gfx target we ship.  We disable the
# optional examples/ (CMAKE_DISABLE_FIND_PACKAGE_hip=ON): they compile HIP
# device code (examples/vecwork.hip) for a single auto-detected arch -- and on
# this CPU-only build node detection falls back to gfx1100, which neither
# matches our multi-arch set nor is needed by the exporter.
# Built WITH the Prometheus exporter (-DHSA_SNOOP_PROMETHEUS=ON fetches
# prometheus-cpp via CMake); the build is rejected if the resulting binary lacks
# --prometheus.  At runtime it must run as root in the host PID namespace,
# privileged (tracefs + pagemap + process_vm_readv), to see the vLLM/LMCache GPU
# processes -- see monitoring/docker-compose.monitoring.yml.
RUN git clone "${HSA_SNOOP_GIT_URL}" /app/hsa-snoop && \
    git -C /app/hsa-snoop checkout "${HSA_SNOOP_SHA}" && \
    cmake -S /app/hsa-snoop -B /app/hsa-snoop/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DHSA_SNOOP_PROMETHEUS=ON \
        -DCMAKE_DISABLE_FIND_PACKAGE_hip=ON && \
    cmake --build /app/hsa-snoop/build --parallel "$(nproc)" && \
    cmake --install /app/hsa-snoop/build --prefix /usr/local && \
    test -x /usr/local/bin/hsa-snoop && \
    { /usr/local/bin/hsa-snoop --help 2>&1 || true; } | grep -q -- '--prometheus' \
        || { echo "ERROR: hsa-snoop built without Prometheus mode (--prometheus absent)" >&2; exit 1; } && \
    echo "PASS: hsa-snoop with Prometheus exporter installed"

# ----- 7. Runtime environment ------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    NIXL_PLUGIN_DIR=/opt/nixl/lib/x86_64-linux-gnu/plugins \
    PYTHONPATH=/opt/nixl/lib/python3/dist-packages \
    LD_LIBRARY_PATH=/opt/nixl/lib/x86_64-linux-gnu:/opt/nixl/lib:/opt/rocnixl-ucx/lib:/opt/rocm/lib

# vLLM OpenAI-compatible server is the default entrypoint.
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
