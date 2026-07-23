# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# AIC: Ubuntu 24.04 + ROCm 7.2.2 inference stack.
#
#   Base image: rocm/dev-ubuntu-24.04:7.2.4-complete  (Ubuntu 24.04, ROCm 7.2.4, Python 3.12)
#   vLLM:       installed from pre-built ROCm wheel (bundles its own torch/triton/flash-attn)
#   LMCache:    LMCache/LMCache v0.5.1 release + 7 AMD patches
#   hipFile:    ROCm/rocm-systems (GDS/P2PDMA NVMe transfers)
#   NIXL:       ai-dynamo/nixl v1.3.1 release + nixl-rocm-ais-mt.patch (ROCm/HIP + AIS_MT)
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
#     --build-arg ROCM_ARCH=gfx942 \
#     -f Dockerfile \
#     -t rocm-aic .
#
#   # Behind AMD Zscaler proxy — pass cert as a BuildKit secret (never baked into image):
#   DOCKER_BUILDKIT=1 docker build \
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
# Wheels: the build also produces pip-installable wheels for the patched LMCache
# and ROCm NIXL under /wheels.  Extract them with the `wheels` stage:
#   docker build --target wheels --output type=local,dest=./wheels .
# (the default target remains the full runtime image -- see stages 8/9 below).

# The main build stage is named `build`; two trailing stages at the end of this
# file expose the pip wheels (`wheels`) and the final runtime image (`runtime`,
# the default target -- identical to `build`, so existing `docker build` callers
# are unaffected).  Publishing wheels: `docker build --target wheels
# --output type=local,dest=<dir>` extracts /wheels without loading the image.
FROM rocm/dev-ubuntu-24.04:7.2.4-complete AS build

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
# ai-dynamo/nixl release tag + nixl-rocm-ais-mt.patch (ROCm/HIP + AIS_MT).
# The patch is AIS_MT-only (it no longer relocates upstream GDS_MT), so it applies
# cleanly on both v1.3.0 and v1.3.1 -- bump NIXL_REF and re-run the patch --check.
ARG NIXL_GIT_URL=https://github.com/ai-dynamo/nixl.git
ARG NIXL_REF=v1.3.1

# hsa-snoop: HSA AQL queue/dispatch snooper + Prometheus exporter (includes AIS metrics).
ARG HSA_SNOOP_GIT_URL=https://github.com/sbates130272/hsa-snoop.git
ARG HSA_SNOOP_REF=v1.0.0

# LMCache: upstream repo, pinned to the v0.5.1 release tag.
# All 7 AMD patches in patches/lmcache/ apply cleanly against this tag.
ARG LMCACHE_GIT_URL=https://github.com/LMCache/LMCache.git
ARG LMCACHE_REF=v0.5.1

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
# components (NIXL, hipFile, hsa-snoop) declare a boost dependency, and the full
# metapackage drags in libboost-mpi-dev -> openmpi-bin/-dev + gfortran + automake,
# which emit a raft of update-alternatives man-page warnings and bloat the image.
# Headers stay available for any transitive #include.
#
# libopenmpi3t64 (openmpi RUNTIME libs) is required on its own, though: the ROCm
# torch bundled in the vLLM wheel has a hard NEEDED on libmpi_cxx.so.40, loaded
# via torch._load_global_deps() on `import torch` -- it was previously satisfied
# incidentally by libboost-all-dev pulling openmpi.  We add just the runtime lib
# (not openmpi-bin/-dev), so torch imports but the mpirun/mpicc alternatives that
# caused the warnings are never registered.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        curl \
        git \
        libboost-dev \
        libibverbs-dev \
        ibverbs-providers \
        libmount-dev \
        libopenmpi3t64 \
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
RUN curl -fsSL https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv && \
    ln -s /root/.local/bin/uvx /usr/local/bin/uvx

# ----- 3. vLLM (pre-built ROCm wheel — bundles its own torch/triton/flash-attn)
# vllm 0.23.0+rocm722 requires Python 3.12; rocm/dev-ubuntu-24.04 ships 3.12 by default.
# The wheel comes from AMD's wheel server; --extra-index-url lets uv find the +rocm722 variant.
# The uv download cache is a BuildKit cache mount (not in the layer), so a rebuild
# reuses the large vLLM/torch wheels instead of re-downloading them.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        "vllm==${VLLM_VERSION}+${VLLM_ROCM_VARIANT}" \
        --extra-index-url "https://wheels.vllm.ai/rocm/${VLLM_VERSION}/${VLLM_ROCM_VARIANT}"

# ----- 4. LMCache (upstream release + AMD patches) ---------------------------
# Shallow-clone the pinned release tag directly (no history/other branches).
RUN git clone --depth 1 --branch "${LMCACHE_REF}" "${LMCACHE_GIT_URL}" /app/LMCache

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
            echo "The LMCache pin (${LMCACHE_REF}) likely drifted; rebase this patch against it." >&2; \
            exit 1; \
        fi; \
    done

# Build the LMCache wheel (with the AMD patches + HIP extension) and install THAT,
# instead of `pip install -e .`, so the wheel we publish is byte-for-byte what the
# image runs.  The wheel lands in /wheels (extracted by the `wheels` stage below).
#
# ROCM_TARGET_LST pins the arch that hipcc auto-detects for the host-side .cpp
# files (which torch compiles without an explicit --offload-arch): on a GPU-less
# build host, rocm_agent_enumerator can report a phantom arch (e.g. gfx1105) that
# this ROCm's clang rejects, breaking the compile.  GPU kernels still get the full
# multi-arch set from PYTORCH_ROCM_ARCH via torch, so a single valid arch here
# (the first of the list) is enough to satisfy the host compiles.
# MAX_JOBS / CMAKE_BUILD_PARALLEL_LEVEL parallelize the HIP extension compile
# (torch's build_ext is single-threaded otherwise -- the dominant build step).
# The pip build cache is a BuildKit cache mount, so repeated builds reuse it.
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p /wheels && \
    printf '%s\n' "${ROCM_ARCH%%;*}" > /etc/rocm_target.lst && \
    cd /app/LMCache && \
    PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        CXX=hipcc \
        BUILD_WITH_HIP=1 \
        ROCM_TARGET_LST=/etc/rocm_target.lst \
        MAX_JOBS="$(nproc)" \
        CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
        python3 -m pip wheel --no-build-isolation --no-deps -w /wheels . && \
    uv pip install --system --no-cache /wheels/lmcache-*.whl

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "matplotlib>=3.8,<4"

# Remove any NIXL wheels that upstream packages may have pulled — we build from source below.
# Also purge stray CUDA-only deps (cupy-cuda12x, cufile-python, cuda-pathfinder): if they
# remain, vLLM/LMCache can try to import non-existent CUDA .so files at runtime on ROCm.
RUN uv pip uninstall --system \
        nixl nixl-cu12 nixl-cu13 \
        cupy-cuda12x cufile-python cuda-pathfinder 2>/dev/null || true

# ----- 5. ROCm/rocm-systems hipFile (DEB + Python binding) ------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone --no-checkout --filter=blob:none \
        https://github.com/ROCm/rocm-systems.git /app/rocm-systems && \
    cd /app/rocm-systems && \
    git sparse-checkout set --cone projects/hipfile && \
    git checkout "${HIPFILE_SHA}" && \
    cd projects/hipfile && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_HIP_PLATFORM=amd -DAIS_INSTALL_TOOLS=ON \
          -DBUILD_TESTING=OFF \
          -DGPU_TARGETS="${ROCM_ARCH}" \
          -DCMAKE_HIP_ARCHITECTURES="${ROCM_ARCH}" .. && \
    cmake --build . -j"$(nproc)" && \
    cpack -G DEB && \
    dpkg -i hipfile*deb && \
    cd ../python && \
    uv pip install --system -e . \
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
        UV_SYSTEM_CERTS=1 \
        /tmp/nixl-scripts/build-nixl.sh

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
    cmake --build /app/hsa-snoop/build --parallel "$(nproc)" && \
    cmake --install /app/hsa-snoop/build --prefix /usr/local && \
    test -x /usr/local/bin/hsa-snoop && \
    { /usr/local/bin/hsa-snoop --help 2>&1 || true; } | grep -q -- '--prometheus' \
        || { echo "ERROR: hsa-snoop built without Prometheus mode (--prometheus absent)" >&2; exit 1; } && \
    echo "PASS: hsa-snoop v${HSA_SNOOP_REF} with Prometheus exporter (incl. AIS metrics) installed"

# ----- 7. Runtime environment ------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    NIXL_PLUGIN_DIR=/opt/nixl/lib/x86_64-linux-gnu/plugins \
    PYTHONPATH=/opt/nixl/lib/python3/dist-packages \
    LD_LIBRARY_PATH=/opt/nixl/lib/x86_64-linux-gnu:/opt/nixl/lib:/opt/rocnixl-ucx/lib:/opt/rocm/lib

# vLLM OpenAI-compatible server is the default entrypoint.
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

# ----- 8. Wheel export stage -------------------------------------------------
# A scratch image whose only content is the pip wheels built above (patched
# LMCache + ROCm NIXL).  Extract them WITHOUT building/loading the full image:
#   docker build --target wheels --output type=local,dest=./wheels .
# The nightly-wheels GitHub workflow uses exactly this to publish the wheels.
FROM scratch AS wheels
COPY --from=build /wheels /

# ----- 9. Runtime image (default target) -------------------------------------
# Aliases the `build` stage so the DEFAULT build target (no --target) is the full
# runtime image, exactly as before -- Makefile/compose/run-build-distribute.sh
# and the existing Docker CI all keep getting the runtime image unchanged.  Must
# be the LAST stage in this file so BuildKit treats it as the default.
FROM build AS runtime
