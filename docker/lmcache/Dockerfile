# check=skip=InvalidDefaultArgInFrom
# ROCM_VERSION deliberately carries no default here -- docker/base/Dockerfile
# is its single source of truth and the Makefile forwards it to every build.
# That leaves the fallback ROCM_BASE_IMAGE default unresolvable at lint time,
# which is harmless: the `base` stage is always replaced by the base
# build-context supplied by `make build`.
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# AIC LMCache image: torch + torchvision + NIXL (from aic-base) + LMCache v0.5.5
# (patched).
#
# AITER and FlashAttention live in aic-vllm — LMCache is a KV-cache
# storage/transfer service and runs no inference kernels itself.
# NIXL is inherited from aic-base (the shared transport layer).
# hsa-snoop is a separate sidecar image (docker/hsa-snoop/Dockerfile) that
# attaches to this container's PID namespace at runtime.
#
# Consumed by the `lmcache` and `lmcache-coordinator` compose services.  Built
# independently from aic-vllm (both share aic-base) so the two images can be
# built in parallel after the base image is available.
#
#   lmcache — standalone lmcache-server (MP mode)
#               L1:  GPU/CPU DRAM (--l1-size-gb)  OR  hipFile NVMe slab (GDS L1 mode)
#               L2a: nixl_store AIS_MT → local NVMe (hipFile P2PDMA)
#               L2b: nixl_store POSIX  → NFS-over-RDMA
#
# Stages:
#   build     LMCache + runtime ENV                                  (sections 5-8)
#   wheels    scratch image holding just /wheels                             (section 9)
#   runtime   default target, aliases `build`
#
# Build (from repo root — build context is this tree):
#   BASE_TAG=$(ROCM_ARCH=gfx950 docker/scripts/aic-image-tag.sh docker/base/Dockerfile)
#   DOCKER_BUILDKIT=1 docker buildx build \
#     --build-arg ROCM_ARCH=gfx950 \
#     --build-context base=docker-image://aic-base:${BASE_TAG} \
#     -f docker/lmcache/Dockerfile \
#     -t aic-lmcache:${BASE_TAG} \
#     .
#
# Or via make:
#   make dist-build-lmcache ROCM_ARCH=gfx950

# =============================================================================
# Fallback base stage: used when no --build-context base= is provided (e.g.
# when building this Dockerfile standalone without a pre-built aic-base image).
# When --build-context base=docker-image://aic-base:<TAG> IS supplied, BuildKit
# overrides this stage with the pre-built image and skips this FROM entirely.
# Single source of truth: ARG ROCM_VERSION in docker/base/Dockerfile.
# The Makefile reads it from there and forwards it to every build, so this
# carries no default and must not be given one -- a second literal here is
# exactly the drift this removes.
ARG ROCM_VERSION
ARG ROCM_BASE_IMAGE=rocm/dev-ubuntu-24.04:${ROCM_VERSION}-full
# hadolint ignore=DL3006
FROM ${ROCM_BASE_IMAGE} AS base

# LMCache: inherits torch + torchvision + NIXL from aic-base.
# =============================================================================
FROM base AS build

ARG ROCM_ARCH

# Parallel compile jobs for C++/HIP source builds (LMCache HIP extension).
ARG BUILD_JOBS=

# LMCache: upstream repo, pinned to the v0.5.5 release tag.
# Optional open-PR heads from patches/lmcache.pull-requests are cherry-picked
# first; AMD patches in patches/lmcache/ apply on top of that tree.
ARG LMCACHE_GIT_URL=https://github.com/LMCache/LMCache.git
ARG LMCACHE_REF=v0.5.5

# NIXL_REF/NIXL_GIT_URL carried for the image label and CI patch checks —
# NIXL itself is built and installed in aic-base.
ARG NIXL_GIT_URL=https://github.com/ai-dynamo/nixl.git
ARG NIXL_REF=v1.4.1

# AITER_REF/FLASH_ATTN_REF carried for the README version-sync check —
# AITER and FlashAttention are built in aic-vllm, not here.
ARG AITER_REF=v0.1.22.post1
ARG FLASH_ATTN_REF=v2.8.3.post1

# ----- 5. LMCache (upstream release + AMD patches) ---------------------------
COPY patches/ /app/patches/
# CuPy-backed GPU allocator module — shipped separately so the patch only
# modifies existing LMCache files (avoids new-file patch format issues).
COPY docker/lmcache/paged_cupy_memory_allocator.py \
     /app/paged_cupy_memory_allocator.py
COPY docker/scripts/apply-github-prs.sh /usr/local/bin/apply-github-prs.sh

RUN git clone --depth 1 --branch "${LMCACHE_REF}" "${LMCACHE_GIT_URL}" /app/LMCache

# Cherry-pick optional LMCache PR heads listed in patches/lmcache.pull-requests.
# Line format:
#   <pr-number>
#   <remote-name> <pr-number>
#   <remote-name>=<remote-url> <pr-number>
# Blank lines and # comments are ignored. Inline remote URLs must point at
# GitHub repositories that expose refs/pull/<pr-number>/head. Stacked PRs are
# allowed: dependency commits not already present are cherry-picked first.
RUN chmod +x /usr/local/bin/apply-github-prs.sh && \
    /usr/local/bin/apply-github-prs.sh \
        /app/LMCache \
        /app/patches/lmcache.pull-requests \
        "${LMCACHE_REF}"

# Apply every patch under patches/lmcache/ in lexical order after optional PRs.
# hadolint ignore=SC3054,SC3044
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

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir grpcio-tools

# Install the CuPy-backed paged allocator into LMCache's package tree so
# the patch's import of paged_cupy_memory_allocator works.
RUN cp /app/paged_cupy_memory_allocator.py \
       /app/LMCache/lmcache/v1/memory_allocators/paged_cupy_memory_allocator.py

# CuPy (ROCm variant): provides hipMalloc without pulling in the ROCm compute
# stack. Used by the NIXL GPU buffer allocator (LMCACHE_NIXL_USE_CUPY=1).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir cupy-rocm-7-0

RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p /wheels && \
    printf '%s\n' "${ROCM_ARCH%%,*}" > /etc/rocm_target.lst && \
    cd /app/LMCache && \
    PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        CXX=hipcc \
        BUILD_WITH_HIP=1 \
        ROCM_TARGET_LST=/etc/rocm_target.lst \
        MAX_JOBS="${BUILD_JOBS:-$(nproc)}" \
        CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS:-$(nproc)}" \
        python3 -m pip wheel --no-build-isolation --no-deps -w /wheels . && \
    uv pip install --system --no-cache /wheels/lmcache-*.whl

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "matplotlib>=3.8,<4" "openai>=1.0,<2"


# ----- 6. hipFile — provided by the ROCm 7.14 base image (GA since 7.14) ------
# No source build needed.

RUN rm -rf /app/patches /app/LMCache /wheels && \
    find /usr/lib/python3 /usr/local/lib/python3 -name '__pycache__' -type d \
         -exec rm -rf {} + 2>/dev/null; true

# ----- 8. Runtime environment ------------------------------------------------
# NIXL_PLUGIN_DIR, PYTHONPATH (incl. NIXL), LD_LIBRARY_PATH, and
# HSA_ENABLE_IPC_MODE_LEGACY are set in aic-base and inherited here.
# LMCACHE_NIXL_USE_CUPY=1: activate the CuPy GPU buffer allocator so
# librocblas/libMIOpen/etc. are never dlopened in the lmcache process.
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    SAFETENSORS_FAST_GPU=1 \
    HIP_FORCE_DEV_KERNARG=1 \
    LMCACHE_NIXL_USE_CUPY=1

ARG AIC_VERSION
ARG ROCM_BASE_IMAGE
ARG ROCM_VERSION
RUN test -n "$AIC_VERSION" || { echo "ERROR: AIC_VERSION is not set (read it from VERSION)"; exit 1; }
LABEL org.opencontainers.image.title="aic-lmcache" \
      org.opencontainers.image.description="AMD Infinity Context: ROCm LMCache tiered-cache service" \
      org.opencontainers.image.version="${AIC_VERSION}" \
      org.opencontainers.image.base.name="${ROCM_BASE_IMAGE}" \
      ai.amd.aic.version="${AIC_VERSION}" \
      ai.amd.aic.rocm="${ROCM_VERSION}" \
      ai.amd.aic.lmcache="${LMCACHE_REF}" \
      ai.amd.aic.nixl="${NIXL_REF}" \
      ai.amd.aic.hipfile="rocm-${ROCM_VERSION}"

ENTRYPOINT ["python3", "-m", "lmcache.server"]

# ----- 9. Wheel export stage -------------------------------------------------
FROM scratch AS wheels
COPY --from=build /wheels /

# ----- 10. Runtime image (default target) ------------------------------------
FROM build AS runtime
