# check=skip=InvalidDefaultArgInFrom
# ROCM_BASE_IMAGE is a build arg with no default here (docker/docker-bake.hcl
# owns it), so the FROM lines are blank at lint time.
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# AIC LMCache image: torch + torchvision + NIXL (from aic-base) + LMCache
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
# Build args: docker/docker-bake.hcl is the only owner of every pin, URL and
# base image, so no ARG here has a default and each consuming stage stops when
# one is unset.  Build through bake (from repo root), which also wires in
# aic-base and its wheels:
#   ROCM_ARCH=gfx950 docker buildx bake -f docker/docker-bake.hcl lmcache --load
#
# or hand bake's resolved args to a raw build against an aic-base built earlier
# (base-wheels is the `--target wheels` export of docker/base/Dockerfile):
#   mapfile -t args < <(ROCM_ARCH=gfx950 docker/scripts/aic-bake-env.sh --build-args lmcache)
#   docker buildx build "${args[@]}" \
#     --build-context base=docker-image://aic-base:<tag> \
#     --build-context base-wheels=/tmp/aic-base-wheels \
#     -f docker/lmcache/Dockerfile -t aic-lmcache:dev .
#
# Or via make:
#   make build ROCM_ARCH=gfx950            (local)
#   make dist-build AIC_ROCM_ARCH=gfx950   (Slurm)

# =============================================================================
# Fallback base stage: used when no --build-context base= is provided.  When
# --build-context base=docker-image://aic-base:<TAG> IS supplied, BuildKit
# overrides this stage with the pre-built image and skips this FROM entirely.
# build_lmcache_wheel always starts from ROCM_BASE_IMAGE.  Unset, the build
# stops at the FROM: base name (${ROCM_BASE_IMAGE}) should not be blank
ARG ROCM_BASE_IMAGE
# hadolint ignore=DL3006
FROM ${ROCM_BASE_IMAGE} AS base

FROM scratch AS base-wheels

# =============================================================================
# build_lmcache_wheel — compile the LMCache HIP extension with the FULL ROCm
# toolchain (ROCM_BASE_IMAGE = -full, not the slim aic-base runtime).
# This stage is NOT replaced by --build-context base=oci-layout://...
# because only the stage named "base" is overridden by that flag.
# torch is COPYed from aic-base (which has it) into this full-ROCm stage.
# The compiled wheel is COPYed into the runtime build stage below.
# =============================================================================
# hadolint ignore=DL3006
FROM ${ROCM_BASE_IMAGE} AS build_lmcache_wheel

ARG ROCM_ARCH
ARG BUILD_JOBS
ARG LMCACHE_GIT_URL
ARG LMCACHE_REF

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN : "${LMCACHE_GIT_URL:?unset: build through docker/docker-bake.hcl}" \
      "${LMCACHE_REF:?unset: build through docker/docker-bake.hcl}"

# Install git + minimal build tools (the -full ROCm image does not include them).
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libnuma1 \
        python3-dev \
        python3-pip

RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

RUN --mount=type=cache,target=/root/.cache/pip \
      pip3 install --break-system-packages build

COPY patches/lmcache/ /app/patches/lmcache/
COPY docker/scripts/apply-github-prs.sh /usr/local/bin/apply-github-prs.sh

RUN git clone --depth 1 --branch "${LMCACHE_REF}" "${LMCACHE_GIT_URL}" /app/LMCache

WORKDIR /app/LMCache

RUN --mount=type=bind,source=patches/lmcache,target=/app/patches/lmcache/ \
    --mount=type=bind,source=docker/scripts/apply-github-prs.sh,target=/tmp/apply-github-prs.sh \
    bash /tmp/apply-github-prs.sh \
        /app/LMCache \
        /app/patches/lmcache.pull-requests \
        "${LMCACHE_REF}"

# hadolint ignore=SC3054,SC3044
RUN shopt -s nullglob && \
    patches=(/app/patches/lmcache/*.patch) && \
    if [ "${#patches[@]}" -eq 0 ]; then \
        echo "ERROR: no patches found in /app/patches/lmcache/" >&2; exit 1; \
    fi && \
    for patch in "${patches[@]}"; do \
        if git apply "$patch"; then \
            echo "Applied: $patch"; \
        else \
            echo "ERROR: LMCache patch does not apply cleanly: $patch" >&2; \
            exit 1; \
        fi; \
    done

WORKDIR /

# Apply the ROCm event-IPC edit to the LMCache source before the wheel is
# built: event IPC is replaced by a producer-side sync (hipIpcOpenEventHandle
# is broken on gfx1201 — see tests/ipc/).  docker/vllm/Dockerfile runs the same
# script on the PyPI LMCache client of its `emulate` stage.  The script exits
# non-zero if an anchor is missing.
COPY docker/lmcache/rocm_event_ipc_patch.py /tmp/rocm_event_ipc_patch.py
RUN AIC_LMCACHE_SRC=/app/LMCache python3 /tmp/rocm_event_ipc_patch.py && \
    rm /tmp/rocm_event_ipc_patch.py

# Build the LMCache wheel with full ROCm headers available.  LMCache pins an
# exact torch in [build-system].requires; relax it so the constraint to the
# base image's torch wheel can be satisfied.
WORKDIR /app/LMCache
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,from=base-wheels,source=/,target=/base-wheels \
    mkdir -p /wheels && \
    printf '%s\n' "${ROCM_ARCH%%;*}" > /etc/rocm_target.lst && \
    (shopt -s nullglob; for f in /base-wheels/torch-*.whl; do \
        printf 'torch @ file://%s\n' "$f" > /tmp/build-constraints.txt; \
    done) && \
    sed -i -E 's/"torch==[0-9.]+"/"torch"/' pyproject.toml && grep -q '"torch",' pyproject.toml && \
    PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        CXX=hipcc \
        BUILD_WITH_HIP=1 \
        ROCM_TARGET_LST=/etc/rocm_target.lst \
        MAX_JOBS="${BUILD_JOBS:-$(nproc)}" \
        CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS:-$(nproc)}" \
        python3 -m build --wheel \
          "$(test -f /tmp/build-constraints.txt && echo --dependency-constraints-txt=/tmp/build-constraints.txt)" \
          --outdir /wheels

WORKDIR /

# =============================================================================
# Runtime build stage (FROM aic-base via --build-context base=oci-layout://...)
# =============================================================================
# LMCache: inherits torch + torchvision + NIXL from aic-base.
# =============================================================================
FROM base AS build

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ROCM_ARCH
ARG BUILD_JOBS
ARG LMCACHE_GIT_URL
ARG LMCACHE_REF
ARG CUPY_VERSION
# For the image label; NIXL itself comes from aic-base.
ARG NIXL_REF
RUN : "${LMCACHE_GIT_URL:?unset: build through docker/docker-bake.hcl}" \
      "${LMCACHE_REF:?unset: build through docker/docker-bake.hcl}" \
      "${CUPY_VERSION:?unset: build through docker/docker-bake.hcl}" \
      "${NIXL_REF:?unset: build through docker/docker-bake.hcl}"

# LMCache (the wheel from build_lmcache_wheel) and CuPy for ROCm, which its MP
# server imports, in one RUN so that nothing removed below reaches a layer.
# pip3 comes from aic-base.  No grpcio-tools: the wheel ships its generated
# gRPC stubs, and an unpinned one wants a newer grpcio and protobuf than
# LMCache allows.
#
# - The constraint file pins what aic-base installed (torch, numpy,
#   setuptools, ...), so a requirement that would replace one of them stops
#   the build instead of leaving a second copy in this layer.
# - aic-base's packaging is Debian-owned (no RECORD), so pip cannot upgrade it
#   in place and setuptools_scm needs a newer one; a pip-owned copy goes in
#   first, as docker/vllm/Dockerfile does.
# - requirements/cli.txt (openai, matplotlib) is not installed: `lmcache
#   server` no longer imports the bench subcommands' openai, which the gate at
#   the end re-checks on every LMCACHE_REF bump.  `lmcache bench` and the
#   cache-simulator plots do not work in this image.
# - The uninstalled packages are wheel requirements that neither the MP server
#   nor the coordinator imports.
# - DEVICE_TYPE=cpu keeps the import gates off GPU detection.
# - No bytecode is shipped (as before, when a later RUN deleted it), so a
#   container whose user cannot write to dist-packages compiles on every
#   start instead of only the first.  The closing find removes what the gates
#   wrote anyway: a child process that drops the environment, numba's cache
#   directories.
# - llvm-strip is the ROCm toolchain's, from aic-base.  --strip-unneeded keeps
#   .dynsym and cupy's .hip_fatbin device code.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,from=build_lmcache_wheel,source=/wheels/,target=/wheels/ \
    export PYTHONDONTWRITEBYTECODE=1 DEVICE_TYPE=cpu && \
    DP="$(echo /usr/local/lib/python3.*/dist-packages)" && \
    pip3 install --break-system-packages --no-compile --ignore-installed packaging && \
    pip3 list --path "${DP}" --format=freeze > /tmp/base-pins.txt && \
    pip3 install --break-system-packages --no-compile \
      -c /tmp/base-pins.txt \
      /wheels/lmcache-*.whl \
      "cupy-rocm-7-0==${CUPY_VERSION}" && \
    rm /tmp/base-pins.txt && \
    pip3 uninstall -y --break-system-packages \
      transformers tokenizers hf-xet regex safetensors redis \
      pytest pluggy iniconfig && \
    find "${DP}/cupy" "${DP}/cupy_backends" "${DP}/cupyx" "${DP}/llvmlite" \
         "${DP}/lmcache" "${DP}/nvtx" "${DP}/numba" "${DP}/cryptography" \
         -type f -name '*.so*' -exec llvm-strip --strip-unneeded {} + && \
    llvm-readelf -S "${DP}"/cupy/cuda/cub.*.so | grep -c '\.hip_fatbin' > /dev/null && \
    lmcache server --help > /dev/null && \
    python3 -c 'import lmcache.cuda_ops, lmcache.lmcache_fs, lmcache.lmcache_native, lmcache.lmcache_redis; import lmcache.v1.mp_coordinator.app' && \
    find "${DP}" -name __pycache__ -type d -exec rm -rf {} +

# ----- 6. hipFile — provided by the ROCm base image (GA since ROCm 7.14) ------
# No source build needed.

# ----- 8. Runtime environment ------------------------------------------------
# NIXL_PLUGIN_DIR, PYTHONPATH (incl. NIXL), and LD_LIBRARY_PATH are set in
# aic-base and inherited here.  HSA_ENABLE_IPC_MODE_LEGACY is intentionally NOT
# set: aic-vllm leaves it unset too, and the two must agree.  The default
# (non-legacy) mode crosses containers that share a network namespace; see
# docker-compose.yml.
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    SAFETENSORS_FAST_GPU=1 \
    HIP_FORCE_DEV_KERNARG=1

ARG AIC_VERSION
ARG ROCM_BASE_IMAGE
ARG ROCM_VERSION
RUN : "${AIC_VERSION:?unset: build through docker/docker-bake.hcl}" \
      "${ROCM_VERSION:?unset: build through docker/docker-bake.hcl}" \
      "${ROCM_BASE_IMAGE:?unset: build through docker/docker-bake.hcl}"
LABEL org.opencontainers.image.title="aic-lmcache" \
      org.opencontainers.image.description="AMD Infinity Context: ROCm LMCache tiered-cache service" \
      org.opencontainers.image.version="${AIC_VERSION}" \
      org.opencontainers.image.base.name="${ROCM_BASE_IMAGE}" \
      ai.amd.aic.version="${AIC_VERSION}" \
      ai.amd.aic.rocm="${ROCM_VERSION}" \
      ai.amd.aic.lmcache="${LMCACHE_REF}" \
      ai.amd.aic.nixl="${NIXL_REF}" \
      ai.amd.aic.hipfile="rocm-${ROCM_VERSION}"

ENTRYPOINT ["lmcache", "server"]

# ----- 9. Wheel export stage -------------------------------------------------
# Copy from build_lmcache_wheel (not build): the build stage only bind-mounts
# /wheels for the install, to keep the runtime image lean.
FROM scratch AS wheels
COPY --from=build_lmcache_wheel /wheels /

# ----- 10. Runtime image (default target) ------------------------------------
FROM build AS runtime

# Asserts that the ZMQ port vLLM connects to is bound and that the HTTP API
# answers /metrics with a 200.  Services that run this image with another
# entrypoint (the coordinator) override the check in compose.
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=18 \
    CMD ["python3", "-c", "import os, socket, urllib.request as u; socket.create_connection(('127.0.0.1', int(os.getenv('LMCACHE_PORT', '6555'))), 2).close(); u.build_opener(u.ProxyHandler({})).open('http://127.0.0.1:8080/metrics', timeout=3).close()"]
