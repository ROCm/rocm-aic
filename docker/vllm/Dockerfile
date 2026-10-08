# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# AIC vLLM image: torch + torchvision (from aic-base) + vLLM v0.29.0 + llm-emu plugin
# + the LMCache wheel built by docker/lmcache/Dockerfile.
#
# Consumed by the `vllm` compose service.  Shares aic-base with aic-lmcache, but
# its default target needs aic-lmcache's wheels, so build aic-lmcache first.
#
# Stages:
#   vllm      torch + vLLM source build + llm-emu                   (sections 3, 3b)
#   emulate   CPU-only runtime (--target emulate)                   (section 3c)
#   runtime   default target: `vllm` + aic-lmcache's lmcache wheel  (section 3d)
#
# Build (from repo root — build context is this tree):
#   BASE_TAG=$(ROCM_ARCH=gfx950 docker/scripts/aic-image-tag.sh docker/base/Dockerfile)
#   DOCKER_BUILDKIT=1 docker buildx build \
#     --build-arg ROCM_ARCH=gfx950 \
#     --build-context base=docker-image://aic-base:${BASE_TAG} \
#     -f docker/lmcache/Dockerfile --target wheels \
#     --output type=local,dest=/tmp/aic-lmcache-wheels \
#     .
#   DOCKER_BUILDKIT=1 docker buildx build \
#     --build-arg ROCM_ARCH=gfx950 \
#     --build-context base=docker-image://aic-base:${BASE_TAG} \
#     --build-context lmcache-wheels=/tmp/aic-lmcache-wheels \
#     -f docker/vllm/Dockerfile \
#     -t aic-vllm:${BASE_TAG} \
#     .
#
# Or via make:
#   make dist-build-vllm ROCM_ARCH=gfx950
#
# Emulation-only (no GPU kernels):
#   DOCKER_BUILDKIT=1 docker buildx build \
#     --target emulate \
#     --build-arg VLLM_TARGET_DEVICE=empty \
#     --build-context base=docker-image://aic-base:${BASE_TAG} \
#     -f docker/vllm/Dockerfile \
#     -t aic-vllm:emulate .

# Fallback base stage: used when no --build-context base= is provided (e.g.
# when building this Dockerfile standalone without a pre-built aic-base image).
# When --build-context base=docker-image://aic-base:<TAG> IS supplied, BuildKit
# overrides this stage with the pre-built image and skips this FROM entirely.
ARG ROCM_VERSION=7.14.1
ARG ROCM_BASE_IMAGE=rocm/dev-ubuntu-24.04:${ROCM_VERSION}-full
# hadolint ignore=DL3006
FROM ${ROCM_BASE_IMAGE} AS base

# Empty default for the `lmcache-wheels` build context (the --target wheels
# export of docker/lmcache/Dockerfile).  Only `runtime` reads it, and fails on
# an empty one, so a build that forgot the context stops instead of shipping a
# vLLM that cannot load LMCacheMPConnector.  `emulate` never reads it.
FROM scratch AS lmcache-wheels

# =============================================================================
# vLLM stage — inherits from the final stage of docker/base/ (build_torchvision)
# which already has pytorch + torchvision installed system-wide.
# =============================================================================
FROM base AS vllm

ARG ROCM_ARCH

# vLLM: built from source at this tag.
ARG VLLM_GIT_URL=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=v0.29.0

# `rocm` compiles HIP kernels for ROCM_ARCH.  `empty` is the no-extension build
# usable only for emulation (--target emulate); do NOT ship it to a GPU node.
ARG VLLM_TARGET_DEVICE=rocm

# ----- 3. vLLM (built from source at pinned tag) ---------------------------------
RUN test -n "$ROCM_ARCH" || { echo "ERROR: ROCM_ARCH is not set"; exit 1; }

RUN git clone --depth 1 --branch "${VLLM_REF}" "${VLLM_GIT_URL}" /app/vllm

COPY patches/ /app/patches/

# Patch setup.py so _is_hip() short-circuits on VLLM_TARGET_DEVICE=rocm at
# build time, without probing torch.version.hip before it is installed.
RUN sed -i 's/def _is_hip() -> bool:/def _is_hip() -> bool:\n    if VLLM_TARGET_DEVICE == "rocm":\n        return True/' /app/vllm/setup.py && \
    grep -A2 '^def _is_hip() -> bool:$' /app/vllm/setup.py | \
        grep -Fxq '    if VLLM_TARGET_DEVICE == "rocm":'

# Apply every patch under patches/vllm/ in lexical order.
# hadolint ignore=SC3054,SC3044,SC2098
RUN cd /app/vllm && \
    shopt -s nullglob && \
    patches=(/app/patches/vllm/*.patch) && \
    if [ "${#patches[@]}" -gt 0 ]; then \
        for patch in "${patches[@]}"; do \
            if git apply "$patch"; then \
                echo "Applied: $patch"; \
            else \
                echo "ERROR: vLLM patch does not apply cleanly: $patch" >&2; \
                echo "The vLLM pin (${VLLM_REF}) likely drifted; rebase this patch against it." >&2; \
                exit 1; \
            fi; \
        done; \
    fi

# --no-build-isolation means nothing installs vLLM's [build-system].requires.
# rocm.txt lists setuptools-rust/-scm itself; common.txt does not, so the
# `empty` path adds build/rust.txt for them.
# hadolint ignore=SC2098
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    mkdir -p /wheels && \
    cd /app/vllm && \
    printf 'torch>=2.13,<2.14\n' > /tmp/torch-constraints.txt && \
    if [ "${VLLM_TARGET_DEVICE}" = "empty" ]; then \
        pip3 install -r requirements/common.txt \
            -r requirements/build/rust.txt \
            --constraint /tmp/torch-constraints.txt; \
    else \
        pip3 install -r requirements/rocm.txt \
            --constraint /tmp/torch-constraints.txt; \
    fi && \
    VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE}" \
        PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        MAX_JOBS="$(nproc)" \
        CCACHE_DIR=/root/.cache/ccache \
        VLLM_VERSION_OVERRIDE="${VLLM_REF#v}+${VLLM_TARGET_DEVICE}" \
        pip3 wheel --no-build-isolation --no-deps --no-cache-dir -w /wheels . && \
    pip3 install --no-cache-dir /wheels/vllm-*.whl

# ----- 3b. vllm_emulator (llm-emu profile-driven emulation plugin) -----------
ARG LLM_EMU_GIT_URL=https://github.com/AKafakA/llm-emu.git
ARG LLM_EMU_REF=b71c602
ENV AIC_EMULATOR_EXAMPLE_PACK=/opt/llm-emu/profiles/A40-Q8-Qwen3-8B.json \
    AIC_EMULATOR_DEFAULT_PACK=/opt/llm-emu/profiles/MI300X-Qwen3-8B.json
# hadolint ignore=SC3044
RUN git clone "${LLM_EMU_GIT_URL}" /app/llm-emu && \
    git -C /app/llm-emu checkout "${LLM_EMU_REF}" && \
    shopt -s nullglob && \
    for patch in /app/patches/llm-emu/*.patch; do \
        if git -C /app/llm-emu apply "$patch"; then \
            echo "Applied: $patch"; \
        else \
            echo "ERROR: llm-emu patch does not apply cleanly: $patch" >&2; \
            echo "The llm-emu pin (${LLM_EMU_REF}) likely drifted; rebase this patch against it." >&2; \
            exit 1; \
        fi; \
    done && \
    pip3 install --no-cache-dir --no-build-isolation --no-deps /app/llm-emu/ && \
    python3 -c "import vllm_emulator.hooks.executor_hook, vllm_emulator.oracle, \
vllm_emulator.profile.loader, vllm_emulator.profiler.trace_profiler; \
print('vllm_emulator subpackages import OK')" && \
    mkdir -p /opt/llm-emu/profiles && \
    cp /app/llm-emu/example_profiling_data/*.json /opt/llm-emu/profiles/ && \
    test -r "${AIC_EMULATOR_EXAMPLE_PACK}"

# ----- Runtime ENV + metadata -------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    SAFETENSORS_FAST_GPU=1 \
    HIP_FORCE_DEV_KERNARG=1 \
    PYTHONPATH=/opt/rocm/share/amd_smi

COPY profiles/*.json /opt/llm-emu/profiles/
RUN test -r "${AIC_EMULATOR_DEFAULT_PACK}" || \
    { echo "ERROR: default profile pack missing: ${AIC_EMULATOR_DEFAULT_PACK}" >&2; exit 1; }

ARG AIC_VERSION
ARG ROCM_BASE_IMAGE
ARG ROCM_VERSION
RUN test -n "$AIC_VERSION" || { echo "ERROR: AIC_VERSION is not set (read it from VERSION)"; exit 1; }
LABEL org.opencontainers.image.title="aic-vllm" \
      org.opencontainers.image.description="AMD Infinity Context: ROCm vLLM inference service" \
      org.opencontainers.image.version="${AIC_VERSION}" \
      org.opencontainers.image.base.name="${ROCM_BASE_IMAGE}" \
      ai.amd.aic.version="${AIC_VERSION}" \
      ai.amd.aic.rocm="${ROCM_VERSION}" \
      ai.amd.aic.vllm="${VLLM_REF}" \
      ai.amd.aic.llm-emu="${LLM_EMU_REF}"

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

# =============================================================================
# ----- 3c. emulate — emulation-only runtime image (--target emulate) ---------
# =============================================================================
FROM vllm AS emulate
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    SAFETENSORS_FAST_GPU=1 \
    PYTHONPATH=/opt/rocm/share/amd_smi
COPY profiles/*.json /opt/llm-emu/profiles/
RUN test -r "${AIC_EMULATOR_DEFAULT_PACK}" || \
    { echo "ERROR: default profile pack missing: ${AIC_EMULATOR_DEFAULT_PACK}" >&2; exit 1; }
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

# ----- 3d. runtime: default target -------------------------------------------
# vLLM's LMCacheMPConnector imports lmcache in the vLLM process, so this image
# installs the same lmcache wheel aic-lmcache runs: one build, with its HIP
# extension, on both ends of the MP protocol.  Installed after vLLM, as the
# pre-split image did.
FROM vllm AS runtime
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN --mount=type=bind,from=lmcache-wheels,target=/aic-wheels \
    shopt -s nullglob && \
    lmc=(/aic-wheels/lmcache-*.whl) && \
    if [ "${#lmc[@]}" -ne 1 ]; then \
        echo "ERROR: expected one lmcache wheel in the lmcache-wheels build context, found:" >&2; \
        ls -l /aic-wheels >&2; \
        echo "Pass --build-context lmcache-wheels=DIR, exported by: docker buildx build -f docker/lmcache/Dockerfile --target wheels --output type=local,dest=DIR ." >&2; \
        exit 1; \
    fi && \
    uv pip install --system --no-cache "${lmc[@]}" && \
    { uv pip uninstall --system \
        nixl nixl-cu12 nixl-cu13 \
        cupy-cuda12x cufile-python cuda-pathfinder 2>/dev/null || true; } && \
    python3 -c 'import lmcache.integration.vllm.vllm_multi_process_adapter, vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector; print("LMCacheMPConnector imports OK")'

# Cross-container HIP IPC with the lmcache server; must match aic-lmcache's ENV.
ENV HSA_ENABLE_IPC_MODE_LEGACY=1
