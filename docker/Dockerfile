# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# AIC base image: Ubuntu 24.04 + ROCm 7.14.1 + PyTorch (source) + torchvision (source).
#
# This is the shared foundation consumed by docker/vllm/Dockerfile and
# docker/lmcache/Dockerfile via --build-context base=docker-image://aic-base:TAG.
# Both images bind-mount the wheel artifacts from the build_pytorch and
# build_torchvision stages at build time (--mount=type=bind,from=build_pytorch,...).
#
# Stages:
#   pip-wheels        scratch stub for --build-context pip-wheels override
#   base              ROCm base + system deps + uv                    (sections 0-2)
#   build_pytorch     PyTorch 2.13 built from source                  (section 2b)
#   build_torchvision torchvision built from source                   (section 2c)
#
# Build (from repo root):
#   DOCKER_BUILDKIT=1 docker build \
#     --build-arg ROCM_ARCH=gfx950 \
#     -f docker/base/Dockerfile \
#     -t aic-base:$(docker/scripts/aic-image-tag.sh) \
#     .
#
# Or via make:
#   make dist-build-base ROCM_ARCH=gfx950

ARG ROCM_VERSION=7.14.1
# ROCm 7.14 publishes the hipFile-bearing variant as `-full` (not `-complete`).
ARG ROCM_BASE_IMAGE=rocm/dev-ubuntu-24.04:${ROCM_VERSION}-full

# Empty default for the `pip-wheels` build context.  A caller that passes
# --build-context pip-wheels=<dir> overrides this stage exactly as before.
FROM scratch AS pip-wheels

# ROCM_BASE_IMAGE has a tagged default but remains overridable by callers.
# hadolint ignore=DL3006
FROM ${ROCM_BASE_IMAGE} AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ROCM_VERSION
ARG ROCM_BASE_IMAGE
ARG ROCM_ARCH

# ----- 0. Corporate CA (optional, Zscaler/AMD) --------------------------------
# Pass a CA cert at build time via BuildKit secret to reach HTTPS through a
# corporate proxy.  Omit entirely for builds outside the proxy.
#
#   docker build --secret id=tls_cert,src=/path/to/corp-ca.crt ...
#
RUN --mount=type=secret,id=tls_cert,required=false \
    if [ -f /run/secrets/tls_cert ] && [ -s /run/secrets/tls_cert ]; then \
        cp /run/secrets/tls_cert /usr/local/share/ca-certificates/corp-ca.crt && \
        update-ca-certificates; \
    fi

# ----- 1. System dependencies ------------------------------------------------
# hadolint ignore=SC2015
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libboost-dev \
        libdrm-dev \
        libfmt-dev \
        libibverbs-dev \
        ibverbs-providers \
        libmount-dev \
        libmsgpack-dev \
        libopenmpi3t64 \
        libsqlite3-dev \
        ninja-build \
        nvme-cli \
        pkg-config \
        python3-dev \
        bpftrace \
    && update-ca-certificates && \
    # libdrm-dev pulls in old libhsa-runtime64-1/libhsakmt1 (ROCm 5.7 era) which
    # break hipIpcGetMemHandle.  Remove them so the ROCm 7.14 runtime is the only one.
    DEBIAN_FRONTEND=noninteractive apt-get remove -y --purge \
        libhsa-runtime64-1 libhsakmt1 2>/dev/null || true

RUN git config --global advice.detachedHead false

RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

ENV PIP_ROOT_USER_ACTION=ignore \
    UV_LINK_MODE=copy \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# ----- 2. uv (fast Python package manager) -----------------------------------
ARG UV_VERSION=0.12.15
RUN curl -fsSL "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv && \
    ln -s /root/.local/bin/uvx /usr/local/bin/uvx

# =============================================================================
# PyTorch source build — avoids ROCm 7.2 wheel artifacts on a ROCm 7.14 host.
# =============================================================================
FROM base AS build_pytorch

ARG ROCM_ARCH
ARG PYTORCH_REPO=https://github.com/ROCm/pytorch.git
ARG PYTORCH_BRANCH=release/2.13

# torch 2.13 emits float->__bf16 libcalls absent from gcc-11; build with gcc-13.
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc-13 g++-13 pkg-config liblzma-dev libnuma-dev python3.12-venv && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-13

RUN git clone --branch "${PYTORCH_BRANCH}" "${PYTORCH_REPO}" /app/pytorch

RUN --mount=type=cache,target=/root/.cache/uv \
    cd /app/pytorch && \
    uv pip install --system -r requirements.txt && \
    git submodule update --init --recursive

ENV PYTORCH_ROCM_ARCH=${ROCM_ARCH}

# hadolint ignore=SC2086
RUN --mount=type=cache,target=/root/.cache/ccache \
    cd /app/pytorch && \
    python3 tools/amd_build/build_amd.py && \
    CCACHE_DIR=/root/.cache/ccache \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    CMAKE_PREFIX_PATH=$(python3 -c 'import sys; print(sys.prefix)') \
        python3 setup.py bdist_wheel --dist-dir=/app/pytorch/dist

# =============================================================================
# torchvision source build — must compile against our source-built PyTorch.
# Pin TORCHVISION_BRANCH to the release matching PyTorch 2.13 (v0.28.x series).
# NOTE: v0.29.0 requires torch::stable::permute which is not present in
# ROCm/pytorch@release/2.13; stay on v0.28.0 until the ROCm fork advances.
# =============================================================================
FROM build_pytorch AS build_torchvision

ARG ROCM_ARCH
ARG TORCHVISION_REPO=https://github.com/pytorch/vision.git
ARG TORCHVISION_BRANCH=v0.28.0

RUN git clone --depth 1 --branch "${TORCHVISION_BRANCH}" "${TORCHVISION_REPO}" /app/vision

# hadolint ignore=SC2086,DL3008
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libnuma-dev && \
    pip3 install setuptools wheel packaging /app/pytorch/dist/*.whl && \
    cd /app/vision && \
    USE_ROCM=1 USE_CUDA=0 \
    PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
    CCACHE_DIR=/root/.cache/ccache \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
        python3 setup.py bdist_wheel --dist-dir=/app/vision/dist

# =============================================================================
# runtime — default target: pytorch + torchvision installed system-wide.
# Downstream Dockerfiles (docker/vllm, docker/lmcache) use this stage via
# --build-context base=docker/base so they get a fully working torch env
# without bind-mounting intermediate wheel directories.
# =============================================================================
FROM build_torchvision AS base_runtime

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir /app/vision/dist/*.whl

# =============================================================================
# vLLM stage — inherits from the final stage of docker/base/ (build_torchvision)
# which already has pytorch + torchvision installed system-wide.
# =============================================================================
FROM base_runtime AS vllm

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

# hadolint ignore=SC2098
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    mkdir -p /wheels && \
    cd /app/vllm && \
    printf 'torch>=2.13,<2.14\n' > /tmp/torch-constraints.txt && \
    pip3 install -r requirements/rocm.txt \
        --constraint /tmp/torch-constraints.txt && \
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
# Full LMCache stack — AITER, FlashAttention, LMCache, NIXL, hsa-snoop.
# Inherits from the final stage of docker/base/ (build_torchvision), which
# already has pytorch + torchvision installed system-wide.
# =============================================================================
FROM vllm AS build

ARG ROCM_ARCH

# Parallel compile jobs for C++/HIP source builds (LMCache, hsa-snoop).
ARG BUILD_JOBS=

# ROCm attention backends — validated pair, update together after revalidation.
ARG AITER_GIT_URL=https://github.com/ROCm/aiter.git
ARG AITER_REF=v0.1.22.post1
ARG FLASH_ATTN_GIT_URL=https://github.com/Dao-AILab/flash-attention.git
ARG FLASH_ATTN_REF=v2.8.3.post1

# ai-dynamo/nixl release tag + nixl-rocm-ais-mt.patch (ROCm/HIP + AIS_MT).
ARG NIXL_GIT_URL=https://github.com/ai-dynamo/nixl.git
ARG NIXL_REF=v1.4.1

# hsa-snoop: HSA AQL queue/dispatch snooper + Prometheus exporter (includes AIS metrics).
ARG HSA_SNOOP_GIT_URL=https://github.com/sbates130272/hsa-snoop.git
ARG HSA_SNOOP_REF=v1.1.1

# LMCache: upstream repo, pinned to the v0.5.5 release tag.
# All AMD patches in patches/lmcache/ apply cleanly against this tag.
ARG LMCACHE_GIT_URL=https://github.com/LMCache/LMCache.git
ARG LMCACHE_REF=v0.5.5

# Optional: enable UCX fast-path for NIXL (default off).
ARG AIC_UCX_FAST=

# ----- 4. ROCm attention backends (AITER + FlashAttention) ------------------
# hadolint ignore=SC2039
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    mkdir -p /wheels && \
    pip3 install --no-cache-dir \
        "pandas==3.0.3" \
        "psutil==7.2.2" \
        "pyyaml==6.0.3" \
        "einops==0.8.2" \
        "pybind11==3.0.4" \
        "ninja==1.13.0" \
        "flydsl==0.2.4" && \
    git clone --depth 1 --branch "${AITER_REF}" "${AITER_GIT_URL}" /app/aiter && \
    cd /app/aiter && \
    git submodule update --init --recursive && \
    PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        MAX_JOBS="${BUILD_JOBS:-$(nproc)}" \
        CCACHE_DIR=/root/.cache/ccache \
        python3 setup.py bdist_wheel --dist-dir=/wheels && \
    pip3 install --no-cache-dir --no-deps /wheels/amd_aiter-*.whl && \
    python3 -c 'import importlib.util, pathlib; spec = importlib.util.find_spec("aiter"); assert spec and spec.submodule_search_locations; print("AITER installed:", next(iter(spec.submodule_search_locations)))'

# FlashAttention: CDNA-only (gfx9xx — MI100/MI200/MI300/MI355X).
# hadolint ignore=SC2039
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    _cdna_only() { \
        _a=""; for _a in $(echo "$1" | tr ';' '\n'); do \
            [ "${_a#gfx9}" != "${_a}" ] || return 1; done; return 0; }; \
    if _cdna_only "${ROCM_ARCH}"; then \
        git clone --recurse-submodules "${FLASH_ATTN_GIT_URL}" /app/flash-attention && \
        cd /app/flash-attention && \
        git checkout "${FLASH_ATTN_REF}" && \
        git submodule update --init --recursive && \
        GPU_ARCHS="${ROCM_ARCH}" \
            PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
            MAX_JOBS="${BUILD_JOBS:-$(nproc)}" \
            CCACHE_DIR=/root/.cache/ccache \
            python3 setup.py bdist_wheel --dist-dir=/wheels && \
        pip3 install --no-cache-dir --no-deps /wheels/flash_attn-*.whl && \
        python3 -c 'import flash_attn; print("FlashAttention installed:", flash_attn.__file__)' && \
        { \
            echo "AITER_REF=${AITER_REF}"; \
            echo "FLASH_ATTN_REF=${FLASH_ATTN_REF}"; \
            python3 -c 'import importlib.util; spec = importlib.util.find_spec("aiter"); assert spec and spec.submodule_search_locations; print("AITER_PACKAGE=" + next(iter(spec.submodule_search_locations)))'; \
            python3 -c 'import flash_attn; print("FLASH_ATTN_PACKAGE=" + flash_attn.__file__)'; \
        } > /etc/rocm-aic-attention-backends.env; \
    else \
        echo "Skipping FlashAttention: ${ROCM_ARCH} contains non-CDNA arch(es); using Triton attention" && \
        { \
            echo "AITER_REF=${AITER_REF}"; \
            echo "FLASH_ATTN_REF="; \
            python3 -c 'import importlib.util; spec = importlib.util.find_spec("aiter"); assert spec and spec.submodule_search_locations; print("AITER_PACKAGE=" + next(iter(spec.submodule_search_locations)))'; \
        } > /etc/rocm-aic-attention-backends.env; \
    fi

# ----- 5. LMCache (upstream release + AMD patches) ---------------------------
COPY patches/ /app/patches/

RUN git clone --depth 1 --branch "${LMCACHE_REF}" "${LMCACHE_GIT_URL}" /app/LMCache

# Apply every patch under patches/lmcache/ in lexical order.
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

RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p /wheels && \
    printf '%s\n' "${ROCM_ARCH%%;*}" > /etc/rocm_target.lst && \
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
    uv pip install --system "matplotlib>=3.8,<4"

# Remove any NIXL wheels that upstream packages may have pulled — we build from source below.
# Also purge stray CUDA-only deps (cupy-cuda12x, cufile-python, cuda-pathfinder).
RUN uv pip uninstall --system \
        nixl nixl-cu12 nixl-cu13 \
        cupy-cuda12x cufile-python cuda-pathfinder 2>/dev/null || true

# Remove the standalone `triton_kernels` wheel so vLLM uses the copy it vendors.
#
# The ROCm base image ships `triton_kernels 1.0.0+amd.rocm7.2.0`, which is the
# post-3.8 API generation: it exposes `triton_kernels.matmul`, having renamed
# away `matmul_ogs`.  vLLM ${VLLM_REF} still imports `triton_kernels.matmul_ogs`
# (fused_moe/config.py) and vendors a matching legacy copy at
# vllm/third_party/triton_kernels/ for exactly this reason.  But its probe
# prefers whatever is installed top-level:
#
#   def has_triton_kernels() -> bool:
#       is_available = _has_module("triton_kernels") or _has_module(
#           "vllm.third_party.triton_kernels")
#
# so the AMD wheel shadows the vendored copy and every MXFP4 model start logs
# "Failed to import Triton kernels ... No module named
# 'triton_kernels.matmul_ogs'", leaving vLLM on a fallback MoE path instead of
# the one it ships and tests against.
#
# NOTE: this does NOT fix gpt-oss-120b on gfx950.  That model still aborts
# EngineCore during CUDA-graph capture with an LLVM assertion inside Triton's
# JIT (llvm/ADT/Sequence.h:275 `Begin <= End`), with this removal applied and
# the matmul_ogs import succeeding -- so the assertion is an independent Triton
# codegen bug, not a consequence of the shadowing.
#
# Nothing else imports the top-level package -- verified by grepping every
# non-vLLM .py under dist-packages in the built image -- so removing it simply
# lets vLLM's own pairing win.  Regression-checked on gfx950 with Qwen3-8B:
# all three cliff arms pass.
RUN uv pip uninstall --system triton_kernels 2>/dev/null || true; \
    if python3 -c "import importlib.util,sys; \
sys.exit(0 if importlib.util.find_spec('triton_kernels') is None else 1)"; then \
        echo "triton_kernels: standalone package removed"; \
    else \
        echo "ERROR: standalone triton_kernels still importable after uninstall" >&2; exit 1; \
    fi; \
    vendored=/usr/local/lib/python3.12/dist-packages/vllm/third_party/triton_kernels/matmul_ogs.py; \
    if [ -f "$vendored" ]; then \
        echo "triton_kernels: vLLM vendored copy provides matmul_ogs"; \
    else \
        echo "ERROR: vLLM no longer vendors triton_kernels/matmul_ogs.py ($vendored)." >&2; \
        echo "       Re-check the triton_kernels API generation this VLLM_REF expects." >&2; \
        exit 1; \
    fi

# ----- 6. hipFile — provided by the ROCm 7.14 base image (GA since 7.14) ------
# No source build needed.

# ----- 7. NIXL (ai-dynamo/nixl upstream + nixl-rocm-ais-mt.patch) ------------
COPY docker/scripts /tmp/nixl-scripts
COPY patches/nixl/ /tmp/nixl-patches/
# hadolint ignore=SC3054,SC3044
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
        BUILD_JOBS="${BUILD_JOBS}" \
        AIC_UCX_FAST="${AIC_UCX_FAST}" \
        UV_SYSTEM_CERTS=1 \
        /tmp/nixl-scripts/build-nixl.sh

# ----- 7b. hsa-snoop (HSA AQL queue snooper + Prometheus exporter) -----------
RUN git clone --depth 1 --branch "${HSA_SNOOP_REF}" "${HSA_SNOOP_GIT_URL}" /app/hsa-snoop && \
    cmake -S /app/hsa-snoop -B /app/hsa-snoop/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DHSA_SNOOP_PROMETHEUS=ON \
        -DCMAKE_DISABLE_FIND_PACKAGE_hip=ON && \
    cmake --build /app/hsa-snoop/build --parallel "${BUILD_JOBS:-$(nproc)}" && \
    cmake --install /app/hsa-snoop/build --prefix /usr/local && \
    test -x /usr/local/bin/hsa-snoop && \
    # hadolint ignore=SC2015
    { /usr/local/bin/hsa-snoop --help 2>&1 || true; } | grep -q -- '--prometheus' \
        || { echo "ERROR: hsa-snoop built without Prometheus mode (--prometheus absent)" >&2; exit 1; } && \
    echo "PASS: hsa-snoop ${HSA_SNOOP_REF} with Prometheus exporter (incl. AIS metrics) installed"

# ----- 8. Runtime environment ------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    NIXL_PLUGIN_DIR=/opt/nixl/lib/x86_64-linux-gnu/plugins \
    PYTHONPATH=/opt/rocm/share/amd_smi:/opt/nixl/lib/python3/dist-packages \
    LD_LIBRARY_PATH=/opt/nixl/lib/x86_64-linux-gnu:/opt/nixl/lib:/opt/rocnixl-ucx/lib:/opt/rocm/lib \
    HSA_ENABLE_IPC_MODE_LEGACY=1 \
    SAFETENSORS_FAST_GPU=1 \
    HIP_FORCE_DEV_KERNARG=1

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
      ai.amd.aic.aiter="${AITER_REF}" \
      ai.amd.aic.flash-attention="${FLASH_ATTN_REF}" \
      ai.amd.aic.lmcache="${LMCACHE_REF}" \
      ai.amd.aic.nixl="${NIXL_REF}" \
      ai.amd.aic.hipfile="rocm-${ROCM_VERSION}" \
      ai.amd.aic.hsa-snoop="${HSA_SNOOP_REF}"

ENTRYPOINT ["python3", "-m", "lmcache.server"]

# ----- 9. Wheel export stage -------------------------------------------------
FROM scratch AS wheels
COPY --from=build /wheels /

# ----- 10. Runtime image (default target) ------------------------------------
# Resets the entrypoint to vLLM: the lmcache build stage sets it to
# lmcache.server, but the default target must serve vLLM (the compose
# lmcache service overrides the entrypoint via its entrypoint: directive).
FROM build AS runtime
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
