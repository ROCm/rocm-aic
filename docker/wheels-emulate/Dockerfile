# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# CPU-only wheel builder for the vllm-emu (emulation) mode.  Produces two
# pip-installable wheels without any ROCm or GPU dependency:
#
#   vllm-<ver>+empty-cp312-cp312-linux_x86_64.whl  (vLLM scheduler + HTTP, no GPU kernels)
#   vllm_emulator-<ver>-py3-none-any.whl            (llm-emu profile-driven emulation plugin)
#
# These two wheels, combined with a CPU torch from pytorch.org/whl/cpu, are
# sufficient to run the emulate-mode vLLM server on any x86_64 Linux host
# without installing ROCm.
#
# Stages:
#   build   compile vllm (empty device) + build vllm_emulator wheel
#   wheels  scratch image holding just the .whl files
#
# Build:
#   WHEELS_DIR=/tmp/emulate-wheels docker buildx bake \
#     -f docker/docker-bake.hcl wheels-emulate
#
# Or standalone:
#   docker buildx build -f docker/wheels-emulate/Dockerfile \
#     --target wheels --output type=local,dest=/tmp/emulate-wheels .
#
FROM python:3.12-bookworm AS build

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG VLLM_GIT_URL=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=v0.29.0
ARG LLM_EMU_GIT_URL=https://github.com/AKafakA/llm-emu.git
ARG LLM_EMU_REF=b71c602
ARG BUILD_JOBS=
ARG AIC_VERSION=

# Build tools: cmake + ninja for vllm's CMake extension (even with empty device,
# vllm's setup.py drives cmake for Pybind11 wrappers).
# hadolint ignore=DL3008
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cmake ninja-build git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CPU torch — the same series as the ROCm build so the wheel ABI matches.
RUN pip3 install --no-cache-dir \
        "torch~=2.13.0" torchvision \
        --index-url https://download.pytorch.org/whl/cpu

# vllm [build-system] requires.
RUN pip3 install --no-cache-dir \
        setuptools "setuptools-scm>=8" setuptools-rust wheel build

# Copy vllm and llm-emu patches from the repo root.
COPY patches/vllm/    /app/patches/vllm/
COPY patches/llm-emu/ /app/patches/llm-emu/

# ----- vLLM source build (empty device: no GPU kernels) -----------------------
RUN git clone --depth 1 --branch "${VLLM_REF}" "${VLLM_GIT_URL}" /app/vllm

WORKDIR /app/vllm
# Apply vLLM patches (same set as the ROCm build).
# hadolint ignore=SC3054,SC3044,SC2098
RUN shopt -s nullglob && \
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

# Install vLLM build deps (common.txt + build/rust.txt for the empty-device path).
# hadolint ignore=SC2098
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    mkdir -p /wheels && \
    printf 'torch>=2.13,<2.14\n' > /tmp/torch-constraints.txt && \
    pip3 install -r requirements/common.txt \
        -r requirements/build/rust.txt \
        --constraint /tmp/torch-constraints.txt && \
    VLLM_TARGET_DEVICE=empty \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        MAX_JOBS="${BUILD_JOBS:-$(nproc)}" \
        CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS:-$(nproc)}" \
        CCACHE_DIR=/root/.cache/ccache CCACHE_MAXSIZE=10G \
        VLLM_VERSION_OVERRIDE="${VLLM_REF#v}+empty" \
        pip3 wheel --no-build-isolation --no-deps --no-cache-dir -w /wheels . && \
    rm -f /tmp/torch-constraints.txt

WORKDIR /

# ----- vllm_emulator (llm-emu) -----------------------------------------------
# hadolint ignore=SC3044
RUN git clone --filter=blob:none --no-checkout "${LLM_EMU_GIT_URL}" /app/llm-emu && \
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
    pip3 wheel --no-build-isolation --no-deps -w /wheels /app/llm-emu/ && \
    pip3 install --no-cache-dir --no-build-isolation --no-deps /app/llm-emu/ && \
    python3 -c "import vllm_emulator.hooks.executor_hook, vllm_emulator.oracle, \
vllm_emulator.profile.loader, vllm_emulator.profiler.trace_profiler; \
print('vllm_emulator subpackages import OK')" && \
    rm -rf /app/llm-emu /app/patches

# =============================================================================
# wheels — scratch image holding just the two emulate wheels.
#
#   docker buildx bake -f docker/docker-bake.hcl wheels-emulate
# =============================================================================
FROM scratch AS wheels
COPY --from=build /wheels /
