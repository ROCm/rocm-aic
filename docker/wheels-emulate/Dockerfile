# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# CPU-only wheel builder for the vllm-emu (emulation) mode.  Produces two
# pip-installable wheels without any ROCm or GPU dependency:
#
#   vllm-<ver>+empty-cp312-cp312-linux_x86_64.whl  (vLLM scheduler + HTTP, no GPU kernels)
#   llm_emu-<ver>-py3-none-any.whl                  (llm-emu profile-driven emulation plugin,
#                                                    imported as vllm_emulator)
#
# These two wheels, combined with a CPU torch from pytorch.org/whl/cpu, are
# sufficient to run the emulate-mode vLLM server on any x86_64 Linux host
# without installing ROCm.
#
# Stages:
#   build   compile vllm (empty device) + build llm_emu wheel
#   wheels  scratch image holding just the .whl files
#
# Build args: docker/docker-bake.hcl is the only owner of every pin and URL, so
# no ARG here has a default and the build stage stops when one is unset.
# Build through bake (from repo root):
#   WHEELS_DIR=/tmp/emulate-wheels docker buildx bake \
#     -f docker/docker-bake.hcl wheels-emulate
#
# or hand bake's resolved args to a raw build:
#   mapfile -t args < <(docker/scripts/aic-bake-env.sh --build-args wheels-emulate)
#   docker buildx build "${args[@]}" -f docker/wheels-emulate/Dockerfile \
#     --target wheels --output type=local,dest=/tmp/emulate-wheels .
#
FROM python:3.12-bookworm AS build

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG VLLM_GIT_URL
ARG VLLM_REF
ARG LLM_EMU_GIT_URL
ARG LLM_EMU_REF
ARG PYTORCH_SERIES
ARG BUILD_JOBS
RUN : "${VLLM_GIT_URL:?unset: build through docker/docker-bake.hcl}" \
      "${VLLM_REF:?unset: build through docker/docker-bake.hcl}" \
      "${LLM_EMU_GIT_URL:?unset: build through docker/docker-bake.hcl}" \
      "${LLM_EMU_REF:?unset: build through docker/docker-bake.hcl}" \
      "${PYTORCH_SERIES:?unset: build through docker/docker-bake.hcl}"

# Build tools: cmake + ninja for vllm's CMake extension (even with empty device,
# vllm's setup.py drives cmake for Pybind11 wrappers).
# hadolint ignore=DL3008
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cmake ninja-build git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CPU torch — the same series as the ROCm build so the wheel ABI matches.
RUN pip3 install --no-cache-dir \
        "torch~=${PYTORCH_SERIES}.0" torchvision \
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
    printf 'torch~=%s.0\n' "${PYTORCH_SERIES}" > /tmp/torch-constraints.txt && \
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
    VLLM_EMULATOR_ENABLE_ORACLE=1 python3 -c "import importlib.util as u, types; \
from vllm_emulator.hooks.executor_hook import ExecutorEmulatorHook as H; \
assert u.find_spec('vllm.v1.worker.gpu.model_runner') or u.find_spec('vllm.v1.worker.gpu_model_runner'), 'vLLM model-runner module moved'; \
h = H.__new__(H); h._filler_token_id = 0; \
o = h._create_fake_output(types.SimpleNamespace(num_scheduled_tokens={'r': 1}, scheduled_new_reqs=[])); \
assert o.req_ids == ['r'] and o.sampled_token_ids == [[0]], o; \
print('vllm_emulator fake ModelRunnerOutput OK')" && \
    rm -rf /app/llm-emu /app/patches

# =============================================================================
# wheels — scratch image holding just the two emulate wheels.
#
#   docker buildx bake -f docker/docker-bake.hcl wheels-emulate
# =============================================================================
FROM scratch AS wheels
COPY --from=build /wheels /
