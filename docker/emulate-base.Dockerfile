# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Minimal base for the emulate (CPU-only) build of docker/vllm/Dockerfile.
# Used by .github/workflows/rocm-aic-emulate-test.yml instead of the full
# ROCm+PyTorch aic-base image (which takes hours to build on a GitHub runner).
#
# Provides Python 3.12, the build tools vLLM's source compilation needs, and
# a CPU-only PyTorch wheel from pytorch.org/whl/cpu.
FROM python:3.12-bookworm
# Match the SHELL used by docker/base/Dockerfile so bash-specific syntax in
# docker/vllm/Dockerfile (shopt, arrays) works when this is the base context.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:git-core/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ninja-build

RUN pip3 install --no-cache-dir \
        "torch~=2.13.0" torchvision \
        --index-url https://download.pytorch.org/whl/cpu

# Pre-install all vLLM build-system requirements (from its pyproject.toml
# [build-system].requires) so pip can prepare vLLM's metadata without failing
# when resolving dependencies from inside the /app/vllm source tree.
RUN pip3 install --no-cache-dir \
        setuptools "setuptools-scm>=8" setuptools-rust wheel build
# Install uv + uvx so COPY --from=base /usr/local/bin/uv[x] in vllm/Dockerfile succeeds.
RUN curl -fsSL "https://astral.sh/uv/0.12.15/install.sh" | sh \
    && ln -s /root/.local/bin/uv  /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx
# python:3.12-bookworm uses site-packages; the ROCm Ubuntu base uses dist-packages.
# Symlink dist-packages -> site-packages so COPY --from=build_vllm dist-packages/ works.
RUN ln -sf /usr/local/lib/python3.12/site-packages \
           /usr/local/lib/python3.12/dist-packages
