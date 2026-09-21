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
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build git build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir \
        "torch>=2.3,<3" torchvision \
        --index-url https://download.pytorch.org/whl/cpu
# Pre-install all vLLM build-system requirements (from its pyproject.toml
# [build-system].requires) so pip can prepare vLLM's metadata without failing
# when resolving dependencies from inside the /app/vllm source tree.
RUN pip3 install --no-cache-dir \
        setuptools "setuptools-scm>=8" setuptools-rust wheel build
