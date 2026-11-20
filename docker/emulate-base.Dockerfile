# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Minimal base for the emulate (CPU-only) build of docker/vllm/Dockerfile.
# Used by .github/workflows/rocm-aic-emulate-test.yml instead of the full
# ROCm+PyTorch aic-base image (which takes hours to build on a GitHub runner).
#
# Provides Python 3.14, the build tools vLLM's source compilation needs, and
# a CPU-only PyTorch wheel from pytorch.org/whl/cpu, and the image's own pip3,
# which docker/vllm/Dockerfile's `vllm` stage installs with.
#
# The Python minor must equal ROCM_BASE_IMAGE's (Ubuntu 26.04: 3.14).  The
# build_vllm stage builds the vLLM wheel on that image and tags it for its
# interpreter (cp314), and the `vllm` stage pip-installs it onto this base.
FROM python:3.14-bookworm
# Match the SHELL used by docker/base/Dockerfile so bash-specific syntax in
# docker/vllm/Dockerfile (shopt, arrays) works when this is the base context.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# docker/docker-bake.hcl owns it; the workflows pass it from
#   docker/scripts/aic-bake-env.sh --get PYTORCH_SERIES
ARG PYTORCH_SERIES
RUN : "${PYTORCH_SERIES:?unset: pass it from docker/scripts/aic-bake-env.sh --get PYTORCH_SERIES}"
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build git build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir \
        "torch~=${PYTORCH_SERIES}.0" torchvision \
        --index-url https://download.pytorch.org/whl/cpu
# Pre-install all vLLM build-system requirements (from its pyproject.toml
# [build-system].requires) so pip can prepare vLLM's metadata without failing
# when resolving dependencies from inside the /app/vllm source tree.
RUN pip3 install --no-cache-dir \
        setuptools "setuptools-scm>=8" setuptools-rust wheel build
# python:3.14-bookworm uses site-packages; the ROCm Ubuntu base uses dist-packages.
# Symlink dist-packages -> site-packages so paths written for the ROCm base resolve.
RUN ln -sf /usr/local/lib/python3.14/site-packages \
           /usr/local/lib/python3.14/dist-packages

# Build parent, not a service: the emulate image defines its own check.
HEALTHCHECK NONE
