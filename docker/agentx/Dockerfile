# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# aic-agentx: aiperf-based AgentX benchmark runner.
#
# Runs the AgentX MVP scenario (inferencex-agentx-mvp) against a Lemonade or
# vLLM endpoint and exports OTEL metrics to the AIC monitoring stack.
#
# This image is intentionally thin (python:3.12-slim + aiperf).  It does NOT
# include ROCm or a GPU runtime — it is a pure benchmark driver that runs
# alongside the inference stack, not on the GPU node.
#
# TLS/proxy configuration is handled entirely at runtime via environment
# variables (REQUESTS_CA_BUNDLE, SSL_CERT_FILE, NO_PROXY, etc.) passed by
# the caller — no certs are baked into the image.
#
# Build (from repo root):
#   DOCKER_BUILDKIT=1 docker buildx build \
#     -f docker/agentx/Dockerfile \
#     -t aic-agentx:latest \
#     .
#
# Or via make:
#   make agentx-build
#
# Or via bake:
#   docker buildx bake -f docker/docker-bake.hcl agentx --load

ARG AIPERF_VERSION=latest

FROM python:3.12-slim

ARG AIPERF_VERSION

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Install aiperf with OTEL support.  When AIPERF_VERSION=latest the constraint
# is unconstrained; pin it (e.g. AIPERF_VERSION=0.4.2) for reproducible builds.
RUN if [ "${AIPERF_VERSION}" = "latest" ]; then \
      pip install 'aiperf[otel]'; \
    else \
      pip install "aiperf[otel]==${AIPERF_VERSION}"; \
    fi

# Bind-mount the host HF cache at runtime so tokenizer snapshots are reused
# without downloading them inside the container.
ENV HF_HOME=/hf_cache
ENV AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
ENV AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800

WORKDIR /results
ENTRYPOINT ["agentx"]
