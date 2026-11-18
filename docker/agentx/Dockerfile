# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# aic-agentx: aiperf, the client of the AgentX benchmark.
#
# benchmarks/agentx/run_agentx.sh runs `aiperf profile --scenario
# inferencex-agentx-mvp` from it against an OpenAI-compatible endpoint, on the
# stack's docker network.
#
# This image is intentionally thin (python:3.12-slim + aiperf).  It does NOT
# include ROCm or a GPU runtime — it is a pure benchmark driver that needs
# only network access to the endpoint.
#
# TLS/proxy configuration is handled entirely at runtime via environment
# variables (REQUESTS_CA_BUNDLE, SSL_CERT_FILE, NO_PROXY, etc.) passed by
# the caller — no certs are baked into the image.
#
# Build (from repo root).  docker/docker-bake.hcl is the only owner of
# AIPERF_VERSION; there is no default here and the build stops without it:
#   docker buildx bake -f docker/docker-bake.hcl agentx --load
#
# or a raw build with bake's resolved args:
#   mapfile -t args < <(docker/scripts/aic-bake-env.sh --build-args agentx)
#   docker buildx build "${args[@]}" -f docker/agentx/Dockerfile \
#     -t aic-agentx:latest .

FROM python:3.12-slim

ARG AIPERF_VERSION

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# The otel extra adds aiperf's OTLP metrics exporter (--otel-url); the AIC
# monitoring stack has no OTLP receiver, so nothing here uses it by default.
# AIPERF_VERSION=latest installs whatever PyPI serves on the day of the build.
RUN : "${AIPERF_VERSION:?unset: build through docker/docker-bake.hcl}" && \
    if [ "${AIPERF_VERSION}" = "latest" ]; then \
      pip install 'aiperf[otel]'; \
    else \
      pip install "aiperf[otel]==${AIPERF_VERSION}"; \
    fi

# Bind-mount the host HF cache at runtime so the tokenizer and the trace
# dataset are reused instead of downloaded on every run.
ENV HF_HOME=/hf_cache
ENV AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
ENV AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800

# aiperf is the only console script the package installs.
WORKDIR /results
ENTRYPOINT ["aiperf"]
