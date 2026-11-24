# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# aic-client: every benchmark client of the stack in one image, driven by
# `aic-bench <subcommand>` (benchmarks/aic_bench.py).
#
# It holds no ROCm and no GPU runtime: a client needs only network access to
# the endpoint, so it runs on the stack's docker network.  lm_eval (the
# accuracy gate) stays out because it pulls in torch.
#
#   /opt/aic/venv         the benchmark scripts' dependencies and agent-kv
#   /opt/aic/aiperf-venv  aiperf (AgentX), apart so that its pins cannot clash
#                         with the main venv; on PATH as `aiperf`
#   /opt/aic/benchmarks   the repo's benchmarks/ as of the build; a checkout
#                         bind-mounted at /repo takes precedence over it
#
# TLS/proxy configuration is handled entirely at runtime via environment
# variables (REQUESTS_CA_BUNDLE, SSL_CERT_FILE, NO_PROXY, etc.) passed by
# the caller; no certs are baked into the image.
#
# Build (from repo root):
#   docker buildx bake -f docker/docker-bake.hcl client --load

FROM python:3.12-slim

# docker/docker-bake.hcl owns these.  The defaults stay only because compose
# also builds this image (the `client` service) and passes its own; they must
# equal bake's, which tests/test_aic_bake_env.py checks.
ARG AIPERF_VERSION=0.13.0
ARG AGENT_KV_GIT_URL=https://github.com/sbates130272/agent-kv
ARG AGENT_KV_REF=19fcc86be6e267839889567ce5d20dd0fed6bbaf

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# agent-kv has no releases; GitHub serves any commit as a tarball, so the image
# needs no git.
RUN python -m venv /opt/aic/venv \
    && /opt/aic/venv/bin/pip install \
        openai httpx pyyaml tqdm \
        transformers huggingface-hub \
        prometheus-client aiohttp \
        matplotlib pandas numpy pytest \
        "agent-kv[tiktoken] @ ${AGENT_KV_GIT_URL}/archive/${AGENT_KV_REF}.tar.gz"

# The otel extra adds aiperf's OTLP metrics exporter (--otel-url); the AIC
# monitoring stack has no OTLP receiver, so nothing here uses it by default.
# AIPERF_VERSION=latest installs whatever PyPI serves on the day of the build.
RUN python -m venv /opt/aic/aiperf-venv \
    && if [ "${AIPERF_VERSION}" = "latest" ]; then \
         /opt/aic/aiperf-venv/bin/pip install 'aiperf[otel]'; \
       else \
         /opt/aic/aiperf-venv/bin/pip install "aiperf[otel]==${AIPERF_VERSION}"; \
       fi \
    && ln -s /opt/aic/aiperf-venv/bin/aiperf /usr/local/bin/aiperf

# The container runs as whatever uid the caller picks, which has no passwd
# entry and owns nothing in the image: every path a tool writes by default
# points at /tmp, and the mount points exist world-writable for a run that
# mounts nothing there.
ENV PATH=/opt/aic/venv/bin:${PATH} \
    HOME=/tmp \
    HF_HOME=/hf_cache \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 \
    AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
RUN mkdir -m 1777 /hf_cache /logs

COPY benchmarks/ /opt/aic/benchmarks/
COPY --chmod=755 docker/client/aic-bench /usr/local/bin/aic-bench

WORKDIR /opt/aic
CMD ["sleep", "infinity"]

HEALTHCHECK NONE
