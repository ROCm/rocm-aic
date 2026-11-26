# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Slim images for MORI UMBP server processes.
#
# Targets:
#   mori-umbp-server    MORI UMBP standalone KV-pool server + ROCm HIP runtime
#   mori-umbp-master    Fleet GlobalBlockIndex coordinator; CPU-only, no GPU
#
# Both consume the MORI wheel built by docker/vllm/Dockerfile's wheels-mori
# stage.  Build via bake:
#   docker buildx bake -f docker/docker-bake.hcl mori-umbp-server mori-umbp-master

ARG ROCM_BASE_IMAGE
ARG ROCM_RUNTIME_IMAGE

# --- Source for AMD apt signing key (avoids network fetch) -------------------
# hadolint ignore=DL3006
FROM ${ROCM_BASE_IMAGE} AS key-source

# --- MORI wheel source (placeholder; bake overrides with wheels-mori target) -
FROM scratch AS wheels-mori

# --- Extract MORI C++ binaries from the wheel archive -----------------------
# The wheel zip contains mori/umbp_standalone_server and mori/umbp_master as
# compiled ELF binaries (confirmed by build_mori's zipfile -l check).
FROM ubuntu:26.04 AS extract
# hadolint ignore=DL3008
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 && \
    rm -rf /var/lib/apt/lists/*
COPY --from=wheels-mori / /tmp/wheels/
RUN python3 -m zipfile -e /tmp/wheels/amd_mori-*.whl /tmp/mori-extracted/ && \
    install -m 0755 /tmp/mori-extracted/mori/umbp_standalone_server /usr/local/bin/umbp_standalone_server && \
    install -m 0755 /tmp/mori-extracted/mori/umbp_master /usr/local/bin/umbp_master

# =============================================================================
# mori-umbp-server — MORI UMBP standalone server
#
# Needs the ROCm HIP runtime (libamdhip64 + libhsa-runtime64) for the shared
# GPU memory management HIP IPC handles MORI uses to move KV tokens.  Also
# needs gRPC, ibverbs and libnuma at runtime.
#
# Shares network, IPC and PID namespaces with vLLM (like lmcache): vLLM joins
# this container's netns so the gRPC unix socket and the HIP IPC handles
# (passed over @hsa<pid> abstract socket) are reachable.
# =============================================================================
FROM ${ROCM_RUNTIME_IMAGE} AS mori-umbp-server
ARG ROCM_VERSION
ARG AIC_VERSION

# hadolint ignore=DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN : "${ROCM_VERSION:?unset: build through docker/docker-bake.hcl}"

# TLS CA cert (corporate proxy).  Installed before any HTTPS apt access.
RUN --mount=type=secret,id=tls_cert,required=false \
    if [ -f /run/secrets/tls_cert ] && [ -s /run/secrets/tls_cert ]; then \
        mkdir -p /usr/local/share/ca-certificates && \
        cp /run/secrets/tls_cert /usr/local/share/ca-certificates/corp-ca.crt; \
    fi

# Step 1: ca-certificates (required before HTTPS AMD apt repo).
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates && \
    update-ca-certificates

# Step 2: AMD apt repo + minimal ROCm HIP runtime + gRPC/ibverbs/numa.
# Key and sources list copied from key-source (the full ROCm toolchain image)
# so the repo URL tracks the base image automatically.
COPY --from=key-source /etc/apt/keyrings/amdrocm.gpg /etc/apt/keyrings/amdrocm.gpg
COPY --from=key-source /etc/apt/sources.list.d/rocm.list /etc/apt/sources.list.d/amdrocm.list
ARG BLAS_CACHE_BUST
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        amdrocm-runtime \
        amdrocm-base \
        ibverbs-providers \
        libgrpc++-dev \
        libnuma1 \
        libpci3 \
        librdmacm1t64 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=extract /usr/local/bin/umbp_standalone_server /usr/local/bin/umbp_standalone_server

LABEL org.opencontainers.image.vendor="Advanced Micro Devices, Inc."
LABEL org.opencontainers.image.title="aic-mori-umbp"
LABEL org.opencontainers.image.description="MORI UMBP standalone KV-pool server"
LABEL org.opencontainers.image.version="${AIC_VERSION}"

ENTRYPOINT ["/usr/local/bin/umbp_standalone_server"]

# =============================================================================
# mori-umbp-master — MORI GlobalBlockIndex fleet coordinator
#
# Pure CPU/network service; no GPU required.
# gRPC on :50051; Prometheus /metrics on :9091.
# One per fleet — on all other nodes set UMBP_MASTER_ADDRESS to point here
# and leave the mori-master profile off.
# =============================================================================
FROM ubuntu:26.04 AS mori-umbp-master
ARG AIC_VERSION

# hadolint ignore=DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# TLS CA cert.
RUN --mount=type=secret,id=tls_cert,required=false \
    if [ -f /run/secrets/tls_cert ] && [ -s /run/secrets/tls_cert ]; then \
        mkdir -p /usr/local/share/ca-certificates && \
        cp /run/secrets/tls_cert /usr/local/share/ca-certificates/corp-ca.crt; \
    fi

# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgrpc++-dev \
        libprotobuf-dev && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=extract /usr/local/bin/umbp_master /usr/local/bin/umbp_master

LABEL org.opencontainers.image.vendor="Advanced Micro Devices, Inc."
LABEL org.opencontainers.image.title="aic-mori-umbp-master"
LABEL org.opencontainers.image.description="MORI UMBP fleet GlobalBlockIndex coordinator"
LABEL org.opencontainers.image.version="${AIC_VERSION}"

ENTRYPOINT ["/usr/local/bin/umbp_master"]
