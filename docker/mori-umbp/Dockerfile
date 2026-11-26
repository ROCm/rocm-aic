# check=skip=InvalidDefaultArgInFrom
# The FROM images are build args with no default here (docker/docker-bake.hcl
# owns them), so they are blank at lint time.
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Slim images for MORI's UMBP server processes: the ROCm HIP runtime plus the
# UMBP programs and MORI shared libraries from the MORI wheel, with no Python,
# torch or vLLM.
#
# Targets:
#   mori-umbp-server    umbp_standalone_server, the node-wide KV pool
#   mori-umbp-master    umbp_master, the fleet GlobalBlockIndex coordinator
#
# The two share every layer and differ only in entrypoint: umbp_master uses no
# GPU but links the same HIP runtime as the server.
#
# Both take the MORI wheel from the `wheels-mori` build context (the wheels-mori
# stage of docker/vllm/Dockerfile).  Build via bake:
#   docker buildx bake -f docker/docker-bake.hcl mori-umbp-server mori-umbp-master

ARG ROCM_BASE_IMAGE
ARG ROCM_RUNTIME_IMAGE

# Source of the AMD apt signing key and sources list.
# hadolint ignore=DL3006
FROM ${ROCM_BASE_IMAGE} AS key-source

# Empty default for the `wheels-mori` build context.
FROM scratch AS wheels-mori

# The UMBP programs and the libmori_*.so they load.  Their RUNPATH starts with
# $ORIGIN, so they stay in one directory.
# hadolint ignore=DL3006
FROM ${ROCM_BASE_IMAGE} AS extract
RUN --mount=type=bind,from=wheels-mori,target=/w \
    python3 -m zipfile -e /w/amd_mori-*.whl /tmp/x && \
    mkdir -p /opt/mori && \
    install -m 0755 /tmp/x/mori/umbp_standalone_server /tmp/x/mori/umbp_master /opt/mori/ && \
    install -m 0644 /tmp/x/mori/libmori_*.so /opt/mori/

# hadolint ignore=DL3006
FROM ${ROCM_RUNTIME_IMAGE} AS runtime
ARG ROCM_VERSION
ARG AIC_VERSION

# hadolint ignore=DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN : "${ROCM_VERSION:?unset: build through docker/docker-bake.hcl}" \
      "${AIC_VERSION:?unset: build through docker/docker-bake.hcl}"

# Corporate CA; the next RUN installs ca-certificates and picks it up.
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
        ca-certificates && \
    update-ca-certificates

# ROCm HIP runtime and what the UMBP programs link.  curl is for the master's
# compose healthcheck.
COPY --from=key-source /etc/apt/keyrings/amdrocm.gpg /etc/apt/keyrings/amdrocm.gpg
COPY --from=key-source /etc/apt/sources.list.d/rocm.list /etc/apt/sources.list.d/amdrocm.list
ARG BLAS_CACHE_BUST
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        amdrocm-base \
        amdrocm-runtime \
        curl \
        ibverbs-providers \
        libdrm-amdgpu1 \
        libdrm2 \
        libgrpc++1.51t64 \
        libibverbs1 \
        libnuma1 \
        libpci3 \
        libprotobuf32t64 \
        librdmacm1t64 && \
    _core="$(find /opt/rocm -maxdepth 1 -type d -name 'core-*' | head -n 1)" && \
    test -n "${_core}" && ln -sfn "${_core}" /opt/rocm/core

COPY --from=extract /opt/mori /opt/mori

# The programs also link ROCm's bundled zlib and numa, which live in the sysdeps
# directory beside copies of common system libraries; the wrappers put it on
# the loader path of these two programs only.  The ldd check catches a runtime
# library missing from the list above.
# /run/umbp holds the server's sockets; world-writable so the named volume
# compose mounts there takes that mode and a non-root container can bind.
RUN _libs=/opt/rocm/core/lib:/opt/rocm/core/lib/rocm_sysdeps/lib && \
    for _bin in umbp_standalone_server umbp_master; do \
        _missing="$(LD_LIBRARY_PATH="${_libs}" ldd "/opt/mori/${_bin}" | grep 'not found' || true)"; \
        [ -z "${_missing}" ] || { echo "ERROR: ${_bin} has unresolved libraries:" >&2; echo "${_missing}" >&2; exit 1; }; \
        printf '#!/bin/sh\nLD_LIBRARY_PATH="%s${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" exec "%s" "$@"\n' \
            "${_libs}" "/opt/mori/${_bin}" > "/usr/local/bin/${_bin}" && \
        chmod 0755 "/usr/local/bin/${_bin}"; \
    done && \
    mkdir -p /run/umbp && chmod 1777 /run/umbp

LABEL org.opencontainers.image.vendor="Advanced Micro Devices, Inc." \
      org.opencontainers.image.version="${AIC_VERSION}"

# =============================================================================
# mori-umbp-master — MORI GlobalBlockIndex fleet coordinator.
# gRPC on :50051; Prometheus /metrics on :9091.  One per fleet.
# =============================================================================
FROM runtime AS mori-umbp-master
LABEL org.opencontainers.image.title="aic-mori-umbp-master" \
      org.opencontainers.image.description="MORI UMBP fleet GlobalBlockIndex coordinator"
ENTRYPOINT ["umbp_master"]

# =============================================================================
# mori-umbp-server — MORI UMBP standalone server.
#
# vLLM joins this container's network and PID namespaces (as it does
# lmcache's), so the gRPC unix socket and the HIP IPC handles are reachable.
# =============================================================================
FROM runtime AS mori-umbp-server
LABEL org.opencontainers.image.title="aic-mori-umbp" \
      org.opencontainers.image.description="MORI UMBP standalone KV-pool server"
ENTRYPOINT ["umbp_standalone_server"]
