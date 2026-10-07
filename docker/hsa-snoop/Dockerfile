# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# aic-hsa-snoop sidecar image: HSA AQL queue/dispatch snooper + Prometheus
# exporter (includes AIS GPU metrics).
#
# Runs as a privileged sidecar in the PID namespace of the lmcache container
# so it can observe both vLLM and LMCache GPU dispatches without needing
# host PID. Requires tracefs mounted from the host (compose handles this).
#
# Stages:
#   build    compile hsa-snoop from source against ROCm HSA headers
#   runtime  minimal ROCm runtime + hsa-snoop binary only
#
# Build (from repo root):
#   docker buildx build -f docker/hsa-snoop/Dockerfile \
#     -t aic-hsa-snoop:local .
#
# Or via make:
#   make build-exporters

ARG ROCM_VERSION=7.14.1
ARG ROCM_BUILD_IMAGE=rocm/dev-ubuntu-24.04:${ROCM_VERSION}-full
# rocm/dev-ubuntu-24.04:7.14.x (no -full suffix) does not exist in the AMD repo.
# Use ubuntu:24.04 + the HSA runtime library (only libhsa-runtime64.so is needed).
ARG ROCM_RUNTIME_IMAGE=ubuntu:24.04

# =============================================================================
# Build stage — needs ROCm HSA headers and cmake/ninja
# =============================================================================
# hadolint ignore=DL3006
FROM ${ROCM_BUILD_IMAGE} AS build

ARG HSA_SNOOP_GIT_URL=https://github.com/sbates130272/hsa-snoop.git
ARG HSA_SNOOP_REF=v1.1.1
ARG BUILD_JOBS=

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cmake ninja-build git ca-certificates

RUN git clone --depth 1 --branch "${HSA_SNOOP_REF}" "${HSA_SNOOP_GIT_URL}" /app/hsa-snoop && \
    cmake -S /app/hsa-snoop -B /app/hsa-snoop/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DHSA_SNOOP_PROMETHEUS=ON \
        -DCMAKE_DISABLE_FIND_PACKAGE_hip=ON && \
    cmake --build /app/hsa-snoop/build --parallel "${BUILD_JOBS:-$(nproc)}" && \
    cmake --install /app/hsa-snoop/build --prefix /usr/local && \
    test -x /usr/local/bin/hsa-snoop && \
    # hadolint ignore=SC2015
    { /usr/local/bin/hsa-snoop --help 2>&1 || true; } | grep -q -- '--prometheus' \
        || { echo "ERROR: hsa-snoop built without Prometheus mode" >&2; exit 1; } && \
    strip --strip-debug /usr/local/bin/hsa-snoop

# =============================================================================
# Runtime stage — plain ROCm runtime image + binary only (~3-4 GB vs ~18 GB)
# =============================================================================
# hadolint ignore=DL3006
FROM ${ROCM_RUNTIME_IMAGE} AS runtime

ARG HSA_SNOOP_REF=v1.1.1
ARG ROCM_VERSION
ARG ROCM_RUNTIME_IMAGE

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# hsa-snoop needs libhsa-runtime64.so to intercept HSA AQL dispatches.
# Copy the AMD apt signing key from the build stage (which has the -full image).
# hadolint ignore=DL3008
COPY --from=build /etc/apt/keyrings/amdrocm.gpg /etc/apt/keyrings/amdrocm.gpg
COPY --from=build /etc/apt/sources.list.d/rocm.list /etc/apt/sources.list.d/amdrocm.list
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates && \
    update-ca-certificates && \
    apt-get update && \
    _mm=$(echo "${ROCM_VERSION}" | cut -d. -f1-2) && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        "amdrocm-runtime${_mm}"

COPY --from=build /usr/local/bin/hsa-snoop /usr/local/bin/hsa-snoop

LABEL org.opencontainers.image.title="aic-hsa-snoop" \
      org.opencontainers.image.description="AMD Infinity Context: HSA AQL snooper + Prometheus exporter sidecar" \
      ai.amd.aic.hsa-snoop="${HSA_SNOOP_REF}" \
      ai.amd.aic.rocm="${ROCM_VERSION}"

EXPOSE 9488

ENTRYPOINT ["/usr/local/bin/hsa-snoop"]
CMD ["--all", "--prometheus", "--prometheus-port", "9488", "--mem-snoop"]
