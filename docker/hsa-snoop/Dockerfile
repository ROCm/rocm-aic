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
#   runtime  plain ubuntu + the hsa-snoop binary -- no ROCm userspace at all
#
# The runtime stage deliberately installs NO ROCm packages.  hsa-snoop reads
# kernel interfaces (ftrace, KFD debugfs) rather than calling into the ROCm
# userspace stack: its CMakeLists links only Threads, src/aql.h states it
# "deliberately does NOT #include <hsa/hsa.h>", and `ldd` on the built binary
# resolves to libstdc++/libgcc_s/libc/libm and nothing else.  Compose backs
# this up -- the service gets /dev/kfd, /sys/kernel/debug and
# /sys/kernel/tracing, not a ROCm install.
#
# Installing "amdrocm-runtime${MM}" here used to cost 965 MB of /opt/rocm, of
# which 843 MB was LLVM/Clang/Flang (libMLIR 170 MB, libLLVM 128 MB,
# libclang-cpp 86 MB, flang-23 83 MB, bbc 73 MB ...) -- a Fortran compiler in a
# metrics sidecar.  Image: ~1.19 GB -> ~80 MB.
#
# Nothing working was lost.  hsa-snoop opportunistically popen()s
# /opt/rocm/bin/rocminfo for GPU metric labels and reads /opt/rocm/.info/version
# for a version label, but neither file was present even with the package
# installed (it ships no rocminfo), so both paths already took their
# fall-backs.  To get those labels, COPY rocminfo and its ldd closure from the
# build stage -- tens of MB -- rather than reinstating the package.
#
# Build (from repo root):
#   docker buildx build -f docker/hsa-snoop/Dockerfile \
#     -t aic-hsa-snoop:local .
#
# Or via make:
#   make build-exporters

ARG ROCM_VERSION=7.14.1
ARG ROCM_BUILD_IMAGE=rocm/dev-ubuntu-24.04:${ROCM_VERSION}-full
# rocm/dev-ubuntu-24.04:7.14.x (no -full suffix) does not exist in the AMD repo,
# so the runtime stage starts from plain ubuntu.  It stays plain: hsa-snoop
# needs no ROCm userspace (see the stage notes above).  This previously read
# "only libhsa-runtime64.so is needed" -- the binary does not link that either.
ARG ROCM_RUNTIME_IMAGE=ubuntu:24.04

# =============================================================================
# Build stage — needs ROCm HSA headers and cmake/ninja
# =============================================================================
# hadolint ignore=DL3006
FROM ${ROCM_BUILD_IMAGE} AS build

ARG HSA_SNOOP_GIT_URL=https://github.com/sbates130272/hsa-snoop.git
ARG HSA_SNOOP_REF=v1.2.0
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
# Runtime stage — plain ubuntu + the binary (~80 MB vs ~1.19 GB)
# =============================================================================
# hadolint ignore=DL3006
FROM ${ROCM_RUNTIME_IMAGE} AS runtime

ARG HSA_SNOOP_REF=v1.2.0
ARG ROCM_VERSION
ARG ROCM_RUNTIME_IMAGE

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# The binary's entire link closure: libstdc++6 pulls libgcc_s, and libc/libm
# are already in the base.  Stated explicitly rather than relying on apt having
# dragged libstdc++6 in as its own dependency.
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libstdc++6

COPY --from=build /usr/local/bin/hsa-snoop /usr/local/bin/hsa-snoop

# Fail the build, not the deployment, if the binary ever grows a dependency
# this stage does not provide -- an upstream change that starts linking the HSA
# runtime would otherwise only show up as a sidecar crash-looping in compose.
RUN ldd /usr/local/bin/hsa-snoop > /tmp/ldd.txt 2>&1; \
    cat /tmp/ldd.txt; \
    if grep -qi 'not found' /tmp/ldd.txt; then \
        echo "ERROR: unresolved shared libraries in the slim runtime (see above)." >&2; \
        exit 1; \
    fi; \
    if grep -qiE 'libhsa|librocm|libamd|/opt/rocm' /tmp/ldd.txt; then \
        echo "ERROR: hsa-snoop now links ROCm libraries this slim runtime does not ship." >&2; \
        echo "       COPY them from the build stage, or reinstate amdrocm-runtime here." >&2; \
        exit 1; \
    fi; \
    /usr/local/bin/hsa-snoop --help >/dev/null || { \
        echo "ERROR: hsa-snoop --help failed in the slim runtime." >&2; exit 1; }; \
    rm -f /tmp/ldd.txt

LABEL org.opencontainers.image.title="aic-hsa-snoop" \
      org.opencontainers.image.description="AMD Infinity Context: HSA AQL snooper + Prometheus exporter sidecar" \
      ai.amd.aic.hsa-snoop="${HSA_SNOOP_REF}" \
      ai.amd.aic.rocm="${ROCM_VERSION}"

EXPOSE 9488

ENTRYPOINT ["/usr/local/bin/hsa-snoop"]
CMD ["--all", "--prometheus", "--prometheus-port", "9488", "--mem-snoop"]
