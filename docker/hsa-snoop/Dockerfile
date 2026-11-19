# check=skip=InvalidDefaultArgInFrom
# The FROM images are build args with no default here (docker/docker-bake.hcl
# owns them), so they are blank at lint time.
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
# Build args: docker/docker-bake.hcl is the only owner of every pin, URL and
# base image, so no ARG here has a default and each stage stops when one is
# unset.  Build through bake (from repo root):
#   docker buildx bake -f docker/docker-bake.hcl hsa-snoop --load
#
# or hand bake's resolved args to a raw build:
#   mapfile -t args < <(docker/scripts/aic-bake-env.sh --build-args hsa-snoop)
#   docker buildx build "${args[@]}" -f docker/hsa-snoop/Dockerfile \
#     -t aic-hsa-snoop:dev .
#
# Or via make:
#   make build                  (bake's hsa-snoop target; tags aic-hsa-snoop:latest)
#   make dist-build-exporters   (Slurm)

# An unset image arg stops the build at its FROM:
#   base name (${ROCM_BUILD_IMAGE}) should not be blank
ARG ROCM_BUILD_IMAGE
# AMD publishes no non-full rocm/dev-ubuntu-26.04:${ROCM_VERSION} image, so the
# runtime stage starts from plain ubuntu.  It stays plain: hsa-snoop needs no
# ROCm userspace (see the stage notes above).
ARG ROCM_RUNTIME_IMAGE

# =============================================================================
# Build stage — needs ROCm HSA headers and cmake/ninja
# =============================================================================
# hadolint ignore=DL3006
FROM ${ROCM_BUILD_IMAGE} AS build

ARG HSA_SNOOP_GIT_URL
ARG HSA_SNOOP_REF
ARG BUILD_JOBS

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN : "${HSA_SNOOP_GIT_URL:?unset: build through docker/docker-bake.hcl}" \
      "${HSA_SNOOP_REF:?unset: build through docker/docker-bake.hcl}"

# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cmake ninja-build git ca-certificates

RUN git clone --filter=blob:none --no-checkout "${HSA_SNOOP_GIT_URL}" /app/hsa-snoop && \
    git -C /app/hsa-snoop checkout "${HSA_SNOOP_REF}" && \
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

ARG HSA_SNOOP_REF
ARG ROCM_VERSION
ARG ROCM_RUNTIME_IMAGE

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Both are image labels.
RUN : "${HSA_SNOOP_REF:?unset: build through docker/docker-bake.hcl}" \
      "${ROCM_VERSION:?unset: build through docker/docker-bake.hcl}"

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
CMD ["--all", "--prometheus", "--prometheus-port", "9488", "--mem-snoop", "--poll-us", "1000"]

# The runtime image has bash but neither curl nor python, so probe over
# /dev/tcp.  TCP connect only: the exporter port accepting is the liveness
# signal.  The port is the CMD default above; a CMD override that changes it
# has to override this check too.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/9488"]
