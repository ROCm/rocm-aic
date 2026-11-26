# check=skip=InvalidDefaultArgInFrom
# The FROM images are build args with no default here (docker/docker-bake.hcl
# owns them), so they are blank at lint time.
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# AIC base image: Ubuntu 26.04 + ROCm + PyTorch (source) + torchvision (source)
# + NIXL (KV-transfer transport).
#
# This is the shared foundation consumed by docker/vllm/Dockerfile and
# docker/lmcache/Dockerfile via --build-context base=docker-image://aic-base:TAG.
# Both images bind-mount the wheel artifacts from the build_pytorch and
# build_torchvision stages at build time (--mount=type=bind,from=build_pytorch,...).
#
# Stages:
#   pip-wheels        scratch stub; --build-context pip-wheels=<dir> supplies the
#                     published wheel set that runtime-from-wheels installs
#   base              ROCm-full base + system deps                    (sections 0-1)
#   build_pytorch     PyTorch built from source                       (section 2b)
#   build_torchvision torchvision built from source                   (section 2c)
#   build_nixl        NIXL built from source (uses ROCm headers)      (section 2d)
#   wheels            scratch image holding just the nixl_rocm wheel (--target wheels)
#   build_clr_patch   patched libamdhip64.so (IPC event ring fix)     (section 2e)
#   rocm-apt-fetch    the ROCm apt key and source list, downloaded
#   rocm-apt          those two files: from base, or from rocm-apt-fetch
#   runtime-os        Ubuntu 26.04 + 13 ROCm runtime pkgs             (section 3)
#   runtime-from-wheels  runtime-os + torch + NIXL from published wheels, nothing
#                     compiled (`make build-from-wheels`)
#   runtime           runtime-os + torch + NIXL from the build stages (default)
#
# Build args: docker/docker-bake.hcl is the only owner of every pin, URL and
# base image, so no ARG here has a default and each consuming stage stops when
# one is unset.  Build through bake (from repo root):
#   ROCM_ARCH=gfx950 docker buildx bake -f docker/docker-bake.hcl base --load
#
# or hand bake's resolved args to a raw build:
#   mapfile -t args < <(ROCM_ARCH=gfx950 docker/scripts/aic-bake-env.sh --build-args base)
#   docker buildx build "${args[@]}" -f docker/base/Dockerfile -t aic-base:dev .
#
# Or via make:
#   make build ROCM_ARCH=gfx950         (local)
#   make dist-build AIC_ROCM_ARCH=gfx950   (Slurm)

# An unset image arg stops the build at its FROM:
#   base name (${ROCM_BASE_IMAGE}) should not be blank
# Build base: full ROCm toolchain (compiler, headers, math libs) for source builds.
ARG ROCM_BASE_IMAGE
# Runtime base: plain Ubuntu + AMD apt repo + 13 ROCm runtime packages.  AMD
# publishes only the -full variant of rocm/dev-ubuntu-26.04, so the runtime
# stage starts from Ubuntu instead; the AMD apt signing key is COPYed from the
# base stage.
ARG ROCM_RUNTIME_IMAGE
# The stage runtime-os copies that key from: `base`, or `rocm-apt-fetch` for a
# build that compiles nothing and so has no other use for the full image.
ARG ROCM_APT_FROM

# Empty default for the `pip-wheels` build context, which only the
# runtime-from-wheels stage reads.  A caller that passes
# --build-context pip-wheels=<dir> overrides this stage.
FROM scratch AS pip-wheels

# hadolint ignore=DL3006
FROM ${ROCM_BASE_IMAGE} AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ROCM_VERSION
ARG ROCM_BASE_IMAGE
ARG ROCM_ARCH

# ----- 0. Corporate CA (optional, Zscaler/AMD) --------------------------------
# Pass a CA cert at build time via BuildKit secret to reach HTTPS through a
# corporate proxy.  Omit entirely for builds outside the proxy.
#
#   docker build --secret id=tls_cert,src=/path/to/corp-ca.crt ...
#
RUN --mount=type=secret,id=tls_cert,required=false \
    if [ -f /run/secrets/tls_cert ] && [ -s /run/secrets/tls_cert ]; then \
        cp /run/secrets/tls_cert /usr/local/share/ca-certificates/corp-ca.crt && \
        update-ca-certificates; \
    fi

# ----- 1. System dependencies ------------------------------------------------
# hadolint ignore=SC2015
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libboost-dev \
        libdrm-dev \
        libfmt-dev \
        libibverbs-dev \
        ibverbs-providers \
        libmount-dev \
        libmsgpack-dev \
        libsqlite3-dev \
        ninja-build \
        nvme-cli \
        pkg-config \
        python3-dev \
        bpftrace \
    && update-ca-certificates && \
    # Ubuntu 26.04 packages its own ROCm 7.1 (pulled in by libucx0/libopenmpi40)
    # with the same sonames as ROCm 10.x; it breaks hipIpcGetMemHandle.  Nothing
    # above pulls it in today; the purge guards against a future dependency.
    { DEBIAN_FRONTEND=noninteractive apt-get remove -y --purge \
        libhsa-runtime64-1 libhsakmt1 libamdhip64-7 libamd-comgr3 2>/dev/null || true; }

RUN git config --global advice.detachedHead false

RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

ENV PIP_ROOT_USER_ACTION=ignore \
    PIP_BREAK_SYSTEM_PACKAGES=1

FROM base AS build_triton

# =============================================================================
# Triton source build — source pin matches the vLLM-validated ref.
# =============================================================================
ARG ROCM_ARCH
ARG TRITON_GIT_URL
ARG TRITON_REF
RUN : "${TRITON_GIT_URL:?unset: build through docker/docker-bake.hcl}" \
      "${TRITON_REF:?unset: build through docker/docker-bake.hcl}"

# Triton's [build-system].requires pins cmake==4.0 and --no-isolation enforces it.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --break-system-packages \
      'cmake==4.0' \
      ninja \
      'pybind11>=2.13.1' \
      setuptools

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-venv \
      python3-build

RUN git clone --depth 1 --branch "${TRITON_REF}" "${TRITON_GIT_URL}" /app/triton

# TRITON_OFFLINE_BUILD=1 to avoid downloading large CUDA binaries
# However, it depends on a specific version of llvm+mlir that is not available in Ubuntu 26.04
WORKDIR /app/triton

RUN --mount=type=cache,target=/root/.triton/ \
    TRITON_APPEND_CMAKE_ARGS="-DCMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH=FALSE" \
       python3 -m build --wheel --no-isolation --outdir=/app/triton/dist

WORKDIR /

# Slim the wheel itself (unpack, edit, repack so RECORD stays valid), so the
# `wheels` target and every downstream install get the small one; kept apart
# from the build above so a change here does not recompile Triton.
#   - libtriton.so/libproton.so carry ~700 MB of DWARF and .symtab;
#   - backends/nvidia/{bin,lib,include} (ptxas, nvdisasm, cupti, CUDA headers)
#     and the instrumentation test library are never used on ROCm.  The nvidia
#     backend's .py files stay: `import triton` imports triton.backends.nvidia.
# hadolint ignore=SC2086
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install wheel && \
    python3 -m wheel unpack -d /tmp/whl /app/triton/dist/triton-*.whl && \
    d=$(echo /tmp/whl/triton-*) && \
    /opt/rocm/core/lib/llvm/bin/llvm-strip --strip-unneeded \
        "${d}/triton/_C/libtriton.so" "${d}/triton/_C/libproton.so" && \
    rm -rf "${d}/triton/backends/nvidia/bin" \
           "${d}/triton/backends/nvidia/lib" \
           "${d}/triton/backends/nvidia/include" \
           "${d}/triton/instrumentation/libGPUInstrumentationTestLib.so" && \
    if /opt/rocm/core/lib/llvm/bin/llvm-readelf -S "${d}/triton/_C/libtriton.so" | grep -F '.debug_'; then \
        echo "ERROR: libtriton.so still carries debug sections" >&2; exit 1; \
    fi && \
    test -f "${d}/triton/backends/nvidia/compiler.py" && \
    rm /app/triton/dist/triton-*.whl && \
    python3 -m wheel pack -d /app/triton/dist "${d}" && \
    rm -rf /tmp/whl && \
    ls -l /app/triton/dist

# =============================================================================
# PyTorch source build — no upstream wheel for this SHA; source pin is authoritative.
# =============================================================================
FROM base AS build_pytorch

ARG ROCM_ARCH
ARG PYTORCH_GIT_URL
# Cloned then checked out, so a SHA, branch or tag all work.
ARG PYTORCH_REF
# Not read by the build.  Declared because bake's base target carries it: the
# release notes and aic-bake-env.sh read the torch series from that target.
ARG PYTORCH_SERIES
RUN : "${PYTORCH_GIT_URL:?unset: build through docker/docker-bake.hcl}" \
      "${PYTORCH_REF:?unset: build through docker/docker-bake.hcl}" \
      "${ROCM_VERSION:?unset: build through docker/docker-bake.hcl}"

# torch 2.13 emits float->__bf16 libcalls absent from gcc-11; build with gcc-13.
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
        g++-13 \
        gcc-13 \
        liblzma-dev \
        libnuma-dev \
        patchelf \
        pkg-config \
        python3-venv && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-13

# Filter out unnecessary submodules, including submodules' submodules
# After clone, remove the sections from .gitmodules, which is used during CMake
# config time to make sure that they are initialised.
#
# Note: ignore git config errors as some references are not in pytorch
# .gitmodules file
#
# Ignored warnings:
# > Quote this to prevent word splitting
# All variable expansions are quoted
# > In POSIX sh, array references are undefined
# This stage uses bash as shell, see SHELL step above (coming from base stage)
# hadolint ignore=SC2046,SC3054
RUN export UNUSED_SUBMODULES=(3rdparty/composable_kernel \
                              android/libs/fbjni \
                              external/cpuinfo \
                              external/cutlass \
                              external/googletest \
                              external/hipify_torch \
                              libkineto/third_party/googletest \
                              third_party/NVTX \
                              third_party/VulkanMemoryAllocator \
                              third_party/cudnn_frontend \
                              third_party/cutlass \
                              third_party/flash-attention \
                              third_party/gemmlowp/gemmlowp \
                              third_party/googletest \
                              third_party/ittapi \
                              third_party/kleidiai \
                              third_party/mimalloc) && \
    git clone --filter=blob:none --no-checkout "${PYTORCH_GIT_URL}" /app/pytorch && \
    git -C /app/pytorch checkout "${PYTORCH_REF}" && \
    git -C /app/pytorch $(for m in "${UNUSED_SUBMODULES[@]}"; do printf -- ' -c submodule.%s.update=none' "$m"; done) \
        submodule update --init --depth 1 --recursive && \
    for m in "${UNUSED_SUBMODULES[@]}"; do \
      git config -f /app/pytorch/.gitmodules --remove-section submodule."$m" || true; \
    done

WORKDIR /app/pytorch
# PyTorch pins exact versions of pip, wheel and packaging.  Where the image's
# copies are Debian-owned (no RECORD) pip cannot upgrade them in place, so
# --ignore-installed puts pip-owned copies in /usr/local, which shadows
# /usr/lib/python3/dist-packages.  The check stops the build if that flag left
# two versions of one distribution in /usr/local.  `build[uv]` is the only
# thing that would bring uv in; plain `build` is all `python3 -m build` needs.
RUN --mount=type=cache,target=/root/.cache/pip \
    sed -i 's/^build\[uv\]/build/' requirements.txt requirements-build.txt && \
    python3 -m pip install --ignore-installed \
      -r requirements.txt \
      -r requirements-build.txt \
      auditwheel==6.4.2 && \
    python3 -c 'import collections, importlib.metadata as m, re, sysconfig; \
    n = collections.Counter(re.sub("[-_.]+", "-", d.metadata["Name"] or "").lower() for d in m.distributions(path=[sysconfig.get_path("purelib")])); \
    dup = sorted(k for k, v in n.items() if v > 1); \
    assert not dup, f"two versions installed side by side: {dup}"'

COPY patches/pytorch/ /app/patches/pytorch/

# Apply patches to pytorch and its submodules before the build.
# Uses `patch -p1` (not `git apply`) so patches to submodule files work without
# git submodule boundary issues.
# hadolint ignore=SC2061,SC3044
RUN if compgen -G "/app/patches/pytorch/*.patch" > /dev/null; then \
        for p in /app/patches/pytorch/*.patch; do \
            echo "Applying pytorch patch: $p"; \
            patch -p1 -d /app/pytorch < "$p" || { \
                echo "ERROR: patch does not apply cleanly: $p" >&2; exit 1; }; \
        done; \
    fi

# The wheel backend (scikit-build-core) rejects pyproject.toml unless every
# project.license-files entry exists; the submodules skipped above take theirs
# with them, so stand in a placeholder for each missing one.
RUN python3 -c 'import pathlib, tomllib; \
    missing = [p for p in map(pathlib.Path, tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["license-files"]) if not p.exists()]; \
    [(p.parent.mkdir(parents=True, exist_ok=True), p.write_text("Submodule not checked out for this build.\n")) for p in missing]; \
    print("placeholder license files:", *missing, sep="\n  ")'

ENV PYTORCH_ROCM_ARCH=${ROCM_ARCH}

# Cap compile parallelism.  Without this the torch_cpu/ATen compile fans out to
# every core; the heavier translation units peak well above 1 GB each, so on a
# host with less RAM than ~2 GB per core this exhausts memory and can hard-hang
# the machine.  Empty means "all cores", matching docker/lmcache/Dockerfile.
ARG BUILD_JOBS

# hadolint ignore=SC2086,SC2015
RUN --mount=type=cache,target=/root/.cache/ccache \
    python3 tools/amd_build/build_amd.py && \
    CCACHE_DIR=/root/.cache/ccache CCACHE_MAXSIZE=25G \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    MAX_JOBS="${BUILD_JOBS:-$(nproc)}" \
    CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS:-$(nproc)}" \
    CMAKE_PREFIX_PATH=$(python3 -c 'import sys; print(sys.prefix)') \
    PYTORCH_ROCM_ARCH=$(echo "${PYTORCH_ROCM_ARCH}" | tr ',' ' ') \
    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    USE_ITT=0 \
    BUILD_TEST=0 \
    PYTHON_SIX_SOURCE_DIR=$(python3 -c "import sys; v=sys.version_info; print(f'/usr/local/lib/python{v.major}.{v.minor}/dist-packages/')") \
        python3 -m build --wheel --no-isolation --outdir=/app/pytorch/dist_tmp && \
    _mm=$(echo "${ROCM_VERSION}" | cut -d. -f1-2) && \
    GPU_ARCH_TYPE=rocm \
      DESIRED_CUDA=rocm${ROCM_VERSION} \
      GPU_ARCH_VERSION=${ROCM_VERSION} \
      PYTORCH_ROCM_ARCH=$(echo "${PYTORCH_ROCM_ARCH}" | tr ',' ' ') \
      ROCM_HOME=/opt/rocm/core-${_mm} \
      python3 .ci/wheel/linux/repair_wheel.py /app/pytorch/dist_tmp/ /app/pytorch/dist/ && \
    { find /app/pytorch -mindepth 1 -maxdepth 1 ! -name 'dist' -exec rm -rf {} + || \
      echo "WARNING: PyTorch source cleanup failed; see find's error above" >&2; }
WORKDIR /

# Slim the wheel as for Triton, in its own layer so a change here does not
# rebuild PyTorch: libtorch_{cpu,hip,python}.so ship unstripped (.symtab only,
# no DWARF) and torch/bin/protoc* is a build tool.
# hadolint ignore=SC2086
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install wheel && \
    python3 -m wheel unpack -d /tmp/whl /app/pytorch/dist/torch-*.whl && \
    d=$(echo /tmp/whl/torch-*) && \
    /opt/rocm/core/lib/llvm/bin/llvm-strip --strip-unneeded \
        "${d}/torch/lib/libtorch_cpu.so" \
        "${d}/torch/lib/libtorch_hip.so" \
        "${d}/torch/lib/libtorch_python.so" && \
    rm -f "${d}"/torch/bin/protoc* && \
    rm /app/pytorch/dist/torch-*.whl && \
    python3 -m wheel pack -d /app/pytorch/dist "${d}" && \
    rm -rf /tmp/whl && \
    ls -l /app/pytorch/dist

# =============================================================================
# torchvision source build — must compile against our source-built PyTorch.
# pytorch/vision has no v0.30 tag yet (the series that pairs with torch 2.15),
# so TORCHVISION_REF is the newest release tag.
# =============================================================================
FROM build_pytorch AS build_torchvision

ARG ROCM_ARCH
# Re-declared: ARG does not carry across a FROM.
ARG BUILD_JOBS
ARG TORCHVISION_GIT_URL
# Shallow clone, so this must be a branch or tag, not a SHA.
ARG TORCHVISION_REF
RUN : "${TORCHVISION_GIT_URL:?unset: build through docker/docker-bake.hcl}" \
      "${TORCHVISION_REF:?unset: build through docker/docker-bake.hcl}" \
      "${ROCM_ARCH:?empty: torchvision compiles its HIP ops per arch, set ROCM_ARCH}"

RUN git clone --depth 1 --branch "${TORCHVISION_REF}" "${TORCHVISION_GIT_URL}" /app/vision

WORKDIR /app/vision
# FORCE_CUDA=1 (as TheRock builds it): a BuildKit RUN has no GPU, so setup.py
# would otherwise skip the HIP ops and link each hipified twin beside its
# original (pytorch/vision#9740).  The last line fails the build unless the
# wheel carries HIP device code for every arch in ROCM_ARCH.
# hadolint ignore=SC2086,DL3008
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libnuma-dev && \
    pip3 install setuptools wheel packaging /app/pytorch/dist/*.whl && \
    USE_ROCM=1 USE_CUDA=0 \
    FORCE_CUDA=1 TORCHVISION_USE_NVJPEG=0 \
    PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
    CCACHE_DIR=/root/.cache/ccache CCACHE_MAXSIZE=25G \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    MAX_JOBS="${BUILD_JOBS:-$(nproc)}" \
    CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS:-$(nproc)}" \
        python3 setup.py bdist_wheel --dist-dir=/app/vision/dist && \
    python3 -c "import glob, zipfile; so = zipfile.ZipFile(glob.glob('/app/vision/dist/torchvision-*.whl')[0]).read('torchvision/_C_stable.so'); missing = [a for a in '${ROCM_ARCH}'.split(';') if ('amdhsa--' + a).encode() not in so]; assert b'hipLaunchKernel' in so and not missing, f'torchvision HIP ops missing for {missing}'"

WORKDIR /

# =============================================================================
# build_nixl — NIXL source build using the ROCm-full toolchain.
# Output (/opt/nixl, /opt/rocnixl-ucx) is copied to the slim runtime stage so
# the runtime image never needs cmake, clang, or ROCm headers.
# =============================================================================
FROM base AS build_nixl

ARG UCX_GIT_URL
ARG UCX_REF
ARG BUILD_JOBS
ARG AIC_UCX_FAST
RUN : "${UCX_GIT_URL:?unset: build through docker/docker-bake.hcl}" \
      "${UCX_REF:?unset: build through docker/docker-bake.hcl}"

# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libaio-dev \
        libibverbs-dev \
        libltdl-dev \
        libnuma-dev \
        librdmacm-dev \
        libtool \
        liburing-dev \
        pkg-config \
        rdma-core

# numpy is required by NIXL's Python bindings at install time.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install numpy numba

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install meson meson-python ninja patchelf pybind11 tomlkit

# UCX (NIXL's transport) in its own layer: a NIXL ref or patch change must not
# rebuild it.  The source and build trees are removed in the layer they are
# created in.
# hadolint ignore=SC3054,SC3030,SC3024
RUN set -eu; \
    git clone --depth 1 --branch "${UCX_REF}" "${UCX_GIT_URL}" /tmp/ucx-src; \
    cd /tmp/ucx-src; \
    ./autogen.sh; \
    mkdir build; \
    cd build; \
    ucx_configure_flags=( \
        --prefix=/opt/rocnixl-ucx \
        --enable-shared \
        --disable-static \
        --disable-doxygen-doc \
        --enable-devel-headers \
        --with-rocm=/opt/rocm \
        --with-verbs \
        --with-dm \
        --enable-mt \
    ); \
    # AIC_UCX_FAST=1 trades UCX features for compile time (dist-build-fast).
    if [ "${AIC_UCX_FAST:-0}" = "1" ]; then \
        ucx_configure_flags+=( \
            --disable-logging \
            --disable-debug \
            --disable-assertions \
            --disable-params-check \
            --without-knem \
            --without-xpmem \
            --without-ugni \
            --without-java \
        ); \
    else \
        ucx_configure_flags+=(--enable-optimizations); \
    fi; \
    ../configure "${ucx_configure_flags[@]}"; \
    make -j"${BUILD_JOBS:-$(nproc)}"; \
    make install; \
    # UCX is built with -g.  Stripped here, before NIXL links and bundles it.
    # Non-ELF executables (scripts) make llvm-strip fail, hence the `|| true`;
    # the check below is what catches a strip that did nothing.
    find /opt/rocnixl-ucx -type f \( -name '*.so' -o -name '*.so.*' -o -perm -111 \) \
        -exec sh -c '/opt/rocm/core/lib/llvm/bin/llvm-strip --strip-unneeded "$1" 2>/dev/null || true' _ {} \; ; \
    if /opt/rocm/core/lib/llvm/bin/llvm-readelf -S /opt/rocnixl-ucx/lib/libucp.so | grep -F '.debug_'; then \
        echo "ERROR: libucp.so still carries debug sections" >&2; exit 1; \
    fi; \
    ldconfig; \
    /opt/rocnixl-ucx/bin/ucx_info -v; \
    rm -rf /tmp/ucx-src

# Declared after the UCX layer so they stay out of its cache key.
ARG NIXL_GIT_URL
ARG NIXL_REF
RUN : "${NIXL_GIT_URL:?unset: build through docker/docker-bake.hcl}" \
      "${NIXL_REF:?unset: build through docker/docker-bake.hcl}"

COPY docker/scripts/clone-nixl.sh /tmp/nixl-scripts/clone-nixl.sh
COPY patches/nixl/ /tmp/nixl-patches/
# NIXL_REQUIRE_ROCM=1 makes clone-nixl.sh reject a ref without the wheel_variant
# meson option.
# hadolint ignore=SC3054,SC3044
RUN chmod +x /tmp/nixl-scripts/clone-nixl.sh && \
    NIXL_GIT_URL="${NIXL_GIT_URL}" NIXL_REF="${NIXL_REF}" NIXL_REQUIRE_ROCM=1 \
        NIXL_DEST=/tmp/nixl /tmp/nixl-scripts/clone-nixl.sh && \
    shopt -s nullglob && \
    nixl_patches=(/tmp/nixl-patches/*.patch) && \
    if [ "${#nixl_patches[@]}" -eq 0 ]; then \
        echo "ERROR: no patches found in /tmp/nixl-patches/" >&2; exit 1; \
    fi && \
    for patch in "${nixl_patches[@]}"; do \
        if git -C /tmp/nixl apply --verbose "$patch"; then \
            echo "[nixl] applied $(basename "$patch")"; \
        else \
            echo "ERROR: nixl patch does not apply cleanly: $patch" >&2; exit 1; \
        fi; \
    done && \
    rm -rf /tmp/nixl-patches /tmp/nixl-scripts

# NIXL build, mandatory-plugin checks and the nixl-rocm wheel.  The patched
# checkout has native HIP + the AIS_MT plugin (nixl-rocm-ais-mt.patch).
# hadolint ignore=SC3054,SC3030,SC3060
RUN --mount=type=cache,target=/root/.cache/pip \
    set -eu; \
    jobs="${BUILD_JOBS:-$(nproc)}"; \
    prefix=/opt/nixl; \
    libdir="${prefix}/lib/x86_64-linux-gnu"; \
    py_site="${prefix}/lib/python3/dist-packages"; \
    export PKG_CONFIG_PATH="/opt/rocnixl-ucx/lib/pkgconfig:${PKG_CONFIG_PATH:-}"; \
    export LD_LIBRARY_PATH="${libdir}:${prefix}/lib:/opt/rocnixl-ucx/lib:${LD_LIBRARY_PATH:-}"; \
    meson_args=( \
        -Dwheel_variant=rocm \
        -Drocm_ais_path=/opt/rocm \
        -Ducx_path=/opt/rocnixl-ucx \
        -Ddisable_gds_backend=true \
        -Dbuild_tests=false \
        -Dbuild_examples=false \
        -Ddisable_plugins=OBJ,AZURE_BLOB \
    ); \
    cd /tmp/nixl; \
    meson setup build --prefix="${prefix}" "${meson_args[@]}"; \
    ninja -C build -j"${jobs}"; \
    ninja -C build install; \
    ldconfig; \
    python3 -m pip uninstall -y nixl nixl-cu12 nixl-cu13 2>/dev/null || true; \
    # Meson installs the bindings as nixl_rocm.  The upstream meta wheel depends
    # on nixl-rocm on PyPI and its __init__.py only probes CUDA backends, so pip
    # install of the meta wheel or the repo-root pyproject.toml is wrong on
    # ROCm; nixl_rocm_shim.py (copied in below) provides `import nixl` instead.
    if [ ! -d "${py_site}/nixl_rocm" ]; then \
        echo "ERROR: ${py_site}/nixl_rocm not found after meson install" >&2; exit 1; \
    fi; \
    plugin_dir="${libdir}/plugins"; \
    if [ ! -d "${plugin_dir}" ]; then plugin_dir="${prefix}/lib/nixl/plugins"; fi; \
    # AIS_MT (LMCache's l2-adapter backend) is mandatory; its absence means
    # hipFile was not found at NIXL build time.
    ais_plugin="$(find build -name libplugin_AIS_MT.so -print -quit)"; \
    if [ -z "${ais_plugin}" ]; then \
        echo "ERROR: libplugin_AIS_MT.so not built (hipFile missing at NIXL build time?)" >&2; exit 1; \
    fi; \
    echo "PASS: ${ais_plugin}"; \
    # The Prometheus telemetry exporter is mandatory too.  NIXL builds it only
    # when its prometheus-cpp CMake subproject resolves.
    prom_plugin="$(find build -name libtelemetry_exporter_prometheus.so -print -quit)"; \
    if [ -z "${prom_plugin}" ]; then \
        echo "ERROR: prometheus telemetry exporter plugin NOT built (prometheus-cpp subproject unavailable at NIXL build time); this build is not permitted without it" >&2; exit 1; \
    fi; \
    echo "PASS: prometheus telemetry exporter plugin: ${prom_plugin}"; \
    # The plugin loads the prometheus-cpp/civetweb shared libs of the CMake
    # subproject at runtime.  NIXL refs whose `ninja install` leaves them in the
    # build tree need them copied to a dir on the runtime LD_LIBRARY_PATH
    # (v1.5.2 installs them itself, as libcore/libpull).
    install -Dm0755 "${prom_plugin}" "${plugin_dir}/$(basename "${prom_plugin}")"; \
    find build \( -name 'libprometheus-cpp*.so*' -o -name 'libcivetweb*.so*' \) \
        -exec install -Dm0755 -t "${libdir}" {} +; \
    ldconfig; \
    # A plugin that cannot resolve its shared libs is as good as absent.
    if ldd "${plugin_dir}/$(basename "${prom_plugin}")" | grep "not found" >&2; then \
        echo "ERROR: prometheus telemetry plugin has unresolved shared libs" >&2; exit 1; \
    fi; \
    find "${prefix}" -type f \( -name '*.so' -o -name '*.so.*' -o -perm -111 \) \
        -exec sh -c '/opt/rocm/core/lib/llvm/bin/llvm-strip --strip-unneeded "$1" 2>/dev/null || true' _ {} \; ; \
    # Wheel: same meson args minus --prefix, which meson-python manages.  It is
    # named nixl-rocm (upstream's pyproject defaults to nixl-cu12) and bundles
    # libnixl and the NIXL plugins; UCX and the ROCm runtime stay external.
    # CXX must not be hipcc here: meson would use it as the host C++ compiler
    # and hipcc fails meson's compiler sanity check.
    if [ -x contrib/tomlutil.py ]; then \
        ./contrib/tomlutil.py --wheel-name nixl-rocm pyproject.toml; \
    fi; \
    mkdir -p /app/nixl/dist; \
    # No --check-build-dependencies: upstream's [build-system].requires pins
    # torch==2.11.*, which must not be enforced.
    env -u CXX ROCM_PATH=/opt/rocm \
        python3 -m pip wheel --no-build-isolation --no-deps --wheel-dir /app/nixl/dist \
            "${meson_args[@]/#/-Csetup-args=}" "-Ccompile-args=-j${jobs}" .; \
    nixl_whl="$(find /app/nixl/dist -maxdepth 1 -name 'nixl*rocm*.whl' -print -quit)"; \
    if [ -z "${nixl_whl}" ]; then \
        echo "ERROR: no nixl*rocm*.whl in /app/nixl/dist" >&2; exit 1; \
    fi; \
    echo "PASS: NIXL wheel built -> ${nixl_whl}"; \
    rm -rf /tmp/nixl

# meson-python installs a versioned library without the soname link the loader
# asks for (a wheel cannot hold a symlink).  The wheel then carries
# prometheus-cpp as libcore.so.1.3.0 and libpull.so.1.3.0 while the Prometheus
# telemetry exporters need libcore.so.1.3 and libpull.so.1.3: from a pip install
# they do not load, and creating an agent fails once
# NIXL_TELEMETRY_EXPORTER=prometheus selects one.  Store each such library
# under its soname; nothing refers to the full name.  /opt/nixl is unaffected:
# `ninja install` made the links there.
#
# The check installs the result on its own and creates an agent the way the
# lmcache server does.  LD_LIBRARY_PATH is emptied and NIXL_PLUGIN_DIR points
# into the wheel, so its libraries are the only ones on offer; the assert on
# /proc/self/maps fails the build should one come from /opt/nixl all the same.
#
# In its own layer so that a change here does not rebuild NIXL.
RUN --mount=type=cache,target=/root/.cache/pip \
    set -eu; \
    nixl_whl="$(find /app/nixl/dist -maxdepth 1 -name 'nixl*rocm*.whl' -print -quit)"; \
    python3 -m pip install wheel; \
    python3 -m wheel unpack -d /tmp/nixl-wheel "${nixl_whl}"; \
    find /tmp/nixl-wheel -type f -name '*.so.*' | while read -r lib; do \
        soname="$(patchelf --print-soname "${lib}")" || { \
            echo "ERROR: ${lib} is not an ELF library" >&2; exit 1; }; \
        if [ -n "${soname}" ] && [ "${soname}" != "$(basename "${lib}")" ]; then \
            mv "${lib}" "$(dirname "${lib}")/${soname}"; \
            echo "nixl wheel: $(basename "${lib}") stored as ${soname}"; \
        fi; \
    done; \
    rm "${nixl_whl}"; \
    python3 -m wheel pack -d /app/nixl/dist /tmp/nixl-wheel/*; \
    rm -rf /tmp/nixl-wheel; \
    test -f "${nixl_whl}"; \
    python3 -m pip install --no-deps --no-compile --target /tmp/nixl-wheel-check "${nixl_whl}"; \
    env LD_LIBRARY_PATH= PYTHONPATH=/tmp/nixl-wheel-check \
        NIXL_PLUGIN_DIR=/tmp/nixl-wheel-check/.nixl_rocm.mesonpy.libs/plugins \
        NIXL_TELEMETRY_ENABLE=y NIXL_TELEMETRY_EXPORTER=prometheus NIXL_TELEMETRY_PROMETHEUS_PORT=19090 \
        python3 -c 'import nixl_rocm; from nixl_rocm._api import nixl_agent, nixl_agent_config; \
    assert nixl_rocm.__file__.startswith("/tmp/nixl-wheel-check/"), nixl_rocm.__file__; \
    a = nixl_agent("wheel-check", nixl_agent_config(backends=[])); a.create_backend("POSIX"); \
    stray = sorted({line.split()[-1] for line in open("/proc/self/maps") if "/opt/nixl/" in line}); \
    assert not stray, f"loaded from /opt/nixl, not from the wheel: {stray}"; \
    print("PASS: the nixl_rocm wheel stands alone:", sorted(a.backends))'; \
    rm -rf /tmp/nixl-wheel-check

COPY docker/base/nixl_rocm_shim.py /opt/nixl/lib/python3/dist-packages/nixl/__init__.py
RUN PYTHONPATH=/opt/nixl/lib/python3/dist-packages \
    NIXL_PLUGIN_DIR=/opt/nixl/lib/x86_64-linux-gnu/plugins \
    LD_LIBRARY_PATH=/opt/nixl/lib/x86_64-linux-gnu:/opt/nixl/lib:/opt/rocnixl-ucx/lib \
        python3 -c "import nixl; print('nixl import OK:', nixl.__file__)"

# =============================================================================
# wheels — scratch image holding just the nixl_rocm pip wheel (--target wheels;
# the `wheels-nixl` bake target).
#
# NIXL is built ONCE here, in aic-base, and both aic-vllm and aic-lmcache
# inherit the result; there is exactly one nixl_rocm wheel for the whole stack,
# not one per image.  The `runtime` stage below copies only /opt/nixl and
# /opt/rocnixl-ucx out of build_nixl, so the wheel never reaches the image.
# Must stay above runtime, the default target.
# =============================================================================
FROM scratch AS wheels
COPY --from=build_nixl /app/nixl/dist/*.whl /
COPY --from=build_triton /app/triton/dist/*.whl /
COPY --from=build_pytorch /app/pytorch/dist/*.whl /
COPY --from=build_torchvision /app/vision/dist/*.whl /


# =============================================================================
# build_clr_patch — patch libamdhip64.so to fix IPC event signal ring overflow.
#
# ROCm/rocm-systems PRs #11511 and #11749 fix IPCEventEmulated::streamWait to
# wrap read_index with % IPC_SIGNALS_PER_EVENT (ring size = 32).  Without this,
# the 33rd cross-process use of the same IPC event handle reads past the end of
# the signal array, causing hipErrorInvalidValue.  The PRs are open on 'develop'
# and not yet in any released ROCm build, so we patch and rebuild here.
# =============================================================================
FROM base AS build_clr_patch

ARG CLR_GIT_URL
ARG CLR_REF
ARG BUILD_JOBS
RUN : "${CLR_GIT_URL:?unset: build through docker/docker-bake.hcl}" \
      "${CLR_REF:?unset: build through docker/docker-bake.hcl}"

# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cmake git ninja-build python3-dev

COPY patches/hip-ipc-signal-slot-index-fix.patch /tmp/hip-ipc-signal-slot-index-fix.patch

RUN pip3 install --break-system-packages CppHeaderParser && \
    git clone --filter=blob:none --no-checkout --depth 1 --branch "${CLR_REF}" "${CLR_GIT_URL}" /app/rocm-systems

WORKDIR /app/rocm-systems
RUN git sparse-checkout init --cone && \
    git sparse-checkout set projects/clr projects/hip projects/hrr && \
    git checkout HEAD && \
    patch -p1 < /tmp/hip-ipc-signal-slot-index-fix.patch
WORKDIR /

# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libgl-dev

RUN cmake -G Ninja \
        -S /app/rocm-systems/projects/clr \
        -B /app/clr-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=/opt/rocm/core/lib/llvm/bin/clang \
        -DCMAKE_CXX_COMPILER=/opt/rocm/core/lib/llvm/bin/clang++ \
        -DHIP_COMMON_DIR=/app/rocm-systems/projects/hip \
        -DHIP_PLATFORM=amd \
        -DROCM_PATH=/opt/rocm/core \
        -DCMAKE_PREFIX_PATH=/opt/rocm/core \
        -DCLR_BUILD_HIP=ON \
        -DCLR_BUILD_OCL=OFF \
        -DROCM_KPACK_ENABLED=ON \
        -D__HIP_ENABLE_PCH=ON && \
    cmake --build /app/clr-build --target amdhip64 \
        --parallel "${BUILD_JOBS:-$(nproc)}" && \
    ls /app/clr-build/hipamd/lib/libamdhip64.so* && \
    echo "libamdhip64.so built OK"

# =============================================================================
# rocm-apt — the ROCm apt signing key and source list runtime-os installs from.
#
# ROCM_APT_FROM names the stage they come from:
#   base            the full ROCm image.  A source build has it anyway, and the
#                   repo URL then follows that image (ROCm 10.0 moved it).
#   rocm-apt-fetch  AMD's published key and ROCM_APT_REPO_URL.  For
#                   runtime-from-wheels, which needs nothing else from the full
#                   image: ~10 GB is too much to pull for two small files.
# =============================================================================
# hadolint ignore=DL3006
FROM ${ROCM_RUNTIME_IMAGE} AS rocm-apt-fetch

ARG ROCM_APT_KEYRING_URL
ARG ROCM_APT_REPO_URL

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN : "${ROCM_APT_KEYRING_URL:?unset: build through docker/docker-bake.hcl}" \
      "${ROCM_APT_REPO_URL:?unset: build through docker/docker-bake.hcl}"

# The key is published ASCII-armored and apt wants a binary keyring under a
# .gpg name.  The armor body is base64, so it is decoded here without
# installing gnupg; a key that is already binary is taken as is.  apt verifies
# the result when runtime-os runs `apt-get update`.
# hadolint ignore=DL3008
RUN --mount=type=secret,id=tls_cert,required=false \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl && \
    if [ -s /run/secrets/tls_cert ]; then \
        cp /run/secrets/tls_cert /usr/local/share/ca-certificates/corp-ca.crt && \
        update-ca-certificates; \
    fi && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL --retry 3 -o /tmp/rocm.key "${ROCM_APT_KEYRING_URL}" && \
    if grep -q '^-----BEGIN PGP PUBLIC KEY BLOCK-----' /tmp/rocm.key; then \
        awk '/^-----BEGIN /{body=1; next} /^-----END /{body=0} body && NF && !/:/ && !/^=/' /tmp/rocm.key \
            | base64 -d > /etc/apt/keyrings/amdrocm.gpg; \
    else \
        cp /tmp/rocm.key /etc/apt/keyrings/amdrocm.gpg; \
    fi && \
    test -s /etc/apt/keyrings/amdrocm.gpg && \
    printf 'deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] %s stable main\n' "${ROCM_APT_REPO_URL}" \
        > /etc/apt/sources.list.d/rocm.list

# hadolint ignore=DL3006
FROM ${ROCM_APT_FROM} AS rocm-apt

# =============================================================================
# runtime — plain ROCm image + 13 ROCm runtime packages + torch + NIXL.
#
# Starts FROM ubuntu:26.04 instead of the -full ROCm image (~18 GB).  AMD
# stopped publishing a lightweight runtime image (no -full suffix) for Ubuntu
# 26.04; we build from plain Ubuntu and add only the 13
# ROCm packages that torch actually dlopens at process start (~2 GB), confirmed
# by /proc/self/maps audit with a live GPU.  ~15 GB of -full content excluded.
#
# ROCm runtime packages installed (~2 GB total):
#   amdrocm-runtime     HIP + HSA + comgr + hipRTC        (47 MB)
#   amdrocm-base        rocm-core + profiler-register      (5 MB)
#   amdrocm-profiler-base  profiler SDK + rocTX + AQL      (122 MB)
#   amdrocm-dnn-host    MIOpen                             (463 MB)
#   amdrocm-blas-host   rocBLAS + hipBLAS + hipBLASLt + rocRoller  (273 MB)
#   amdrocm-blas<mm>-<arch>  per-arch hipBLASLt TensileLibrary device libs,
#                        one per entry of ROCM_ARCH — torch's linear GEMM
#                        fails with HIPBLAS_STATUS_INVALID_VALUE without them
#   amdrocm-fft-host    hipFFT + rocFFT                    (13 MB)
#   amdrocm-rand-host   hipRAND + rocRAND                  (52 MB)
#   amdrocm-rccl-host   RCCL                               (4 MB)
#   amdrocm-solver-host hipSOLVER + rocSOLVER              (29 MB)
#   amdrocm-sparse-host hipSPARSE + rocSPARSE              (74 MB)
#   amdrocm-amdsmi      AMD SMI lib + Python bindings      (36 MB)
#   amdrocm-llvm        libLLVM + libclang (JIT needed)    (839 MB)
#   amdrocm-math-common rocm-openblas + host-math          (97 MB)
#   amdrocm-hipfile     hipFile I/O + ais-check CLI tool   (~5 MB)
#
# Downstream Dockerfiles (docker/vllm, docker/lmcache) use this stage via
# --build-context base=oci-layout://... so they get a fully working torch env.
#
# Two stages: runtime-os is everything below the Python stack, and `runtime`
# (the last stage, so the default target) adds the Python stack from the build
# stages.  runtime-from-wheels sits on the same runtime-os.
# =============================================================================
# hadolint ignore=DL3006
FROM ${ROCM_RUNTIME_IMAGE} AS runtime-os

ARG ROCM_VERSION
ARG ROCM_BASE_IMAGE
ARG ROCM_RUNTIME_IMAGE
ARG ROCM_ARCH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN : "${ROCM_VERSION:?unset: build through docker/docker-bake.hcl}"

# Corporate CA cert (needed for pip at runtime if behind a corporate proxy).
# ubuntu:26.04 has no ca-certificates yet; the next RUN installs it and runs
# update-ca-certificates, which picks this file up.
RUN --mount=type=secret,id=tls_cert,required=false \
    if [ -f /run/secrets/tls_cert ] && [ -s /run/secrets/tls_cert ]; then \
        mkdir -p /usr/local/share/ca-certificates && \
        cp /run/secrets/tls_cert /usr/local/share/ca-certificates/corp-ca.crt; \
    fi

# AMD stopped publishing the lightweight runtime image (no -full suffix) for
# Ubuntu 26.04.  Use ubuntu:26.04 as the base and add the
# AMD apt repo using the signing key COPYed from the build-time base stage
# (which already has it from the -full image — no network fetch needed).
# Version-specific package names are computed from ROCM_VERSION (10.2.0 → "10.2").
# hadolint ignore=DL3008
# Step 1: ca-certificates must be installed before adding the HTTPS AMD repo,
# otherwise apt-get update cannot verify the TLS connection and silently skips
# the AMD packages.
# BLAS_CACHE_BUST invalidates this layer whenever its value changes, to
# reinstall the ROCm runtime packages on demand.
ARG BLAS_CACHE_BUST
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates && \
    update-ca-certificates

# Step 2: add AMD ROCm apt repo (requires ca-certificates for HTTPS), then
# install the 13 runtime packages + system deps.  Tools among those deps serve
# what runs inside the containers:
#   curl          HTTP probes, scrapes and compose healthchecks
#   make          AITER's JIT builds some kernels (paged attention) with make
#   python3-venv  ensurepip, for `python3 -m venv`
#   python3-pip   the one pip for the stack: aic-vllm and aic-lmcache inherit it
#                 for their own installs, and it stays in the images (the
#                 rocjitsu VM test and the accuracy job pip-install in them)
# Key and sources list copied from the rocm-apt stage, which a source build
# points at the build-time base stage so that the repo URL tracks the base
# image automatically (ROCm 10.0 changed the URL).
#
# amdrocm-runtime-dev carries the HIP development headers AITER's JIT needs
# (hip/hip_runtime.h and friends), but Depends on amdrocm-llvm-dev: 2 GB of
# LLVM static archives and C++ API headers nothing here links.  An empty
# package of the same name and version, installed first and held, satisfies
# that dependency instead, here and for the amdrocm-*-dev packages aic-vllm
# installs on top.
#
# Everything this RUN deletes or strips was installed by it: done in a later
# RUN, the bytes would stay in this layer.  dpkg still lists the files.
#   - LLVM: the flang and MLIR toolchains, clang-tidy and the refactoring
#     tools, the tblgen tools, and compiler-rt's sanitizer, fuzzer, profile and
#     xray runtimes (-fsanitize and -fprofile-generate no longer link).
#     clang, lld, libLLVM, libclang-cpp, builtins and crtbegin/crtend stay.
#   - rocprofiler Python bindings for interpreters other than the image's,
#     static archives, test and sample trees, the VA-API video driver.
#   - symbol tables of the largest ROCm libraries (--strip-unneeded keeps
#     .dynsym).  libhsa-runtime64 and libhipfile are left alone so uprobe
#     tracing of the HSA and hipFile paths can still resolve their symbols
#     (hsa-snoop itself uses a kernel kprobe); libamdhip64 is not stripped
#     either, though it ships without a .symtab.
#   - the packaged libamdhip64 is emptied: the patched build is copied over it
#     in a RUN of its own below, so this layer need not carry the original and
#     a CLR_REF or patch change does not re-run this RUN.
COPY --from=rocm-apt /etc/apt/keyrings/amdrocm.gpg /etc/apt/keyrings/amdrocm.gpg
COPY --from=rocm-apt /etc/apt/sources.list.d/rocm.list /etc/apt/sources.list.d/amdrocm.list
# hadolint ignore=SC2046,SC2086,DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    echo "blas cache bust: ${BLAS_CACHE_BUST}" && \
    apt-get update && \
    _mm=$(echo "${ROCM_VERSION}" | cut -d. -f1-2) && \
    R="/opt/rocm/core-${_mm}" && L="${R}/lib/llvm" && \
    _stub_ver=$(apt-cache policy "amdrocm-llvm-dev${_mm}" | awk '$1 == "Candidate:" {print $2}') && \
    if [ -z "${_stub_ver}" ] || [ "${_stub_ver}" = "(none)" ]; then \
        echo "ERROR: no apt candidate for amdrocm-llvm-dev${_mm}" >&2; exit 1; \
    fi && \
    mkdir -p /tmp/llvm-dev-stub/DEBIAN && \
    printf 'Package: %s\nVersion: %s\nArchitecture: amd64\nMaintainer: rocm-aic\nDescription: empty stand-in for the LLVM static libraries and headers\n' \
        "amdrocm-llvm-dev${_mm}" "${_stub_ver}" > /tmp/llvm-dev-stub/DEBIAN/control && \
    dpkg-deb -b /tmp/llvm-dev-stub /tmp/llvm-dev-stub.deb && \
    dpkg -i /tmp/llvm-dev-stub.deb && \
    apt-mark hold "amdrocm-llvm-dev${_mm}" && \
    rm -rf /tmp/llvm-dev-stub /tmp/llvm-dev-stub.deb && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl \
        git \
        ibverbs-providers \
        libaio1t64 \
        libgomp1 \
        libnuma1 \
        librdmacm1t64 \
        liburing2 \
        make \
        python3-dev \
        python3-pip \
        python3-venv \
        "amdrocm-amdsmi${_mm}" \
        "amdrocm-base${_mm}" \
        "amdrocm-blas-host${_mm}" \
        "amdrocm-dnn-host${_mm}" \
        "amdrocm-fft-host${_mm}" \
        "amdrocm-hipfile${_mm}" \
        "amdrocm-llvm${_mm}" \
        "amdrocm-math-common${_mm}" \
        "amdrocm-profiler-base${_mm}" \
        "amdrocm-rand-host${_mm}" \
        "amdrocm-rccl-host${_mm}" \
        "amdrocm-runtime${_mm}" \
        "amdrocm-runtime-dev${_mm}" \
        "amdrocm-solver-host${_mm}" \
        "amdrocm-sparse-host${_mm}" \
        "amdrocm-sysdeps${_mm}" \
    && _arch_pkgs="" \
    && for _arch in $(echo "${ROCM_ARCH}" | tr ',;' '  '); do \
          _arch_pkgs="${_arch_pkgs} \
                      amdrocm-blas${_mm}-${_arch} \
                      amdrocm-rand${_mm}-${_arch} \
                      amdrocm-rccl${_mm}-${_arch}"; \
       done \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ${_arch_pkgs} \
    && update-ca-certificates \
    && update-alternatives --install /opt/rocm/core core "${R}" 1375129220 \
    && rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED \
    && test -f "${R}/include/hip/hip_runtime.h" \
    && if [ -e "${L}/lib/libLLVMCore.a" ]; then \
           echo "ERROR: the real amdrocm-llvm-dev${_mm} was installed over the stub" >&2; exit 1; \
       fi \
    && (cd "${L}/bin" && rm -f \
            flang* amdflang bbc fir-* tco f18-parse-demo mlir-* \
            clang-tidy run-clang-tidy clang-doc clang-change-namespace clang-move \
            find-all-symbols clang-include-cleaner clang-include-fixer clang-query \
            clang-reorder-fields clang-apply-replacements modularize pp-trace \
            llvm-tblgen clang-tblgen tblgen-to-irdl tblgen-lsp-server \
            yaml-bench llvm-test-mustache-spec) \
    && rm -f "${L}"/lib/libMLIR*.so* "${L}"/lib/libmlir_*.so* \
    && rm -rf "${L}"/lib/clang/*/finclude \
    && find "${L}/lib/clang" -type f \( -name 'libflang_rt*' \
            -o -name 'libclang_rt.*san*' -o -name 'libclang_rt.fuzzer*' \
            -o -name 'libclang_rt.memprof*' -o -name 'libclang_rt.xray*' \
            -o -name 'libclang_rt.scudo*' -o -name 'libclang_rt.profile*' \
            -o -name 'libclang_rt.stats*' -o -name 'libclang_rt.safestack*' \
            -o -name 'libclang_rt.cfi*' -o -name 'libclang_rt.ctx_profile*' \
            -o -name 'libclang_rt.dd-*' \) -delete \
    && _py=$(python3 -c 'import sys; print("python%d.%d" % sys.version_info[:2])') \
    && for d in "${R}"/lib/python3.*; do \
           if [ "$(basename "${d}")" != "${_py}" ]; then rm -rf "${d}"; fi; \
       done \
    && rm -rf "${R}/lib/libamd_smi_static.a" "${R}/lib/libamdsminic.a" \
              "${R}/share/amd_smi/tests" "${R}/share/amd_smi/example" \
              "${R}/share/roctracer/test" \
              "${R}/share/rocprofiler-sdk/tests" "${R}/share/rocprofiler-sdk/samples" \
              "${R}"/lib/rocm_sysdeps/lib/librocm_sysdeps_gallium_drv_video.so* \
              "${R}"/lib/rocm_sysdeps/lib/*.a \
              /usr/lib/python3.*/config-*/libpython3.*.a \
    && find "${R}/lib" "${L}/lib" -maxdepth 1 -type f \( \
            -name 'librocroller.so.*' -o -name 'librocsparse.so.*' \
            -o -name 'librocrand.so.*' -o -name 'librocsolver.so.*' \
            -o -name 'librocblas.so.*' -o -name 'librocprofiler-sdk.so.*' \
            -o -name 'libLLVM.so.*' -o -name 'libclang-cpp.so.*' \) \
            -exec "${L}/bin/llvm-strip" --strip-unneeded {} + \
    && test -s "${R}/lib/libamdhip64.so" \
    && : > "$(readlink -f "${R}/lib/libamdhip64.so")" \
    && "${L}/bin/clang++" --version

RUN git config --global advice.detachedHead false

# ROCm version info — at build time, AITER's setup.py determines HIP version by
# reading one of the following files:
# - /opt/rocm/.info/version
# - /opt/rocm/include/hip/hip_version.h
# The runtime packages install the
# ROCm toolchain but the header file are in the development package.
#
# Also expose clang++/clang as c++/cc so AITER JIT can link module_aiter_core.so.
RUN ln -sfn "/opt/rocm/core/lib/llvm/bin/clang++" /usr/local/bin/c++ && \
    ln -sfn "/opt/rocm/core/lib/llvm/bin/clang"   /usr/local/bin/cc

ENV PIP_ROOT_USER_ACTION=ignore \
    PIP_BREAK_SYSTEM_PACKAGES=1

# =============================================================================
# runtime-from-wheels — aic-base assembled from PUBLISHED wheels: nothing is
# compiled.  `make build-from-wheels` builds it (bake target base-from-wheels)
# with one release's wheel set as the `pip-wheels` build context, for a node
# that should run the stack without the source builds above.
#
# It differs from `runtime` in what no wheel carries:
#   - libamdhip64 is the packaged one, not build_clr_patch's.  The patch fixes
#     the signal ring of cross-process GPU events, which both images replace
#     with a producer-side sync (docker/lmcache/rocm_event_ipc_patch.py).
#   - NIXL is the nixl_rocm wheel rather than the /opt/nixl install, and there
#     is no UCX: the POSIX and AIS_MT plugins load, the UCX one does not.
# =============================================================================
FROM runtime-os AS runtime-from-wheels

ARG ROCM_VERSION
ARG ROCM_ARCH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# The released torch, torchvision, triton and nixl_rocm, with the pins and the
# cleanup of `runtime`.  Exactly one of each: with one missing, pip would
# resolve the requirement from PyPI, where torch is a CUDA build.
# hadolint ignore=SC2015,SC2206
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,from=pip-wheels,target=/w \
    shopt -s nullglob && \
    for pat in 'torch-*.whl' 'torchvision-*.whl' 'triton-*.whl' 'nixl*rocm*.whl'; do \
        hits=(/w/${pat}); \
        if [ "${#hits[@]}" -ne 1 ]; then \
            echo "ERROR: expected exactly one ${pat} in the pip-wheels build context, found ${#hits[@]}" >&2; \
            ls -l /w >&2; exit 1; \
        fi; \
    done && \
    pip3 install --no-compile \
        /w/torch-*.whl /w/torchvision-*.whl /w/triton-*.whl /w/nixl*rocm*.whl \
        'numpy<2.5' 'setuptools>=78.1.1,<80' && \
    pip3 uninstall -y \
        nixl nixl-cu12 nixl-cu13 \
        cupy-cuda12x cufile-python cuda-pathfinder && \
    _tl=$(python3 -c 'import sysconfig; print(sysconfig.get_path("platlib"))')/torch/lib && \
    for f in "${_tl}"/libaotriton_v2.so.*; do \
        if [ -f "${f}" ] && [ ! -L "${_tl}/libaotriton_v2.so" ] && cmp -s "${f}" "${_tl}/libaotriton_v2.so"; then \
            ln -sf "$(basename "${f}")" "${_tl}/libaotriton_v2.so"; \
        fi; \
    done && \
    { find /usr/local/lib/python3.* -name '__pycache__' -type d \
         -exec rm -rf {} + || \
      echo "WARNING: __pycache__ cleanup failed; see find's error above" >&2; }

# runtime-os emptied the packaged libamdhip64 for `runtime` to fill with the
# patched build.  Put the packaged file back.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    _mm=$(echo "${ROCM_VERSION}" | cut -d. -f1-2) && \
    _lib=$(readlink -f "/opt/rocm/core-${_mm}/lib/libamdhip64.so") && \
    _pkg=$(dpkg -S "${_lib}" | cut -d: -f1) && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --reinstall "${_pkg}" && \
    test -s "${_lib}"

# NIXL from the wheel, laid out where `runtime` keeps it, so the ENV below is
# the same: the `nixl` shim on PYTHONPATH (off dist-packages, where a pip
# `nixl` cannot replace it), and the wheel's libraries, with plugins/ beside
# them, at /opt/nixl/lib/x86_64-linux-gnu.
#
# ldconfig -n is for wheels built before build_nixl stored versioned
# libraries under their sonames.  Those carry prometheus-cpp as
# libcore.so.1.3.0 and libpull.so.1.3.0 with no libcore.so.1.3 or
# libpull.so.1.3 beside them, which is what the telemetry exporters ask the
# loader for (the directory itself they reach through their own RUNPATH).
# Without the links, creating a NIXL agent with the Prometheus exporter that
# compose selects fails with "Failed to load telemetry plugin", and the
# lmcache server with it.  On a repaired wheel it changes nothing.
COPY docker/base/nixl_rocm_shim.py /opt/nixl/lib/python3/dist-packages/nixl/__init__.py
RUN _libs=$(python3 -c 'import sysconfig; print(sysconfig.get_path("platlib"))')/.nixl_rocm.mesonpy.libs && \
    test -f "${_libs}/plugins/libplugin_POSIX.so" && \
    ldconfig -n "${_libs}" && \
    ln -sfn "${_libs}" /opt/nixl/lib/x86_64-linux-gnu

RUN _mm=$(echo "${ROCM_VERSION}" | cut -d. -f1-2) && \
    echo "/opt/rocm/core-${_mm}/lib" > /etc/ld.so.conf.d/rocm.conf && \
    ldconfig

ENV NIXL_PLUGIN_DIR=/opt/nixl/lib/x86_64-linux-gnu/plugins \
    PYTHONPATH=/opt/rocm/core/share/amd_smi:/opt/nixl/lib/python3/dist-packages \
    LD_LIBRARY_PATH=/opt/nixl/lib/x86_64-linux-gnu:/opt/nixl/lib:/opt/rocnixl-ucx/lib:/opt/rocm/core/lib \
    ROCM_PATH=/opt/rocm/core \
    PATH=/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# The checks `runtime` ends on, plus what this stage put in place by hand: the
# HIP runtime loads, every NIXL plugin but UCX resolves its libraries, and an
# agent comes up the way the lmcache server creates one, telemetry included.
RUN _arch=$(echo "${ROCM_ARCH}" | tr ',;' '  ' | awk '{print $1}') && \
    printf '#include <hip/hip_runtime.h>\n__global__ void k(float* a, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) a[i] += 1.0f; }\n' > /tmp/k.hip && \
    hipcc -x hip ${_arch:+--offload-arch="${_arch}"} -O3 -fPIC -shared /tmp/k.hip -o /tmp/k.so && \
    rm -f /tmp/k.hip /tmp/k.so && \
    PYTHONDONTWRITEBYTECODE=1 python3 -c 'import ctypes, nixl, numpy, setuptools, torch, torchvision, triton; \
    ctypes.CDLL("libamdhip64.so"); \
    assert torch.version.hip, "torch is not a ROCm build"; \
    print("torch", torch.__version__, "hip", torch.version.hip, "torchvision", torchvision.__version__, "triton", triton.__version__, "numpy", numpy.__version__, "setuptools", setuptools.__version__)' && \
    for _p in "${NIXL_PLUGIN_DIR}"/*.so; do \
        if [ "${_p##*/}" != libplugin_UCX.so ] && ldd "${_p}" | grep 'not found'; then \
            echo "ERROR: ${_p} does not resolve the libraries above" >&2; exit 1; \
        fi; \
    done && \
    NIXL_TELEMETRY_ENABLE=y NIXL_TELEMETRY_EXPORTER=prometheus NIXL_TELEMETRY_PROMETHEUS_PORT=19090 \
        PYTHONDONTWRITEBYTECODE=1 python3 -c 'from nixl._api import nixl_agent, nixl_agent_config; \
    a = nixl_agent("build-check", nixl_agent_config(backends=[])); a.create_backend("POSIX"); \
    assert "POSIX" in a.backends, a.backends; print("NIXL agent OK:", sorted(a.backends), "of", a.get_plugin_list())' && \
    pip3 --version

# Build parent, not a service: downstream images define their own checks.
HEALTHCHECK NONE

# =============================================================================
# runtime — runtime-os + the Python stack from the build stages.  Last, so it
# is what a build with no --target produces.
# =============================================================================
FROM runtime-os AS runtime

ARG ROCM_VERSION
ARG ROCM_ARCH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# torch, torchvision, triton and nixl-rocm with their dependencies.  numpy and
# setuptools are capped to what vLLM resolves to (numba needs numpy < 2.5, vLLM
# setuptools < 80), so aic-vllm does not add second copies over these.
#
# --no-compile: no bytecode in the layer.  A root container compiles into its
# own writable layer on first import; a non-root one (the emulator services,
# the accuracy client) cannot write to dist-packages and pays the uncached
# import on every start.
#
# The uninstall guards against upstream deps pulling CUDA-only nixl variants
# that would shadow the source-built install from build_nixl.
#
# The torch wheel carries libaotriton_v2.so twice (a wheel cannot hold a
# symlink); link the duplicate back, in the RUN that installs it.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,from=build_nixl,source=/app/nixl/dist,target=/wheel-nixl/ \
    --mount=type=bind,from=build_pytorch,source=/app/pytorch/dist,target=/wheel-pytorch/ \
    --mount=type=bind,from=build_torchvision,source=/app/vision/dist,target=/wheel-vision/ \
    --mount=type=bind,from=build_triton,source=/app/triton/dist,target=/wheel-triton/ \
    pip3 install --no-compile /wheel-*/*.whl 'numpy<2.5' 'setuptools>=78.1.1,<80' && \
    pip3 uninstall -y \
        nixl nixl-cu12 nixl-cu13 \
        cupy-cuda12x cufile-python cuda-pathfinder && \
    _tl=$(python3 -c 'import sysconfig; print(sysconfig.get_path("platlib"))')/torch/lib && \
    for f in "${_tl}"/libaotriton_v2.so.*; do \
        if [ -f "${f}" ] && [ ! -L "${_tl}/libaotriton_v2.so" ] && cmp -s "${f}" "${_tl}/libaotriton_v2.so"; then \
            ln -sf "$(basename "${f}")" "${_tl}/libaotriton_v2.so"; \
        fi; \
    done && \
    { find /usr/local/lib/python3.* -name '__pycache__' -type d \
         -exec rm -rf {} + || \
      echo "WARNING: __pycache__ cleanup failed; see find's error above" >&2; }

# Patched libamdhip64.so, from build_clr_patch: fixes the IPC event signal ring
# overflow (ROCm#11511/#11749).  read_index must be wrapped with
# % IPC_SIGNALS_PER_EVENT (32) or the 33rd cross-process import of the same
# event handle reads past the signal array.  Fills the file emptied above.
RUN --mount=type=bind,from=build_clr_patch,source=/app/clr-build/hipamd/lib,target=/clr-build/ \
    _mm=$(echo "${ROCM_VERSION}" | cut -d. -f1-2) && \
    cp /clr-build/libamdhip64.so "/opt/rocm/core-${_mm}/lib/libamdhip64.so" && \
    test -s "/opt/rocm/core-${_mm}/lib/libamdhip64.so"

# NIXL native libs and Python bindings — compiled in build_nixl against the
# ROCm-full toolchain; only the output directories are needed at runtime.
COPY --from=build_nixl /opt/nixl/ /opt/nixl/
COPY --from=build_nixl /opt/rocnixl-ucx/ /opt/rocnixl-ucx/

# The AMD packages install to /opt/rocm/core-<major.minor>/ (not /opt/rocm/
# directly).  The -full image creates symlinks; with explicit packages we must
# use the versioned path, derived from ROCM_VERSION (10.2.0 -> core-10.2).
RUN _mm=$(echo "${ROCM_VERSION}" | cut -d. -f1-2) && \
    echo "/opt/rocm/core-${_mm}/lib" > /etc/ld.so.conf.d/rocm.conf && \
    ldconfig

ENV NIXL_PLUGIN_DIR=/opt/nixl/lib/x86_64-linux-gnu/plugins \
    PYTHONPATH=/opt/rocm/core/share/amd_smi:/opt/nixl/lib/python3/dist-packages \
    LD_LIBRARY_PATH=/opt/nixl/lib/x86_64-linux-gnu:/opt/nixl/lib:/opt/rocnixl-ucx/lib:/opt/rocm/core/lib \
    ROCM_PATH=/opt/rocm/core \
    PATH=/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# CPU-only checks of what the slimming above could break: hipcc still compiles
# device code with the pruned LLVM and without amdrocm-llvm-dev, and the Python
# stack imports from the repacked wheels.
RUN _arch=$(echo "${ROCM_ARCH}" | tr ',;' '  ' | awk '{print $1}') && \
    printf '#include <hip/hip_runtime.h>\n__global__ void k(float* a, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) a[i] += 1.0f; }\n' > /tmp/k.hip && \
    hipcc -x hip ${_arch:+--offload-arch="${_arch}"} -O3 -fPIC -shared /tmp/k.hip -o /tmp/k.so && \
    rm -f /tmp/k.hip /tmp/k.so && \
    PYTHONDONTWRITEBYTECODE=1 python3 -c 'import nixl, numpy, setuptools, torch, torchvision, triton; \
    print("torch", torch.__version__, "torchvision", torchvision.__version__, "triton", triton.__version__, "numpy", numpy.__version__, "setuptools", setuptools.__version__)' && \
    pip3 --version

# Build parent, not a service: downstream images define their own checks.
HEALTHCHECK NONE
