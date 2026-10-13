# syntax=docker/dockerfile:1.28
#
# vLLM + LMCache (MP mode) + NIXL on ROCm, for AMD Instinct GPUs.
#
# ROCm 10.1 ships as Python wheels (TheRock layout): rocm-sdk-core, a per-GFX
# rocm-sdk-device-<arch>, and a matching per-GFX amd-torch-device-<arch>. We
# consume those prebuilt -- building ROCm and PyTorch from source is neither
# practical nor useful here. vLLM, LMCache, NIXL and UCX are built from source.
#
# Installing only the device wheels for one GPU_ARCH is the whole slimming
# story: the stock rocm/pytorch image carries all 25 targets at 24 GB of
# site-packages; one target is 6.7 GB.

ARG RUNTIME_BASE=ubuntu:26.04
ARG PYTHON_VERSION=3.14

####################  python + venv  ####################
FROM ${RUNTIME_BASE} AS python-base

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=500

# apt lists and the apt archive are cache mounts, so they never land in a layer.
RUN rm -f /etc/apt/apt.conf.d/docker-clean
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv libpython${PYTHON_VERSION} \
        libnuma1 libgomp1 libdrm2 \
    && python${PYTHON_VERSION} -m venv /opt/venv

COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv

####################  ROCm runtime, one GPU arch  ####################
FROM python-base AS rocm-runtime

# MI210/MI250 = gfx90a, MI300X/MI325X = gfx942, MI350X = gfx950.
ARG GPU_ARCH=gfx90a
ARG ROCM_VERSION=10.1.0
ARG TORCH_VERSION=2.13.0+rocm10.1.0
# vLLM pulls torchvision in, and the PyPI build is ABI-incompatible with ROCm
# torch, so the ROCm build is installed here, before vLLM's deps resolve.
# torchvision's metadata requires a bare `torch`, so nothing catches a version
# skew: 0.28.0 against torch 2.14 fails at load with an undefined c10::cuda
# symbol and silently leaves torchvision::nms unregistered. 2.13.0/0.28.0 is
# the matched rocm10.1.0 pair on the stable index -- keep them in lockstep.
ARG TORCHVISION_VERSION=0.28.0+rocm10.1.0
ARG ROCM_INDEX=https://stable.repo.amd.com/rocm/whl-next/
ARG PYTHON_VERSION

ENV SITE_PACKAGES=/opt/venv/lib/python${PYTHON_VERSION}/site-packages
ENV SDK_DEV=${SITE_PACKAGES}/_rocm_sdk_devel \
    SDK_CORE=${SITE_PACKAGES}/_rocm_sdk_core
# Paths per vLLM's docker/Dockerfile.rock_base -- there is no /opt/rocm in the
# wheel layout, so every consumer has to be pointed at the SDK packages.
ENV ROCM_PATH=${SDK_DEV} \
    ROCM_HOME=${SDK_DEV} \
    HIP_PATH=${SDK_DEV} \
    HIP_CLANG_PATH=${SDK_DEV}/lib/llvm/bin \
    HIP_DEVICE_LIB_PATH=${SDK_CORE}/lib/llvm/amdgcn/bitcode \
    PYTORCH_ROCM_ARCH=${GPU_ARCH} \
    HSA_NO_SCRATCH_RECLAIM=1
ENV LD_LIBRARY_PATH=/opt/ucx/lib:${SDK_DEV}/lib:${SDK_DEV}/lib/rocm_sysdeps/lib:${SDK_CORE}/lib:${SDK_CORE}/lib/rocm_sysdeps/lib
ENV PATH=/opt/venv/bin:/opt/ucx/bin:${SDK_DEV}/bin:${SDK_DEV}/lib/llvm/bin:${SDK_CORE}/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --index-url ${ROCM_INDEX} --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match \
        "rocm-sdk-core==${ROCM_VERSION}" \
        "rocm-sdk-libraries==${ROCM_VERSION}" \
        "rocm-sdk-device-${GPU_ARCH}==${ROCM_VERSION}" \
        "torch==${TORCH_VERSION}" \
        "amd-torch-device-${GPU_ARCH}==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "amd-torchvision-device-${GPU_ARCH}==${TORCHVISION_VERSION}"

####################  build toolchain  ####################
FROM rocm-runtime AS build-base

ARG GPU_ARCH
ARG ROCM_VERSION
ARG ROCM_INDEX
ARG PYTHON_VERSION
ARG MAX_JOBS=64
ENV MAX_JOBS=${MAX_JOBS} \
    CCACHE_DIR=/root/.cache/ccache \
    # hipcc is a wrapper script, so hash the real compiler, not the wrapper.
    CCACHE_COMPILERCHECK=content \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    # vLLM's expensive work is HIP kernels, which CMake routes through this one
    # rather than the C/CXX launchers above.
    CMAKE_HIP_COMPILER_LAUNCHER=ccache
# Puts ccache's compiler shims ahead of the real ones, which is what catches
# UCX's autotools build -- it invokes the compiler by name and reads neither
# the CMake launchers nor meson's. /usr/lib/ccache has to come before the venv:
# the SDK wheels ship their own hipcc in /opt/venv/bin, which would otherwise
# shadow the shim. The dir holds compiler shims only, so python and everything
# else still resolve from the venv as before.
ENV PATH=/usr/lib/ccache:/opt/venv/bin:${PATH}

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ccache cmake ninja-build pkg-config patchelf \
        autoconf automake libtool \
        python${PYTHON_VERSION}-dev \
        libnuma-dev libdrm-dev libibverbs-dev librdmacm-dev rdma-core ibverbs-providers \
        libssl-dev libcurl4-openssl-dev libaio-dev liburing-dev \
    # apt's ccache only shims compilers it installed, so hipcc gets one by hand.
    # Invoked through this name, ccache compiles with the next hipcc on PATH
    # (the SDK's). This is the only thing that caches LMCache's HIP kernels:
    # torch's cpp_extension execs $CXX as a single binary, so a "ccache hipcc"
    # command string is not an option.
    && ln -sf "$(command -v ccache)" /usr/lib/ccache/hipcc

# hipcc and the HIP headers live in rocm-sdk-devel; `rocm-sdk init` expands the
# devel tree and links in the device wheels already installed.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --index-url ${ROCM_INDEX} --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match "rocm-sdk-devel==${ROCM_VERSION}" \
    && rocm-sdk init \
    && uv pip install "cmake<4" ninja meson meson-python pybind11 patchelf \
        "setuptools>=77,<80" setuptools-scm setuptools-rust wheel packaging build

####################  UCX (NIXL's transport for AMD GPU memory)  ####################
FROM build-base AS ucx

ARG UCX_REF=v1.22.0
RUN --mount=type=cache,target=/root/.cache/ccache \
    git clone --depth 1 -b ${UCX_REF} https://github.com/openucx/ucx.git /src/ucx \
    && cd /src/ucx && ./autogen.sh \
    && ./contrib/configure-release-mt \
        --prefix=/opt/ucx \
        --enable-shared --disable-static --disable-doxygen-doc \
        --enable-optimizations --enable-cma --enable-devel-headers \
        --with-rocm=${ROCM_PATH} --without-cuda --with-verbs --with-dm --without-gdrcopy \
    && make -j${MAX_JOBS} && make install-strip \
    && rm -rf /src/ucx

####################  NIXL  ####################
FROM ucx AS nixl

ARG NIXL_REF=v1.5.0
ENV PKG_CONFIG_PATH=/opt/ucx/lib/pkgconfig
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    git clone --depth 1 -b ${NIXL_REF} https://github.com/ai-dynamo/nixl.git /src/nixl \
    && cd /src/nixl \
    && uv build --wheel --no-build-isolation -o /wheels \
        -Csetup-args=-Drocm_path=${ROCM_PATH} \
        -Csetup-args=-Ducx_path=/opt/ucx \
        -Csetup-args=-Dwheel_variant=rocm \
    # The backend wheel above exposes `nixl_rocm`; the top-level `nixl` module
    # that LMCache imports lives in the separate meta package, which dispatches
    # to nixl_rocm when no CUDA stack is present. Deps are emptied because the
    # backend wheel is installed directly rather than resolved from an index.
    && cd src/bindings/python/nixl-meta \
    && sed -e "s/@VERSION@/${NIXL_REF#v}/" -e "s/@WHEEL_DEPS@//" \
           -e "s/setuptools>=80.9.0/setuptools/" pyproject.toml.in > pyproject.toml \
    && cp /src/nixl/LICENSE . \
    && python3 -m build --wheel --no-isolation -o /wheels \
    && rm -rf /src/nixl

####################  LMCache (HIP extensions)  ####################
FROM build-base AS lmcache

ARG LMCACHE_REF=v0.5.5
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    git clone --depth 1 -b ${LMCACHE_REF} https://github.com/LMCache/LMCache.git /src/lmcache \
    && cd /src/lmcache \
    && uv pip install -r requirements/build.txt \
    # LMCache's ROCm profile pins -std=c++17 for the non-HIP sources; it targets
    # the ROCm 7.x / torch 2.9 era. torch 2.14 headers require C++20.
    && sed -i 's/-std=c++17/-std=c++20/' setup_extensions/build_profiles/rocm.py \
    # numpy<=2.2.6 is pinned for the PyPI nixl's numba dependency. We build NIXL
    # from source and it declares only torch+numpy, so the ceiling does not
    # apply -- and 2.2.6 has no cp314 wheel, forcing a source build downstream.
    && sed -i 's/^numpy<=2\.2\.6$/numpy/' requirements/common.txt \
    # 381 MB of cupy built against ROCm 7.0, pulled in unconditionally for HIP
    # builds. On ROCm 10.1 it cannot even import (librocfft.so.0 is gone), and
    # every use site is a lazy import inside CUDA-only paths -- GDS/cuFile and
    # RawCudaIPCWrapper -- that the MP GPU IPC path never reaches.
    && sed -i '/^cupy-rocm/d' requirements/rocm_core.txt \
    # setup.py directly, as LMCache's own Dockerfile does: `uv build
    # --no-build-isolation` fails to expose setuptools to the backend on 3.14.
    # torch's cpp_extension execs $CXX as a single binary, so CXX="ccache hipcc"
    # is looked up as one literal name and fails. The hipcc shim created in
    # build-base is what routes this through ccache instead.
    && BUILD_WITH_HIP=1 CXX=hipcc python3 setup.py bdist_wheel --dist-dir=/wheels \
    && rm -rf /src/lmcache

####################  vLLM  ####################
FROM build-base AS vllm-wheel

ARG VLLM_REF=v0.31.0
ENV VLLM_TARGET_DEVICE=rocm
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    git clone --depth 1 -b ${VLLM_REF} https://github.com/vllm-project/vllm.git /src/vllm \
    && cd /src/vllm \
    # Drops the torch pins so the build uses the ROCm torch already installed.
    # requirements/build/rocm.txt is deliberately not used: it carries an
    # --extra-index-url (repo.amd.com/rocm/whl-multi-arch) that 403s and then
    # shadows PyPI for every other requirement. build-base has these deps.
    && python3 tools/use_existing_torch.py \
    # Mooncake is an alternative KV-transfer backend; this image uses the
    # LMCache MP connector with NIXL instead. Its ROCm wheels stop at cp313,
    # so keeping it would pin the whole image to an older Python.
    && sed -i '/mooncake-transfer-engine-rocm/d' requirements/rocm.txt \
    && uv build --wheel --no-build-isolation -o /wheels \
    && rm -rf /src/vllm

####################  shared runtime: ROCm + UCX + NIXL + LMCache  ####################
FROM rocm-runtime AS runtime-base

ARG PYTHON_VERSION

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        libibverbs1 librdmacm1 rdma-core ibverbs-providers libatomic1 \
        # triton JIT-compiles its HIP shim as a CPython extension at engine
        # start, so the runtime genuinely needs a C compiler and Python.h.
        # Without them vLLM dies in HIPUtils with "Failed to find C compiler"
        # / "Python.h: No such file or directory". ~90 MB, unavoidable.
        gcc libc6-dev python${PYTHON_VERSION}-dev

# vLLM's ROCm platform plugin probes `import amdsmi`, and without it falls back
# to UnspecifiedPlatform ("Failed to infer device type"). The SDK ships bindings
# matched to its own libamd_smi, so put those on the path rather than taking the
# PyPI amdsmi, which is built for ROCm 7.x.
RUN echo "${SITE_PACKAGES}/_rocm_sdk_core/share/amd_smi" > ${SITE_PACKAGES}/_rocm_amdsmi.pth

COPY --from=ucx /opt/ucx /opt/ucx
# PyPI only: torch and the ROCm SDK are already installed, and the ROCm index
# also carries an old numpy with no cp314 wheel, which would be built from
# source here -- in a stage that deliberately has no compiler.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=bind,from=nixl,source=/wheels,target=/wheels/nixl \
    --mount=type=bind,from=lmcache,source=/wheels,target=/wheels/lmcache \
    uv pip install /wheels/nixl/*.whl /wheels/lmcache/*.whl \
    && find /opt/venv -name '__pycache__' -prune -exec rm -rf {} + \
    && python3 -c "import lmcache, nixl; print('lmcache + nixl OK')"

####################  target: LMCache MP cache server  ####################
FROM runtime-base AS lmcache-server

EXPOSE 6555 8080
# MP mode: a standalone cache server the vLLM workers talk to over ZMQ.
ENTRYPOINT ["python3", "-m", "lmcache.v1.multiprocess.server"]
CMD ["--host", "0.0.0.0", "--port", "6555", "--l1-size-gb", "32", "--eviction-policy", "LRU", "--max-workers", "4"]

####################  target: vLLM server  ####################
FROM runtime-base AS vllm

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=bind,from=vllm-wheel,source=/wheels,target=/wheels/vllm \
    uv pip install /wheels/vllm/*.whl \
    && find /opt/venv -name '__pycache__' -prune -exec rm -rf {} + \
    && python3 -c "import vllm; print('vllm', vllm.__version__)"

# `mp-serve <model> [args]` runs the cache server alongside vLLM in this
# container, which is the only arrangement in which the GPU IPC handoff works.
COPY --chmod=755 deploy/mp-serve /usr/local/bin/mp-serve

EXPOSE 8000
ENTRYPOINT ["vllm", "serve"]
