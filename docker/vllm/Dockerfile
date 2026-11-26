# check=skip=InvalidDefaultArgInFrom
# ROCM_BASE_IMAGE is a build arg with no default here (docker/docker-bake.hcl
# owns it), so the FROM lines are blank at lint time.
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# AIC vLLM image: torch + torchvision + NIXL (from aic-base) + AITER +
# FlashAttention + vLLM + llm-emu plugin.
#
# Consumed by the `vllm` compose service.  Shares aic-base with aic-lmcache, but
# its default target needs aic-lmcache's wheels, so build aic-lmcache first.
#
# Stages:
#   build_mori  MORI wheel with UMBP, installed by `runtime` only
#   vllm      AITER + FlashAttention + vLLM source build + llm-emu  (sections 3-3b)
#   wheels    scratch image holding vllm/amd_aiter/flash_attn .whl  (section 3c)
#   emulate   CPU-only runtime + the PyPI lmcache client           (section 3d)
#   runtime   default target: `vllm` + aic-lmcache's lmcache wheel  (section 3e)
#
# Build args: docker/docker-bake.hcl is the only owner of every pin, URL and
# base image, so no ARG here has a default and each consuming stage stops when
# one is unset.  Build through bake (from repo root), which also wires in
# aic-base, its wheels and the lmcache wheels:
#   ROCM_ARCH=gfx950 docker buildx bake -f docker/docker-bake.hcl vllm --load
#   ROCM_ARCH=gfx950 docker buildx bake -f docker/docker-bake.hcl emulate --load
#
# or hand bake's resolved args to a raw build against images built earlier:
#   export ROCM_ARCH=gfx950
#   mapfile -t basew_args < <(docker/scripts/aic-bake-env.sh --build-args base-wheels)
#   docker buildx build "${basew_args[@]}" \
#     -f docker/base/Dockerfile --target wheels \
#     --output type=local,dest=/tmp/aic-base-wheels .
#   mapfile -t lmc_args < <(docker/scripts/aic-bake-env.sh --build-args lmcache-wheels)
#   docker buildx build "${lmc_args[@]}" \
#     --build-context base=docker-image://aic-base:<tag> \
#     --build-context base-wheels=/tmp/aic-base-wheels \
#     -f docker/lmcache/Dockerfile --target wheels \
#     --output type=local,dest=/tmp/aic-lmcache-wheels .
#   mapfile -t args < <(docker/scripts/aic-bake-env.sh --build-args vllm)
#   docker buildx build "${args[@]}" \
#     --build-context base=docker-image://aic-base:<tag> \
#     --build-context base-wheels=/tmp/aic-base-wheels \
#     --build-context lmcache-wheels=/tmp/aic-lmcache-wheels \
#     -f docker/vllm/Dockerfile -t aic-vllm:dev .
# (`--build-args emulate` plus `--target emulate` for the CPU-only image.)
#
# Or via make:
#   make build ROCM_ARCH=gfx950            (local)
#   make dist-build AIC_ROCM_ARCH=gfx950   (Slurm)

# Fallback base stage: used when no --build-context base= is provided.  When
# --build-context base=docker-image://aic-base:<TAG> IS supplied, BuildKit
# overrides this stage with the pre-built image and skips this FROM entirely.
# build_vllm always starts from ROCM_BASE_IMAGE.  Unset, the build stops at the
# FROM: base name (${ROCM_BASE_IMAGE}) should not be blank
ARG ROCM_BASE_IMAGE
# hadolint ignore=DL3006
FROM ${ROCM_BASE_IMAGE} AS base

# Empty default for the `lmcache-wheels` build context (the --target wheels
# export of docker/lmcache/Dockerfile).  Only `runtime` reads it, and fails on
# an empty one, so a build that forgot the context stops instead of shipping a
# vLLM that cannot load LMCacheMPConnector.  `emulate` never reads it.
FROM scratch AS lmcache-wheels

FROM scratch AS base-wheels

# =============================================================================
# build_mori — MORI wheel with UMBP, for the `runtime` stage only (`emulate`
# never references it, so a CPU-only build skips it).  The wheel carries the
# `mori` package vLLM's UMBPStoreConnector imports plus the
# umbp_standalone_server and umbp_master binaries.
#
# MORI versions itself from its git tag (setuptools_scm), so MORI_REF must be a
# tag or branch.  Its setup.py drops UMBP with only a warning when gRPC or
# Protobuf is missing, hence the check that the server is in the wheel.
# BUILD_TORCH_SYMM=OFF: that extension is MORI-CCO's, and this stage has no torch.
# =============================================================================
FROM ${ROCM_BASE_IMAGE} AS build_mori

# hadolint ignore=DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG MORI_GIT_URL
ARG MORI_REF
RUN : "${MORI_GIT_URL:?unset: build through docker/docker-bake.hcl}" \
      "${MORI_REF:?unset: build through docker/docker-bake.hcl}"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        git \
        libgrpc++-dev \
        libgrpc-dev \
        libibverbs-dev \
        libnuma-dev \
        libpci-dev \
        libprotobuf-dev \
        ninja-build \
        protobuf-compiler \
        protobuf-compiler-grpc \
        python3-dev \
        python3-pip

WORKDIR /app/mori
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth 1 --branch "${MORI_REF}" --recurse-submodules --shallow-submodules \
        "${MORI_GIT_URL}" . && \
    _rocm="${ROCM_PATH:-/opt/rocm}" && \
    CMAKE_PREFIX_PATH="${_rocm}:${_rocm}/lib/rocm_sysdeps" \
        BUILD_UMBP=ON \
        BUILD_TORCH_SYMM=OFF \
        pip3 wheel --no-deps -w /wheels . && \
    python3 -m zipfile -l /wheels/amd_mori-*.whl > /tmp/mori-wheel.txt && \
    for _bin in umbp_standalone_server umbp_master; do \
        grep -q "mori/${_bin}\$" /tmp/mori-wheel.txt || \
            { echo "ERROR: MORI wheel has no ${_bin}: UMBP was not built" >&2; exit 1; }; \
    done && \
    rm /tmp/mori-wheel.txt
WORKDIR /

# =============================================================================
# build_vllm — compile AITER, FlashAttention, and vLLM against the full ROCm
# image (which has hipcc + HIP CMake files).  Torch is COPYed in from aic-base
# so the built wheels are compatible with the slim runtime stage.
# =============================================================================
FROM ${ROCM_BASE_IMAGE} AS build_vllm

# hadolint ignore=DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ROCM_ARCH
ARG BUILD_JOBS
ARG VLLM_GIT_URL
ARG VLLM_REF
ARG VLLM_TARGET_DEVICE
ARG AITER_GIT_URL
ARG AITER_REF
ARG FLASH_ATTN_GIT_URL
ARG FLASH_ATTN_REF
# AITER kernels are JIT-compiled on first use, which costs 5-10 min on every
# cold container -- and the only way to pay it at build time used to be running
# vLLM on a real GPU after the build, which a CPU-only build node cannot do.
# AITER's own setup.py can compile them ahead of time instead.  No GPU is
# needed -- the arch comes from GPU_ARCHS/PYTORCH_ROCM_ARCH -- but it only
# works for a SINGLE arch, since AITER's build-time codegen keys on one gfx.
# Off by default (docker/docker-bake.hcl) so the default multi-arch image is
# unchanged.
#
# The levels are NOT ordered by cost; read them as different MHA policies:
#   0  off -- JIT at runtime
#   1  named-variant mode: drops the generic MHA modules and adds back five
#      specific bf16 mha_varlen_fwd instances (req_md_names in setup.py).
#      This is the cheap one.
#   2  everything except _bwd/_tune -- which KEEPS the whole MHA family
#      (libmha_fwd, module_mha_fwd, module_mha_varlen_fwd).  Those three alone
#      took >85 min of a 128-core allocation on job 68147805, because CK emits
#      a template instance per dtype x head-dim x mask x bias x lse x dropout
#      x skip x qscale combination -- instance fan-out, not arch fan-out.
#   3  fmha_v3 only
# Prefer 1 for single-arch builds; 2 only when you need the full MHA set.
ARG AITER_PREBUILD_KERNELS
RUN : "${VLLM_GIT_URL:?unset: build through docker/docker-bake.hcl}" \
      "${VLLM_REF:?unset: build through docker/docker-bake.hcl}" \
      "${VLLM_TARGET_DEVICE:?unset: build through docker/docker-bake.hcl}" \
      "${AITER_GIT_URL:?unset: build through docker/docker-bake.hcl}" \
      "${AITER_REF:?unset: build through docker/docker-bake.hcl}" \
      "${FLASH_ATTN_GIT_URL:?unset: build through docker/docker-bake.hcl}" \
      "${FLASH_ATTN_REF:?unset: build through docker/docker-bake.hcl}" \
      "${AITER_PREBUILD_KERNELS:?unset: build through docker/docker-bake.hcl}"

# git and ca-certificates are not in the -full image; install before cloning.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        git \
        ninja-build \
        python3-dev \
        python3-pip

# Bring torch + Python packages from aic-base so wheels link against the same torch.
# Note that `emulate` has no base-wheels context
# The Debian-installed packaging has no RECORD, so pip cannot upgrade it in
# place when the wheels need a newer one; install a pip-owned copy first.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,from=base-wheels,source=/,target=/base-wheels \
    pip3 install --break-system-packages --ignore-installed packaging && \
    find /base-wheels -type f -name "*.whl" -exec pip3 install --break-system-packages {} +

RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

COPY patches/vllm/ /app/patches/vllm/
COPY patches/llm-emu/ /app/patches/llm-emu/
COPY patches/aiter/ /app/patches/aiter/

# ----- 3. ROCm attention backends (AITER + FlashAttention) ----------------------
# Skipped when VLLM_TARGET_DEVICE=empty (CPU-only emulate build: no hipcc).
#
# On the AOT path (AITER_PREBUILD_KERNELS != 0) the RUN below exports two
# FlyDSL knobs.  GPU_ARCHS is what AITER (since v0.1.29) filters the AOT job
# list by; patches/aiter/02 adds the escape hatch on the AOT failure
# assertion.  Both patches/aiter scripts exit non-zero if their anchor is
# missing.  AITER_FLYDSL_AOT_TOLERATE_FAILURES=1 covers the residual
# failures that survive the arch filter: on gfx942 those were the
# `wint4` (INT4-weight) MoE kernels -- 51 of the 352 arch-correct jobs on job
# 68149555, all named flydsl_moe{1,2}_abf16_wint4_bf16_*.  AITER lists them
# under gfx942's cu_num but they did not compile for it in v0.1.22.post1.  Not
# re-measured on v0.1.29.post1: revisit whether it is still needed.
#
# The AOT worker count has no knob of its own any more
# (AITER_FLYDSL_AOT_WORKERS is gone in v0.1.29): setup.py adopts MAX_JOBS as
# AITER_MAX_JOBS, and AITER further caps it at 80% of the CPUs and by free
# memory.
#
# AITER_USE_SYSTEM_TRITON=1: without it AITER's setup.py uninstalls the
# source-built Triton from base-wheels and pip-installs triton==3.8.0 from
# pypi.amd.com (and swallows a failed install, leaving no Triton at all).  It
# only takes effect when a triton dist is installed, hence the check before it.
#
# AITER versions itself from its git tag (setuptools_scm), and --revision
# fetches no tags, so it is cloned with --branch: AITER_REF must be a tag or
# branch, as VLLM_REF already must.
RUN [ "${VLLM_TARGET_DEVICE}" = "empty" ] || \
    test -n "$ROCM_ARCH" || { echo "ERROR: ROCM_ARCH is not set"; exit 1; }

# WORKDIR creates /app/aiter even when the build is skipped; the clone below
# targets that empty directory.
WORKDIR /app/aiter
# hadolint ignore=SC2039
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    if [ "${VLLM_TARGET_DEVICE}" = "empty" ]; then \
        echo "Skipping AITER build (VLLM_TARGET_DEVICE=empty)"; \
    else \
        mkdir -p /wheels && \
        git clone --depth 1 --branch "${AITER_REF}" "${AITER_GIT_URL}" --recurse-submodules . && \
        pip3 install --no-cache-dir -r requirements.txt && \
        _prebuild="${AITER_PREBUILD_KERNELS}" && \
        case "${ROCM_ARCH}" in \
            *\;*) [ "${_prebuild}" = "0" ] || \
                      echo "AITER AOT prebuild off: ROCM_ARCH='${ROCM_ARCH}' is multi-arch"; \
                  _prebuild=0 ;; \
        esac && \
        echo "AITER PREBUILD_KERNELS=${_prebuild} (0 = JIT at runtime)" && \
        _jobs="${BUILD_JOBS:-$(nproc)}" && \
        if [ "${_prebuild}" != "0" ]; then \
            export GPU_ARCHS="${ROCM_ARCH}"; \
            export AITER_FLYDSL_AOT_TOLERATE_FAILURES=1; \
            python3 /app/patches/aiter/02-flydsl-aot-tolerate-failures.py /app/aiter || exit 1; \
            python3 /app/patches/aiter/03-aiter-prebuild-threads.py /app/aiter || exit 1; \
            : "${AITER_PREBUILD_THREADS:=$(( _jobs / 8 ))}"; \
            [ "${AITER_PREBUILD_THREADS}" -lt 1 ] && AITER_PREBUILD_THREADS=1; \
            [ "${AITER_PREBUILD_THREADS}" -gt 32 ] && AITER_PREBUILD_THREADS=32; \
            export AITER_PREBUILD_THREADS; \
            echo "AITER CK prebuild: ${AITER_PREBUILD_THREADS} concurrent modules, ninja up to -j${_jobs} each"; \
        fi && \
        python3 -c 'import importlib.metadata as m; print("Triton kept for AITER build:", m.version("triton"))' && \
        AITER_USE_SYSTEM_TRITON=1 \
            PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
            PREBUILD_KERNELS="${_prebuild}" \
            TORCH_DONT_CHECK_COMPILER_ABI=1 \
            MAX_JOBS="${_jobs}" \
            CCACHE_DIR=/root/.cache/ccache CCACHE_MAXSIZE=25G \
            python3 setup.py bdist_wheel --dist-dir=/wheels && \
        pip3 install --no-cache-dir --no-deps /wheels/amd_aiter-*.whl && \
        python3 -c 'import importlib.util; spec = importlib.util.find_spec("aiter"); assert spec and spec.submodule_search_locations; print("AITER installed:", next(iter(spec.submodule_search_locations)))' && \
        ( cd / && \
          { GPU_ARCHS="${ROCM_ARCH}" PYTORCH_ROCM_ARCH="${ROCM_ARCH}" TORCH_DONT_CHECK_COMPILER_ABI=1 \
                python3 -c 'import aiter' || :; } && \
          test -f "$(python3 -c 'import importlib.util as u, os; print(os.path.join(next(iter(u.find_spec("aiter").submodule_search_locations)), "jit", "module_aiter_core.so"))')" ); \
    fi

# FlashAttention: CDNA-only (gfx9xx — MI100/MI200/MI300/MI355X).  Built whenever
# ROCM_ARCH names at least one gfx9 arch, and compiled for only those: in a mixed
# list the other archs fall back to Triton attention.
# Skipped when VLLM_TARGET_DEVICE=empty (CPU-only emulate build).
# As with AITER, WORKDIR creates /app/flash-attention even when it is skipped.
# torch 2.13+ headers need C++20; this pin predates the upstream switch
# (Dao-AILab/flash-attention#2899), which is the same flag change.
WORKDIR /app/flash-attention
# hadolint ignore=SC2039
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    _gfx9_archs() { \
        _o=""; for _a in $(echo "$1" | tr ';' '\n'); do \
            [ "${_a#gfx9}" = "${_a}" ] || _o="${_o:+${_o};}${_a}"; done; \
        echo "${_o}"; }; \
    touch /etc/rocm-aic-attention-backends.env && \
    if [ "${VLLM_TARGET_DEVICE}" = "empty" ]; then \
        echo "Skipping FlashAttention (VLLM_TARGET_DEVICE=empty)"; \
    elif _fa_arch="$(_gfx9_archs "${ROCM_ARCH}")"; [ -n "${_fa_arch}" ]; then \
        git clone --filter=blob:none --no-checkout "${FLASH_ATTN_GIT_URL}" . && \
        git checkout "${FLASH_ATTN_REF}" && \
        git submodule update --init --recursive --depth 1 && \
        sed -i 's/-std=c++17/-std=c++20/g' setup.py && \
        grep -q -- '-std=c++20' setup.py && \
        GPU_ARCHS="${_fa_arch}" \
            PYTORCH_ROCM_ARCH="${_fa_arch}" \
            HIPFLAGS="-Wno-deprecated-attributes" \
            CXXFLAGS="-Wno-deprecated-attributes" \
            MAX_JOBS="${BUILD_JOBS:-$(nproc)}" \
            CCACHE_DIR=/root/.cache/ccache CCACHE_MAXSIZE=25G \
            python3 setup.py bdist_wheel --dist-dir=/wheels && \
        pip3 install --no-cache-dir --no-deps /wheels/flash_attn-*.whl && \
        python3 -c 'import flash_attn; print("FlashAttention installed:", flash_attn.__file__)' && \
        { \
            echo "AITER_REF=${AITER_REF}"; \
            echo "FLASH_ATTN_REF=${FLASH_ATTN_REF}"; \
            python3 -c 'import importlib.util; spec = importlib.util.find_spec("aiter"); assert spec and spec.submodule_search_locations; print("AITER_PACKAGE=" + next(iter(spec.submodule_search_locations)))'; \
            python3 -c 'import flash_attn; print("FLASH_ATTN_PACKAGE=" + flash_attn.__file__)'; \
        } > /etc/rocm-aic-attention-backends.env; \
    else \
        echo "Skipping FlashAttention: ${ROCM_ARCH} contains no gfx9 arch; using Triton attention" && \
        { \
            echo "AITER_REF=${AITER_REF}"; \
            echo "FLASH_ATTN_REF="; \
            python3 -c 'import importlib.util; spec = importlib.util.find_spec("aiter"); assert spec and spec.submodule_search_locations; print("AITER_PACKAGE=" + next(iter(spec.submodule_search_locations)))' 2>/dev/null || true; \
        } > /etc/rocm-aic-attention-backends.env; \
    fi
WORKDIR /

# ----- 3b. vLLM (built from source at pinned tag) --------------------------------

RUN git clone --depth 1 --branch "${VLLM_REF}" "${VLLM_GIT_URL}" /app/vllm

# Patch setup.py so _is_hip() short-circuits on VLLM_TARGET_DEVICE=rocm at
# build time, without probing torch.version.hip before it is installed.
# amd-quark (quantization tooling) has no release for Python 3.14; drop the pin.
RUN sed -i 's/def _is_hip() -> bool:/def _is_hip() -> bool:\n    if VLLM_TARGET_DEVICE == "rocm":\n        return True/' /app/vllm/setup.py && \
    grep -A2 '^def _is_hip() -> bool:$' /app/vllm/setup.py | \
        grep -Fxq '    if VLLM_TARGET_DEVICE == "rocm":' && \
    sed -i '/^amd-quark==/d' /app/vllm/requirements/rocm.txt && \
    ! grep -q '^amd-quark' /app/vllm/requirements/rocm.txt

# Apply every patch under patches/vllm/ in lexical order.
WORKDIR /app/vllm
# hadolint ignore=SC3054,SC3044,SC2098
RUN shopt -s nullglob && \
    patches=(/app/patches/vllm/*.patch) && \
    if [ "${#patches[@]}" -gt 0 ]; then \
        for patch in "${patches[@]}"; do \
            if git apply "$patch"; then \
                echo "Applied: $patch"; \
            else \
                echo "ERROR: vLLM patch does not apply cleanly: $patch" >&2; \
                echo "The vLLM pin (${VLLM_REF}) likely drifted; rebase this patch against it." >&2; \
                exit 1; \
            fi; \
        done; \
    fi

# --no-build-isolation means nothing installs vLLM's [build-system].requires.
# rocm.txt lists setuptools-rust/-scm itself; common.txt does not, so the
# `empty` path adds build/rust.txt for them.
# The constraint keeps pip from swapping the ROCm torch for a PyPI build.
#
# vLLM's extensions build as RelWithDebInfo; their .debug_* sections are
# stripped from the wheel before anything installs it, so neither aic-vllm nor
# the `wheels` export carries them.  The tools are looked up before the compile
# so a missing one costs seconds, not the whole build.
ARG PYTORCH_SERIES
# hadolint ignore=SC2098
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    : "${PYTORCH_SERIES:?unset: build through docker/docker-bake.hcl}" && \
    _strip="$(command -v llvm-strip || command -v strip)" && \
    python3 -m wheel version && \
    mkdir -p /wheels && \
    printf 'torch~=%s.0\n' "${PYTORCH_SERIES}" > /tmp/torch-constraints.txt && \
    if [ "${VLLM_TARGET_DEVICE}" = "empty" ]; then \
        pip3 install -r requirements/common.txt \
            -r requirements/build/rust.txt \
            --constraint /tmp/torch-constraints.txt; \
    else \
        pip3 install -r requirements/rocm.txt \
            --constraint /tmp/torch-constraints.txt; \
    fi && \
    VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE}" \
        PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        MAX_JOBS="${BUILD_JOBS:-$(nproc)}" \
        CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS:-$(nproc)}" \
        CCACHE_DIR=/root/.cache/ccache CCACHE_MAXSIZE=25G \
        VLLM_VERSION_OVERRIDE="${VLLM_REF#v}+${VLLM_TARGET_DEVICE}" \
        pip3 wheel --no-build-isolation --no-deps --no-cache-dir -w /wheels . && \
    python3 -m wheel unpack -d /tmp/vllm-whl /wheels/vllm-*.whl && \
    rm /wheels/vllm-*.whl && \
    find /tmp/vllm-whl -type f -name '*.so' -exec "${_strip}" --strip-debug {} + && \
    python3 -m wheel pack -d /wheels /tmp/vllm-whl/vllm-* && \
    rm -rf /tmp/vllm-whl && \
    pip3 install --no-cache-dir /wheels/vllm-*.whl && \
    python3 -c 'import importlib.metadata as m; print("vLLM installed:", m.version("vllm"))' && \
    rm -rf /app/vllm /tmp/torch-constraints.txt && \
    { find /usr/lib/python3 /usr/local/lib/python3.* -name '__pycache__' -type d \
         -exec rm -rf {} + || \
      echo "WARNING: __pycache__ cleanup failed; see find's error above" >&2; }
WORKDIR /

# ----- 3b. vllm_emulator (llm-emu profile-driven emulation plugin) -----------
ARG LLM_EMU_GIT_URL
ARG LLM_EMU_REF
ENV AIC_EMULATOR_EXAMPLE_PACK=/opt/llm-emu/profiles/A40-Q8-Qwen3-8B.json \
    AIC_EMULATOR_DEFAULT_PACK=/opt/llm-emu/profiles/MI300X-Qwen3-8B.json

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install build

# llm-emu was originally built for vllm 0.18
# However, our patches make it compatible with aic's vllm version
# Edit pyproject.toml to remove that version restriction
#
# The second check below builds the hook's fake ModelRunnerOutput against the
# installed vLLM (on the emulator platform, so it needs no GPU): at v0.34.0 the
# imports alone passed while the first scheduled step raised TypeError.
# hadolint ignore=SC3044
RUN : "${LLM_EMU_GIT_URL:?unset: build through docker/docker-bake.hcl}" \
      "${LLM_EMU_REF:?unset: build through docker/docker-bake.hcl}" && \
    git clone --filter=blob:none --no-checkout "${LLM_EMU_GIT_URL}" /app/llm-emu && \
    git -C /app/llm-emu checkout "${LLM_EMU_REF}" && \
    shopt -s nullglob && \
    for patch in /app/patches/llm-emu/*.patch; do \
        if git -C /app/llm-emu apply "$patch"; then \
            echo "Applied: $patch"; \
        else \
            echo "ERROR: llm-emu patch does not apply cleanly: $patch" >&2; \
            echo "The llm-emu pin (${LLM_EMU_REF}) likely drifted; rebase this patch against it." >&2; \
            exit 1; \
        fi; \
    done

WORKDIR /app/llm-emu

RUN sed -i 's/vllm==[0-9\.]\+/vllm/g' pyproject.toml && \
    python3 -m build --no-isolation --wheel --outdir /wheels/ && \
    pip3 install --no-cache-dir --no-deps /wheels/llm_emu-*.whl && \
    python3 -c "import vllm_emulator.hooks.executor_hook, vllm_emulator.oracle, \
vllm_emulator.profile.loader, vllm_emulator.profiler.trace_profiler; \
print('vllm_emulator subpackages import OK')" && \
    VLLM_EMULATOR_ENABLE_ORACLE=1 python3 -c "import importlib.util as u, types; \
from vllm_emulator.hooks.executor_hook import ExecutorEmulatorHook as H; \
assert u.find_spec('vllm.v1.worker.gpu.model_runner') or u.find_spec('vllm.v1.worker.gpu_model_runner'), 'vLLM model-runner module moved'; \
h = H.__new__(H); h._filler_token_id = 0; \
o = h._create_fake_output(types.SimpleNamespace(num_scheduled_tokens={'r': 1}, scheduled_new_reqs=[])); \
assert o.req_ids == ['r'] and o.sampled_token_ids == [[0]], o; \
print('vllm_emulator fake ModelRunnerOutput OK')" && \
    mkdir -p /opt/llm-emu/profiles && \
    cp /app/llm-emu/example_profiling_data/*.json /opt/llm-emu/profiles/ && \
    test -r "${AIC_EMULATOR_EXAMPLE_PACK}" && \
    rm -rf /app/llm-emu /app/patches /app/aiter /app/flash-attention && \
    { find /usr/lib/python3 /usr/local/lib/python3.* -name '__pycache__' -type d \
         -exec rm -rf {} + || \
      echo "WARNING: __pycache__ cleanup failed; see find's error above" >&2; }

WORKDIR /

# =============================================================================
# wheels — scratch image holding just the wheels built above:
#   vllm, amd_aiter, and (CDNA-only arch sets) flash_attn.
#
# The `vllm` runtime stage below is FROM base and copies only dist-packages,
# so /wheels never reaches a published image; it lives in build_vllm alone.
#
#   docker buildx bake -f docker/docker-bake.hcl wheels-vllm
# =============================================================================
FROM scratch AS wheels
COPY --from=build_vllm /wheels /

# =============================================================================
# vllm — slim runtime stage: aic-base + packages built in build_vllm
# =============================================================================
FROM base AS vllm

# pip3 comes with the base: aic-base installs python3-pip once for this image
# and aic-lmcache, and the emulate base is a python image.
#
# When building emulated, the base context does not contain any pytorch wheels
#
# No bytecode is shipped (--no-compile, and PYTHONDONTWRITEBYTECODE on every
# build-time python3 below): it is about 400 MB of layers that a root container
# regenerates on first import, which costs a few seconds once.  A non-root
# container (the emulator services under AIC_CONTAINER_USER, the accuracy
# client) cannot write into dist-packages, so it pays that on every start.
#
# The uninstall drops distributions vLLM declares but this stack never imports:
# the Mooncake connector, the runai_streamer load format with its cloud SDKs,
# tilelang (and z3), HF `datasets` benchmarks, sentry and the FastAPI cloud
# CLI.  It shares the RUN with the install so the files never reach a layer;
# `pip3 check` reports them as missing requirements from here on.  pip skips a
# name that is not installed, which the emulate build relies on.  The
# imports run on llm-emu's emulator platform because vLLM's ROCm platform
# module needs a GPU to import.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,from=build_vllm,source=/wheels/,target=/wheels-vllm \
    --mount=type=bind,from=base-wheels,source=/,target=/base-wheels \
    pip3 install --break-system-packages --no-compile --ignore-installed packaging && \
    find /base-wheels /wheels-vllm -type f -name "*.whl" -exec pip3 install --break-system-packages --no-compile {} + && \
    pip3 uninstall -y --break-system-packages \
        tilelang z3-solver ml-dtypes \
        mooncake-transfer-engine-rocm \
        runai-model-streamer runai-model-streamer-s3 \
        runai-model-streamer-gcs runai-model-streamer-azure \
        google-cloud-storage google-resumable-media google-cloud-core google-crc32c \
        azure-storage-blob azure-identity azure-core msal msal-extensions \
        datasets pyarrow multiprocess \
        supervisor fastapi-cloud-cli sentry-sdk rignore fastar && \
    PYTHONDONTWRITEBYTECODE=1 VLLM_EMULATOR_ENABLE_ORACLE=1 python3 -c 'import vllm.entrypoints.openai.api_server, vllm.benchmarks.serve, vllm_emulator.hooks.executor_hook; print("vLLM serving imports OK without the unused backends")'

COPY --from=build_vllm /usr/local/bin/vllm /usr/local/bin/vllm
COPY --from=build_vllm /opt/llm-emu/      /opt/llm-emu/
COPY --from=build_vllm /etc/rocm-aic-attention-backends.env \
                       /etc/rocm-aic-attention-backends.env
# Normalise the shebang: pip writes an absolute interpreter path from the build
# stage; #!/usr/bin/env python3 works on both the ROCm and CPU emulate bases.
RUN sed -i '1s|.*|#!/usr/bin/env python3|' /usr/local/bin/vllm

ARG VLLM_REF
ARG AITER_REF
ARG FLASH_ATTN_REF
ARG LLM_EMU_REF
ENV AIC_EMULATOR_EXAMPLE_PACK=/opt/llm-emu/profiles/A40-Q8-Qwen3-8B.json \
    AIC_EMULATOR_DEFAULT_PACK=/opt/llm-emu/profiles/MI300X-Qwen3-8B.json

# ----- Runtime ENV + metadata -------------------------------------------------
# PYTHONPATH and LD_LIBRARY_PATH (including NIXL paths) are inherited from
# aic-base; do not redefine them here or the NIXL entries will be lost.
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    SAFETENSORS_FAST_GPU=1 \
    HIP_FORCE_DEV_KERNARG=1

COPY profiles/*.json /opt/llm-emu/profiles/
RUN test -r "${AIC_EMULATOR_DEFAULT_PACK}" || \
    { echo "ERROR: default profile pack missing: ${AIC_EMULATOR_DEFAULT_PACK}" >&2; exit 1; }

ARG AIC_VERSION
ARG ROCM_BASE_IMAGE
ARG ROCM_VERSION
RUN : "${AIC_VERSION:?unset: build through docker/docker-bake.hcl}" \
      "${ROCM_VERSION:?unset: build through docker/docker-bake.hcl}" \
      "${ROCM_BASE_IMAGE:?unset: build through docker/docker-bake.hcl}"
LABEL org.opencontainers.image.title="aic-vllm" \
      org.opencontainers.image.description="AMD Infinity Context: ROCm vLLM inference service" \
      org.opencontainers.image.version="${AIC_VERSION}" \
      org.opencontainers.image.base.name="${ROCM_BASE_IMAGE}" \
      ai.amd.aic.version="${AIC_VERSION}" \
      ai.amd.aic.rocm="${ROCM_VERSION}" \
      ai.amd.aic.vllm="${VLLM_REF}" \
      ai.amd.aic.llm-emu="${LLM_EMU_REF}" \
      ai.amd.aic.aiter="${AITER_REF}" \
      ai.amd.aic.flash-attention="${FLASH_ATTN_REF}"

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

# =============================================================================
# ----- 3c. emulate — emulation-only runtime image (--target emulate) ---------
# =============================================================================
FROM vllm AS emulate

# ----- LMCache client (KV transfer connector PyPI build, emulate target only) --
# Provides LMCacheMPConnector, which vLLM loads in-process for
# --kv-transfer-config.  It must be the same release as the aic-lmcache server:
# the MP wire protocol carries auto-numbered opcodes and positionally-decoded
# msgpack payloads with no version negotiation, so a mismatched pair misroutes
# calls rather than failing cleanly.
#
# docker/docker-bake.hcl owns LMCACHE_REF and passes it to both builds, so this
# carries no default and must not be given one.  No build isolation needed:
# torch is already here.
#
# Only this target installs the PyPI client; `runtime` installs aic-lmcache's
# wheel instead, so neither image carries lmcache in two layers.  The PyPI
# wheel is the CUDA 13 build: it pulls cupy-cuda13x and cuda-python, dropped
# below.
#
# LMCache caps opentelemetry-api at 1.40.0 (and prometheus_client at 0.24.1)
# while vLLM v0.34.0 asks for opentelemetry >= 1.45.1, so pip downgrades the
# opentelemetry set here and reports the conflict.  LMCache's metrics export
# breaks on the newer set; vLLM only uses opentelemetry for optional tracing.
# The two 0.66b2 distributions belong to the newer set and are left orphaned
# by the downgrade, so they go too; the checks fail the build if vLLM's tracing
# module cannot load its exporters on the downgraded set, or if the client is
# not the pinned release.
#
# The constraint file holds numpy, setuptools and torch at what the image
# already has, so resolving LMCache's dependencies fails instead of moving
# them.  LMCache's S3 and Bigtable adapters and nvtx are unused in the vLLM
# process and go the same way as the vLLM extras above.
ARG LMCACHE_REF
RUN --mount=type=cache,target=/root/.cache/pip \
    : "${LMCACHE_REF:?unset: build through docker/docker-bake.hcl}" && \
    _ver="${LMCACHE_REF#v}" && \
    export PYTHONDONTWRITEBYTECODE=1 && \
    python3 -c 'import importlib.metadata as m; print(*(f"{n}=={m.version(n)}" for n in ("numpy", "setuptools", "torch")), sep="\n")' > /tmp/held-constraints.txt && \
    pip3 install --break-system-packages --no-compile \
        --constraint /tmp/held-constraints.txt "lmcache==${_ver}" && \
    rm /tmp/held-constraints.txt && \
    pip3 uninstall -y --break-system-packages \
        nixl nixl-cu12 nixl-cu13 \
        cupy-cuda12x cupy-cuda13x cuda-python cuda-bindings cuda-core \
        cufile-python cuda-pathfinder \
        opentelemetry-exporter-otlp-common \
        opentelemetry-exporter-http-transport \
        awscrt nvtx \
        google-cloud-bigtable google-cloud-monitoring grpc-google-iam-v1 \
        google-cloud-core google-crc32c && \
    python3 -c 'import importlib.metadata as m, sys; v = m.version("lmcache"); assert v == sys.argv[1], f"lmcache {v} installed, expected {sys.argv[1]}"' "${_ver}" && \
    VLLM_EMULATOR_ENABLE_ORACLE=1 python3 -c 'import vllm.entrypoints.openai.api_server, lmcache.integration.vllm.vllm_multi_process_adapter; print("vLLM + lmcache MP adapter imports OK")' && \
    python3 -c 'import opentelemetry.exporter.otlp.proto.grpc.trace_exporter, opentelemetry.exporter.otlp.proto.http.trace_exporter, vllm.tracing.otel as o; assert o.is_otel_available(), "vLLM tracing cannot load opentelemetry"; print("vLLM tracing imports OK on opentelemetry", __import__("importlib.metadata").metadata.version("opentelemetry-api"))'

# gfx1201: apply the event-IPC producer-side-sync patch to the lmcache CLIENT
# installed above.  vLLM is the producer on the store path, so its export_event
# must synchronize locally + emit b"" (hipIpcOpenEventHandle is broken on
# gfx1201 — see tests/ipc/), and its import_event must never open the server's
# handle.  The lmcache image runs the same script on its source tree before
# building the wheel, so `runtime` gets the edit with that wheel.
# AIC_LMCACHE_SRC points the patcher at the pip-installed package
# (site-packages), not the /app source tree; it exits non-zero if an anchor is
# missing.
COPY docker/lmcache/rocm_event_ipc_patch.py /tmp/rocm_event_ipc_patch.py
RUN export PYTHONDONTWRITEBYTECODE=1 && \
    AIC_LMCACHE_SRC="$(python3 -c 'import lmcache, os; print(os.path.dirname(os.path.dirname(lmcache.__file__)))')" \
        python3 /tmp/rocm_event_ipc_patch.py && \
    rm /tmp/rocm_event_ipc_patch.py

ENV PYTHONPATH=/opt/rocm/share/amd_smi
COPY profiles/*.json /opt/llm-emu/profiles/
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

# /health is 200 only while the engine is alive (503 once it has died).  Port
# 8000 is what the compose service serves on; the emulated model loads fast.
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=20 \
    CMD ["curl", "-fsS", "--noproxy", "*", "-o", "/dev/null", "http://127.0.0.1:8000/health"]

# ----- 3d. runtime: default target -------------------------------------------
# vLLM's LMCacheMPConnector imports lmcache in the vLLM process, so this image
# installs the same lmcache wheel aic-lmcache runs: one build, with its HIP
# extension, on both ends of the MP protocol.  Installed after vLLM, as the
# pre-split image did.
# The shared `vllm` stage has no lmcache, so this is the image's only copy and
# the install does what `emulate` does for the PyPI client: the same held
# constraints, opentelemetry downgrade, uninstall list and checks (see there).
# The direct_url.json assertion is the proof the wheel, not a PyPI build, went
# in; the wheel already carries the event-IPC edit.
# Do not try to import the vllm lmcache_mp_connector at this step, since it
# requires a GPU to be found by amdsmi. This import check will be done during
# the smoke test when a GPU is present in the container.
FROM vllm AS runtime
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# The wheel requires a bare cupy-rocm-7-0; pin it to what aic-lmcache installs.
ARG CUPY_VERSION
ARG LMCACHE_REF
RUN : "${CUPY_VERSION:?unset: build through docker/docker-bake.hcl}" \
      "${LMCACHE_REF:?unset: build through docker/docker-bake.hcl}"

# Install make, needed by aiter
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        make

# hadolint ignore=SC3054,SC3044
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,from=lmcache-wheels,target=/aic-wheels \
    shopt -s nullglob && \
    lmc=(/aic-wheels/lmcache-*.whl) && \
    if [ "${#lmc[@]}" -ne 1 ]; then \
        echo "ERROR: expected one lmcache wheel in the lmcache-wheels build context, found:" >&2; \
        ls -l /aic-wheels >&2; \
        echo "Pass --build-context lmcache-wheels=DIR, exported by: docker buildx build -f docker/lmcache/Dockerfile --target wheels --output type=local,dest=DIR ." >&2; \
        exit 1; \
    fi && \
    export PYTHONDONTWRITEBYTECODE=1 && \
    python3 -c 'import importlib.metadata as m; print(*(f"{n}=={m.version(n)}" for n in ("numpy", "setuptools", "torch")), sep="\n")' > /tmp/held-constraints.txt && \
    printf 'cupy-rocm-7-0==%s\n' "${CUPY_VERSION}" >> /tmp/held-constraints.txt && \
    pip3 install --break-system-packages --no-compile \
        --constraint /tmp/held-constraints.txt "${lmc[@]}" && \
    rm /tmp/held-constraints.txt && \
    pip3 uninstall -y --break-system-packages \
        nixl nixl-cu12 nixl-cu13 \
        cupy-cuda12x cupy-cuda13x cuda-python cuda-bindings cuda-core \
        cufile-python cuda-pathfinder \
        opentelemetry-exporter-otlp-common \
        opentelemetry-exporter-http-transport \
        awscrt nvtx \
        google-cloud-bigtable google-cloud-monitoring grpc-google-iam-v1 \
        google-cloud-core google-crc32c && \
    python3 -c 'import importlib.metadata as m, sys; d = sorted(x.name for x in m.distributions() if x.name.startswith("cupy")); assert d == ["cupy-rocm-7-0"], f"expected only cupy-rocm-7-0, found {d}"; v = m.version("cupy-rocm-7-0"); assert v == sys.argv[1], f"cupy-rocm-7-0 {v} installed, expected {sys.argv[1]}"' "${CUPY_VERSION}" && \
    python3 -c 'import importlib.metadata as m, sys; d = [x for x in m.distributions() if x.name == "lmcache"]; assert len(d) == 1 and d[0].read_text("direct_url.json"), "lmcache is not the single aic-lmcache wheel build"; v = d[0].version; assert v == sys.argv[1], f"lmcache {v} installed, expected {sys.argv[1]}"' "${LMCACHE_REF#v}" && \
    VLLM_EMULATOR_ENABLE_ORACLE=1 python3 -c 'import vllm.entrypoints.openai.api_server, vllm.benchmarks.serve, lmcache.integration.vllm.vllm_multi_process_adapter; print("vLLM + lmcache MP adapter imports OK")' && \
    python3 -c 'import opentelemetry.exporter.otlp.proto.grpc.trace_exporter, opentelemetry.exporter.otlp.proto.http.trace_exporter, vllm.tracing.otel as o; assert o.is_otel_available(), "vLLM tracing cannot load opentelemetry"; print("vLLM tracing imports OK on opentelemetry", __import__("importlib.metadata").metadata.version("opentelemetry-api"))'

# Install make and other packages needed by aiter to build kernels JIT
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    _mm=$(echo "${ROCM_VERSION}" | cut -d. -f1-2) && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        make \
        "amdrocm-blas-dev${_mm}" \
        "amdrocm-ccl-dev${_mm}" \
        "amdrocm-hipblas-common-dev${_mm}" \
        "amdrocm-solver-dev${_mm}" \
        "amdrocm-sparse-dev${_mm}"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages --no-compile amdsmi

# For fastsafetensors' dlopen("libamdhip64.so").  Not in the shared `vllm`
# stage: CI's emulate build swaps in a base with no ROCm.  Prepend, so
# aic-base's NIXL/UCX entries survive.
ENV LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
RUN PYTHONDONTWRITEBYTECODE=1 python3 -c 'import ctypes; ctypes.CDLL("libamdhip64.so")'

# ----- MORI / UMBP (AIC_KV_BACKEND=mori) --------------------------------------
# The `umbp` and `umbp-master` compose services run this image too, so the two
# binaries are linked onto PATH.  `import mori` is left to the smoke test: like
# vLLM's ROCm platform module it may need a GPU.  The ldd check is what catches
# a runtime library missing from the list below.
# /run/umbp holds the server's sockets; world-writable so the named volume
# compose mounts there takes that mode and a non-root container can bind.
ARG MORI_REF
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ibverbs-providers \
        libgrpc++1.51t64 \
        libibverbs1 \
        libnuma1 \
        libpci3 \
        libprotobuf32t64

# hadolint ignore=SC3054,SC3044
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,from=build_mori,source=/wheels,target=/wheels-mori \
    : "${MORI_REF:?unset: build through docker/docker-bake.hcl}" && \
    pip3 install --break-system-packages --no-compile --no-deps /wheels-mori/amd_mori-*.whl && \
    _pkg="$(PYTHONDONTWRITEBYTECODE=1 python3 -c 'import importlib.util as u; print(next(iter(u.find_spec("mori").submodule_search_locations)))')" && \
    for _bin in umbp_standalone_server umbp_master; do \
        test -x "${_pkg}/${_bin}" || { echo "ERROR: ${_pkg}/${_bin} missing" >&2; exit 1; }; \
        _missing="$(ldd "${_pkg}/${_bin}" | grep 'not found' || true)"; \
        [ -z "${_missing}" ] || { echo "ERROR: ${_bin} has unresolved libraries:" >&2; echo "${_missing}" >&2; exit 1; }; \
        ln -s "${_pkg}/${_bin}" "/usr/local/bin/${_bin}"; \
    done && \
    mkdir -p /run/umbp && chmod 1777 /run/umbp
LABEL ai.amd.aic.mori="${MORI_REF}"

# Same probe as emulate.  start-period is sized for loading and compiling a
# large model (gpt-oss-120b); the first success marks the container healthy
# immediately, so a long period only delays unhealthy reporting.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15m --retries=5 \
    CMD ["curl", "-fsS", "--noproxy", "*", "-o", "/dev/null", "http://127.0.0.1:8000/health"]
