# check=skip=InvalidDefaultArgInFrom
# ROCM_VERSION deliberately carries no default here -- docker/base/Dockerfile
# is its single source of truth and the Makefile forwards it to every build.
# That leaves the fallback ROCM_BASE_IMAGE default unresolvable at lint time,
# which is harmless: the `base` stage is always replaced by the base
# build-context supplied by `make build`.
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# AIC vLLM image: torch + torchvision + NIXL (from aic-base) + AITER +
# FlashAttention + vLLM v0.29.0 + llm-emu plugin.
#
# Consumed by the `vllm` compose service.  Shares aic-base with aic-lmcache, but
# its default target needs aic-lmcache's wheels, so build aic-lmcache first.
#
# Stages:
#   vllm      AITER + FlashAttention + vLLM source build + llm-emu  (sections 3-3b)
#   wheels    scratch image holding just the vLLM/AITER/FA wheels   (--target wheels)
#   emulate   CPU-only runtime (--target emulate)                   (section 3c)
#   runtime   default target: `vllm` + aic-lmcache's lmcache wheel  (section 3d)
#
# Build (from repo root — build context is this tree):
#   BASE_TAG=$(ROCM_ARCH=gfx950 docker/scripts/aic-image-tag.sh docker/base/Dockerfile)
#   DOCKER_BUILDKIT=1 docker buildx build \
#     --build-arg ROCM_ARCH=gfx950 \
#     --build-context base=docker-image://aic-base:${BASE_TAG} \
#     -f docker/lmcache/Dockerfile --target wheels \
#     --output type=local,dest=/tmp/aic-lmcache-wheels \
#     .
#   DOCKER_BUILDKIT=1 docker buildx build \
#     --build-arg ROCM_ARCH=gfx950 \
#     --build-context base=docker-image://aic-base:${BASE_TAG} \
#     --build-context lmcache-wheels=/tmp/aic-lmcache-wheels \
#     -f docker/vllm/Dockerfile \
#     -t aic-vllm:${BASE_TAG} \
#     .
#
# Or via make:
#   make dist-build-vllm ROCM_ARCH=gfx950
#
# Emulation-only (no GPU kernels):
#   DOCKER_BUILDKIT=1 docker buildx build \
#     --target emulate \
#     --build-arg VLLM_TARGET_DEVICE=empty \
#     --build-context base=docker-image://aic-base:${BASE_TAG} \
#     -f docker/vllm/Dockerfile \
#     -t aic-vllm:emulate .

# Fallback base stage: used when no --build-context base= is provided (e.g.
# when building this Dockerfile standalone without a pre-built aic-base image).
# When --build-context base=docker-image://aic-base:<TAG> IS supplied, BuildKit
# overrides this stage with the pre-built image and skips this FROM entirely.
# Single source of truth: ARG ROCM_VERSION in docker/base/Dockerfile.
# The Makefile reads it from there and forwards it to every build, so this
# carries no default and must not be given one -- a second literal here is
# exactly the drift this removes.
ARG ROCM_VERSION
# ROCM_BASE_IMAGE is overridden by --build-context base=oci-layout://... (make build)
# or by --build-arg ROCM_BASE_IMAGE=<ref> (dist build).  The default intentionally
# references the -full image so standalone builds still work; pass the named
# build-context to avoid pulling it.
ARG ROCM_BASE_IMAGE=rocm/dev-ubuntu-24.04:${ROCM_VERSION}-full
# hadolint ignore=DL3006
FROM ${ROCM_BASE_IMAGE} AS base

# Empty default for the `lmcache-wheels` build context (the --target wheels
# export of docker/lmcache/Dockerfile).  Only `runtime` reads it, and fails on
# an empty one, so a build that forgot the context stops instead of shipping a
# vLLM that cannot load LMCacheMPConnector.  `emulate` never reads it.
FROM scratch AS lmcache-wheels

# =============================================================================
# build_vllm — compile AITER, FlashAttention, and vLLM against the full ROCm
# image (which has hipcc + HIP CMake files).  Torch is COPYed in from aic-base
# so the built wheels are compatible with the slim runtime stage.
# =============================================================================
FROM ${ROCM_BASE_IMAGE} AS build_vllm

# hadolint ignore=DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ROCM_ARCH
ARG BUILD_JOBS=
ARG VLLM_GIT_URL=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=v0.29.0
ARG VLLM_TARGET_DEVICE=rocm
ARG AITER_GIT_URL=https://github.com/ROCm/aiter.git
ARG AITER_REF=v0.1.22.post1
ARG FLASH_ATTN_GIT_URL=https://github.com/Dao-AILab/flash-attention.git
ARG FLASH_ATTN_REF=v2.8.3.post1

# git and ca-certificates are not in the -full image; install before cloning.
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cmake git ca-certificates ninja-build python3-dev

# Bring torch + Python packages from aic-base so wheels link against the same torch.
COPY --from=base /usr/local/lib/python3.12/dist-packages/ \
                 /usr/local/lib/python3.12/dist-packages/
COPY --from=base /usr/local/bin/uv  /usr/local/bin/uv
COPY --from=base /usr/local/bin/uvx /usr/local/bin/uvx
RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

COPY patches/vllm/ /app/patches/vllm/
COPY patches/llm-emu/ /app/patches/llm-emu/
COPY patches/aiter/ /app/patches/aiter/

# ----- 3. ROCm attention backends (AITER + FlashAttention) ----------------------
# Skipped when VLLM_TARGET_DEVICE=empty (CPU-only emulate build: no hipcc).
RUN test -n "$ROCM_ARCH" || { echo "ERROR: ROCM_ARCH is not set"; exit 1; }

# hadolint ignore=SC2039
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    if [ "${VLLM_TARGET_DEVICE}" = "empty" ]; then \
        echo "Skipping AITER build (VLLM_TARGET_DEVICE=empty)"; \
    else \
        mkdir -p /wheels && \
        pip3 install --no-cache-dir \
            "pandas==3.0.3" \
            "psutil==7.2.2" \
            "pyyaml==6.0.3" \
            "einops==0.8.2" \
            "pybind11==3.0.4" \
            "ninja==1.13.0" \
            "flydsl==0.2.4" && \
        git clone --depth 1 --branch "${AITER_REF}" "${AITER_GIT_URL}" /app/aiter && \
        cd /app/aiter && \
        git submodule update --init --recursive && \
        PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
            TORCH_DONT_CHECK_COMPILER_ABI=1 \
            MAX_JOBS="${BUILD_JOBS:-$(nproc)}" \
            CCACHE_DIR=/root/.cache/ccache CCACHE_MAXSIZE=25G \
            python3 setup.py bdist_wheel --dist-dir=/wheels && \
        pip3 install --no-cache-dir --no-deps /wheels/amd_aiter-*.whl && \
        python3 -c 'import importlib.util; spec = importlib.util.find_spec("aiter"); assert spec and spec.submodule_search_locations; print("AITER installed:", next(iter(spec.submodule_search_locations)))' && \
        PYTORCH_ROCM_ARCH="${ROCM_ARCH}" TORCH_DONT_CHECK_COMPILER_ABI=1 \
            python3 /app/patches/aiter/01-jit-use-pytorch-rocm-arch.py; \
    fi

# FlashAttention: CDNA-only (gfx9xx — MI100/MI200/MI300/MI355X).
# Skipped when VLLM_TARGET_DEVICE=empty (CPU-only emulate build).
# hadolint ignore=SC2039
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    _cdna_only() { \
        _a=""; for _a in $(echo "$1" | tr ';' '\n'); do \
            [ "${_a#gfx9}" != "${_a}" ] || return 1; done; return 0; }; \
    touch /etc/rocm-aic-attention-backends.env && \
    if [ "${VLLM_TARGET_DEVICE}" = "empty" ]; then \
        echo "Skipping FlashAttention (VLLM_TARGET_DEVICE=empty)"; \
    elif _cdna_only "${ROCM_ARCH}"; then \
        git clone --depth 1 --branch "${FLASH_ATTN_REF}" \
            --recurse-submodules --shallow-submodules \
            "${FLASH_ATTN_GIT_URL}" /app/flash-attention && \
        python3 /app/patches/vllm/02-flash-attn-deprecated-attrs.py && \
        cd /app/flash-attention && \
        GPU_ARCHS="${ROCM_ARCH}" \
            PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
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
        echo "Skipping FlashAttention: ${ROCM_ARCH} contains non-CDNA arch(es); using Triton attention" && \
        { \
            echo "AITER_REF=${AITER_REF}"; \
            echo "FLASH_ATTN_REF="; \
            python3 -c 'import importlib.util; spec = importlib.util.find_spec("aiter"); assert spec and spec.submodule_search_locations; print("AITER_PACKAGE=" + next(iter(spec.submodule_search_locations)))' 2>/dev/null || true; \
        } > /etc/rocm-aic-attention-backends.env; \
    fi

# ----- 3b. vLLM (built from source at pinned tag) --------------------------------

RUN git clone --depth 1 --branch "${VLLM_REF}" "${VLLM_GIT_URL}" /app/vllm

# Patch setup.py so _is_hip() short-circuits on VLLM_TARGET_DEVICE=rocm at
# build time, without probing torch.version.hip before it is installed.
RUN sed -i 's/def _is_hip() -> bool:/def _is_hip() -> bool:\n    if VLLM_TARGET_DEVICE == "rocm":\n        return True/' /app/vllm/setup.py && \
    grep -A2 '^def _is_hip() -> bool:$' /app/vllm/setup.py | \
        grep -Fxq '    if VLLM_TARGET_DEVICE == "rocm":'

# Apply every patch under patches/vllm/ in lexical order.
# hadolint ignore=SC3054,SC3044,SC2098
RUN cd /app/vllm && \
    shopt -s nullglob && \
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
# hadolint ignore=SC2098
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    mkdir -p /wheels && \
    cd /app/vllm && \
    printf 'torch>=2.13,<2.14\n' > /tmp/torch-constraints.txt && \
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
    pip3 install --no-cache-dir /wheels/vllm-*.whl && \
    python3 -c 'import importlib.metadata as m; print("vLLM installed:", m.version("vllm"))' && \
    rm -rf /app/vllm /tmp/torch-constraints.txt && \
    { find /usr/lib/python3 /usr/local/lib/python3.12 -name '__pycache__' -type d \
         -exec rm -rf {} + || \
      echo "WARNING: __pycache__ cleanup failed; see find's error above" >&2; }

# ----- 3b. vllm_emulator (llm-emu profile-driven emulation plugin) -----------
ARG LLM_EMU_GIT_URL=https://github.com/AKafakA/llm-emu.git
ARG LLM_EMU_REF=b71c602
ENV AIC_EMULATOR_EXAMPLE_PACK=/opt/llm-emu/profiles/A40-Q8-Qwen3-8B.json \
    AIC_EMULATOR_DEFAULT_PACK=/opt/llm-emu/profiles/MI300X-Qwen3-8B.json
# hadolint ignore=SC3044
RUN git clone --filter=blob:none --no-checkout "${LLM_EMU_GIT_URL}" /app/llm-emu && \
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
    done && \
    pip3 install --no-cache-dir --no-build-isolation --no-deps /app/llm-emu/ && \
    python3 -c "import vllm_emulator.hooks.executor_hook, vllm_emulator.oracle, \
vllm_emulator.profile.loader, vllm_emulator.profiler.trace_profiler; \
print('vllm_emulator subpackages import OK')" && \
    mkdir -p /opt/llm-emu/profiles && \
    cp /app/llm-emu/example_profiling_data/*.json /opt/llm-emu/profiles/ && \
    test -r "${AIC_EMULATOR_EXAMPLE_PACK}" && \
    rm -rf /app/llm-emu /app/patches /app/aiter /app/flash-attention && \
    { find /usr/lib/python3 /usr/local/lib/python3.12 -name '__pycache__' -type d \
         -exec rm -rf {} + || \
      echo "WARNING: __pycache__ cleanup failed; see find's error above" >&2; }

# =============================================================================
# wheels — scratch image holding just the vLLM, AITER and FlashAttention (CDNA
# only) pip wheels build_vllm installed (--target wheels; the `wheels-vllm` bake
# target).  build_vllm keeps /wheels for this: the stages below copy only
# site-packages, uv, /opt/llm-emu and the backends env file out of it, so the
# wheels never reach an image.
# =============================================================================
FROM scratch AS wheels
COPY --from=build_vllm /wheels/ /

# =============================================================================
# vllm — slim runtime stage: aic-base + packages built in build_vllm
# =============================================================================
FROM base AS vllm

# Copy all installed Python packages (vllm, aiter, flash_attn, llm-emu, deps)
# from the full-ROCm build stage into the slim aic-base runtime.
COPY --from=build_vllm /usr/local/lib/python3.12/dist-packages/ \
                       /usr/local/lib/python3.12/dist-packages/
COPY --from=build_vllm /usr/local/bin/uv  /usr/local/bin/uv
COPY --from=build_vllm /usr/local/bin/uvx /usr/local/bin/uvx
COPY --from=build_vllm /opt/llm-emu/      /opt/llm-emu/
COPY --from=build_vllm /etc/rocm-aic-attention-backends.env \
                       /etc/rocm-aic-attention-backends.env

ARG VLLM_REF
ARG AITER_REF
ARG FLASH_ATTN_REF
ARG LLM_EMU_REF
ENV AIC_EMULATOR_EXAMPLE_PACK=/opt/llm-emu/profiles/A40-Q8-Qwen3-8B.json \
    AIC_EMULATOR_DEFAULT_PACK=/opt/llm-emu/profiles/MI300X-Qwen3-8B.json

# ----- LMCache client (KV transfer connector used by vllm service) -----------
# Provides LMCacheMPConnector, which vLLM loads in-process for
# --kv-transfer-config.  It must be the same release as the aic-lmcache server:
# the MP wire protocol carries auto-numbered opcodes and positionally-decoded
# msgpack payloads with no version negotiation, so a mismatched pair misroutes
# calls rather than failing cleanly.
#
# Single source of truth: ARG LMCACHE_REF in docker/lmcache/Dockerfile.  The
# Makefile reads it there and forwards it here, so this carries no default and
# must not be given one.  No build isolation needed: torch is already here.
ARG LMCACHE_REF
# hadolint ignore=SC2015
RUN --mount=type=cache,target=/root/.cache/pip \
    test -n "${LMCACHE_REF}" || { \
        echo "ERROR: LMCACHE_REF is empty; build via 'make build'" >&2; exit 1; } && \
    _ver="${LMCACHE_REF#v}" && \
    pip3 install --no-cache-dir "lmcache==${_ver}" && \
    { uv pip uninstall --system \
        nixl nixl-cu12 nixl-cu13 \
        cupy-cuda12x cufile-python cuda-pathfinder 2>/dev/null || true; }

# gfx1201: apply the event-IPC producer-side-sync patch to the lmcache CLIENT
# installed above.  vLLM is the producer on the store path, so its export_event
# must synchronize locally + emit b"" (hipIpcOpenEventHandle is broken on
# gfx1201 — see tests/ipc/).  Only the event patch is applied here; the VMM
# memory patches belong to the lmcache server image.  AIC_LMCACHE_SRC points the
# patcher at the pip-installed package (site-packages), not the /app source tree.
COPY docker/lmcache/rocm_event_ipc_patch.py /tmp/rocm_event_ipc_patch.py
COPY docker/lmcache/diag_register_kv_caches.py /tmp/diag_register_kv_caches.py
RUN AIC_LMCACHE_SRC="$(python3 -c 'import lmcache, os; print(os.path.dirname(os.path.dirname(lmcache.__file__)))')" \
        python3 /tmp/rocm_event_ipc_patch.py && \
    AIC_LMCACHE_SRC="$(python3 -c 'import lmcache, os; print(os.path.dirname(os.path.dirname(lmcache.__file__)))')" \
        python3 /tmp/diag_register_kv_caches.py && \
    rm /tmp/rocm_event_ipc_patch.py /tmp/diag_register_kv_caches.py

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
RUN test -n "$AIC_VERSION" || { echo "ERROR: AIC_VERSION is not set (read it from VERSION)"; exit 1; }
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
ENV PYTHONPATH=/opt/rocm/share/amd_smi
COPY profiles/*.json /opt/llm-emu/profiles/
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

# ----- 3d. runtime: default target -------------------------------------------
# vLLM's LMCacheMPConnector imports lmcache in the vLLM process, so this image
# installs the same lmcache wheel aic-lmcache runs: one build, with its HIP
# extension, on both ends of the MP protocol.  Installed after vLLM, as the
# pre-split image did.
# Do not try to import the vllm lmcache_mp_connector at this step, since it
# requires a GPU to be found by amdsmi. This import check will be done during
# the smoke test when a GPU is present in the container.
FROM vllm AS runtime
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN --mount=type=bind,from=lmcache-wheels,target=/aic-wheels \
    shopt -s nullglob && \
    lmc=(/aic-wheels/lmcache-*.whl) && \
    if [ "${#lmc[@]}" -ne 1 ]; then \
        echo "ERROR: expected one lmcache wheel in the lmcache-wheels build context, found:" >&2; \
        ls -l /aic-wheels >&2; \
        echo "Pass --build-context lmcache-wheels=DIR, exported by: docker buildx build -f docker/lmcache/Dockerfile --target wheels --output type=local,dest=DIR ." >&2; \
        exit 1; \
    fi && \
    uv pip install --system --no-cache "${lmc[@]}" && \
    { uv pip uninstall --system \
        nixl nixl-cu12 nixl-cu13 \
        cupy-cuda12x cufile-python cuda-pathfinder 2>/dev/null || true; } && \
    python3 -c 'import lmcache.integration.vllm.vllm_multi_process_adapter; print("lmcache MP adapter imports OK")'

# For fastsafetensors' dlopen("libamdhip64.so").  Not in the shared `vllm`
# stage: CI's emulate build swaps in a base with no ROCm.  Prepend, so
# aic-base's NIXL/UCX entries survive.
ENV LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
RUN python3 -c 'import ctypes; ctypes.CDLL("libamdhip64.so")'
