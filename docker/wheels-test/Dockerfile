# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# CI-only test harness: rebuild the aic-vllm and aic-lmcache runtime images from
# PUBLISHED pip wheels instead of from source.
#
# Why this exists
# ---------------
# The nightly wheels release is, today, an untested by-product: it is built,
# checked for file names by .github/scripts/workflows/check-wheels.sh, and
# uploaded.  Nothing ever installs it and runs a kernel.  This Dockerfile is the
# missing half -- .github/workflows/rocm-aic-gfx1250-wheels-test.yml installs
# the released wheels onto the published aic-base image and drives the rocjitsu
# gfx1250 VM test against the result, so a broken wheel fails CI rather than a
# downstream user's `pip install`.
#
# It also keeps the GitHub-hosted runner within its disk budget: aic-base is
# pulled once and the two heavy images (aic-vllm, aic-lmcache) are reconstituted
# from ~180 MB of wheels rather than pulled whole.
#
# THIS IS NOT A PRODUCT IMAGE.  docker/vllm/Dockerfile and
# docker/lmcache/Dockerfile remain the single source of truth for what ships.
# This file reproduces the FINAL STATE of their `runtime` stages from wheels,
# and it will drift from them.  The drift is deliberately made loud: the
# verification step in the workflow imports the same symbols those Dockerfiles
# assert on, so a missing piece fails here instead of silently serving a
# different stack than production.  When either runtime stage changes, change
# this too.
#
# Everything torch-shaped comes from aic-base
# -------------------------------------------
# docker/base/Dockerfile's `runtime` stage already carries torch + torchvision +
# NIXL + the 13 ROCm runtime packages, all compiled for ROCM_ARCH.  That is why
# a wheels-only path is viable at all: there is no gfx1250 torch on PyPI, and
# none is needed, because aic-base has one.
#
# Build (both targets need the same two contexts):
#   docker buildx build -f docker/wheels-test/Dockerfile --target vllm \
#     --build-arg AIC_BASE_IMAGE=ghcr.io/rocm/rocm-aic/aic-base:gfx1250 \
#     --build-context wheels=/path/to/downloaded/wheels \
#     -t aic-vllm:<tag> .

# hadolint ignore=DL3006
ARG AIC_BASE_IMAGE
# The wheel set downloaded from the release, as a build context.
FROM scratch AS wheels

# =============================================================================
# vllm — aic-base + the vllm / amd_aiter / lmcache wheels.
#
# Mirrors the end state of docker/vllm/Dockerfile's `runtime` stage: vLLM and
# AITER from their own wheels, plus the SAME locally-built lmcache wheel the
# aic-lmcache server runs.  That last point is load-bearing and is why the
# lmcache wheel is installed here from the release rather than from PyPI: the
# LMCacheMPConnector wire protocol carries auto-numbered opcodes and
# positionally-decoded msgpack payloads with no version negotiation, so a
# mismatched client/server pair misroutes calls instead of failing cleanly.
# =============================================================================
# hadolint ignore=DL3006
FROM ${AIC_BASE_IMAGE} AS vllm

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Fail loudly and early on an incomplete wheel set.  Without this the install
# below "succeeds" against whatever pip can reach on PyPI -- which for `vllm` is
# a CUDA build that imports fine and dies at the first kernel launch.  That is
# exactly the failure docs/PIP_WHEELS.md warns about, so it must not be
# reachable from CI.
# hadolint ignore=SC2015
RUN --mount=type=bind,from=wheels,target=/w \
    shopt -s nullglob && \
    for pat in 'vllm-*.whl' 'amd_aiter-*.whl' 'lmcache-*.whl' 'nixl*rocm*.whl'; do \
        hits=(/w/${pat}); \
        if [ "${#hits[@]}" -ne 1 ]; then \
            echo "ERROR: expected exactly one ${pat} in the wheels context, found ${#hits[@]}" >&2; \
            ls -l /w >&2; exit 1; \
        fi; \
    done && \
    echo "wheel set OK"

# Install the AIC wheels WITH their dependencies, but under a constraint file
# that pins torch/torchvision to the versions aic-base already carries.
#
# The constraint is the whole trick.  aic-base's torch is a source build for
# ROCM_ARCH and exists nowhere on PyPI, so an unconstrained resolve sees vLLM's
# `torch>=X` requirement unsatisfied-looking and installs the PyPI CUDA wheel
# over it -- which imports fine and dies at the first kernel launch.  Pinning to
# the exact local version makes the requirement resolve against what is already
# installed instead.
#
# Deliberately NOT --no-deps: vLLM's pure-Python requirements (msgspec, openai,
# transformers, ...) are not in aic-base and the server does not start without
# them.
# hadolint ignore=SC2015
RUN --mount=type=bind,from=wheels,target=/w \
    --mount=type=cache,target=/root/.cache/uv \
    _tv="$(python3 -c 'import torch; print(torch.__version__)')" && \
    _tvv="$(python3 -c 'import torchvision; print(torchvision.__version__)' 2>/dev/null || true)" && \
    echo "pinning to aic-base torch=${_tv} torchvision=${_tvv:-<absent>}" && \
    { printf 'torch==%s\n' "${_tv}"; \
      [ -n "${_tvv}" ] && printf 'torchvision==%s\n' "${_tvv}"; } > /tmp/c.txt && \
    uv pip install --system --no-cache --constraint /tmp/c.txt \
        /w/vllm-*.whl /w/amd_aiter-*.whl /w/lmcache-*.whl && \
    rm -f /tmp/c.txt && \
    _after="$(python3 -c 'import torch; print(torch.__version__)')" && \
    if [ "${_after}" != "${_tv}" ]; then \
        echo "ERROR: torch changed from ${_tv} to ${_after} during the wheel install." >&2; \
        echo "       aic-base's ROCm torch was replaced; the image would fail at first kernel launch." >&2; \
        exit 1; \
    fi && \
    echo "torch intact: ${_after}"

# Drop the CUDA-flavoured packages the wheels' metadata may have dragged in.
# aic-base supplies the ROCm NIXL at /opt/nixl; a pip `nixl` would shadow it.
RUN { uv pip uninstall --system \
        nixl nixl-cu12 nixl-cu13 \
        cupy-cuda12x cufile-python cuda-pathfinder 2>/dev/null || true; }

# Post-install patches applied to the lmcache CLIENT, mirroring
# docker/vllm/Dockerfile's `vllm` stage.  vLLM is the producer on the store
# path, so its export_event must synchronize locally.
COPY docker/lmcache/rocm_event_ipc_patch.py /tmp/rocm_event_ipc_patch.py
COPY docker/lmcache/diag_register_kv_caches.py /tmp/diag_register_kv_caches.py
RUN AIC_LMCACHE_SRC="$(python3 -c 'import lmcache, os; print(os.path.dirname(os.path.dirname(lmcache.__file__)))')" \
        python3 /tmp/rocm_event_ipc_patch.py && \
    AIC_LMCACHE_SRC="$(python3 -c 'import lmcache, os; print(os.path.dirname(os.path.dirname(lmcache.__file__)))')" \
        python3 /tmp/diag_register_kv_caches.py && \
    rm /tmp/rocm_event_ipc_patch.py /tmp/diag_register_kv_caches.py

# PYTHONPATH and LD_LIBRARY_PATH (including the NIXL entries) are inherited from
# aic-base; do not redefine them or those entries are lost.  LD_LIBRARY_PATH is
# PREPENDED, matching docker/vllm/Dockerfile's runtime stage, for
# fastsafetensors' dlopen("libamdhip64.so").
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    SAFETENSORS_FAST_GPU=1 \
    HIP_FORCE_DEV_KERNARG=1
ENV LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

RUN python3 -c 'import ctypes; ctypes.CDLL("libamdhip64.so")'

ARG AIC_VERSION
ARG AIC_BASE_IMAGE
LABEL org.opencontainers.image.title="aic-vllm" \
      org.opencontainers.image.description="AMD Infinity Context: ROCm vLLM, reconstituted from published wheels (CI test harness)" \
      org.opencontainers.image.version="${AIC_VERSION}" \
      org.opencontainers.image.base.name="${AIC_BASE_IMAGE}" \
      ai.amd.aic.version="${AIC_VERSION}" \
      ai.amd.aic.built-from="published-wheels"

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

# =============================================================================
# lmcache — aic-base + the lmcache wheel and its non-wheel runtime deps.
#
# Mirrors docker/lmcache/Dockerfile's `runtime` stage.  The ROCm IPC patches are
# NOT re-applied here: they are compiled into the lmcache wheel itself by that
# file's build_lmcache_wheel stage, so the published wheel already carries them.
# =============================================================================
# hadolint ignore=DL3006
FROM ${AIC_BASE_IMAGE} AS lmcache

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN --mount=type=bind,from=wheels,target=/w \
    --mount=type=cache,target=/root/.cache/uv \
    shopt -s nullglob && \
    lmc=(/w/lmcache-*.whl) && \
    if [ "${#lmc[@]}" -ne 1 ]; then \
        echo "ERROR: expected exactly one lmcache wheel, found ${#lmc[@]}" >&2; \
        ls -l /w >&2; exit 1; \
    fi && \
    _torch_ver="$(python3 -c 'import torch; print(torch.__version__)')" && \
    printf 'torch==%s\n' "${_torch_ver}" > /tmp/c.txt && \
    uv pip install --system --no-cache --constraint /tmp/c.txt "${lmc[@]}" && \
    rm -f /tmp/c.txt

# The `lmcache` CLI imports every subcommand eagerly, so `lmcache server` dies
# without the bench subcommands' openai.  Upstream lists those deps only in
# requirements/cli.txt (the lmcache-cli package), not in this wheel's metadata.
# This is the same gap mk/bench.mk's [6b.1] hot-patch works around in the guest;
# installing it here is what lets that hot-patch become a no-op.
#
# cupy-rocm-7-0 activates the CuPy GPU buffer allocator (LMCACHE_NIXL_USE_CUPY=1
# below) so librocblas/libMIOpen are never dlopened in the lmcache process.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache \
        openai "matplotlib>=3.8,<4" cupy-rocm-7-0 && \
    { uv pip uninstall --system \
        nixl nixl-cu12 nixl-cu13 \
        cupy-cuda12x cufile-python cuda-pathfinder 2>/dev/null || true; }

# HSA_ENABLE_IPC_MODE_LEGACY is intentionally NOT set: aic-vllm leaves it unset
# too, and the two must agree.
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    SAFETENSORS_FAST_GPU=1 \
    HIP_FORCE_DEV_KERNARG=1 \
    LMCACHE_NIXL_USE_CUPY=1

ARG AIC_VERSION
ARG AIC_BASE_IMAGE
LABEL org.opencontainers.image.title="aic-lmcache" \
      org.opencontainers.image.description="AMD Infinity Context: ROCm LMCache, reconstituted from published wheels (CI test harness)" \
      org.opencontainers.image.version="${AIC_VERSION}" \
      org.opencontainers.image.base.name="${AIC_BASE_IMAGE}" \
      ai.amd.aic.version="${AIC_VERSION}" \
      ai.amd.aic.built-from="published-wheels"

ENTRYPOINT ["python3", "-m", "lmcache.server"]
