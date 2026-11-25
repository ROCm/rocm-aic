# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Rebuild the aic-vllm and aic-lmcache runtime images from PUBLISHED pip wheels
# instead of from source.  Two callers:
#   - CI (rocm-aic-gfx1250-wheels-test.yml), on the published gfx1250 aic-base;
#   - `make build-from-wheels` (bake group `from-wheels`), on an aic-base that
#     docker/base/Dockerfile's runtime-from-wheels stage assembles from the
#     same wheel set, for a node whose arch has no published aic-base.
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

# Default so a bare `docker build` resolves, and so linters do not read the
# FROM as blank.  The workflow always passes --build-arg AIC_BASE_IMAGE.
ARG AIC_BASE_IMAGE=ghcr.io/rocm/rocm-aic/aic-base:gfx1250
# The wheel set downloaded from the release, as a build context.
FROM scratch AS wheels

# aic-base.  A `base` build context replaces this stage: the bake from-wheels
# targets pass target:base-from-wheels.
# hadolint ignore=DL3006
FROM ${AIC_BASE_IMAGE} AS base

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
FROM base AS vllm

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Passed by the bake from-wheels targets only; with none set the image is the
# one CI has always built.  ROCM_ARCH adds flash_attn when it names a gfx9
# arch, CUPY_VERSION pins cupy-rocm-7-0 as docker/vllm/Dockerfile does, and
# ROCM_VERSION adds what AITER's JIT needs.
ARG ROCM_ARCH
ARG CUPY_VERSION
ARG ROCM_VERSION

# An aic-base published before pip replaced uv has no pip3; add it so this
# harness does not depend on the tag having been republished.
# hadolint ignore=DL3008
RUN command -v pip3 > /dev/null || { \
        apt-get update && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-pip && \
        rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED && \
        rm -rf /var/lib/apt/lists/*; }

# Fail loudly and early on an incomplete wheel set.  Without this the install
# below "succeeds" against whatever pip can reach on PyPI -- which for `vllm` is
# a CUDA build that imports fine and dies at the first kernel launch.  That is
# exactly the failure docs/PIP_WHEELS.md warns about, so it must not be
# reachable from CI.
# SC2206: ${pat} is UNQUOTED on purpose -- it is a glob that must expand into
# the array.  Same deliberate exception .github/scripts/workflows/check-wheels.sh
# takes for the identical construct.
# hadolint ignore=SC2015,SC2206
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
#
# Two installs, in docker/vllm/Dockerfile's order (its `vllm` stage, then its
# `runtime` stage) and with its uninstalls after each.  One resolve of both
# wheels is impossible: vLLM v0.34.0 requires opentelemetry-api >= 1.45.1 and
# LMCache v0.5.8 caps it at 1.40.0.  Installed second, lmcache takes
# opentelemetry down to what it allows, which vLLM's tracing still imports.
#
# The uninstalls drop distributions this stack never imports, and the
# CUDA-flavoured ones the wheels' metadata drags in: aic-base supplies the ROCm
# NIXL at /opt/nixl, and a pip `nixl` would shadow it.
#
# aic-base's packaging is Debian-owned (no RECORD), so pip cannot upgrade it in
# place when a wheel needs a newer one; install a pip-owned copy first, as
# docker/vllm/Dockerfile does.
#
# flash_attn goes in when ROCM_ARCH names a gfx9 arch, llm_emu when the wheel
# set has it.  vLLM's ROCm platform needs a GPU to import, so the import checks
# at the end run on llm_emu's CPU platform, and are skipped without that wheel.
# hadolint ignore=SC2015
RUN --mount=type=bind,from=wheels,target=/w \
    --mount=type=cache,target=/root/.cache/pip \
    export PYTHONDONTWRITEBYTECODE=1 && \
    shopt -s nullglob && \
    _emu=(/w/llm_emu-*.whl) && \
    _extra=("${_emu[@]}") && \
    case ";${ROCM_ARCH};" in *";gfx9"*) \
        _fa=(/w/flash_attn-*.whl); \
        if [ "${#_fa[@]}" -ne 1 ]; then \
            echo "ERROR: ROCM_ARCH=${ROCM_ARCH} needs exactly one flash_attn wheel in the wheels context, found ${#_fa[@]}" >&2; \
            exit 1; \
        fi; \
        _extra+=("${_fa[@]}") ;; \
    esac && \
    pip3 install --no-compile --ignore-installed packaging && \
    _tv="$(python3 -c 'import torch; print(torch.__version__)')" && \
    _tvv="$(python3 -c 'import torchvision; print(torchvision.__version__)' 2>/dev/null || true)" && \
    echo "pinning to aic-base torch=${_tv} torchvision=${_tvv:-<absent>}" && \
    printf 'torch==%s\n' "${_tv}" > /tmp/c.txt && \
    if [ -n "${_tvv}" ]; then printf 'torchvision==%s\n' "${_tvv}" >> /tmp/c.txt; fi && \
    pip3 install --no-compile --constraint /tmp/c.txt \
        /w/vllm-*.whl /w/amd_aiter-*.whl "${_extra[@]}" && \
    pip3 uninstall -y \
        tilelang z3-solver ml-dtypes \
        mooncake-transfer-engine-rocm \
        runai-model-streamer runai-model-streamer-s3 \
        runai-model-streamer-gcs runai-model-streamer-azure \
        google-cloud-storage google-resumable-media google-cloud-core google-crc32c \
        azure-storage-blob azure-identity azure-core msal msal-extensions \
        datasets pyarrow multiprocess \
        supervisor fastapi-cloud-cli sentry-sdk rignore fastar && \
    python3 -c 'import importlib.metadata as m; print(*(f"{n}=={m.version(n)}" for n in ("numpy", "setuptools")), sep="\n")' >> /tmp/c.txt && \
    if [ -n "${CUPY_VERSION}" ]; then printf 'cupy-rocm-7-0==%s\n' "${CUPY_VERSION}" >> /tmp/c.txt; fi && \
    pip3 install --no-compile --constraint /tmp/c.txt /w/lmcache-*.whl && \
    rm -f /tmp/c.txt && \
    pip3 uninstall -y \
        nixl nixl-cu12 nixl-cu13 \
        cupy-cuda12x cupy-cuda13x cuda-python cuda-bindings cuda-core \
        cufile-python cuda-pathfinder \
        opentelemetry-exporter-otlp-common \
        opentelemetry-exporter-http-transport \
        awscrt nvtx \
        google-cloud-bigtable google-cloud-monitoring grpc-google-iam-v1 \
        google-cloud-core google-crc32c && \
    _after="$(python3 -c 'import torch; print(torch.__version__)')" && \
    if [ "${_after}" != "${_tv}" ]; then \
        echo "ERROR: torch changed from ${_tv} to ${_after} during the wheel install." >&2; \
        echo "       aic-base's ROCm torch was replaced; the image would fail at first kernel launch." >&2; \
        exit 1; \
    fi && \
    echo "torch intact: ${_after}" && \
    if [ "${#_emu[@]}" -gt 0 ]; then \
        VLLM_EMULATOR_ENABLE_ORACLE=1 python3 -c 'import vllm.entrypoints.openai.api_server, vllm.benchmarks.serve, vllm_emulator.hooks.executor_hook, lmcache.integration.vllm.vllm_multi_process_adapter; print("vLLM + lmcache MP adapter imports OK")' && \
        python3 -c 'import opentelemetry.exporter.otlp.proto.grpc.trace_exporter, opentelemetry.exporter.otlp.proto.http.trace_exporter, vllm.tracing.otel as o; assert o.is_otel_available(), "vLLM tracing cannot load opentelemetry"; print("vLLM tracing imports OK on opentelemetry", __import__("importlib.metadata").metadata.version("opentelemetry-api"))'; \
    else \
        echo "no llm_emu wheel in the set: vLLM import checks skipped"; \
    fi

# What docker/vllm/Dockerfile's runtime stage adds for AITER, which compiles
# some kernels on first use: make, the ROCm math headers and amdsmi.
# hadolint ignore=DL3008,DL3013
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip \
    if [ -n "${ROCM_VERSION}" ]; then \
        _mm=$(echo "${ROCM_VERSION}" | cut -d. -f1-2) && \
        apt-get update && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            make \
            "amdrocm-blas-dev${_mm}" \
            "amdrocm-ccl-dev${_mm}" \
            "amdrocm-hipblas-common-dev${_mm}" \
            "amdrocm-solver-dev${_mm}" \
            "amdrocm-sparse-dev${_mm}" && \
        pip3 install --no-compile amdsmi; \
    fi

# Post-install patches applied to the lmcache CLIENT, mirroring
# docker/vllm/Dockerfile's `emulate` stage.  vLLM is the producer on the store
# path, so its export_event must synchronize locally.
COPY docker/lmcache/rocm_event_ipc_patch.py /tmp/rocm_event_ipc_patch.py
RUN AIC_LMCACHE_SRC="$(python3 -c 'import lmcache, os; print(os.path.dirname(os.path.dirname(lmcache.__file__)))')" \
        python3 /tmp/rocm_event_ipc_patch.py && \
    rm /tmp/rocm_event_ipc_patch.py

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
# The base's name when a `base` build context replaced AIC_BASE_IMAGE.
ARG AIC_BASE_NAME
LABEL org.opencontainers.image.title="aic-vllm" \
      org.opencontainers.image.description="AMD Infinity Context: ROCm vLLM, reconstituted from published wheels (CI test harness)" \
      org.opencontainers.image.version="${AIC_VERSION}" \
      org.opencontainers.image.base.name="${AIC_BASE_NAME:-${AIC_BASE_IMAGE}}" \
      ai.amd.aic.version="${AIC_VERSION}" \
      ai.amd.aic.built-from="published-wheels"

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

# =============================================================================
# lmcache — aic-base + the lmcache wheel and its non-wheel runtime deps.
#
# Mirrors docker/lmcache/Dockerfile's `runtime` stage.  build_lmcache_wheel
# runs rocm_event_ipc_patch.py on the source before `pip wheel`, so the
# published wheel already carries the event-IPC edit and needs nothing beside
# it in site-packages.  The patcher is re-run against the installed package
# anyway: it reports "SKIP (already applied)" for a wheel built that way,
# applies the edit to one that was not, and fails the build if it can do
# neither.
# =============================================================================
FROM base AS lmcache

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set by the bake from-wheels targets: the cupy-rocm-7-0 aic-vllm is pinned to.
ARG CUPY_VERSION

# An aic-base published before pip replaced uv has no pip3; add it so this
# harness does not depend on the tag having been republished.
# hadolint ignore=DL3008
RUN command -v pip3 > /dev/null || { \
        apt-get update && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-pip && \
        rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED && \
        rm -rf /var/lib/apt/lists/*; }

RUN --mount=type=bind,from=wheels,target=/w \
    --mount=type=cache,target=/root/.cache/pip \
    shopt -s nullglob && \
    lmc=(/w/lmcache-*.whl) && \
    if [ "${#lmc[@]}" -ne 1 ]; then \
        echo "ERROR: expected exactly one lmcache wheel, found ${#lmc[@]}" >&2; \
        ls -l /w >&2; exit 1; \
    fi && \
    _torch_ver="$(python3 -c 'import torch; print(torch.__version__)')" && \
    printf 'torch==%s\n' "${_torch_ver}" > /tmp/c.txt && \
    if [ -n "${CUPY_VERSION}" ]; then printf 'cupy-rocm-7-0==%s\n' "${CUPY_VERSION}" >> /tmp/c.txt; fi && \
    pip3 install --ignore-installed packaging && \
    pip3 install --constraint /tmp/c.txt "${lmc[@]}" && \
    rm -f /tmp/c.txt

# The `lmcache` CLI imports every subcommand eagerly, so `lmcache server` dies
# without the bench subcommands' openai.  Upstream lists those deps only in
# requirements/cli.txt (the lmcache-cli package), not in this wheel's metadata.
# This is the same gap mk/bench.mk's [6b.1] hot-patch works around in the guest;
# installing it here is what lets that hot-patch become a no-op.
#
# cupy-rocm-7-0: LMCache's MP server imports cupy.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install \
        openai "matplotlib>=3.8,<4" "cupy-rocm-7-0${CUPY_VERSION:+==${CUPY_VERSION}}" && \
    pip3 uninstall -y \
        nixl nixl-cu12 nixl-cu13 \
        cupy-cuda12x cufile-python cuda-pathfinder

# Confirm (or apply) the ROCm event-IPC edit; see the stage comment above.
COPY docker/lmcache/rocm_event_ipc_patch.py /tmp/rocm_event_ipc_patch.py
RUN AIC_LMCACHE_SRC="$(python3 -c 'import lmcache, os; print(os.path.dirname(os.path.dirname(lmcache.__file__)))')" \
        python3 /tmp/rocm_event_ipc_patch.py && \
    rm /tmp/rocm_event_ipc_patch.py

# The checks docker/lmcache/Dockerfile ends its build on: the server CLI and the
# compiled extensions import.  DEVICE_TYPE=cpu keeps them off GPU detection.
RUN export PYTHONDONTWRITEBYTECODE=1 DEVICE_TYPE=cpu && \
    lmcache server --help > /dev/null && \
    python3 -c 'import lmcache.cuda_ops, lmcache.lmcache_fs, lmcache.lmcache_native, lmcache.lmcache_redis; import lmcache.v1.mp_coordinator.app'

# HSA_ENABLE_IPC_MODE_LEGACY is intentionally NOT set: aic-vllm leaves it unset
# too, and the two must agree.
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    SAFETENSORS_FAST_GPU=1 \
    HIP_FORCE_DEV_KERNARG=1

ARG AIC_VERSION
ARG AIC_BASE_IMAGE
# The base's name when a `base` build context replaced AIC_BASE_IMAGE.
ARG AIC_BASE_NAME
LABEL org.opencontainers.image.title="aic-lmcache" \
      org.opencontainers.image.description="AMD Infinity Context: ROCm LMCache, reconstituted from published wheels (CI test harness)" \
      org.opencontainers.image.version="${AIC_VERSION}" \
      org.opencontainers.image.base.name="${AIC_BASE_NAME:-${AIC_BASE_IMAGE}}" \
      ai.amd.aic.version="${AIC_VERSION}" \
      ai.amd.aic.built-from="published-wheels"

ENTRYPOINT ["python3", "-m", "lmcache.server"]
