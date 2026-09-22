# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# AIC LMCache image: torch + torchvision (from aic-base) + AITER + FlashAttention +
# LMCache v0.5.5 (patched) + NIXL + hsa-snoop.
#
# Consumed by the `lmcache` and `lmcache-coordinator` compose services.  Built
# independently from aic-vllm (both share aic-base) so the two images can be
# built in parallel after the base image is available.
#
#   lmcache — standalone lmcache-server (MP mode)
#               L1:  GPU/CPU DRAM (--l1-size-gb)  OR  hipFile NVMe slab (GDS L1 mode)
#               L2a: nixl_store AIS_MT → local NVMe (hipFile P2PDMA)
#               L2b: nixl_store POSIX  → NFS-over-RDMA
#
# Stages:
#   build     AITER + FlashAttn + LMCache + NIXL + hsa-snoop + runtime ENV  (sections 4-8)
#   wheels    scratch image holding just /wheels                             (section 9)
#   runtime   default target, aliases `build`
#
# Build (from repo root — build context is this tree):
#   BASE_TAG=$(ROCM_ARCH=gfx950 docker/scripts/aic-image-tag.sh docker/base/Dockerfile)
#   DOCKER_BUILDKIT=1 docker buildx build \
#     --build-arg ROCM_ARCH=gfx950 \
#     --build-context base=docker-image://aic-base:${BASE_TAG} \
#     -f docker/lmcache/Dockerfile \
#     -t aic-lmcache:${BASE_TAG} \
#     .
#
# Or via make:
#   make dist-build-lmcache ROCM_ARCH=gfx950

# =============================================================================
# Full LMCache stack — AITER, FlashAttention, LMCache, NIXL, hsa-snoop.
# Inherits from the final stage of docker/base/ (build_torchvision), which
# already has pytorch + torchvision installed system-wide.
# =============================================================================
FROM base AS build

ARG ROCM_ARCH

# Parallel compile jobs for C++/HIP source builds (LMCache, hsa-snoop).
ARG BUILD_JOBS=

# ROCm attention backends — validated pair, update together after revalidation.
ARG AITER_GIT_URL=https://github.com/ROCm/aiter.git
ARG AITER_REF=v0.1.22.post1
ARG FLASH_ATTN_GIT_URL=https://github.com/Dao-AILab/flash-attention.git
ARG FLASH_ATTN_REF=v2.8.3.post1

# ai-dynamo/nixl release tag + nixl-rocm-ais-mt.patch (ROCm/HIP + AIS_MT).
ARG NIXL_GIT_URL=https://github.com/ai-dynamo/nixl.git
ARG NIXL_REF=v1.4.1

# hsa-snoop: HSA AQL queue/dispatch snooper + Prometheus exporter (includes AIS metrics).
ARG HSA_SNOOP_GIT_URL=https://github.com/sbates130272/hsa-snoop.git
ARG HSA_SNOOP_REF=v1.1.1

# LMCache: upstream repo, pinned to the v0.5.5 release tag.
# Optional open-PR heads from patches/lmcache.pull-requests are cherry-picked
# first; AMD patches in patches/lmcache/ apply on top of that tree.
ARG LMCACHE_GIT_URL=https://github.com/LMCache/LMCache.git
ARG LMCACHE_REF=v0.5.5

# Optional: enable UCX fast-path for NIXL (default off).
ARG AIC_UCX_FAST=

# ----- 4. ROCm attention backends (AITER + FlashAttention) ------------------
# hadolint ignore=SC2039
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
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
        CCACHE_DIR=/root/.cache/ccache \
        python3 setup.py bdist_wheel --dist-dir=/wheels && \
    pip3 install --no-cache-dir --no-deps /wheels/amd_aiter-*.whl && \
    python3 -c 'import importlib.util, pathlib; spec = importlib.util.find_spec("aiter"); assert spec and spec.submodule_search_locations; print("AITER installed:", next(iter(spec.submodule_search_locations)))'

# FlashAttention: CDNA-only (gfx9xx — MI100/MI200/MI300/MI355X).
# hadolint ignore=SC2039
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    _cdna_only() { \
        _a=""; for _a in $(echo "$1" | tr ';' '\n'); do \
            [ "${_a#gfx9}" != "${_a}" ] || return 1; done; return 0; }; \
    if _cdna_only "${ROCM_ARCH}"; then \
        git clone --recurse-submodules "${FLASH_ATTN_GIT_URL}" /app/flash-attention && \
        cd /app/flash-attention && \
        git checkout "${FLASH_ATTN_REF}" && \
        git submodule update --init --recursive && \
        GPU_ARCHS="${ROCM_ARCH}" \
            PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
            MAX_JOBS="${BUILD_JOBS:-$(nproc)}" \
            CCACHE_DIR=/root/.cache/ccache \
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
            python3 -c 'import importlib.util; spec = importlib.util.find_spec("aiter"); assert spec and spec.submodule_search_locations; print("AITER_PACKAGE=" + next(iter(spec.submodule_search_locations)))'; \
        } > /etc/rocm-aic-attention-backends.env; \
    fi

# ----- 5. LMCache (upstream release + AMD patches) ---------------------------
COPY patches/ /app/patches/

RUN git clone --depth 1 --branch "${LMCACHE_REF}" "${LMCACHE_GIT_URL}" /app/LMCache

RUN git -C /app/LMCache config user.name "AIC Image Build" && \
    git -C /app/LMCache config user.email "rocm-aic@noreply.invalid"

# Cherry-pick optional LMCache PR heads listed in patches/lmcache.pull-requests.
# Line format:
#   <pr-number>
#   <remote-name> <pr-number>
#   <remote-name>=<remote-url> <pr-number>
# Blank lines and # comments are ignored. Alternate remotes must expose
# GitHub-style refs/pull/<pr-number>/head refs.
# hadolint ignore=SC2206,SC3054,SC3044
RUN cd /app/LMCache && \
    if [ -f /app/patches/lmcache.pull-requests ]; then \
        while IFS= read -r raw_line || [ -n "$raw_line" ]; do \
            line="$(sed -e 's/#.*//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' <<<"$raw_line")" && \
            if [ -z "$line" ]; then \
                continue; \
            fi && \
            parts=($line) && \
            if [ "${#parts[@]}" -eq 1 ]; then \
                remote_spec=origin && \
                pr_number="${parts[0]}"; \
            elif [ "${#parts[@]}" -eq 2 ]; then \
                remote_spec="${parts[0]}" && \
                pr_number="${parts[1]}"; \
            else \
                echo "ERROR: invalid LMCache PR entry: $raw_line" >&2; \
                echo "Expected '<pr-number>', '<remote-name> <pr-number>', or '<remote-name>=<remote-url> <pr-number>'." >&2; \
                exit 1; \
            fi && \
            if [[ ! "$pr_number" =~ ^[0-9]+$ ]]; then \
                echo "ERROR: invalid LMCache PR number in entry: $raw_line" >&2; exit 1; \
            fi && \
            remote_name="${remote_spec%%=*}" && \
            remote_url="" && \
            if [ "${remote_spec#*=}" != "$remote_spec" ]; then \
                remote_url="${remote_spec#*=}"; \
            fi && \
            if ! git remote get-url "$remote_name" >/dev/null 2>&1; then \
                if [ -z "$remote_url" ]; then \
                    echo "ERROR: LMCache remote '$remote_name' is not configured; use '<remote-name>=<remote-url> <pr-number>'." >&2; \
                    exit 1; \
                fi && \
                git remote add "$remote_name" "$remote_url"; \
            fi && \
            pr_ref="refs/remotes/${remote_name}/pr/${pr_number}" && \
            if ! git fetch "$remote_name" "refs/pull/${pr_number}/head:${pr_ref}"; then \
                echo "ERROR: LMCache PR ${remote_name}#${pr_number} could not be fetched via refs/pull/${pr_number}/head; use a remote that exposes GitHub-style PR refs." >&2; \
                exit 1; \
            fi && \
            merge_base="$(git merge-base HEAD "${pr_ref}")" && \
            if [ "$merge_base" != "$(git rev-parse HEAD)" ]; then \
                echo "ERROR: LMCache PR ${remote_name}#${pr_number} is not based on ${LMCACHE_REF}; rebase it or convert it to a patch." >&2; \
                exit 1; \
            fi && \
            commit_count="$(git rev-list --count "HEAD..${pr_ref}")" && \
            if [ "$commit_count" -eq 0 ]; then \
                echo "Skipped LMCache PR ${remote_name}#${pr_number}: already included"; \
                continue; \
            fi && \
            if git cherry-pick -x "HEAD..${pr_ref}"; then \
                echo "Applied LMCache PR ${remote_name}#${pr_number}"; \
            else \
                git cherry-pick --abort >/dev/null 2>&1 || true; \
                echo "ERROR: LMCache PR ${remote_name}#${pr_number} does not cherry-pick cleanly" >&2; \
                exit 1; \
            fi; \
        done < /app/patches/lmcache.pull-requests; \
    fi

# Apply every patch under patches/lmcache/ in lexical order after optional PRs.
# hadolint ignore=SC3054,SC3044
RUN cd /app/LMCache && \
    shopt -s nullglob && \
    patches=(/app/patches/lmcache/*.patch) && \
    if [ "${#patches[@]}" -eq 0 ]; then \
        echo "ERROR: no patches found in /app/patches/lmcache/" >&2; exit 1; \
    fi && \
    for patch in "${patches[@]}"; do \
        if git apply "$patch"; then \
            echo "Applied: $patch"; \
        else \
            echo "ERROR: LMCache patch does not apply cleanly: $patch" >&2; \
            echo "The LMCache pin (${LMCACHE_REF}) likely drifted; rebase this patch against it." >&2; \
            exit 1; \
        fi; \
    done

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir grpcio-tools

RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p /wheels && \
    printf '%s\n' "${ROCM_ARCH%%;*}" > /etc/rocm_target.lst && \
    cd /app/LMCache && \
    PYTORCH_ROCM_ARCH="${ROCM_ARCH}" \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        CXX=hipcc \
        BUILD_WITH_HIP=1 \
        ROCM_TARGET_LST=/etc/rocm_target.lst \
        MAX_JOBS="${BUILD_JOBS:-$(nproc)}" \
        CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS:-$(nproc)}" \
        python3 -m pip wheel --no-build-isolation --no-deps -w /wheels . && \
    uv pip install --system --no-cache /wheels/lmcache-*.whl

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "matplotlib>=3.8,<4"

# Remove any NIXL wheels that upstream packages may have pulled — we build from source below.
# Also purge stray CUDA-only deps (cupy-cuda12x, cufile-python, cuda-pathfinder).
RUN uv pip uninstall --system \
        nixl nixl-cu12 nixl-cu13 \
        cupy-cuda12x cufile-python cuda-pathfinder 2>/dev/null || true

# ----- 6. hipFile — provided by the ROCm 7.14 base image (GA since 7.14) ------
# No source build needed.

# ----- 7. NIXL (ai-dynamo/nixl upstream + nixl-rocm-ais-mt.patch) ------------
COPY docker/scripts /tmp/nixl-scripts
COPY patches/nixl/ /tmp/nixl-patches/
# hadolint ignore=SC3054,SC3044
RUN --mount=type=cache,target=/root/.cache/uv \
    chmod +x /tmp/nixl-scripts/*.sh && \
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
    NIXL_SRC=/tmp/nixl \
        AIS_PATH=/opt/rocm ROCM_PATH=/opt/rocm \
        NIXL_INSTALL_PREFIX=/opt/nixl \
        NIXL_BUILD_WHEEL=1 NIXL_WHEEL_DIR=/wheels \
        BUILD_JOBS="${BUILD_JOBS}" \
        AIC_UCX_FAST="${AIC_UCX_FAST}" \
        UV_SYSTEM_CERTS=1 \
        /tmp/nixl-scripts/build-nixl.sh

# ----- 7b. hsa-snoop (HSA AQL queue snooper + Prometheus exporter) -----------
RUN git clone --depth 1 --branch "${HSA_SNOOP_REF}" "${HSA_SNOOP_GIT_URL}" /app/hsa-snoop && \
    cmake -S /app/hsa-snoop -B /app/hsa-snoop/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DHSA_SNOOP_PROMETHEUS=ON \
        -DCMAKE_DISABLE_FIND_PACKAGE_hip=ON && \
    cmake --build /app/hsa-snoop/build --parallel "${BUILD_JOBS:-$(nproc)}" && \
    cmake --install /app/hsa-snoop/build --prefix /usr/local && \
    test -x /usr/local/bin/hsa-snoop && \
    # hadolint ignore=SC2015
    { /usr/local/bin/hsa-snoop --help 2>&1 || true; } | grep -q -- '--prometheus' \
        || { echo "ERROR: hsa-snoop built without Prometheus mode (--prometheus absent)" >&2; exit 1; } && \
    echo "PASS: hsa-snoop ${HSA_SNOOP_REF} with Prometheus exporter (incl. AIS metrics) installed"

# ----- 8. Runtime environment ------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    NIXL_PLUGIN_DIR=/opt/nixl/lib/x86_64-linux-gnu/plugins \
    PYTHONPATH=/opt/rocm/share/amd_smi:/opt/nixl/lib/python3/dist-packages \
    LD_LIBRARY_PATH=/opt/nixl/lib/x86_64-linux-gnu:/opt/nixl/lib:/opt/rocnixl-ucx/lib:/opt/rocm/lib \
    HSA_ENABLE_IPC_MODE_LEGACY=1 \
    SAFETENSORS_FAST_GPU=1 \
    HIP_FORCE_DEV_KERNARG=1

ARG AIC_VERSION
ARG ROCM_BASE_IMAGE
ARG ROCM_VERSION
RUN test -n "$AIC_VERSION" || { echo "ERROR: AIC_VERSION is not set (read it from VERSION)"; exit 1; }
LABEL org.opencontainers.image.title="aic-lmcache" \
      org.opencontainers.image.description="AMD Infinity Context: ROCm LMCache tiered-cache service" \
      org.opencontainers.image.version="${AIC_VERSION}" \
      org.opencontainers.image.base.name="${ROCM_BASE_IMAGE}" \
      ai.amd.aic.version="${AIC_VERSION}" \
      ai.amd.aic.rocm="${ROCM_VERSION}" \
      ai.amd.aic.aiter="${AITER_REF}" \
      ai.amd.aic.flash-attention="${FLASH_ATTN_REF}" \
      ai.amd.aic.lmcache="${LMCACHE_REF}" \
      ai.amd.aic.nixl="${NIXL_REF}" \
      ai.amd.aic.hipfile="rocm-${ROCM_VERSION}" \
      ai.amd.aic.hsa-snoop="${HSA_SNOOP_REF}"

ENTRYPOINT ["python3", "-m", "lmcache.server"]

# ----- 9. Wheel export stage -------------------------------------------------
FROM scratch AS wheels
COPY --from=build /wheels /

# ----- 10. Runtime image (default target) ------------------------------------
FROM build AS runtime
