# syntax=docker/dockerfile:1
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# ROCm vLLM dev base + hipFile + LMCache + NIXL (ROCm/nixl over HTTPS by
# default; optional NIXL_REF for tag/commit) + llm-d checkout under /opt/llm-d.
# Build with BuildKit, e.g.:
#   export DOCKER_BUILDKIT=1
#   docker buildx build -f Dockerfile -t rocm-inference-stack:latest .
#
FROM rocm/vllm-dev:nightly_main_20260416

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN mkdir -p /app
WORKDIR /app

RUN apt-get update && \
	apt-get install -y --no-install-recommends \
		ca-certificates \
		git \
		cmake \
		libboost-all-dev \
		libmount-dev \
		ninja-build \
		pkg-config && \
	rm -rf /var/lib/apt/lists/*

ARG HIPFILE_SHA=b2509f2
ARG LMCACHE_GIT_URL=https://github.com/amd-ivaganev/LMCache.git
ARG LMCACHE_SHA=bef2e13
ARG GPU_ARCHS=gfx942
ARG WITH_HIPFILE=1

ARG NIXL_GIT_URL=https://github.com/ROCm/nixl.git
ARG NIXL_REF=
ARG LLM_D_GIT_URL=https://github.com/llm-d/llm-d.git
ARG LLM_D_REF=main
ARG LLM_D_SHA=

RUN echo "${GPU_ARCHS}" | tr ' ' ';' > /app/gpu_archs.txt

RUN if [[ "${WITH_HIPFILE}" == "1" ]]; then \
		git clone --no-checkout --filter=blob:none \
			https://github.com/ROCm/rocm-systems.git && \
		cd rocm-systems && \
		git sparse-checkout set --cone projects/hipfile && \
		git checkout "${HIPFILE_SHA}" && \
		cd projects/hipfile && \
		mkdir build && cd build && \
		cmake -DCMAKE_HIP_PLATFORM=amd .. && \
		cmake --build . -j"$(nproc)" && \
		cpack -G DEB && \
		dpkg -i hipfile*deb && \
		cd ../python && \
		pip install --no-cache-dir -e . \
			-Ccmake.define.HIPFILE_INCLUDE_DIR=../include \
			-Ccmake.define.HIP_INCLUDE_DIR=/opt/rocm/include; \
	fi

RUN git clone "${LMCACHE_GIT_URL}" /tmp/LMCache && \
	cd /tmp/LMCache && git checkout "${LMCACHE_SHA}" && \
	pip install --no-cache-dir -r requirements/build.txt && \
	PYTORCH_ROCM_ARCH=$(cat /app/gpu_archs.txt) \
		TORCH_DONT_CHECK_COMPILER_ABI=1 \
		CXX=hipcc \
		BUILD_WITH_HIP=1 \
		python3 -m pip install --no-cache-dir --no-build-isolation -e .

# Optional transitive nixl wheels from LMCache deps; ignore if absent.
RUN pip uninstall -y nixl nixl-cu12 2>/dev/null || true

COPY scripts/clone-nixl.sh /tmp/clone-nixl.sh
COPY scripts/build-nixl.sh /tmp/build-nixl.sh
RUN chmod +x /tmp/clone-nixl.sh /tmp/build-nixl.sh

RUN NIXL_GIT_URL="${NIXL_GIT_URL}" NIXL_REF="${NIXL_REF}" /tmp/clone-nixl.sh

ENV NIXL_SRC=/tmp/nixl
ENV UCX_PREFIX=/opt/rocnixl-ucx
RUN /tmp/build-nixl.sh

RUN rm -rf /opt/llm-d/src && mkdir -p /opt/llm-d && \
	if git clone --depth 1 --branch "${LLM_D_REF}" "${LLM_D_GIT_URL}" /opt/llm-d/src; then \
		:; \
	else \
		git clone "${LLM_D_GIT_URL}" /opt/llm-d/src && \
		cd /opt/llm-d/src && git checkout "${LLM_D_REF}"; \
	fi

RUN if [[ -n "${LLM_D_SHA:-}" ]]; then \
		cd /opt/llm-d/src && \
		git fetch --depth 1 origin "${LLM_D_SHA}" && \
		git checkout FETCH_HEAD; \
	fi

RUN apt-get update && \
	apt-get install -y --no-install-recommends curl ca-certificates && \
	curl -fsSL "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" -o /usr/local/bin/kubectl && \
	chmod +x /usr/local/bin/kubectl && \
	curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash && \
	rm -rf /var/lib/apt/lists/*

ENV NIXL_SRC=
WORKDIR /app
