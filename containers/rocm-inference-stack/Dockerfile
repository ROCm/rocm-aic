# syntax=docker/dockerfile:1
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# ROCm vLLM dev base + hipFile + LMCache + NIXL (private ROCm/nixl via SSH) +
# optional llm-d chart checkout. Build with BuildKit and SSH agent, e.g.:
#   export DOCKER_BUILDKIT=1
#   eval "$(ssh-agent -s)" && ssh-add ~/.ssh/your_github_key
#   docker buildx build --ssh=default -f Dockerfile -t rocm-inference-stack:latest .
#
FROM rocm/vllm-dev:nightly_main_20260416

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN mkdir -p /app
WORKDIR /app

RUN apt-get update && \
	apt-get install -y --no-install-recommends \
		ca-certificates \
		openssh-client \
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

ARG NIXL_GIT_URL=git@github.com:ROCm/nixl.git
ARG NIXL_REF=
ARG LLM_D_GIT_URL=https://github.com/llm-d/llm-d.git
ARG LLM_D_REF=main

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
		python3 -m pip install --no-cache-dir --no-build-isolation -e . && \
	pip uninstall -y nixl nixl-cu12 2>/dev/null || true

COPY scripts/build-nixl.sh /tmp/build-nixl.sh
RUN chmod +x /tmp/build-nixl.sh

RUN --mount=type=ssh \
	bash -c 'set -euo pipefail; \
		mkdir -p /root/.ssh && chmod 700 /root/.ssh; \
		ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null || true; \
		rm -rf /tmp/nixl; \
		if [[ -n "${NIXL_REF}" ]]; then \
			if git clone --depth 1 --branch "${NIXL_REF}" "${NIXL_GIT_URL}" /tmp/nixl; then \
				:; \
			else \
				rm -rf /tmp/nixl; \
				git clone "${NIXL_GIT_URL}" /tmp/nixl; \
				cd /tmp/nixl && git checkout "${NIXL_REF}"; \
			fi; \
		else \
			git clone --depth 1 "${NIXL_GIT_URL}" /tmp/nixl; \
		fi'

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

RUN apt-get update && \
	apt-get install -y --no-install-recommends curl ca-certificates && \
	curl -fsSL "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" -o /usr/local/bin/kubectl && \
	chmod +x /usr/local/bin/kubectl && \
	curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash && \
	rm -rf /var/lib/apt/lists/*

ENV NIXL_SRC=
WORKDIR /app
