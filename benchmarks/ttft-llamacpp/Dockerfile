# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

# ROCm 7.2.1 development image with HIP toolchain
FROM rocm/dev-ubuntu-24.04:7.2.1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN mkdir -p /app
WORKDIR /app

# Build essentials for llama.cpp
RUN apt-get update && \
	apt-get install -y --no-install-recommends \
		cmake curl git ninja-build python3 python3-pip python3-venv && \
	rm -rf /var/lib/apt/lists/*

# Pinned llama.cpp release for deterministic builds
ARG LLAMACPP_TAG=b8799
ARG APPLY_CACHE_DISK_PATCH=0
ARG GPU_ARCHS

# Clone llama.cpp at the pinned tag
RUN git clone --depth 1 --branch "${LLAMACPP_TAG}" \
	https://github.com/ggml-org/llama.cpp.git /app/llama.cpp

# Optionally apply the cache-disk patch
COPY patches/ /app/patches/
RUN if [[ "${APPLY_CACHE_DISK_PATCH}" == "1" ]] && \
	   [[ -f /app/patches/0001-cache-disk.patch ]]; then \
		cd /app/llama.cpp && \
		git apply /app/patches/0001-cache-disk.patch && \
		echo "Applied cache-disk patch"; \
	else \
		echo "Skipping cache-disk patch"; \
	fi

# Build llama.cpp with HIP/ROCm support
RUN cd /app/llama.cpp && \
	cmake -B build -G Ninja \
		-DGGML_HIP=ON \
		-DCMAKE_BUILD_TYPE=Release \
		${GPU_ARCHS:+-DAMDGPU_TARGETS="${GPU_ARCHS}"} && \
	cmake --build build --target llama-server -j "$(nproc)" && \
	cp build/bin/llama-server /usr/local/bin/llama-server

# Python venv for benchmark dependencies
RUN python3 -m venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy benchmark scripts and fetch corpus
COPY scripts/ /app/scripts/
COPY bench_ttft.py /app/bench_ttft.py
RUN /app/scripts/fetch-corpus.sh
