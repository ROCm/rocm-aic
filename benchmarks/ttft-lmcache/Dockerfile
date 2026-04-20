# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

# Base image providing ROCm 7.2.1 and matching PyTorch
FROM rocm/vllm-dev:nightly_main_20260416

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN mkdir -p /app
WORKDIR /app

# Pinned SHAs for deterministic builds
ARG HIPFILE_SHA=b2509f2
ARG LMCACHE_SHA=bef2e13
ARG GPU_ARCHS
ARG WITH_HIPFILE=1

# Scan GPUs if no GPU_ARCHS specified for the build
RUN if [[ -z "${GPU_ARCHS}" ]]; then \
		rocm_agent_enumerator | sort -u | xargs echo | tr ' ' ';' > /app/gpu_archs.txt; \
	else \
		echo ${GPU_ARCHS} > /app/gpu_archs.txt; \
	fi

# Optionally build & install hipFile (needed for AIS backend)
RUN if [[ "${WITH_HIPFILE}" == "1" ]]; then \
		apt-get update && \
		apt-get install -y --no-install-recommends \
			cmake libboost-all-dev libmount-dev ninja-build && \
		rm -rf /var/lib/apt/lists/* && \
		git clone --no-checkout --filter=blob:none \
			https://github.com/ROCm/rocm-systems.git && \
		cd rocm-systems && \
		git sparse-checkout set --cone projects/hipfile && \
		git checkout "${HIPFILE_SHA}" && \
		cd projects/hipfile && \
		mkdir build && cd build && \
		cmake -DCMAKE_HIP_PLATFORM=amd .. && \
		cmake --build . -j && \
		cpack -G DEB && \
		dpkg -i hipfile*deb && \
		cd ../python && \
		pip install --no-cache-dir -e . \
			-Ccmake.define.HIPFILE_INCLUDE_DIR=../include \
			-Ccmake.define.HIP_INCLUDE_DIR=/opt/rocm/include; \
	fi

# Build & install LMCache with ROCm/HIP support.
# Remove NIXL wheels pulled by upstream LMCache unconditionally;
# if they remain, vLLM tries to import non-existent CUDA .so at runtime.
RUN git clone https://github.com/amd-ivaganev/LMCache.git && \
	cd LMCache && git checkout "${LMCACHE_SHA}" && \
	pip install --no-cache-dir \
	    -r requirements/build.txt && \
	PYTORCH_ROCM_ARCH=$(cat /app/gpu_archs.txt) \
		TORCH_DONT_CHECK_COMPILER_ABI=1 \
		CXX=hipcc \
		BUILD_WITH_HIP=1 \
		python3 -m pip install --no-cache-dir \
		    --no-build-isolation -e . && \
	pip uninstall -y nixl nixl-cu12

# Install benchmark Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Transfer configs, scripts, and benchmark code
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY bench_ttft.py /app/bench_ttft.py

# Fetch the benchmark corpus at build time
RUN /app/scripts/fetch-corpus.sh
