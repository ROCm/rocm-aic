# Base image providing ROCm 7.2 and matching PyTorch
FROM rocm/pytorch:latest

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Setup working dir
RUN mkdir -p /app
WORKDIR /app

# Setup prerequisites
RUN apt-get update && \
	apt-get install -y --no-install-recommends \
	cmake bpftrace \
	      libboost-all-dev \
	      libmount-dev \
	      ninja-build \
	      vim && \
	rm -rf /var/lib/apt/lists/*

# Pinned SHAs for deterministic builds
ARG HIPFILE_BASE_SHA=5a7d275
ARG HIPFILE_PATCH_SHA=0d07c19
ARG LMCACHE_SHA=542bb9b
ARG VLLM_SHA=a913b612d
ARG FLASH_ATTN_SHA=0e60e394
ARG AITER_TAG=c3708fb7
ARG GPU_ARCHS

# Scan GPUs if no GPU_ARCHS specified for the build
RUN if [[ -z "${GPU_ARCHS}" ]]; then \
		rocm_agent_enumerator | sort -u | xargs echo | tr ' ' ';' > /app/gpu_archs.txt; \
	else \
		echo ${GPU_ARCHS} > /app/gpu_archs.txt; \
	fi

# Build & install hipFile libs, headers, and python bindings
RUN git clone https://github.com/ROCm/hipFile.git && \
	cd hipFile && \
	git config user.email "noone@example.com" && git config user.name "No User" && \
	git remote add dell https://github.com/glimchb/hipFile.git && git fetch dell && git fetch origin && \
	git checkout "${HIPFILE_BASE_SHA}" && git cherry-pick "${HIPFILE_PATCH_SHA}" && \
	mkdir build && cd build && \
	cmake .. && cmake --build . -j && cpack -G DEB && \
	dpkg -i hipfile*deb && \
	cd ../python && python -m build --wheel && \
	pip install --no-cache-dir dist/hipfile*whl && \
	cd ../..

# Build & install LMCache
RUN git clone https://github.com/glimchb/LMCache.git && \
	cd LMCache && git checkout "${LMCACHE_SHA}" && \
	pip install --no-cache-dir \
	    -r requirements/build.txt && \
	PYTORCH_ROCM_ARCH=$(cat /app/gpu_archs.txt) \
		TORCH_DONT_CHECK_COMPILER_ABI=1 \
		CXX=hipcc \
		BUILD_WITH_HIP=1 \
		python3 -m pip install --no-cache-dir \
		    --no-build-isolation -e . && \
	cd ..

# Build AITER (provides the kernels for vLLM's ROCM_AITER_FA backend)
# SHA pinned to match tag in vLLM's docker/Dockerfile.rocm_base
RUN git clone https://github.com/ROCm/aiter.git && \
	cd aiter && \
	git checkout "${AITER_TAG}" && \
	git submodule update --init --recursive && \
	pip install --no-cache-dir -r requirements.txt && \
	GPU_ARCHS=$(cat /app/gpu_archs.txt) python3 setup.py install && \
	cd ..

# Build flash-attention (CK backend, used by vLLM ROCM_AITER_FA)
# SHA pinned to match vLLM's docker/Dockerfile.rocm_base
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
	cd flash-attention && \
	git checkout "${FLASH_ATTN_SHA}" && \
	git submodule update --init && \
	GPU_ARCHS=$(cat /app/gpu_archs.txt) python3 setup.py install && \
	cd ..

# Build MORI
# TODO, if needed later

# Build vLLM
RUN git clone https://github.com/vllm-project/vllm.git && \
	cd vllm && \
	git checkout "${VLLM_SHA}" && \
	pip install --no-cache-dir /opt/rocm/share/amd_smi && \
	pip install --no-cache-dir --upgrade numba \
	    scipy \
	    'huggingface-hub[cli,hf_transfer]' \
	    setuptools_scm && \
	pip install --no-cache-dir -r requirements/rocm.txt && \
	PYTORCH_ROCM_ARCH=$(cat /app/gpu_archs.txt) python3 setup.py develop

# Transfer configs & run scripts, then fetch corpus
COPY scripts/ /app/scripts/
RUN /app/scripts/fetch-corpus.sh
