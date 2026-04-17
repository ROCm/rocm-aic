# Base image providing ROCm 7.2 and matching PyTorch
FROM rocm/vllm-dev:nightly_main_20260415

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Setup working dir
RUN mkdir -p /app
WORKDIR /app

# Setup debug symbols repository
RUN apt-get update && apt-get install -y --no-install-recommends ubuntu-dbgsym-keyring && \
	(echo "deb http://ddebs.ubuntu.com $(lsb_release -cs) main restricted universe multiverse" && \
	 echo "deb http://ddebs.ubuntu.com $(lsb_release -cs)-updates main restricted universe multiverse" && \
	 echo "deb http://ddebs.ubuntu.com $(lsb_release -cs)-proposed main restricted universe multiverse") | \
	tee /etc/apt/sources.list.d/ddebs.list && \
	rm -rf /var/lib/apt/lists/*

# Setup prerequisites
RUN apt-get update && \
	apt-get install -y --no-install-recommends \
	cmake bpftrace-dbgsym \
	      gdb \
	      libboost-all-dev \
	      libmount-dev \
	      ninja-build \
	      vim && \
	rm -rf /var/lib/apt/lists/*

# Pinned SHAs for deterministic builds
ARG HIPFILE_SHA=b2509f2
ARG LMCACHE_SHA=bef2e13
ARG GPU_ARCHS

# Scan GPUs if no GPU_ARCHS specified for the build
RUN if [[ -z "${GPU_ARCHS}" ]]; then \
		rocm_agent_enumerator | sort -u | xargs echo | tr ' ' ';' > /app/gpu_archs.txt; \
	else \
		echo ${GPU_ARCHS} > /app/gpu_archs.txt; \
	fi

# Build & install hipFile libs, headers, and python bindings
RUN git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git && \
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
	pip install --no-cache-dir -e . -Ccmake.define.HIPFILE_INCLUDE_DIR=../include -Ccmake.define.HIP_INCLUDE_DIR=/opt/rocm/include

# Build & install LMCache
# NOTE: Remove NIXL wheels pulled by upstream LMCache unconditionally. If they remain, VLLM
# tries to import non-existent CUDA shared object at runtime.
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

# Transfer configs & run scripts, then fetch corpus
COPY scripts/ /app/scripts/
RUN /app/scripts/fetch-corpus.sh
