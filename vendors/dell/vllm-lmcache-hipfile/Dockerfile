# Base image providing ROCm 7.2 and matching PyTorch
FROM rocm/pytorch:latest

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Setup working dir
RUN mkdir -p /app
WORKDIR /app

# Setup prerequisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake libmount-dev libboost-all-dev vim && \
    rm -rf /var/lib/apt/lists/*

# Build & install hipFile libs, headers, and python bindings
RUN git clone https://github.com/glimchb/hipFile.git && \
    	cd hipFile && git checkout origin/python && \
	mkdir build && cd build && \
	cmake .. && cmake --build . -j && cpack -G DEB && \
	dpkg -i hipfile*deb && \
	cd ../python && python -m build --wheel && \
	pip install --no-cache-dir dist/hipfile*whl && \
	cd ../..

# Build & install LMCache
RUN git clone https://github.com/glimchb/LMCache.git && \
    	cd LMCache && git checkout origin/hipFile && \
	pip install --no-cache-dir \
	    -r requirements/build.txt && \
	PYTORCH_ROCM_ARCH=$(rocm_agent_enumerator | head -n 1) \
		TORCH_DONT_CHECK_COMPILER_ABI=1 \
		CXX=hipcc \
		BUILD_WITH_HIP=1 \
		python3 -m pip install --no-cache-dir \
		    --no-build-isolation -e . && \
	cd ..

# Build CK
#RUN git clone https://github.com/ROCm/composable_kernel.git && \
#    	cd composable_kernel && mkdir build && cd build && \
#	cmake -D CMAKE_PREFIX_PATH=/opt/rocm \
#	      -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
#	      -D CMAKE_BUILD_TYPE=Release \
#	      -D GPU_TARGETS=$(rocm_agent_enumerator | head -n 1) .. && \
#	      make -j $(nproc --ignore 2) && make install && \
#	      cd ../..

# Build flash-attention
# TODO, if needed later

# Build AITER
# TODO, if needed later

# Build MORI
# TODO, if needed later

# Build vLLM
RUN git clone https://github.com/vllm-project/vllm.git && \
    	cd vllm && \
	pip install --no-cache-dir /opt/rocm/share/amd_smi && \
	pip install --no-cache-dir --upgrade numba \
	    scipy \
	    'huggingface-hub[cli,hf_transfer]' \
	    setuptools_scm && \
	pip install --no-cache-dir -r requirements/rocm.txt && \
	PYTORCH_ROCM_ARCH="gfx942" python3 setup.py develop

# Transfer configs & run scripts, then fetch corpus
COPY scripts/ /app/scripts/
RUN /app/scripts/fetch-corpus.sh
