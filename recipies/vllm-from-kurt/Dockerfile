FROM vllm/vllm-openai-rocm:v0.19.0

ARG ROCM_ARCH
RUN test -n "$ROCM_ARCH" || (echo "ERROR: ROCM_ARCH is not set" && exit 1)

## LMCache is broken (2026-04-21), need to checkout a specific SHA1
#RUN cd /app && \
#    git clone https://github.com/LMCache/LMCache.git && \
#    cd LMCache && \
#    git checkout 33135270223b094bbe567b0c20a318d8b24f0366 && \
#    PYTORCH_ROCM_ARCH=${ROCM_ARCH} \
#        TORCH_DONT_CHECK_COMPILER_ABI=1 \
#        CXX=hipcc \
#        BUILD_WITH_HIP=1 \
#        python3 -m pip install --no-build-isolation -e .


## Riley's lmcache changes (Just before "^^^")
## Riley's changes:
ARG LMCACHE_SHA1=6f24badfa940fadbc5cf66d8266a221211ff1adc
## Riley's changes - 1:
#ARG LMCACHE_SHA1=e98dc3c942ab5216fc2824a57dbc30f39f52a5ee
RUN cd /app && \
    git clone https://github.com/riley-dixon/LMCache.git && \
    cd LMCache && \
    git checkout ${LMCACHE_SHA1} && \
    PYTORCH_ROCM_ARCH=${ROCM_ARCH} \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        CXX=hipcc \
        BUILD_WITH_HIP=1 \
        python3 -m pip install --no-build-isolation --no-cache-dir -e .

RUN apt-get update && \
    apt-get install -y \
        libboost-program-options-dev \
        libmount-dev

RUN cd /app && \
    git clone https://github.com/ROCm/hipFile.git && \
    cd hipFile && \
    mkdir build && \
    cd build && \
    cmake -DGPU_TARGETS=${ROCM_ARCH} -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_PLATFORM=amd .. && \
    cmake --build . -j && \
    cmake --install .

RUN cd /app && \
    git clone https://github.com/ROCm/fio.git && \
    cd fio && \
    git checkout origin/hipFile && \
    CFLAGS="-I/opt/rocm/include -I/app/hipFile/include" \
        LDFLAGS="-L/opt/rocm/lib -L/app/hipFile/build/src/amd_detail -Wl,-rpath,/app/hipFile/build/src/amd_detail" \
        ./configure --enable-libhipfile && \
    make -j

# Riley's hipfile bindings
#RUN pip install scikit-build-core
#RUN pip install hipfile --pre --no-build-isolation

# Riley's hipfile bindings 2.0
COPY hipfile-0.2.0.dev0-cp312-cp312-linux_x86_64.whl /tmp
RUN python3 -m pip install --no-cache-dir scikit-build-core
RUN python3 -m pip install --no-cache-dir /tmp/hipfile-0.2.0.dev0-cp312-cp312-linux_x86_64.whl

# Alex's hipfile bindings
# COPY hipfile /usr/lib/python3/dist-packages/hipfile

COPY scripts/* /app/

# This is the way
RUN echo 'set -o vi' >> /root/.bashrc

ENTRYPOINT ["/bin/bash"]
CMD []
