FROM python:3.12-slim

ARG KVBENCH_GIT_URL=https://github.com/wvaske/llm-kv-passthrough.git
ARG KVBENCH_REF=177cf4e7c46ece8e03a701f89b60cd3a329f45fd
# kvbench's [lmcache] extra is `~=0.5.1`; 0.5.7 removed lmcache.v1.cache_engine.
ARG KVBENCH_LMCACHE_VERSION=0.5.6

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 "${KVBENCH_GIT_URL}" /tmp/kvbench \
    && git -C /tmp/kvbench fetch --depth 1 origin "${KVBENCH_REF}" \
    && git -C /tmp/kvbench checkout "${KVBENCH_REF}" \
    && python -m pip install '/tmp/kvbench[lmcache]' \
        "lmcache==${KVBENCH_LMCACHE_VERSION}" \
    && rm -rf /tmp/kvbench

EXPOSE 8000
