FROM python:3.12-slim

ARG KVBENCH_GIT_URL=https://github.com/wvaske/llm-kv-passthrough.git
ARG KVBENCH_REF=177cf4e7c46ece8e03a701f89b60cd3a329f45fd

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 "${KVBENCH_GIT_URL}" /tmp/kvbench \
    && git -C /tmp/kvbench fetch --depth 1 origin "${KVBENCH_REF}" \
    && git -C /tmp/kvbench checkout "${KVBENCH_REF}" \
    && python -m pip install '/tmp/kvbench[lmcache]' \
    && rm -rf /tmp/kvbench

EXPOSE 8000
