# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Containerized RDMA Prometheus exporter for the aai-day-release metrics stack.
#
# Reproduces the batesste `rdma_setup` role's optional rdma_exporter host service
# in a container so nodes WITHOUT the host systemd service (or without the
# compose plugin -- see the docker-run fallback in .slurm/run-cliff.sbatch) still
# export dedicated RDMA/RoCE metrics on :9879.  Same upstream, same release
# binary, same default flags, so Prometheus/Grafana see identical
# `rdma_port_{rcv,xmit}_data_total` series either way.
#
#   Upstream: https://github.com/yuuki/rdma_exporter
#   Reads RDMA NIC stats from sysfs only -- no extra runtime tools needed.
#
# Multi-arch: TARGETARCH is set automatically by `docker buildx` (amd64/arm64);
# it maps directly onto the release asset suffix.
#
# Build (from aai-day-release/monitoring/):
#   docker build -t aai-day-rdma-exporter:local ./rdma-exporter
#   # or pin a version:
#   docker build --build-arg RDMA_EXPORTER_VERSION=0.4.2 \
#       -t aai-day-rdma-exporter:0.4.2 ./rdma-exporter
#
# Run (host networking; needs sysfs):
#   docker run -d --name aai-day-rdma-exporter --network host \
#       -v /sys:/sys:ro aai-day-rdma-exporter:local

FROM debian:12-slim AS fetch

# Matches the rdma_setup role's pinned default (0.3.0); latest upstream is 0.4.2.
ARG RDMA_EXPORTER_VERSION=0.3.0
# Provided by buildx; default keeps a plain `docker build` working on x86_64.
ARG TARGETARCH=amd64

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl; \
    url="https://github.com/yuuki/rdma_exporter/releases/download/v${RDMA_EXPORTER_VERSION}/rdma_exporter_${RDMA_EXPORTER_VERSION}_linux_${TARGETARCH}.tar.gz"; \
    mkdir -p /tmp/extract; \
    curl -fSL "$url" -o /tmp/rdma_exporter.tar.gz; \
    tar -xzf /tmp/rdma_exporter.tar.gz -C /tmp/extract; \
    install -m0755 "$(find /tmp/extract -type f -name rdma_exporter | head -1)" \
        /usr/local/bin/rdma_exporter; \
    /usr/local/bin/rdma_exporter --help >/dev/null 2>&1 || true

FROM debian:12-slim

LABEL org.opencontainers.image.title="aai-day-rdma-exporter" \
      org.opencontainers.image.description="Prometheus RDMA exporter (yuuki/rdma_exporter) for the aai-day-release metrics stack" \
      org.opencontainers.image.source="https://github.com/yuuki/rdma_exporter" \
      org.opencontainers.image.licenses="MIT"

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    mkdir -p /var/lib/rdma-exporter

COPY --from=fetch /usr/local/bin/rdma_exporter /usr/local/bin/rdma_exporter

# Mirrors the role's systemd WorkingDirectory=/var/lib/rdma-exporter.
WORKDIR /var/lib/rdma-exporter

EXPOSE 9879

# Defaults mirror the role's systemd ExecStart; override the CMD to change them.
ENTRYPOINT ["/usr/local/bin/rdma_exporter"]
CMD ["--listen-address=:9879", "--metrics-path=/metrics", "--health-path=/health"]
