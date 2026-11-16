# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Containerized RDMA Prometheus exporter for the  metrics stack.
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
# Build (from repository root).  docker/docker-bake.hcl is the only owner of
# RDMA_EXPORTER_VERSION; there is no default here and the build stops without it:
#   docker buildx bake -f docker/docker-bake.hcl rdma-exporter --load
#   # or pin a version:
#   RDMA_EXPORTER_VERSION=<version> docker buildx bake \
#       -f docker/docker-bake.hcl rdma-exporter --load
#   # or a raw build with bake's resolved args:
#   mapfile -t args < <(docker/scripts/aic-bake-env.sh --build-args rdma-exporter)
#   docker build "${args[@]}" -t aic-rdma-exporter:local ./docker/rdma-exporter
#
# Run (on the aic bridge, no published port: Prometheus scrapes it by name;
# needs sysfs):
#   docker run -d --name aic-rdma-exporter --network aic \
#       -v /sys:/sys:ro aic-rdma-exporter:local

FROM debian:12-slim AS fetch

ARG RDMA_EXPORTER_VERSION
# Provided by buildx; default keeps a plain `docker build` working on x86_64.
ARG TARGETARCH=amd64

RUN set -eux; \
    : "${RDMA_EXPORTER_VERSION:?unset: build through docker/docker-bake.hcl}"; \
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

LABEL org.opencontainers.image.title="aic-rdma-exporter" \
      org.opencontainers.image.description="Prometheus RDMA exporter (yuuki/rdma_exporter) for the  metrics stack" \
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

# debian-slim ships bash but neither curl nor python, so probe over /dev/tcp.
# Asserts an HTTP 200 from the exporter's own health endpoint.  The port and
# path are the CMD defaults above; a CMD override that changes them has to
# override this check too.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/9879 && printf 'GET /health HTTP/1.0\\r\\n\\r\\n' >&3 && read -r status <&3 && [[ $status == *' 200 '* ]]"]
