# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Containerized NVMe Prometheus exporter for the aai-day-release metrics stack.
#
# Reproduces the batesste `nvme_exporter_setup` Ansible host service in a
# container so nodes WITHOUT the host systemd service (or without the compose
# plugin -- see the docker-run fallback in .slurm/run-cliff.sbatch) still export
# dedicated NVMe metrics on :9998.  Same upstream, same release binary, same
# default flags, so Prometheus/Grafana see identical `nvme_*` series either way.
#
#   Upstream: https://github.com/E4-Computer-Engineering/nvme-exporter
#   Binary shells out to nvme-cli, so the runtime image ships `nvme-cli`.
#
# Multi-arch: TARGETARCH is set automatically by `docker buildx` (amd64/arm64);
# it maps directly onto the release asset suffix.
#
# Build (from aai-day-release/monitoring/):
#   docker build -t aai-day-nvme-exporter:local ./nvme-exporter
#   # or pin a version:
#   docker build --build-arg NVME_EXPORTER_VERSION=3.0.0 \
#       -t aai-day-nvme-exporter:3.0.0 ./nvme-exporter
#
# Run (host networking; needs the NVMe devices + sysfs):
#   docker run -d --name aai-day-nvme-exporter --network host --pid host \
#       --privileged -v /dev:/dev -v /sys:/sys:ro aai-day-nvme-exporter:local

FROM debian:12-slim AS fetch

# Matches the nvme_exporter_setup role's "latest" at authoring time (v3.0.0).
ARG NVME_EXPORTER_VERSION=3.0.0
# Provided by buildx; default keeps a plain `docker build` working on x86_64.
ARG TARGETARCH=amd64

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl; \
    url="https://github.com/E4-Computer-Engineering/nvme-exporter/releases/download/v${NVME_EXPORTER_VERSION}/nvme_exporter_${NVME_EXPORTER_VERSION}_linux_${TARGETARCH}.tar.gz"; \
    mkdir -p /tmp/extract; \
    curl -fSL "$url" -o /tmp/nvme_exporter.tar.gz; \
    tar -xzf /tmp/nvme_exporter.tar.gz -C /tmp/extract; \
    install -m0755 "$(find /tmp/extract -type f -name nvme_exporter | head -1)" \
        /usr/local/bin/nvme_exporter; \
    /usr/local/bin/nvme_exporter --help >/dev/null 2>&1 || true

FROM debian:12-slim

LABEL org.opencontainers.image.title="aai-day-nvme-exporter" \
      org.opencontainers.image.description="Prometheus NVMe exporter (E4-Computer-Engineering/nvme-exporter) for the aai-day-release metrics stack" \
      org.opencontainers.image.source="https://github.com/E4-Computer-Engineering/nvme-exporter" \
      org.opencontainers.image.licenses="MIT"

# The exporter invokes nvme-cli; ca-certificates is harmless and aids debugging.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends nvme-cli ca-certificates; \
    rm -rf /var/lib/apt/lists/*

COPY --from=fetch /usr/local/bin/nvme_exporter /usr/local/bin/nvme_exporter

EXPOSE 9998

# Defaults mirror the role's systemd ExecStart; override the CMD to change them.
ENTRYPOINT ["/usr/local/bin/nvme_exporter"]
CMD ["--web.listen-address=:9998", "--web.telemetry-path=/metrics"]
