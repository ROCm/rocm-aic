# shellcheck shell=bash
# shellcheck disable=SC2015  # `launch && log ok || log fail` is intentional; log() never fails
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Shared metrics-capture helpers: bring up the Prometheus sidecar + the exporter
# fleet via `docker run` (the Markham GPU nodes have docker but not the compose
# plugin), health-check each exporter, summarize what landed in the TSDB, and
# tear the stack down.  SOURCED (not executed) by:
#
#   .slurm/run-cliff.sbatch          -- capture metrics across the cliff sweep
#   .slurm/run-build-distribute.sh   -- the `smoke-test` exporter sanity check
#
# so both drive metrics identically and there is one place to fix.
#
# Caller contract (set before calling the functions; the defaults below keep a
# bare source harmless and the linter quiet):
#   log()             -- logger; a no-op '[monitoring] ...' printer is provided
#                        if the caller has not defined one.
#   AAI_IMAGE         -- aai-day image tag (hsa-snoop / ais-snoop run from it).
#   MON_DIR           -- path to the monitoring/ dir (prometheus.yml, configs).
#   AAI_METRICS_DIR   -- Prometheus TSDB dir (bind-mounted into the container).
#   MON_COMPOSE       -- optional compose file; used when the plugin is present.
#   AAI_EXPORTERS     -- 1 to also launch the exporter fleet (0 = Prometheus only).
#   AAI_MONITORING    -- 1 to enable start_monitoring/stop_monitoring at all.
#   Optional: AAI_PROM_IMAGE/PORT/RETENTION, AAI_{NODE,AMDGPU,NVME,RDMA}_EXPORTER_IMAGE,
#             AAI_{NVME,RDMA}_EXPORTER_ARGS, {NVME,RDMA}_EXPORTER_PORT,
#             {HSA,AIS}_SNOOP_PORT, AIS_KFD_SYMBOL.

# --- Caller-provided vars: defaults so the lib is self-contained + SC2154-clean.
: "${AAI_IMAGE:=rocm-aic-aai-day:latest}"
: "${AAI_MONITORING:=1}"
: "${AAI_EXPORTERS:=1}"
: "${AAI_METRICS_DIR:=}"
: "${MON_DIR:=}"
: "${MON_COMPOSE:=}"
# Provide a fallback logger only if the caller has not defined one (run-cliff.sbatch
# and run-build-distribute.sh each define their own prefixed log()).
declare -F log >/dev/null 2>&1 || log() { printf '[monitoring] %s\n' "$*" >&2; }

have_compose() { docker compose version >/dev/null 2>&1; }

mon_profile() {  # echo the compose --profile args for the exporters profile
    [[ "${AAI_EXPORTERS}" == "1" ]] && printf -- '--profile\nexporters\n'
}

# Compose service container_names -- reused by the docker-run fallback so both
# paths produce identically-named containers and share one teardown.
MON_CONTAINERS=(aai-day-prometheus aai-day-node-exporter aai-day-amdgpu-exporter
                aai-day-nvme-exporter aai-day-rdma-exporter aai-day-hsa-snoop
                aai-day-ais-snoop)

# True (0) if something is already listening on the given local TCP port -- used
# to skip launching a containerized exporter when a host exporter serves it
# already (Prometheus scrapes the host one on the same port either way).
_port_in_use() { timeout 1 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2>/dev/null; }

# True (0) if the /metrics endpoint on the given local port actually serves AMD
# GPU device-metrics series (gpu_* or amd_gpu_*).  Distinguishes a real GPU
# exporter from a stray/empty listener occupying the port (see amdgpu launch).
_endpoint_serves_gpu() {
    curl -fsS --max-time 2 "http://127.0.0.1:$1/metrics" 2>/dev/null | grep -qE '^(amd_)?gpu_'
}

# Launch one exporter container unless its port is already served on the host.
# $1=container name  $2=port  rest=docker run flags + image + command.
_run_exporter() {
    local name="$1" port="$2"; shift 2
    if _port_in_use "${port}"; then
        log "  ${name}: :${port} already served (host exporter?), skipping container"
        return 0
    fi
    docker rm -f "${name}" >/dev/null 2>&1 || true
    docker run -d --name "${name}" "$@" >/dev/null \
        && log "  ${name} on :${port}" \
        || log "  ${name}: docker run failed"
}

# docker run fallback for nodes without the compose plugin (the Markham GPU
# nodes have docker but not `docker compose`).  Mirrors the services in
# monitoring/docker-compose.monitoring.yml: same images, host networking, ports,
# mounts, and container names.  Containerized exporters are skipped when a host
# exporter already serves the port (see _run_exporter).  Idempotent (rm -f first).
_monitoring_run_up() {
    docker rm -f "${MON_CONTAINERS[@]}" >/dev/null 2>&1 || true

    # prometheus (always, our capture process -- not port-skipped): scrape
    # localhost targets, TSDB on AAI_METRICS_DIR.
    docker run -d --name aai-day-prometheus --network host --restart unless-stopped \
        --user "$(id -u):$(id -g)" \
        -v "${MON_DIR}/prometheus/prometheus.yml":/etc/prometheus/prometheus.yml:ro \
        -v "${MON_DIR}/prometheus/rules":/etc/prometheus/rules:ro \
        -v "${AAI_METRICS_DIR}":/prometheus \
        "${AAI_PROM_IMAGE:-prom/prometheus:v2.55.1}" \
        --config.file=/etc/prometheus/prometheus.yml \
        --storage.tsdb.path=/prometheus \
        --storage.tsdb.retention.time="${AAI_PROM_RETENTION:-90d}" \
        --web.enable-lifecycle \
        --web.listen-address=":${AAI_PROM_PORT:-9090}" >/dev/null \
        && log "  prometheus on :${AAI_PROM_PORT:-9090}" \
        || log "  prometheus: docker run failed"

    [[ "${AAI_EXPORTERS}" == "1" ]] || return 0

    # node-exporter: host CPU/mem/net + NVMe (diskstats/nvme) + RDMA (infiniband).
    _run_exporter aai-day-node-exporter 9100 \
        --network host --pid host --restart unless-stopped -v /:/host:ro,rslave \
        "${AAI_NODE_EXPORTER_IMAGE:-quay.io/prometheus/node-exporter:v1.8.2}" \
        --path.rootfs=/host --collector.diskstats --collector.nvme \
        --collector.infiniband --web.listen-address=":9100"

    # amdgpu-exporter: AMD device-metrics-exporter (gpu_* metrics) on :5000.
    # A bare TCP check (_port_in_use) is NOT enough here: some nodes have an
    # unrelated/empty listener on :5000 (e.g. a stray promhttp handler) that
    # serves no gpu_* series, and skipping our container in favour of it leaves
    # the GPU panels blank.  So: reuse :5000 only if it actually serves gpu_*
    # metrics; if :5000 is busy but is a phantom, launch ours on :5050 instead
    # (prometheus.yml scrapes both); if :5000 is free, use it.
    if _port_in_use 5000 && _endpoint_serves_gpu 5000; then
        log "  aai-day-amdgpu-exporter: real GPU exporter already on :5000, reusing"
    else
        local amdgpu_port=5000 amdgpu_cfg="${MON_DIR}/amdgpu-exporter/config.json"
        if _port_in_use 5000; then
            amdgpu_port=5050
            amdgpu_cfg="/tmp/aai-amdgpu-config.${SLURM_JOB_ID:-$$}.json"
            sed 's/"ServerPort":[[:space:]]*5000/"ServerPort": 5050/' \
                "${MON_DIR}/amdgpu-exporter/config.json" > "${amdgpu_cfg}"
            log "  aai-day-amdgpu-exporter: :5000 busy but not a GPU exporter (phantom); using :5050"
        fi
        docker rm -f aai-day-amdgpu-exporter >/dev/null 2>&1 || true
        docker run -d --name aai-day-amdgpu-exporter \
            --network host --restart unless-stopped --device /dev/kfd --device /dev/dri \
            -v /sys:/sys:ro \
            -v "${amdgpu_cfg}":/etc/metrics/config.json:ro \
            "${AAI_AMDGPU_EXPORTER_IMAGE:-rocm/device-metrics-exporter:v1.4.2}" >/dev/null \
            && log "  aai-day-amdgpu-exporter on :${amdgpu_port}" \
            || log "  aai-day-amdgpu-exporter: docker run failed"
    fi

    # nvme-exporter (:9998, dedicated NVMe metrics).  No standard published image
    # (batesste host service); set AAI_NVME_EXPORTER_IMAGE to containerize it,
    # else rely on the host service or node-exporter's nvme/diskstats collectors.
    if [[ -n "${AAI_NVME_EXPORTER_IMAGE:-}" ]]; then
        # shellcheck disable=SC2086  # AAI_NVME_EXPORTER_ARGS is an intentional arg list
        _run_exporter aai-day-nvme-exporter "${NVME_EXPORTER_PORT:-9998}" \
            --network host --pid host --privileged --restart unless-stopped \
            -v /dev:/dev -v /sys:/sys:ro \
            "${AAI_NVME_EXPORTER_IMAGE}" ${AAI_NVME_EXPORTER_ARGS:-}
    else
        log "  nvme-exporter: set AAI_NVME_EXPORTER_IMAGE to containerize (else host service / node-exporter nvme collector)"
    fi

    # rdma-exporter (:9879, yuuki-style, sysfs).  Build-your-own image (no
    # published one -- github.com/yuuki/rdma_exporter ships a Dockerfile); set
    # AAI_RDMA_EXPORTER_IMAGE to containerize, else rely on the host service or
    # node-exporter's infiniband collector.
    if [[ -n "${AAI_RDMA_EXPORTER_IMAGE:-}" ]]; then
        # shellcheck disable=SC2086  # AAI_RDMA_EXPORTER_ARGS is an intentional arg list
        _run_exporter aai-day-rdma-exporter "${RDMA_EXPORTER_PORT:-9879}" \
            --network host --restart unless-stopped -v /sys:/sys:ro \
            "${AAI_RDMA_EXPORTER_IMAGE}" ${AAI_RDMA_EXPORTER_ARGS:-}
    else
        log "  rdma-exporter: set AAI_RDMA_EXPORTER_IMAGE to containerize (else host service / node-exporter infiniband collector)"
    fi

    # hsa-snoop (:9488): HSA AQL queue snooper from the aai-day image.  It installs
    # its kprobe via tracefs at /sys/kernel/tracing -- so BOTH debugfs and tracefs
    # must be mounted; with only /sys/kernel/debug it fails with ENOENT ("failed
    # to install kprobe").  Needs root + privileged + host PID ns to see the
    # vLLM/LMCache GPU processes.
    if _port_in_use "${HSA_SNOOP_PORT:-9488}"; then
        log "  hsa-snoop: :${HSA_SNOOP_PORT:-9488} already served, skipping container"
    else
        docker rm -f aai-day-hsa-snoop >/dev/null 2>&1 || true
        local -a _tracefs=()
        [[ -d /sys/kernel/tracing ]] && _tracefs=(-v /sys/kernel/tracing:/sys/kernel/tracing)
        docker run -d --name aai-day-hsa-snoop --network host --pid host --privileged \
            --restart unless-stopped --device /dev/kfd --device /dev/dri \
            -v /sys/kernel/debug:/sys/kernel/debug "${_tracefs[@]}" \
            --entrypoint /usr/local/bin/hsa-snoop "${AAI_IMAGE}" \
            --all --prometheus --prometheus-port "${HSA_SNOOP_PORT:-9488}" >/dev/null \
            && log "  hsa-snoop on :${HSA_SNOOP_PORT:-9488}" \
            || log "  hsa-snoop: docker run failed"
    fi

    # ais-snoop (:9489): AIS (AMD Infinity Storage / hipFile GDS/P2PDMA) KFD
    # kprobe exporter from the aai-day image.  Sibling of hsa-snoop: same
    # privileged + host-PID + tracefs model, plus /lib/modules + /usr/src so
    # bpftrace can resolve kernel symbols (and /sys/kernel/btf when present, for
    # future arg introspection).  The kprobe target is AIS_KFD_SYMBOL -- unset
    # ->ais_snoop_up=0{reason="no_symbol_configured"}, so it's harmless to launch
    # before the symbol is chosen.
    if _port_in_use "${AIS_SNOOP_PORT:-9489}"; then
        log "  ais-snoop: :${AIS_SNOOP_PORT:-9489} already served, skipping container"
    else
        docker rm -f aai-day-ais-snoop >/dev/null 2>&1 || true
        local -a _aisfs=()
        [[ -d /sys/kernel/tracing ]] && _aisfs+=(-v /sys/kernel/tracing:/sys/kernel/tracing)
        [[ -d /sys/kernel/btf ]] && _aisfs+=(-v /sys/kernel/btf:/sys/kernel/btf:ro)
        docker run -d --name aai-day-ais-snoop --network host --pid host --privileged \
            --restart unless-stopped --device /dev/kfd --device /dev/dri \
            -v /sys/kernel/debug:/sys/kernel/debug \
            -v /lib/modules:/lib/modules:ro -v /usr/src:/usr/src:ro "${_aisfs[@]}" \
            -e AIS_KFD_SYMBOL="${AIS_KFD_SYMBOL:-}" \
            --entrypoint /usr/local/bin/ais-snoop "${AAI_IMAGE}" \
            --prometheus-port "${AIS_SNOOP_PORT:-9489}" >/dev/null \
            && log "  ais-snoop on :${AIS_SNOOP_PORT:-9489} (symbol='${AIS_KFD_SYMBOL:-<unset>}')" \
            || log "  ais-snoop: docker run failed"
    fi
}

start_monitoring() {
    [[ "${AAI_MONITORING}" == "1" ]] || { log "monitoring disabled (AAI_MONITORING=0)"; return 0; }
    mkdir -p "${AAI_METRICS_DIR}" 2>/dev/null || { log "monitoring: cannot create ${AAI_METRICS_DIR}, skipping"; AAI_MONITORING=0; return 0; }
    log "starting metrics capture -> ${AAI_METRICS_DIR} (exporters=${AAI_EXPORTERS})"
    if have_compose && [[ -f "${MON_COMPOSE}" ]]; then
        local -a profile; mapfile -t profile < <(mon_profile)
        AAI_METRICS_DIR="${AAI_METRICS_DIR}" PROM_UID="$(id -u)" PROM_GID="$(id -g)" \
            IMAGE_NAME="${AAI_IMAGE}" \
            AIS_KFD_SYMBOL="${AIS_KFD_SYMBOL:-}" AIS_SNOOP_PORT="${AIS_SNOOP_PORT:-9489}" \
            docker compose -f "${MON_COMPOSE}" "${profile[@]}" up -d \
            || log "monitoring: compose up failed (continuing without metrics)"
    else
        # No compose plugin on this node -> equivalent docker-run sidecar.
        log "monitoring: 'docker compose' unavailable; using docker run sidecar"
        _monitoring_run_up
    fi
}

stop_monitoring() {
    [[ "${AAI_MONITORING}" == "1" ]] || return 0
    log "stopping metrics capture (TSDB retained at ${AAI_METRICS_DIR})"
    # Compose down when available; then rm any docker-run containers (same names).
    if have_compose && [[ -f "${MON_COMPOSE}" ]]; then
        local -a profile; mapfile -t profile < <(mon_profile)
        AAI_METRICS_DIR="${AAI_METRICS_DIR}" IMAGE_NAME="${AAI_IMAGE}" \
            docker compose -f "${MON_COMPOSE}" "${profile[@]}" down >/dev/null 2>&1 || true
    fi
    docker rm -f "${MON_CONTAINERS[@]}" >/dev/null 2>&1 || true
}

# --- Health check + TSDB summary (used by the build smoke-test) ---------------
# curl one exporter's /metrics and report whether it serves its expected series.
# Purely informational: prints OK/WARN and returns 0/1, never exits.
# $1=display name  $2=port  $3=extended-regex the metrics should contain.
_check_endpoint() {
    local name="$1" port="$2" re="$3" body n
    body="$(curl -fsS --max-time 3 "http://127.0.0.1:${port}/metrics" 2>/dev/null)" || {
        log "  WARN ${name} :${port} -- /metrics not responding"; return 1; }
    n="$(printf '%s\n' "${body}" | grep -cE "${re}" 2>/dev/null || true)"
    if [[ "${n:-0}" -gt 0 ]]; then
        log "  OK   ${name} :${port} -- ${n} ${re} series"; return 0
    fi
    log "  WARN ${name} :${port} -- responding but no ${re} series"; return 1
}

# Probe every exporter's /metrics endpoint.  Informational only (the caller
# decides whether to act on the result); the amdgpu exporter is retried on :5050
# to match _monitoring_run_up's phantom-:5000 fallback.
monitoring_healthcheck() {
    log "exporter health check (/metrics):"
    _check_endpoint node-exporter   9100                           '^node_'       || true
    if ! _check_endpoint amdgpu-exporter 5000 '^(amd_)?gpu_'; then
        _check_endpoint amdgpu-exporter 5050 '^(amd_)?gpu_' || true
    fi
    _check_endpoint nvme-exporter   "${NVME_EXPORTER_PORT:-9998}"  '^nvme_'       || true
    _check_endpoint rdma-exporter   "${RDMA_EXPORTER_PORT:-9879}"  '^rdma_'       || true
    _check_endpoint hsa-snoop       "${HSA_SNOOP_PORT:-9488}"      '^hsa_'        || true
    _check_endpoint ais-snoop       "${AIS_SNOOP_PORT:-9489}"      '^ais_'        || true
    _check_endpoint prometheus      "${AAI_PROM_PORT:-9090}"       '^prometheus_' || true
}

# After a scrape window, ask Prometheus which targets are up (confirms data
# actually landed in the TSDB) and print the TSDB path/size.  Uses the aai-day
# image's python3 to parse the query JSON (no jq on the host).  Informational.
monitoring_tsdb_summary() {
    local prom="http://127.0.0.1:${AAI_PROM_PORT:-9090}"
    log "Prometheus scrape targets (query up):"
    if ! curl -fsS --max-time 3 "${prom}/-/ready" >/dev/null 2>&1; then
        log "  WARN Prometheus not ready at ${prom}"; return 0
    fi
    docker run --rm --network host -e PROM_URL="${prom}" \
        --entrypoint python3 "${AAI_IMAGE}" -c '
import json, os, sys, urllib.request
prom = os.environ["PROM_URL"]
try:
    d = json.load(urllib.request.urlopen(prom + "/api/v1/query?query=up", timeout=5))
except Exception as e:
    print("WARN could not query up:", e); sys.exit(0)
res = d.get("data", {}).get("result", [])
if not res:
    print("WARN no up series yet"); sys.exit(0)
for r in sorted(res, key=lambda x: x["metric"].get("job", "")):
    m = r["metric"]
    print("up=%s job=%s instance=%s" % (r["value"][1], m.get("job", "?"), m.get("instance", "?")))
' 2>&1 | sed 's/^/    /' || log "  WARN tsdb summary query failed"
    if [[ -n "${AAI_METRICS_DIR}" && -d "${AAI_METRICS_DIR}" ]]; then
        log "  TSDB: ${AAI_METRICS_DIR} ($(du -sh "${AAI_METRICS_DIR}" 2>/dev/null | cut -f1))"
    fi
}
