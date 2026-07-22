# shellcheck shell=bash
# shellcheck disable=SC2015  # `launch && log ok || log fail` is intentional; log() never fails
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Shared metrics-capture helpers: bring up the Prometheus capture stack + the
# exporter fleet via `docker compose`, health-check each exporter, summarize what
# landed in the TSDB, and tear the stack down.  SOURCED (not executed) by:
#
#   .slurm/run-cliff.sbatch          -- capture metrics across the cliff sweep
#   .slurm/run-build-distribute.sh   -- the `smoke-test` exporter sanity check
#
# so both drive metrics identically and there is one place to fix.
#
# The whole stack is `docker compose` (v2) only.  The old docker-run fallback for
# nodes without the compose plugin is gone; ensure_compose() below installs the
# plugin user-locally ($HOME/.docker/cli-plugins, shared across the Slurm nodes)
# when it is missing, so `docker compose` always resolves.
#
# Caller contract (set before calling the functions; the defaults below keep a
# bare source harmless and the linter quiet):
#   log()             -- logger; a no-op '[monitoring] ...' printer is provided
#                        if the caller has not defined one.
#   AIC_IMAGE         -- AIC image tag (hsa-snoop runs from it).
#   MON_DIR           -- path to the monitoring/ dir (prometheus.yml, configs).
#   AIC_METRICS_DIR   -- Prometheus TSDB dir (bind-mounted into the container).
#   MON_COMPOSE       -- optional compose file; used when the plugin is present.
#   AIC_EXPORTERS     -- 1 to also launch the exporter fleet (0 = Prometheus only).
#   AIC_MONITORING    -- 1 to enable start_monitoring/stop_monitoring at all.
#   Optional: AIC_PROM_IMAGE/PORT/RETENTION, AIC_{NODE,AMDGPU,NVME,RDMA}_EXPORTER_IMAGE,
#             AIC_{NVME,RDMA}_EXPORTER_ARGS, {NVME,RDMA}_EXPORTER_PORT, HSA_SNOOP_PORT.

# --- Caller-provided vars: defaults so the lib is self-contained + SC2154-clean.
: "${AIC_IMAGE:=rocm-aic:latest}"
: "${AIC_MONITORING:=1}"
: "${AIC_EXPORTERS:=1}"
: "${AIC_METRICS_DIR:=}"
: "${MON_DIR:=}"
: "${MON_COMPOSE:=}"
# Provide a fallback logger only if the caller has not defined one (run-cliff.sbatch
# and run-build-distribute.sh each define their own prefixed log()).
declare -F log >/dev/null 2>&1 || log() { printf '[monitoring] %s\n' "$*" >&2; }

have_compose() { docker compose version >/dev/null 2>&1; }

# Ensure the `docker compose` (v2) plugin is available.  Docker checks
# $HOME/.docker/cli-plugins before the system dir, and $HOME is shared across the
# Slurm/SPUR nodes, so a user-local install fixes every node without root.  No-op
# when compose is already present.  Returns non-zero if it still cannot be made
# available (callers treat that as "skip metrics", never fatal).
: "${COMPOSE_PLUGIN_VERSION:=v2.40.0}"
ensure_compose() {
    have_compose && return 0
    log "docker compose plugin missing; installing ${COMPOSE_PLUGIN_VERSION} -> ~/.docker/cli-plugins"
    local arch; arch="$(uname -m)"
    mkdir -p "${HOME}/.docker/cli-plugins" 2>/dev/null || return 1
    if curl -fsSL "https://github.com/docker/compose/releases/download/${COMPOSE_PLUGIN_VERSION}/docker-compose-linux-${arch}" \
            -o "${HOME}/.docker/cli-plugins/docker-compose" 2>/dev/null \
       && chmod +x "${HOME}/.docker/cli-plugins/docker-compose" 2>/dev/null \
       && have_compose; then
        log "docker compose installed: $(docker compose version --short 2>/dev/null)"
        return 0
    fi
    log "WARN: could not install docker compose (no egress?); metrics capture unavailable"
    return 1
}

mon_profile() {  # echo the compose --profile args for the exporter fleet
    [[ "${AIC_EXPORTERS}" == "1" ]] || return 0
    printf -- '--profile\nexporters\n'
    # Also enable the fabric exporters (nvme/rdma) when their images are provided
    # (built by run-build-distribute.sh build-exporters, loaded on the node).
    [[ -n "${AIC_NVME_EXPORTER_IMAGE:-}${AIC_RDMA_EXPORTER_IMAGE:-}" ]] \
        && printf -- '--profile\nexporters-fabric\n'
}

# Compose service container_names -- used to sweep up any leftovers from a
# crashed run before/after a compose up/down (compose down handles the rest).
MON_CONTAINERS=(aic-prometheus aic-node-exporter aic-amdgpu-exporter
                aic-nvme-exporter aic-rdma-exporter aic-hsa-snoop)

start_monitoring() {
    [[ "${AIC_MONITORING}" == "1" ]] || { log "monitoring disabled (AIC_MONITORING=0)"; return 0; }
    mkdir -p "${AIC_METRICS_DIR}" 2>/dev/null || { log "monitoring: cannot create ${AIC_METRICS_DIR}, skipping"; AIC_MONITORING=0; return 0; }
    ensure_compose || { log "monitoring: docker compose unavailable, skipping"; AIC_MONITORING=0; return 0; }
    [[ -f "${MON_COMPOSE}" ]] || { log "monitoring: ${MON_COMPOSE} not found, skipping"; AIC_MONITORING=0; return 0; }
    log "starting metrics capture -> ${AIC_METRICS_DIR} (exporters=${AIC_EXPORTERS})"
    local -a profile; mapfile -t profile < <(mon_profile)
    AIC_METRICS_DIR="${AIC_METRICS_DIR}" PROM_UID="$(id -u)" PROM_GID="$(id -g)" \
        IMAGE_NAME="${AIC_IMAGE}" \
        docker compose -f "${MON_COMPOSE}" "${profile[@]}" up -d \
        || log "monitoring: compose up failed (continuing without metrics)"
}

stop_monitoring() {
    [[ "${AIC_MONITORING}" == "1" ]] || return 0
    log "stopping metrics capture (TSDB retained at ${AIC_METRICS_DIR})"
    if have_compose && [[ -f "${MON_COMPOSE}" ]]; then
        local -a profile; mapfile -t profile < <(mon_profile)
        AIC_METRICS_DIR="${AIC_METRICS_DIR}" IMAGE_NAME="${AIC_IMAGE}" \
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
# decides whether to act on the result).
monitoring_healthcheck() {
    log "exporter health check (/metrics):"
    _check_endpoint node-exporter   9100                           '^node_'       || true
    _check_endpoint amdgpu-exporter 5000                           '^(amd_)?gpu_' || true
    _check_endpoint nvme-exporter   "${NVME_EXPORTER_PORT:-9998}"  '^nvme_'       || true
    _check_endpoint rdma-exporter   "${RDMA_EXPORTER_PORT:-9879}"  '^rdma_'       || true
    _check_endpoint hsa-snoop       "${HSA_SNOOP_PORT:-9488}"      '^hsa_\|^ais_' || true
    _check_endpoint prometheus      "${AIC_PROM_PORT:-9090}"       '^prometheus_' || true
}

# After a scrape window, ask Prometheus which targets are up (confirms data
# actually landed in the TSDB) and print the TSDB path/size.  Uses the AIC
# image's python3 to parse the query JSON (no jq on the host).  Informational.
monitoring_tsdb_summary() {
    local prom="http://127.0.0.1:${AIC_PROM_PORT:-9090}"
    log "Prometheus scrape targets (query up):"
    if ! curl -fsS --max-time 3 "${prom}/-/ready" >/dev/null 2>&1; then
        log "  WARN Prometheus not ready at ${prom}"; return 0
    fi
    docker run --rm --network host -e PROM_URL="${prom}" \
        --entrypoint python3 "${AIC_IMAGE}" -c '
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
    # The vllm/lmcache/nixl jobs are the inference stack, which the exporter
    # sanity check does not start -- so up=0 for those is expected here.
    log "  note: vllm/lmcache/nixl up=0 is expected unless an inference stack is serving"
    if [[ -n "${AIC_METRICS_DIR}" && -d "${AIC_METRICS_DIR}" ]]; then
        log "  TSDB: ${AIC_METRICS_DIR} ($(du -sh "${AIC_METRICS_DIR}" 2>/dev/null | cut -f1))"
    fi
}
