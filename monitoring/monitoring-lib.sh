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
#             AIC_{NVME,RDMA}_EXPORTER_ARGS, {NVME,RDMA}_EXPORTER_PORT, HSA_SNOOP_PORT,
#             AIC_HSA_SNOOP_PID_MODE.

export PATH="/usr/local/bin:${PATH}"

# --- Caller-provided vars: defaults so the lib is self-contained + SC2154-clean.
: "${AIC_IMAGE:=rocm-aic:latest}"
: "${AIC_MONITORING:=1}"
: "${AIC_EXPORTERS:=1}"
: "${AIC_METRICS_DIR:=}"
: "${MON_DIR:=}"
: "${MON_COMPOSE:=}"
# Single `=`, not `:=`: an explicitly empty value means "no PID sharing" and
# must survive.  With `:=` it was silently rewritten to "host", so a caller
# that deliberately opted out of PID sharing got pid:host instead.
: "${AIC_HSA_SNOOP_PID_MODE=host}"
# Provide a fallback logger only if the caller has not defined one (run-cliff.sbatch
# and run-build-distribute.sh each define their own prefixed log()).
declare -F log >/dev/null 2>&1 || log() { printf '[monitoring] %s\n' "$*" >&2; }

# aic_url, for the host-side Prometheus queries below.
# shellcheck source=scripts/aic-endpoints.sh
source "${BASH_SOURCE[0]%/*}/../scripts/aic-endpoints.sh"

# --- GPU visibility resolution ------------------------------------------------
# Resolve GPU visibility for this job and export the values the compose files and
# `docker run` call sites interpolate:
#
#   AIC_ROCR_VISIBLE  absolute host GPU IDs   -> ROCR_VISIBLE_DEVICES
#   AIC_HIP_VISIBLE   relative indices 0..n-1 -> HIP_VISIBLE_DEVICES + CUDA_VISIBLE_DEVICES
aic_resolve_gpu_visibility() {
    local rocr n hip id from

    _aic_set_gpu_visibility() {
        local device_ids="$1"
        [[ "${device_ids}" =~ ^[^,[:space:]]+(,[^,[:space:]]+)*$ ]] || {
            printf 'aic_resolve_gpu_visibility: malformed GPU visibility: %s\n' "${device_ids}" >&2
            return 1
        }
        n="$(awk -F, '{print NF}' <<<"${device_ids}")"
        hip="$(seq -s, 0 "$((n - 1))")"
        export AIC_ROCR_VISIBLE="${device_ids}" AIC_HIP_VISIBLE="${hip}"
    }

    # Standard Slurm has already selected the GPU and exposes that choice via
    # its normal visibility environment.  Do not replace it with GPU 0.
    if [[ "${AIC_SPUR_CLUSTER:-0}" != "1" ]]; then
        rocr="${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-${GPU:-0}}}}"
        _aic_set_gpu_visibility "${rocr}" || return 1
        log "GPU visibility: ROCR=${AIC_ROCR_VISIBLE} HIP=${AIC_HIP_VISIBLE}"
        return 0
    fi

    # SPUR exports the job's positional host GPU indices into the batch
    # environment.  Never query the controller for them: its GetJob device id
    # is a PCI-location ID, not an index.
    if [[ -n "${SPUR_JOB_GPUS:-}" ]]; then
        rocr="${SPUR_JOB_GPUS}" from=SPUR_JOB_GPUS
    elif [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
        rocr="${ROCR_VISIBLE_DEVICES}" from=ROCR_VISIBLE_DEVICES
    else
        printf 'aic_resolve_gpu_visibility: SPUR_JOB_GPUS and ROCR_VISIBLE_DEVICES are empty for job %s (is a GPU requested?)\n' \
            "${SLURM_JOB_ID:-<none>}" >&2
        return 1
    fi

    # Positional indices are small; anything >= 64 (8 GPUs x 8 partitions) is
    # an ID of some other kind and must not reach `docker run`.
    [[ "${rocr}" =~ ^[0-9]+(,[0-9]+)*$ ]] \
        || { printf 'aic_resolve_gpu_visibility: malformed %s for job %s: %s\n' "${from}" "${SLURM_JOB_ID:-<none>}" "${rocr}" >&2; return 1; }
    for id in ${rocr//,/ }; do
        (( ${#id} <= 2 && 10#${id} < 64 )) \
            || { printf 'aic_resolve_gpu_visibility: %s for job %s is not a GPU index: %s\n' "${from}" "${SLURM_JOB_ID:-<none>}" "${rocr}" >&2; return 1; }
    done

    _aic_set_gpu_visibility "${rocr}" || return 1
    log "GPU allocation for job ${SLURM_JOB_ID:-<none>} (${from}): ROCR=${AIC_ROCR_VISIBLE} HIP=${AIC_HIP_VISIBLE}"
}

# Pre-flight check: detect a broken containerd overlay snapshotter before
# wasting time loading images or creating containers.  The symptom is:
#   "apply layer error: failed to open database file: ...metadata.db"
# which occurs on cluster nodes where containerd lost its snapshot store.
# Returns non-zero (caller should exit 1) if Docker or the snapshotter is broken.
aic_docker_preflight() {
    local host; host=$(hostname)
    docker info >/dev/null 2>&1 \
        || { printf '[preflight] FATAL: docker daemon unreachable on %s\n' "${host}" >&2; return 1; }
    docker system df >/dev/null 2>&1 \
        || { printf '[preflight] FATAL: docker overlay snapshotter broken on %s (containerd metadata.db missing?) -- resubmit to a different node\n' "${host}" >&2; return 1; }
    log "docker preflight OK on ${host}"
}

have_compose() { docker compose version >/dev/null 2>&1; }

# Ensure the `docker compose` (v2) plugin is available.  Docker checks
# $HOME/.docker/cli-plugins before the system dir, and $HOME is shared across the
# Slurm/SPUR nodes, so a user-local install fixes every node without root.  No-op
# when compose is already present.  Returns non-zero if it still cannot be made
# available (callers treat that as "skip metrics", never fatal).
: "${COMPOSE_PLUGIN_VERSION:=v5.5.1}"
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
    case "${AIC_EXPORTERS}" in
    1)
        # Full exporter fleet: node-exporter, amdgpu, hsa-snoop (all require pid:host).
        # Use only on nodes where the spur-authz plugin allows --pid=host.
        #
        # AIC_CPU_SMOKE=1: use the cpu-only exporter subset (node+nvme+rdma, no GPU
        # exporters).  Default: full exporters profile (includes amdgpu + hsa-snoop).
        if [[ "${AIC_CPU_SMOKE:-0}" == "1" ]]; then
            printf -- '--profile\nexporters-cpu\n'
        else
            printf -- '--profile\nexporters\n'
        fi
        # Also enable the fabric exporters (nvme/rdma) when their images are provided
        # (built by run-build-distribute.sh build-exporters, loaded on the node).
        [[ -n "${AIC_NVME_EXPORTER_IMAGE:-}${AIC_RDMA_EXPORTER_IMAGE:-}" ]] \
            && printf -- '--profile\nexporters-fabric\n'
        ;;
    safe)
        # Safe subset: amdgpu-exporter + hsa-snoop + rdma-exporter (container PID,
        # /sys:ro only).  Works on SPUR nodes where spur-authz blocks host-namespace
        # containers and /dev bind mounts.
        printf -- '--profile\nexporters-safe\n'
        # Add exporters-fabric (nvme + rdma) only on non-SPUR nodes; nvme-exporter
        # requires /dev:/dev:rw which spur-authz policy B2 blocks on SPUR.
        [[ "${AIC_SPUR_CLUSTER:-0}" != "1" ]] \
            && [[ -n "${AIC_RDMA_EXPORTER_IMAGE:-}" ]] \
            && docker image inspect "${AIC_RDMA_EXPORTER_IMAGE}" >/dev/null 2>&1 \
            && printf -- '--profile\nexporters-fabric\n'
        ;;
    esac
    # 0 or unset: no exporters
}

# Compose service container_names -- used to sweep up any leftovers from a
# crashed run before/after a compose up/down (compose down handles the rest).
MON_CONTAINERS=(aic-prometheus aic-grafana aic-loki aic-alloy
                aic-node-exporter aic-amdgpu-exporter
                aic-nvme-exporter aic-rdma-exporter aic-hsa-snoop
                aic-vllm-emulator)

start_monitoring() {
    [[ "${AIC_MONITORING}" == "1" ]] || { log "monitoring disabled (AIC_MONITORING=0)"; return 0; }
    mkdir -p "${AIC_METRICS_DIR}" 2>/dev/null || { log "monitoring: cannot create ${AIC_METRICS_DIR}, skipping"; AIC_MONITORING=0; return 0; }
    ensure_compose || { log "monitoring: docker compose unavailable, skipping"; AIC_MONITORING=0; return 0; }
    [[ -f "${MON_COMPOSE}" ]] || { log "monitoring: ${MON_COMPOSE} not found, skipping"; AIC_MONITORING=0; return 0; }
    log "starting metrics capture -> ${AIC_METRICS_DIR} (exporters=${AIC_EXPORTERS})"
    # Sweep any leftover containers from a previous run on this node so port
    # bindings don't conflict with the new compose up.
    docker rm -f "${MON_CONTAINERS[@]}" >/dev/null 2>&1 || true
    docker network inspect aic-network >/dev/null 2>&1 \
        || docker network create aic-network >/dev/null \
        || log "monitoring: could not create aic-network (continuing)"
    local -a profile; mapfile -t profile < <(mon_profile)
    local -a compose_files=(-f "${MON_COMPOSE}")
    # On SPUR, layer in the SPUR override (tmpfs rootfs for node-exporter,
    # no /dev bind for nvme-exporter) so spur-authz policies B1/B2 don't
    # block the compose up.
    if [[ "${AIC_SPUR_CLUSTER:-0}" == "1" ]]; then
        local _spur_override="${MON_COMPOSE%/*}/docker-compose.spur.yml"
        [[ -f "${_spur_override}" ]] \
            && compose_files+=(-f "${_spur_override}") \
            || log "monitoring: WARN spur override ${_spur_override} not found"
    fi
    # vllm carries no profile and client is in monitoring-base, so a bare `up`
    # would start both; the caller owns those.  Name the services instead of
    # relying on profiles alone.  `config` needs the same env as `up`: unset,
    # AIC_HSA_SNOOP_PID_MODE makes hsa-snoop depend on the inactive lmcache
    # service and the whole project fails validation.
    local -a env=(AIC_METRICS_DIR="${AIC_METRICS_DIR}" PROM_UID="$(id -u)" PROM_GID="$(id -g)"
                  IMAGE_NAME="${AIC_IMAGE}" AIC_HSA_SNOOP_PID_MODE="${AIC_HSA_SNOOP_PID_MODE}")
    local -a dc=(docker compose "${compose_files[@]}" --profile monitoring-base "${profile[@]}")
    local -a svc
    mapfile -t svc < <(env "${env[@]}" "${dc[@]}" config --services \
        | grep -vxE "vllm|client${AIC_MON_SKIP_SERVICES:+|${AIC_MON_SKIP_SERVICES// /|}}")
    # An empty list would make `up` start every active service, vllm included.
    if [[ "${#svc[@]}" -eq 0 ]]; then
        log "monitoring: no services resolved from compose config (continuing without metrics)"
        return 0
    fi
    env "${env[@]}" "${dc[@]}" up -d --no-deps "${svc[@]}" \
        || log "monitoring: compose up failed (continuing without metrics)"
}

stop_monitoring() {
    [[ "${AIC_MONITORING}" == "1" ]] || return 0
    log "stopping metrics capture (TSDB retained at ${AIC_METRICS_DIR})"
    # Not `docker compose down`: it is project-wide, so it also stops vllm (no
    # profile) while a test is still serving, and blocks on vLLM's unreaped
    # EngineCore children until the job's time limit.  Removing exactly the
    # containers start_monitoring brought up is bounded and enough.  Stop first
    # so prometheus gets SIGTERM and closes its WAL cleanly; rm -f alone kills.
    timeout 60 docker stop -t 10 "${MON_CONTAINERS[@]}" >/dev/null 2>&1 || true
    timeout 60 docker rm -f "${MON_CONTAINERS[@]}" >/dev/null 2>&1 || true
}

# --- Health check + TSDB summary (used by the build smoke-test) ---------------
# curl one exporter's /metrics and report whether it serves its expected series.
# Purely informational: prints OK/WARN and returns 0/1, never exits.
# $1=display name  $2=base URL  $3=extended-regex the metrics should contain.
_check_endpoint() {
    local name="$1" url="$2" re="$3" body n
    body="$(curl -fsS --max-time 3 "${url}/metrics" 2>/dev/null)" || {
        log "  WARN ${name} ${url} -- /metrics not responding"; return 1; }
    n="$(printf '%s\n' "${body}" | grep -cE "${re}" 2>/dev/null || true)"
    if [[ "${n:-0}" -gt 0 ]]; then
        log "  OK   ${name} ${url} -- ${n} ${re} series"; return 0
    fi
    log "  WARN ${name} ${url} -- responding but no ${re} series"; return 1
}

# Query prometheus for an exporter job's scrape status.  Returns 0 if at least
# one instance is up=1; logs OK/WARN.  Purely informational, never exits.
# Exporters live on aic-network only (no host port); prometheus is the proxy.
# $1=prometheus base URL  $2=display name  $3=prometheus job name
# $4=optional series regex to spot-check
_check_job() {
    local prom="$1" name="$2" job="$3" re="${4:-}"
    local result n_up
    result="$(curl -fsS --max-time 3 \
        "${prom}/api/v1/query?query=up%7Bjob%3D%22${job}%22%7D" 2>/dev/null)" || {
        log "  WARN ${name} -- prometheus not reachable"; return 1; }
    n_up="$(printf '%s' "${result}" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(sum(1 for r in d.get('data',{}).get('result',[]) if r.get('value',['','0'])[1]=='1'))
" 2>/dev/null || echo 0)"
    if [[ "${n_up:-0}" -gt 0 ]]; then
        if [[ -n "${re}" ]]; then
            # Spot-check series count from prometheus metric store
            local n_series
            n_series="$(curl -fsS --max-time 3 \
                "${prom}/api/v1/query?query=count%28${re}%7Bjob%3D%22${job}%22%7D%29" \
                2>/dev/null \
                | python3 -c "
import json,sys
d=json.load(sys.stdin)
r=d.get('data',{}).get('result',[])
print(int(r[0]['value'][1]) if r else 0)
" 2>/dev/null || echo 0)"
            log "  OK   ${name} (job=${job} up=${n_up} series=${n_series})"
        else
            log "  OK   ${name} (job=${job} up=${n_up})"
        fi
        return 0
    fi
    log "  WARN ${name} (job=${job} up=0)"; return 1
}

# Check every known exporter via prometheus scrape status.
# Prometheus is the only service with a host port (9090); all exporters are on
# aic-network only and are reached by prometheus via container name.
monitoring_healthcheck() {
    log "exporter health check (via prometheus scrape status):"
    local prom
    prom="$(aic_url aic-prometheus 9090)" || {
        log "  WARN prometheus -- cannot resolve its published port"; return 0; }
    # Direct check for prometheus itself
    _check_endpoint prometheus "${prom}" '^prometheus_' || true
    # All other exporters: checked via prometheus up metric
    _check_job "${prom}" node-exporter   node_exporter        || true
    _check_job "${prom}" amdgpu-exporter amd_metrics_exporter || true
    _check_job "${prom}" nvme-exporter   nvme_exporter        || true
    _check_job "${prom}" rdma-exporter   rdma_exporter        || true
    _check_job "${prom}" hsa-snoop       hsa_snoop            || true
}

# After a scrape window, ask Prometheus which targets are up (confirms data
# actually landed in the TSDB) and print the TSDB path/size.  Uses the AIC
# image's python3 to parse the query JSON (no jq on the host).  Informational.
monitoring_tsdb_summary() {
    local prom
    prom="$(aic_url aic-prometheus 9090)" || {
        log "  WARN Prometheus: cannot resolve its published port"; return 0; }
    log "Prometheus scrape targets (query up):"
    if ! curl -fsS --max-time 3 "${prom}/-/ready" >/dev/null 2>&1; then
        log "  WARN Prometheus not ready at ${prom}"; return 0
    fi
    # A client on the aic bridge, addressing prometheus by name (the host URL
    # above is for this shell's curl only).
    docker run --rm --network aic -e PROM_URL="http://aic-prometheus:9090" \
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
