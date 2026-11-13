# shellcheck shell=bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Service discovery for the AIC stack.  SOURCED by host processes (make
# recipes, sbatch scripts, workflow steps) that must reach a container.
# Also runnable:
#
#   scripts/aic-endpoints.sh url <container> <container-port>
#   scripts/aic-endpoints.sh bridge-url <container> <container-port>
#   scripts/aic-endpoints.sh probe <container> <container-port> <path>
#   scripts/aic-endpoints.sh fetch <url> [<json-body>]
#   scripts/aic-endpoints.sh list
#   scripts/aic-endpoints.sh check-stack <compose-project-dir>
#
# Clients reach the stack from its bridge (the network named ${AIC_STACK}, see
# below), by container name: aic_fetch for a single request, aic_bridge_url for
# the name a containerized client should use.  Containers that join another's
# network namespace (vllm joins ${AIC_STACK}-lmcache's via `network_mode:
# container:`) are not on the bridge themselves; both follow the chain to the
# container that owns the namespace.
#
#   AIC_FETCH_IMAGE=aic-vllm:<tag> aic_fetch "$(aic_bridge_url "${AIC_STACK}-vllm-gpu0" 8000)/v1/models"
#
# Only Grafana, which people open in a browser, is published to the host, as
#
#   "${AIC_PUBLISH_ADDR:-127.0.0.1}:${AIC_GRAFANA_PORT:-}:3000"
#
# so its host port is ephemeral (Docker picks a free one) unless pinned.  Ask
# Docker where it landed instead of assuming a number:
#
#   url="$(aic_url "${AIC_STACK}-grafana" 3000)" || exit 1
#
# An ephemeral port changes whenever the publishing container is recreated, so
# resolve after `up`/restart and never cache the result across one.

# AIC_STACK names one stack on the Docker host: its compose project, its bridge
# network, and the prefix of every container name, so address a container as
# "${AIC_STACK}-lmcache", never as a bare aic-lmcache.  On that network the
# containers also answer to their fixed aic-* aliases, which is all a client on
# the bridge (aic_fetch, a benchmark) needs.  Exported because compose reads it.
#
# It defaults to aic-$SLURM_JOB_ID inside a Slurm (or SPUR) job, so two jobs on
# one node never share a name, and to aic elsewhere; set it to run a second
# stack by hand.  AIC_STACK_DERIVED records a value this default produced, so
# one inherited from elsewhere -- the login-node shell whose environment sbatch
# passes on -- is recomputed for this job instead of reused.  The Makefile
# applies the same rule.
if [[ -z "${AIC_STACK:-}" || "${AIC_STACK}" == "${AIC_STACK_DERIVED:-}" ]]; then
    AIC_STACK="aic${SLURM_JOB_ID:+-${SLURM_JOB_ID}}"
    AIC_STACK_DERIVED="${AIC_STACK}"
fi
export AIC_STACK AIC_STACK_DERIVED

# Print the container that owns <container>'s network namespace.  Fails, naming
# the container, if it does not exist or is not running.
_aic_netns_owner() {
    local owner="$1" state running mode name depth=0
    while :; do
        state="$(docker inspect -f '{{.State.Running}} {{.HostConfig.NetworkMode}} {{.Name}}' "${owner}" 2>&1)" || {
            echo "aic_url: no such container '${owner}': ${state//$'\n'/ }" >&2
            return 1
        }
        read -r running mode name <<< "${state}"
        name="${name#/}"
        if [[ "${running}" != "true" ]]; then
            echo "aic_url: container '${name}' is not running" >&2
            return 1
        fi
        if [[ "${mode}" != container:* ]]; then
            printf '%s\n' "${name}"
            return 0
        fi
        owner="${mode#container:}"
        if (( ++depth > 8 )); then
            echo "aic_url: network_mode chain from '$1' is longer than 8 hops" >&2
            return 1
        fi
    done
}

# aic_url <container> <container-port>
# Print http://<addr>:<host-port> for a TCP port of <container>.  Non-zero, with
# a message on stderr, if the container is not running or the port is not
# published.
aic_url() {
    local ctr="${1:-}" port="${2:-}" owner out line addr hostport
    if [[ -z "${ctr}" || -z "${port}" ]]; then
        echo "usage: aic_url <container> <container-port>" >&2
        return 2
    fi
    owner="$(_aic_netns_owner "${ctr}")" || return 1
    out="$(docker port "${owner}" "${port}/tcp" 2>&1)" || {
        echo "aic_url: ${ctr} port ${port}/tcp is not published" \
             "(network namespace owner: ${owner}): ${out}" >&2
        return 1
    }
    # One "addr:port" line per bind; prefer the IPv4 one.
    line="$(printf '%s\n' "${out}" | grep -v '^\[' | head -n1)"
    [[ -n "${line}" ]] || line="$(printf '%s\n' "${out}" | head -n1)"
    if [[ -z "${line}" ]]; then
        echo "aic_url: docker port printed nothing for ${owner} ${port}/tcp" >&2
        return 1
    fi
    addr="${line%:*}"
    hostport="${line##*:}"
    case "${addr}" in
        0.0.0.0) addr=127.0.0.1 ;;
        '[::]')  addr='[::1]' ;;
    esac
    printf 'http://%s:%s\n' "${addr}" "${hostport}"
}

# aic_bridge_url <container> <container-port>
# Print http://<name>:<port> for reaching <container> from the aic bridge, the
# counterpart of aic_url for clients that run in a container (aic_fetch, a
# benchmark).  The name is the container owning <container>'s network namespace,
# which unlike an alias like aic-vllm says which of its services is meant.
aic_bridge_url() {
    local owner
    if (( $# != 2 )); then
        echo "usage: aic_bridge_url <container> <container-port>" >&2
        return 2
    fi
    owner="$(_aic_netns_owner "$1")" || return 1
    printf 'http://%s:%s\n' "${owner}" "$2"
}

# aic_probe <container> <container-port> <path>
# Succeed if <container> answers http://127.0.0.1:<port><path> from inside its
# own network namespace, so it needs no published port.  Silent, for readiness
# loops.  The container's image needs curl.
aic_probe() {
    if (( $# != 3 )); then
        echo "usage: aic_probe <container> <container-port> <path>" >&2
        return 2
    fi
    docker exec "$1" curl -fsS --max-time 3 -o /dev/null \
        "http://127.0.0.1:$2$3" >/dev/null 2>&1
}

# aic_fetch <url> [<json-body>]
# Send one request from a throwaway container on the stack's bridge and print the
# response body, so the target needs no published port: address it by name
# (http://aic-vllm:8000/v1/models).  GET, or a JSON POST when a body is given.
# Fails as curl does: 7 on a connection failure, 28 on a timeout, 22 on an HTTP
# error status, whose body is still printed.
#
# curl runs from AIC_FETCH_IMAGE; set it to an image already on the node (any
# AIC image) where a registry pull is unwelcome.  AIC_FETCH_TIMEOUT bounds the
# request in seconds (default: 60).
aic_fetch() {
    if (( $# < 1 || $# > 2 )); then
        echo "usage: aic_fetch <url> [<json-body>]" >&2
        return 2
    fi
    local -a run=(docker run --rm --network "${AIC_STACK}" --entrypoint curl)
    local -a req=(-sS --fail-with-body --max-time "${AIC_FETCH_TIMEOUT:-60}")
    local image="${AIC_FETCH_IMAGE:-curlimages/curl:8.23.0}"
    if (( $# == 1 )); then
        "${run[@]}" "${image}" "${req[@]}" "$1"
    else
        # On stdin, so curl cannot read a body starting with @ as a file name.
        printf '%s' "$2" | "${run[@]}" -i "${image}" "${req[@]}" \
            -H 'Content-Type: application/json' --data-binary @- "$1"
    fi
}

# aic_endpoints
# Print a URL for every published port of every running container on the
# stack's bridge (by default only Grafana's), and note which containers share a
# bridged container's network namespace, whose ports would be listed on the
# owner's row.  Nothing here knows a container name or port.
aic_endpoints() {
    local net="${AIC_STACK}" ctr line cport bind addr hostport id joined ports names
    local -a owners=()
    names="$(docker ps --filter "network=${net}" --format '{{.Names}}')" || {
        echo "aic_endpoints: docker ps failed" >&2
        return 1
    }
    [[ -z "${names}" ]] || mapfile -t owners <<< "${names}"
    if (( ${#owners[@]} == 0 )); then
        echo "no containers are running on the '${net}' network" >&2
        return 1
    fi
    for ctr in "${owners[@]}"; do
        id="$(docker inspect -f '{{.Id}}' "${ctr}")" || return 1
        # Everything that joined this container's namespace with container:<id>.
        joined="$(docker ps --format '{{.Names}}' | while read -r j; do
            if [[ "$(docker inspect -f '{{.HostConfig.NetworkMode}}' "${j}")" == "container:${id}" ]]; then
                printf '%s ' "${j}"
            fi
        done)"
        ports="$(docker port "${ctr}")" || return 1
        while IFS= read -r line; do
            [[ -n "${line}" ]] || continue
            cport="${line%% -> *}"
            bind="${line##* -> }"
            [[ "${bind}" == '['* ]] && continue    # the IPv6 twin of a 0.0.0.0 bind
            addr="${bind%:*}"; hostport="${bind##*:}"
            [[ "${addr}" == 0.0.0.0 ]] && addr=127.0.0.1
            printf '%-24s %-10s http://%s:%s%s\n' "${ctr}" "${cport}" "${addr}" "${hostport}" \
                "${joined:+  (also serves: ${joined% })}"
        done <<< "${ports}"
    done
}

# _aic_stack_dirs <stack>
# Print the compose project directory of every container of <stack>, once each.
_aic_stack_dirs() {
    docker ps -a --filter "label=com.docker.compose.project=$1" \
        --format '{{.Label "com.docker.compose.project.working_dir"}}' | sort -u
}

# aic_stack_check <compose-project-dir>
# Fail, suggesting a free name, if AIC_STACK names a stack started from another
# checkout.  Compose would adopt that stack's containers as this one's and
# recreate or remove them; a Slurm job never gets here with a name in use, but
# two checkouts on one workstation both default to aic.  Restarting this
# checkout's own stack is fine.
aic_stack_check() {
    local dir="${1:-}" dirs user name n=1
    if (( $# != 1 )); then
        echo "usage: aic_stack_check <compose-project-dir>" >&2
        return 2
    fi
    dirs="$(_aic_stack_dirs "${AIC_STACK}")" || {
        echo "aic_stack_check: docker ps failed" >&2
        return 1
    }
    [[ -z "${dirs}" || "${dirs}" == "${dir}" ]] && return 0
    # Compose project names are lower case.
    user="$(id -un)"; user="${user,,}"
    name="${AIC_STACK}-${user}"
    while [[ -n "$(_aic_stack_dirs "${name}")" ]]; do
        name="${AIC_STACK}-${user}$(( ++n ))"
    done
    echo "AIC_STACK=${AIC_STACK} is already in use by the stack started from ${dirs//$'\n'/, }." >&2
    echo "Run this one under its own name, e.g. AIC_STACK=${name}, and pass the same" >&2
    echo "AIC_STACK to every later command that should reach it." >&2
    return 1
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    case "${1:-}" in
        url)         shift; aic_url "$@" ;;
        bridge-url)  shift; aic_bridge_url "$@" ;;
        probe)       shift; aic_probe "$@" ;;
        fetch)       shift; aic_fetch "$@" ;;
        list)        aic_endpoints ;;
        check-stack) shift; aic_stack_check "$@" ;;
        *)           echo "usage: ${0} url <container> <container-port> | bridge-url <container> <container-port> | probe <container> <container-port> <path> | fetch <url> [<json-body>] | list | check-stack <compose-project-dir>" >&2; exit 2 ;;
    esac
fi
