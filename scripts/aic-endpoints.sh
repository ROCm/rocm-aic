# shellcheck shell=bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Host-side service discovery for the AIC stack.  SOURCED by host processes
# (make recipes, sbatch scripts, workflow steps) that must reach a container.
# Also runnable:
#
#   scripts/aic-endpoints.sh url <container> <container-port>
#   scripts/aic-endpoints.sh list
#
# Nothing in the stack binds a fixed host port by default.  A service that has a
# host-side consumer is published as
#
#   "${AIC_PUBLISH_ADDR:-127.0.0.1}:${<SVC>_HOST_PORT:-}:<container-port>"
#
# so the host port is ephemeral (Docker picks a free one) unless <SVC>_HOST_PORT
# pins it.  Ask Docker where it landed instead of assuming a number:
#
#   url="$(aic_url aic-vllm-gpu0 8000)" || exit 1
#
# An ephemeral port changes whenever the publishing container is recreated, so
# resolve after `up`/restart and never cache the result across one.
#
# Containers that join another's network namespace (vllm joins aic-lmcache's via
# `network_mode: container:`) publish nothing themselves; aic_url follows the
# chain to the container that owns the namespace.

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

# aic_endpoints
# Print a URL for every published port of every running container on the aic
# bridge, and note which containers share a bridged container's network
# namespace (vllm shares aic-lmcache's, so vLLM's :8000 is listed on the
# aic-lmcache row).  Nothing here knows a container name or port.
aic_endpoints() {
    local net="${AIC_NETWORK:-aic}" ctr line cport bind addr hostport id joined ports names
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

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    case "${1:-}" in
        url)  shift; aic_url "$@" ;;
        list) aic_endpoints ;;
        *)    echo "usage: ${0} url <container> <container-port> | list" >&2; exit 2 ;;
    esac
fi
