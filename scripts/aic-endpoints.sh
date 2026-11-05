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
#   scripts/aic-endpoints.sh bridge-url <container> <container-port>
#   scripts/aic-endpoints.sh probe <container> <container-port> <path>
#   scripts/aic-endpoints.sh fetch <url> [<json-body>]
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

# aic_bridge_url <container> <container-port>
# Print http://<name>:<port> for reaching <container> from the aic bridge, the
# counterpart of aic_url for clients that run in a container (aic_fetch, a
# benchmark).  The name is the container owning <container>'s network namespace,
# not a shared alias like aic-vllm: container names are unique on the node, an
# alias is not when two stacks share the bridge.
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
# loops.  Uses the image's python3: the AIC images carry no curl.
aic_probe() {
    if (( $# != 3 )); then
        echo "usage: aic_probe <container> <container-port> <path>" >&2
        return 2
    fi
    docker exec "$1" python3 -c \
        'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=3)' \
        "http://127.0.0.1:$2$3" >/dev/null 2>&1
}

# aic_fetch <url> [<json-body>]
# Send one request from a throwaway container on the aic bridge and print the
# response body, so the target needs no published port: address it by name
# (http://aic-vllm:8000/v1/models).  GET, or a JSON POST when a body is given.
# Non-zero, with the reason on stderr, on a connection failure (7) or an HTTP
# error status (22), as `curl -fsS` would be.
#
# The client runs from AIC_FETCH_IMAGE, which needs python3; set it to an image
# already on the node (any AIC image) where a registry pull is unwelcome.
# AIC_FETCH_TIMEOUT bounds the request in seconds (default: 60).
aic_fetch() {
    if (( $# < 1 || $# > 2 )); then
        echo "usage: aic_fetch <url> [<json-body>]" >&2
        return 2
    fi
    docker run --rm --network "${AIC_NETWORK:-aic}" \
        -e AIC_FETCH_TIMEOUT="${AIC_FETCH_TIMEOUT:-60}" \
        --entrypoint python3 "${AIC_FETCH_IMAGE:-python:3.12-slim}" -c '
import os, sys, urllib.error, urllib.request
url = sys.argv[1]
body = sys.argv[2].encode() if len(sys.argv) > 2 else None
req = urllib.request.Request(url, data=body)
if body is not None:
    req.add_header("Content-Type", "application/json")
try:
    with urllib.request.urlopen(req, timeout=float(os.environ["AIC_FETCH_TIMEOUT"])) as r:
        sys.stdout.buffer.write(r.read())
except urllib.error.HTTPError as e:
    detail = e.read()[:500].decode(errors="replace")
    sys.stderr.write(f"aic_fetch: {url}: HTTP {e.code}: {detail}\n")
    sys.exit(22)
except Exception as e:
    sys.stderr.write(f"aic_fetch: {url}: {e}\n")
    sys.exit(7)
' "$@"
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
        url)        shift; aic_url "$@" ;;
        bridge-url) shift; aic_bridge_url "$@" ;;
        probe)      shift; aic_probe "$@" ;;
        fetch)      shift; aic_fetch "$@" ;;
        list)       aic_endpoints ;;
        *)          echo "usage: ${0} url <container> <container-port> | bridge-url <container> <container-port> | probe <container> <container-port> <path> | fetch <url> [<json-body>] | list" >&2; exit 2 ;;
    esac
fi
