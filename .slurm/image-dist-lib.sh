# shellcheck shell=bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Node-side image distribution helpers.  SOURCED (not executed) by the jobs that
# .slurm/run-build-distribute.sh submits, and by the .sbatch scripts that run the
# images it builds.
#
# A build leaves one artifact per image in AIC_IMAGE_DIR, and every consumer
# goes through aic_image_fetch, so none of them care which kind it is:
#
#   <name>.tar.zst | .tar.gz | .tar   a `docker save` stream (AIC_DIST_BACKEND=tarball)
#   <name>.ref                        one line, "<storage> <repo>@<digest> <tag>...":
#                                     the image sits in a registry whose storage is
#                                     <storage> on shared NFS (AIC_DIST_BACKEND=registry)
#
# There is no long-running registry.  A job that needs one serves <storage> from
# its own node for as long as it needs it, on 127.0.0.1 with a Docker-chosen port
# (nodes are multi-tenant: no fixed host ports), so dockerd -- and the BuildKit
# inside it -- reach it as an insecure-by-default loopback registry.  Concurrent
# jobs each run their own registry on the same storage: blobs are content-addressed
# and every upload is staged in its own directory, so only a tag can race, and
# consumers pull by digest, never by tag.

# aic_registry_start LOGTAG STORAGE
# Serve STORAGE from a registry container on this node.  Sets AIC_REG_NAME,
# AIC_REG_PORT and AIC_REG_STORAGE for aic_registry_stop / aic_image_store.
aic_registry_start() {
    local tag="$1" storage="$2" image="${AIC_REGISTRY_IMAGE:-registry:3.1.2}" ports _
    AIC_REG_NAME="aic-registry-${SLURM_JOB_ID:-nojob}-$$-${RANDOM}"
    AIC_REG_STORAGE="${storage}"
    mkdir -p "${storage}" \
        || { echo "[${tag}] FATAL: cannot create registry storage ${storage}" >&2; return 1; }
    _aic_registry_image "${tag}" "${storage}" "${image}" || return 1
    docker run -d --name "${AIC_REG_NAME}" --user "$(id -u):$(id -g)" \
        -p 127.0.0.1::5000 -v "${storage}:/var/lib/registry" \
        -e REGISTRY_LOG_LEVEL=info "${image}" >/dev/null \
        || { echo "[${tag}] FATAL: could not start a registry (${image}) on $(hostname)" >&2; return 1; }
    # Not `| head -1`: under the caller's pipefail, head's early exit can SIGPIPE
    # docker and fail the lookup.
    ports="$(docker port "${AIC_REG_NAME}" 5000/tcp)"
    AIC_REG_PORT="${ports%%$'\n'*}"
    AIC_REG_PORT="${AIC_REG_PORT##*:}"
    for _ in $(seq 1 60); do
        curl -sf "http://127.0.0.1:${AIC_REG_PORT}/v2/" >/dev/null && break
        sleep 1
    done
    if ! curl -sf "http://127.0.0.1:${AIC_REG_PORT}/v2/" >/dev/null; then
        echo "[${tag}] FATAL: registry ${AIC_REG_NAME} not answering on 127.0.0.1:${AIC_REG_PORT}" >&2
        aic_registry_stop "${tag}" 1
        return 1
    fi
    echo "[${tag}] registry ${AIC_REG_NAME} on 127.0.0.1:${AIC_REG_PORT}, storage ${storage}"
}

# _aic_registry_image LOGTAG STORAGE IMAGE
# Make the registry IMAGE present in this node's dockerd without touching Docker
# Hub when possible.
_aic_registry_image() {
    local tag="$1" storage="$2" image="$3" cache tmp
    docker image inspect "${image}" >/dev/null 2>&1 && return 0
    cache="${storage%/}/aic-images/$(aic_image_stem "${image}").tar"
    if [[ -f "${cache}" ]]; then
        echo "[${tag}] loading ${image} on $(hostname) from ${cache}"
        docker load -i "${cache}" >/dev/null && return 0
        echo "[${tag}] WARN: could not load ${cache}; pulling ${image} instead" >&2
    fi
    docker pull -q "${image}" >/dev/null \
        || { echo "[${tag}] FATAL: could not pull ${image} on $(hostname), and no usable ${cache}" >&2; return 1; }
    # The pull succeeded, so a failed save costs only the next node a pull.
    # `docker save -o` creates the file 0600; other users share the storage.
    tmp="${cache}.partial.$$"
    if mkdir -p "${cache%/*}" && docker save -o "${tmp}" "${image}" \
        && chmod 0644 "${tmp}" && mv -f "${tmp}" "${cache}"; then
        echo "[${tag}] saved ${image} to ${cache}"
    else
        echo "[${tag}] WARN: could not save ${image} to ${cache}" >&2
        rm -f "${tmp}" || echo "[${tag}] WARN: could not remove ${tmp}" >&2
    fi
}

# aic_registry_stop LOGTAG [SHOW_LOG]
# Remove the registry aic_registry_start started, first printing the tail of its
# log when SHOW_LOG is 1.
aic_registry_stop() {
    local tag="$1" show_log="${2:-0}" rc
    [[ -n "${AIC_REG_NAME:-}" ]] || return 0
    if [[ "${show_log}" == "1" ]]; then
        echo "[${tag}] registry ${AIC_REG_NAME} log (last 20 lines):"
        docker logs --tail 20 "${AIC_REG_NAME}" 2>&1 | sed 's/^/    /' || {
            rc=$?
            echo "[${tag}] WARN: could not read the registry log (exit ${rc})" >&2
        }
    fi
    docker rm -f "${AIC_REG_NAME}" >/dev/null \
        || echo "[${tag}] WARN: could not remove registry container ${AIC_REG_NAME}" >&2
    AIC_REG_NAME="" AIC_REG_PORT=""
}

# aic_metadata_digest FILE
# Print the image digest a `docker buildx build|bake --metadata-file FILE` recorded.
aic_metadata_digest() {
    local d
    d="$(grep -m1 -oE '"containerimage\.digest": *"sha256:[0-9a-f]{64}"' "$1")" \
        || { echo "no containerimage.digest in $1" >&2; return 1; }
    d="${d##*\"sha256:}"
    printf 'sha256:%s\n' "${d%\"}"
}

# aic_image_store LOGTAG ARTIFACT PUSHED DIGEST TAG...
# Record in ARTIFACT (a .ref) that PUSHED -- localhost:<port>/<repo>:<tag>, just
# pushed to the running registry with manifest DIGEST -- is this build's image,
# to be tagged as each TAG wherever it is fetched.
aic_image_store() {
    local tag="$1" artifact="$2" pushed="$3" digest="$4" repo tmp
    shift 4
    [[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]] \
        || { echo "[${tag}] FATAL: no digest for ${pushed} (got '${digest}')" >&2; return 1; }
    repo="${pushed#*/}"
    repo="${repo%:*}"
    tmp="${artifact}.partial.$$"
    { printf '%s %s@%s %s\n' "${AIC_REG_STORAGE}" "${repo}" "${digest}" "$*" > "${tmp}" \
        && mv -f "${tmp}" "${artifact}"; } \
        || { echo "[${tag}] FATAL: could not write ${artifact}" >&2; return 1; }
    # The docker driver also keeps the image in dockerd under the port-specific
    # name it was pushed as, which no later job can use; drop the name.
    if docker image inspect "${pushed}" >/dev/null 2>&1; then
        docker rmi "${pushed}" >/dev/null \
            || echo "[${tag}] WARN: could not untag ${pushed} on $(hostname)" >&2
    fi
    echo "[${tag}] pushed ${repo}@${digest} -> ${artifact}"
}

# aic_image_stem IMAGE
# Print the file name a build gives IMAGE's artifact, before any arch suffix and
# extension: name:tag with '/' and ':' made '-' (repo/rocm-aic:1.0 ->
# repo-rocm-aic-1.0).
aic_image_stem() {
    printf '%s\n' "${1//[\/:]/-}"
}

# aic_image_artifact DIR STEM
# Print the newest artifact in DIR named STEM.ref, .tar.zst, .tar.gz or .tar.
# STEM may be a glob: "$(aic_image_stem IMAGE)-*" matches IMAGE built for any
# arch.  In-progress .partial.<pid> files never match.  Non-zero if none does.
aic_image_artifact() {
    local dir="$1" stem="$2" f newest=""
    # shellcheck disable=SC2086  # STEM is meant to glob
    for f in "${dir}"/${stem}.ref "${dir}"/${stem}.tar.zst "${dir}"/${stem}.tar.gz "${dir}"/${stem}.tar; do
        [[ -f "${f}" ]] || continue    # an unmatched glob stays literal
        [[ -z "${newest}" || "${f}" -nt "${newest}" ]] && newest="${f}"
    done
    [[ -n "${newest}" ]] || return 1
    printf '%s\n' "${newest}"
}

# aic_image_fetch LOGTAG IMAGE ARTIFACT
# Make IMAGE present in this node's dockerd from ARTIFACT.  A .ref is always
# pulled: by digest, so it is exactly what the build pushed, and layers already
# on the node are not fetched again.  A tarball is loaded only when forced
# (AIC_FORCE_LOAD=1), when IMAGE is absent, or when the tarball is newer than the
# one last loaded here: a per-uid+image marker under /var/tmp records the mtime
# last loaded, and comparing two build-side mtimes avoids build/node clock skew.
aic_image_fetch() {
    case "$3" in
        *.ref) _aic_image_pull "$@" ;;
        *) _aic_image_load "$@" ;;
    esac
}

_aic_image_pull() {
    local tag="$1" image="$2" artifact="$3" src t rc=0
    local -a fields
    read -r -a fields < "${artifact}" \
        || { echo "[${tag}] ERROR: cannot read ${artifact}" >&2; return 1; }
    [[ "${fields[1]:-}" == *@sha256:* ]] \
        || { echo "[${tag}] ERROR: ${artifact} names no image digest" >&2; return 1; }
    # The build's storage must be mounted here at the path it recorded.  Starting
    # a registry on a missing one would create it empty, and the pull would then
    # fail with an opaque "manifest unknown".
    [[ -d "${fields[0]}/docker/registry/v2" ]] || {
        echo "[${tag}] ERROR: registry storage ${fields[0]} (from ${artifact}) is not on $(hostname)" >&2
        return 1
    }
    aic_registry_start "${tag}" "${fields[0]}" || return 1
    src="localhost:${AIC_REG_PORT}/${fields[1]}"
    echo "[${tag}] pulling ${image} on $(hostname) (${fields[1]})"
    docker pull -q "${src}" >/dev/null || rc=$?
    if (( rc == 0 )); then
        for t in "${image}" "${fields[@]:2}"; do
            docker tag "${src}" "${t}" || { rc=$?; break; }
        done
        # Drop the port-specific name the pull left behind.
        docker rmi "${src}" >/dev/null \
            || echo "[${tag}] WARN: could not untag ${src} on $(hostname)" >&2
    fi
    aic_registry_stop "${tag}" "$(( rc != 0 ))"
    (( rc == 0 )) || echo "[${tag}] ERROR: could not fetch ${image} from ${fields[0]} (exit ${rc})" >&2
    return "${rc}"
}

_aic_image_load() {
    local tag="$1" image="$2" tarball="$3" marker tarm loadedm have
    local -
    set -o pipefail
    marker="/var/tmp/aic-loaded-$(id -u)-$(printf '%s' "${image}" | tr '/:' '--').mtime"
    tarm="$(stat -c %Y "${tarball}" 2>/dev/null || echo 0)"
    loadedm="$(cat "${marker}" 2>/dev/null || echo 0)"
    have="$(docker images -q "${image}")"
    if [[ "${AIC_FORCE_LOAD:-0}" != "1" && -n "${have}" && "${tarm}" -le "${loadedm}" ]]; then
        echo "[${tag}] ${image} up to date on $(hostname) (id ${have}); AIC_FORCE_LOAD=1 forces a reload"
        return 0
    fi
    echo "[${tag}] loading ${image} on $(hostname) from ${tarball} (tarball=${tarm} last-loaded=${loadedm} present=$([[ -n "${have}" ]] && echo yes || echo no) force=${AIC_FORCE_LOAD:-0})"
    case "${tarball}" in
        *.zst) zstd -dc "${tarball}" | docker load >/dev/null ;;
        *.gz) gzip -dc "${tarball}" | docker load >/dev/null ;;
        *) docker load -i "${tarball}" >/dev/null ;;
    esac || { echo "[${tag}] ERROR: could not load ${tarball}" >&2; return 1; }
    echo "${tarm}" > "${marker}" \
        || echo "[${tag}] WARN: could not record the load in ${marker}" >&2
}
