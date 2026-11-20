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
#
# Garbage collection (aic_registry_gc) deletes blobs, so it must never overlap
# a job serving the storage: a push may just have been told a blob exists.
# Every registry holds a shared lock on <storage>/aic-gc/lock while it runs,
# and the collector holds it exclusively.  The lock is the NFS server's
# (/shared_nfs is mounted local_lock=none), so it spans nodes.  Each registry
# also records what its jobs used (<storage>/aic-gc/used), which is how the
# collector tells a cache a build just reused from one nobody has touched.

# aic_registry_start LOGTAG STORAGE
# Serve STORAGE from a registry container on this node.  Sets AIC_REG_NAME,
# AIC_REG_PORT and AIC_REG_STORAGE for aic_registry_stop / aic_image_store.
aic_registry_start() {
    local tag="$1" storage="$2"
    mkdir -p "${storage}" \
        || { echo "[${tag}] FATAL: cannot create registry storage ${storage}" >&2; return 1; }
    _aic_registry_lock "${tag}" "${storage}" s "${AIC_REGISTRY_LOCK_WAIT:-3600}" || return 1
    _aic_registry_run "${tag}" "${storage}" || { _aic_registry_unlock "${tag}"; return 1; }
}

# _aic_registry_lock LOGTAG STORAGE s|x WAIT_SECONDS
# Take STORAGE's lock shared (s) or exclusive (x), waiting up to WAIT_SECONDS.
# Sets AIC_REG_LOCK_FD; the lock is held until _aic_registry_unlock.
_aic_registry_lock() {
    local tag="$1" storage="$2" mode="$3" wait="$4" lock="${2%/}/aic-gc/lock" fd
    # Whoever can create it does; a user who can only read the storage still
    # needs it to exist, since a missing lock would let a collection run under
    # their pull.
    if [[ ! -e "${lock}" ]] && ! { mkdir -p "${lock%/*}" && : >> "${lock}"; }; then
        echo "[${tag}] FATAL: cannot create ${lock}; ask the owner of ${storage} to create it" >&2
        return 1
    fi
    # An exclusive lock over NFS needs a descriptor open for writing.
    if [[ "${mode}" == x ]]; then
        exec {fd}<>"${lock}" || { echo "[${tag}] FATAL: cannot open ${lock} for writing" >&2; return 1; }
    else
        exec {fd}<"${lock}" || { echo "[${tag}] FATAL: cannot open ${lock}" >&2; return 1; }
    fi
    # 75 = held by someone else; any other failure is the locking itself (no
    # working NFS lock service on this host), which waiting will not fix.
    local rc=0
    flock -n -E 75 "-${mode}" "${fd}" || rc=$?
    if (( rc == 75 )); then
        if [[ "${mode}" == x ]]; then
            echo "[${tag}] ${storage} is in use; waiting up to ${wait}s for its jobs to finish"
        else
            echo "[${tag}] ${storage} is being garbage-collected; waiting up to ${wait}s"
        fi
        rc=0
        flock -w "${wait}" -E 75 "-${mode}" "${fd}" || rc=$?
    fi
    if (( rc != 0 )); then
        if (( rc == 75 )); then
            echo "[${tag}] FATAL: no lock on ${lock} after ${wait}s" >&2
        else
            echo "[${tag}] FATAL: cannot lock ${lock} on $(hostname) (flock exit ${rc}); is NFS locking working here?" >&2
        fi
        exec {fd}<&-
        return 1
    fi
    AIC_REG_LOCK_FD="${fd}"
}

# _aic_registry_unlock LOGTAG
_aic_registry_unlock() {
    local fd="${AIC_REG_LOCK_FD:-}"
    [[ -n "${fd}" ]] || return 0
    flock -u "${fd}" || echo "[$1] WARN: could not release the registry storage lock" >&2
    exec {fd}<&-
    AIC_REG_LOCK_FD=""
}

# _aic_registry_run LOGTAG STORAGE [DOCKER_RUN_ARG...]
# Start the registry container; the caller holds the storage lock.
_aic_registry_run() {
    local tag="$1" storage="$2" image="${AIC_REGISTRY_IMAGE:-registry:3.1.2}" ports _
    shift 2
    AIC_REG_NAME="aic-registry-${SLURM_JOB_ID:-nojob}-$$-${RANDOM}"
    AIC_REG_STORAGE="${storage}"
    _aic_registry_image "${tag}" "${storage}" "${image}" || return 1
    docker run -d --name "${AIC_REG_NAME}" --user "$(id -u):$(id -g)" \
        -p 127.0.0.1::5000 -v "${storage}:/var/lib/registry" \
        -e REGISTRY_LOG_LEVEL=info "$@" "${image}" >/dev/null \
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
        _aic_registry_rm "${tag}" 1
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
# log when SHOW_LOG is 1, and release the storage.
aic_registry_stop() {
    if [[ -n "${AIC_REG_NAME:-}" ]]; then
        _aic_registry_stamp "$1"
        _aic_registry_rm "$@"
    fi
    _aic_registry_unlock "$1"
}

# _aic_registry_stamp LOGTAG
# Record, for aic_registry_gc, every manifest the running registry served or
# stored: one file per <repo>@<tag-or-digest> under <storage>/aic-gc/used, its
# mtime the last use.  The registry's access log is the only record of a read.
_aic_registry_stamp() {
    local tag="$1" dir="${AIC_REG_STORAGE%/}/aic-gc/used" used ref name n=0
    used="$(docker logs "${AIC_REG_NAME}" 2>/dev/null \
        | sed -nE 's#.*"(GET|HEAD|PUT) /v2/([a-z0-9._/-]+)/manifests/([A-Za-z0-9_.:-]+) HTTP/[0-9.]+" 20[01] .*#\2@\3#p' \
        | sort -u)" || {
        echo "[${tag}] WARN: could not read registry ${AIC_REG_NAME}'s log; what this job used is not recorded for garbage collection" >&2
        return 0
    }
    [[ -n "${used}" ]] || return 0
    mkdir -p "${dir}" || { echo "[${tag}] WARN: cannot create ${dir}; usage not recorded" >&2; return 0; }
    # The collector deletes nothing until usage has been recorded for its
    # whole retention window.
    [[ -e "${dir%/*}/stamping-since" ]] || : >> "${dir%/*}/stamping-since" \
        || echo "[${tag}] WARN: cannot create ${dir%/*}/stamping-since" >&2
    # Percent-encoded: a repo name holds '/', and SPUR's NFS server rejects ':'
    # (sha256:...) in a file name with EINVAL.
    while IFS= read -r ref; do
        name="${ref//\//%2F}"
        touch "${dir}/${name//:/%3A}" && n=$(( n + 1 )) \
            || echo "[${tag}] WARN: cannot record use of ${ref} in ${dir}" >&2
    done <<<"${used}"
    echo "[${tag}] recorded use of ${n} manifest(s) in ${dir}"
}

# _aic_registry_rm LOGTAG [SHOW_LOG]
_aic_registry_rm() {
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

# aic_registry_gc LOGTAG STORAGE RETAIN_DAYS DRY_RUN [REF_DIR...]
# Delete what no job has used in RETAIN_DAYS from STORAGE (see
# .slurm/registry-gc.py for what is kept), with the registry's own tools: the
# manifests through its API, then the blobs nothing references with
# `registry garbage-collect`.  DRY_RUN=1 only reports.  Holds the storage
# exclusively throughout, waiting up to AIC_CACHE_GC_LOCK_WAIT seconds for the
# jobs serving it to finish; jobs that start meanwhile wait for it.
aic_registry_gc() {
    local tag="$1" storage="$2" days="$3" dry="$4" d rc=0 out
    local image="${AIC_REGISTRY_IMAGE:-registry:3.1.2}"
    local py="${BASH_SOURCE[0]%/*}/registry-gc.py"
    local -a refs=()
    shift 4
    for d in "$@"; do [[ -d "${d}" ]] && refs+=(--ref-dir "${d}"); done
    if [[ ! -d "${storage%/}/docker/registry/v2" ]]; then
        echo "[${tag}] ${storage} holds no registry; nothing to collect"
        return 0
    fi
    # A dry run only reads, so it neither waits for jobs nor makes them wait.
    if [[ "${dry}" == 1 ]]; then
        echo "[${tag}] dry run over ${storage} (retain ${days} days)"
        python3 -I "${py}" "${storage}" --retain-days "${days}" "${refs[@]}"
        return
    fi
    # A subshell, so the EXIT trap is its own.  A container outlives a killed
    # job, so the trap removes the ones that change the storage before the
    # subshell's exit releases the lock.  Slurm sends SIGTERM seconds before
    # SIGKILL; only a bare SIGKILL skips it.
    (
        gc_name=""
        trap '[[ -z "${AIC_REG_NAME:-}" ]] || _aic_registry_rm "${tag}"
              [[ -z "${gc_name}" ]] || docker rm -f "${gc_name}" >/dev/null \
                  || echo "[${tag}] WARN: could not remove ${gc_name}" >&2' EXIT
        _aic_registry_lock "${tag}" "${storage}" x "${AIC_CACHE_GC_LOCK_WAIT:-3600}" || exit 1
        _aic_registry_gc_idle "${tag}" || exit 0
        echo "[${tag}] collecting ${storage} (retain ${days} days)"
        _aic_registry_run "${tag}" "${storage}" -e REGISTRY_STORAGE_DELETE_ENABLED=true || exit 1
        python3 -I "${py}" "${storage}" --retain-days "${days}" "${refs[@]}" \
            --registry "http://127.0.0.1:${AIC_REG_PORT}" || rc=$?
        _aic_registry_rm "${tag}" "$(( rc != 0 ))"
        (( rc == 0 )) || exit "${rc}"
        gc_name="aic-registry-gc-${SLURM_JOB_ID:-nojob}-$$-${RANDOM}"
        # Blob deletions are one line each; keep the summary.
        out="$(docker run --rm --name "${gc_name}" --user "$(id -u):$(id -g)" \
            -v "${storage}:/var/lib/registry" \
            "${image}" garbage-collect /etc/distribution/config.yml 2>&1)" || rc=$?
        gc_name=""
        if (( rc == 0 )); then
            grep -E 'blobs marked' <<<"${out}" | sed "s/^/[${tag}] /"
        else
            echo "[${tag}] ERROR: registry garbage-collect failed (exit ${rc}):" >&2
            tail -n 20 <<<"${out}" | sed 's/^/    /' >&2
        fi
        exit "${rc}"
    )
}

# _aic_registry_gc_idle LOGTAG
# Non-zero, after saying why, while one of the caller's Slurm jobs named like
# AIC_CACHE_GC_IDLE_JOBS (a glob; default aic-*) runs.  The storage lock already
# excludes every job whose image-dist-lib.sh takes it; this covers jobs from
# checkouts that predate the lock.  Empty AIC_CACHE_GC_IDLE_JOBS skips the check.
_aic_registry_gc_idle() {
    local tag="$1" pat="${AIC_CACHE_GC_IDLE_JOBS-aic-*}" q id name state busy=""
    [[ -n "${pat}" ]] || return 0
    command -v squeue >/dev/null 2>&1 || return 0
    q="$(squeue --me -h -o '%i %j %T')" \
        || { echo "[${tag}] squeue failed; not collecting, since a running job cannot be ruled out" >&2; return 1; }
    while read -r id name state; do
        [[ -n "${id}" && "${id}" != "${SLURM_JOB_ID:-}" ]] || continue
        [[ "${state}" == RUNNING || "${state}" == COMPLETING ]] || continue
        # shellcheck disable=SC2053  # pat is a glob
        [[ "${name}" == ${pat} ]] && busy+=" ${id}(${name})"
    done <<<"${q}"
    [[ -z "${busy}" ]] && return 0
    echo "[${tag}] skipped: running job(s)${busy} may use the storage without its lock"
    return 1
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
