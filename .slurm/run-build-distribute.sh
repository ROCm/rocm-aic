#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Build the aic-release inference image on an alola compile node and make it
# available across the cluster, using save -> shared BeeGFS scratch -> load.
#
# Image *distribution* uses save -> shared BeeGFS scratch -> load: each node keeps
# its images in local /docker/overlay2, so an image built on one node is invisible
# to the rest.  /scratch (BeeGFS) and /home (NFS) *are* shared on every node, so
# this script builds the image once on a CPU compile node, `docker save`s it to a
# tarball on /scratch, then `docker load`s it on each target node -- every node
# reading the one shared tarball, no per-node copy.
#
# Image *build cache* is optional and off by default.  Two backends:
#   * File-based (simplest): set AIC_CACHE_DIR to a dir on shared /scratch and the
#     build switches to `docker buildx` with a local cache -- every good layer is
#     written under that dir (in a per-arch subdir), so a later build on ANY node
#     reads it back and resumes from the step that failed instead of from scratch.
#     No registry, auth, or TLS needed; /scratch is shared on every node.
#   * Registry: set AIC_CACHE_REF to a registry ref instead (takes precedence over
#     AIC_CACHE_DIR).  Pushes/pulls layers to a registry; needs `docker login`.
# Either backend uses a docker-container buildx builder (the default `docker`
# driver cannot export type=local/registry cache); the script creates it on the
# build node on demand.
#
# The image bakes GPU arch(es) into hipFile and the LMCache HIP extension when it
# compiles them (NIXL AIS_MT is host-only, so arch-independent).  AIC_ROCM_ARCH is a
# ';'-separated list; by default it covers every gfx the vLLM ROCm wheel supports,
# so one image runs on any of them.  Narrow it (e.g. AIC_ROCM_ARCH=gfx942) for a
# faster, smaller single-arch build.
#
# Usage (run from the aic-release/ tree root; paths resolve relative to this script):
#
#   # Build on a compile node AND load onto MI300X targets in one go:
#   AIC_TARGETS=ctr-cx63-mi300x-3,ctr-cx64-mi300x-4 \
#     bash .slurm/run-build-distribute.sh all
#
#   # Just build + save the tarball:
#   bash .slurm/run-build-distribute.sh build
#
#   # Just load an already-saved tarball onto targets:
#   AIC_TARGETS=ctr-cx63-mi300x-3,ctr-cx64-mi300x-4 \
#     bash .slurm/run-build-distribute.sh load
#
#   # Push the built image to a registry (pull-based distribution):
#   AIC_PUSH_REF=registry-sc-harbor.amd.com/<proj>/rocm-aic:latest \
#     bash .slurm/run-build-distribute.sh push
#
# Commands:
#   build   Build the image on AIC_BUILD_NODE, then save the tarball to AIC_IMAGE_DIR
#   build-exporters
#           Build the fabric exporter images (nvme_exporter / rdma_exporter) from
#           monitoring/*/Dockerfile and save their tarballs to AIC_IMAGE_DIR, so
#           bare cliff nodes (no batesste host service) can containerize them.
#   load    Load the saved tarball on every node in AIC_TARGETS, then verify
#           (also loads the exporter tarballs when present)
#   push    Tag the built image as AIC_PUSH_REF and `docker push` it to a registry
#           (loads from the shared tarball first if the image is not present)
#   test    Smoke-test the image on a GPU+NVMe node (loads it there if missing):
#           checks GPU visibility + arch, vLLM / LMCache / hipFile, ais-check
#           (HIP+amdgpu AIS support), and the NIXL AIS_MT plugin (hard fail if
#           AIS_MT or ais-check fail)
#   all     build, build-exporters, then load   (default)
#
# Key environment:
#   AIC_ROCM_ARCH        gfx arch(es) baked in; ';'-list   (default: all vLLM archs)
#   AIC_IMAGE            image name:tag                    (default: rocm-aic:latest)
#   AIC_IMAGE_DIR        shared dir for the tarball        (default: /scratch/$USER/images)
#   AIC_FORCE_LOAD       test/push: force a reload from the tarball even when the
#                        node's image is already current (default: 0).  By default
#                        a node auto-reloads only when the /scratch tarball is
#                        newer than what it last loaded (tracked per node via a
#                        marker under /var/tmp), so a rebuild is picked up
#                        automatically without setting this.
#   AIC_TARGETS          comma-separated nodes to load     (required for load/all)
#   AIC_PUSH_REF         registry-qualified ref to push the final image to
#                        (required for push; needs `docker login <registry>` first)
#                        (e.g. registry-sc-harbor.amd.com/<proj>/rocm-aic:latest)
#
#   AIC_BUILD_CONSTRAINT Slurm -C feature expr for the build node
#                        (default: MARKHAM&CPUONLY -- CPU-only alola build boxes
#                         on the same Markham /scratch).
#                         Used only when AIC_BUILD_NODE is unset.
#   AIC_BUILD_NODE       pin an exact build node via --nodelist (overrides
#                        AIC_BUILD_CONSTRAINT)             (default: unset)
#   AIC_BUILD_LOCAL      set to 1 to build on THIS host, no Slurm  (default: unset)
#   AIC_BUILD_PARTITION  Slurm partition for build + load  (default: defq)
#   AIC_BUILD_CPUS       --cpus-per-task for the build job (default: 32)
#   AIC_BUILD_TIME       build job time limit              (default: 02:00:00)
#   AIC_LOAD_TIME        per-node load job time limit      (default: 00:30:00)
#
#   AIC_CACHE_DIR        base dir on shared /scratch for a file-based BuildKit
#                        cache; when set, the build uses `docker buildx` with a
#                        type=local cache under <dir>/<arch> so a failed build
#                        resumes from the last good layer on any node.  No registry
#                        or auth needed.  (default: unset -- plain `docker build`)
#   AIC_CACHE_REF        registry ref for a shared BuildKit cache instead of a dir;
#                        takes precedence over AIC_CACHE_DIR.  Uses --cache-to/
#                        --cache-from type=registry.  Requires `docker login` first.
#                        (e.g. registry-sc-harbor.amd.com/<proj>/rocm-aic:buildcache)
#                        (default: unset)
#   AIC_CACHE_MODE       cache mode: min | max              (default: max)
#   AIC_BUILDX_BUILDER   docker-container buildx builder name (default: aic-cache)
#   AIC_CACHE_INSECURE   set to 1 when AIC_CACHE_REF has an untrusted TLS cert
#                        (self-signed / private-CA HTTPS, e.g. the in-cluster
#                        Artifactory): the docker-container builder does NOT inherit
#                        the daemon's insecure-registries, so it is told to skip
#                        cert verification explicitly                (default: unset)
#
#   AIC_TEST_CONSTRAINT  Slurm -C feature expr for the test node
#                        (default: MARKHAM&GFX942&NVME -- MI300X + local NVMe).
#                        Used only when AIC_TEST_NODE is unset.
#   AIC_TEST_NODE        pin an exact test node via --nodelist  (default: unset)
#   AIC_TEST_TIME        test job time limit               (default: 00:20:00)
#   AIC_TEST_CPUS        --cpus-per-task for the test job  (default: 8)
#   AIC_TEST_MEM         --mem for the test job            (default: 32G)
#   AIC_SMOKE_EXPORTERS  test: 1 to also stand up the exporter fleet + Prometheus
#                        after the in-image checks, health-check each /metrics
#                        endpoint, and leave a TSDB under logs/<job-id>/prometheus
#                        (informational -- never changes the exit code); 0 to skip
#                        (default: 1)
#   AIC_SMOKE_SCRAPE_S   test: seconds to let Prometheus scrape before the health
#                        check / TSDB summary                       (default: 45)
#
#   AIC_SPUR_CLUSTER     set to 1 when running on a SPUR-based Slurm controller.
#                        SPUR's sbatch does not support --parsable, --wait, or
#                        --overcommit.  When set, _sbatch_run writes the job script to
#                        a temp file (SPUR requires a file, not stdin), submits without
#                        those flags, parses the job id from "Submitted batch job NNNN",
#                        and polls squeue until the job leaves the queue.  cmd_load and
#                        cmd_push also drop --overcommit from their srun calls.
#                        (default: 0)
#   AIC_SPUR_CONTROLLER  SPUR controller address passed as --controller to every
#                        sbatch/srun/squeue call when AIC_SPUR_CLUSTER=1.
#                        (default: $SPUR_CONTROLLER_ADDR if set, else
#                         http://crs-m2m-cpu-spur-005.crusoe.amd.com:6817)
#
#   AIC_TLS_CERT         corporate CA cert (BuildKit secret, never baked into image)
#                        (default: $HOME/certs/zscaler-ca.crt if it exists; else none)
#   AIC_COMPRESS         zstd | gzip | none                (default: zstd if available,
#                                                            else gzip)
#
set -euo pipefail

# --- Resolve paths (script lives at aic-release/.slurm/) ------------------
# The tree is self-contained: the Docker build context IS aic-release/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIC_DAY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Defaults ----------------------------------------------------------------
# Multi-arch by default: every gfx the vLLM ROCm wheel ships kernels for.
# hipFile + the LMCache HIP extension are compiled for all of these (cmake and
# PYTORCH_ROCM_ARCH both accept ';'-lists); NIXL AIS_MT is host-only, so arch-
# independent.  NOTE: the RDNA entries (gfx11xx/gfx12xx) have no NVMe-DMA /
# Infinity-Storage hardware -- if hipFile fails to build for them, narrow this
# to the CDNA set "gfx90a;gfx942;gfx950".  Override via AIC_ROCM_ARCH.
AIC_ROCM_ARCH="${AIC_ROCM_ARCH:-gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201}"
AIC_IMAGE="${AIC_IMAGE:-rocm-aic:latest}"
AIC_IMAGE_DIR="${AIC_IMAGE_DIR:-/scratch/${USER}/images}"
AIC_SPUR_CLUSTER="${AIC_SPUR_CLUSTER:-0}"
AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER:-${SPUR_CONTROLLER_ADDR:-http://crs-m2m-cpu-spur-005.crusoe.amd.com:6817}}"

# When running on SPUR, default the partition to amd-spur (the only partition)
# and clear the build/test constraints (SPUR nodes have no MARKHAM/CPUONLY/GFX942
# feature labels; node selection is done by partition or explicit --nodelist).
if [[ "${AIC_SPUR_CLUSTER}" == "1" ]]; then
    AIC_BUILD_PARTITION="${AIC_BUILD_PARTITION:-amd-spur}"
    # Use ${VAR-default} (not ${VAR:-default}) so an explicitly set empty string
    # ("AIC_BUILD_CONSTRAINT=") is honoured as "no constraint".
    AIC_BUILD_CONSTRAINT="${AIC_BUILD_CONSTRAINT-}"
    AIC_TEST_CONSTRAINT="${AIC_TEST_CONSTRAINT-}"
else
    AIC_BUILD_PARTITION="${AIC_BUILD_PARTITION:-defq}"
    AIC_BUILD_CONSTRAINT="${AIC_BUILD_CONSTRAINT:-MARKHAM&CPUONLY}"
    AIC_TEST_CONSTRAINT="${AIC_TEST_CONSTRAINT:-MARKHAM&GFX942&NVME}"
fi
AIC_BUILD_CPUS="${AIC_BUILD_CPUS:-32}"
AIC_BUILD_TIME="${AIC_BUILD_TIME:-02:00:00}"
AIC_LOAD_TIME="${AIC_LOAD_TIME:-00:30:00}"
AIC_TARGETS="${AIC_TARGETS:-}"
AIC_PUSH_REF="${AIC_PUSH_REF:-}"
AIC_CACHE_DIR="${AIC_CACHE_DIR:-}"
AIC_CACHE_REF="${AIC_CACHE_REF:-}"
AIC_CACHE_MODE="${AIC_CACHE_MODE:-max}"
AIC_BUILDX_BUILDER="${AIC_BUILDX_BUILDER:-aic-cache}"
AIC_CACHE_INSECURE="${AIC_CACHE_INSECURE:-}"
AIC_TEST_TIME="${AIC_TEST_TIME:-00:20:00}"
AIC_TEST_CPUS="${AIC_TEST_CPUS:-8}"
AIC_TEST_MEM="${AIC_TEST_MEM:-32G}"

# --- Fabric exporter images (nvme_exporter / rdma_exporter) -------------------
# Built from monitoring/*/Dockerfile and distributed alongside the main image so
# bare cliff nodes (no batesste host service) can containerize them.  Names must
# match run-cliff.sbatch's defaults so the tarballs written here are found there.
# Versions default to the batesste host-service versions for Grafana parity.
AIC_NVME_EXPORTER_IMAGE="${AIC_NVME_EXPORTER_IMAGE:-aic-nvme-exporter:latest}"
AIC_RDMA_EXPORTER_IMAGE="${AIC_RDMA_EXPORTER_IMAGE:-aic-rdma-exporter:latest}"
AIC_NVME_EXPORTER_VERSION="${AIC_NVME_EXPORTER_VERSION:-3.0.0}"
AIC_RDMA_EXPORTER_VERSION="${AIC_RDMA_EXPORTER_VERSION:-0.3.0}"

# Corporate CA: default to the conventional path only if it actually exists.
if [[ -z "${AIC_TLS_CERT:-}" && -r "${HOME}/certs/zscaler-ca.crt" ]]; then
    AIC_TLS_CERT="${HOME}/certs/zscaler-ca.crt"
fi
AIC_TLS_CERT="${AIC_TLS_CERT:-}"

log()  { printf '[build-distribute] %s\n' "$*" >&2; }
die()  { printf '[build-distribute] ERROR: %s\n' "$*" >&2; exit 1; }

# --- Compression: pick tool + file extension --------------------------------
_pick_compress() {
    local choice="${AIC_COMPRESS:-}"
    if [[ -z "${choice}" ]]; then
        if command -v zstd >/dev/null 2>&1; then choice=zstd; else choice=gzip; fi
    fi
    case "${choice}" in
        zstd) COMPRESS_EXT="tar.zst"; COMPRESS_CMD="zstd -T0 -3 -q"; DECOMPRESS_CMD="zstd -dc" ;;
        gzip) COMPRESS_EXT="tar.gz";  COMPRESS_CMD="gzip";           DECOMPRESS_CMD="gzip -dc" ;;
        none) COMPRESS_EXT="tar";     COMPRESS_CMD="cat";            DECOMPRESS_CMD="cat" ;;
        *)    die "AIC_COMPRESS must be zstd, gzip, or none (got '${choice}')" ;;
    esac
    AIC_COMPRESS="${choice}"
}

# --- Filesystem-safe tag for the arch value --------------------------------
# AIC_ROCM_ARCH may be a ';'-separated multi-arch list, which is not a valid
# filename/path component; map separators to '-' for use in the tarball name
# and the per-arch cache dir (e.g. "gfx90a;gfx942" -> "gfx90a-gfx942").
_arch_tag() {
    printf '%s' "${AIC_ROCM_ARCH}" | tr ';,: ' '----' | tr -s '-' | sed 's/^-//;s/-$//'
}

# --- Tarball path (name:tag + arch, sanitized for a filename) ----------------
_tarball_path() {
    local base
    base="$(printf '%s' "${AIC_IMAGE}" | tr '/:' '--')"
    printf '%s/%s-%s.%s' "${AIC_IMAGE_DIR}" "${base}" "$(_arch_tag)" "${COMPRESS_EXT}"
}

# --- Exporter tarball path ($1=image name:tag) -------------------------------
# The exporters bake a host-CPU-arch binary (not a gfx arch), so unlike the main
# image their tarball name carries no arch tag -- just the sanitized name:tag.
_exporter_tarball_path() {
    local base
    base="$(printf '%s' "$1" | tr '/:' '--')"
    printf '%s/%s.%s' "${AIC_IMAGE_DIR}" "${base}" "${COMPRESS_EXT}"
}

# --- sbatch dispatch (mirrors run-cliff.sbatch's per-job logging) -------------
# Submit BODY as an sbatch batch job whose output streams into
# logs/<job-id>/<logname>.out under the tree -- the SAME per-job structure
# run-cliff.sbatch uses for logs/<job-id>/cliff.out.  Unlike `make cliff-submit`
# (fire-and-forget), this BLOCKS until the job finishes and returns its exit
# code, so the chained goals `make dist-build dist-push smoke-test` still run in
# order and stop on failure.  The job's log is live-tailed while it runs so
# `make dist-build` / `make smoke-test` still show progress in the terminal.
#
#   $1    = job name       (e.g. aic-build)
#   $2    = log basename   (build -> logs/<job-id>/build.out)
#   $3    = body script    (the work; runs on the compute node)
#   $4..  = extra sbatch options (node selection, cpus, mem, gres, time, ...)
_sbatch_run() {
    local jobname="$1" logname="$2" body="$3"; shift 3
    command -v sbatch >/dev/null 2>&1 || die "sbatch not found; set AIC_BUILD_LOCAL=1 to build here"

    # Batch script = shebang + a prologue that creates the per-job log dir and
    # redirects everything into <logname>.out, then the caller's body.
    # AIC_DAY_DIR is absolute and on shared storage, so it resolves on the
    # compute node without relying on SLURM_SUBMIT_DIR.  --output=/dev/null
    # discards any pre-redirect output (there is none here).
    local script
    script="$(cat <<PROLOGUE
#!/bin/bash
_logdir="${AIC_DAY_DIR}/logs/\${SLURM_JOB_ID:-manual}"
mkdir -p "\${_logdir}" 2>/dev/null && exec >>"\${_logdir}/${logname}.out" 2>&1
PROLOGUE
)"
    script+=$'\n'"${body}"

    local jobid="" logfile="" rc=0

    if [[ "${AIC_SPUR_CLUSTER}" == "1" ]]; then
        # SPUR sbatch does not support --parsable, --wait, or reading the script
        # from stdin.  Write the script to a temp file, submit without those
        # flags, parse "Submitted batch job NNNN" from stdout, then poll squeue
        # until the job is no longer in the queue.
        local tmpscript; tmpscript="$(mktemp --suffix=.sh)"
        printf '%s\n' "${script}" > "${tmpscript}"
        chmod +x "${tmpscript}"

        local submit_out
        submit_out="$(sbatch \
            --controller="${AIC_SPUR_CONTROLLER}" \
            --job-name="${jobname}" \
            --partition="${AIC_BUILD_PARTITION}" \
            --output=/dev/null \
            "$@" \
            "${tmpscript}" 2>&1)" || { rm -f "${tmpscript}"; die "sbatch submission failed: ${submit_out}"; }
        rm -f "${tmpscript}"

        jobid="$(printf '%s\n' "${submit_out}" | grep -oE '[0-9]+$' | tail -1)"
        [[ -n "${jobid}" ]] || die "could not parse job id from sbatch output: ${submit_out}"
        logfile="${AIC_DAY_DIR}/logs/${jobid}/${logname}.out"
        log "submitted ${jobname} as job ${jobid} (partition ${AIC_BUILD_PARTITION})"
        log "log: ${logfile}"

        # Poll squeue until the job leaves the queue, printing new log lines each
        # iteration. Avoids tail -F which hangs on SPUR when the background
        # subshell cannot be reliably killed inside an SSH heredoc.
        # SPUR squeue ignores -j; filter by job ID in awk.
        local last_line=0

        _print_new_lines() {
            if [[ -f "${logfile}" ]]; then
                local total; total=$(wc -l < "${logfile}" 2>/dev/null || echo 0)
                if (( total > last_line )); then
                    tail -n +"$((last_line + 1))" "${logfile}" 2>/dev/null
                    last_line=${total}
                fi
            fi
        }

        # Wait up to 60s for the job to appear.
        local appear_tries=0
        until squeue --controller="${AIC_SPUR_CONTROLLER}" -j "${jobid}" -h 2>/dev/null | awk '{print $1}' | grep -qx "${jobid}" \
              || (( appear_tries >= 60 )); do
            sleep 1; appear_tries=$((appear_tries + 1))
        done

        # Poll until the job leaves the queue, streaming new log lines.
        while squeue --controller="${AIC_SPUR_CONTROLLER}" -j "${jobid}" -h 2>/dev/null | awk '{print $1}' | grep -qx "${jobid}"; do
            _print_new_lines
            sleep 10
        done

        # Flush any remaining lines after job completes.
        sleep 2
        _print_new_lines

        # Read the real exit code from sacct ("<code>:<signal>" format).
        # SPUR sacct ignores -j and returns all jobs; grep for the exact job ID
        # in JobID+ExitCode output to avoid picking up an unrelated row.
        local acct_exit
        acct_exit="$(sacct --controller="${AIC_SPUR_CONTROLLER}" -j "${jobid}" \
            --format=JobID,ExitCode --noheader 2>/dev/null \
            | awk -v id="${jobid}" '$1==id{split($2,a,":"); print a[1]; exit}')"
        rc="${acct_exit:-1}"
    else
        # Standard Slurm path: --parsable prints the bare job id; --wait blocks
        # until the job finishes and exits with the job's exit code.
        local idfile; idfile="$(mktemp)"
        local -a _stdbuf=(); command -v stdbuf >/dev/null 2>&1 && _stdbuf=(stdbuf -oL)
        "${_stdbuf[@]}" sbatch --parsable --wait \
            --job-name="${jobname}" \
            --partition="${AIC_BUILD_PARTITION}" \
            --output=/dev/null \
            "$@" \
            <<<"${script}" >"${idfile}" &
        local sb_pid=$!

        # Wait briefly for the parsable job id to land in the temp file.
        local tries=0
        while [[ ! -s "${idfile}" ]] && kill -0 "${sb_pid}" 2>/dev/null && (( tries < 150 )); do
            sleep 0.2; tries=$((tries + 1))
        done
        jobid="$(head -n1 "${idfile}" 2>/dev/null | tr -d '[:space:]' | cut -d';' -f1)"

        logfile="${AIC_DAY_DIR}/logs/${jobid:-unknown}/${logname}.out"
        if [[ -n "${jobid}" ]]; then
            log "submitted ${jobname} as job ${jobid} (partition ${AIC_BUILD_PARTITION})"
            log "log: ${logfile}"
        else
            log "submitted ${jobname} (job id not yet available; partition ${AIC_BUILD_PARTITION})"
        fi

        local tail_pid=""
        if [[ -n "${jobid}" ]]; then
            ( tail -F "${logfile}" 2>/dev/null ) & tail_pid=$!
        fi

        wait "${sb_pid}" || rc=$?
        if [[ -n "${tail_pid}" ]]; then
            sleep 1
            kill "${tail_pid}" >/dev/null 2>&1 || true
            wait "${tail_pid}" 2>/dev/null || true
        fi
        rm -f "${idfile}" 2>/dev/null || true
    fi

    return "${rc}"
}

# --- build: build the image on a compile node, save tarball to shared scratch -
cmd_build() {
    _pick_compress
    local tarball; tarball="$(_tarball_path)"

    log "image      : ${AIC_IMAGE}  (arch ${AIC_ROCM_ARCH})"
    log "tarball    : ${tarball}  (compress: ${AIC_COMPRESS})"
    if [[ -n "${AIC_TLS_CERT}" ]]; then
        [[ -r "${AIC_TLS_CERT}" ]] || die "AIC_TLS_CERT not readable: ${AIC_TLS_CERT}"
        log "tls cert   : ${AIC_TLS_CERT} (BuildKit secret)"
    fi

    # BuildKit secret arg for the corporate CA, only when a cert was provided.
    local _secret_arg=""
    [[ -n "${AIC_TLS_CERT}" ]] && _secret_arg="--secret id=tls_cert,src=${AIC_TLS_CERT}"

    # --- Build program: plain `docker build`, or `docker buildx` with a shared
    #     registry cache when AIC_CACHE_REF is set.  The registry cache pushes each
    #     good layer as it builds, so a failed build resumes from the last good
    #     layer on ANY node instead of restarting from scratch.  It needs the
    #     docker-container buildx driver (the default `docker` driver cannot export
    #     type=registry cache); the driver is created on the build node on demand.
    #     In the buildx path we do NOT `--load` + `docker save`; instead we export
    #     the tarball straight from BuildKit with `--output type=docker,dest=-`
    #     (see the build+save block below).  `docker save` on a large BuildKit
    #     image (34GB+) deadlocks -- the build finishes but `docker save` hangs
    #     forever writing 0 bytes -- so exporting from BuildKit sidesteps it.
    #     `_build_program` (plain `docker build`) is used only by the no-cache path.
    local _build_program="DOCKER_BUILDKIT=1 docker build"
    local _cache_args=""
    local _builder_setup=""
    if [[ -n "${AIC_CACHE_REF}" || -n "${AIC_CACHE_DIR}" ]]; then
        case "${AIC_CACHE_MODE}" in
            min|max) ;;
            *) die "AIC_CACHE_MODE must be min or max (got '${AIC_CACHE_MODE}')" ;;
        esac
        # _pre / _mkdir run before the build; _cfg_arg tweaks builder creation.
        local _cfg_arg="" _pre="" _mkdir=""
        if [[ -n "${AIC_CACHE_REF}" ]]; then
            # Registry backend (takes precedence over AIC_CACHE_DIR).
            log "build cache: registry ${AIC_CACHE_REF} (mode ${AIC_CACHE_MODE}, builder ${AIC_BUILDX_BUILDER})"
            _cache_args="--cache-from type=registry,ref=${AIC_CACHE_REF} --cache-to type=registry,ref=${AIC_CACHE_REF},mode=${AIC_CACHE_MODE}"
            # Optional buildkitd config for a cache registry with an untrusted TLS
            # cert (self-signed / private-CA HTTPS).  The docker-container builder
            # does NOT inherit /etc/docker/daemon.json's insecure-registries, so it
            # must be told to skip cert verification.  `insecure = true` keeps HTTPS
            # but skips verification; we do NOT set `http = true` since these
            # registries speak HTTPS (plain HTTP gets a 400).
            if [[ "${AIC_CACHE_INSECURE}" == "1" ]]; then
                local _cache_host="${AIC_CACHE_REF%%/*}"
                _cfg_arg=" --config /tmp/buildkitd-aic.toml"
                _pre="printf '[registry.\"%s\"]\n  insecure = true\n' '${_cache_host}' > /tmp/buildkitd-aic.toml; "
                log "build cache: skipping TLS verification for ${_cache_host} (AIC_CACHE_INSECURE=1)"
            fi
        else
            # File-based backend: type=local cache under <dir>/<arch> on shared
            # /scratch.  buildx reads/writes these paths from the sbatch job (which
            # has /scratch mounted), so no registry or auth is involved.
            local _cdir
            _cdir="${AIC_CACHE_DIR%/}/$(_arch_tag)"
            log "build cache: local dir ${_cdir} (mode ${AIC_CACHE_MODE}, builder ${AIC_BUILDX_BUILDER})"
            _cache_args="--cache-from type=local,src=${_cdir} --cache-to type=local,dest=${_cdir},mode=${AIC_CACHE_MODE}"
            _mkdir="mkdir -p '${_cdir}'; "
        fi
        # Create the docker-container builder once per node (idempotent), then
        # bootstrap it so its BuildKit is ready before the build starts.
        _builder_setup="${_pre}${_mkdir}if ! docker buildx inspect ${AIC_BUILDX_BUILDER} >/dev/null 2>&1; then echo '[build] creating buildx builder ${AIC_BUILDX_BUILDER} (docker-container)'; docker buildx create --name ${AIC_BUILDX_BUILDER} --driver docker-container${_cfg_arg} >/dev/null; fi; docker buildx inspect --bootstrap ${AIC_BUILDX_BUILDER} >/dev/null"
    fi

    # The build + save block runs on ONE node so the saved tarball comes from the
    # image that was just built.  Values are baked in here (not passed via env)
    # to keep it robust regardless of sbatch environment propagation.
    #
    # Build with plain `docker build` (not `make build`) so the only requirement
    # on the build node is docker itself -- the compose plugin is not installed
    # on every CPU node and is only needed to *run* the stack, not to build it.
    local remote_script
    if [[ -n "${AIC_CACHE_REF}" || -n "${AIC_CACHE_DIR}" ]]; then
        # Cache/buildx path: stream a docker-format tar straight from the
        # docker-container builder into the compressor with
        # `--output type=docker,dest=-`.  This deliberately avoids `--load` +
        # `docker save`, whose daemon export path deadlocks on large BuildKit
        # images (the build finishes but `docker save` hangs forever at 0 bytes).
        # `set -o pipefail` (from set -euo pipefail) makes a build failure fail
        # the whole pipeline instead of writing a truncated tarball.
        remote_script="$(cat <<REMOTE
set -euo pipefail
command -v docker >/dev/null 2>&1 || { echo 'docker not found on build node' >&2; exit 1; }
echo "[build] host=\$(hostname) docker=\$(docker --version)"
cd "${AIC_DAY_DIR}"
${_builder_setup}
mkdir -p "${AIC_IMAGE_DIR}"
tmp="${tarball}.partial.\$\$"
docker buildx build --builder ${AIC_BUILDX_BUILDER} --progress=plain --output type=docker,dest=- \
    --build-arg ROCM_ARCH="${AIC_ROCM_ARCH}" \
    ${_secret_arg} \
    ${_cache_args} \
    -f "${AIC_DAY_DIR}/docker/Dockerfile" \
    -t "${AIC_IMAGE}" \
    "${AIC_DAY_DIR}" | ${COMPRESS_CMD} > "\${tmp}"
mv -f "\${tmp}" "${tarball}"
echo "[build] saved \$(du -h "${tarball}" | cut -f1) -> ${tarball}"
REMOTE
)"
    else
        # No-cache path: plain `docker build` into the local daemon, then
        # `docker save` the result.  Fine for smaller/simpler builds.
        remote_script="$(cat <<REMOTE
set -euo pipefail
command -v docker >/dev/null 2>&1 || { echo 'docker not found on build node' >&2; exit 1; }
echo "[build] host=\$(hostname) docker=\$(docker --version)"
cd "${AIC_DAY_DIR}"
${_builder_setup}
${_build_program} \
    --build-arg ROCM_ARCH="${AIC_ROCM_ARCH}" \
    ${_secret_arg} \
    ${_cache_args} \
    -f "${AIC_DAY_DIR}/docker/Dockerfile" \
    -t "${AIC_IMAGE}" \
    "${AIC_DAY_DIR}"
echo "[build] built ${AIC_IMAGE}"
mkdir -p "${AIC_IMAGE_DIR}"
tmp="${tarball}.partial.\$\$"
docker save "${AIC_IMAGE}" | ${COMPRESS_CMD} > "\${tmp}"
mv -f "\${tmp}" "${tarball}"
echo "[build] saved \$(du -h "${tarball}" | cut -f1) -> ${tarball}"
REMOTE
)"
    fi

    if [[ "${AIC_BUILD_LOCAL:-}" == "1" ]]; then
        log "building locally on $(hostname) (AIC_BUILD_LOCAL=1)"
        bash -c "${remote_script}"
    else
        # Pin an exact node if AIC_BUILD_NODE is set; otherwise let Slurm choose
        # any idle node matching the CPU-only build constraint.
        local -a _sel
        if [[ -n "${AIC_BUILD_NODE:-}" ]]; then
            _sel=(--nodelist="${AIC_BUILD_NODE}")
            log "building on ${AIC_BUILD_NODE} via sbatch (partition ${AIC_BUILD_PARTITION})"
        else
            if [[ -n "${AIC_BUILD_CONSTRAINT:-}" ]]; then
                _sel=(--constraint="${AIC_BUILD_CONSTRAINT}")
            fi
            log "building via sbatch (partition ${AIC_BUILD_PARTITION}, constraint ${AIC_BUILD_CONSTRAINT})"
        fi
        _sbatch_run aic-build build "${remote_script}" \
            "${_sel[@]}" \
            --nodes=1 --ntasks=1 \
            --cpus-per-task="${AIC_BUILD_CPUS}" \
            --time="${AIC_BUILD_TIME}"
    fi
    log "build complete: ${tarball}"
}

# --- build-exporters: build the fabric exporter images, save tarballs ---------
# The nvme_exporter / rdma_exporter images are small (Debian slim + a prebuilt
# release binary) and gfx-arch-independent.  Like cmd_build, we do NOT `docker
# save`: on a node whose default builder is the docker-container driver (the
# `aic-cache` builder this script creates), `docker build` builds inside BuildKit
# and `docker save` can miss the layer blobs -- yielding a truncated tarball
# (observed: a 1.5K "image").  Instead we export a docker-format tar straight
# from BuildKit with `--output type=docker,dest=-` piped into the compressor,
# which captures the full image regardless of the node's default builder/driver.
# Runs on a build-class node (or locally with AIC_BUILD_LOCAL=1); needs docker +
# reachability to GitHub/Debian.
cmd_build_exporters() {
    _pick_compress
    local nvme_tar rdma_tar
    nvme_tar="$(_exporter_tarball_path "${AIC_NVME_EXPORTER_IMAGE}")"
    rdma_tar="$(_exporter_tarball_path "${AIC_RDMA_EXPORTER_IMAGE}")"

    log "exporter images: ${AIC_NVME_EXPORTER_IMAGE} (nvme v${AIC_NVME_EXPORTER_VERSION}), ${AIC_RDMA_EXPORTER_IMAGE} (rdma v${AIC_RDMA_EXPORTER_VERSION})"
    log "tarballs   : ${nvme_tar}, ${rdma_tar}  (compress: ${AIC_COMPRESS})"

    local remote_script
    remote_script="$(cat <<REMOTE
set -euo pipefail
command -v docker >/dev/null 2>&1 || { echo 'docker not found on build node' >&2; exit 1; }
echo "[build-exporters] host=\$(hostname) docker=\$(docker --version)"
mkdir -p "${AIC_IMAGE_DIR}"
# A docker-container builder is required to stream a docker-format tar from
# BuildKit (the default 'docker' driver cannot export type=docker to stdout).
# Reuse/create the same builder cmd_build uses; idempotent, then bootstrap it.
if ! docker buildx inspect ${AIC_BUILDX_BUILDER} >/dev/null 2>&1; then
    echo "[build-exporters] creating buildx builder ${AIC_BUILDX_BUILDER} (docker-container)"
    docker buildx create --name ${AIC_BUILDX_BUILDER} --driver docker-container >/dev/null
fi
docker buildx inspect --bootstrap ${AIC_BUILDX_BUILDER} >/dev/null
tmp="${nvme_tar}.partial.\$\$"
docker buildx build --builder ${AIC_BUILDX_BUILDER} --output type=docker,dest=- \
    --build-arg NVME_EXPORTER_VERSION="${AIC_NVME_EXPORTER_VERSION}" \
    -t "${AIC_NVME_EXPORTER_IMAGE}" "${AIC_DAY_DIR}/monitoring/nvme-exporter" | ${COMPRESS_CMD} > "\${tmp}"
mv -f "\${tmp}" "${nvme_tar}"
tmp="${rdma_tar}.partial.\$\$"
docker buildx build --builder ${AIC_BUILDX_BUILDER} --output type=docker,dest=- \
    --build-arg RDMA_EXPORTER_VERSION="${AIC_RDMA_EXPORTER_VERSION}" \
    -t "${AIC_RDMA_EXPORTER_IMAGE}" "${AIC_DAY_DIR}/monitoring/rdma-exporter" | ${COMPRESS_CMD} > "\${tmp}"
mv -f "\${tmp}" "${rdma_tar}"
echo "[build-exporters] saved \$(du -h "${nvme_tar}" | cut -f1) -> ${nvme_tar}"
echo "[build-exporters] saved \$(du -h "${rdma_tar}" | cut -f1) -> ${rdma_tar}"
REMOTE
)"

    if [[ "${AIC_BUILD_LOCAL:-}" == "1" ]]; then
        log "building exporters locally on $(hostname) (AIC_BUILD_LOCAL=1)"
        bash -c "${remote_script}"
    else
        local -a _sel
        if [[ -n "${AIC_BUILD_NODE:-}" ]]; then
            _sel=(--nodelist="${AIC_BUILD_NODE}")
            log "building exporters on ${AIC_BUILD_NODE} via sbatch (partition ${AIC_BUILD_PARTITION})"
        else
            if [[ -n "${AIC_BUILD_CONSTRAINT:-}" ]]; then
                _sel=(--constraint="${AIC_BUILD_CONSTRAINT}")
            fi
            log "building exporters via sbatch (partition ${AIC_BUILD_PARTITION}, constraint ${AIC_BUILD_CONSTRAINT})"
        fi
        local -a _exp_overcommit=(); [[ "${AIC_SPUR_CLUSTER}" != "1" ]] && _exp_overcommit=(--overcommit)
        _sbatch_run aic-build-exporters build-exporters "${remote_script}" \
            "${_sel[@]}" \
            --nodes=1 --ntasks=1 \
            --cpus-per-task=2 --mem=8G "${_exp_overcommit[@]}" \
            --time="${AIC_LOAD_TIME}"
    fi
    log "exporter build complete: ${nvme_tar}, ${rdma_tar}"
}

# --- load: docker load the tarball on every target node, then verify ---------
# NOTE: load stays on `srun` (not the sbatch/_sbatch_run path used by build/test):
# it is a MULTI-NODE fan-out (--nodelist=<N nodes> --ntasks-per-node=1 runs the
# docker load on every target at once), whereas an sbatch batch script runs on
# only the first allocated node.  push likewise stays on srun (a quick single
# registry op).  build/build-exporters/test are the single-node "do work + log it"
# jobs that map cleanly onto run-cliff.sbatch's per-job logs/<job-id>/ structure.
cmd_load() {
    _pick_compress
    local tarball; tarball="$(_tarball_path)"

    [[ -n "${AIC_TARGETS}" ]] || die "set AIC_TARGETS=node1,node2,... for the load step"
    command -v srun >/dev/null 2>&1 || die "srun not found; cannot load onto remote nodes"
    [[ -r "${tarball}" ]] || die "tarball not found: ${tarball} (run 'build' first)"

    # Also ship the fabric exporter tarballs when they exist (built by
    # 'build-exporters').  Best-effort: absent tarballs are simply skipped, so a
    # main-image-only build still loads fine.  Paths carry no spaces (/scratch...).
    local tarballs="${tarball}" et
    for et in "$(_exporter_tarball_path "${AIC_NVME_EXPORTER_IMAGE}")" \
              "$(_exporter_tarball_path "${AIC_RDMA_EXPORTER_IMAGE}")"; do
        [[ -r "${et}" ]] && { tarballs+=" ${et}"; log "  + exporter tarball: ${et}"; }
    done

    local n; n="$(awk -F, '{print NF}' <<<"${AIC_TARGETS}")"
    log "loading ${AIC_IMAGE} (+ present exporter images) onto ${n} node(s): ${AIC_TARGETS}"
    log "tarball: ${tarball}"

    # Small, oversubscribable request so the load can slip in alongside running
    # GPU jobs -- docker load needs no GPU.  --overcommit is dropped on SPUR
    # (unsupported); harmless on standard Slurm.
    local -a _overcommit_arg=(); [[ "${AIC_SPUR_CLUSTER}" != "1" ]] && _overcommit_arg=(--overcommit)
    local -a _spur_ctl_arg=(); [[ "${AIC_SPUR_CLUSTER}" == "1" ]] && _spur_ctl_arg=(--controller="${AIC_SPUR_CONTROLLER}")
    srun \
        "${_spur_ctl_arg[@]}" \
        --job-name=aic-load \
        --partition="${AIC_BUILD_PARTITION}" \
        --nodelist="${AIC_TARGETS}" \
        --nodes="${n}" --ntasks-per-node=1 \
        --cpus-per-task=2 --mem=8G "${_overcommit_arg[@]}" \
        --time="${AIC_LOAD_TIME}" \
        bash -c "
set -euo pipefail
command -v docker >/dev/null 2>&1 || { echo \"\$(hostname): docker not found\" >&2; exit 1; }
for _tb in ${tarballs}; do
    ${DECOMPRESS_CMD} \"\${_tb}\" | docker load >/dev/null && echo \"\$(hostname): loaded \${_tb}\"
done
"
    log "load complete on: ${AIC_TARGETS}"
}

# --- push: tag the built image as AIC_PUSH_REF and push it to a registry ------
# Registry-based (pull) distribution as an alternative to the save->scratch->load
# tarball flow.  Runs on a build-class node; if that node does not already have
# the image locally (e.g. Slurm placed this job on a different node than build),
# it loads it from the shared tarball first, then tags and pushes.  Registry
# creds come from ~/.docker/config.json on shared /home; `docker login` once.
cmd_push() {
    _pick_compress
    local tarball; tarball="$(_tarball_path)"

    [[ -n "${AIC_PUSH_REF}" ]] || die "set AIC_PUSH_REF=registry/host/path:tag for the push step"
    command -v srun >/dev/null 2>&1 || die "srun not found; cannot run the push job"
    [[ -r "${tarball}" ]] || die "tarball not found: ${tarball} (run 'build' first)"

    log "pushing ${AIC_IMAGE} -> ${AIC_PUSH_REF}"
    log "tarball (load fallback): ${tarball}"

    local remote_script
    remote_script="$(cat <<REMOTE
set -euo pipefail
command -v docker >/dev/null 2>&1 || { echo "\$(hostname): docker not found" >&2; exit 1; }
echo "[push] host=\$(hostname) docker=\$(docker --version)"
# Load the image from the shared tarball only when needed (see cmd_test for the
# marker/mtime rationale): reload when the tarball is newer than the last load
# here, when the image is absent, or when forced.
_marker="/var/tmp/aic-loaded-\$(id -u)-\$(echo '${AIC_IMAGE}' | tr '/:' '__').mtime"
_tar_mtime="\$(stat -c %Y '${tarball}' 2>/dev/null || echo 0)"
_have_img="\$(docker images -q '${AIC_IMAGE}')"
_loaded_mtime="\$(cat "\${_marker}" 2>/dev/null || echo 0)"
if [ "${AIC_FORCE_LOAD:-0}" = "1" ] || [ -z "\${_have_img}" ] || [ "\${_tar_mtime}" -gt "\${_loaded_mtime}" ]; then
    echo "[push] loading ${AIC_IMAGE} from ${tarball} (tarball=\${_tar_mtime} last-loaded=\${_loaded_mtime} present=\$([ -n "\${_have_img}" ] && echo yes || echo no) force=${AIC_FORCE_LOAD:-0})"
    ${DECOMPRESS_CMD} '${tarball}' | docker load >/dev/null
    echo "\${_tar_mtime}" > "\${_marker}" 2>/dev/null || true
else
    echo "[push] image up to date on \$(hostname) (id \${_have_img}); AIC_FORCE_LOAD=1 forces a reload"
fi
docker tag '${AIC_IMAGE}' '${AIC_PUSH_REF}'
docker push '${AIC_PUSH_REF}'
echo "[push] pushed ${AIC_PUSH_REF}"
REMOTE
)"

    # Reuse the build-node selection: push needs no GPU, just docker + the creds
    # on shared /home.  Small, oversubscribable request so it can slip in.
    local -a _sel
    if [[ -n "${AIC_BUILD_NODE:-}" ]]; then
        _sel=(--nodelist="${AIC_BUILD_NODE}")
        log "pushing from ${AIC_BUILD_NODE} via srun (partition ${AIC_BUILD_PARTITION})"
    else
        if [[ -n "${AIC_BUILD_CONSTRAINT:-}" ]]; then
            _sel=(--constraint="${AIC_BUILD_CONSTRAINT}")
        fi
        log "pushing via srun (partition ${AIC_BUILD_PARTITION}, constraint ${AIC_BUILD_CONSTRAINT})"
    fi
    local -a _push_overcommit=(); [[ "${AIC_SPUR_CLUSTER}" != "1" ]] && _push_overcommit=(--overcommit)
    local -a _push_spur_ctl=(); [[ "${AIC_SPUR_CLUSTER}" == "1" ]] && _push_spur_ctl=(--controller="${AIC_SPUR_CONTROLLER}")
    srun \
        "${_push_spur_ctl[@]}" \
        --job-name=aic-push \
        --partition="${AIC_BUILD_PARTITION}" \
        "${_sel[@]}" \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=2 --mem=8G "${_push_overcommit[@]}" \
        --time="${AIC_LOAD_TIME}" \
        bash -c "${remote_script}"
    log "push complete: ${AIC_PUSH_REF}"
}

# --- test: smoke-test the image on a GPU+NVMe node ---------------------------
# Loads the image on the node if absent, then runs a container that verifies GPU
# visibility/arch and that the key stack components import/resolve, that AIS is
# usable (ais-check: HIP runtime + amdgpu driver), and that the NIXL AIS_MT
# plugin is present.  No HF token or model download -- this validates the *image*
# (and the node's AIS runtime support), not an end-to-end serve.
cmd_test() {
    _pick_compress
    local tarball; tarball="$(_tarball_path)"
    command -v sbatch >/dev/null 2>&1 || die "sbatch not found; cannot run the GPU test job"
    [[ -r "${tarball}" ]] || die "tarball not found: ${tarball} (run 'build' first)"

    # After the in-image checks, optionally stand up the exporter fleet +
    # Prometheus (via monitoring/monitoring-lib.sh, shared with the cliff),
    # scrape briefly, health-check each /metrics endpoint, and leave a TSDB under
    # logs/<job-id>/prometheus to sanity-check.  Informational only -- these never
    # change the smoke-test's exit code (the in-image checks alone govern that).
    local _smoke_exporters="${AIC_SMOKE_EXPORTERS:-1}"
    local _smoke_scrape_s="${AIC_SMOKE_SCRAPE_S:-45}"
    local nvme_tar rdma_tar
    nvme_tar="$(_exporter_tarball_path "${AIC_NVME_EXPORTER_IMAGE}")"
    rdma_tar="$(_exporter_tarball_path "${AIC_RDMA_EXPORTER_IMAGE}")"

    # In-container checks live in a standalone script on shared /scratch (visible
    # on the GPU node) and are bind-mounted in -- avoids nested shell quoting.
    mkdir -p "${AIC_IMAGE_DIR}"
    local smoketest="${AIC_IMAGE_DIR}/aic-smoketest.sh"
    cat > "${smoketest}" <<'SMOKE'
#!/bin/bash
# Runs INSIDE the rocm-aic image.  EXPECT_ARCH is passed via docker -e.
set -uo pipefail
fail=0
note()  { printf '[smoketest] %s\n' "$*"; }
check() { local d="$1"; shift; if "$@" >/tmp/_ck 2>&1; then note "OK   ${d}"; \
          else note "FAIL ${d}"; sed 's/^/           /' /tmp/_ck; fail=1; fi; }

note "container: $(uname -srm)"

# GPU visibility + arch match (EXPECT_ARCH may be a ';'-separated arch list)
if command -v rocminfo >/dev/null 2>&1; then
    gfx="$(rocminfo 2>/dev/null | grep -om1 'gfx[0-9a-z]*' || true)"
    if [ -n "${gfx}" ]; then
        note "OK   GPU visible: ${gfx} (image built for ${EXPECT_ARCH:-?})"
        if [ -n "${EXPECT_ARCH:-}" ]; then
            case ";${EXPECT_ARCH};" in
                *";${gfx};"*) : ;;  # GPU arch is in the image's arch set
                *) note "WARN GPU arch ${gfx} not in image arch set ${EXPECT_ARCH}" ;;
            esac
        fi
    else
        note "FAIL no GPU reported by rocminfo"; fail=1
    fi
else
    note "FAIL rocminfo not found"; fail=1
fi

check "import vllm"    python3 -c 'import vllm; print("vllm", vllm.__version__)'
check "import lmcache" python3 -c 'import lmcache; print("lmcache", getattr(lmcache, "__version__", "?"))'
check "lmcache CLI"    command -v lmcache
check "ais-stats (hipFile)" command -v ais-stats
# ais-check reports AIS readiness across 4 components: kernel P2PDMA, HIP runtime,
# amdgpu driver, and a hipFile-capable mounted volume.  Two of those (P2PDMA and
# the volume) depend on the *run environment*, not the image -- so ais-check is
# INFORMATIONAL here (we print its report but never fail on its exit code); full
# AIS validation happens in the cliff run, which mounts a real NVMe volume.  We do
# hard-fail if the ais-check binary is missing, since that is an image defect.
if command -v ais-check >/dev/null 2>&1; then
    note "INFO ais-check (image/driver AIS pass; P2PDMA + volume depend on deployment):"
    ais-check 2>&1 | sed 's/^/           /'
else
    note "FAIL ais-check not found on PATH (image build problem)"; fail=1
fi

# Kernel release + block-device layout (informational) -- context for the
# ais-check volume/P2PDMA table above.  lsblk/nvme read the host's /sys and
# /dev, so they reflect the node's real disks/NVMe.
note "INFO kernel release: $(uname -r)"
if command -v lsblk >/dev/null 2>&1; then
    note "INFO lsblk:"
    lsblk 2>&1 | sed 's/^/           /'
else
    note "INFO lsblk not available in image"
fi
if command -v nvme >/dev/null 2>&1; then
    note "INFO nvme list:"
    nvme list 2>&1 | sed 's/^/           /'
else
    note "INFO nvme (nvme-cli) not installed in image"
fi

# NIXL plugins, incl. the AIS_MT (hipFile) backend.  AIS_MT is the only hipFile
# backend and is mandatory, so its absence is a hard failure (matches the build).
plug="${NIXL_PLUGIN_DIR:-/opt/nixl/lib/x86_64-linux-gnu/plugins}"
if [ -d "${plug}" ]; then
    note "OK   NIXL plugins: $(printf '%s ' "${plug}"/*)"
    shopt -s nullglob nocaseglob; ais_mt=("${plug}"/*ais_mt*); shopt -u nocaseglob
    if [ "${#ais_mt[@]}" -gt 0 ]; then
        note "OK   AIS_MT plugin: ${ais_mt[*]}"
    else
        note "FAIL no AIS_MT plugin in ${plug}"; fail=1
    fi
else
    note "FAIL NIXL plugin dir missing: ${plug}"; fail=1
fi

[ "${fail}" -eq 0 ] && note "ALL CHECKS PASSED" || note "SOME CHECKS FAILED"
exit "${fail}"
SMOKE
    chmod +x "${smoketest}"

    local -a _sel
    if [[ -n "${AIC_TEST_NODE:-}" ]]; then
        _sel=(--nodelist="${AIC_TEST_NODE}")
        log "testing on ${AIC_TEST_NODE} via sbatch (partition ${AIC_BUILD_PARTITION})"
    else
        _sel=(--constraint="${AIC_TEST_CONSTRAINT}")
        log "testing via sbatch (partition ${AIC_BUILD_PARTITION}, constraint ${AIC_TEST_CONSTRAINT})"
    fi
    log "image: ${AIC_IMAGE}  smoketest: ${smoketest}"

    # docker run mirrors the compose vllm service's device/ipc/cap setup so the
    # GPU is reachable; entrypoint is overridden to run the smoke test.
    local remote_script
    remote_script="$(cat <<REMOTE
set -euo pipefail
command -v docker >/dev/null 2>&1 || { echo "\$(hostname): docker not found" >&2; exit 1; }
echo "[test] host=\$(hostname) docker=\$(docker --version)"
# Load the image from the shared tarball only when needed.  A node-local marker
# records the tarball mtime that was last loaded here; we reload when the tarball
# is newer (a rebuild happened), when the image is absent, or when forced.  We
# compare the tarball's current mtime against the previously-recorded tarball
# mtime -- both are build-side values, so there is no build/test clock skew.
_marker="/var/tmp/aic-loaded-\$(id -u)-\$(echo '${AIC_IMAGE}' | tr '/:' '__').mtime"
_tar_mtime="\$(stat -c %Y '${tarball}' 2>/dev/null || echo 0)"
_have_img="\$(docker images -q '${AIC_IMAGE}')"
_loaded_mtime="\$(cat "\${_marker}" 2>/dev/null || echo 0)"
if [ "${AIC_FORCE_LOAD:-0}" = "1" ] || [ -z "\${_have_img}" ] || [ "\${_tar_mtime}" -gt "\${_loaded_mtime}" ]; then
    echo "[test] loading ${AIC_IMAGE} from ${tarball} (tarball=\${_tar_mtime} last-loaded=\${_loaded_mtime} present=\$([ -n "\${_have_img}" ] && echo yes || echo no) force=${AIC_FORCE_LOAD:-0})"
    ${DECOMPRESS_CMD} '${tarball}' | docker load >/dev/null
    echo "\${_tar_mtime}" > "\${_marker}" 2>/dev/null || true
else
    echo "[test] image up to date on \$(hostname) (id \${_have_img}, tarball mtime \${_tar_mtime} not newer than last load); AIC_FORCE_LOAD=1 forces a reload"
fi
# Expose the node's kernel config read-only so ais-check's P2PDMA probe can read
# /boot/config-* or /lib/modules/*/build/.config (informational; both may be
# absent on a given node, in which case the mounts are simply skipped).
kmounts=""
[ -d /boot ] && kmounts="\${kmounts} -v /boot:/boot:ro"
[ -d /lib/modules ] && kmounts="\${kmounts} -v /lib/modules:/lib/modules:ro"
# In-image checks govern the exit code; capture it so the exporter phase below
# (informational) can run regardless and we still exit with the real result.
img_rc=0
docker run --rm \
    --device /dev/kfd --device /dev/dri \
    --ipc host \
    --cap-add SYS_PTRACE --cap-add SYS_ADMIN \
    --security-opt seccomp=unconfined \
    \${kmounts} \
    -e EXPECT_ARCH='${AIC_ROCM_ARCH}' \
    -v '${smoketest}':/tmp/aic-smoketest.sh:ro \
    --entrypoint /bin/bash \
    '${AIC_IMAGE}' /tmp/aic-smoketest.sh || img_rc=\$?

# --- exporter + Prometheus sanity check (informational; never fails the test) --
# Stands up the same exporter fleet + Prometheus the cliff uses (docker-run path;
# GPU nodes lack the compose plugin), scrapes briefly, curls each /metrics, and
# leaves a TSDB under logs/<job-id>/prometheus.  All best-effort: missing images
# or absent hardware -> WARN and continue.
if [ '${_smoke_exporters}' = "1" ]; then
    set +e
    echo "[test] === exporter + Prometheus sanity check (scrape ${_smoke_scrape_s}s) ==="
    log() { printf '[test] %s\n' "\$*"; }
    # Best-effort load the fabric exporter images from /scratch, then advertise
    # them to the lib only when actually present on the node.
    if [ -r '${nvme_tar}' ]; then ${DECOMPRESS_CMD} '${nvme_tar}' | docker load >/dev/null 2>&1 || true; fi
    if [ -r '${rdma_tar}' ]; then ${DECOMPRESS_CMD} '${rdma_tar}' | docker load >/dev/null 2>&1 || true; fi
    docker image inspect '${AIC_NVME_EXPORTER_IMAGE}' >/dev/null 2>&1 && export AIC_NVME_EXPORTER_IMAGE='${AIC_NVME_EXPORTER_IMAGE}'
    docker image inspect '${AIC_RDMA_EXPORTER_IMAGE}' >/dev/null 2>&1 && export AIC_RDMA_EXPORTER_IMAGE='${AIC_RDMA_EXPORTER_IMAGE}'
    AIC_IMAGE='${AIC_IMAGE}'
    MON_DIR='${AIC_DAY_DIR}/monitoring'
    AIC_METRICS_DIR="\${_logdir}/prometheus"
    AIC_EXPORTERS=1
    AIC_MONITORING=1
    AIS_KFD_SYMBOL="\${AIS_KFD_SYMBOL:-kfd_ioctl_ais}"
    # shellcheck source=/dev/null
    source '${AIC_DAY_DIR}/monitoring/monitoring-lib.sh'
    mkdir -p "\${AIC_METRICS_DIR}"
    start_monitoring
    echo "[test] scraping metrics for ${_smoke_scrape_s}s ..."
    sleep '${_smoke_scrape_s}'
    monitoring_healthcheck
    monitoring_tsdb_summary
    stop_monitoring
    echo "[test] exporter sanity check complete (TSDB at \${AIC_METRICS_DIR})"
fi

exit \${img_rc}
REMOTE
)"

    local -a _gres_arg=(); [[ "${AIC_SPUR_CLUSTER}" != "1" ]] && _gres_arg=(--gres=gpu:1)
    _sbatch_run aic-test smoke-test "${remote_script}" \
        "${_sel[@]}" \
        "${_gres_arg[@]}" \
        --nodes=1 --ntasks=1 \
        --cpus-per-task="${AIC_TEST_CPUS}" --mem="${AIC_TEST_MEM}" \
        --time="${AIC_TEST_TIME}"
    log "test complete"
}

# --- main --------------------------------------------------------------------
main() {
    local sub="${1:-all}"
    case "${sub}" in
        build)           cmd_build ;;
        build-exporters) cmd_build_exporters ;;
        load)            cmd_load ;;
        push)            cmd_push ;;
        test)            cmd_test ;;
        all)             cmd_build; cmd_build_exporters; cmd_load ;;
        -h|--help|help)
            sed -n '2,70p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            ;;
        *) die "unknown command '${sub}' (use: build | build-exporters | load | push | test | all | help)" ;;
    esac
}

main "$@"
