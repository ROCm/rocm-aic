#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Build the aic-release inference image on a CPU-only build node and make it
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
#   AIC_TARGETS=<node-a>,<node-b> \
#     bash .slurm/run-build-distribute.sh all
#
#   # Just build + save the tarball:
#   bash .slurm/run-build-distribute.sh build
#
#   # Just load an already-saved tarball onto targets:
#   AIC_TARGETS=<node-a>,<node-b> \
#     bash .slurm/run-build-distribute.sh load
#
#   # Push the built image to a registry (pull-based distribution):
#   AIC_PUSH_REF=<your-registry>/<project>/rocm-aic:latest \
#     bash .slurm/run-build-distribute.sh push
#
# Commands:
#   build   Build the image on AIC_BUILD_NODE, then save the tarball to AIC_IMAGE_DIR
#   build-exporters
#           Build the fabric exporter images (nvme_exporter / rdma_exporter) from
#           monitoring/*/Dockerfile and save their tarballs to AIC_IMAGE_DIR, so
#           bare cliff nodes can containerize them (no host-installed exporter service needed).
#   load    Load the saved tarball on every node in AIC_TARGETS, then verify
#           (also loads the exporter tarballs when present)
#   push    Tag the built image as AIC_PUSH_REF and `docker push` it to a registry
#           (loads from the shared tarball first if the image is not present)
#   test    Smoke-test the image on a GPU+NVMe node (loads it there if missing):
#           checks GPU visibility + arch, vLLM / LMCache / hipFile, ais-check
#           (HIP+amdgpu AIS support), and the NIXL AIS_MT plugin (hard fail if
#           AIS_MT or ais-check fail)
#   tiny-test  End-to-end serve check on a GPU node: brings up the compose MP
#           stack (standalone lmcache server + vLLM LMCacheMPConnector) with a tiny
#           model (Qwen/Qwen2.5-0.5B-Instruct) and asserts one non-empty chat
#           completion.  Exercises the full connector path a smoke-test cannot.
#   reset-test  L1+L2 retrieval check on a GPU node: submits `make vllm-reset-test`
#           via sbatch (1 GiB L1 + NIXL POSIX L2 on local NVMe), floods the cache,
#           POSTs /reset_prefix_cache, and asserts both L1 and L2 hits.
#   all     build, build-exporters, then load   (default)
#
# Key environment:
#   AIC_ROCM_ARCH        gfx arch(es) baked in; ';'-list   (default: all vLLM archs)
#   AIC_IMAGE            full image name:tag
#   AIC_IMAGE_NAME       image name only
#   AIC_IMAGE_DIR        shared dir for the tarball        (default: /scratch/$USER/images)
#   HF_HOME              persistent Hugging Face cache used by tiny-test
#                        (default: <AIC_IMAGE_DIR>/tiny-hf)
#   ROCM_VERSION, VLLM_VERSION, VLLM_ROCM_VARIANT, LMCACHE_REF,
#   NIXL_REF, HIPFILE_SHA, HSA_SNOOP_REF
#                        optional Docker build-arg overrides.
#   AIC_FORCE_LOAD       test/push: force a reload from the tarball even when the
#                        node's image is already current (default: 0).  By default
#                        a node auto-reloads only when the /scratch tarball is
#                        newer than what it last loaded (tracked per node via a
#                        marker under /var/tmp), so a rebuild is picked up
#                        automatically without setting this.
#   AIC_TARGETS          comma-separated nodes to load     (required for load/all)
#   AIC_PUSH_REF         registry-qualified ref to push the final image to
#                        (required for push; needs `docker login <registry>` first)
#                        (e.g. <your-registry>/<project>/rocm-aic:latest)
#
#   AIC_BUILD_CONSTRAINT Slurm -C feature expr for the build node
#                        (default: <site>&CPUONLY -- CPU-only build nodes).
#                         Used only when AIC_BUILD_NODE is unset.
#   AIC_BUILD_NODE       pin an exact build node via --nodelist (overrides
#                        AIC_BUILD_CONSTRAINT)             (default: unset)
#   AIC_BUILD_LOCAL      set to 1 to build on THIS host, no Slurm  (default: unset)
#   AIC_BUILD_PARTITION  Slurm partition for build + load  (default: defq)
#   AIC_PIP_WHEELS_DIR   dir of pre-downloaded torch wheels on the build node,
#                        passed as the `pip-wheels` build context.  An empty dir is
#                        used when it is absent, so torch falls back to the index
#                        URL          (default: /opt/pip-cache/wheels)
#   AIC_BUILD_MIN_DISK_GB  minimum free space on the build node's / before the
#                        build starts; below this it fails immediately rather
#                        than dying later on truncated apt metadata (default: 150)
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
#                        (e.g. <your-registry>/<project>/rocm-aic:buildcache)
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
#                        (default: <site>&GFX942&NVME -- MI300X + local NVMe).
#                        Used only when AIC_TEST_NODE is unset.
#   AIC_TEST_NODE        pin an exact test node via --nodelist  (default: unset)
#   AIC_RESET_FLOOD      flood prompts for reset-test; must overflow the 1 GiB L1
#                        or no chunks reach L2            (default: 400)
#   AIC_TEST_ARCH        gfx arch of the test node, used as ROCM_ARCH for the
#                        tiny-test compose stack.  Auto-detected by
#                        .slurm/aic-test-arch.sh when unset (gfx950 on SPUR,
#                        otherwise from the Slurm node/constraint/partition)
#   AIC_SLURM_ACCOUNT    --account for every submission (default: unset = Slurm default)
#   AIC_TEST_ACCOUNT     --account for test + tiny-test only, when the test partition
#                        restricts AllowAccounts   (default: $AIC_SLURM_ACCOUNT)
#   AIC_TEST_PARTITION   Slurm partition for test + tiny-test, when the GPU nodes
#                        live outside the build partition (e.g. the MI300X storage
#                        node in `storage`)   (default: $AIC_BUILD_PARTITION)
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
#                        (default: $SPUR_CONTROLLER_ADDR)
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
[[ -s "${AIC_DAY_DIR}/VERSION" ]] || {
    printf '[build-distribute] ERROR: VERSION is missing or empty\n' >&2
    exit 1
}
AIC_VERSION="$(<"${AIC_DAY_DIR}/VERSION")"

# --- Defaults ----------------------------------------------------------------
# Multi-arch by default: every gfx the vLLM ROCm wheel ships kernels for.
# hipFile + the LMCache HIP extension are compiled for all of these (cmake and
# PYTORCH_ROCM_ARCH both accept ';'-lists); NIXL AIS_MT is host-only, so arch-
# independent.  NOTE: the RDNA entries (gfx11xx/gfx12xx) have no NVMe-DMA /
# Infinity-Storage hardware -- if hipFile fails to build for them, narrow this
# to the CDNA set "gfx90a;gfx942;gfx950".  Override via AIC_ROCM_ARCH.
AIC_ROCM_ARCH="${AIC_ROCM_ARCH:-gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201}"
AIC_UCX_FAST="${AIC_UCX_FAST:-}"
# Default the image ref to the version-derived tag (see docker/scripts/aic-image-tag.sh)
# so the saved tarball, load, push, and the cliff's tarball glob all agree on the
# same name without hardcoding a version here.  Falls back to :latest if the helper
# can't resolve the pins.
#
# Callers must pass one of:
#   AIC_IMAGE_NAME  name only for the version-derived tag to be appended here.
#   AIC_IMAGE       a complete name:tag, used verbatim.  Wins over AIC_IMAGE_NAME.
_aic_tag="$(bash "${AIC_DAY_DIR}/docker/scripts/aic-image-tag.sh" 2>/dev/null || true)"
AIC_IMAGE="${AIC_IMAGE:-${AIC_IMAGE_NAME:-rocm-aic}:${_aic_tag:-latest}}"
AIC_IMAGE_DIR="${AIC_IMAGE_DIR:-/scratch/${USER}/images}"
AIC_PIP_WHEELS_DIR="${AIC_PIP_WHEELS_DIR:-/opt/pip-cache/wheels}"
AIC_BUILD_MIN_DISK_GB="${AIC_BUILD_MIN_DISK_GB:-150}"
# Slurm account.  Empty = let Slurm pick the default association.  Needed when a
# partition sets AllowAccounts (e.g. `storage` is gds,auto -- the default `ais`
# association is rejected with "Invalid account or account/partition combination").
AIC_SLURM_ACCOUNT="${AIC_SLURM_ACCOUNT:-}"
AIC_TEST_ACCOUNT="${AIC_TEST_ACCOUNT:-${AIC_SLURM_ACCOUNT}}"
AIC_SPUR_CLUSTER="${AIC_SPUR_CLUSTER:-0}"
# Only *required* when AIC_SPUR_CLUSTER=1 -- the check lives in the branch below
# so a plain (non-SPUR) run doesn't need SPUR_CONTROLLER_ADDR in the environment.
AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER:-${SPUR_CONTROLLER_ADDR:-}}"

# When running on SPUR, default the partition to amd-spur (the only partition)
# and clear the build/test constraints (SPUR nodes have no MARKHAM/CPUONLY/GFX942
# feature labels; node selection is done by partition or explicit --nodelist).
if [[ "${AIC_SPUR_CLUSTER}" == "1" ]]; then
    : "${AIC_SPUR_CONTROLLER:?set SPUR_CONTROLLER_ADDR or AIC_SPUR_CONTROLLER before using AIC_SPUR_CLUSTER=1 (\`source .env.aic\`)}"
    AIC_BUILD_PARTITION="${AIC_BUILD_PARTITION:-amd-spur}"
    AIC_IMAGE_DIR="${AIC_IMAGE_DIR:-${AIC_SHARED_NFS:-/shared_nfs}/${USER}/images}"
    # Use ${VAR-default} (not ${VAR:-default}) so an explicitly set empty string
    # ("AIC_BUILD_CONSTRAINT=") is honoured as "no constraint".
    AIC_BUILD_CONSTRAINT="${AIC_BUILD_CONSTRAINT-}"
    AIC_TEST_CONSTRAINT="${AIC_TEST_CONSTRAINT-}"
else
    AIC_BUILD_PARTITION="${AIC_BUILD_PARTITION:-defq}"
    AIC_BUILD_CONSTRAINT="${AIC_BUILD_CONSTRAINT:-CPUONLY}"
    AIC_TEST_CONSTRAINT="${AIC_TEST_CONSTRAINT:-GFX942&NVME}"
fi
AIC_PIP_WHEELS_DIR="${AIC_PIP_WHEELS_DIR:-}"
# Test jobs land in the build partition unless the GPU nodes live elsewhere.
AIC_TEST_PARTITION="${AIC_TEST_PARTITION:-${AIC_BUILD_PARTITION}}"
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
AIC_TEST_TIME="${AIC_TEST_TIME:-00:45:00}"
AIC_TEST_CPUS="${AIC_TEST_CPUS:-8}"
AIC_TEST_MEM="${AIC_TEST_MEM:-32G}"

# --- tiny-test: end-to-end serve check with a tiny model ---------------------
# Brings up the compose MP stack (standalone lmcache + vLLM LMCacheMPConnector)
# with a small model and asserts one non-empty chat completion.  A fast functional
# gate that exercises the connector path a smoke-test cannot.  The model is
# downloaded once into HF_HOME (a persistent shared HF cache) and reused.
AIC_TINY_MODEL="${AIC_TINY_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
HF_HOME="${HF_HOME:-${AIC_IMAGE_DIR}/tiny-hf}"
AIC_TINY_TIME="${AIC_TINY_TIME:-00:25:00}"
AIC_TINY_CPUS="${AIC_TINY_CPUS:-8}"
AIC_TINY_MEM="${AIC_TINY_MEM:-32G}"
AIC_TINY_READY_TIMEOUT="${AIC_TINY_READY_TIMEOUT:-120}"   # x5s = up to 10 min for weights + download

# reset-test: sbatch wrapper around `make vllm-reset-test` (L1+L2 retrieval).
AIC_RESET_MODEL="${AIC_RESET_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
AIC_RESET_TIME="${AIC_RESET_TIME:-01:00:00}"
AIC_RESET_CPUS="${AIC_RESET_CPUS:-16}"
AIC_RESET_MEM="${AIC_RESET_MEM:-64G}"
# Cap the GPU KV cache so vLLM spills into LMCache sooner.  The test also POSTs
# /reset_prefix_cache to flush the GPU prefix cache outright, so this is belt-and-
# braces rather than load-bearing.
AIC_RESET_GPU_UTIL="${AIC_RESET_GPU_UTIL:-0.20}"
AIC_RESET_MAX_LEN="${AIC_RESET_MAX_LEN:-8192}"
# POSIX slot file must be >= the model KV chunk size or every L2 write fails and
# the test reports "no L2 hits".  Compose defaults to 16 MiB, which is too small
# for Qwen2.5-3B at bf16 (~36 MiB/chunk).  Slots are sparse, so oversizing is cheap.
AIC_RESET_SLOT_SIZE="${AIC_RESET_SLOT_SIZE:-67108864}"
# Host dir for the NIXL POSIX L2 pool.  Empty = pick a writable NVMe mount on the
# node (the compose default /mnt/lmcache-nvme does not exist on every GPU node,
# and _prep_dirs cannot mkdir under /mnt without root).
AIC_RESET_NVME_DATA="${AIC_RESET_NVME_DATA:-}"
# 1 = also stand up Prometheus + the exporter fleet and keep the TSDB under
# logs/<job-id>/prometheus.  The two aic-*-exporter:local images are unpushed local
# build artifacts, so they are loaded from their tarballs onto the test node below;
# set 0 to skip monitoring entirely (the test itself only scrapes lmcache's :8080).
AIC_RESET_MONITORING="${AIC_RESET_MONITORING:-1}"
# vllm_reset_test.py defaults AIC_TEST_FLOOD to 50, which only reaches ~33% of the
# 1 GiB L1 for a 3B model -- L1 never overflows, nothing reaches L2, and the test
# always reports "no L2 hits".  400 gives ~14 chunks x 400 >> the ~222-chunk cap.
AIC_RESET_FLOOD="${AIC_RESET_FLOOD:-400}"

# --- Fabric exporter images (nvme_exporter / rdma_exporter) -------------------
# Built from monitoring/*/Dockerfile and distributed alongside the main image so
# bare cliff nodes can containerize them (no host-installed exporter service needed).  Names
# must match run-cliff.sbatch's defaults so the tarballs written here are found there.
# Versions default to the Grafana-parity versions; override via AIC_NVME/RDMA_EXPORTER_VERSION.
AIC_NVME_EXPORTER_IMAGE="${AIC_NVME_EXPORTER_IMAGE:-aic-nvme-exporter:latest}"
AIC_RDMA_EXPORTER_IMAGE="${AIC_RDMA_EXPORTER_IMAGE:-aic-rdma-exporter:latest}"
AIC_NVME_EXPORTER_VERSION="${AIC_NVME_EXPORTER_VERSION:-3.0.0}"  # matches host-service default; override as needed
AIC_RDMA_EXPORTER_VERSION="${AIC_RDMA_EXPORTER_VERSION:-0.3.0}"  # matches host-service default; override as needed

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

    # Partition for this submission.  Callers that need a different one than the
    # build partition (cmd_test / cmd_tiny_test -> AIC_TEST_PARTITION) declare
    # `local _SBATCH_PARTITION=...`, which dynamic scoping makes visible here.
    local _part="${_SBATCH_PARTITION:-${AIC_BUILD_PARTITION}}"
    local _acct="${_SBATCH_ACCOUNT:-${AIC_SLURM_ACCOUNT}}"
    local -a _acct_arg=(); [[ -n "${_acct}" ]] && _acct_arg=(--account="${_acct}")

    # Batch script = shebang + a prologue that creates the per-job log dir and
    # redirects everything into <logname>.out, then the caller's body.
    # AIC_DAY_DIR is absolute and on shared storage, so it resolves on the
    # compute node without relying on SLURM_SUBMIT_DIR.  --output=/dev/null
    # discards any pre-redirect output (there is none here).
    local script
    # The body runs in a subshell so its own `trap EXIT` cannot clobber the
    # outer exit-file write.  The outer EXIT trap always fires last and records
    # the subshell's real exit code regardless of what traps the body set.
    script="$(cat <<PROLOGUE
#!/bin/bash
_logdir="${AIC_DAY_DIR}/logs/\${SLURM_JOB_ID:-manual}"
_exitfile="${AIC_DAY_DIR}/logs/\${SLURM_JOB_ID:-manual}/${logname}.exit"
mkdir -p "\${_logdir}" 2>/dev/null && exec >>"\${_logdir}/${logname}.out" 2>&1
( ${body} )
_body_rc=\$?
echo "\${_body_rc}" > "\${_exitfile}" 2>/dev/null || true
exit "\${_body_rc}"
PROLOGUE
)"

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
            --partition="${_part}" \
            "${_acct_arg[@]}" \
            --output=/dev/null \
            "$@" \
            "${tmpscript}" 2>&1)" || { rm -f "${tmpscript}"; die "sbatch submission failed: ${submit_out}"; }
        rm -f "${tmpscript}"

        jobid="$(printf '%s\n' "${submit_out}" | grep -oE '[0-9]+$' | tail -1)"
        [[ -n "${jobid}" ]] || die "could not parse job id from sbatch output: ${submit_out}"
        logfile="${AIC_DAY_DIR}/logs/${jobid}/${logname}.out"
        log "submitted ${jobname} as job ${jobid} (partition ${_part})"
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

        # SPUR ignores -j and may return a large queue.  Consume all of squeue's
        # output before deciding whether the job is present: an early-exiting
        # grep -q closes the pipe and makes squeue fail with SIGPIPE under
        # pipefail, which looks like the job disappeared.
        _spur_job_is_queued() {
            squeue --controller="${AIC_SPUR_CONTROLLER}" -j "${jobid}" -h 2>/dev/null |
                awk -v id="${jobid}" '
                    $1 == id { found = 1 }
                    END { exit found ? 0 : 1 }
                '
        }

        # Wait up to 60s for the job to appear.
        local appear_tries=0
        until _spur_job_is_queued || (( appear_tries >= 60 )); do
            sleep 1; appear_tries=$((appear_tries + 1))
        done

        # Poll until the job leaves the queue, streaming new log lines.
        while _spur_job_is_queued; do
            _print_new_lines
            sleep 10
        done

        # Flush any remaining lines after job completes.
        sleep 2
        _print_new_lines

        # Read the real exit code from sacct ("<code>:<signal>" format).
        # SPUR sacct ignores -j and returns all jobs; grep for the exact job ID
        # in JobID+ExitCode output to avoid picking up an unrelated row.
        # Use the exit code written by the job script itself (under the log dir)
        # as the authoritative source; fall back to sacct only when absent.
        local acct_exit exit_file="${AIC_DAY_DIR}/logs/${jobid}/${logname}.exit"
        # Wait up to 10s for the exit file to appear on NFS (it is written by the
        # job's EXIT trap; NFS open/close may lag a few seconds after job end).
        local ef_tries=0
        until [[ -f "${exit_file}" ]] || (( ef_tries >= 10 )); do
            sleep 1; ef_tries=$((ef_tries + 1))
        done
        if [[ -f "${exit_file}" ]]; then
            acct_exit="$(tr -d '[:space:]' < "${exit_file}" 2>/dev/null)"
            log "exit code from file: ${acct_exit} (${exit_file})"
        else
            acct_exit="$(sacct --controller="${AIC_SPUR_CONTROLLER}" -j "${jobid}" \
                --format=JobID,ExitCode --noheader 2>/dev/null \
                | awk -v id="${jobid}" '
                    $1 == id && !found {
                        split($2, fields, ":")
                        code = fields[1]
                        found = 1
                    }
                    END { if (found) print code }
                ')"
            log "exit code from sacct: ${acct_exit:-<empty>} (job ${jobid})"
        fi
        rc="${acct_exit:-1}"
    else
        # Standard Slurm path: --parsable prints the bare job id; --wait blocks
        # until the job finishes and exits with the job's exit code.
        local idfile; idfile="$(mktemp)"
        local -a _stdbuf=(); command -v stdbuf >/dev/null 2>&1 && _stdbuf=(stdbuf -oL)
        "${_stdbuf[@]}" sbatch --parsable --wait \
            --job-name="${jobname}" \
            --partition="${_part}" \
            "${_acct_arg[@]}" \
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
            log "submitted ${jobname} as job ${jobid} (partition ${_part})"
            log "log: ${logfile}"
        else
            log "submitted ${jobname} (job id not yet available; partition ${_part})"
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
    # Alias ref (name:latest) built + saved alongside the versioned ref.
    local latest_ref="${AIC_IMAGE%:*}:latest"
    # Forward every version override for Docker tag naming.
    local _version_build_args="" _version_arg _version_value
    printf -v _version_value '%q' "${AIC_VERSION}"
    _version_build_args+=" --build-arg AIC_VERSION=${_version_value}"
    for _version_arg in \
        ROCM_VERSION VLLM_VERSION VLLM_ROCM_VARIANT \
        LMCACHE_REF NIXL_REF HIPFILE_SHA HSA_SNOOP_REF; do
        if [[ -v "${_version_arg}" ]]; then
            printf -v _version_value '%q' "${!_version_arg}"
            _version_build_args+=" --build-arg ${_version_arg}=${_version_value}"
        fi
    done

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
            _cache_args="--cache-from type=local,src=${_cdir} --cache-to type=local,dest=${_cdir},mode=${AIC_CACHE_MODE},ignore-error=true"
            _mkdir="mkdir -p '${_cdir}'; "
        fi
        # Create the docker-container builder once per node (idempotent), then
        # bootstrap it so its BuildKit is ready before the build starts.
        _builder_setup="${_pre}${_mkdir}if ! docker buildx inspect ${AIC_BUILDX_BUILDER} >/dev/null 2>&1; then echo '[build] creating buildx builder ${AIC_BUILDX_BUILDER} (docker-container)'; docker buildx create --name ${AIC_BUILDX_BUILDER} --driver docker-container${_cfg_arg} >/dev/null; fi; docker buildx inspect --bootstrap ${AIC_BUILDX_BUILDER} >/dev/null"
    fi

    # pip-wheels build context: supply a local wheel cache dir to skip the 6 GB
    # torch download.  If AIC_PIP_WHEELS_DIR is unset, fall back to an empty
    # sentinel dir on shared storage so BuildKit does not try to pull the
    # non-existent docker.io/library/pip-wheels image.
    local _pip_wheels_dir="${AIC_PIP_WHEELS_DIR}"
    if [[ -z "${_pip_wheels_dir}" ]]; then
        _pip_wheels_dir="${AIC_DAY_DIR}/.empty-pip-wheels"
        mkdir -p "${_pip_wheels_dir}"
    fi
    local _pip_wheels_arg="--build-context pip-wheels=${_pip_wheels_dir}"
    log "pip-wheels context: ${_pip_wheels_dir}"

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
# BuildKit writes a temp dir for config injection into TMPDIR (defaults to /tmp).
# On SPUR compute nodes /tmp may not be writable for this user; use $HOME/tmp instead.
mkdir -p "\${HOME}/.tmp-rocm-aic-cicd"
export TMPDIR="\${HOME}/.tmp-rocm-aic-cicd"
cd "${AIC_DAY_DIR}"
${_builder_setup}
mkdir -p "${AIC_IMAGE_DIR}"
# Prune the BuildKit cache before building to prevent the builder's local
# volume from filling the node's disk and silently killing the export.
echo "[build] pruning BuildKit cache on \$(hostname) before build ..."
docker buildx prune --builder ${AIC_BUILDX_BUILDER} --force 2>/dev/null || true
echo "[build] disk after prune: \$(df -h / | awk 'NR==2{print \$4" avail / "\$2" total ("\$5" used)"}')"
# Fail fast on a full disk.  A node whose / is full does not error cleanly: apt
# writes truncated InRelease files and the build dies ~15 min in with a baffling
# "GPG error: At least one invalid signature was encountered" on every repo.
_avail_g="\$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')"
if [ "\${_avail_g:-0}" -lt "${AIC_BUILD_MIN_DISK_GB}" ]; then
    echo "[build] ERROR: only \${_avail_g}G free on / at \$(hostname); need >= ${AIC_BUILD_MIN_DISK_GB}G" >&2
    echo "[build] Pin a node with room via AIC_BUILD_NODE=<node>, or lower AIC_BUILD_MIN_DISK_GB." >&2
    exit 1
fi
# The Dockerfile mounts a named build context 'pip-wheels' (pre-downloaded torch
# wheels, ~6 GB).  A named context that is NOT supplied is resolved by BuildKit as
# an IMAGE ref -- docker.io/library/pip-wheels:latest -- which fails the pull, so
# it must always be passed.  Point it at an empty dir when the wheel cache is
# absent on this node: the Dockerfile probes for the wheel and falls back to the
# pytorch.org index URL when it is not there.
_wheels_dir="${AIC_PIP_WHEELS_DIR}"
if [ ! -d "\${_wheels_dir}" ]; then
    _wheels_dir="\${TMPDIR:-/tmp}/aic-empty-wheels"
    mkdir -p "\${_wheels_dir}"
    echo "[build] no wheel cache at ${AIC_PIP_WHEELS_DIR}; using empty pip-wheels context (torch comes from the index URL)"
else
    echo "[build] pip-wheels build context: \${_wheels_dir}"
fi
tmp="${tarball}.partial.\$\$"
docker buildx build --builder ${AIC_BUILDX_BUILDER} --progress=plain --output type=docker,dest=- \
    --build-context pip-wheels="\${_wheels_dir}" \
    --build-arg ROCM_ARCH="${AIC_ROCM_ARCH}" \
    --build-arg AIC_UCX_FAST="${AIC_UCX_FAST}" \
    ${_version_build_args} \
    ${_secret_arg} \
    ${_cache_args} \
    ${_pip_wheels_arg} \
    -f "${AIC_DAY_DIR}/docker/Dockerfile" \
    -t "${AIC_IMAGE}" \
    -t "${latest_ref}" \
    "${AIC_DAY_DIR}" | ${COMPRESS_CMD} > "\${tmp}"
_rc=("\${PIPESTATUS[@]}")
set -o pipefail
if [ "\${_rc[1]}" -ne 0 ]; then
    echo "[build] ERROR: compressor exited \${_rc[1]}; tarball may be corrupt" >&2; exit 1
fi
if [ "\${_rc[0]}" -ne 0 ]; then
    echo "[build] ERROR: docker buildx exited \${_rc[0]}; build failed (patch apply error or Dockerfile issue)" >&2
    rm -f "\${tmp}"
    exit 1
fi
mv -f "\${tmp}" "${tarball}"
echo "[build] saved \$(du -h "${tarball}" | cut -f1) -> ${tarball}"
exit 0
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
# Fail fast on a full disk.  A node whose / is full does not error cleanly: apt
# writes truncated InRelease files and the build dies ~15 min in with a baffling
# "GPG error: At least one invalid signature was encountered" on every repo.
_avail_g="\$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')"
if [ "\${_avail_g:-0}" -lt "${AIC_BUILD_MIN_DISK_GB}" ]; then
    echo "[build] ERROR: only \${_avail_g}G free on / at \$(hostname); need >= ${AIC_BUILD_MIN_DISK_GB}G" >&2
    echo "[build] Pin a node with room via AIC_BUILD_NODE=<node>, or lower AIC_BUILD_MIN_DISK_GB." >&2
    exit 1
fi
# The Dockerfile mounts a named build context 'pip-wheels' (pre-downloaded torch
# wheels, ~6 GB).  A named context that is NOT supplied is resolved by BuildKit as
# an IMAGE ref -- docker.io/library/pip-wheels:latest -- which fails the pull, so
# it must always be passed.  Point it at an empty dir when the wheel cache is
# absent on this node: the Dockerfile probes for the wheel and falls back to the
# pytorch.org index URL when it is not there.
_wheels_dir="${AIC_PIP_WHEELS_DIR}"
if [ ! -d "\${_wheels_dir}" ]; then
    _wheels_dir="\${TMPDIR:-/tmp}/aic-empty-wheels"
    mkdir -p "\${_wheels_dir}"
    echo "[build] no wheel cache at ${AIC_PIP_WHEELS_DIR}; using empty pip-wheels context (torch comes from the index URL)"
else
    echo "[build] pip-wheels build context: \${_wheels_dir}"
fi
${_build_program} \
    --build-context pip-wheels="\${_wheels_dir}" \
    --build-arg ROCM_ARCH="${AIC_ROCM_ARCH}" \
    --build-arg AIC_UCX_FAST="${AIC_UCX_FAST}" \
    ${_version_build_args} \
    ${_secret_arg} \
    ${_cache_args} \
    -f "${AIC_DAY_DIR}/docker/Dockerfile" \
    -t "${AIC_IMAGE}" \
    -t "${latest_ref}" \
    "${AIC_DAY_DIR}"
echo "[build] built ${AIC_IMAGE}"
mkdir -p "${AIC_IMAGE_DIR}"
tmp="${tarball}.partial.\$\$"
docker save "${AIC_IMAGE}" "${latest_ref}" | ${COMPRESS_CMD} > "\${tmp}"
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

    log "exporter images : ${AIC_NVME_EXPORTER_IMAGE} (nvme v${AIC_NVME_EXPORTER_VERSION}), ${AIC_RDMA_EXPORTER_IMAGE} (rdma v${AIC_RDMA_EXPORTER_VERSION})"
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
set +o pipefail
docker buildx build --builder ${AIC_BUILDX_BUILDER} --output type=docker,dest=- \
    --build-arg NVME_EXPORTER_VERSION="${AIC_NVME_EXPORTER_VERSION}" \
    -t "${AIC_NVME_EXPORTER_IMAGE}" "${AIC_DAY_DIR}/monitoring/nvme-exporter" | ${COMPRESS_CMD} > "\${tmp}"
_rc=("\${PIPESTATUS[@]}")
set -o pipefail
if [ "\${_rc[1]}" -ne 0 ]; then
    echo "[build-exporters] ERROR: compressor exited \${_rc[1]} for nvme image; tarball may be corrupt" >&2; exit 1
fi
if [ "\${_rc[0]}" -ne 0 ]; then
    echo "[build-exporters] WARN: docker buildx exited \${_rc[0]} for nvme image (cache lock race?); tarball written, continuing"
fi
mv -f "\${tmp}" "${nvme_tar}"
tmp="${rdma_tar}.partial.\$\$"
set +o pipefail
docker buildx build --builder ${AIC_BUILDX_BUILDER} --output type=docker,dest=- \
    --build-arg RDMA_EXPORTER_VERSION="${AIC_RDMA_EXPORTER_VERSION}" \
    -t "${AIC_RDMA_EXPORTER_IMAGE}" "${AIC_DAY_DIR}/monitoring/rdma-exporter" | ${COMPRESS_CMD} > "\${tmp}"
_rc=("\${PIPESTATUS[@]}")
set -o pipefail
if [ "\${_rc[1]}" -ne 0 ]; then
    echo "[build-exporters] ERROR: compressor exited \${_rc[1]} for rdma image; tarball may be corrupt" >&2; exit 1
fi
if [ "\${_rc[0]}" -ne 0 ]; then
    echo "[build-exporters] WARN: docker buildx exited \${_rc[0]} for rdma image (cache lock race?); tarball written, continuing"
fi
mv -f "\${tmp}" "${rdma_tar}"
echo "[build-exporters] saved \$(du -h "${nvme_tar}" | cut -f1) -> ${nvme_tar}"
echo "[build-exporters] saved \$(du -h "${rdma_tar}" | cut -f1) -> ${rdma_tar}"
exit 0
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

    # Seen by _sbatch_run via dynamic scoping; the GPU test nodes may sit in a
    # different partition than the CPU-only build nodes.
    local _SBATCH_PARTITION="${AIC_TEST_PARTITION}"
    local _SBATCH_ACCOUNT="${AIC_TEST_ACCOUNT}"
    local -a _sel
    if [[ -n "${AIC_TEST_NODE:-}" ]]; then
        _sel=(--nodelist="${AIC_TEST_NODE}")
        log "testing on ${AIC_TEST_NODE} via sbatch (partition ${AIC_TEST_PARTITION})"
    else
        _sel=(--constraint="${AIC_TEST_CONSTRAINT}")
        log "testing via sbatch (partition ${AIC_TEST_PARTITION}, constraint ${AIC_TEST_CONSTRAINT})"
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
    # Compose-only monitoring needs MON_COMPOSE set (the docker-run fallback is
    # gone); without it start_monitoring skips the whole exporter/Prometheus stack.
    MON_COMPOSE='${AIC_DAY_DIR}/monitoring/docker-compose.monitoring.yml'
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

# --- tiny-test: end-to-end serve check (compose MP stack + a tiny model) ------
# Loads the image on a GPU node if needed, brings up the SAME compose stack the
# cliff/`make up` use (standalone lmcache server + vLLM LMCacheMPConnector), waits
# for the endpoint, and asserts one non-empty chat completion.  Unlike smoke-test
# (which validates the image in isolation) this exercises the full MP connector
# path end-to-end -- the functional gate wired into CI after smoke-test.
cmd_tiny_test() {
    _pick_compress
    local tarball; tarball="$(_tarball_path)"
    command -v sbatch >/dev/null 2>&1 || die "sbatch not found; cannot run the GPU tiny-test job"
    [[ -r "${tarball}" ]] || die "tarball not found: ${tarball} (run 'build' first)"

    local _SBATCH_PARTITION="${AIC_TEST_PARTITION}"
    local _SBATCH_ACCOUNT="${AIC_TEST_ACCOUNT}"

    # ROCM_ARCH for the compose stack must be the arch of the node we land on --
    # NOT AIC_ROCM_ARCH, which for a release build is the whole ';'-separated
    # list of archs baked into the image.  Resolved here (lazily: the detection
    # costs an sinfo call) so plain tiny-test on the multi-arch tarball still
    # hands the node a single, correct arch.
    local _test_arch="${AIC_TEST_ARCH:-}"
    if [[ -z "${_test_arch}" ]]; then
        if [[ "${AIC_ROCM_ARCH}" != *";"* ]]; then
            _test_arch="${AIC_ROCM_ARCH}"   # single-arch build: already exact
        else
            _test_arch="$(bash "${SCRIPT_DIR}/aic-test-arch.sh" 2>/dev/null || true)"
            [[ -n "${_test_arch}" ]] || _test_arch="${AIC_ROCM_ARCH}"
        fi
    fi
    log "test-node arch: ${_test_arch}  (image built for ${AIC_ROCM_ARCH})"

    local -a _sel
    if [[ -n "${AIC_TEST_NODE:-}" ]]; then
        _sel=(--nodelist="${AIC_TEST_NODE}")
        log "tiny-test on ${AIC_TEST_NODE} via sbatch (partition ${AIC_TEST_PARTITION})"
    else
        _sel=(--constraint="${AIC_TEST_CONSTRAINT}")
        log "tiny-test via sbatch (partition ${AIC_TEST_PARTITION}, constraint ${AIC_TEST_CONSTRAINT})"
    fi
    log "image: ${AIC_IMAGE}  model: ${AIC_TINY_MODEL}  hf: ${HF_HOME}"

    local remote_script
    remote_script="$(cat <<REMOTE
set -uo pipefail
command -v docker >/dev/null 2>&1 || { echo "\$(hostname): docker not found" >&2; exit 1; }
echo "[tiny-test] host=\$(hostname) docker=\$(docker --version)"

# Load the image from the shared tarball only when needed (same marker logic as
# smoke-test): reload when forced, absent, or the tarball is newer.
_marker="/var/tmp/aic-loaded-\$(id -u)-\$(echo '${AIC_IMAGE}' | tr '/:' '__').mtime"
_tar_mtime="\$(stat -c %Y '${tarball}' 2>/dev/null || echo 0)"
_have_img="\$(docker images -q '${AIC_IMAGE}')"
_loaded_mtime="\$(cat "\${_marker}" 2>/dev/null || echo 0)"
if [ "${AIC_FORCE_LOAD:-0}" = "1" ] || [ -z "\${_have_img}" ] || [ "\${_tar_mtime}" -gt "\${_loaded_mtime}" ]; then
    echo "[tiny-test] loading ${AIC_IMAGE} from ${tarball}"
    ${DECOMPRESS_CMD} '${tarball}' | docker load >/dev/null
    echo "\${_tar_mtime}" > "\${_marker}" 2>/dev/null || true
else
    echo "[tiny-test] image up to date on \$(hostname) (id \${_have_img})"
fi

cd '${AIC_DAY_DIR}'
# docker compose v2 only -- install user-locally if the node lacks the plugin.
# shellcheck source=/dev/null
source '${AIC_DAY_DIR}/monitoring/monitoring-lib.sh'
ensure_compose || { echo "[tiny-test] docker compose unavailable and could not be installed" >&2; exit 1; }

# Tiny-model MP stack env.  Small footprint; the tiny model is downloaded online
# into the persistent HF_HOME forwarded by the Makefile.
export IMAGE_REF='${AIC_IMAGE}'
export IMAGE_NAME='${AIC_IMAGE%:*}'
export ROCM_ARCH='${_test_arch}'
export GPU=0
export VLLM_MODEL='${AIC_TINY_MODEL}'
export HF_HOME='${HF_HOME}'
export HF_TOKEN='${HF_TOKEN:-}'
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export LOG="\${_logdir}"
export NVME_DATA=/tmp/aic-tiny-nvme NFS_DATA=/tmp/aic-tiny-nfs
export VLM_GPU_MEMORY_UTILIZATION=0.30
export VLM_MAX_MODEL_LEN=4096
export VLM_MAX_NUM_BATCHED_TOKENS=4096
export VLM_ATTENTION_BACKEND=TRITON_ATTN
export VLM_KV_CACHE_DTYPE=auto
export LMCACHE_L1_SIZE_GB=4
# DRAM-only L1, no L2 tier: the minimal MP config -- robust on a node with only
# /tmp (AIS_MT/GDS and file-based L2 need a real NVMe/GDS volume). tiny-test only
# needs to prove the LMCacheMPConnector round-trip + serve works end-to-end.
export AIC_L2_BACKEND=none
export VLLM_IPC_MODE=service:lmcache
export VLLM_PID_MODE=service:lmcache
# Host MUST be the lmcache container, not localhost: the vllm service shares only
# lmcache's PID/IPC namespaces, not its network namespace, so both sit on the aic
# bridge with their own loopback.  'tcp://localhost' makes vLLM dial itself and the
# engine core dies after the 300s MP connect timeout.  Keep this in step with the
# Makefile's _MP_CONNECTOR_JSON.
export KV_TRANSFER_ARG="--kv-transfer-config '{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://aic-lmcache\",\"lmcache.mp.port\":${LMCACHE_PORT:-6555}}}'"
mkdir -p "\${HF_HOME}" /tmp/aic-tiny-nvme /tmp/aic-tiny-nfs

compose() { docker compose -f '${AIC_DAY_DIR}/docker/docker-compose.yml' "\$@"; }
cleanup() {
    local svc c
    for svc in vllm lmcache; do
        timeout 30 compose logs --no-color --no-log-prefix "\$svc" > "\${_logdir}/tiny-\${svc}.log" 2>&1 || true
    done
    # vLLM shares lmcache's PID ns (for cross-container HIP IPC), which blocks docker
    # from reaping vLLM's EngineCore children -> compose down/docker rm hang.  Force-
    # kill the stack first, then tear down (one MP stack per node under host net).
    pkill -9 -f 'vllm.entrypoints.openai' 2>/dev/null || true
    pkill -9 -f 'EngineCore'              2>/dev/null || true
    pkill -9 -f 'lmcache server'          2>/dev/null || true
    sleep 2
    timeout 60 compose --profile cache down --remove-orphans --timeout 5 >/dev/null 2>&1 || true
    for c in aic-vllm-gpu0 aic-lmcache; do timeout 30 docker rm -f "\$c" >/dev/null 2>&1 || true; done
    rm -rf /tmp/aic-tiny-nvme /tmp/aic-tiny-nfs 2>/dev/null || true
}
trap cleanup EXIT

echo "[tiny-test] bringing up compose MP stack (model=${AIC_TINY_MODEL}) ..."
if ! compose --profile cache up -d; then
    echo "[tiny-test] FAIL: compose up failed" >&2
    compose logs --tail 60 --no-color lmcache 2>&1 | sed 's/^/  [lmcache] /'
    compose logs --tail 60 --no-color vllm    2>&1 | sed 's/^/  [vllm]    /'
    exit 1
fi

# Wait for the vLLM endpoint (weights load + one-time model download).
# The API is published on the 'aic' compose network ONLY -- docker-compose.yml has
# no ports: mapping -- so a host-side curl can never connect and just burns the
# full timeout.  Probe from inside the vllm container on its own loopback: that
# needs no second container to be healthy first.
_vllm="aic-vllm-gpu\${GPU:-0}"
ready=0
for _i in \$(seq 1 ${AIC_TINY_READY_TIMEOUT}); do
    if docker exec "\${_vllm}" curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then ready=1; break; fi
    sleep 5
done
if [ "\${ready}" != "1" ]; then
    echo "[tiny-test] FAIL: vLLM never became ready" >&2
    compose logs --tail 80 --no-color vllm 2>&1 | sed 's/^/  [vllm] /'
    exit 1
fi
echo "[tiny-test] endpoint ready; sending one chat completion ..."

# One real completion; assert a NON-EMPTY assistant content came back through the
# LMCacheMPConnector path.
resp="\$(docker exec "\${_vllm}" curl -fsS http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"${AIC_TINY_MODEL}","messages":[{"role":"user","content":"Reply with the single word: pong"}],"max_tokens":16,"temperature":0}' 2>&1)" || {
    echo "[tiny-test] FAIL: completion request failed: \${resp}" >&2; exit 1; }
echo "[tiny-test] response: \${resp}"
if printf '%s' "\${resp}" | grep -qE '"content"[[:space:]]*:[[:space:]]*"[^"]+'; then
    echo "[tiny-test] OK: got a non-empty completion via LMCacheMPConnector"
    exit 0
fi
echo "[tiny-test] FAIL: empty or missing completion content" >&2
exit 1
REMOTE
)"

    local -a _gres_arg=(); [[ "${AIC_SPUR_CLUSTER}" != "1" ]] && _gres_arg=(--gres=gpu:1)
    _sbatch_run aic-tiny-test tiny-test "${remote_script}" \
        "${_sel[@]}" \
        "${_gres_arg[@]}" \
        --nodes=1 --ntasks=1 \
        --cpus-per-task="${AIC_TINY_CPUS}" --mem="${AIC_TINY_MEM}" \
        --time="${AIC_TINY_TIME}"
    log "tiny-test complete"
}

# --- reset-test: L1+L2 retrieval check on a GPU node -------------------------
# A thin sbatch wrapper around the local `make vllm-reset-test` target: loads the
# image if needed, picks a writable NVMe mount, then delegates.  Deliberately does
# NOT re-implement the compose bring-up or the pass/fail logic, so the batch path
# and the interactive path cannot drift apart.
cmd_reset_test() {
    _pick_compress
    local tarball; tarball="$(_tarball_path)"
    command -v sbatch >/dev/null 2>&1 || die "sbatch not found; cannot run the GPU reset-test job"
    [[ -r "${tarball}" ]] || die "tarball not found: ${tarball} (run 'build' first)"
    [[ -n "${HF_TOKEN:-}" || -r "${HF_TOKEN_FILE:-/nonexistent}" ]] ||
        die "reset-test needs HF_TOKEN (or HF_TOKEN_FILE); make vllm-reset-test gates on it"

    local _SBATCH_PARTITION="${AIC_TEST_PARTITION}"
    local _SBATCH_ACCOUNT="${AIC_TEST_ACCOUNT}"

    local _test_arch="${AIC_TEST_ARCH:-}"
    if [[ -z "${_test_arch}" ]]; then
        if [[ "${AIC_ROCM_ARCH}" != *";"* ]]; then
            _test_arch="${AIC_ROCM_ARCH}"
        else
            _test_arch="$(bash "${SCRIPT_DIR}/aic-test-arch.sh" 2>/dev/null || true)"
            [[ -n "${_test_arch}" ]] || _test_arch="${AIC_ROCM_ARCH}"
        fi
    fi

    # Monitoring needs the two locally-built exporter images.  They are never
    # pushed, so compose would try to pull them and fail; load them from their
    # tarballs on the node instead.  Missing tarballs are a hard error here rather
    # than a compose pull failure ten minutes in.
    local _nvme_tar="" _rdma_tar=""
    if [[ "${AIC_RESET_MONITORING}" == "1" ]]; then
        _nvme_tar="$(_exporter_tarball_path "${AIC_NVME_EXPORTER_IMAGE}")"
        _rdma_tar="$(_exporter_tarball_path "${AIC_RDMA_EXPORTER_IMAGE}")"
        log "monitoring: on  (Prometheus + exporters; TSDB kept under logs/<job-id>/prometheus)"
        for _t in "${_nvme_tar}" "${_rdma_tar}"; do
            [[ -r "${_t}" ]] || die "monitoring needs ${_t} (run 'make dist-build-exporters', or set AIC_RESET_MONITORING=0)"
        done
    else
        log "monitoring: off (AIC_RESET_MONITORING=0)"
    fi

    local -a _sel
    if [[ -n "${AIC_TEST_NODE:-}" ]]; then
        _sel=(--nodelist="${AIC_TEST_NODE}")
        log "reset-test on ${AIC_TEST_NODE} via sbatch (partition ${AIC_TEST_PARTITION})"
    else
        _sel=(--constraint="${AIC_TEST_CONSTRAINT}")
        log "reset-test via sbatch (partition ${AIC_TEST_PARTITION}, constraint ${AIC_TEST_CONSTRAINT})"
    fi
    log "image: ${AIC_IMAGE}  model: ${AIC_RESET_MODEL}  arch: ${_test_arch}"
    log "l1: 1GiB (forced by the make target)  l2: nixl_posix  slot: ${AIC_RESET_SLOT_SIZE}B  gpu-util: ${AIC_RESET_GPU_UTIL}"

    local remote_script
    remote_script="$(cat <<REMOTE
set -uo pipefail
command -v docker >/dev/null 2>&1 || { echo "\$(hostname): docker not found" >&2; exit 1; }
echo "[reset-test] host=\$(hostname) docker=\$(docker --version)"

_marker="/var/tmp/aic-loaded-\$(id -u)-\$(echo '${AIC_IMAGE}' | tr '/:' '__').mtime"
_tar_mtime="\$(stat -c %Y '${tarball}' 2>/dev/null || echo 0)"
_have_img="\$(docker images -q '${AIC_IMAGE}')"
_loaded_mtime="\$(cat "\${_marker}" 2>/dev/null || echo 0)"
if [ "${AIC_FORCE_LOAD:-0}" = "1" ] || [ -z "\${_have_img}" ] || [ "\${_tar_mtime}" -gt "\${_loaded_mtime}" ]; then
    echo "[reset-test] loading ${AIC_IMAGE} from ${tarball}"
    ${DECOMPRESS_CMD} '${tarball}' | docker load >/dev/null
    echo "\${_tar_mtime}" > "\${_marker}" 2>/dev/null || true
else
    echo "[reset-test] image up to date on \$(hostname) (id \${_have_img})"
fi

if [ '${AIC_RESET_MONITORING}' = "1" ]; then
    for _pair in '${AIC_NVME_EXPORTER_IMAGE}|${_nvme_tar}' '${AIC_RDMA_EXPORTER_IMAGE}|${_rdma_tar}'; do
        _img="\${_pair%%|*}"; _tar="\${_pair##*|}"
        if [ -n "\$(docker images -q "\${_img}")" ]; then
            echo "[reset-test] \${_img} already present"
        else
            echo "[reset-test] loading \${_img} from \${_tar}"
            ${DECOMPRESS_CMD} "\${_tar}" | docker load >/dev/null ||
                { echo "[reset-test] FAIL: could not load \${_img}" >&2; exit 1; }
        fi
    done
fi

cd '${AIC_DAY_DIR}'
# shellcheck source=/dev/null
source '${AIC_DAY_DIR}/monitoring/monitoring-lib.sh'
ensure_compose || { echo "[reset-test] docker compose unavailable and could not be installed" >&2; exit 1; }

# Pick the L2 pool directory.  Prefer an explicit override, else the first
# writable NVMe mount on this node.
_nvme='${AIC_RESET_NVME_DATA}'
if [ -z "\${_nvme}" ]; then
    for _d in /mnt/nixl-nvme-0 /mnt/nixl-nvme-* /mnt/lmcache-nvme /mnt/nvme*; do
        if [ -d "\${_d}" ] && [ -w "\${_d}" ]; then _nvme="\${_d}/aic-reset-test"; break; fi
    done
fi
if [ -z "\${_nvme}" ]; then
    echo "[reset-test] FAIL: no writable NVMe mount found; set AIC_RESET_NVME_DATA" >&2
    exit 1
fi
_nfs="\${_nvme%/*}/aic-reset-nfs"
# NFS_DATA is unused with the POSIX L2 backend, but _prep_dirs mkdir -p's it
# unconditionally and its /mnt/lmcache-nfs default needs root.
mkdir -p "\${_nvme}" "\${_nfs}" || { echo "[reset-test] FAIL: cannot create \${_nvme} / \${_nfs}" >&2; exit 1; }
echo "[reset-test] L2 pool dir: \${_nvme} (\$(df -h --output=avail "\${_nvme}" | tail -1 | tr -d ' ') avail)"

export HF_TOKEN='${HF_TOKEN:-}'
export HF_TOKEN_FILE='${HF_TOKEN_FILE:-}'
# Cache-backed launch: vLLM joins lmcache's PID ns so cross-container HIP IPC works.
export VLLM_PID_MODE="\${VLLM_PID_MODE:-service:lmcache}"
# The test script asks for AIC_TEST_MODEL; vLLM serves VLLM_MODEL.  Keep them equal.
export AIC_TEST_MODEL='${AIC_RESET_MODEL}'
export AIC_TEST_FLOOD='${AIC_RESET_FLOOD}'
# compose defaults these to :local while this script builds/saves them as :latest
# (see AIC_NVME_EXPORTER_IMAGE above) -- pass the tags we actually loaded, or
# compose looks for an image that is not here and tries to pull it.
export AIC_NVME_EXPORTER_IMAGE='${AIC_NVME_EXPORTER_IMAGE}'
export AIC_RDMA_EXPORTER_IMAGE='${AIC_RDMA_EXPORTER_IMAGE}'

compose() { docker compose -f '${AIC_DAY_DIR}/docker/docker-compose.yml' "\$@"; }
cleanup() {
    local svc c
    for svc in vllm lmcache; do
        timeout 30 compose logs --no-color --no-log-prefix "\$svc" > "\${_logdir}/reset-\${svc}.log" 2>&1 || true
    done
    pkill -9 -f 'vllm.entrypoints.openai' 2>/dev/null || true
    pkill -9 -f 'EngineCore'              2>/dev/null || true
    pkill -9 -f 'lmcache server'          2>/dev/null || true
    sleep 2
    timeout 60 compose --profile cache --profile monitoring down --remove-orphans --timeout 5 >/dev/null 2>&1 || true
    for c in aic-vllm-gpu0 aic-lmcache aic-client aic-prometheus aic-grafana; do
        timeout 30 docker rm -f "\$c" >/dev/null 2>&1 || true
    done
    rm -rf "\${_nvme}" "\${_nfs}" 2>/dev/null || true
}
trap cleanup EXIT

if [ '${AIC_RESET_MONITORING}' = "1" ]; then
    mkdir -p "\${_logdir}/prometheus"
    echo "[reset-test] Prometheus TSDB -> \${_logdir}/prometheus"
fi
echo "[reset-test] running make vllm-reset-test ..."
_out="\${_logdir}/reset-make.out"
set -o pipefail
make -C '${AIC_DAY_DIR}' vllm-reset-test \
    IMAGE_REF='${AIC_IMAGE}' \
    ROCM_ARCH='${_test_arch}' \
    VLLM_MODEL='${AIC_RESET_MODEL}' \
    NVME_DATA="\${_nvme}" \
    NFS_DATA="\${_nfs}" \
    HF_HOME='${HF_HOME}' \
    LOG="\${_logdir}" \
    AIC_METRICS_DIR="\${_logdir}/prometheus" \
    VLM_GPU_MEMORY_UTILIZATION='${AIC_RESET_GPU_UTIL}' \
    VLM_MAX_MODEL_LEN='${AIC_RESET_MAX_LEN}' \
    LMCACHE_NIXL_POSIX_SLOT_SIZE='${AIC_RESET_SLOT_SIZE}' \
    AIC_RESET_MONITORING='${AIC_RESET_MONITORING}' 2>&1 | tee "\${_out}"
_rc=\${PIPESTATUS[0]}
if [ "\${_rc}" -eq 0 ]; then
    echo "[reset-test] PASS: L1 and L2 retrieval both verified"
else
    # 2/3/4 are vllm_reset_test.py verdicts -- but make itself also exits 2 on any
    # recipe failure (a mkdir in _prep_dirs, compose up, the health wait).  Only read
    # them as cache verdicts once the python test has actually produced output,
    # otherwise a setup failure gets reported as "no L1 hits".
    # Read the verdict off the test's OWN output lines.  vllm_reset_test.py exits
    # 2/3/4 to mean no-L1 / no-L2 / neither, but make collapses every recipe failure
    # to its own exit 2, so \${_rc} cannot distinguish them -- mapping it reported
    # "no L1 hits" for a run where L1 passed and L2 failed.
    if grep -qE "(L1|L2) retrieval (PASS|FAIL)" "\${_out}" 2>/dev/null; then
        grep -E "(L1|L2) retrieval (PASS|FAIL)" "\${_out}" | sed 's/^/[reset-test]   /' >&2
        grep -q "L2 retrieval FAIL" "\${_out}" &&
            echo "[reset-test] hint: no L2 hits -- flood may be too small to overflow L1 (raise AIC_RESET_FLOOD), or the KV chunk exceeds LMCACHE_NIXL_POSIX_SLOT_SIZE" >&2
        echo "[reset-test] FAIL: see the verdict lines above" >&2
    else
        echo "[reset-test] FAIL: setup failed before the cache test ran (make exited \${_rc})" >&2
        echo "[reset-test] last 20 lines of \${_out}:" >&2
        tail -20 "\${_out}" >&2 2>/dev/null || true
    fi
fi
exit "\${_rc}"
REMOTE
)"

    local -a _gres_arg=(); [[ "${AIC_SPUR_CLUSTER}" != "1" ]] && _gres_arg=(--gres=gpu:1)
    _sbatch_run aic-reset-test reset-test "${remote_script}" \
        "${_sel[@]}" \
        "${_gres_arg[@]}" \
        --nodes=1 --ntasks=1 \
        --cpus-per-task="${AIC_RESET_CPUS}" --mem="${AIC_RESET_MEM}" \
        --time="${AIC_RESET_TIME}"
    log "reset-test complete"
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
        tiny-test)       cmd_tiny_test ;;
        reset-test)      cmd_reset_test ;;
        all)             cmd_build; cmd_build_exporters; cmd_load ;;
        -h|--help|help)
            sed -n '2,70p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            ;;
        *) die "unknown command '${sub}' (use: build | build-exporters | load | push | test | tiny-test | all | help)" ;;
    esac
}

main "$@"
