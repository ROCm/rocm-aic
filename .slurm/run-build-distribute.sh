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
#   build-emulate
#           Build the CPU-only EMULATION image (Dockerfile `emulate` stage +
#           VLLM_TARGET_DEVICE=empty): vLLM + the llm-emu plugin with no GPU
#           kernels compiled at all, so it takes minutes rather than hours.
#           Tagged AIC_EMULATE_IMAGE (default rocm-aic:7.14-emulate) with its
#           own tarball, so it never overwrites the real GPU image.
#   emulate-test
#           End-to-end serve check of that image on a CPU-ONLY node: brings up
#           the compose `emulate` profile, asserts a non-empty completion, and
#           asserts the llm-emu hook (not a real forward pass) produced it
#   profile-capture
#           Capture an AMD profile pack: runs the FULL image on a GPU node with
#           VLLM_EMULATOR_TRACE_STEP_CYCLE=1 (a REAL serve, real weights, real
#           kernels), drives a `vllm bench serve` sweep over
#           (input len x concurrency), then builds a profile pack from the step
#           trace.  Pack + trace + real-hardware benchmark results land in
#           AIC_CAPTURE_DIR.  Needs AIC_ROCM_ARCH set to the arch the image
#           tarball was built for (e.g. AIC_ROCM_ARCH=gfx942).
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
#   all     build, build-exporters, then load   (default)
#
# Key environment:
#   AIC_ROCM_ARCH        gfx arch(es) baked in; ';'-list   (default: all vLLM archs)
#   AIC_IMAGE            image name:tag                    (default: rocm-aic:latest)
#   AIC_IMAGE_DIR        shared dir for the tarball        (default: /scratch/$USER/images)
#   HF_HOME              persistent Hugging Face cache used by tiny-test
#                        (default: <AIC_IMAGE_DIR>/tiny-hf)
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

# --- Defaults ----------------------------------------------------------------
# Multi-arch by default: every gfx the vLLM ROCm wheel ships kernels for.
# hipFile + the LMCache HIP extension are compiled for all of these (cmake and
# PYTORCH_ROCM_ARCH both accept ';'-lists); NIXL AIS_MT is host-only, so arch-
# independent.  NOTE: the RDNA entries (gfx11xx/gfx12xx) have no NVMe-DMA /
# Infinity-Storage hardware -- if hipFile fails to build for them, narrow this
# to the CDNA set "gfx90a;gfx942;gfx950".  Override via AIC_ROCM_ARCH.
AIC_ROCM_ARCH="${AIC_ROCM_ARCH:-gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201}"
AIC_UCX_FAST="${AIC_UCX_FAST:-}"
AIC_IMAGE="${AIC_IMAGE:-${IMAGE_NAME:-rocm-aic}:${IMAGE_TAG:-7.14-latest}}"
AIC_IMAGE_DIR="${AIC_IMAGE_DIR:-/scratch/${USER}/images}"
AIC_SPUR_CLUSTER="${AIC_SPUR_CLUSTER:-0}"

# --- Docker build target / vLLM build device --------------------------------
# Empty AIC_BUILD_TARGET = the Dockerfile's default (full runtime image).  The
# `build-emulate` command below overrides both to produce the CPU-only
# emulation image.  Set directly for a one-off custom target.
AIC_BUILD_TARGET="${AIC_BUILD_TARGET:-}"
AIC_VLLM_TARGET_DEVICE="${AIC_VLLM_TARGET_DEVICE:-}"

# --- build-emulate / emulate-test defaults ----------------------------------
# The emulation image stops at the Dockerfile's `emulate` stage (vLLM + the
# llm-emu plugin, no LMCache HIP ext / NIXL / hsa-snoop) and builds vLLM with
# VLLM_TARGET_DEVICE=empty, so NO GPU kernels are compiled at all -- minutes,
# not hours.  The arch below only picks torch's device extra from AMD's wheel
# index; nothing in this image runs on a GPU.
AIC_EMULATE_IMAGE="${AIC_EMULATE_IMAGE:-${IMAGE_NAME:-rocm-aic}:${IMAGE_TAG_EMULATE:-7.14-emulate}}"
AIC_EMULATE_ARCH="${AIC_EMULATE_ARCH:-gfx942}"
AIC_EMULATE_VLLM_DEVICE="${AIC_EMULATE_VLLM_DEVICE:-empty}"
AIC_EMULATE_MODEL="${AIC_EMULATE_MODEL:-Qwen/Qwen3-8B}"
AIC_EMULATE_PROFILE_PACK="${AIC_EMULATE_PROFILE_PACK:-/opt/llm-emu/profiles/MI300X-Qwen3-8B.json}"
AIC_EMULATE_TIME="${AIC_EMULATE_TIME:-00:30:00}"
AIC_EMULATE_CPUS="${AIC_EMULATE_CPUS:-8}"
AIC_EMULATE_MEM="${AIC_EMULATE_MEM:-32G}"
AIC_EMULATE_READY_TIMEOUT="${AIC_EMULATE_READY_TIMEOUT:-96}"   # x5s = up to 8 min
# Host dir bind-mounted at /profiles in the emulator container, for packs
# captured on real hardware (see profile-capture).  Unset = the image's
# example pack only.
AIC_EMULATE_PACK_HOST="${AIC_EMULATE_PACK_HOST:-}"

# --- profile-capture defaults (real GPU -> profile pack) ---------------------
# A REAL serve on a REAL GPU with VLLM_EMULATOR_TRACE_STEP_CYCLE=1, driven by a
# `vllm bench serve` sweep, then turned into a profile pack the emulator can
# replay.  This is the only way to get AMD latencies into emulation: the pack
# shipped in the image is an NVIDIA A40 capture.
AIC_CAPTURE_MODEL="${AIC_CAPTURE_MODEL:-Qwen/Qwen3-8B}"
AIC_CAPTURE_DIR="${AIC_CAPTURE_DIR:-${AIC_IMAGE_DIR}/profiles}"
AIC_CAPTURE_HF_HOME="${AIC_CAPTURE_HF_HOME:-${AIC_IMAGE_DIR}/capture-hf}"
# NOTE: GFX942 matches BOTH MI300X and MI300A (the APU) -- they are different
# machines with different memory systems, so pin AIC_CAPTURE_NODE (or add a
# site/feature term) when the pack is meant to describe one of them.
AIC_CAPTURE_CONSTRAINT="${AIC_CAPTURE_CONSTRAINT-GFX942}"
AIC_CAPTURE_GPU="${AIC_CAPTURE_GPU:-0}"
AIC_CAPTURE_TIME="${AIC_CAPTURE_TIME:-01:30:00}"
AIC_CAPTURE_CPUS="${AIC_CAPTURE_CPUS:-32}"
AIC_CAPTURE_MEM="${AIC_CAPTURE_MEM:-128G}"
AIC_CAPTURE_READY_TIMEOUT="${AIC_CAPTURE_READY_TIMEOUT:-180}"  # x5s = up to 15 min (weights download)
AIC_CAPTURE_MAX_MODEL_LEN="${AIC_CAPTURE_MAX_MODEL_LEN:-8192}"
AIC_CAPTURE_MAX_BATCHED_TOKENS="${AIC_CAPTURE_MAX_BATCHED_TOKENS:-4096}"
AIC_CAPTURE_GPU_UTIL="${AIC_CAPTURE_GPU_UTIL:-0.85}"
# Extra serve flags for the capture.  Prefix caching is OFF by default: with it
# on, a sweep that revisits an input length serves the second pass out of the
# prefix cache, so the trace mixes cold and warm prefills AND the real-hardware
# reference numbers stop being comparable to a cold emulated replay.  Capture
# the compute cost; let the emulator's own (unmodified) prefix cache decide what
# is reused at serve time.  Whatever is set here must also be set for
# emulate-validate -- a pack is only valid for its capture configuration.
AIC_CAPTURE_EXTRA_ARGS="${AIC_CAPTURE_EXTRA_ARGS---no-enable-prefix-caching}"
# Sweep points as "isl,osl,concurrency,num_prompts".  Chosen to populate both
# axes the oracle buckets on: total scheduled tokens (isl drives the prefill /
# chunked-prefill cells) and concurrency (drives the decode cells).
AIC_CAPTURE_SWEEP="${AIC_CAPTURE_SWEEP:-256,64,1,16 256,64,8,64 1024,128,1,12 1024,128,4,32 1024,128,16,64 1024,128,64,128 4096,128,1,8 4096,128,8,32 4096,128,32,64}"

# When running on SPUR, default the partition to amd-spur (the only partition)
# and clear the build/test constraints (SPUR nodes have no MARKHAM/CPUONLY/GFX942
# feature labels; node selection is done by partition or explicit --nodelist).
if [[ "${AIC_SPUR_CLUSTER}" == "1" ]]; then
    # Only SPUR needs a controller address -- every `--controller=` below is
    # inside an AIC_SPUR_CLUSTER=1 branch.  Demanding it unconditionally (as
    # this once did, one line after defaulting AIC_SPUR_CLUSTER to 0) aborted
    # every ordinary defq build on a host that has no SPUR controller at all.
    AIC_SPUR_CONTROLLER="${AIC_SPUR_CONTROLLER:-${SPUR_CONTROLLER_ADDR:?set SPUR_CONTROLLER_ADDR or AIC_SPUR_CONTROLLER before using AIC_SPUR_CLUSTER=1}}"
    AIC_BUILD_PARTITION="${AIC_BUILD_PARTITION:-amd-spur}"
    AIC_IMAGE_DIR="${AIC_IMAGE_DIR:-${AIC_SHARED_NFS:-/shared_nfs}/${USER}/images}"
    # Use ${VAR-default} (not ${VAR:-default}) so an explicitly set empty string
    # ("AIC_BUILD_CONSTRAINT=") is honoured as "no constraint".
    AIC_BUILD_CONSTRAINT="${AIC_BUILD_CONSTRAINT-}"
    AIC_TEST_CONSTRAINT="${AIC_TEST_CONSTRAINT-}"
else
    # Never read on this path; defined only so `set -u` stays satisfied.
    AIC_SPUR_CONTROLLER=""
    AIC_BUILD_PARTITION="${AIC_BUILD_PARTITION:-defq}"
    AIC_BUILD_CONSTRAINT="${AIC_BUILD_CONSTRAINT:-CPUONLY}"
    AIC_TEST_CONSTRAINT="${AIC_TEST_CONSTRAINT:-GFX942&NVME}"
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
                | awk -v id="${jobid}" '$1==id{split($2,a,":"); print a[1]; exit}')"
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
    [[ -n "${AIC_BUILD_TARGET}" ]] && log "target     : ${AIC_BUILD_TARGET} (docker --target)"
    [[ -n "${AIC_VLLM_TARGET_DEVICE}" ]] && \
        log "vllm device: ${AIC_VLLM_TARGET_DEVICE} (VLLM_TARGET_DEVICE build-arg)"
    if [[ -n "${AIC_TLS_CERT}" ]]; then
        [[ -r "${AIC_TLS_CERT}" ]] || die "AIC_TLS_CERT not readable: ${AIC_TLS_CERT}"
        log "tls cert   : ${AIC_TLS_CERT} (BuildKit secret)"
    fi

    # BuildKit secret arg for the corporate CA, only when a cert was provided.
    local _secret_arg=""
    [[ -n "${AIC_TLS_CERT}" ]] && _secret_arg="--secret id=tls_cert,src=${AIC_TLS_CERT}"

    # Optional Dockerfile stage + vLLM build device (see the `emulate` stage).
    local _target_arg="" _vllm_device_arg=""
    [[ -n "${AIC_BUILD_TARGET}" ]] && _target_arg="--target ${AIC_BUILD_TARGET}"
    [[ -n "${AIC_VLLM_TARGET_DEVICE}" ]] && \
        _vllm_device_arg="--build-arg VLLM_TARGET_DEVICE=${AIC_VLLM_TARGET_DEVICE}"

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
            # Per-target subdir as well as per-arch: an `emulate` build shares
            # no post-torch layers with a full build, and keeping their caches
            # apart stops one from evicting/locking the other's entries.
            local _cdir
            _cdir="${AIC_CACHE_DIR%/}/$(_arch_tag)${AIC_BUILD_TARGET:+-${AIC_BUILD_TARGET}}"
            log "build cache: local dir ${_cdir} (mode ${AIC_CACHE_MODE}, builder ${AIC_BUILDX_BUILDER})"
            _cache_args="--cache-from type=local,src=${_cdir} --cache-to type=local,dest=${_cdir},mode=${AIC_CACHE_MODE},ignore-error=true"
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
echo "[build] disk after prune: \$(df -h / | awk 'NR==2{print \$3\" free / \"\$2\" total (\"\$5\" used)\"}')"
tmp="${tarball}.partial.\$\$"
docker buildx build --builder ${AIC_BUILDX_BUILDER} --progress=plain --output type=docker,dest=- \
    --build-arg ROCM_ARCH="${AIC_ROCM_ARCH}" \
    --build-arg AIC_UCX_FAST="${AIC_UCX_FAST}" \
    ${_vllm_device_arg} \
    ${_target_arg} \
    ${_secret_arg} \
    ${_cache_args} \
    -f "${AIC_DAY_DIR}/docker/Dockerfile" \
    -t "${AIC_IMAGE}" \
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
${_build_program} \
    --build-arg ROCM_ARCH="${AIC_ROCM_ARCH}" \
    --build-arg AIC_UCX_FAST="${AIC_UCX_FAST}" \
    ${_vllm_device_arg} \
    ${_target_arg} \
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

# --- build-emulate: CPU-only emulation image (no GPU kernels compiled) --------
# Same machinery as `build`, pointed at the Dockerfile's `emulate` stage with
# VLLM_TARGET_DEVICE=empty.  That stage stops after vLLM + the llm-emu plugin,
# and `empty` is upstream vLLM's no-extension build -- so this compiles no HIP
# kernels, no LMCache HIP extension, no NIXL and no hsa-snoop.  The result runs
# the full vLLM scheduler/HTTP stack on a CPU-only node with the forward pass
# replaced by a profile-pack latency draw.  Tagged separately from the GPU image
# (AIC_EMULATE_IMAGE), so its tarball never overwrites the real one.
_use_emulate_image() {
    AIC_IMAGE="${AIC_EMULATE_IMAGE}"
    AIC_ROCM_ARCH="${AIC_EMULATE_ARCH}"
}

cmd_build_emulate() {
    _use_emulate_image
    AIC_BUILD_TARGET="emulate"
    AIC_VLLM_TARGET_DEVICE="${AIC_EMULATE_VLLM_DEVICE}"
    log "build-emulate: emulation-only image, no GPU kernels compiled"
    cmd_build
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

    local -a _sel
    if [[ -n "${AIC_TEST_NODE:-}" ]]; then
        _sel=(--nodelist="${AIC_TEST_NODE}")
        log "tiny-test on ${AIC_TEST_NODE} via sbatch (partition ${AIC_BUILD_PARTITION})"
    else
        _sel=(--constraint="${AIC_TEST_CONSTRAINT}")
        log "tiny-test via sbatch (partition ${AIC_BUILD_PARTITION}, constraint ${AIC_TEST_CONSTRAINT})"
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
export IMAGE_NAME='${AIC_IMAGE}'
export ROCM_ARCH='${AIC_ROCM_ARCH}'
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
export VLLM_PID_MODE=service:lmcache
export KV_TRANSFER_ARG="--kv-transfer-config '{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://localhost\",\"lmcache.mp.port\":6555}}'"
mkdir -p "\${HF_HOME}" /tmp/aic-tiny-nvme /tmp/aic-tiny-nfs

compose() { docker compose -f '${AIC_DAY_DIR}/docker/docker-compose.yml' "\$@"; }
cleanup() {
    local svc c
    for svc in vllm lmcache; do
        # \`timeout\` runs a program, not the compose() shell function above --
        # calling it with "compose" silently ran /usr/bin/compose (mailcap) and
        # wrote its error message here instead of the service log.
        timeout 30 docker compose -f '${AIC_DAY_DIR}/docker/docker-compose.yml' \
            logs --no-color --no-log-prefix "\$svc" > "\${_logdir}/tiny-\${svc}.log" 2>&1 || true
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
ready=0
for _i in \$(seq 1 ${AIC_TINY_READY_TIMEOUT}); do
    if curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1; then ready=1; break; fi
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
resp="\$(curl -fsS http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"${AIC_TINY_MODEL}","messages":[{"role":"user","content":"Reply with the single word: pong"}],"max_tokens":16,"temperature":0}' 2>&1)" || {
    echo "[tiny-test] FAIL: completion request failed: \${resp}" >&2; exit 1; }
echo "[tiny-test] response: \${resp}"
if grep -qE '"content"[[:space:]]*:[[:space:]]*"[^"]+' <<<"\${resp}"; then
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

# --- emulate-test: end-to-end serve check of the emulation image on a CPU node -
# Loads the `emulate` tarball on a CPU-only node, brings up the compose `emulate`
# profile (vLLM serving with the llm-emu executor hook), and asserts:
#   1. the OpenAI endpoint comes up and answers a chat completion non-empty;
#   2. the emulator hook actually engaged  -- otherwise a broken plugin would
#      quietly fall back to real execution and this test would pass for the
#      wrong reason (on a CPU node it would instead crash, but on a GPU node it
#      would silently measure real inference);
#   3. no model weights were loaded (the emulator stubs load_model).
cmd_emulate_test() {
    _use_emulate_image
    _pick_compress
    local tarball; tarball="$(_tarball_path)"
    command -v sbatch >/dev/null 2>&1 || die "sbatch not found; cannot run the emulate-test job"
    [[ -r "${tarball}" ]] || die "tarball not found: ${tarball} (run 'build-emulate' first)"

    # CPU-only node: no --gres, no GPU constraint.  AIC_EMULATE_TEST_CONSTRAINT
    # defaults to the build constraint, which already selects CPU-only nodes.
    local _constraint="${AIC_EMULATE_TEST_CONSTRAINT-${AIC_BUILD_CONSTRAINT}}"
    local -a _sel=()
    if [[ -n "${AIC_EMULATE_TEST_NODE:-}" ]]; then
        _sel=(--nodelist="${AIC_EMULATE_TEST_NODE}")
        log "emulate-test on ${AIC_EMULATE_TEST_NODE} via sbatch (partition ${AIC_BUILD_PARTITION})"
    else
        [[ -n "${_constraint}" ]] && _sel=(--constraint="${_constraint}")
        log "emulate-test via sbatch (partition ${AIC_BUILD_PARTITION}, constraint ${_constraint:-<none>})"
    fi
    log "image: ${AIC_IMAGE}  model: ${AIC_EMULATE_MODEL}  pack: ${AIC_EMULATE_PROFILE_PACK}"

    local remote_script
    remote_script="$(cat <<REMOTE
set -uo pipefail
command -v docker >/dev/null 2>&1 || { echo "\$(hostname): docker not found" >&2; exit 1; }
echo "[emulate-test] host=\$(hostname) docker=\$(docker --version)"
echo "[emulate-test] GPU devices present: \$(ls /dev/kfd /dev/dri 2>/dev/null | tr '\n' ' ' || echo none)"

# Load the image from the shared tarball only when needed (same marker logic as
# smoke-test): reload when forced, absent, or the tarball is newer.
_marker="/var/tmp/aic-loaded-\$(id -u)-\$(echo '${AIC_IMAGE}' | tr '/:' '__').mtime"
_tar_mtime="\$(stat -c %Y '${tarball}' 2>/dev/null || echo 0)"
_have_img="\$(docker images -q '${AIC_IMAGE}')"
_loaded_mtime="\$(cat "\${_marker}" 2>/dev/null || echo 0)"
if [ "${AIC_FORCE_LOAD:-0}" = "1" ] || [ -z "\${_have_img}" ] || [ "\${_tar_mtime}" -gt "\${_loaded_mtime}" ]; then
    echo "[emulate-test] loading ${AIC_IMAGE} from ${tarball}"
    ${DECOMPRESS_CMD} '${tarball}' | docker load >/dev/null
    echo "\${_tar_mtime}" > "\${_marker}" 2>/dev/null || true
else
    echo "[emulate-test] image up to date on \$(hostname) (id \${_have_img})"
fi

cd '${AIC_DAY_DIR}'
# shellcheck source=/dev/null
source '${AIC_DAY_DIR}/monitoring/monitoring-lib.sh'
ensure_compose || { echo "[emulate-test] docker compose unavailable and could not be installed" >&2; exit 1; }

export IMAGE_NAME='${AIC_IMAGE}'
export VLLM_MODEL='${AIC_EMULATE_MODEL}'
export VLLM_EMULATOR_PROFILE_PACK='${AIC_EMULATE_PROFILE_PACK}'
# Host dir of real-hardware packs, bind-mounted at /profiles by the compose
# service.  Set AIC_EMULATE_PACK_HOST + point AIC_EMULATE_PROFILE_PACK at
# /profiles/<pack>.json to replay an MI300X/MI355X capture.
[ -n '${AIC_EMULATE_PACK_HOST}' ] && export EMU_PROFILE_PACK_HOST='${AIC_EMULATE_PACK_HOST}'
export HF_HOME='${AIC_TINY_HF_HOME}'
export HF_TOKEN='${HF_TOKEN:-}'
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export LOG="\${_logdir}"
mkdir -p "\${HF_HOME}" "\${_logdir}/vllm-emulator"

# NB: \`timeout\` cannot run a shell function, so the calls below spell out
# \`docker compose\` rather than wrapping the compose() helper.
compose() { docker compose -f '${AIC_DAY_DIR}/docker/docker-compose.yml' "\$@"; }
cleanup() {
    timeout 60 docker compose -f '${AIC_DAY_DIR}/docker/docker-compose.yml' \
        --profile emulate logs --no-color --no-log-prefix vllm-emulator \
        > "\${_logdir}/emulate-vllm.log" 2>&1 || true
    timeout 60 docker compose -f '${AIC_DAY_DIR}/docker/docker-compose.yml' \
        --profile emulate down --remove-orphans --timeout 10 >/dev/null 2>&1 || true
    timeout 30 docker rm -f aic-vllm-emulator >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[emulate-test] bringing up the emulate compose profile (model=${AIC_EMULATE_MODEL}) ..."
if ! compose --profile emulate up -d vllm-emulator; then
    echo "[emulate-test] FAIL: compose up failed" >&2
    compose --profile emulate logs --tail 60 --no-color vllm-emulator 2>&1 | sed 's/^/  [emu] /'
    exit 1
fi

ready=0
for _i in \$(seq 1 ${AIC_EMULATE_READY_TIMEOUT}); do
    if curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1; then ready=1; break; fi
    # Bail out early if the container died rather than waiting the full timeout.
    if [ -z "\$(docker ps -q -f name=aic-vllm-emulator)" ]; then
        echo "[emulate-test] FAIL: the emulator container exited during startup" >&2
        compose --profile emulate logs --tail 120 --no-color vllm-emulator 2>&1 | sed 's/^/  [emu] /'
        exit 1
    fi
    sleep 5
done
if [ "\${ready}" != "1" ]; then
    echo "[emulate-test] FAIL: the emulator endpoint never became ready" >&2
    compose --profile emulate logs --tail 120 --no-color vllm-emulator 2>&1 | sed 's/^/  [emu] /'
    exit 1
fi
echo "[emulate-test] endpoint ready; sending one chat completion ..."

# The container maps no /dev/kfd or /dev/dri (see the compose service), so it
# has no GPU access at all whatever the host has -- assert that, otherwise the
# "runs without a GPU" claim rests on the node happening to have none.
if docker exec aic-vllm-emulator test -e /dev/kfd 2>/dev/null; then
    kfd_in_container=yes
else
    kfd_in_container=no
fi
echo "[emulate-test] /dev/kfd inside the container: \${kfd_in_container}"

resp="\$(curl -fsS http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"${AIC_EMULATE_MODEL}","messages":[{"role":"user","content":"Reply with the single word: pong"}],"max_tokens":16,"temperature":0}' 2>&1)" || {
    echo "[emulate-test] FAIL: completion request failed: \${resp}" >&2; exit 1; }
echo "[emulate-test] response: \${resp}"

rc=0
if [ "\${kfd_in_container}" = "no" ]; then
    echo "[emulate-test] OK: the emulator container has no GPU device at all"
else
    echo "[emulate-test] FAIL: /dev/kfd is visible inside the emulator container" >&2; rc=1
fi
# Assert on TOKENS, not text.  The emulator emits one fixed filler token id per
# step (100 -> a lone byte-level BPE token in Qwen's vocab), so the decoded
# string is legitimately empty; what the test must prove is that generated
# tokens flowed all the way through vLLM's real sampling/detokenize/HTTP path.
completion_tokens="\$(printf '%s' "\${resp}" | grep -oE '"completion_tokens"[[:space:]]*:[[:space:]]*[0-9]+' | grep -oE '[0-9]+\$')"
if [ -n "\${completion_tokens}" ] && [ "\${completion_tokens}" -gt 0 ]; then
    echo "[emulate-test] OK: emulated engine generated \${completion_tokens} tokens" \
        "(decoded text is meaningless by design -- filler token)"
else
    echo "[emulate-test] FAIL: no tokens generated (completion_tokens=\${completion_tokens:-<absent>})" >&2; rc=1
fi

# Prove the emulator -- not a real forward pass -- produced that output.
# Grep a FILE, not a pipe: \`grep -q\` exits on the first match, the writer then
# dies of SIGPIPE, and \`set -o pipefail\` reports the whole pipeline as failed --
# an intermittent false negative that depends on how big the log is.
logfile="\${_logdir}/emulate-checks.log"
compose --profile emulate logs --no-color --no-log-prefix vllm-emulator \
    > "\${logfile}" 2>&1 || true
if grep -q '\[ExecutorEmulatorHook\] Enabled' "\${logfile}"; then
    echo "[emulate-test] OK: executor hook active ->" \
        "\$(grep -m1 '\[ExecutorEmulatorHook\] Enabled' "\${logfile}")"
else
    echo "[emulate-test] FAIL: the llm-emu executor hook never activated" >&2; rc=1
fi
if grep -q 'load_model SKIPPED' "\${logfile}"; then
    echo "[emulate-test] OK: model weights were never loaded (emulator stub)"
else
    echo "[emulate-test] WARN: no 'load_model SKIPPED' line; weights may have been loaded"
fi
if grep -q '\[ExecutorHook\] step=' "\${logfile}"; then
    echo "[emulate-test] OK: steps served from the profile pack ->" \
        "\$(grep -m1 '\[ExecutorHook\] step=' "\${logfile}")"
else
    echo "[emulate-test] FAIL: no emulated step was recorded" >&2; rc=1
fi

[ "\${rc}" -eq 0 ] && echo "[emulate-test] ALL CHECKS PASSED" || echo "[emulate-test] SOME CHECKS FAILED"
exit \${rc}
REMOTE
)"

    _sbatch_run aic-emulate-test emulate-test "${remote_script}" \
        "${_sel[@]}" \
        --nodes=1 --ntasks=1 \
        --cpus-per-task="${AIC_EMULATE_CPUS}" --mem="${AIC_EMULATE_MEM}" \
        --time="${AIC_EMULATE_TIME}"
    log "emulate-test complete"
}

# --- emulate-validate: replay a captured pack and diff against real hardware --
# The only question that matters about a profile pack is "does the emulator
# reproduce the machine it was captured from?".  This runs the SAME
# `vllm bench serve` points the capture ran, against the emulator on a CPU-only
# node, and prints real-vs-emulated TTFT / TPOT / output-throughput deltas.
cmd_emulate_validate() {
    _use_emulate_image
    _pick_compress
    local tarball; tarball="$(_tarball_path)"
    command -v sbatch >/dev/null 2>&1 || die "sbatch not found; cannot run the validate job"
    [[ -r "${tarball}" ]] || die "tarball not found: ${tarball} (run 'build-emulate' first)"
    [[ -n "${AIC_VALIDATE_PACK:-}" ]] || die "set AIC_VALIDATE_PACK=/path/to/pack.json (from 'profile-capture')"
    [[ -r "${AIC_VALIDATE_PACK}" ]] || die "pack not readable: ${AIC_VALIDATE_PACK}"

    local pack_dir pack_file
    pack_dir="$(cd "$(dirname "${AIC_VALIDATE_PACK}")" && pwd)"
    pack_file="$(basename "${AIC_VALIDATE_PACK}")"
    local real_dir="${AIC_VALIDATE_REAL_DIR:-${pack_dir}/bench}"
    local model="${AIC_VALIDATE_MODEL:-${AIC_EMULATE_MODEL}}"
    local sweep="${AIC_VALIDATE_SWEEP:-1024,128,1,12 1024,128,16,64 4096,128,8,32}"
    local max_len="${AIC_VALIDATE_MAX_MODEL_LEN:-8192}"
    local max_bt="${AIC_VALIDATE_MAX_BATCHED_TOKENS:-4096}"
    # Must match the capture's serve flags (see AIC_CAPTURE_EXTRA_ARGS): a pack
    # only describes the configuration it was captured under, and a warm-vs-cold
    # prefix cache alone changes TTFT by an order of magnitude.
    local extra_args="${AIC_VALIDATE_EXTRA_ARGS---no-enable-prefix-caching}"
    # Oracle neighbour selection for the replay: 1 = nearest cell, `auto` =
    # adaptive-K Shepard pooling, which smooths thinly-sampled cells.
    local oracle_k="${AIC_VALIDATE_ORACLE_K:-1}"

    local -a _sel=()
    if [[ -n "${AIC_EMULATE_TEST_NODE:-}" ]]; then
        _sel=(--nodelist="${AIC_EMULATE_TEST_NODE}")
    else
        local _c="${AIC_EMULATE_TEST_CONSTRAINT-${AIC_BUILD_CONSTRAINT}}"
        [[ -n "${_c}" ]] && _sel=(--constraint="${_c}")
    fi
    log "emulate-validate: pack ${pack_dir}/${pack_file}"
    log "model ${model}  sweep ${sweep}  real results ${real_dir}"

    local remote_script
    remote_script="$(cat <<REMOTE
set -uo pipefail
echo "[validate] host=\$(hostname)"
_marker="/var/tmp/aic-loaded-\$(id -u)-\$(echo '${AIC_IMAGE}' | tr '/:' '__').mtime"
_tar_mtime="\$(stat -c %Y '${tarball}' 2>/dev/null || echo 0)"
_have_img="\$(docker images -q '${AIC_IMAGE}')"
_loaded_mtime="\$(cat "\${_marker}" 2>/dev/null || echo 0)"
if [ "${AIC_FORCE_LOAD:-0}" = "1" ] || [ -z "\${_have_img}" ] || [ "\${_tar_mtime}" -gt "\${_loaded_mtime}" ]; then
    echo "[validate] loading ${AIC_IMAGE} from ${tarball}"
    ${DECOMPRESS_CMD} '${tarball}' | docker load >/dev/null
    echo "\${_tar_mtime}" > "\${_marker}" 2>/dev/null || true
fi

cd '${AIC_DAY_DIR}'
# shellcheck source=/dev/null
source '${AIC_DAY_DIR}/monitoring/monitoring-lib.sh'
ensure_compose || { echo "[validate] docker compose unavailable" >&2; exit 1; }

export IMAGE_NAME='${AIC_IMAGE}'
export VLLM_MODEL='${model}'
export EMU_PROFILE_PACK_HOST='${pack_dir}'
export VLLM_EMULATOR_PROFILE_PACK='/profiles/${pack_file}'
export VLM_MAX_MODEL_LEN=${max_len}
export VLM_MAX_NUM_BATCHED_TOKENS=${max_bt}
export VLLM_EXTRA_ARGS='${extra_args}'
export VLLM_EMULATOR_ORACLE_K='${oracle_k}'
export HF_HOME='${AIC_CAPTURE_HF_HOME}'
export HF_TOKEN='${HF_TOKEN:-}'
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export LOG="\${_logdir}"
mkdir -p "\${_logdir}/vllm-emulator/bench"

cleanup() {
    timeout 60 docker compose -f '${AIC_DAY_DIR}/docker/docker-compose.yml' \
        --profile emulate logs --no-color --no-log-prefix vllm-emulator \
        > "\${_logdir}/validate-vllm.log" 2>&1 || true
    timeout 60 docker compose -f '${AIC_DAY_DIR}/docker/docker-compose.yml' \
        --profile emulate down --remove-orphans --timeout 10 >/dev/null 2>&1 || true
    timeout 30 docker rm -f aic-vllm-emulator >/dev/null 2>&1 || true
}
trap cleanup EXIT
compose() { docker compose -f '${AIC_DAY_DIR}/docker/docker-compose.yml' "\$@"; }

echo "[validate] starting the emulator with pack /profiles/${pack_file} ..."
compose --profile emulate up -d vllm-emulator || { echo "[validate] FAIL: compose up" >&2; exit 1; }
ready=0
for _i in \$(seq 1 ${AIC_EMULATE_READY_TIMEOUT}); do
    curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1 && { ready=1; break; }
    [ -z "\$(docker ps -q -f name=aic-vllm-emulator)" ] && break
    sleep 5
done
if [ "\${ready}" != "1" ]; then
    echo "[validate] FAIL: emulator never became ready" >&2
    compose --profile emulate logs --tail 120 --no-color vllm-emulator 2>&1 | sed 's/^/  [emu] /'
    exit 1
fi
# Fail loudly if the pack did not load -- the hook prints this only on success,
# and a missing pack silently leaves the emulator disabled.
# Grep a FILE, not a pipe: \`docker compose logs | grep -q\` makes grep exit on
# the first match, compose dies of SIGPIPE, and \`set -o pipefail\` then reports
# the whole pipeline as failed -- intermittently, depending on log size.
compose --profile emulate logs --no-color --no-log-prefix vllm-emulator \
    > "\${_logdir}/validate-startup.log" 2>&1 || true
if grep -q '\[ExecutorEmulatorHook\] Enabled' "\${_logdir}/validate-startup.log"; then
    grep -m1 '\[ExecutorEmulatorHook\] Enabled' "\${_logdir}/validate-startup.log" | sed 's/^/[validate] /'
else
    echo "[validate] FAIL: emulator hook did not activate with this pack" >&2
    tail -60 "\${_logdir}/validate-startup.log" | sed 's/^/  [emu] /'
    exit 1
fi
echo "[validate] emulator up on the captured pack; running the sweep ..."

model_tag="\$(echo '${model}' | tr '/' '-')"
for point in ${sweep}; do
    isl="\$(echo "\${point}" | cut -d, -f1)"; osl="\$(echo "\${point}" | cut -d, -f2)"
    conc="\$(echo "\${point}" | cut -d, -f3)"; nprompts="\$(echo "\${point}" | cut -d, -f4)"
    echo "[validate] --- emulated isl=\${isl} osl=\${osl} concurrency=\${conc} prompts=\${nprompts} ---"
    docker exec aic-vllm-emulator vllm bench serve \
        --host localhost --port 8000 --model '${model}' \
        --dataset-name random \
        --random-input-len "\${isl}" --random-output-len "\${osl}" \
        --num-prompts "\${nprompts}" --max-concurrency "\${conc}" \
        --ignore-eos --seed 0 \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result --result-dir /var/log/aic-vllm-emulator/bench \
        --result-filename "emu-\${model_tag}-isl\${isl}-osl\${osl}-c\${conc}.json" \
        2>&1 | tail -25 | sed 's/^/  [bench] /'
done

echo "[validate] === real (MI-series capture) vs emulated ==="
python3 - '${real_dir}' "\${_logdir}/vllm-emulator/bench" <<'PYEOF'
import json, os, glob, sys
real_dir, emu_dir = sys.argv[1], sys.argv[2]
rows, worst = [], 0.0
for emu_path in sorted(glob.glob(os.path.join(emu_dir, "emu-*.json"))):
    real_path = os.path.join(real_dir, "real-" + os.path.basename(emu_path)[4:])
    if not os.path.exists(real_path):
        print(f"  (no real counterpart for {os.path.basename(emu_path)})")
        continue
    e, r = json.load(open(emu_path)), json.load(open(real_path))
    tag = os.path.basename(emu_path)[4:-5]
    for metric in ("mean_ttft_ms", "mean_tpot_ms", "output_throughput"):
        rv, ev = r.get(metric), e.get(metric)
        if not rv:
            continue
        delta = (ev - rv) / rv * 100.0
        worst = max(worst, abs(delta))
        rows.append((tag, metric, rv, ev, delta))
w = max([len(t) for t, *_ in rows] + [10])
print(f"  {'point'.ljust(w)}  {'metric':<18} {'real':>12} {'emulated':>12} {'delta':>9}")
for tag, metric, rv, ev, delta in rows:
    print(f"  {tag.ljust(w)}  {metric:<18} {rv:12.2f} {ev:12.2f} {delta:+8.1f}%")
print(f"\n  worst absolute delta: {worst:.1f}%")
PYEOF
exit 0
REMOTE
)"

    _sbatch_run aic-emulate-validate emulate-validate "${remote_script}" \
        "${_sel[@]}" \
        --nodes=1 --ntasks=1 \
        --cpus-per-task="${AIC_EMULATE_CPUS}" --mem="${AIC_EMULATE_MEM}" \
        --time="${AIC_EMULATE_TIME}"
    log "emulate-validate complete"
}

# --- profile-capture: build an AMD profile pack from a real GPU serve ---------
# Runs the FULL image on a GPU node with VLLM_EMULATOR_TRACE_STEP_CYCLE=1, which
# makes the patched engine core append one
#   {total_tokens, num_new_reqs, num_decode_seqs, sum_kv, step_cycle_us}
# record per step to a JSONL trace.  A `vllm bench serve` sweep drives the two
# axes the emulator's oracle buckets on (batch total-tokens x concurrency), then
# vllm_emulator.profile.build_serving_profile_filtered turns the trace into a
# profile pack.  Everything lands in AIC_CAPTURE_DIR on shared storage.
#
# NOTE: this is a REAL serve -- real weights, real kernels.  The emulator hook
# is NOT enabled here (VLLM_EMULATOR_ENABLE_ORACLE is deliberately unset);
# tracing alone is a passive measurement.
cmd_profile_capture() {
    _pick_compress
    local tarball; tarball="$(_tarball_path)"
    command -v sbatch >/dev/null 2>&1 || die "sbatch not found; cannot run the capture job"
    [[ -r "${tarball}" ]] || die "tarball not found: ${tarball} (run 'build' with AIC_ROCM_ARCH=<arch> first)"

    local -a _sel=()
    if [[ -n "${AIC_CAPTURE_NODE:-}" ]]; then
        _sel=(--nodelist="${AIC_CAPTURE_NODE}")
        log "profile-capture on ${AIC_CAPTURE_NODE} via sbatch (partition ${AIC_BUILD_PARTITION})"
    else
        [[ -n "${AIC_CAPTURE_CONSTRAINT}" ]] && _sel=(--constraint="${AIC_CAPTURE_CONSTRAINT}")
        log "profile-capture via sbatch (partition ${AIC_BUILD_PARTITION}, constraint ${AIC_CAPTURE_CONSTRAINT:-<none>})"
    fi
    log "image: ${AIC_IMAGE}  model: ${AIC_CAPTURE_MODEL}"
    log "output dir: ${AIC_CAPTURE_DIR}   hf cache: ${AIC_CAPTURE_HF_HOME}"
    log "sweep (isl,osl,concurrency,prompts): ${AIC_CAPTURE_SWEEP}"

    mkdir -p "${AIC_CAPTURE_DIR}" "${AIC_CAPTURE_HF_HOME}"

    local remote_script
    remote_script="$(cat <<REMOTE
set -uo pipefail
command -v docker >/dev/null 2>&1 || { echo "\$(hostname): docker not found" >&2; exit 1; }
echo "[capture] host=\$(hostname) docker=\$(docker --version)"
command -v rocm_agent_enumerator >/dev/null 2>&1 && \
    echo "[capture] host GPU arch(es): \$(rocm_agent_enumerator | grep -E '^gfx' | sort -u | tr '\n' ' ')"

_marker="/var/tmp/aic-loaded-\$(id -u)-\$(echo '${AIC_IMAGE}' | tr '/:' '__').mtime"
_tar_mtime="\$(stat -c %Y '${tarball}' 2>/dev/null || echo 0)"
_have_img="\$(docker images -q '${AIC_IMAGE}')"
_loaded_mtime="\$(cat "\${_marker}" 2>/dev/null || echo 0)"
if [ "${AIC_FORCE_LOAD:-0}" = "1" ] || [ -z "\${_have_img}" ] || [ "\${_tar_mtime}" -gt "\${_loaded_mtime}" ]; then
    echo "[capture] loading ${AIC_IMAGE} from ${tarball}"
    ${DECOMPRESS_CMD} '${tarball}' | docker load >/dev/null
    echo "\${_tar_mtime}" > "\${_marker}" 2>/dev/null || true
else
    echo "[capture] image up to date on \$(hostname) (id \${_have_img})"
fi

# Honour Slurm's GPU allocation: with --gres=gpu:1 the scheduler exports the
# device index it gave us, and these nodes are shared -- hard-coding device 0
# would land on somebody else's GPU.  AIC_CAPTURE_GPU is only the fallback.
_gpu="\${ROCR_VISIBLE_DEVICES:-\${HIP_VISIBLE_DEVICES:-${AIC_CAPTURE_GPU}}}"
echo "[capture] using GPU index \${_gpu} (slurm ROCR_VISIBLE_DEVICES=\${ROCR_VISIBLE_DEVICES:-<unset>})"

stamp="\$(date +%Y%m%d-%H%M%S)"
model_tag="\$(echo '${AIC_CAPTURE_MODEL}' | tr '/' '-')"
# Write the trace to NODE-LOCAL disk, not the shared filesystem: the tracer
# writes from inside the engine loop, and a shared-FS stall shows up as a
# fabricated multi-hundred-ms "step" in the very next sample.  Copied to
# AIC_CAPTURE_DIR when the run ends.
trace_local_host="/var/tmp/aic-capture-\${SLURM_JOB_ID:-\$\$}"
trace_name="step-trace-\${model_tag}-\${stamp}.jsonl"
trace="/tracelocal/\${trace_name}"
mkdir -p "\${trace_local_host}" '${AIC_CAPTURE_DIR}/bench' '${AIC_CAPTURE_HF_HOME}'

cleanup() {
    docker logs aic-vllm-capture > "\${_logdir}/capture-vllm.log" 2>&1 || true
    docker rm -f aic-vllm-capture >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[capture] starting a REAL serve with step-cycle tracing -> \${trace}"
docker run -d --name aic-vllm-capture \
    --device /dev/kfd --device /dev/dri \
    --network host --ipc host \
    --cap-add CAP_SYS_ADMIN --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v '${AIC_CAPTURE_HF_HOME}':/hf \
    -v '${AIC_CAPTURE_DIR}':/trace \
    -v "\${trace_local_host}":/tracelocal \
    -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub \
    -e VLLM_CACHE_ROOT=/hf/vllm -e VLLM_CONFIG_ROOT=/hf/vllm_config \
    -e HF_TOKEN='${HF_TOKEN:-}' -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
    -e ROCR_VISIBLE_DEVICES="\${_gpu}" \
    -e VLLM_ROCM_USE_AITER=1 \
    -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:False \
    -e PYTHONUNBUFFERED=1 \
    -e VLLM_EMULATOR_TRACE_STEP_CYCLE=1 \
    -e VLLM_EMULATOR_STEP_TRACE_OUTPUT="\${trace}" \
    '${AIC_IMAGE}' \
    --model '${AIC_CAPTURE_MODEL}' \
    --host 0.0.0.0 --port 8000 \
    --max-model-len ${AIC_CAPTURE_MAX_MODEL_LEN} \
    --max-num-batched-tokens ${AIC_CAPTURE_MAX_BATCHED_TOKENS} \
    --gpu-memory-utilization ${AIC_CAPTURE_GPU_UTIL} \
    --attention-backend TRITON_ATTN \
    --disable-access-log-for-endpoints "/health,/metrics,/v1/models" \
    ${AIC_CAPTURE_EXTRA_ARGS} \
    >/dev/null || { echo "[capture] FAIL: docker run failed" >&2; exit 1; }

ready=0
for _i in \$(seq 1 ${AIC_CAPTURE_READY_TIMEOUT}); do
    if curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1; then ready=1; break; fi
    if [ -z "\$(docker ps -q -f name=aic-vllm-capture)" ]; then
        echo "[capture] FAIL: the vLLM container exited during startup" >&2
        docker logs --tail 120 aic-vllm-capture 2>&1 | sed 's/^/  [vllm] /'
        exit 1
    fi
    sleep 5
done
if [ "\${ready}" != "1" ]; then
    echo "[capture] FAIL: vLLM never became ready" >&2
    docker logs --tail 120 aic-vllm-capture 2>&1 | sed 's/^/  [vllm] /'
    exit 1
fi
echo "[capture] endpoint ready; running the sweep ..."

# Sanity: the tracer must have written its header (GPU + model metadata), or
# the pack would come out with gpu=unknown and no model_config.
# Grep a FILE, not a pipe: \`grep -q\` exits on the first match, \`docker logs\`
# then dies of SIGPIPE, and \`set -o pipefail\` turns that into a false failure.
docker logs aic-vllm-capture > "\${_logdir}/capture-startup.log" 2>&1 || true
if grep -q '\[StepCycleTracer\] Header written' "\${_logdir}/capture-startup.log"; then
    grep -m1 '\[StepCycleTracer\] Header written' "\${_logdir}/capture-startup.log" | sed 's/^/[capture] /'
else
    echo "[capture] FAIL: the step-cycle tracer did not start (no header)" >&2
    tail -60 "\${_logdir}/capture-startup.log" | sed 's/^/  [vllm] /'
    exit 1
fi

sweep_rc=0
_seed=0
for point in ${AIC_CAPTURE_SWEEP}; do
    _seed=\$((_seed + 1))
    isl="\$(echo "\${point}" | cut -d, -f1)"
    osl="\$(echo "\${point}" | cut -d, -f2)"
    conc="\$(echo "\${point}" | cut -d, -f3)"
    nprompts="\$(echo "\${point}" | cut -d, -f4)"
    echo "[capture] --- sweep isl=\${isl} osl=\${osl} concurrency=\${conc} prompts=\${nprompts} ---"
    docker exec aic-vllm-capture vllm bench serve \
        --host localhost --port 8000 \
        --model '${AIC_CAPTURE_MODEL}' \
        --dataset-name random \
        --random-input-len "\${isl}" --random-output-len "\${osl}" \
        --num-prompts "\${nprompts}" --max-concurrency "\${conc}" \
        --ignore-eos --seed "\${_seed}" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result --result-dir /trace/bench \
        --result-filename "real-\${model_tag}-isl\${isl}-osl\${osl}-c\${conc}.json" \
        2>&1 | sed 's/^/  [bench] /'
    [ "\${PIPESTATUS[0]}" -eq 0 ] || { echo "[capture] WARN: sweep point \${point} failed"; sweep_rc=1; }
done

# Stop the server so the tracer's buffered records are written out by the
# shutdown path, then give the file a moment to land on shared storage.
echo "[capture] sweep done (rc=\${sweep_rc}); stopping the server ..."
docker stop -t 60 aic-vllm-capture >/dev/null 2>&1 || true
sleep 5

if [ ! -s "\${trace_local_host}/\${trace_name}" ]; then
    echo "[capture] FAIL: no trace records were written" >&2
    exit 1
fi
cp "\${trace_local_host}/\${trace_name}" '${AIC_CAPTURE_DIR}'/
rm -rf "\${trace_local_host}"
host_trace='${AIC_CAPTURE_DIR}'/"\${trace_name}"
echo "[capture] trace: \$(wc -l < "\${host_trace}") lines, \$(du -h "\${host_trace}" | cut -f1)"

pack='${AIC_CAPTURE_DIR}'/"\${model_tag}-\${stamp}.json"
echo "[capture] building profile pack -> \${pack}"
docker run --rm \
    -v '${AIC_CAPTURE_DIR}':/trace \
    --entrypoint python3 '${AIC_IMAGE}' \
    -m vllm_emulator.profile.build_serving_profile_filtered \
    "/trace/\${trace_name}" "/trace/\$(basename "\${pack}")" \
    2>&1 | sed 's/^/  [pack] /'
[ "\${PIPESTATUS[0]}" -eq 0 ] || { echo "[capture] FAIL: profile-pack build failed" >&2; exit 1; }

# The pack must validate against the emulator's own schema, and must carry the
# AMD GPU identity -- a pack that says gpu=unknown is useless for comparison.
docker run --rm -v '${AIC_CAPTURE_DIR}':/trace --entrypoint python3 '${AIC_IMAGE}' -c "
import json, sys
from vllm_emulator.profile.loader import load_profile_pack
p = load_profile_pack('/trace/\$(basename "\${pack}")')
mc = p.get('model_config', {})
gpu = mc.get('gpu', {})
cells = len(p.get('step_cycle_2d_distribution', []))
cells3d = len(p.get('step_cycle_3d_distribution', []))
samples = sum(c['num_samples'] for c in p.get('step_cycle_2d_distribution', []))
kv_bytes = gpu.get('available_kv_cache_bytes')
print(f\"pack OK: gpu={p.get('gpu_model')} model={p.get('model_name')} \"
      f\"cells={cells} (3d {cells3d}) samples={samples} \"
      f\"mem={gpu.get('gpu_memory_bytes')} cu={gpu.get('gpu_sm_count')}\")
print(f\"  measured KV pool: {kv_bytes} bytes\" if kv_bytes else
      \"  WARNING: no available_kv_cache_bytes in the pack -- the emulator will \"
      \"ESTIMATE the KV pool, so its admission point will not match the capture\")
print(f\"  captured under: max_num_batched_tokens={mc.get('max_num_batched_tokens')} \"
      f\"prefix_caching={mc.get('enable_prefix_caching')} \"
      f\"async_scheduling={mc.get('async_scheduling')} \"
      f\"kv_cache_dtype={mc.get('kv_cache_dtype')} vllm={mc.get('vllm_version')}\")
if p.get('gpu_model') in (None, 'unknown') or not cells:
    sys.exit('pack is missing GPU identity or has no cells')
" 2>&1 | sed 's/^/[capture] /'
[ "\${PIPESTATUS[0]}" -eq 0 ] || { echo "[capture] FAIL: pack validation failed" >&2; exit 1; }

# Record exactly how this pack was produced, next to the pack.
cat > "\${pack%.json}.capture.txt" <<META
model                 : ${AIC_CAPTURE_MODEL}
image                 : ${AIC_IMAGE}
node                  : \$(hostname)
serve flags           : --max-model-len ${AIC_CAPTURE_MAX_MODEL_LEN} --max-num-batched-tokens ${AIC_CAPTURE_MAX_BATCHED_TOKENS} --gpu-memory-utilization ${AIC_CAPTURE_GPU_UTIL} --attention-backend TRITON_ATTN ${AIC_CAPTURE_EXTRA_ARGS}
env                   : VLLM_ROCM_USE_AITER=1
                        (the effective serving config -- async scheduling, prefix
                        caching, kv-cache-dtype, KV pool size -- is recorded
                        inside the pack itself, under model_config)
sweep (isl,osl,c,n)   : ${AIC_CAPTURE_SWEEP}
trace                 : \${trace_name}
real-hardware results : bench/real-\${model_tag}-*.json
META
echo "[capture] wrote \${pack%.json}.capture.txt"
echo "[capture] PACK: \${pack}"
exit \${sweep_rc}
REMOTE
)"

    local -a _gres_arg=(); [[ "${AIC_SPUR_CLUSTER}" != "1" ]] && _gres_arg=(--gres=gpu:1)
    _sbatch_run aic-profile-capture profile-capture "${remote_script}" \
        "${_sel[@]}" \
        "${_gres_arg[@]}" \
        --nodes=1 --ntasks=1 \
        --cpus-per-task="${AIC_CAPTURE_CPUS}" --mem="${AIC_CAPTURE_MEM}" \
        --time="${AIC_CAPTURE_TIME}"
    log "profile-capture complete; packs in ${AIC_CAPTURE_DIR}"
}

# --- main --------------------------------------------------------------------
main() {
    local sub="${1:-all}"
    case "${sub}" in
        build)           cmd_build ;;
        build-emulate)   cmd_build_emulate ;;
        build-exporters) cmd_build_exporters ;;
        load)            cmd_load ;;
        push)            cmd_push ;;
        test)            cmd_test ;;
        tiny-test)       cmd_tiny_test ;;
        emulate-test)    cmd_emulate_test ;;
        emulate-validate) cmd_emulate_validate ;;
        profile-capture) cmd_profile_capture ;;
        all)             cmd_build; cmd_build_exporters; cmd_load ;;
        -h|--help|help)
            sed -n '2,70p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            ;;
        *) die "unknown command '${sub}' (use: build | build-emulate | build-exporters | load | push | test | tiny-test | emulate-test | emulate-validate | profile-capture | all | help)" ;;
    esac
}

main "$@"
