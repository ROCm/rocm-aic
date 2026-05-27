#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Merge a per-job or legacy HF tree into the golden Hub cache.
#
#   .slurm/scripts/sync-hf-golden-cache.sh \\
#     /scratch/$USER/vllm-lmcache-hipfile/lmcache-341744/hf
#
set -euo pipefail

SRC="${1:-}"
: "${VLH_HF_HOME:=/scratch/${USER}/vllm-lmcache-hipfile/hf}"
DEST="${VLH_HF_HOME}"

if [[ -z "${SRC}" ]]; then
    echo "usage: $0 <source-hf-dir>" >&2
    echo "  e.g. /scratch/\$USER/vllm-lmcache-hipfile/lmcache-341744/hf" >&2
    exit 1
fi
if [[ ! -d "${SRC}" ]]; then
    echo "error: source not found: ${SRC}" >&2
    exit 1
fi

mkdir -p "${DEST}/hub" "${DEST}/datasets" "${DEST}/vllm" \
    "${DEST}/vllm_config" "${DEST}/torch" "${DEST}/torch_inductor"

echo "sync-hf-golden: ${SRC}/ -> ${DEST}/"
rsync -a "${SRC}/" "${DEST}/"

grp="$(id -gn 2>/dev/null || echo "${USER}")"
if ! chown -R "${USER}:${grp}" "${DEST}" 2>/dev/null; then
    sudo chown -R "${USER}:${grp}" "${DEST}" 2>/dev/null \
        || echo "warn: chown ${DEST} failed (docker may have written as root)" >&2
fi

if [[ -d "${DEST}/hub/models--openai--gpt-oss-120b" ]]; then
    du -sh "${DEST}/hub/models--openai--gpt-oss-120b"
    find "${DEST}/hub/models--openai--gpt-oss-120b" -name '*.safetensors' 2>/dev/null \
        | wc -l | xargs -I{} echo "  safetensors shards: {}"
    find "${DEST}/hub/models--openai--gpt-oss-120b" -name '*.incomplete' 2>/dev/null \
        | wc -l | xargs -I{} echo "  incomplete blobs: {}"
fi

echo "golden cache: ${DEST}"
