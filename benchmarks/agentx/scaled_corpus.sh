#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Generate a token-scaled copy of the AgentX trace corpus for models whose
# context is too small for the public one, and print its directory.
#
# A scaled corpus is an AgentX-shaped workload, NOT the AgentX benchmark: its
# results are not comparable with published AgentX or InferenceX numbers.  See
# scale_corpus.py for what scaling keeps and what it changes.
#
# Usage:
#   ./benchmarks/agentx/scaled_corpus.sh [OPTIONS]
#
# Options:
#   --scale N        Divide every token count by N (default: 8)
#   --dataset REPO   Hugging Face dataset repo to scale
#                    (default: semianalysisai/cc-traces-weka-062126)
#   --cache-dir DIR  Where corpora are kept (default: ~/.cache/aic-agentx)
#   --image REF      aic-agentx image ref (default: aic-agentx:latest)
#   --force          Regenerate a corpus that already exists
#   --print-dir      Only print the directory the corpus has or would have
#
# The corpus lands in <cache-dir>/<dataset name>-scaled-<N>/, one JSON file per
# trace plus SCALED-CORPUS.txt (source, divisor and how many traces fit common
# context sizes).  An existing corpus is reused.  The directory is the only
# thing printed on stdout; progress goes to stderr.
#
# Environment:
#   AGENTX_CORPUS_SCALE, AGENTX_CORPUS_DATASET, AGENTX_CORPUS_CACHE,
#   AGENTX_IMAGE_REF   the options above
#   HF_HOME            Host Hugging Face cache (default: ~/.cache/huggingface);
#                      the dataset is downloaded there once
#   HF_TOKEN           Passed through when set (the default dataset is public)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCALE="${AGENTX_CORPUS_SCALE:-8}"
DATASET="${AGENTX_CORPUS_DATASET:-semianalysisai/cc-traces-weka-062126}"
CACHE_DIR="${AGENTX_CORPUS_CACHE:-${HOME}/.cache/aic-agentx}"
IMAGE="${AGENTX_IMAGE_REF:-aic-agentx:latest}"
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
FORCE="false"
PRINT_DIR="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale)     SCALE="$2";     shift 2 ;;
    --dataset)   DATASET="$2";   shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --image)     IMAGE="$2";     shift 2 ;;
    --force)     FORCE="true";     shift ;;
    --print-dir) PRINT_DIR="true"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ ! "${SCALE}" =~ ^[0-9]+$ ]] || (( SCALE < 2 )); then
  echo "ERROR: the scale must be an integer >= 2 (token counts are divided by it), got '${SCALE}'" >&2
  exit 1
fi

if [[ -z "${CACHE_DIR}" || -z "${DATASET##*/}" ]]; then
  echo "ERROR: the cache directory and the dataset must not be empty" >&2
  exit 1
fi

NAME="${DATASET##*/}-scaled-${SCALE}"
OUT="${CACHE_DIR}/${NAME}"

if [[ "${PRINT_DIR}" == "true" ]]; then
  printf '%s\n' "${OUT}"
  exit 0
fi

if [[ -f "${OUT}/SCALED-CORPUS.txt" && "${FORCE}" != "true" ]]; then
  echo "Reusing the scaled corpus in ${OUT} (--force regenerates it):" >&2
  sed 's/^/  /' "${OUT}/SCALED-CORPUS.txt" >&2
  printf '%s\n' "${OUT}"
  exit 0
fi

if ! docker image inspect "${IMAGE}" &>/dev/null; then
  echo "ERROR: image ${IMAGE} not found. Build it with: make agentx-build" >&2
  exit 1
fi

mkdir -p "${CACHE_DIR}" "${HF_CACHE}"
# Built next to the final name and renamed, so an interrupted run never leaves
# a directory that looks complete.
TMP="$(mktemp -d "${CACHE_DIR}/.${NAME}.XXXXXX")"
trap 'rm -rf "${TMP}"' EXIT

ENV_ARGS=()
[[ -z "${HF_TOKEN:-}" ]] || ENV_ARGS+=(-e HF_TOKEN)

echo "Scaling ${DATASET} by 1/${SCALE} into ${OUT} ..." >&2
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  "${ENV_ARGS[@]}" \
  -v "${HF_CACHE}:/hf_cache" \
  -v "${SCRIPT_DIR}/scale_corpus.py:/aic/scale_corpus.py:ro" \
  -v "${TMP}:/out" \
  --entrypoint python3 \
  "${IMAGE}" /aic/scale_corpus.py \
    --dataset "${DATASET}" --divisor "${SCALE}" --output /out >&2

rm -rf "${OUT}"
mv "${TMP}" "${OUT}"
chmod 755 "${OUT}"
trap - EXIT
printf '%s\n' "${OUT}"
