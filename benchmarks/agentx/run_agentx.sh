#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Run the AgentX MVP benchmark (inferencex-agentx-mvp scenario) against an
# OpenAI-compatible endpoint using the aic-agentx container.
#
# Usage:
#   ./benchmarks/agentx/run_agentx.sh [OPTIONS]
#
# Options:
#   --endpoint URL         Inference server endpoint (default: http://localhost:8000)
#   --model MODEL          Model ID (default: Qwen/Qwen2.5-7B-Instruct)
#   --concurrency N        Parallel workers (default: 2)
#   --duration SECS        Benchmark duration in seconds (default: 900)
#   --max-context N        Max context length in tokens (default: 65536)
#   --otel-url URL         OTEL collector endpoint (default: disabled)
#   --cache-bust STRATEGY  first_turn_prefix|none|all (default: first_turn_prefix)
#   --output-dir DIR       Results directory (default: logs/agentx/<timestamp>)
#   --image REF            aic-agentx image ref (default: aic-agentx:latest)
#   --dry-run              Print command without running
#
# Environment:
#   HF_TOKEN              HuggingFace token for dataset access
#   HF_HOME               Host HF cache dir (default: ~/.cache/huggingface)
#   AGENTX_OTEL_URL       Override --otel-url (convenience env var)
#   REQUESTS_CA_BUNDLE / SSL_CERT_FILE / NO_PROXY / no_proxy
#                         TLS/proxy settings — passed through to the container
#                         as-is from the calling shell environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── defaults ─────────────────────────────────────────────────────────────────
ENDPOINT="${BENCH_ENDPOINT:-${AGENTX_ENDPOINT:-http://localhost:8000}}"
MODEL="${AGENTX_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CONCURRENCY="${AGENTX_CONCURRENCY:-2}"
DURATION="${AGENTX_DURATION:-900}"
MAX_CONTEXT="${AGENTX_MAX_CONTEXT:-65536}"
OTEL_URL="${AGENTX_OTEL_URL:-}"
CACHE_BUST="${AGENTX_CACHE_BUST:-first_turn_prefix}"
OUTPUT_DIR=""
IMAGE="${AGENTX_IMAGE:-aic-agentx:latest}"
DRY_RUN="false"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
# LOG follows the rocm-aic convention: defaults to <repo_root>/logs
LOG_BASE="${LOG:-${REPO_ROOT}/logs}"

# ── arg parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint)     ENDPOINT="$2";     shift 2 ;;
    --model)        MODEL="$2";        shift 2 ;;
    --concurrency)  CONCURRENCY="$2";  shift 2 ;;
    --duration)     DURATION="$2";     shift 2 ;;
    --max-context)  MAX_CONTEXT="$2";  shift 2 ;;
    --otel-url)     OTEL_URL="$2";     shift 2 ;;
    --cache-bust)   CACHE_BUST="$2";   shift 2 ;;
    --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
    --image)        IMAGE="$2";        shift 2 ;;
    --dry-run)      DRY_RUN="true";    shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ── output directory ─────────────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%dT%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${LOG_BASE}/agentx/${TIMESTAMP}_c${CONCURRENCY}}"
mkdir -p "${OUTPUT_DIR}"

LOG="${OUTPUT_DIR}/run.log"
exec > >(tee -a "${LOG}") 2>&1

echo "=== AIC AgentX Benchmark — $(date) ==="
echo "Endpoint:    ${ENDPOINT}"
echo "Model:       ${MODEL}"
echo "Concurrency: ${CONCURRENCY}"
echo "Duration:    ${DURATION}s"
echo "Max context: ${MAX_CONTEXT} tokens"
echo "Cache bust:  ${CACHE_BUST}"
echo "OTEL:        ${OTEL_URL:-disabled}"
echo "Output:      ${OUTPUT_DIR}"
echo "Image:       ${IMAGE}"
echo ""

# ── ensure image exists ───────────────────────────────────────────────────────
if ! docker image inspect "${IMAGE}" &>/dev/null; then
  echo "Image ${IMAGE} not found. Build with: make agentx-build" >&2
  exit 1
fi

# ── check endpoint is reachable ──────────────────────────────────────────────
echo "Checking inference endpoint at ${ENDPOINT}/v1/models ..."
if ! curl -sf --max-time 10 "${ENDPOINT}/v1/models" > /dev/null; then
  echo "ERROR: endpoint ${ENDPOINT} not reachable." >&2
  exit 1
fi

# Verify the requested model is loaded
curl -sf "${ENDPOINT}/v1/models" | python3 -c "
import sys, json
models = {m['id'] for m in json.load(sys.stdin)['data']}
model = '${MODEL}'
if model not in models:
    print(f'ERROR: model {model!r} not loaded. Loaded: {models}', file=sys.stderr)
    sys.exit(1)
print(f'Model {model!r} confirmed.')
" || exit 1

# ── tokenizer: use locally-cached snapshot (no network needed at runtime) ────
# Look for a Qwen3-8B snapshot — used as the tokenizer for token counting.
TOKENIZER_SNAP="$(ls -d "${HF_HOME}/hub/models--Qwen--Qwen3-8B/snapshots"/*/ 2>/dev/null | head -1)"
TOKENIZER_ARG=()
if [[ -n "${TOKENIZER_SNAP}" ]]; then
  TOKENIZER_CONTAINER="/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/$(basename "${TOKENIZER_SNAP}")"
  TOKENIZER_ARG=(--tokenizer "${TOKENIZER_CONTAINER}")
fi

# ── OTEL args ─────────────────────────────────────────────────────────────────
OTEL_ARGS=()
if [[ -n "${OTEL_URL}" ]]; then
  OTEL_ARGS=(
    --otel-url "${OTEL_URL}"
    --stream metrics timing
    --otel-resource-attributes
      "concurrency=${CONCURRENCY}"
      "model=${MODEL}"
      "run_id=$(basename "${OUTPUT_DIR}")"
  )
fi

# ── aiperf args ───────────────────────────────────────────────────────────────
AIPERF_ARGS=(
  profile
    --scenario inferencex-agentx-mvp
    --url "${ENDPOINT}"
    --model "${MODEL}"
    --max-context-length "${MAX_CONTEXT}"
    --endpoint-type chat
    --public-dataset semianalysis_cc_traces_weka_with_subagents
    --concurrency "${CONCURRENCY}"
    --streaming
    --api-key ignore
    --use-server-token-count
    "${TOKENIZER_ARG[@]}"
    --extra-inputs "ignore_eos:true"
    --cache-bust "${CACHE_BUST}"
    --system-idle-gap-cap-seconds 10
    --benchmark-duration "${DURATION}"
    --random-seed 20260707
    --ui simple
    --output-artifact-dir "/results/$(basename "${OUTPUT_DIR}")"
    "${OTEL_ARGS[@]}"
)

echo "Command: aiperf ${AIPERF_ARGS[*]}"
echo ""

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "(dry-run — not executing)"
  exit 0
fi

# ── run ───────────────────────────────────────────────────────────────────────
echo "Starting benchmark at $(date)..."
mkdir -p "${LOG_BASE}"
docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -v "${LOG_BASE}:/results" \
  -v "${HF_HOME}:/hf_cache" \
  -e "HF_TOKEN=${HF_TOKEN:-}" \
  -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}" \
  -e "REQUESTS_CA_BUNDLE=${REQUESTS_CA_BUNDLE:-}" \
  -e "SSL_CERT_FILE=${SSL_CERT_FILE:-}" \
  -e "NO_PROXY=${NO_PROXY:-}" \
  -e "no_proxy=${no_proxy:-}" \
  "${IMAGE}" "${AIPERF_ARGS[@]}" || {
    echo "aiperf exited with status $?. See ${LOG}" >&2
    exit 1
  }

echo ""
echo "Benchmark complete at $(date)."
echo "Results: ${OUTPUT_DIR}"
