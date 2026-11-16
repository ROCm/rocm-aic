#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Run the AgentX benchmark against an OpenAI-compatible endpoint using the
# aic-agentx container.
#
# Usage:
#   ./benchmarks/agentx/run_agentx.sh [OPTIONS]
#
# Options:
#   --endpoint URL         Inference server endpoint (default: http://localhost:8000)
#   --model MODEL          Model ID (default: Qwen/Qwen2.5-7B-Instruct)
#   --tokenizer MODEL      Tokenizer ID (default: same as --model)
#   --concurrency N        Parallel workers (default: 2)
#   --duration SECS        Benchmark duration in seconds (default: 3600)
#   --server-metrics URL   Prometheus metrics URL(s) passed to agentx; repeat for
#                          multiple nodes (e.g. prefill + decode in disagg setups)
#   --otel-url URL         OTEL collector endpoint (default: disabled)
#   --output-dir DIR       Results directory (default: logs/agentx/<timestamp>)
#   --image REF            aic-agentx image ref (default: aic-agentx:latest)
#   --dry-run              Print command without running
#
# Environment:
#   HF_TOKEN              HuggingFace token for dataset access
#   HF_HOME               Host HF cache dir (default: ~/.cache/huggingface)
#   AGENTX_SERVER_METRICS Space-separated Prometheus metrics URLs
#   AGENTX_OTEL_URL       OTEL collector endpoint; enables OTEL when set
#   REQUESTS_CA_BUNDLE / SSL_CERT_FILE / NO_PROXY / no_proxy
#                         TLS/proxy settings — passed through to the container
#                         as-is from the calling shell environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── defaults ─────────────────────────────────────────────────────────────────
ENDPOINT="${BENCH_ENDPOINT:-${AGENTX_ENDPOINT:-http://localhost:8000}}"
MODEL="${AGENTX_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER="${AGENTX_TOKENIZER:-}"       # defaults to MODEL if empty
CONCURRENCY="${AGENTX_CONCURRENCY:-2}"
DURATION="${AGENTX_DURATION:-3600}"
OTEL_URL="${AGENTX_OTEL_URL:-}"
# Space-separated Prometheus /metrics URLs (one per node; supports disagg setups)
SERVER_METRICS="${AGENTX_SERVER_METRICS:-}"
OUTPUT_DIR=""
IMAGE="${AGENTX_IMAGE:-aic-agentx:latest}"
DRY_RUN="false"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
LOG_BASE="${LOG:-${REPO_ROOT}/logs}"

# ── arg parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint)       ENDPOINT="$2";        shift 2 ;;
    --model)          MODEL="$2";           shift 2 ;;
    --tokenizer)      TOKENIZER="$2";       shift 2 ;;
    --concurrency)    CONCURRENCY="$2";     shift 2 ;;
    --duration)       DURATION="$2";        shift 2 ;;
    --server-metrics) SERVER_METRICS="${SERVER_METRICS:+${SERVER_METRICS} }$2"; shift 2 ;;
    --otel-url)       OTEL_URL="$2";        shift 2 ;;
    --output-dir)     OUTPUT_DIR="$2";      shift 2 ;;
    --image)          IMAGE="$2";           shift 2 ;;
    --dry-run)        DRY_RUN="true";       shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

TOKENIZER="${TOKENIZER:-${MODEL}}"

# ── output directory ─────────────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%dT%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${LOG_BASE}/agentx/${TIMESTAMP}_c${CONCURRENCY}}"
mkdir -p "${OUTPUT_DIR}"

LOG="${OUTPUT_DIR}/run.log"
exec > >(tee -a "${LOG}") 2>&1

echo "=== AIC AgentX Benchmark — $(date) ==="
echo "Endpoint:       ${ENDPOINT}"
echo "Model:          ${MODEL}"
echo "Tokenizer:      ${TOKENIZER}"
echo "Concurrency:    ${CONCURRENCY}"
echo "Duration:       ${DURATION}s"
echo "Server metrics: ${SERVER_METRICS:-none}"
echo "OTEL:           ${OTEL_URL:-disabled}"
echo "Output:         ${OUTPUT_DIR}"
echo "Image:          ${IMAGE}"
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

curl -sf "${ENDPOINT}/v1/models" | python3 -c "
import sys, json
models = {m['id'] for m in json.load(sys.stdin)['data']}
model = '${MODEL}'
if model not in models:
    print(f'ERROR: model {model!r} not loaded. Loaded: {models}', file=sys.stderr)
    sys.exit(1)
print(f'Model {model!r} confirmed.')
" || exit 1

# ── server-metrics args ───────────────────────────────────────────────────────
SERVER_METRICS_ARGS=()
for url in ${SERVER_METRICS}; do
  SERVER_METRICS_ARGS+=(--server-metrics "${url}")
done

# ── OTEL args ─────────────────────────────────────────────────────────────────
OTEL_ARGS=()
if [[ -n "${OTEL_URL}" ]]; then
  OTEL_ARGS=(
    --otel-url "${OTEL_URL}"
    --otel-resource-attributes
      "concurrency=${CONCURRENCY}"
      "model=${MODEL}"
      "run_id=$(basename "${OUTPUT_DIR}")"
  )
fi

# ── agentx args ───────────────────────────────────────────────────────────────
AGENTX_ARGS=(
  profile
    --url "${ENDPOINT}"
    --model "${MODEL}"
    --tokenizer "${TOKENIZER}"
    --tokenizer-trust-remote-code
    --concurrency "${CONCURRENCY}"
    --benchmark-duration "${DURATION}"
    --public-dataset semianalysis_cc_traces_weka_062126
    --output-artifact-dir "/results/$(basename "${OUTPUT_DIR}")"
    "${SERVER_METRICS_ARGS[@]}"
    "${OTEL_ARGS[@]}"
)

echo "Command: agentx ${AGENTX_ARGS[*]}"
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
  "${IMAGE}" "${AGENTX_ARGS[@]}" || {
    echo "agentx exited with status $?. See ${LOG}" >&2
    exit 1
  }

echo ""
echo "Benchmark complete at $(date)."
echo "Results: ${OUTPUT_DIR}"
