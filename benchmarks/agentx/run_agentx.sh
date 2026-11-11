#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Run the AgentX benchmark (`aiperf profile --scenario inferencex-agentx-mvp`)
# against the running AIC stack, from a throwaway aic-client container on the
# stack's docker network.
#
# Usage:
#   ./benchmarks/agentx/run_agentx.sh [OPTIONS] [-- EXTRA_AIPERF_ARGS...]
#
# Options:
#   --endpoint URL         Server to benchmark, as the client container reaches
#                          it (default: the running <stack>-vllm-gpu<GPU>, by
#                          name on the docker network); a server on the docker
#                          host is http://host.docker.internal:<port>
#   --network NAME         Docker network the client joins (default: the
#                          stack's, AIC_STACK)
#   --model MODEL          Served model name (default: VLLM_MODEL, else the
#                          model the endpoint serves)
#   --tokenizer MODEL      Tokenizer ID (default: same as --model)
#   --concurrency N        Concurrent trajectories (default: 2)
#   --duration SECS        Benchmark duration (default: 1800).  Under 900 the
#                          run is not a valid AgentX submission
#   --max-context N        Drop traces whose peak prompt + output exceeds N
#                          tokens (default: the served model's max_model_len)
#   --dataset NAME         aiperf public dataset alias
#                          (default: semianalysis_cc_traces_weka_062126)
#   --corpus DIR           Replay a local weka-trace directory instead of the
#                          public dataset; not a valid AgentX submission
#   --corpus-scale N       Replay the public corpus with token counts divided
#                          by N (generated on first use by scaled_corpus.sh);
#                          an AgentX-shaped workload, NOT the AgentX benchmark
#   --seed N               Random seed (default: 42)
#   --slice-duration SECS  Width of the time slices in the export (default: 30)
#   --server-metrics URL   Extra Prometheus /metrics URL to record; repeatable.
#                          `none` records only the endpoint's own /metrics
#                          (default: the stack's LMCache metrics port)
#   --output-dir DIR       Run directory (default: <LOG>/agentx/<timestamp>_c<N>)
#   --image REF            aic-client image ref (default: aic-client:latest)
#   --dry-run              Print the command and exit; needs neither the image
#                          nor a running stack
#
# The run directory holds run.log and everything aiperf exports (see README.md).
# The client runs as the invoking user.
#
# Environment:
#   BENCH_ENDPOINT, AIC_NETWORK, AGENTX_MODEL, AGENTX_TOKENIZER,
#   AGENTX_CONCURRENCY, AGENTX_DURATION, AGENTX_MAX_CONTEXT, AGENTX_DATASET,
#   AGENTX_CORPUS, AGENTX_CORPUS_SCALE, AGENTX_SEED, AGENTX_SLICE_DURATION,
#   AGENTX_SERVER_METRICS (space-separated), AIC_CLIENT_IMAGE
#                         the options above
#   AGENTX_EXTRA_ARGS     Extra aiperf arguments, space-separated (for example
#                         `--otel-url http://collector:4318`; the AIC
#                         monitoring stack has no OTLP receiver)
#   GPU                   Index of the vLLM container to benchmark (default: 0)
#   LOG                   Log root (default: <repo>/logs)
#   HF_HOME               Host Hugging Face cache (default: ~/.cache/huggingface)
#   HF_TOKEN, NO_PROXY, no_proxy, REQUESTS_CA_BUNDLE,
#   SSL_CERT_FILE         Passed through to the client when set (a CA bundle
#                         file is mounted at the same path)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=scripts/aic-endpoints.sh
source "${REPO_ROOT}/scripts/aic-endpoints.sh"

# aiperf's inferencex-agentx-mvp scenario rejects a shorter run.
MIN_VALID_DURATION=900

# ── defaults ─────────────────────────────────────────────────────────────────
ENDPOINT="${BENCH_ENDPOINT:-}"
NETWORK="${AIC_NETWORK:-${AIC_STACK}}"
MODEL="${AGENTX_MODEL:-${VLLM_MODEL:-}}"
TOKENIZER="${AGENTX_TOKENIZER:-}"
CONCURRENCY="${AGENTX_CONCURRENCY:-2}"
DURATION="${AGENTX_DURATION:-1800}"
MAX_CONTEXT="${AGENTX_MAX_CONTEXT:-}"
DATASET="${AGENTX_DATASET:-semianalysis_cc_traces_weka_062126}"
CORPUS="${AGENTX_CORPUS:-}"
CORPUS_SCALE="${AGENTX_CORPUS_SCALE:-}"
SEED="${AGENTX_SEED:-42}"
SLICE_DURATION="${AGENTX_SLICE_DURATION:-30}"
SERVER_METRICS="${AGENTX_SERVER_METRICS:-}"
IMAGE="${AIC_CLIENT_IMAGE:-aic-client:latest}"
OUTPUT_DIR=""
DRY_RUN="false"
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
# docker reads a relative -v source as a volume name, not a host path.
[[ "${HF_CACHE}" == /* ]] || HF_CACHE="${PWD}/${HF_CACHE}"
LOG_BASE="${LOG:-${REPO_ROOT}/logs}"
VLLM_CONTAINER="${AIC_STACK}-vllm-gpu${GPU:-0}"
EXTRA_ARGS=()
if [[ -n "${AGENTX_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${AGENTX_EXTRA_ARGS}"
fi

# ── arg parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint)       ENDPOINT="$2";        shift 2 ;;
    --network)        NETWORK="$2";         shift 2 ;;
    --model)          MODEL="$2";           shift 2 ;;
    --tokenizer)      TOKENIZER="$2";       shift 2 ;;
    --concurrency)    CONCURRENCY="$2";     shift 2 ;;
    --duration)       DURATION="$2";        shift 2 ;;
    --max-context)    MAX_CONTEXT="$2";     shift 2 ;;
    --dataset)        DATASET="$2";         shift 2 ;;
    --corpus)         CORPUS="$2";          shift 2 ;;
    --corpus-scale)   CORPUS_SCALE="$2";    shift 2 ;;
    --seed)           SEED="$2";            shift 2 ;;
    --slice-duration) SLICE_DURATION="$2";  shift 2 ;;
    --server-metrics) SERVER_METRICS="${SERVER_METRICS:+${SERVER_METRICS} }$2"; shift 2 ;;
    --output-dir)     OUTPUT_DIR="$2";      shift 2 ;;
    --image)          IMAGE="$2";           shift 2 ;;
    --dry-run)        DRY_RUN="true";       shift ;;
    --)               shift; EXTRA_ARGS+=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

for _pair in "concurrency:${CONCURRENCY}" "duration:${DURATION}" "seed:${SEED}" \
             "slice duration:${SLICE_DURATION}" "max context:${MAX_CONTEXT}"; do
  [[ -n "${_pair#*:}" || "${_pair%%:*}" != "max context" ]] || continue
  if [[ ! "${_pair#*:}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${_pair%%:*} must be a whole number, got '${_pair#*:}'" >&2
    exit 1
  fi
done
if [[ -n "${CORPUS}" && -n "${CORPUS_SCALE}" ]]; then
  echo "ERROR: give --corpus or --corpus-scale, not both" >&2
  exit 1
fi

# The preflight runs curl from AIC_FETCH_IMAGE; the client image has curl, so
# it pulls nothing.
AIC_FETCH_IMAGE="${AIC_FETCH_IMAGE:-${IMAGE}}"

# ── endpoint and network ─────────────────────────────────────────────────────
# Nothing in the stack publishes a host port, so the client reaches the server
# by container name on the stack's network.  An --endpoint override may name
# another container there or a remote server; a bridge network routes out
# either way, so the client joins the network whenever it exists.
NOTES=()
NETWORK_ARGS=()
ENDPOINT_GIVEN="${ENDPOINT:+true}"
if docker network inspect "${NETWORK}" &>/dev/null; then
  NETWORK_ARGS=(--network "${NETWORK}")
elif [[ -z "${ENDPOINT_GIVEN}" ]]; then
  # No stack to resolve: only a dry run gets past the lookup below.
  NETWORK_ARGS=(--network "${NETWORK}")
else
  NOTES+=("docker network '${NETWORK}' does not exist: the client runs on Docker's default bridge")
fi

if [[ -z "${ENDPOINT}" ]]; then
  if ! ENDPOINT="$(aic_bridge_url "${VLLM_CONTAINER}" 8000 2>/dev/null)"; then
    if [[ "${DRY_RUN}" != "true" ]]; then
      aic_bridge_url "${VLLM_CONTAINER}" 8000 >/dev/null || true
      echo "ERROR: ${VLLM_CONTAINER} is not running — start the stack with 'make up', or pass --endpoint" >&2
      exit 1
    fi
    ENDPOINT="http://<netns-owner-of-${VLLM_CONTAINER}>:8000"
  fi
fi
ENDPOINT="${ENDPOINT%/}"

# The stack's LMCache server exports Prometheus metrics on 9091; aiperf adds
# them to the endpoint's own /metrics, which it always records.
if [[ -z "${SERVER_METRICS}" && -z "${ENDPOINT_GIVEN}" ]]; then
  if _lmcache="$(aic_bridge_url "${AIC_STACK}-lmcache" 9091 2>/dev/null)"; then
    SERVER_METRICS="${_lmcache}/metrics"
  elif [[ "${DRY_RUN}" == "true" ]]; then
    SERVER_METRICS="http://<netns-owner-of-${AIC_STACK}-lmcache>:9091/metrics"
  fi
fi
SERVER_METRICS_ARGS=()
if [[ -n "${SERVER_METRICS}" && "${SERVER_METRICS}" != "none" ]]; then
  # shellcheck disable=SC2206  # a space-separated URL list, split on purpose
  SERVER_METRICS_ARGS=(--server-metrics ${SERVER_METRICS})
fi

# A server on the docker host is reachable as host.docker.internal.
HOST_ARGS=(--add-host=host.docker.internal:host-gateway)

# One GET from where the client will run.  Not aic_fetch, which cannot add the
# host entry the client gets.
fetch() {
  docker run --rm "${NETWORK_ARGS[@]}" "${HOST_ARGS[@]}" --entrypoint curl \
    "${AIC_FETCH_IMAGE}" \
    -sS --fail-with-body --max-time "${AIC_FETCH_TIMEOUT:-60}" "$1"
}

# ── image and endpoint preflight ─────────────────────────────────────────────
if [[ "${DRY_RUN}" == "true" ]]; then
  [[ -n "${MODEL}" ]] || MODEL="<served-model>"
  [[ -n "${MAX_CONTEXT}" ]] || MAX_CONTEXT="<served-max_model_len>"
else
  if ! docker image inspect "${IMAGE}" &>/dev/null; then
    echo "ERROR: image ${IMAGE} not found. Build it with: make client-build" >&2
    exit 1
  fi
  echo "Checking ${ENDPOINT}/v1/models from ${NETWORK_ARGS[1]:-the default bridge} ..." >&2
  if ! _models="$(fetch "${ENDPOINT}/v1/models")"; then
    echo "ERROR: ${ENDPOINT}/v1/models did not answer: ${_models}" >&2
    exit 1
  fi
  # Prints "<model id>\t<max_model_len or empty>" for the wanted model, or for
  # the first one served when none is named.
  if ! _served="$(printf '%s' "${_models}" | python3 -c '
import json, sys
want = sys.argv[1]
models = json.load(sys.stdin)["data"]
ids = [m["id"] for m in models]
if not ids:
    sys.exit("the endpoint serves no model")
if want and want not in ids:
    sys.exit(f"model {want!r} is not served; served: {ids}")
model = next(m for m in models if m["id"] == (want or ids[0]))
print(model["id"], model.get("max_model_len") or "", sep="\t")
' "${MODEL}")"; then
    echo "ERROR: cannot use ${ENDPOINT}/v1/models (see above)" >&2
    exit 1
  fi
  MODEL="${_served%%$'\t'*}"
  if [[ -z "${MAX_CONTEXT}" ]]; then
    MAX_CONTEXT="${_served#*$'\t'}"
    if [[ -z "${MAX_CONTEXT}" ]]; then
      echo "ERROR: ${ENDPOINT}/v1/models reports no max_model_len for ${MODEL}; pass --max-context" >&2
      exit 1
    fi
    NOTES+=("context limit ${MAX_CONTEXT} is the served model's max_model_len")
  fi
fi
[[ -n "${TOKENIZER}" ]] || TOKENIZER="${MODEL}"

# ── corpus ───────────────────────────────────────────────────────────────────
UNSAFE_REASONS=()
if [[ -n "${CORPUS_SCALE}" ]]; then
  if [[ "${DRY_RUN}" == "true" ]]; then
    CORPUS="$(AIC_CLIENT_IMAGE="${IMAGE}" bash "${SCRIPT_DIR}/scaled_corpus.sh" --scale "${CORPUS_SCALE}" --print-dir)"
  else
    CORPUS="$(AIC_CLIENT_IMAGE="${IMAGE}" bash "${SCRIPT_DIR}/scaled_corpus.sh" --scale "${CORPUS_SCALE}")"
  fi
  UNSAFE_REASONS+=("the corpus is scaled by 1/${CORPUS_SCALE}: an AgentX-shaped workload, NOT the AgentX benchmark; do not compare it with published AgentX or InferenceX numbers")
elif [[ -n "${CORPUS}" ]]; then
  if [[ "${DRY_RUN}" != "true" && ! -d "${CORPUS}" ]]; then
    echo "ERROR: corpus directory ${CORPUS} not found" >&2
    exit 1
  fi
  if [[ -d "${CORPUS}" ]]; then
    CORPUS="$(cd "${CORPUS}" && pwd)"
  fi
  if [[ -f "${CORPUS}/SCALED-CORPUS.txt" ]]; then
    UNSAFE_REASONS+=("the corpus is a scaled one: an AgentX-shaped workload, NOT the AgentX benchmark; do not compare it with published AgentX or InferenceX numbers")
  else
    UNSAFE_REASONS+=("a local corpus is not the pinned AgentX dataset")
  fi
fi
if (( DURATION < MIN_VALID_DURATION )); then
  UNSAFE_REASONS+=("the duration is under ${MIN_VALID_DURATION} s, too short to reach steady state")
fi

DATASET_ARGS=(--public-dataset "${DATASET}")
DATASET_LABEL="${DATASET} (public)"
CORPUS_MOUNT=()
if [[ -n "${CORPUS}" ]]; then
  DATASET_ARGS=(--custom-dataset-type weka-trace --input-file /data)
  DATASET_LABEL="${CORPUS} (local weka-trace directory)"
  CORPUS_MOUNT=(-v "${CORPUS}:/data:ro")
fi
UNSAFE_ARGS=()
if (( ${#UNSAFE_REASONS[@]} )); then
  UNSAFE_ARGS=(--unsafe-override)
fi

# ── output directory ─────────────────────────────────────────────────────────
[[ -n "${OUTPUT_DIR}" ]] || OUTPUT_DIR="${LOG_BASE}/agentx/$(date +%Y%m%dT%H%M%S)_c${CONCURRENCY}"
[[ "${OUTPUT_DIR}" == /* ]] || OUTPUT_DIR="${PWD}/${OUTPUT_DIR}"

# ── client environment ───────────────────────────────────────────────────────
# Passed by name, so no value (a token) appears in the printed command.
ENV_ARGS=()
CA_MOUNTS=()
for _var in HF_TOKEN NO_PROXY no_proxy REQUESTS_CA_BUNDLE SSL_CERT_FILE; do
  [[ -n "${!_var:-}" ]] || continue
  ENV_ARGS+=(-e "${_var}")
  case "${_var}" in
    REQUESTS_CA_BUNDLE|SSL_CERT_FILE)
      if [[ -f "${!_var}" && " ${CA_MOUNTS[*]} " != *" ${!_var}:${!_var}:ro "* ]]; then
        CA_MOUNTS+=(-v "${!_var}:${!_var}:ro")
      fi ;;
  esac
done

AIPERF_ARGS=(
  profile
    --scenario inferencex-agentx-mvp
    "${UNSAFE_ARGS[@]}"
    --url "${ENDPOINT}"
    --model "${MODEL}"
    --tokenizer "${TOKENIZER}"
    "${DATASET_ARGS[@]}"
    --max-context-length "${MAX_CONTEXT}"
    --concurrency "${CONCURRENCY}"
    --benchmark-duration "${DURATION}"
    --random-seed "${SEED}"
    --slice-duration "${SLICE_DURATION}"
    --no-gpu-telemetry
    "${SERVER_METRICS_ARGS[@]}"
    --output-artifact-dir /results
    "${EXTRA_ARGS[@]}"
)

DOCKER_CMD=(
  docker run --rm
    "${NETWORK_ARGS[@]}"
    "${HOST_ARGS[@]}"
    --user "$(id -u):$(id -g)"
    -e HOME=/tmp
    "${ENV_ARGS[@]}"
    -v "${OUTPUT_DIR}:/results"
    -v "${HF_CACHE}:/hf_cache"
    "${CORPUS_MOUNT[@]}"
    "${CA_MOUNTS[@]}"
    -w /results
    --entrypoint aiperf
    "${IMAGE}" "${AIPERF_ARGS[@]}"
)

summary() {
  echo "=== AIC AgentX Benchmark — $(date) ==="
  echo "Endpoint:       ${ENDPOINT}"
  echo "Network:        ${NETWORK_ARGS[1]:-Docker default bridge}"
  echo "Model:          ${MODEL}"
  echo "Tokenizer:      ${TOKENIZER}"
  echo "Dataset:        ${DATASET_LABEL}"
  echo "Context limit:  ${MAX_CONTEXT} tokens"
  echo "Concurrency:    ${CONCURRENCY}"
  echo "Duration:       ${DURATION}s"
  echo "Seed:           ${SEED}"
  echo "Server metrics: ${ENDPOINT}/metrics ${SERVER_METRICS_ARGS[*]:1}"
  echo "Output:         ${OUTPUT_DIR}"
  echo "Image:          ${IMAGE}"
  local _note
  for _note in "${NOTES[@]}"; do
    echo "NOTE: ${_note}"
  done
  for _note in "${UNSAFE_REASONS[@]}"; do
    echo "NOTE: --unsafe-override added, the run is not a valid AgentX submission: ${_note}"
  done
  echo ""
  printf 'Command:'
  printf ' %q' "${DOCKER_CMD[@]}"
  printf '\n\n'
}

if [[ "${DRY_RUN}" == "true" ]]; then
  summary
  echo "(dry-run — not executing)"
  exit 0
fi

# ── run ───────────────────────────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}" "${HF_CACHE}"
RUN_LOG="${OUTPUT_DIR}/run.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

summary
echo "Starting benchmark at $(date)..."
_rc=0
"${DOCKER_CMD[@]}" || _rc=$?
if (( _rc != 0 )); then
  echo "aiperf exited with status ${_rc}. See ${RUN_LOG}" >&2
  echo "If it found no eligible traces, the context limit (${MAX_CONTEXT}) is too small for the" >&2
  echo "dataset: serve a larger context, or use a scaled corpus (--corpus-scale, see README.md)." >&2
  exit 1
fi

echo ""
echo "Benchmark complete at $(date)."
echo "Results: ${OUTPUT_DIR}"
