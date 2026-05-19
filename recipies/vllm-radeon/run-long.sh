#!/bin/bash

# Random chunk + random question from BOOK_DATA_DIR (override CONTEXT_FILE / QUESTION).
# Env: BASE_URL, MODEL, BOOK_SLUG, BOOK_DATA_DIR, QUESTIONS_FILE, CONTEXT_FILE,
#      QUESTION, ITERATIONS.

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
BOOK_SLUG="${BOOK_SLUG:-war-and-peace}"
BOOK_DATA_DIR="${BOOK_DATA_DIR:-${here}/data/${BOOK_SLUG}}"
QUESTIONS_FILE="${QUESTIONS_FILE:-${BOOK_DATA_DIR}/${BOOK_SLUG}.questions.json}"
ITERATIONS="${ITERATIONS:-1}"

if ! [[ "${ITERATIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: ITERATIONS must be a positive integer (got ${ITERATIONS})" >&2
  exit 1
fi

chunks=()
if [[ -n "${CONTEXT_FILE:-}" ]]; then
  if ! [[ -f "${CONTEXT_FILE}" ]]; then
    echo "error: CONTEXT_FILE is not a readable file: ${CONTEXT_FILE}" >&2
    exit 1
  fi
else
  shopt -s nullglob
  chunks=( "${BOOK_DATA_DIR}"/${BOOK_SLUG}-*.txt )
  shopt -u nullglob
  if [[ ${#chunks[@]} -eq 0 ]]; then
    echo "error: no chunk files matching ${BOOK_DATA_DIR}/${BOOK_SLUG}-*.txt" >&2
    exit 1
  fi
fi

n=""
if [[ -n "${QUESTION+x}" ]]; then
  :
else
  if ! [[ -f "${QUESTIONS_FILE}" ]]; then
    echo "error: missing questions file ${QUESTIONS_FILE}" >&2
    exit 1
  fi
  n=$(jq '.questions | length' "${QUESTIONS_FILE}")
  if ! [[ "${n}" =~ ^[0-9]+$ ]] || [[ "${n}" -lt 1 ]]; then
    echo "error: .questions must be a non-empty array in ${QUESTIONS_FILE}" >&2
    exit 1
  fi
fi

resp_file=$(mktemp)
trap 'rm -f "${resp_file}"' EXIT

for ((i = 1; i <= ITERATIONS; i++)); do
  echo "run-long: iteration=${i}/${ITERATIONS}" >&2

  if [[ -n "${CONTEXT_FILE:-}" ]]; then
    ctxt="${CONTEXT_FILE}"
  else
    ctxt="${chunks[RANDOM % ${#chunks[@]}]}"
  fi
  echo "run-long: context=${ctxt}" >&2

  if [[ -n "${QUESTION+x}" ]]; then
    qtext="${QUESTION}"
  else
    qidx=$((RANDOM % n))
    qtext=$(jq -r --argjson idx "${qidx}" '.questions[$idx]' "${QUESTIONS_FILE}")
    echo "run-long: question_index=${qidx} file=${QUESTIONS_FILE}" >&2
  fi

  read -r http_code elapsed < <(
    jq -n \
      --arg model "$MODEL" \
      --arg question "$qtext" \
      --rawfile context "$ctxt" \
      '{
        model: $model,
        messages: [
          { role: "system", content: "Answer using only the provided context." },
          { role: "user", content: ("Context:\n\n" + $context + "\n\nQuestion: " + $question) }
        ],
        max_tokens: 512,
        temperature: 0.2
      }' \
      | curl -sS "${BASE_URL}/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -o "${resp_file}" \
          -w '%{http_code} %{time_total}' \
          -d @-
  )

  if [[ "${http_code}" != 2* ]]; then
    echo "HTTP ${http_code} (client_wall_time_seconds=${elapsed}) iteration=${i}/${ITERATIONS}" >&2
    cat "${resp_file}" >&2
    exit 1
  fi

  if [[ "${i}" -gt 1 ]]; then
    echo
  fi
  jq --argjson http "${http_code}" --argjson wall "${elapsed}" \
    --argjson iter "${i}" --argjson total "${ITERATIONS}" '
    . + {
      http_status: $http,
      client_wall_time_seconds: $wall,
      run_long_iteration: $iter,
      run_long_iterations_total: $total
    }
  ' "${resp_file}"
done
