#!/bin/bash

# Random chunk + random question from one book or the whole library.
# Env: BASE_URL, MODEL, BOOK_DATA_ROOT, BOOK_SLUG, BOOK_DATA_DIR,
#      QUESTIONS_FILE, CONTEXT_FILE, QUESTION, ITERATIONS,
#      RUN_LONG_SEED, RUN_LONG_WORKER, RUN_LONG_COMBINE_CHUNKS.
# Library mode (default): unset BOOK_SLUG; scans BOOK_DATA_ROOT/*/ for fixtures.
# Single-book mode: set BOOK_SLUG (e.g. war-and-peace).
# RUN_LONG_COMBINE_CHUNKS=2 concatenates two random 10k chunks (~20k words).
# Parallel load: use run-long-parallel.sh (distinct RUN_LONG_SEED per worker).

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
BOOK_DATA_ROOT="${BOOK_DATA_ROOT:-${here}/data}"
ITERATIONS="${ITERATIONS:-1}"
RUN_LONG_WORKER="${RUN_LONG_WORKER:-}"
RUN_LONG_COMBINE_CHUNKS="${RUN_LONG_COMBINE_CHUNKS:-1}"

if ! [[ "${ITERATIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: ITERATIONS must be a positive integer (got ${ITERATIONS})" >&2
  exit 1
fi

if ! [[ "${RUN_LONG_COMBINE_CHUNKS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: RUN_LONG_COMBINE_CHUNKS must be a positive integer (got ${RUN_LONG_COMBINE_CHUNKS})" >&2
  exit 1
fi

if [[ -n "${CONTEXT_FILE:-}" && "${RUN_LONG_COMBINE_CHUNKS}" -gt 1 ]]; then
  echo "run-long: CONTEXT_FILE set; ignoring RUN_LONG_COMBINE_CHUNKS" >&2
fi

if [[ -n "${RUN_LONG_SEED+x}" && -n "${RUN_LONG_SEED}" ]]; then
  if ! [[ "${RUN_LONG_SEED}" =~ ^[0-9]+$ ]]; then
    echo "error: RUN_LONG_SEED must be a non-negative integer (got ${RUN_LONG_SEED})" >&2
    exit 1
  fi
  RANDOM="${RUN_LONG_SEED}"
  _seed_log=" seed=${RUN_LONG_SEED}"
else
  _seed_log=""
fi

if [[ -n "${RUN_LONG_WORKER}" ]]; then
  if ! [[ "${RUN_LONG_WORKER}" =~ ^[0-9]+$ ]]; then
    echo "error: RUN_LONG_WORKER must be a non-negative integer (got ${RUN_LONG_WORKER})" >&2
    exit 1
  fi
  _worker_log=" worker=${RUN_LONG_WORKER}"
else
  _worker_log=""
fi

BOOK_SLUG_MODE="${BOOK_SLUG:-}"
BOOK_DATA_DIR=""
QUESTIONS_FILE=""

library_books=()
library_slugs=()

discover_library_books() {
  library_books=()
  library_slugs=()
  local d slug questions
  shopt -s nullglob
  for d in "${BOOK_DATA_ROOT}"/*/; do
    slug=$(basename "${d%/}")
    questions="${d}/${slug}.questions.json"
    if [[ -f "${questions}" ]] && compgen -G "${d}/${slug}-"'*.txt' > /dev/null; then
      library_books+=("${d%/}")
      library_slugs+=("${slug}")
    fi
  done
  shopt -u nullglob
}

if [[ -n "${BOOK_SLUG_MODE}" ]]; then
  BOOK_DATA_DIR="${BOOK_DATA_DIR:-${BOOK_DATA_ROOT}/${BOOK_SLUG_MODE}}"
  QUESTIONS_FILE="${QUESTIONS_FILE:-${BOOK_DATA_DIR}/${BOOK_SLUG_MODE}.questions.json}"
elif [[ -n "${CONTEXT_FILE:-}" ]]; then
  :
else
  discover_library_books
  if [[ ${#library_books[@]} -eq 0 ]]; then
    echo "error: no book directories under ${BOOK_DATA_ROOT} " \
         "(expected <slug>/<slug>.questions.json and <slug>-*.txt chunks; " \
         "run make data-all)" >&2
    exit 1
  fi
  echo "run-long: library mode books=${#library_books[@]} root=${BOOK_DATA_ROOT}${_worker_log}${_seed_log}" >&2
fi

chunks=()
if [[ -n "${CONTEXT_FILE:-}" ]]; then
  if ! [[ -f "${CONTEXT_FILE}" ]]; then
    echo "error: CONTEXT_FILE is not a readable file: ${CONTEXT_FILE}" >&2
    exit 1
  fi
elif [[ -n "${BOOK_SLUG_MODE}" ]]; then
  shopt -s nullglob
  chunks=( "${BOOK_DATA_DIR}"/${BOOK_SLUG_MODE}-*.txt )
  shopt -u nullglob
  if [[ ${#chunks[@]} -eq 0 ]]; then
    echo "error: no chunk files matching ${BOOK_DATA_DIR}/${BOOK_SLUG_MODE}-*.txt" >&2
    exit 1
  fi
fi

n=""
if [[ -n "${QUESTION+x}" ]]; then
  :
elif [[ -n "${BOOK_SLUG_MODE}" ]]; then
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
combined_file=""
if [[ "${RUN_LONG_COMBINE_CHUNKS}" -gt 1 && -z "${CONTEXT_FILE:-}" ]]; then
  combined_file=$(mktemp)
fi
trap 'rm -f "${resp_file}" "${combined_file}"' EXIT

# Pick N distinct chunk paths from pool (bash 3+ compatible).
pick_distinct_chunks() {
  local n=$1
  shift
  local -a pool=("$@")
  local -a picked=()
  local c p found attempts=0
  local max_attempts=$(( n * 32 ))

  if [[ ${#pool[@]} -lt n ]]; then
    echo "error: need ${n} chunks but only ${#pool[@]} available" >&2
    return 1
  fi

  while [[ ${#picked[@]} -lt n && attempts -lt max_attempts ]]; do
    attempts=$((attempts + 1))
    c="${pool[RANDOM % ${#pool[@]}]}"
    found=0
    for p in "${picked[@]}"; do
      if [[ "${p}" == "${c}" ]]; then
        found=1
        break
      fi
    done
    if [[ "${found}" -eq 0 ]]; then
      picked+=("${c}")
    fi
  done

  if [[ ${#picked[@]} -lt n ]]; then
    echo "error: could not pick ${n} distinct chunks" >&2
    return 1
  fi

  printf '%s\n' "${picked[@]}"
}

build_combined_context() {
  local out=$1
  shift
  local -a parts=("$@")
  local part
  : > "${out}"
  for part in "${parts[@]}"; do
    cat "${part}" >> "${out}"
    printf '\n\n' >> "${out}"
  done
}

for ((i = 1; i <= ITERATIONS; i++)); do
  echo "run-long: iteration=${i}/${ITERATIONS}${_worker_log}${_seed_log}" >&2

  iter_slug=""
  iter_questions_file=""
  iter_chunks=()
  iter_context_sources=()
  ctxt=""

  if [[ -n "${CONTEXT_FILE:-}" ]]; then
    ctxt="${CONTEXT_FILE}"
    iter_context_sources=("${CONTEXT_FILE}")
  elif [[ -n "${BOOK_SLUG_MODE}" ]]; then
    iter_slug="${BOOK_SLUG_MODE}"
    iter_questions_file="${QUESTIONS_FILE}"
    iter_chunks=( "${chunks[@]}" )
  else
    bidx=$((RANDOM % ${#library_books[@]}))
    BOOK_DATA_DIR="${library_books[bidx]}"
    iter_slug="${library_slugs[bidx]}"
    iter_questions_file="${BOOK_DATA_DIR}/${iter_slug}.questions.json"
    shopt -s nullglob
    iter_chunks=( "${BOOK_DATA_DIR}"/${iter_slug}-*.txt )
    shopt -u nullglob
    if [[ ${#iter_chunks[@]} -eq 0 ]]; then
      echo "error: no chunks in ${BOOK_DATA_DIR}" >&2
      exit 1
    fi
    echo "run-long: book=${iter_slug}" >&2
  fi

  if [[ "${RUN_LONG_COMBINE_CHUNKS}" -gt 1 ]]; then
    mapfile -t iter_context_sources < <(
      pick_distinct_chunks "${RUN_LONG_COMBINE_CHUNKS}" "${iter_chunks[@]}"
    ) || exit 1
    build_combined_context "${combined_file}" "${iter_context_sources[@]}"
    ctxt="${combined_file}"
    echo "run-long: context_mode=combine chunks=${RUN_LONG_COMBINE_CHUNKS}" >&2
    for src in "${iter_context_sources[@]}"; do
      echo "run-long: context_part=${src}" >&2
    done
    echo "run-long: context_words=$(wc -w < "${ctxt}")" >&2
  elif [[ -z "${ctxt}" ]]; then
    ctxt="${iter_chunks[RANDOM % ${#iter_chunks[@]}]}"
    iter_context_sources=("${ctxt}")
  fi

  if [[ ${#iter_context_sources[@]} -eq 0 ]]; then
    iter_context_sources=("${ctxt}")
  fi
  echo "run-long: context=${ctxt}" >&2

  if [[ -n "${QUESTION+x}" ]]; then
    qtext="${QUESTION}"
  else
    if [[ -z "${iter_questions_file}" ]]; then
      iter_questions_file="${QUESTIONS_FILE}"
    fi
    if [[ -z "${iter_slug}" && -n "${BOOK_SLUG_MODE}" ]]; then
      iter_slug="${BOOK_SLUG_MODE}"
    fi
    if [[ -z "${iter_questions_file}" ]]; then
      echo "error: no questions file for iteration ${i}" >&2
      exit 1
    fi
    n=$(jq '.questions | length' "${iter_questions_file}")
    if ! [[ "${n}" =~ ^[0-9]+$ ]] || [[ "${n}" -lt 1 ]]; then
      echo "error: .questions must be a non-empty array in ${iter_questions_file}" >&2
      exit 1
    fi
    qidx=$((RANDOM % n))
    qtext=$(jq -r --argjson idx "${qidx}" '.questions[$idx]' "${iter_questions_file}")
    echo "run-long: question_index=${qidx} file=${iter_questions_file}" >&2
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
  _jq_seed='null'
  _jq_worker='null'
  if [[ -n "${RUN_LONG_SEED+x}" && -n "${RUN_LONG_SEED}" ]]; then
    _jq_seed="${RUN_LONG_SEED}"
  fi
  if [[ -n "${RUN_LONG_WORKER}" ]]; then
    _jq_worker="${RUN_LONG_WORKER}"
  fi
  _sources_json=$(printf '%s\n' "${iter_context_sources[@]}" | jq -R . | jq -s .)
  jq --argjson http "${http_code}" --argjson wall "${elapsed}" \
    --argjson iter "${i}" --argjson total "${ITERATIONS}" \
    --argjson seed "${_jq_seed}" --argjson worker "${_jq_worker}" \
    --argjson combine "${RUN_LONG_COMBINE_CHUNKS}" \
    --argjson sources "${_sources_json}" \
    --arg book "${iter_slug:-${BOOK_SLUG_MODE:-}}" \
    --arg context "${ctxt}" \
    --arg questions_file "${iter_questions_file:-${QUESTIONS_FILE:-}}" '
    . + {
      http_status: $http,
      client_wall_time_seconds: $wall,
      run_long_iteration: $iter,
      run_long_iterations_total: $total,
      run_long_seed: $seed,
      run_long_worker: $worker,
      run_long_combine_chunks: $combine,
      run_long_book: (if $book == "" then null else $book end),
      run_long_context: $context,
      run_long_context_sources: $sources,
      run_long_questions_file: (if $questions_file == "" then null else $questions_file end)
    }
  ' "${resp_file}"
done
