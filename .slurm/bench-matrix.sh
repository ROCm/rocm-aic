#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Serial driver for a bench-serve.sh matrix.
#
# Serial, not parallel, and not negotiable: every arm uses the same fixed
# container names (aic-vllm-gpu0, aic-lmcache) and the same host ports, so two
# concurrent arms would fight over both -- and a co-resident arm would perturb
# the very numbers being measured.
#
# Each arm appends one RESULT line to ${OUT}.  A failed arm records its failure
# and the matrix continues; one bad configuration must not cost the whole run.
#
# Env:
#   AIC_DAY_DIR   repo root                          (required)
#   AIC_IMAGE     image ref                          (required)
#   OUT           results file      (default: logs/bench/results-<jobid>.txt)
#   ARMS          space-separated arm names to run   (default: all)
set -uo pipefail
: "${AIC_DAY_DIR:?}" "${AIC_IMAGE:?}"
cd "${AIC_DAY_DIR}" || exit 1

OUT="${OUT:-${AIC_DAY_DIR}/logs/bench/results-${SLURM_JOB_ID:-$$}.txt}"
mkdir -p "$(dirname "${OUT}")"

# GPUs 4-7.  0-3 are left to whatever else shares this node.
export HF_HOME="${HF_HOME:-/projects/hf_cache}"
BASE_ISL=2048 BASE_OSL=512 BASE_CONC="1 8 32 64 128"

run() {  # run <name> <KEY=VAL>...
    local name="$1"; shift
    if [ -n "${ARMS:-}" ] && [[ " ${ARMS} " != *" ${name} "* ]]; then return 0; fi
    printf '\n===== %s : %s =====\n' "$(date -Is)" "${name}"
    local line
    line=$(env RUN_NAME="${name}" AIC_DAY_DIR="${AIC_DAY_DIR}" AIC_IMAGE="${AIC_IMAGE}" \
        HF_HOME="${HF_HOME}" ISL="${BASE_ISL}" OSL="${BASE_OSL}" \
        CONC_LIST="${BASE_CONC}" READY_TRIES=180 "$@" \
        bash "${AIC_DAY_DIR}/.slurm/bench-serve.sh" 2>&1 | tee /dev/stderr |
        grep '^RESULT:')
    printf '%s\n' "${line:-RESULT: name=${name} status=crashed}" >> "${OUT}"
    sleep 10   # let the GPUs settle before the next arm claims them
}

# --- correctness gate -------------------------------------------------------
run gate-conn-l1     TP=1 DEVS=4 CONNECTOR=1 L2=none  CONC_LIST="8 32"
run gate-conn-l2nvme TP=1 DEVS=4 CONNECTOR=1 L2=nvme  CONC_LIST="8 32" L1_GB=4

# --- TP scaling (no connector: isolates vLLM itself) ------------------------
run tp1 TP=1 DEVS=4       CONNECTOR=0
run tp2 TP=2 DEVS=4,5     CONNECTOR=0
run tp4 TP=4 DEVS=4,5,6,7 CONNECTOR=0

# --- connector overhead at TP=1 and TP=4 ------------------------------------
run tp1-conn TP=1 DEVS=4       CONNECTOR=1
run tp4-conn TP=4 DEVS=4,5,6,7 CONNECTOR=1

# --- optimisation knobs, all at TP=1, all against the tp1 baseline ----------
run opt-eager        TP=1 DEVS=4 CONNECTOR=0 EAGER=1
run opt-noprefix     TP=1 DEVS=4 CONNECTOR=0 PREFIX_CACHE=0
run opt-batch2048    TP=1 DEVS=4 CONNECTOR=0 BATCHED_TOK=2048
run opt-batch8192    TP=1 DEVS=4 CONNECTOR=0 BATCHED_TOK=8192
run opt-batch16384   TP=1 DEVS=4 CONNECTOR=0 BATCHED_TOK=16384
run opt-util95       TP=1 DEVS=4 CONNECTOR=0 GPU_UTIL=0.95

printf '\n===== matrix complete: %s =====\n' "$(date -Is)"
cat "${OUT}"
