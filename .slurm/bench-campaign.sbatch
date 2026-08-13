#!/bin/bash
#SBATCH --job-name=aic-bench
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=logs/bench/campaign-%j.out
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Slurm submitter for the serving-benchmark harness: claim a whole node, sweep
# any stale containers, then run some subset of bench-matrix.sh, bench-matrix2.sh
# and l2-gate.sh into one results file.
#
# --exclusive and --gres=gpu:8 are correctness requirements, not a resource grab.
# Most arms use a single GPU, but the numbers are only comparable across arms if
# nothing else is resident on the node -- a co-tenant perturbs exactly what is
# being measured -- and the TP=8 arm needs all eight devices to exist.
#
# Env:
#   STAGES  which stages to run   (default: "matrix matrix2 gate")
#   ARMS    arm filter, applied to both matrix stages    (default: every arm)
#   OUT     results file    (default: logs/bench/results-<jobid>.txt)
#   AIC_IMAGE / HF_HOME     as elsewhere in .slurm/
#
# Submit:
#   source .env.aic
#   sbatch --partition="${AIC_TEST_PARTITION}" --account="${AIC_TEST_ACCOUNT}" \
#          .slurm/bench-campaign.sbatch
#
# A subset -- note --export=ALL, or the job starts with an empty environment:
#   sbatch --partition="${AIC_TEST_PARTITION}" --account="${AIC_TEST_ACCOUNT}" \
#          --export=ALL,STAGES=matrix,ARMS="tp2 tp4" \
#          .slurm/bench-campaign.sbatch
set -uo pipefail

AIC_DAY_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
export AIC_DAY_DIR
export AIC_IMAGE="${AIC_IMAGE:-rocm-aic:0.1.0-rocm7.14.0-vllm0.26.0-lmcache0.5.3-nixl1.3.2-hsasnoop1.0.0}"
export HF_HOME="${HF_HOME:-/projects/hf_cache}"
cd "${AIC_DAY_DIR}" || exit 1

STAGES="${STAGES:-matrix matrix2 gate}"
# One file for every stage: both matrix scripts append, so a campaign's arms stay
# together and .slurm/bench-table.py renders the lot in one pass.
OUT="${OUT:-${AIC_DAY_DIR}/logs/bench/results-${SLURM_JOB_ID:-$$}.txt}"
export OUT
mkdir -p "$(dirname "${OUT}")"

echo "=== campaign start $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"
echo "=== stages='${STAGES}' arms='${ARMS:-all}' out=${OUT}"

# A previous run that was killed rather than exited never ran its EXIT trap, so
# its containers are still holding GPU memory and the container names this run
# needs.  Sweep them before anything else.
for c in aic-vllm-gpu0 aic-lmcache aic-lmcache-coordinator aic-client; do
    docker rm -f "$c" >/dev/null 2>&1 && echo "removed stale ${c}"
done
docker network rm aic >/dev/null 2>&1 || true

for stage in ${STAGES}; do
    echo "=== stage ${stage} $(date -Is)"
    case "${stage}" in
        matrix)  bash "${AIC_DAY_DIR}/.slurm/bench-matrix.sh" ;;
        matrix2) bash "${AIC_DAY_DIR}/.slurm/bench-matrix2.sh" ;;
        # The gate's failures are reported, not fatal: it runs last, and one bad
        # L2 backend must not discard a matrix that already completed.
        gate)    bash "${AIC_DAY_DIR}/.slurm/l2-gate.sh" || echo "l2-gate FAILED (rc=$?)" ;;
        *)       echo "unknown stage '${stage}' -- expected matrix|matrix2|gate" >&2 ;;
    esac
done

echo "=== campaign complete $(date -Is)"
