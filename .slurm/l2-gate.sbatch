#!/bin/bash
#SBATCH --job-name=aic-gate2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=logs/bench/gate2-%j.out
set -uo pipefail
export AIC_DAY_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
export AIC_IMAGE="${AIC_IMAGE:-rocm-aic:0.1.0-rocm7.14.0-vllm0.26.0-lmcache0.5.3-nixl1.3.2-hsasnoop1.0.0}"
export HF_HOME="${HF_HOME:-/projects/hf_cache}"
cd "${AIC_DAY_DIR}" || exit 1
for c in aic-vllm-gpu0 aic-lmcache aic-lmcache-coordinator aic-client; do docker rm -f "$c" >/dev/null 2>&1; done
L2_BACKENDS="${L2_BACKENDS:-nixl_posix local_disk nixl}" bash "${AIC_DAY_DIR}/.slurm/l2-gate.sh"
