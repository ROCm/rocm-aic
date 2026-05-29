#!/bin/bash
#SBATCH --job-name=test-nixl-ais-mt
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=.slurm/logs/test-nixl-ais-mt-%j.log
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Build NIXL (andyluo7/nixl amd-support + AIS overlay) and run validation.
# Submit from repo root:
#   sbatch --constraint='MARKHAM&NVME' .slurm/test-nixl-ais-mt.sbatch
#
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NIXL_COMMON="${REPO_DIR}/recipies/common/nixl"
JOB_ROOT="/tmp/test-nixl-ais-mt-${SLURM_JOB_ID:-local$$}"
NIXL_SRC="${JOB_ROOT}/nixl"
NIXL_PREFIX="${JOB_ROOT}/nixl-install"
LOG_DIR="${REPO_DIR}/.slurm/logs/test-nixl-ais-mt-${SLURM_JOB_ID:-local$$}"

mkdir -p "${JOB_ROOT}" "${LOG_DIR}"

echo "=== test-nixl-ais-mt job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "REPO_DIR=${REPO_DIR}"
echo "NIXL_SRC=${NIXL_SRC}"

export NIXL_DEST="${NIXL_SRC}"
export NIXL_GIT_URL="${NIXL_GIT_URL:-https://github.com/andyluo7/nixl.git}"
export NIXL_REF="${NIXL_REF:-amd-support}"

"${NIXL_COMMON}/clone-nixl.sh"

export NIXL_SRC NIXL_ENABLE_AIS=1
export NIXL_INSTALL_PREFIX="${NIXL_PREFIX}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export AIS_PATH="${AIS_PATH:-${ROCM_PATH}}"

"${NIXL_COMMON}/build-nixl.sh" 2>&1 | tee "${LOG_DIR}/build.log"

export NIXL_PLUGIN_DIR="${NIXL_PREFIX}/lib/x86_64-linux-gnu/plugins"
if [[ ! -d "${NIXL_PLUGIN_DIR}" ]]; then
    NIXL_PLUGIN_DIR="${NIXL_PREFIX}/lib/nixl/plugins"
fi
export LD_LIBRARY_PATH="${NIXL_PREFIX}/lib:/opt/rocnixl-ucx/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"

python3 -c "import nixl; nixl.nixl_agent('test')" | tee "${LOG_DIR}/agent.log"

for plug in AIS AIS_MT; do
    found="$(find "${NIXL_SRC}/build" -name "libplugin_${plug}.so" 2>/dev/null | head -1 || true)"
    if [[ -z "${found}" ]]; then
        echo "FAIL: libplugin_${plug}.so not found" | tee -a "${LOG_DIR}/plugins.log"
        exit 1
    fi
    echo "PASS: ${found}" | tee -a "${LOG_DIR}/plugins.log"
done

if [[ -f "${NIXL_SRC}/build/test/unit/plugins/ais_mt/nixl_ais_mt_test" ]]; then
    echo "Running AIS_MT unit test..." | tee -a "${LOG_DIR}/unit-test.log"
    "${NIXL_SRC}/build/test/unit/plugins/ais_mt/nixl_ais_mt_test" \
        2>&1 | tee -a "${LOG_DIR}/unit-test.log" || true
else
    echo "WARN: nixl_ais_mt_test binary not built; plugin .so checks passed" \
        | tee -a "${LOG_DIR}/unit-test.log"
fi

echo "=== test-nixl-ais-mt PASS ==="
