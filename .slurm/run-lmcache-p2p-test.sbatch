#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Sbatch wrapper for run-lmcache-p2p-test.sh.
# Submitted by: make test-lmcache-p2p-spur
#
#SBATCH --job-name=aic-lmcache-p2p
#SBATCH --partition=amd-spur
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=60

set -euo pipefail
export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
exec bash "${SLURM_SUBMIT_DIR}/.slurm/run-lmcache-p2p-test.sh"
