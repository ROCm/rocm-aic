#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Deprecated: vllm-from-kurt-mi300.sbatch and recipies/vllm-from-kurt/ were removed.
# Use run-vllm-radeon.sh for the vllm-radeon Slurm workflow.
#
set -euo pipefail
echo "NOTE: vllm-from-kurt Slurm recipe was removed; delegating to run-vllm-radeon.sh" >&2
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run-vllm-radeon.sh" "$@"
