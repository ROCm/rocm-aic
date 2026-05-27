#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Thin wrapper for vllm-lmcache-hipfile and vllm-lmcache-nixl Slurm jobs.
# Shared logic lives in recipies/common/slurm/lib/recipe-common.sh.

if [[ -z "${REPO_DIR:-}" ]]; then
	REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
# shellcheck source=../../../recipies/common/slurm/lib/recipe-common.sh
source "${REPO_DIR}/recipies/common/slurm/lib/recipe-common.sh"
