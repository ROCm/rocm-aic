#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Submit vllm-radeon MI300 job (Docker build + run + long_doc_qa benchmark).
#
# Set RADEON_NVME_BASE to your node NVMe mount (e.g. /mnt/nvme) so HF + LMCache
# do not fill the root filesystem.
#
#   export HF_TOKEN=...                    # or HF_TOKEN_FILE=~/.hf_token
#   export RADEON_NVME_BASE=/mnt/nvme
#   bash .slurm/run-vllm-radeon.sh
#
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p .slurm/logs

_exports="ALL"
if [[ -n "${RADEON_NVME_BASE:-}" ]]; then
    _exports="${_exports},RADEON_NVME_BASE=${RADEON_NVME_BASE}"
fi
for _flag in \
    RADEON_NVME_BLK_BPFTRACE \
    RADEON_NVME_SMART_LOG \
    RADEON_LMCACHE_ENABLE_KV_EVENTS \
    RADEON_VFS_BPFTRACE \
    RADEON_BENCHMARK \
    RADEON_SKIP_BUILD; do
    if [[ -n "${!_flag:-}" ]]; then
        _exports="${_exports},${_flag}=${!_flag}"
    fi
done

# Defaults when not set in the environment
: "${RADEON_NVME_BLK_BPFTRACE:=1}"
: "${RADEON_NVME_SMART_LOG:=1}"
: "${RADEON_LMCACHE_ENABLE_KV_EVENTS:=1}"
_exports="${_exports},RADEON_NVME_BLK_BPFTRACE=${RADEON_NVME_BLK_BPFTRACE}"
_exports="${_exports},RADEON_NVME_SMART_LOG=${RADEON_NVME_SMART_LOG}"
_exports="${_exports},RADEON_LMCACHE_ENABLE_KV_EVENTS=${RADEON_LMCACHE_ENABLE_KV_EVENTS}"

out="$(sbatch --export="${_exports}" .slurm/vllm-radeon-mi300.sbatch)"
echo "${out}"
jid="${out##* }"
echo "Log:  tail -f .slurm/logs/vllm-radeon-${jid}.log"
echo "Artifacts: .slurm/logs/vllm-radeon-${jid}/"
echo "Done:  sacct -j ${jid} --format=JobID,State,ExitCode,Elapsed,End"
