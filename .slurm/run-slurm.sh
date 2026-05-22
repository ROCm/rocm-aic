#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Submit vllm-from-kurt MI300 job with NVMe block bpftrace, LMCache VFS bpftrace
# (when KURT_NVME_BLK_BPFTRACE=1), SMART snapshots, and LMCache KV events.
# KURT_NVME_BASE — **set KURT_NVME_BASE to your local NVMe mount** (e.g.
# /mnt/nvme) so HF + LMCache do not fill the root filesystem; see sbatch header.
#
# Exclusive-node mkfs + auto device (only when an unmounted nvme*n* exists):
#   --export=ALL,KURT_NVME_MKFS=1,KURT_NVME_AUTO_DEVICE=1,KURT_NVME_MOUNT=/mnt/kurt-nvme,...
# Or set KURT_NVME_DEVICE=/dev/nvme0n1 explicitly with KURT_NVME_MKFS=1.
#
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

out="$(sbatch \
  --export=ALL,KURT_NVME_BLK_BPFTRACE=1,KURT_NVME_SMART_LOG=1,KURT_LMCACHE_ENABLE_KV_EVENTS=1 \
  .slurm/vllm-from-kurt-mi300.sbatch)"
echo "${out}"
jid="${out##* }"
echo "Log: tail -f .slurm/logs/vllm-from-kurt-mi300-${jid}.log"
echo "Done:  sacct -j ${jid} --format=JobID,State,ExitCode,Elapsed,End"

# Example (local NVMe + traces): sbatch --export=ALL,KURT_NVME_BASE=/mnt/nvme,KURT_NVME_BLK_BPFTRACE=1 \
#   .slurm/vllm-from-kurt-mi300.sbatch
