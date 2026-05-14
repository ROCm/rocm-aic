#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# List NVMe namespace block devices (nvme*n*) with lsblk mount state.
# Does not mount or format. Use on Slurm compute nodes to pick KURT_NVME_DEVICE.
#
# Exit 0 always; prints human lines to stdout. Machine-readable candidates:
#   CANDIDATE\t/dev/nvme0n1\tsize_gib\treason
#
# Usage: ./discover_unmounted_nvme.sh

set -euo pipefail

json="${1:-}"

has_mount_under() {
  local dev="$1"
  lsblk -nr "${dev}" -o MOUNTPOINT 2>/dev/null | awk '
    NF == 0 { next }
    $1 == "-" { next }
    { print $1; exit 0 }
  ' | grep -q .
}

size_gib() {
  local dev="$1"
  lsblk -ndo SIZE -b "${dev}" 2>/dev/null | awk '{printf "%.1f", $1/1024/1024/1024}'
}

emit_candidate() {
  local dev="$1" reason="$2"
  local sz
  sz="$(size_gib "${dev}")"
  if [[ "${json}" == "--json" ]]; then
    printf '{"device":"%s","size_gib":%s,"note":"%s"}\n' "${dev}" "${sz}" "${reason}"
  else
    printf 'CANDIDATE\t%s\t%s GiB\t%s\n' "${dev}" "${sz}" "${reason}"
  fi
}

if [[ "${json}" != "" && "${json}" != "--json" ]]; then
  echo "usage: $0 [--json]" >&2
  exit 2
fi

echo "=== NVMe block devices (lsblk) ==="
if ! lsblk -d -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,MODEL,TRAN 2>/dev/null | grep -E '^nvme|NAME'; then
  lsblk -d -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,MODEL,TRAN 2>/dev/null || true
fi

echo ""
echo "=== Whole-disk namespaces (TYPE=disk) with no active mount points ==="
cand_count=0
while read -r name type _rest; do
  [[ "${type}" == "disk" ]] || continue
  [[ "${name}" =~ ^nvme[0-9]+n[0-9]+$ ]] || continue
  dev="/dev/${name}"
  [[ -b "${dev}" ]] || continue
  if has_mount_under "${dev}"; then
    echo "skip ${dev}: has mounted descendant or self-mounted"
    continue
  fi
  reason="no lsblk MOUNTPOINT on this disk or its partitions"
  emit_candidate "${dev}" "${reason}"
  cand_count=$((cand_count + 1))
done < <(lsblk -d -n -o NAME,TYPE 2>/dev/null || true)

if [[ "${cand_count}" -eq 0 ]]; then
  echo "(no unmounted whole-disk nvme* namespaces found; site layout may use partitions only)"
fi

echo ""
echo "=== hints ==="
echo "Set KURT_NVME_DEVICE=/dev/nvmeXn1 and KURT_NVME_MOUNT=/mnt/... then KURT_NVME_MKFS=1 for first-time XFS (DESTRUCTIVE)."
echo "KURT_NVME_AUTO_DEVICE=1 with KURT_NVME_MKFS=1 picks the first candidate above (exclusive nodes only)."
