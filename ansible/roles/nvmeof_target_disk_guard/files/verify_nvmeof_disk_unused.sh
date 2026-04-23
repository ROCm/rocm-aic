#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Exit 0 if this block device (and its subtree) has no mountpoints, no
# non-empty FSTYPE, no swap, and is not an active LVM PV. Otherwise exit
# non-zero with a message on stderr.
set -euo pipefail

dev="${1:?block device path required}"

if [[ ! -e "$dev" ]]; then
	echo "error: path does not exist: $dev" >&2
	exit 1
fi
if [[ ! -b "$dev" ]]; then
	echo "error: not a block device: $dev" >&2
	exit 1
fi

real=$(readlink -f "$dev")

while read -r mp fst; do
	[[ "$mp" == '-' ]] && mp=
	[[ "$fst" == '-' ]] && fst=
	if [[ -n "${mp// }" ]]; then
		echo "error: mountpoint '${mp}' under subtree of ${real} (input ${dev})" >&2
		exit 2
	fi
	if [[ -n "${fst// }" ]]; then
		echo "error: fstype '${fst}' present under subtree of ${real} (input ${dev})" >&2
		exit 3
	fi
done < <(lsblk -nr -o MOUNTPOINT,FSTYPE "$real" 2>/dev/null || true)

if command -v swapon >/dev/null 2>&1; then
	while read -r swapdev; do
		[[ -z "${swapdev// }" ]] && continue
		sreal=$(readlink -f "$swapdev" 2>/dev/null || echo "$swapdev")
		if [[ "$sreal" == "$real" || "$sreal" == "$real"* ]]; then
			echo "error: active swap ${swapdev} uses ${real} (input ${dev})" >&2
			exit 4
		fi
	done < <(swapon --show --noheadings --output=NAME 2>/dev/null || true)
fi

if command -v pvs >/dev/null 2>&1; then
	while read -r pv; do
		[[ -z "${pv// }" ]] && continue
		pv_real=$(readlink -f "$pv" 2>/dev/null || echo "$pv")
		if [[ "$pv_real" == "$real" || "$pv_real" == "$real"* || "$real" == "$pv_real"* ]]; then
			echo "error: ${real} is already used as LVM PV (${pv})" >&2
			exit 5
		fi
	done < <(pvs --noheadings -o pv_name 2>/dev/null || true)
fi

exit 0
