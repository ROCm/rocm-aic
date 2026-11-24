#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Put a dm-delay device under the LMCache L2 mount, or take it away again, so
# nvme_delay_sweep.sh can inject read latency.  The filesystem and its data are
# untouched: the same block device is mounted, through the mapper or directly.
# Needs sudo, and nothing may be using the mount (bring the stack down first).
#
#   dm_delay_setup.sh up   <mount-point>     # mount through /dev/mapper/<name>
#   dm_delay_setup.sh down <mount-point>     # mount the device directly again
#
# Environment:
#   AIC_DM_DEVICE   mapper name (default: kv-delay)

set -euo pipefail

NAME="${AIC_DM_DEVICE:-kv-delay}"
MAPPER="/dev/mapper/${NAME}"

die() { echo "dm_delay_setup: $*" >&2; exit 1; }

(( $# == 2 )) || die "usage: $0 up|down <mount-point>"
action="$1"
mount_point="$(realpath "$2")"
source_dev="$(findmnt -no SOURCE "${mount_point}" 2>/dev/null)" \
    || die "${mount_point} is not a mount point"

case "${action}" in
    up)
        [[ "${source_dev}" != "${MAPPER}" ]] || { echo "${mount_point} is already on ${MAPPER}"; exit 0; }
        [[ "${source_dev}" != /dev/mapper/* ]] || die "${mount_point} is on ${source_dev}, not a plain block device"
        sudo dmsetup info "${NAME}" >/dev/null 2>&1 && die "${MAPPER} already exists but ${mount_point} is not on it"
        sectors="$(sudo blockdev --getsz "${source_dev}")"
        sudo umount "${mount_point}"
        echo "0 ${sectors} delay ${source_dev} 0 0 ${source_dev} 0 0" | sudo dmsetup create "${NAME}"
        sudo mount "${MAPPER}" "${mount_point}"
        echo "${mount_point}: ${source_dev} -> ${MAPPER} (read delay 0 ms)"
        ;;
    down)
        [[ "${source_dev}" == "${MAPPER}" ]] || { echo "${mount_point} is on ${source_dev}, not ${MAPPER}"; exit 0; }
        # "0 <sectors> delay <dev> 0 <read_ms> <dev> 0 <write_ms>"; <dev> is major:minor.
        majmin="$(sudo dmsetup table "${NAME}" | awk '{print $4}')"
        backing="$(lsblk -dnro PATH,MAJ:MIN | awk -v m="${majmin}" '$2 == m {print $1}')"
        [[ -n "${backing}" ]] || die "cannot find the device ${majmin} behind ${MAPPER}"
        sudo umount "${mount_point}"
        sudo dmsetup remove "${NAME}"
        sudo mount "${backing}" "${mount_point}"
        echo "${mount_point}: ${MAPPER} -> ${backing}"
        ;;
    *)  die "usage: $0 up|down <mount-point>" ;;
esac
