<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# amdgpu-dkms-tool.py

Part of [rocm-aic](../../README.md).

## Overview

A Python tool that pulls either a URL or file-based Debian package (`.deb`),
extracts it, applies patches, and repackages with a new version tag and name.
Install the result with `dpkg` like any other DKMS package.

## Patches

Patches live in [patches](./patches). They improve the DKMS install and add
features for [hipFile][ref-hipfile]. Newer `amdgpu-dkms` versions may not
apply cleanly; the example below matches a known-good package.

| Patch | Description |
|---|---|
| 0001-silence-nproc-cached-verbosity | Suppress noisy "nproc (cached)" lines from configure output |
| 0002-provide-vmlinux-for-btf-generation | Symlink vmlinux for BTF generation during DKMS builds |
| 0003-parallel-builds | Set `MAKEFLAGS=-j$(nproc)` so DKMS builds use all cores |
| 0004-fix-amdkcl-missing-prototypes | Add missing prototypes for `amdkcl_init`/`amdkcl_exit` |
| 0005-amdkfd-ais-updates | AIS tracepoints, debugfs latency, sysfs tunables, conditional buffer pinning |
| 0006-amirs-raw-block-device-fix | Fix PCI device lookup for raw block device nodes in AIS I/O |

## Example usage

This example requires AMD VPN access to the internal package host:

```bash
./amdgpu-dkms-tool.py \
   https://mkmartifactory.amd.com:8443/artifactory/amdgpu-deb-local-new/pool/2295296/noble/a/amdgpu-dkms_6.18.8.31200000-2295296.24.04_all.deb \
  --patch patches \
  --version 6.18.8.31200000-2295296.24.05 \
  --output debs/amdgpu-dkms_6.18.8.31200000-2295296.24.05_all.deb
```

## AIS sysfs knobs

Patch 0005 adds runtime-tunable sysfs entries under
`/sys/devices/virtual/kfd/kfd/ais/`. All are read/write (0644) and require root
to write.

| Entry | Type | Default | Purpose |
|---|---|---|---|
| `buffer_pinning` | 0/1 | 1 | Pin GPU BOs to VRAM before AIS I/O. Disable (0) to skip pin/unpin. |
| `p2pdma_distance_use_cache` | 0/1 | 0 | Cache the PCI P2P distance result instead of recalculating each I/O. |
| `p2pdma_distance` | int | 0 | Override P2P distance check. 0 = normal; positive = assume reachable; negative = force-fail. |
| `pci_bdf_override` | 0/1 | 0 | When 1, use BDF from `pci_bdf` instead of discovering from the file. |
| `pci_bdf` | string | (empty) | PCI address (`DDDD:BB:DD.F` or `BB:DD.F`). Only used when `pci_bdf_override=1`. |

### Examples

```bash
# Disable buffer pinning
echo 0 > /sys/devices/virtual/kfd/kfd/ais/buffer_pinning

# Enable P2P distance caching
echo 1 > /sys/devices/virtual/kfd/kfd/ais/p2pdma_distance_use_cache

# Skip P2P distance check (positive = assume reachable)
echo 1 > /sys/devices/virtual/kfd/kfd/ais/p2pdma_distance

# Override which NVMe device is used for P2P
echo "0000:41:00.0" > /sys/devices/virtual/kfd/kfd/ais/pci_bdf
echo 1 > /sys/devices/virtual/kfd/kfd/ais/pci_bdf_override
```

### debugfs latency

When the kernel is built with `CONFIG_DEBUG_FS`, the last AIS operation's
phase-by-phase timing is available at:

```
/sys/kernel/debug/kfd/ais_latency
```

Reading it shows nanosecond breakdowns for each phase: `get_pdev`,
`p2p_distance`, `get_sg_table`, `init_bvec`, `vfs_io`, `update_counters`,
plus `total_ns`, `size_copied`, and `ret`.

<!-- References -->

[ref-hipfile]: https://github.com/ROCm/hipFile
