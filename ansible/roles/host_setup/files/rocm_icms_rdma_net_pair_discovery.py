#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Emit JSON list of RDMA netdev / InfiniBand pairs for rocm-icms udev naming."""

from __future__ import annotations

import json
import os
import re


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def _netdev_driver(netdev: str) -> str:
    link = f"/sys/class/net/{netdev}/device/driver"
    if os.path.islink(link):
        return os.path.basename(os.readlink(link))
    return ""


DRIVER_TO_VENDOR = {
    "mlx5_core": "mlx",
    "bnxt_en": "bnxt",
    "bnxt_re": "bnxt",
    "ionic": "ionic",
}


def _pci_sort_key(device_realpath: str) -> str:
    m = re.search(r"([\da-f]{4}:[\da-f]{2}:[\da-f]{2}\.\d)$", device_realpath)
    if m:
        return m.group(1)
    return device_realpath


def _netdirs_for_ib(ib_dir: str) -> list[str]:
    net_base = os.path.join(ib_dir, "device", "net")
    if not os.path.isdir(net_base):
        return []
    return sorted(
        x
        for x in os.listdir(net_base)
        if os.path.isdir(os.path.join(net_base, x))
    )


def main() -> None:
    ib_root = "/sys/class/infiniband"
    if not os.path.isdir(ib_root):
        print("[]")
        return

    raw: list[dict[str, str]] = []
    for ib in sorted(os.listdir(ib_root)):
        ib_dir = os.path.join(ib_root, ib)
        if not os.path.isdir(ib_dir):
            continue
        ng_file = os.path.join(ib_dir, "node_guid")
        if not os.path.isfile(ng_file):
            continue
        node_guid = _read(ng_file)
        netdirs = _netdirs_for_ib(ib_dir)
        if not netdirs:
            continue
        netdev = netdirs[0]
        drv = _netdev_driver(netdev)
        vendor = DRIVER_TO_VENDOR.get(drv)
        if not vendor:
            continue
        mac = _read(os.path.join("/sys/class/net", netdev, "address")).lower()
        dev_path = os.path.join(ib_dir, "device")
        pci_key = _pci_sort_key(os.path.realpath(dev_path))
        raw.append(
            {
                "pci_key": pci_key,
                "vendor": vendor,
                "mac": mac,
                "node_guid": node_guid,
            }
        )

    raw.sort(key=lambda x: (x["vendor"], x["pci_key"]))

    counts: dict[str, int] = {}
    out: list[dict[str, str]] = []
    for row in raw:
        v = row["vendor"]
        n = counts.get(v, 0)
        counts[v] = n + 1
        out.append(
            {
                "mac": row["mac"],
                "node_guid": row["node_guid"],
                "eth_name": f"{v}-eth{n}",
                "rdma_name": f"{v}-rdma{n}",
            }
        )

    print(json.dumps(out))


if __name__ == "__main__":
    main()
