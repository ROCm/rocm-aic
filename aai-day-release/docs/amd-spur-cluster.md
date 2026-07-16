# AMD SPUR Cluster — Node Inventory

**Collected:** 2026-07-16 | **Nodes surveyed:** `crsuse2-m2m-001`, `-050`, `-100`, `-187`, `-200`, `-301` | **Partition:** `amd-spur`

> Inventory gathered via `srun --partition=amd-spur` against the SPUR controller at
> `http://crs-m2m-cpu-spur-005.crusoe.amd.com:6817` (set via `SPUR_CONTROLLER_ADDR`
> in `/etc/profile.d/`; not exported to non-login shells by default).

---

## Operating System

| Field | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS (Noble Numbat) |
| Kernel | `6.8.0-107-generic` |
| Kernel build date | 2026-03-13 |
| Architecture | x86_64 |
| FQDN | `crsuse2-m2m-187.us-east2-a.compute.internal` |

---

## CPU & Memory

| Field | Value |
|---|---|
| CPU Model | AMD EPYC 9575F 64-Core Processor |
| Sockets | 2 |
| Cores per socket | 59 (118 physical total) |
| Threads | 2 per core → **236 logical CPUs** |
| NUMA nodes | 2 (node0: CPUs 0–117, node1: CPUs 118–235) |
| Total RAM | **2.7 TiB** |
| Swap | None |

---

## GPUs — AMD MI355X (PCI device `0x75a3`, GFX `gfx950`)

8× AMD MI355X GPUs split evenly across the two PCIe root complexes (`0002:` bus and `0003:` bus), operating in **NPS1 / SPX** partition mode.

> **Note:** `rocm-smi --showproductname` reports the card model as `0x75a3` with `GFX Version: gfx950`. The `gfx950` target confirms MI355X (MI300X is `gfx942`). The PCI subsystem ID is also `0x75a3`; libdrm product name lookup fails in the virtualised environment.

| GPU | Temp (Junction) | Power (Socket) | SCLK | MCLK | VRAM% | GPU% |
|---|---|---|---|---|---|---|
| 0 | 58°C | 280 W | 765 MHz | 2000 MHz | 90% | 15% |
| 1 | 58°C | 259 W | 94 MHz | 2000 MHz | 0% | 0% |
| 2 | 59°C | 253 W | 93 MHz | 2000 MHz | 0% | 0% |
| 3 | 59°C | 256 W | 94 MHz | 2000 MHz | 0% | 0% |
| 4 | 61°C | 257 W | 95 MHz | 2000 MHz | 0% | 0% |
| 5 | 61°C | 258 W | 94 MHz | 2000 MHz | 0% | 0% |
| 6 | 61°C | 253 W | 94 MHz | 2000 MHz | 0% | 0% |
| 7 | 59°C | 254 W | 94 MHz | 2000 MHz | 0% | 0% |

Power cap: **1400 W per GPU**. GPU 0 had 90% VRAM in use at survey time (pre-existing job on the node).

---

## NVMe Storage

8× **Micron 7500 PRO** NVMe SSDs, 4 per PCIe bus (symmetric across NUMA nodes).

| Device | Model | Raw Capacity | FW |
|---|---|---|---|
| nvme0n1 – nvme7n1 | Micron MTFDKCC3T8TGP | 3.84 TB each | E3MQ005 |

- **Total raw NVMe: ~30.7 TB** across 8 drives (512B sectors)
- An LVM volume group (`nvme_vg` / `nvme_lv`) was present at survey time — the AAI-day LMCache NVMe tier
- ~745 GB used per drive (~19%) at survey time

---

## TCP/IP Networking

The node has 9 active physical network interfaces (plus Docker bridges):

| Interface | Driver / NIC | MTU | IPv4 | IPv6 (ULA fabric) |
|---|---|---|---|---|
| `ens3` | Mellanox mlx5Gen VF | 1500 | `10.245.146.104/20` | fe80::4ccb:2eff:fed9:8067 |
| `enP2p0s9` | AMD Pensando DSC VF | 9000 | — | `fc01:800:980d:2d59::.../64` |
| `enP2p0s10` | AMD Pensando DSC VF | 9000 | — | `fc01:700:970d:2d59::.../64` |
| `enP2p0s11` | AMD Pensando DSC VF | 9000 | — | `fc01:500:950d:2d59::.../64` |
| `enP2p0s12` | AMD Pensando DSC VF | 9000 | — | `fc01:600:960d:2d59::.../64` |
| `enP3p0s9` | AMD Pensando DSC VF | 9000 | — | `fc01:400:940d:2d59::.../64` |
| `enP3p0s10` | AMD Pensando DSC VF | 9000 | — | `fc01:300:930d:2d59::.../64` |
| `enP3p0s11` | AMD Pensando DSC VF | 9000 | — | `fc01:100:910d:2d59::.../64` |
| `enP3p0s12` | AMD Pensando DSC VF | 9000 | — | `fc01:200:920d:2d59::.../64` |

**Key points:**
- `ens3` is the only interface with an IPv4 address; default route is `10.245.144.1` via DHCP.
- The 8× Pensando DSC VFs carry **IPv6 only** (ULA `fc01::/16` range), MTU 9000 — these are the high-speed compute fabric ports.
- Fabric traffic is **IPv6-only**; no IPv4 on any fabric interface.

---

## RDMA

9 RDMA devices — all **ACTIVE / LinkUp** at survey time. All use **RoCEv2 over Ethernet** (no native InfiniBand).

| RDMA Device | NIC | Netdev | Rate | FW Version |
|---|---|---|---|---|
| `mlx5_0` | Mellanox MT4126 | `ens3` | 200 Gb/s | 28.43.3608 |
| `ionic_0` | AMD Pensando ionic | `enP2p0s9` | 400 Gb/s | 1.117.1-a-63 |
| `ionic_1` | AMD Pensando ionic | `enP2p0s10` | 400 Gb/s | 1.117.1-a-63 |
| `ionic_2` | AMD Pensando ionic | `enP2p0s11` | 400 Gb/s | 1.117.1-a-63 |
| `ionic_3` | AMD Pensando ionic | `enP2p0s12` | 400 Gb/s | 1.117.1-a-63 |
| `ionic_4` | AMD Pensando ionic | `enP3p0s9` | 400 Gb/s | 1.117.1-a-63 |
| `ionic_5` | AMD Pensando ionic | `enP3p0s10` | 400 Gb/s | 1.117.1-a-63 |
| `ionic_6` | AMD Pensando ionic | `enP3p0s11` | 400 Gb/s | 1.117.1-a-63 |
| `ionic_7` | AMD Pensando ionic | `enP3p0s12` | 400 Gb/s | 1.117.1-a-63 |

**Total fabric RDMA bandwidth: 8 × 400 Gb/s = 3.2 Tb/s** (ionic ports).

Each ionic device exposes a 256-entry GID table; only GID indices 0 and 1 are populated per port (the node's own IPv6 addresses).

---

## PCIe Topology

```
Bus 0000: (management / virtio)
  00:01  Red Hat Virtio 1.0 block device
  00:02  Red Hat Virtio 1.0 RNG
  00:03  Mellanox ConnectX mlx5Gen VF  →  ens3 (mlx5_0, 200 Gb/s RoCEv2)

Bus 0002: (NUMA node 0 — 4× GPU + 4× NVMe + 4× DSC)
  00:01  AMD MI355X GPU 0
  00:02  AMD MI355X GPU 1
  00:03  AMD MI355X GPU 2
  00:04  AMD MI355X GPU 3
  00:05  Micron 7500 PRO NVMe (nvme0)
  00:06  Micron 7500 PRO NVMe (nvme1)
  00:07  Micron 7500 PRO NVMe (nvme2)
  00:08  Micron 7500 PRO NVMe (nvme3)
  00:09  AMD Pensando DSC VF  →  enP2p0s9  (ionic_0)
  00:0a  AMD Pensando DSC VF  →  enP2p0s10 (ionic_1)
  00:0b  AMD Pensando DSC VF  →  enP2p0s11 (ionic_2)
  00:0c  AMD Pensando DSC VF  →  enP2p0s12 (ionic_3)

Bus 0003: (NUMA node 1 — 4× GPU + 4× NVMe + 4× DSC)
  00:01  AMD MI355X GPU 4
  00:02  AMD MI355X GPU 5
  00:03  AMD MI355X GPU 6
  00:04  AMD MI355X GPU 7
  00:05  Micron 7500 PRO NVMe (nvme4)
  00:06  Micron 7500 PRO NVMe (nvme5)
  00:07  Micron 7500 PRO NVMe (nvme6)
  00:08  Micron 7500 PRO NVMe (nvme7)
  00:09  AMD Pensando DSC VF  →  enP3p0s9  (ionic_4)
  00:0a  AMD Pensando DSC VF  →  enP3p0s10 (ionic_5)
  00:0b  AMD Pensando DSC VF  →  enP3p0s11 (ionic_6)
  00:0c  AMD Pensando DSC VF  →  enP3p0s12 (ionic_7)
```

GPUs, NVMe drives, and fabric NICs are **symmetrically split 4+4** across the two NUMA nodes and PCIe root complexes.

---

## Filesystem Layout

Consistent across all nodes surveyed. Each node presents the following mount points:

### Local Filesystems

| Mount | Device | Type | Size | Notes |
|---|---|---|---|---|
| `/` | `/dev/vda1` | ext4 | 123 GiB | Virtio root disk |
| `/boot` | `/dev/vda16` | ext4 | 881 MiB | |
| `/boot/efi` | `/dev/vda15` | vfat | 105 MiB | |
| `/mnt/m2m_nobackup` | `/dev/mapper/nvme_vg-nvme_lv` | XFS | **27.9 TiB** | 8× NVMe LVM stripe (see below) |
| `/tmp` | (on `/dev/vda1`) | — | — | Backed by root disk; usage varies per node |
| `/dev/shm` | tmpfs | tmpfs | 1.3 TiB | RAM-backed shared memory |
| `/run` | tmpfs | tmpfs | 275 GiB | |

**`/tmp` is not a tmpfs** — it sits on the 123 GiB root ext4 volume. Usage varies significantly between nodes (observed 3 MiB to 3.7 GiB). Use `/mnt/m2m_nobackup` for large scratch data.

### NVMe LVM Volume (`/mnt/m2m_nobackup`)

All 8 NVMe drives are pooled into a single LVM volume group (`nvme_vg`) with a striped logical volume (`nvme_lv`) mounted as XFS at `/mnt/m2m_nobackup`:

```
nvme0n1–nvme7n1  →  nvme_vg (PVs)  →  nvme_lv (striped RAID, 8 rimages)  →  XFS  →  /mnt/m2m_nobackup
```

- **Total usable:** ~27.9 TiB (XFS on 8-way stripe across 8× 3.84 TB Micron 7500 PRO)
- **XFS stripe params:** `sunit=512, swidth=4096` (aligned to 8-drive stripe)
- **Usage at survey time:** 3.5–6.1 TiB used per node (~12–22%), ~22–25 TiB free
- This is the intended mount point for LMCache NVMe tiers (`NVME_DATA`, `GDS_SLAB_DATA`)
- **No `/scratch`** — the equivalent is `/mnt/m2m_nobackup`

### NFS / Shared Filesystems

All nodes share the same NFS server at `172.27.255.2`, with three volumes mounted:

| Mount | NFS Export (volume UUID) | Type | Size | Access | Notes |
|---|---|---|---|---|---|
| `/home` | `172.27.255.2:/volumes/b0a55a09-...` | NFSv3 | 5.0 TiB | rw | User home dirs; shared across all nodes |
| `/it-shared` | `172.27.255.2:/volumes/bad83b28-...` | NFSv3 | 1.0 TiB | **ro** | IT-managed shared data |
| `/shared_nfs` | `172.27.255.2:/volumes/b2e6868e-...` | NFSv3 | 30 TiB | rw | Large shared storage |

**NFS mount options of note:**
- `/home`: single connection (`nconnect` not set), `timeo=600`, `retrans=2` — lighter-weight
- `/it-shared` and `/shared_nfs`: `nconnect=16` (16 parallel TCP connections), `spread_reads`, `spread_writes`, `remoteports=172.27.255.2-172.27.255.17` (multi-path across 16 server IPs for throughput)

**Space at survey time:**
- `/home`: 3.4 TiB / 5.0 TiB used (67–67%)
- `/it-shared`: 846 GiB / 1.0 TiB used (83%) — **getting full**
- `/shared_nfs`: 24 TiB / 30 TiB used (78–79%) — **getting full**

### `/home` bind-mount detail

`/home/ubuntu` is bind-mounted from `/dev/vda1[/localhome/ubuntu]` (i.e. a subdirectory of the root ext4), overriding the NFS `/home` for the `ubuntu` system account. All other user home directories (e.g. `/home/stebates`) come from NFS and are available identically on every node.

### Storage Summary for AAI-day Workloads

| Tier | Path | Capacity | Recommended use |
|---|---|---|---|
| Local NVMe (fast) | `/mnt/m2m_nobackup` | ~27.9 TiB | LMCache NVMe L2, model weights staging, scratch |
| Shared NFS (large) | `/shared_nfs` | 30 TiB (6.5 TiB free) | Shared model cache, results |
| RAM | `/dev/shm` | 1.3 TiB | LMCache DRAM L1, `AIS_MT` memory mapping |
| Root disk | `/` | 123 GiB | OS only — avoid filling |

---

## Node Homogeneity

Sampled 6 nodes spread across the full numbering range (`-001`, `-050`, `-100`, `-187`, `-200`, `-301`). **All nodes are identical** in every measured dimension:

| Field | Value (all nodes) |
|---|---|
| Kernel | `6.8.0-107-generic` |
| CPU | AMD EPYC 9575F, 2S × 59C, 236 logical CPUs |
| RAM | 2.7 TiB |
| GPUs | 8× MI355X (`gfx950`) |
| NVMe | 9 drives visible (8 physical + 1 nvme-fabrics) |
| RDMA devices | 9 (8× ionic 400 Gb/s + 1× mlx5 200 Gb/s) |
| RDMA link state | All ACTIVE |
| Interface naming | Identical (`ionic_0–7` → `enP2p0s9–12`, `enP3p0s9–12`) |

There is no hardware heterogeneity in this cluster across the sampled population.

---

## Storage Performance

Inter-node storage benchmark run with fio 3.36 (ioengine=`mmap`, `numjobs=8`, `iodepth=1`, `size=16g`, `runtime=30s`, `time_based`, `group_reporting`). Tests run on two nodes (`crsuse2-m2m-050` and `crsuse2-m2m-200`) simultaneously via sbatch. Note: fio was compiled from source as it is not installed on compute nodes; the `mmap` engine was used as `libaio` headers are absent without root access (`libaio` runtime is present but not the dev package). This means results reflect page-cache-assisted I/O rather than O_DIRECT; actual device-level throughput for the NVMe tier will differ under O_DIRECT workloads. NFS results are representative of real client-side throughput.

### Local NVMe LVM (`/mnt/m2m_nobackup`) — 8× Micron 7500 PRO, 8-way XFS stripe

| Block Size | Direction | BW node-050 (MiB/s) | IOPS node-050 | BW node-200 (MiB/s) | IOPS node-200 |
|---|---|---|---|---|---|
| 4 KiB | Random Read  | **733.7** | 187,838 | **755.6** | 193,441 |
| 4 KiB | Random Write | **724.8** | 185,550 | **711.0** | 182,025 |
| 1 MiB | Random Read  | **543.8** | 544 | **536.0** | 536 |
| 1 MiB | Random Write | **765.9** | 766 | **795.9** | 796 |

**Notes:** ~720–756 MiB/s at 4 KiB (~185–193k IOPS) is consistent with the XFS 8-way stripe across 8 NVMe drives. 1 MiB sequential-style random writes exceed reads due to page-cache write coalescing. Both nodes show near-identical performance confirming hardware homogeneity.

### NFS `/home` — `172.27.255.2`, single-connection NFSv3 (`timeo=600`)

| Block Size | Direction | BW node-050 (MiB/s) | IOPS node-050 | BW node-200 (MiB/s) | IOPS node-200 |
|---|---|---|---|---|---|
| 4 KiB | Random Read  | **54.7** | 14,009 | **55.7** | 14,260 |
| 4 KiB | Random Write | **44.9** | 11,484 | **47.7** | 12,219 |
| 1 MiB | Random Read  | **72.5** | 72 | **87.4** | 87 |
| 1 MiB | Random Write | **53.1** | 53 | **59.2** | 59 |

**Notes:** `/home` uses a single NFS TCP connection (`nconnect` not set), which caps throughput. ~55 MiB/s at 4 KiB and ~72–87 MiB/s at 1 MiB are consistent with a lightly loaded single-connection NFSv3 path. Do not use `/home` for high-bandwidth I/O workloads.

### NFS `/shared_nfs` — `172.27.255.2`, 16-connection NFSv3 (`nconnect=16`, multipath)

| Block Size | Direction | BW node-050 (MiB/s) | IOPS node-050 | BW node-200 (MiB/s) | IOPS node-200 |
|---|---|---|---|---|---|
| 4 KiB | Random Read  | **62.2** | 15,918 | **66.1** | 16,930 |
| 4 KiB | Random Write | **38.0** | 9,736 | **36.5** | 9,336 |
| 1 MiB | Random Read  | **79.9** | 80 | **80.3** | 80 |
| 1 MiB | Random Write | **83.9** | 84 | **83.0** | 83 |

**Notes:** `/shared_nfs` has `nconnect=16` and 16-path multipath routing but per-node throughput is still in the ~80–84 MiB/s range at 1 MiB — the aggregate server-side bandwidth is shared across all cluster nodes. 4 KiB random write performance (~37–38 MiB/s) is lower than `/home`, likely due to the server-side striping/RAID overhead. Use `/shared_nfs` for large shared datasets, not latency-sensitive or write-heavy workloads.

### Storage Tier Summary

| Tier | Path | 4K Rand Read | 4K Rand Write | 1M Rand Read | 1M Rand Write | Recommended Use |
|---|---|---|---|---|---|---|
| Local NVMe LVM | `/mnt/m2m_nobackup` | ~745 MiB/s / ~190k IOPS | ~718 MiB/s / ~184k IOPS | ~540 MiB/s | ~780 MiB/s | LMCache NVMe L2, model weights, scratch |
| NFS `/home` | `/home/stebates` etc. | ~55 MiB/s / ~14k IOPS | ~46 MiB/s / ~12k IOPS | ~80 MiB/s | ~56 MiB/s | Small files, code, config only |
| NFS `/shared_nfs` | `/shared_nfs` | ~64 MiB/s / ~16k IOPS | ~37 MiB/s / ~9k IOPS | ~80 MiB/s | ~83 MiB/s | Shared model cache, checkpoints |

**Test methodology:** fio 3.36, `ioengine=mmap`, `numjobs=8`, `iodepth=1`, `size=16g`, `runtime=30s` per test, `group_reporting`. Two nodes tested simultaneously (jobs 8297/8298, 2026-07-16). Scripts: `fio-test.sbatch`, `fio-worker.sh`, `fio-parse.py` in project root.

---

## Network Performance

Inter-node RDMA bandwidth measured with `ib_write_bw` (perftest v6.25, RDMA Write, RC connection, 65536 B messages, 5 s duration). Tests run as batch jobs via `sbatch --nodes=2`, server and client dispatched to separate nodes via SSH over shared NFS, coordinating via a shared IP file. All link states were ACTIVE at test time.

### Test methodology

- Tool: `ib_write_bw -d <device> -i 1 -p <port> -D 5 --report_gbits`
- GID selection: index 3 (IPv4-mapped) for `mlx5_0`; index 1 (ULA IPv6) for `ionic_0`
- Message size: 65536 B (default); BW peak suppressed (`-D` duration mode)
- Node pairs chosen to sample across the full numbering range of the cluster

### Mellanox mlx5_0 — 200 Gb/s RoCEv2 (`ens3`, IPv4)

| Node A | Node B | BW avg (Gb/s) | Msg rate (Mpps) |
|---|---|---|---|
| crsuse2-m2m-036 | crsuse2-m2m-301 | **161.95** | 0.388 |
| crsuse2-m2m-003 | crsuse2-m2m-001 | **160.64** | — |
| crsuse2-m2m-100 | crsuse2-m2m-150 | **162.89** | 0.311 |
| crsuse2-m2m-325 | crsuse2-m2m-290 | **161.42** | — |
| crsuse2-m2m-331 | crsuse2-m2m-002 | **161.32** | — |

**mlx5_0 summary:** 160–163 Gb/s, highly consistent across all node pairs (~81% line rate of 200 Gb/s). No topology-dependent variation observed.

### AMD Pensando ionic_0 — 400 Gb/s RoCEv2 (`enP2p0s9`, IPv6 ULA)

| Node A | Node B | BW avg (Gb/s) | Msg rate (Mpps) |
|---|---|---|---|
| crsuse2-m2m-036 | crsuse2-m2m-301 | **388.27** | 0.741 |
| crsuse2-m2m-003 | crsuse2-m2m-001 | **338.91** | — |
| crsuse2-m2m-100 | crsuse2-m2m-150 | **349.47** | 0.667 |
| crsuse2-m2m-325 | crsuse2-m2m-290 | **389.58** | 0.743 |
| crsuse2-m2m-331 | crsuse2-m2m-002 | **338.98** | — |

**ionic_0 summary:** 339–390 Gb/s (~85–97% line rate of 400 Gb/s). Higher variance than mlx5_0 — pairs with BW in the 338–350 Gb/s range vs 388–390 Gb/s may reflect differences in fabric path or switch topology within the cluster. MTU on ionic is 4096 B (vs 1024 B for mlx5_0), which benefits large-message bandwidth.

### Notes

- Only `ionic_0` (one of eight 400 Gb/s ports per node) was tested. Running all 8 ionic ports simultaneously would yield up to **3.2 Tb/s** aggregate per node pair.
- `mlx5_0` is the management/out-of-band NIC; not intended for compute traffic.
- Occasional "Couldn't listen to port" errors in `stderr` are from ghost ib_write_bw processes left over on nodes from prior failed runs — they do not affect successful test results.
- Tests collected 2026-07-16 using `rdma-test.sbatch` / `rdma-worker.sh` in the project root.

---

## SPUR Cluster Facts

- **Controller:** `http://crs-m2m-cpu-spur-005.crusoe.amd.com:6817`
- **Partition:** `amd-spur`
- **Total nodes in partition:** 256 (255 idle + 1 mix at survey time)
- **Node naming convention:** `crsuse2-m2m-NNN`
- **`SPUR_CONTROLLER_ADDR`** is set system-wide in `/etc/profile.d/` — sourced only by interactive login shells, not by Claude Code or other non-login processes. Export it manually if needed:
  ```bash
  export SPUR_CONTROLLER_ADDR=http://crs-m2m-cpu-spur-005.crusoe.amd.com:6817
  ```
