# Cliff run 67544874 (O_DIRECT LMCache LocalDiskBackend) — COMPLETE ✅ (err=0, full-ladder hold)

**Job:** `aic-cliff-long64k` (67544874) · node ctr-cx66-mi300x-13 · 3 h wall
**Config:** arm = **nvme only** · L2 = **LMCache LocalDiskBackend, O_DIRECT** (`local_disk`,
`use_odirect: true`, 320 GB disk tier) · ISL 64k/60k · YaRN ×2 · DRAM L1 = 64 GB
**Ladder:** BENCH_CONCUR = 1, 16, 32, 64 · iters = 2
**Purpose:** O_DIRECT counterpart to the earlier *buffered* LocalDiskBackend (67538307) — gives
the buffered-vs-O_DIRECT contrast for the native POSIX L2, and a true block-layer device-read number.

_Auto-updated every ~5 min while the job runs._

## Throughput (median tok/s) — vs prior runs (L1=64 GB)

| c | vram | nvme GDS-L2 | gds slab | nvme POSIX (NIXL, O_DIRECT) | LocalDisk buffered (J 67538307, c32 only) | **LocalDisk O_DIRECT (this)** |
|---|---|---|---|---|---|---|
| 1  | 45.5k | 44.6k | 44.8k | 44.9k | — | 44.6k |
| 16 | 6.3k  | 47.0k | 48.4k | 47.0k | — | 46.9k |
| 32 | 6.1k  | 46.6k | 48.0k | 46.1k | 41.9k | 46.3k |
| 64 | 6.0k  | 9.9k ⬇ | 48.7k | 42.3k | — | **40.6k ✅** |

## NVMe read (fill at completion, from TSDB)

O_DIRECT LocalDisk → block-layer reads, so `node_disk_read` should be nonzero (unlike buffered/GDS).

| metric | value |
|---|---|
| NVMe device read (`node_disk_read`, nvme2n1) | **102.1 GB** (real block-layer reads — O_DIRECT) |
| NVMe device write | 97.0 GB |

**Verdict:** O_DIRECT LocalDiskBackend holds the full ladder (44.6/46.9/46.3/**40.6k**, err=0),
doing **102 GB of real block-layer NVMe reads** (O_DIRECT → visible in node_disk, unlike GDS).
Consistent with NIXL-POSIX-O_DIRECT (106 GB reads, 42.3k). Both first-class POSIX O_DIRECT
paths ≈ 40–42k @ c=64 with ~100 GB real device reads; GDS-slab leads at 48.7k via 266.8 GB
GDS P2PDMA. Only nvme+GDS-L2 collapses (9.9k, 0 L2 reads).

## Progress log

| time | elapsed | point | note |
|---|---|---|---|
| (start) | 8:57 | c=16 | O_DIRECT confirmed (log: "Using O_DIRECT for disk I/O: True"); c=1 44.6k, c=16 ~47k, err=0 |
