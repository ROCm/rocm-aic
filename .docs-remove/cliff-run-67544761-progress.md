# Cliff run 67544761 (Job 2 — nvme POSIX L2) — COMPLETE ✅ (err=0, full-ladder hold)

**Job:** `aic-cliff-long64k` (67544761) · node ctr-cx66-mi300x-13 · 3 h wall
**Config:** arm = **nvme only** · L2 = **NIXL POSIX plugin** (`nixl_posix`, cpu buffer)
· ISL 64k / prefix 60k · YaRN ×2 → 65536 · DRAM L1 = 64 GB · NVMe pool 262144
**Ladder:** BENCH_CONCUR = 1, 16, 32, 64 · iters = 2
**This is Job 2** — the 4th line to compare against Job 1 (67538748: vram / nvme GDS-L2 / gds slab).

_Auto-updated every ~5 min while the job runs._

## nvme POSIX-L2 median throughput (tok/s) — vs Job 1 reference

| c | vram (J1) | nvme GDS-L2 (J1) | gds slab (J1) | **nvme POSIX-L2 (this)** |
|---|---|---|---|---|
| 1  | 45.5k | 44.6k | 44.8k | 44.9k |
| 16 | 6.3k  | 47.0k | 48.4k | 47.0k |
| 32 | 6.1k  | 46.6k | 48.0k | 46.1k |
| 64 | 6.0k  | 9.9k ⬇ | 48.7k | **42.3k ✅** |

**Verdict:** NIXL POSIX L2 (O_DIRECT, cpu buffer) **holds the full ladder** — 44.9/47.0/46.1/**42.3k**,
ext_hit 93.6% at c=64, err=0 — where nvme GDS-L2 collapsed to 9.9k. Confirms at production
L1 (64 GB) that a POSIX L2 avoids the DRAM-L1-overflow collapse. Both POSIX L2s (NIXL-POSIX
here, LMCache LocalDiskBackend earlier) and the GDS slab hold; only nvme+GDS-L2 collapses.

**Key question:** does POSIX L2 hold at c=64 (earlier sweep: ~44k) where the GDS-L2 arm
collapsed to 9.9k? If yes → POSIX L2 avoids the DRAM-L1-overflow collapse at production L1.

## Cache hit rates (L1 = vLLM VRAM prefix cache; ext/L2 = LMCache offload tier)

Format `L1% / ext%` (ext = LMCache external offload; blank for vram = no offload).

| c | vram (J1) | nvme GDS-L2 (J1) | gds slab (J1) | nvme POSIX-L2 (this) |
|---|---|---|---|---|
| 1  | 93.7 / — | 93.7 / 0 | 93.7 / 0 | 93.7 / 0 |
| 16 | 6.4 / — | 64.7 / 82.0 | 63.4 / 82.6 | 68.5 / 79.9 |
| 32 | 0 / — | 0 / 93.6 | 0 / 93.6 | 0 / 93.6 |
| 64 | 0 / — | 0 / 40.8 | 0.3 / 93.5 | 0 / 93.6 |

## NVMe read per arm (CORRECTED — use the right metric per backend)

**Metric caveat:** `node_disk_read_bytes` (/proc/diskstats, block layer) does **NOT** capture
GDS reads — true GDS is P2PDMA (NVMe→VRAM direct) and **bypasses the block layer**. So for
GDS-mode arms the correct read metric is LMCache's `gds_l1_bytes_read`, not node_disk. POSIX
O_DIRECT *does* traverse the block layer, so node_disk captures it. (GDS never uses page cache.)

| arm | NVMe read (correct metric) | NVMe write | reality |
|---|---|---|---|
| vram | — | — | no cache/NVMe |
| **nvme GDS-L2** | **0 GB** (`agent_rx`) | 94 GB | issued **no L2 reads** — DRAM-L1 served ≤c32, recompute at c64. The 0 IS the collapse (no DRAM-L1→L2 read-fallthrough), **not** caching |
| **gds slab** | **266.8 GB** (`gds_l1_bytes_read`, real GDS P2PDMA) | (part of 94 GB) | reads straight NVMe→VRAM via GDS; **invisible to node_disk by design**; never page-cached |
| **nvme POSIX-L2 (O_DIRECT)** | **106 GB** (`node_disk_read`; `agent_rx` 101.6 GB) | 97 GB | POSIX O_DIRECT traverses the block layer → node_disk counts it |

> **Corrected finding (thanks to review — my earlier "0 reads → page cache" was wrong):**
> GDS reads are real NVMe→VRAM P2PDMA and simply don't show in `node_disk_read` because GDS
> bypasses the kernel block layer — the gds slab arm genuinely read **266.8 GB off the NVMe**.
> The nvme GDS-L2 arm's 0 reads is also genuine (it never falls through to L2 → collapse),
> not caching. Read-path throughput at c=64: gds-slab **48.7k** (266.8 GB via GDS P2PDMA) vs
> POSIX-L2 **42.3k** (106 GB via O_DIRECT block layer) — GDS's direct path reads a bit faster.
> ais-snoop `ais_kfd` counters read 0 here (known-unreliable PID targeting on shared nodes —
> see [[hsa-snoop-sdma-calibration]]), so `gds_l1_bytes_read` is the trustworthy GDS metric.

## Progress log

| time | elapsed | point | note |
|---|---|---|---|
| (start) | 0:37 | launching | nvme arm, POSIX L2; vLLM starting |
