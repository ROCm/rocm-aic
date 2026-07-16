# Long-ISL cliff report — POST-FIX rerun (verified pool 262144 + slab 320 GB)

Companion to `cliff-long-isl-report.md` (the **pre-fix** runs 67536671 / 67536798 /
67537066). Those runs were confounded by a Makefile propagation bug: `${VAR:-default}`
could not override the Makefile-exported defaults, so **`nixl_pool_size` was silently
4096 slots (~18 GiB) instead of 262144**, and **the GDS slab was 20 GB instead of
320 GB**. See that report's "REVISED ROOT CAUSE" + "Propagation gap" sections.

**This run isolates the fix.** Same config as yesterday's run 2 (job 67536798):
trimmed ladder `BENCH_CONCUR=1,8,16,32,64`, 3 h wall, DRAM L1 64 GB, YaRN ×2 → 65536,
arms vram → nvme → gds. The *only* deliberate change is the corrected pool/slab
sizing (Makefile origin-guarded fix). The goal is to answer the two questions the
pre-fix runs could not:

1. **nvme arm:** with a real 262144-slot pool, does `local_cpu:true` now hold c≥32
   (and c=64)? If it still collapses, the **DRAM-L1→NVMe-L2 read-fallthrough bug is
   real and isolated**. If it holds, the pre-fix collapse was *only* the tiny pool.
2. **gds arm:** with a real 320 GB slab, is the c≥32 collapse **capacity overflow**
   (was 20 GB < 34 GB WS) or a genuine **I/O-rate wall** (as originally claimed)?

## Run — job 67537751 (`cliff-long-64k`, pool/slab-fixed), SUBMITTED ⏳

- **Submitted:** 2026-07-16, `AAI_CLIFF_TIME=03:00:00 BENCH_CONCUR=1,8,16,32,64 make cliff-long-64k`
- **State:** PENDING (Priority) — awaiting scheduling on MARKHAM&GFX942&NVME.
- **Log:** `logs/67537751/cliff.out`

### ✅ Sizing verification (from the new `tiers:` line in cliff.out)

_To be filled the moment the job starts — this is the check that the whole rerun
hinges on. Expect `nixl_pool_size=262144 slots (~1152 GiB)` and `gds_l1=320GB`._

| knob | pre-fix (67536798) | this run (67537751) | expected |
|---|---|---|---|
| `nixl_pool_size` | 4096 (~18 GiB) ❌ | _pending_ | **262144 (~1152 GiB)** |
| `gds_l1` slab | 20 GB ❌ | _pending_ | **320 GB** |
| `max_local_cpu_size` (DRAM L1) | 64 GB ✅ | _pending_ | 64 GB |
| `nixl_buffer` | 8 GiB ✅ | _pending_ | 8 GiB |

## Head-to-head throughput (median tok/s) — pre-fix vs post-fix

_Filled per arm as the sweep runs. Pre-fix column = job 67536798 (yesterday, run 2)._

### Arm A — vram_only (baseline; should be unchanged — no cache involved)
| c | pre-fix vram | post-fix vram | note |
|---|---|---|---|
| 1  | 44.9k | _pending_ | |
| 8  | 27.9k | _pending_ | |
| 16 | 8.7k  | _pending_ | cliff (HBM full ~14 req) |
| 32 | 5.9k  | _pending_ | floor |
| 64 | 5.8k  | _pending_ | client ReadTimeout region |

### Arm B — kvd_v2 nvme (64 GB DRAM L1 + NVMe L2) — the key nvme test
| c | pre-fix nvme | post-fix nvme | pre-fix ext_hit | post-fix ext_hit | note |
|---|---|---|---|---|---|
| 1  | 44.8k | _pending_ | | | |
| 8  | 47.5k | _pending_ | | | warm-cache |
| 16 | 46.8k | _pending_ | 93% | _pending_ | cliff |
| 32 | 45.7k | _pending_ | 93.4% | _pending_ | pre-fix held (pool fit at c=32 even @4096? no — 7488>4096, so pre-fix c=32 actually collapsed in the NVMe-only repro; watch this closely) |
| 64 | 9.75k | _pending_ | 42.6% | _pending_ | **THE test:** pre-fix collapsed (WS>pool AND >DRAM L1). Post-fix pool fits — does it still collapse (→ fallthrough bug real) or hold (→ was just the pool)? |

### Arm C — kvd_v2 gds (GDS NVMe slab L1) — the key gds test
| c | pre-fix gds | post-fix gds | note |
|---|---|---|---|
| 1  | 45.2k | _pending_ | |
| 8  | 48.4k | _pending_ | |
| 16 | 10.97k | _pending_ | cliff — fix held pre-fix |
| 32 | 5.74k | _pending_ | **THE test:** pre-fix slab was 20 GB (<34 GB WS → overflow). Post-fix 320 GB. Does it now hold (→ was capacity) or still collapse (→ real I/O-rate wall)? |
| 64 | 5.7k | _pending_ | |

## Live NIXL / cache telemetry (per arm, from :19090 + vLLM /metrics)

_Captured live while each arm runs (ssh to the node works via pam_slurm_adopt)._

| arm | c | agent_tx_bytes | agent_rx_requests | agent_rx_bytes | ext_hit | verdict |
|---|---|---|---|---|---|---|
| _pending_ | | | | | | |

> **The `agent_rx_requests` counter is the crux for the nvme arm:** pre-fix it was
> frozen (0 reads issued in `local_cpu:true`). If it stays 0 at c=64 *with a
> correctly-sized pool*, the DRAM-L1→NVMe-L2 read-fallthrough bug is confirmed real
> and isolated from the pool-sizing artifact.

## Findings / verdict

_TBD — filled at job completion._
