# Slurm / SPUR Build & Cliff Sweep

For local usage see [QUICK_START.md](QUICK_START.md). On a Slurm cluster the
`dist-*` / `cliff-*` targets build/distribute the image and run the full
three-arm sweep end-to-end (they wrap `.slurm/run-build-distribute.sh` and
`sbatch .slurm/run-cliff.sbatch`):

```bash
# Build the image (+ fabric exporters) on a CPU build node and save tarballs to
# the shared image dir; chain push + smoke-test:
make dist-build dist-push smoke-test AIC_PUSH_REF=<registry>/rocm-aic:latest

# Submit the full sweep (vram_only + kvd_v2 nvme + kvd_v2 gds) on a GPU+NVMe node.
# Output lands in logs/<job-id>/. Pin a node / narrow arms / override the sweep via env:
make cliff-submit AIC_CLIFF_NODE=ctr-s95-mi300x-3
make cliff-submit AIC_CLIFF_ARMS=nvme BENCH_CONCUR=1,8,64
make cliff-short          # 1-point smoke test of the whole flow
```

`smoke-test` validates the image on a GPU+NVMe node (GPU/arch, vLLM + LMCache
imports, `ais-check`, `nvme list`, the NIXL AIS_MT plugin). After those
checks it
also stands up the full exporter fleet + Prometheus (the same
`monitoring/monitoring-lib.sh` the cliff uses), scrapes briefly, health-checks
each `/metrics` endpoint, and leaves a TSDB under `logs/<job-id>/prometheus` to
sanity-check — all **informational** (only the in-image checks affect the exit
code). Tune with `AIC_SMOKE_EXPORTERS=0` (skip) and `AIC_SMOKE_SCRAPE_S=<secs>`
(default 45).

## SPUR cluster

For SPUR-specific setup (controller address, partition name, storage paths)
see [CLAUDE.md](../CLAUDE.md).

Set `AIC_SPUR_CLUSTER=1` to activate the SPUR-aware submission path:

```bash
make cliff-submit AIC_SPUR_CLUSTER=1 AIC_CLIFF_NODE=crsuse2-m2m-042
```
