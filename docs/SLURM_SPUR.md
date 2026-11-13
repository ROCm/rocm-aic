# Slurm / SPUR Build & Cliff Sweep

For local usage see [QUICK_START.md](QUICK_START.md). On a Slurm cluster the
`dist-*` / `cliff-*` targets build/distribute the image and run the full
three-arm sweep end-to-end (they wrap `.slurm/run-build-distribute.sh`; its
`cliff` command submits `.slurm/run-cliff.sbatch` and waits for the job):

```bash
# Build the image (+ fabric exporters) on a CPU build node and save tarballs to
# the shared image dir; chain push + smoke-test:
make dist-build dist-push smoke-test \
  AIC_VLLM_PUSH_REF=<registry>/aic-vllm:latest AIC_LMCACHE_PUSH_REF=<registry>/aic-lmcache:latest

# Run the full sweep (vram_only + kvd_v2 nvme + kvd_v2 gds) on a GPU+NVMe node. The
# target blocks until the job ends and exits with its status.
# Output lands in logs/<job-id>/. Pin a node / narrow arms / override the sweep via env:
make cliff-submit AIC_CLIFF_NODE=<node-name>
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

## Several jobs on one node

Every job's stack is named `aic-<job id>` (`AIC_STACK`): its compose project,
bridge network and container names (`aic-<job id>-lmcache`, ...), so jobs that
land on the same node, CI or manual, never recreate or remove each other's
containers. To look at a job's containers on its node:

```bash
docker ps --filter label=com.docker.compose.project=aic-<job id>
docker logs aic-<job id>-vllm-gpu0
```

A job killed before its teardown (`scancel`, the time limit) leaves its
containers running; `docker compose -p aic-<job id> down -v` removes them, its
network and its volumes, and needs no compose file (if it hangs on vLLM's
EngineCore, `docker rm -f` the job's containers first).

## SPUR cluster

For SPUR-specific setup (controller address, partition name, storage paths)
see [CLAUDE.md](../CLAUDE.md).

Set `AIC_SPUR_CLUSTER=1` to activate the SPUR-aware submission path:

```bash
make cliff-submit AIC_SPUR_CLUSTER=1 AIC_CLIFF_NODE=<node-name>
```
