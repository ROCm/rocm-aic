<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# vllm-lmcache-nixl

ROCm **vLLM** + **LMCache** + **NIXL** (andyluo7/nixl @ pinned `NIXL_SHA` + AIS/hipfile
overlay). Uses LMCache **`enable_nixl_storage`** with **POSIX** (MVP) or **AIS**
backends on AMD Instinct GPUs.

Part of [rocm-aic](../../README.md). Gutenberg benchmarks live in
[benchmarks/llm-prefill-benchmark](../../benchmarks/llm-prefill-benchmark).

## Quick start

```bash
export ROCM_ARCH=gfx942
make -C recipies/vllm-lmcache-nixl build
export HF_TOKEN=...
make -C recipies/vllm-lmcache-nixl run VLN_LMCACHE_IO=nixl-posix
```

## Runtime YAML

For repeated local or Slurm runs, checked-in defaults come from
`runtime-defaults.yaml`. Put local overrides in `runtime.yaml` instead of
exporting each `VLN_*` variable:

```bash
$EDITOR recipies/vllm-lmcache-nixl/runtime.yaml
make -C recipies/vllm-lmcache-nixl run
```

Use `RECIPE_RUNTIME_FILE=/path/to/runtime.yaml` to select another file.
Environment variables override checked-in defaults. When an override YAML file
is detected, mapped runtime env vars are ignored so the file wins;
`make VAR=value` remains an explicit one-off override.

## Storage modes (`VLN_LMCACHE_IO`)

| Value | LMCache / NIXL backend |
|-------|-------------------------|
| `nixl-posix` (default) | NIXL POSIX on local NVMe |
| `ais` | NIXL **AIS_MT** + sync hipFile (default AIS path; see below) |
| `ais_mt` | Same as `ais` (explicit alias) |
| `ais_batch` | NIXL AIS batch hipFile (stub on ROCm hipFile 0.2.x; not recommended) |

## AIS / hipFile (non-compat)

ROCm **hipFile 0.2.x** implements sync ``hipFileRead``/``hipFileWrite`` only;
batch ``hipFileBatchIOGetStatus`` is not implemented on AMD. This recipe
therefore maps ``VLN_LMCACHE_IO=ais`` to NIXL **AIS_MT** (thread-pool sync I/O).
Use ``VLN_LMCACHE_IO=ais_batch`` only to experiment with the batch AIS plugin.

AIS mode uses a **GPU (VRAM) NIXL staging buffer** so `hipFileBufRegister`
succeeds and transfers use the direct hipFile path. The template
``configs/lmcache-nixl-ais.yml`` sets ``nixl_buffer_device: cuda`` (same idea as
``HipFileMemoryAllocator`` in the hipfile recipe).

```bash
make -C recipies/vllm-lmcache-nixl run VLN_LMCACHE_IO=ais DATA=/mnt/lmcache-nvme/ GPU=0
```

``HIPFILE_ALLOW_COMPAT_MODE=false`` is set in ``vllm-server``; if buffer
registration fails, NIXL AIS init **errors** instead of falling back to compat
mode. Rebuild the image after overlay changes (``make build``).

### hipFile ``ais-stats`` (in container)

The image installs ``/app/ais-stats`` (also on ``PATH`` as ``ais-stats``).
hipFile counters are **off** unless ``VLH_HIPFILE_STATS_LEVEL`` is set before
EngineCore starts (``0`` = disabled, ``1`` = basic, ``2`` = detailed, ``3`` =
max):

```bash
make -C recipies/vllm-lmcache-nixl run \
  VLN_LMCACHE_IO=ais \
  VLH_HIPFILE_STATS_LEVEL=1 \
  DATA=/mnt/lmcache-nvme/ GPU=0
```

After AIS traffic, attach to EngineCore and print stats:

```bash
docker exec -it vllm-lmcache-nixl-gpu0 bash -lc \
  'ais-stats -p $(pgrep -f VLLM::EngineCor | head -1) -i'
```

Override staging device or size:

```bash
VLN_NIXL_BUFFER_DEVICE=cuda VLN_NIXL_BUFFER_SIZE=512 make run VLN_LMCACHE_IO=ais
```

At startup you should see ``Backend AIS_MT was instantiated`` **without** a flood
of ``buffer registration failed - will use compat mode`` warnings. The default
**512 MiB** VRAM staging buffer is reserved on GPU **in addition** to vLLM KV
cache. On **16 GiB** GPUs, HIP OOM during warmup is common unless you tune:

```bash
VLN_LMCACHE_IO=ais \
VLN_NIXL_BUFFER_SIZE=256 \
VLN_GPU_MEMORY_UTILIZATION=0.85 \
VLN_NIXL_POOL_SIZE=4096 \
make -C recipies/vllm-lmcache-nixl run
```

Keep ``VLN_NIXL_POOL_SIZE`` in the low thousands unless ``VLN_DOCKER_NOFILE``
is raised and you accept long AIS init (one FD per pool slot; your log used
``262114``, which alone adds ~20 s startup).

After a store, used ``obj_*.bin`` slots should be **~4.5 MiB each** (Qwen2.5-3B
align size), not 0 bytes. Unused pool slots stay **0 bytes** until first write
(lazy ``ftruncate`` before O_DIRECT AIS I/O):

```bash
find /mnt/lmcache-nvme/lmcache -name 'obj_*.bin' -size +0 | wc -l   # used slots
find /mnt/lmcache-nvme/lmcache -name 'obj_*.bin' -size +0 -printf '%s\n' | \
  awk '{s+=$1} END {print s}'   # used bytes (matches rocm_aic_nixl_pool_bytes_total)
```

0-byte slots after traffic on a slot that should have data means AIS writes did
not land; rebuild the image after overlay / LMCache patch updates.

## Build arguments

| ARG | Default |
|-----|---------|
| `NIXL_GIT_URL` | `https://github.com/andyluo7/nixl.git` |
| `NIXL_SHA` | `f72aad2…` (amd-support tip; see `../common/nixl/defaults.mk`) |
| `HIPFILE_SHA` | same as hipfile recipe |
| `LMCACHE_SHA` | pinned in Dockerfile |

NIXL is built via [recipies/common/nixl](../common/nixl) with the AIS plugin
overlay applied at image build time.

## Slurm

From repo root:

```bash
./run-slurm-nixl.sh
```

## Compare to vllm-lmcache-hipfile

| Recipe | KV disk path |
|--------|----------------|
| vllm-lmcache-hipfile | LMCache GdsBackend → hipFile directly |
| vllm-lmcache-nixl | LMCache NixlStorageBackend → NIXL POSIX/AIS |

Both use the same [llm-prefill-benchmark](../../benchmarks/llm-prefill-benchmark)
Gutenberg workload for TTFT measurement.

## NIXL pool size (`VLN_NIXL_POOL_SIZE`)

The NIXL backend pre-allocates a fixed pool of ``obj_*.bin`` slot **files**
under ``$DATA/lmcache/`` (one FD per slot at init). Default **`nixl_pool_size:
4096`** reserves up to that many slots; only **used** slots are sized on disk
(~4.5 MiB each for Qwen2.5-3B). Grafana:
``rocm_aic_nixl_pool_slots_used`` and ``rocm_aic_nixl_pool_bytes_total``.

```bash
make run VLN_NIXL_POOL_SIZE=8192   # up to 8192 slots (~36 GiB if all used)
```

Restart after changing pool size (``clear_gds_dir_before_start`` recreates the
pool). First start with a large pool can take a minute while slots allocate.

**File descriptor limit:** NIXL opens one FD per pool slot during init. Docker
defaults to ~65536; a pool of **131072** fails with ``Too many open files``
unless ``make run`` passes a higher limit (``VLN_DOCKER_NOFILE``, default
**1048576**). Rule of thumb: ``VLN_NIXL_POOL_SIZE + 5000 < VLN_DOCKER_NOFILE``.
