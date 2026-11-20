@AGENTS.md

# CLAUDE.md — AMD Infinity Context (AIC)

## SPUR Cluster Access

Reach the cluster over ssh at `crs-spur.crusoe.amd.com`, a load balancer in
front of the login nodes `crsuse2-slog-00{3,4,5}`. Each backend presents a
**different** host key, so all three must be in `known_hosts` under both the
load-balancer name and their own — pinning only the one you happen to land on
fails intermittently as the balancer re-hashes.

Do **not** export `SPUR_CONTROLLER_ADDR`. Login and compute nodes both set it
in `/etc/environment`, and `pam_env` applies it to non-interactive
`ssh host 'cmd'` sessions too, so `squeue`/`srun` work with no setup. The
cluster's value lists all three controllers (`crs-m2m-cpu-spur-00{5,6,7}`),
which form a Raft quorum; overriding it with a single address silently drops
that redundancy.

- **Partition:** `amd-spur-v2` (since 2026-11-03; `amd-spur` has no nodes and
  jobs submitted to it pend forever on `PartitionConfig`). Scripts and `make`
  read it from `AIC_SPUR_PARTITION`; use that, not the literal name
- **No `--account` flag needed** for `srun`
- Node naming convention: `crsuse2-m2m-v3-NNN` (`amd-spur-v2`; older nodes
  were `crsuse2-m2m-NNN`)
- Controllers are *not* compute nodes and never appear in `sinfo`

## Current stack (2026-11-23)

`docker/docker-bake.hcl` owns every build pin (see `docs/ENVIRONMENT.md`); the
Dockerfiles carry bare `ARG`s with no defaults. Current pins: base image
`rocm/dev-ubuntu-26.04:10.2.0-full` (ROCm 10.2.0, Ubuntu 26.04, Python 3.14),
PyTorch source-built from `ROCm/pytorch@release/2.15`, torchvision `v0.29.2`,
Triton `release/internal/3.8.x`, vLLM `v0.34.0`, LMCache `v0.5.8`, NIXL
`v1.5.2` on UCX `v1.22.0`, AITER `v0.1.29.post1`, hsa-snoop `aba6a306`
(unreleased, `feat/cpu-deload`). KVBench
stays on LMCache 0.5.6. Validate changes on an MI355X before modifying the
stack; gfx1201 built 2026-11-23, MI355X unvalidated.

- **The LMCache connector config must set `"lmcache.mp.use_vmm_api": false`
  unless `patches/vllm/02-vllm-rocm-extensible-kv-fd-exporter.patch` is in
  the build.** LMCache v0.5.8 asks vLLM for fd-exported (VMM) KV memory by
  default, and vLLM v0.34.0 refuses that on ROCm at startup (LMCache issue
  #6620). Every connector JSON in the repo carries `use_vmm_api: false`; a
  hand-written `--kv-transfer-config` needs it too. The ROCm VMM path landed
  in LMCache v0.5.8 (#6626) but requires vLLM PR #67577 (backported by patch
  `02`) — that PR narrows the extensible-KV-cache ROCm guard from
  `needs_single_chunk()` to `KVCacheAccess.REGISTERS in kv_cache_access` so
  fd-exporting connectors keep the extensible cache. With patch `02` applied,
  `use_vmm_api: true` works on gfx1201 (validated 2026-11-23) with
  dma-buf IPC; 300s stress runs complete with zero errors.
  The performance delta vs `use_vmm_api: false` is entirely from the smaller
  extensible KV cache start size (832 blocks, 1.83 GiB on 16 GB gfx1201 at
  0.74 utilization) — GPU hits drop ~33% → ~11%, computed rises ~33% → ~47%.
  On SPUR with more VRAM the extensible cache self-sizes larger. Keep
  `use_vmm_api: false` as the production default; use `LMCACHE_USE_VMM_API=true`
  to test the VMM path. Drop this note once VLLM_REF is bumped to a tag
  containing #67577.
- **Judge KV offload by counters, not log wording.** Use prompts of several
  256-token chunks, restart only vLLM (`make restart-vllm`), and check
  `lmcache_mp_num_finished_stores_total` and
  `vllm:external_prefix_cache_hits_total`. `Session ... has no lookup ipc key,
  skipping touch` is normal for a prompt under one chunk, a repeat against a
  warm vLLM shows no LMCache activity, and the store line now reads
  `Stored N of M requested tokens`.
- **Lazy offload** (`lazy_offload: true` in the Makefile's `_MP_CONNECTOR_JSON`)
  defers a store until vLLM is about to evict the blocks, or about 30 s have
  passed and another request arrives. An idle engine does not flush, and
  `/reset_prefix_cache` or a vLLM restart drops whatever is pending.
  `make vllm-reset-test` therefore runs with lazy offload off, unless you set
  `KV_TRANSFER_ARG` yourself.
- **The KV cache dtype defaults to `auto`** (`VLM_KV_CACHE_DTYPE`). `fp8` gave
  repeated-token glitches with Qwen2.5-3B on gfx1201. `fp8` holds about twice
  the KV tokens, so cliff results from before this change are not comparable.
- **LMCache hit and miss counts come from upstream counters**
  (`lmcache_mp_lookup_hit_l1_keys_total`, `lmcache_mp_lookup_hit_l2_keys_total`,
  requested minus hit tokens for misses); the carried prefetch-counter patch
  was dropped.
- `POST /reset_prefix_cache` still needs `VLLM_SERVER_DEV_MODE=1` (the compose
  default); `?reset_external=true` also clears LMCache.

## Historical ROCm 7.14 torch wheel integration issue (2026-07-20)

The Dockerfile was temporarily updated to use AMD's native ROCm 7.14 wheel index
(`repo.amd.com/rocm/whl-multi-arch/`) for PyTorch. Testing on the SPUR
cluster (MI355X / gfx950) revealed a **blocking incompatibility**:

**`torch 2.12.0+rocm7.14.0` ships Triton 3.7.1, which breaks older vLLM 0.25.1.**
vLLM's engine core fails to start on gfx950 nodes — Triton kernel calls
failed because Triton was disabled at startup due to circular import side-effects
in vLLM's platform detection code.

The fix was to source-build PyTorch (then `ROCm/pytorch@release/2.13` on the
ROCm 7.14.1 base image, with vLLM `v0.29.0`, LMCache `v0.5.6`, AITER
`v0.1.22.post1` and NIXL `v1.4.1`) and to split the single `docker/Dockerfile`
into three independent Dockerfiles (`docker/base/`, `docker/vllm/`,
`docker/lmcache/`) enabling parallel builds. The pins have since moved; see
"Current stack" above.

vLLM v0.29.0 stopped calling `logger.warning_once()` from
`vllm.platforms.rocm._get_gcn_arch()` when the amdsmi probe fails, so the old
ROCm-AIC circular-import patch was retired. The SPUR validation must confirm
that the vLLM ROCm-platform import and engine startup remain healthy.

### Historical unblock options

1. **Use gfx942 (MI300X) nodes** — the Triton issue manifests on gfx950
   (MI355X) specifically; gfx942 nodes have not been fully tested but the
   hipErrorInvalidImage issue is resolved with `AIC_ROCM_ARCH=gfx942`.

2. **Use the current vLLM ROCm pin** — `docker/docker-bake.hcl` pins
   `VLLM_REF` (`v0.29.0` at the time, `v0.34.0` now; it owns every build pin,
   see `docs/ENVIRONMENT.md`); retain this unless SPUR validation proves a
   different version is needed.

3. **Upgrade vLLM further** — only after validating the current pin on SPUR;
   newer vLLM versions may require additional patch rework.

### SPUR cluster submission quirks

- The SPUR scheduler **requires** a GPU request (`--gpus=N`) for every job,
  including CPU-only build jobs. `run-build-distribute.sh` sets `--gpus=1`
  automatically when `AIC_SPUR_CLUSTER=1`. Override with `AIC_BUILD_GPUS=N`
  if you need a different count.
- Do NOT use `--no-requeue` — SPUR sbatch doesn't support it.
- Submit with `--chdir=/shared_nfs/$USER` when `/home` is at quota.
- Use `SLURM_SUBMIT_DIR=/path/to/repo` to tell the cliff script where the repo is.
- Pin to a node with no pre-existing containers: `--nodelist=crsuse2-m2m-042`
  (as of 2026-07-20 crsuse2-m2m-036 has a long-running primus container
  that blocks GPU access for new containers).
- All SPUR nodes are **MI355X (gfx950)**, not MI300X (gfx942).

### Full analysis

See `.docs-remove/rocm-7.14-torch-wheel-integration.md` for the complete
11-bug chain discovered during SPUR testing, including root causes and fix
status for each.

## Clients run on the bridge (only Grafana is published)

SPUR nodes are multi-tenant, so no container binds a fixed host port, and only
Grafana (for browsers) is published at all, on `127.0.0.1`: under `make` on the
lowest free port from 3000, which it keeps across a down/up; from compose alone
on a Docker-chosen port. `make endpoints` prints it. Servers live on the `aic` bridge (or join a
bridged container's netns, as `vllm` joins `aic-lmcache`), and every client runs
in a container there: `aic-client` for make targets, a throwaway
`docker run --network aic` on SPUR. `aic-client` is a built image
(`docker/client/Dockerfile`, bake target `client`, `make client-build`; compose
builds it when missing) holding every benchmark client: the `benchmarks/`
scripts, `agent-kv` and `aiperf` (own venv), no ROCm, no `lm_eval`. One CLI
drives them, `aic-bench <subcommand>` (`make bench BENCH=<subcommand>
BENCH_ARGS='...'`, `make bench-list`); it runs the scripts from `/repo` when
the checkout is mounted, so an edit needs no rebuild, and fills in
`AIC_BENCH_ENDPOINT` / `AIC_BENCH_MODEL` / `AIC_BENCH_OUT`. See
`benchmarks/README.md`. One-off requests from host code go through
`aic_fetch <url> [<json-body>]` in `scripts/aic-endpoints.sh` (set
`AIC_FETCH_IMAGE` to an image already on the node); never `curl localhost:<port>`.
Address a stack by container name, not the shared `aic-vllm` alias, which two
stacks on one node's bridge would both answer to: `aic_bridge_url <container>
<port>` resolves it (`aic-vllm-gpu0` → `http://aic-lmcache:8000`).

## gfx1201 (RDNA4 / RX 9070 XT) local development notes (2026-09-03)

The RX 9070 XT (Navi 48, gfx1201, 16 GB) is supported in the default multi-arch
build (`AIC_ROCM_ARCH` default includes `gfx1201`). Key differences from CDNA:

- **PyTorch and Triton**: both are source-built in `aic-base` (torch 2.15,
  Triton 3.8) and work on gfx1201; serving was validated on this node on
  2026-11-17. Do not switch to wheels from `repo.amd.com` for gfx1201.
- **NIXL AIS_MT**: hipFile P2PDMA is CDNA-only; use `AIC_L2_BACKEND=nixl_posix`
  on gfx1201 nodes (the SPUR default already does this).
- **VFIO**: the GPU is typically VFIO-passed to a VM on workstations. Rebind to
  `amdgpu` before running Docker containers:

  ```bash
  echo "0000:10:00.0" | sudo tee /sys/bus/pci/devices/0000:10:00.0/driver/unbind
  echo "0000:10:00.0" | sudo tee /sys/bus/pci/drivers/amdgpu/bind
  ```

- **LMCache MP connector — host kernel + IPC requirements (2026-09-08, updated)**:
  Cross-container HIP IPC requires **amdgpu-dkms 7.1.3** (from
  `amdgpu-install 31.50` / therock-10.0) for `kfd_ioctl_ipc_import_handle`.
  The in-tree `amdgpu.ko` (kernel `7.0.0-28-generic`) is missing these ioctls.
  Install once per node:

  ```bash
  # Add the AMD amdgpu-install 31.50 repo and install the DKMS module
  # (the 31.50 deb configures https://repo.radeon.com/amdgpu/31.50/ubuntu)
  sudo apt-get install -y amdgpu-dkms   # after adding the 31.50 source
  sudo modprobe -r amdgpu && sudo modprobe amdgpu
  echo "0000:10:00.0" | sudo tee /sys/bus/pci/drivers/amdgpu/bind
  ```

  dma-buf IPC (the default) does not need the DKMS amdgpu ioctls, but vllm
  and lmcache must share a network namespace (ROCR passes the dmabuf fd over
  the abstract socket `@hsa<pid>`) and should share the PID namespace
  (`network_mode: container:aic-lmcache` + `pid: service:lmcache` — the
  compose defaults; no make override needed).

  Validated 2026-11-17 on ROCm 10.1 and 2026-11-19 on ROCm 10.2, both with
  amdgpu-dkms 7.1.9: in the default
  (non-legacy) mode with those shared namespaces, stock HIP memory-handle IPC
  registers, stores and retrieves KV across the two containers, so the AIC
  VMM/SCM_RIGHTS workaround was removed. Cross-process HIP *event* handles are
  still bypassed (`docker/lmcache/rocm_event_ipc_patch.py`).

  The standard vllm-reset-test command on gfx1201:

  ```bash
  make vllm-reset-test \
    ROCM_ARCH=gfx1201 \
    VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct \
    AIC_L2_BACKEND=nixl_posix \
    NVME_DATA=<dir on the data NVMe> \
    VLM_LOAD_FORMAT=auto \
    LMCACHE_L1_SIZE_GB=2 \
    VLM_GPU_MEMORY_UTILIZATION=0.74
  ```

  Additional gfx1201 requirements:
  - `VLM_GPU_MEMORY_UTILIZATION=0.74` on this 16 GB card — vLLM v0.34.0
    introduced the extensible KV cache (on by default), which measures free
    memory after warmup and defaults `gpu_memory_utilization` to 1.0. On a
    16 GB card this still OOMs at startup if too much is pre-allocated, so
    0.74 is still recommended. The reset test (which adds `--enforce-eager`)
    then gets a KV cache of about 152,000 tokens; with CUDA graphs it is far
    smaller. 0.74 only holds for short prompts: prompts of several thousand
    tokens (AgentX, 2026-11-18) ran the engine out of memory at 0.74, and
    with CUDA graphs a 32,768-token context does not fit at all. For those use
    `make up-dev` (eager) with `VLM_GPU_MEMORY_UTILIZATION=0.68`, which gives
    a KV cache of about 124,000 tokens. On a node with more VRAM (or on SPUR)
    removing `VLM_GPU_MEMORY_UTILIZATION` lets the extensible KV cache size
    itself and may increase throughput.
  - `VLM_LOAD_FORMAT=auto` — fastsafetensors GDS is CDNA-only. **Always pass
    this on fresh starts on gfx1201:** without it, fastsafetensors allocates a
    ~600 MiB temporary tensor clone during weight loading that OOMs the 16 GB
    card at 0.74 util; `auto` (safetensors) does not have this overhead.
  - `LMCACHE_L1_SIZE_GB=2` for this 16 GB node — default 20 GiB L1 prevents
    eviction to L2 in the reset test.
  - HF_TOKEN is picked up from the environment; no HF_TOKEN_FILE needed.
  - **Locate the L2 NVMe; never assume a path.** `NVME_DATA` must be a
    directory on a dedicated data NVMe, not on the root disk. Mount points and
    device names differ per node and change (device names can shift after a
    `modprobe -r/modprobe amdgpu` cycle), and a path that exists proves
    nothing: the Makefile default `/mnt/lmcache-nvme` is created on the root
    filesystem if nothing is mounted there, and the stack runs happily on it.
    Before any command that takes `NVME_DATA`:
    1. `lsblk -o NAME,SIZE,MODEL,MOUNTPOINT` to find the data drive (not the
       one holding `/`) and where it is mounted; mount it if it is not.
    2. `findmnt -no SOURCE,TARGET -T <dir>` to confirm the directory you are
       about to pass resolves to that drive.
    3. Use a subdirectory owned by the invoking user (the stack runs
       non-root): `sudo install -d -o "$(id -u)" -g "$(id -g)" <mount>/aic-nvme`
       when the mount point itself is root-owned.

    On snoc-thinkstation as of 2026-11-18 the data drive is the 1.8 TB WD
    Black SN850X, mounted at `/mnt/kv-nvme`, so `NVME_DATA=/mnt/kv-nvme/aic-nvme`.

- **Profile capture** on a local gfx1201 node (no cluster required):

  ```bash
  make capture-profile-local \
    IMAGE_REF=<image> ROCM_ARCH=gfx1201 \
    AIC_CAPTURE_MODEL=Qwen/Qwen2.5-3B-Instruct \
    HF_TOKEN_FILE=~/.cache/huggingface/token
  ```
  Output lands in `profiles/captures/`. The captured gfx1201 pack
  (`Qwen-Qwen2.5-3B-Instruct-20260902-182346.json`) is the first RDNA4 pack
  in the repo; see `prometheus-dump.md` for emulated-vs-real latency numbers.
