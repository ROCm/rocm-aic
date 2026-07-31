# Key Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `ROCM_ARCH` | auto-detected | GPU arch, e.g. `gfx942` |
| `HF_TOKEN` | — | HuggingFace access token (required) |
| `VLLM_MODEL` | `openai/gpt-oss-120b` | Model to serve |
| `NVME_DATA` | `/mnt/lmcache-nvme` | Host path for NVMe L2a pool |
| `NFS_DATA` | `/mnt/lmcache-nfs` | Host path for NFS-over-RDMA L2b pool |
| `GDS_SLAB_DATA` | — | Host path for GDS NVMe slab (GDS L1 mode only) |
| `GDS_MODE` | — | Set to `1` to enable GDS L1 mode |
| `AIC_L2_BACKEND` | `nixl` | LMCache L2 backend: `nixl` (AIS_MT NVMe + POSIX NFS) or `local_disk` (native LocalDiskBackend via a mounted config) |
| `KV_TRANSFER_ARG` | LMCacheMPConnector JSON | vLLM `--kv-transfer-config` arg; empty = plain vLLM (the cliff `vram` baseline). Wrap the JSON in single quotes |
| `LMCACHE_L1_SIZE_GB` | `20` | MP server L1 cap in GiB (DRAM L1 in nvme mode, hipFile slab size in GDS mode) |
| `LMCACHE_CHUNK_SIZE` | `256` | Shared MP-server/coordinator token chunk size |
| `LMCACHE_COORDINATOR_PORT` | `9300` | Coordinator HTTP listen port |
| `LMCACHE_COORDINATOR_URL` | `http://127.0.0.1:9300` | Coordinator URL used by the host-networked MP server |
| `LMCACHE_COORDINATOR_EVENT_FLUSH_INTERVAL` | `1.0` | Seconds between MP-server cache-event flush attempts |
| `LMCACHE_NVME_POOL` | `4096` | NIXL pool slots for NVMe adapter |
| `LMCACHE_NVME_SLOT_SIZE` | `268435456` | NIXL file size per NVMe pool slot, bytes (256 MiB) |
| `LMCACHE_NFS_POOL` | `1024` | NIXL pool slots for NFS adapter |
| `LMCACHE_NFS_SLOT_SIZE` | same as `LMCACHE_NVME_SLOT_SIZE` | NIXL file size per NFS/POSIX pool slot; must fit at least one full KV chunk |
| `VLM_ATTENTION_BACKEND` | `TRITON_ATTN` | vLLM `--attention-backend` (TRITON_ATTN supports KV connectors) |
| `VLM_KV_CACHE_DTYPE` | `fp8` | vLLM `--kv-cache-dtype` (`auto` for non-fp8 arches) |
| `VLLM_EXTRA_ARGS` | — | Extra vLLM args appended verbatim (e.g. `--hf-overrides '{...}'`; single-quote embedded JSON) |
| `COMPOSE_PLUGIN_VERSION` | `v2.40.0` | docker compose v2 plugin version installed by `make ensure-compose` / `ensure_compose` when missing |
| `AIC_TINY_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Model served by `make tiny-test` (end-to-end serve check) |
| `HF_HOME` | `~/.cache/huggingface` (`/shared_nfs/huggingface` on SPUR) | Persistent Hugging Face cache used by normal, cliff, monitoring, and tiny-model paths |
| `TENSOR_PARALLEL_SIZE` | `1` | vLLM tensor parallel degree |
| `GPU` | `0` | ROCR_VISIBLE_DEVICES for the vllm container |
| `AIC_NVME_AUTO` | `1` (cliff) | Auto-detect a dedicated local NVMe for the LMCache tiers: reuse a mounted `aic-lmcache` volume, else format+mount a raw non-root spare, else use a non-root mounted NVMe, else node-local `/tmp`. `0` forces `/tmp`; needs passwordless `sudo` to format/mount |
| `AIC_NVME_MOUNT` | `/mnt/aic-lmcache` | Mountpoint used when auto-provisioning a spare NVMe (left mounted for reuse) |
| `AIC_MONITORING` | `1` | Auto-start the Prometheus sidecar in cliff sbatch runs (`0` to skip) |
| `AIC_METRICS_DIR` | `logs/<job-id>/prometheus` (cliff) | Prometheus TSDB dir — bind-mount an NFS path here |
| `AIC_EXPORTERS` | `1` (cliff) / `0` (make) | Also launch containerized node + AMD GPU exporters |
