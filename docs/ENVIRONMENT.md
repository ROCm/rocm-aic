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
| `LMCACHE_L1_SIZE_GB` | `20` | L1 memory cap in GiB |
| `LMCACHE_NVME_POOL` | `4096` | NIXL pool slots for NVMe adapter |
| `LMCACHE_NFS_POOL` | `1024` | NIXL pool slots for NFS adapter |
| `TENSOR_PARALLEL_SIZE` | `1` | vLLM tensor parallel degree |
| `GPU` | `0` | ROCR_VISIBLE_DEVICES for the vllm container |
| `AIC_NVME_AUTO` | `1` (cliff) | Auto-detect a dedicated local NVMe for the LMCache tiers: reuse a mounted `aic-lmcache` volume, else format+mount a raw non-root spare, else use a non-root mounted NVMe, else node-local `/tmp`. `0` forces `/tmp`; needs passwordless `sudo` to format/mount |
| `AIC_NVME_MOUNT` | `/mnt/aic-lmcache` | Mountpoint used when auto-provisioning a spare NVMe (left mounted for reuse) |
| `AIC_MONITORING` | `1` | Auto-start the Prometheus sidecar in cliff sbatch runs (`0` to skip) |
| `AIC_METRICS_DIR` | `logs/<job-id>/prometheus` (cliff) | Prometheus TSDB dir — bind-mount an NFS path here |
| `AIC_EXPORTERS` | `1` (cliff) / `0` (make) | Also launch containerized node + AMD GPU exporters |
