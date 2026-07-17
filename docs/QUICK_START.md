# Quick Start

All commands run from the repo root with `make <target>`.

## 1. Build the image

```bash
make build ROCM_ARCH=gfx942
```

## 2. Start the stack (standard mode: DRAM L1 + NVMe L2a + NFS L2b)

```bash
make up \
    HF_TOKEN=hf_... \
    NVME_DATA=/mnt/lmcache-nvme \
    NFS_DATA=/mnt/lmcache-nfs \
    VLLM_MODEL=openai/gpt-oss-120b
```

## 3. Start in GDS L1 mode (hipFile NVMe slab as L1, no L2)

```bash
make up-gds-l1 \
    HF_TOKEN=hf_... \
    GDS_SLAB_DATA=/mnt/lmcache-nvme \
    VLLM_MODEL=openai/gpt-oss-120b
```

## 4. Install host-side benchmark dependencies

```bash
make venv
source .venv/bin/activate
```

## 5. Run the cliff benchmark

Run the baseline arm first, then the AIC arm against separate endpoints:

```bash
# Arm A: baseline (vram_only) — must be run against a plain vLLM endpoint
make cliff \
    BENCH_ARM=vram_only \
    BENCH_ENDPOINT=http://localhost:8001 \
    BENCH_MODEL=openai/gpt-oss-120b

# Arm B: AIC (kvd_v2) — run against the vllm-lmcache stack on port 8000
make cliff \
    BENCH_ARM=kvd_v2 \
    BENCH_ENDPOINT=http://localhost:8000 \
    BENCH_MODEL=openai/gpt-oss-120b
```

## 6. Generate cliff charts

```bash
make plot
# → logs/manual/plots/cliff-throughput.png
# → logs/manual/plots/cliff-latency-p50.png
# → logs/manual/plots/cliff-latency-p95.png
```
