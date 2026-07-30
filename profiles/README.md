# Emulator profile packs

Profile packs for the LLM-Emu emulation mode (see [../docs/EMULATE.md](../docs/EMULATE.md)).

Each pack is a step-latency model of **one GPU running one model under one
serving configuration**, captured from a real serve on real hardware. In
emulation mode the pack replaces the GPU forward pass, so the whole vLLM
scheduler / KV-cache / HTTP stack can be exercised on a CPU-only node at
hardware-realistic timings.

Everything in this directory is baked into the image at
`/opt/llm-emu/profiles/`, alongside the upstream NVIDIA A40 example pack that
ships inside `llm-emu` itself.

| Pack | GPU | Model | Notes |
| --- | --- | --- | --- |
| `MI300X-Qwen3-8B.json` | AMD Instinct MI300X (gfx942) | `Qwen/Qwen3-8B` | AIC default; captured 2026-07-30 with `make profile-capture` (13 400 step samples, KV-depth aware, prefix caching off) |
| `A40-Q8-Qwen3-8B.json` | NVIDIA A40 | `Qwen/Qwen3-8B` | upstream example, from the `llm-emu` repo (not in this directory) |

Accuracy of the shipped MI300X pack, measured by `make emulate-validate` (real
MI300X vs emulated, same benchmark points): every metric within **15%** at
concurrency 1 and 8 (TTFT -14.7%/-8.5%, TPOT +7.0%/+8.1%, throughput
-5.1%/-2.2%), but still ~40% out on TTFT and TPOT at concurrency 16. Good enough
to study scheduling and admission behavior at low concurrency, not a substitute
for a real benchmark — see [../docs/EMULATE.md](../docs/EMULATE.md) §4.2.

The pack is self-describing: it records the GPU (including the KV pool vLLM
measured — 147 GiB here) and the serving configuration it was captured under, so
a replay can be checked against the capture rather than trusted.

## Provenance

Every pack has a sibling `<pack>.capture.txt` recording the node, image, serve
flags and benchmark sweep it came from. **A pack is only valid for the serving
configuration it was captured under** — change `--max-num-batched-tokens`,
enable async scheduling, or switch attention backend, and it is stale.

## Adding a pack for a new GPU

```bash
# 1. build the full image for that arch and capture on such a node
AIC_ROCM_ARCH=gfx950 make dist-build
AIC_ROCM_ARCH=gfx950 AIC_CAPTURE_CONSTRAINT=GFX950 make profile-capture

# 2. score it: replay on a CPU node, diff against the real run
AIC_VALIDATE_PACK=/scratch/$USER/images/profiles/<pack>.json make emulate-validate

# 3. copy pack + .capture.txt here, named <GPU>-<Model>.json, and rebuild
```

See [../docs/EMULATE.md](../docs/EMULATE.md) for what "good" looks like and how
to read the validation deltas.
