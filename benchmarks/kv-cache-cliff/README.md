<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# kv-cache-cliff benchmark

Prefill-only **KV-cache cliff** sweep against any OpenAI-compatible vLLM
endpoint. Sends a long shared prefix + per-client unique suffix at increasing
concurrency and watches throughput for the cliff that appears once the
concurrent KV working set exceeds the VRAM (L1) budget and cached prefixes get
evicted. With an external KV tier (e.g. Optimus kvd on NVMe via AIS — AMD
Infinity Storage) the evicted prefixes spill to storage and stream back instead
of re-prefilling, so the cliff flattens.

`run_cliff.py` is endpoint-agnostic — point it at any `/v1` server:

```bash
python3 run_cliff.py \
  --endpoint http://127.0.0.1:8000 \
  --model openai/gpt-oss-120b \
  --concurrencies 16,32,48,64 \
  --isl 60000 --shared-prefix-tokens 60000 \
  --iters 3 --max-tokens 64 \
  --out cliff-$(date +%Y%m%d-%H%M%S).csv
```

See `run_cliff.py --help` for the full flag set and the module docstring for the
two-arm (VRAM-only vs VRAM+external) methodology.

## Driven by the Optimus AIS recipe

The [`vllm-optimus-ais`](../../recipies/vllm-optimus-ais/) recipe bind-mounts
this directory into its container at `/app/cliff` and runs the sweep with
`make cliff` (and `run-this.sh` for an ad-hoc smoke). The CSV columns
(`tput_total`, `ext_hit_pct`, `miss_pct`, `p50_s`/`p95_s`) are documented in
that recipe's README.
