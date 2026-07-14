<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# ais-snoop — AIS KFD kprobe Prometheus exporter

`ais-snoop` attaches a kprobe + kretprobe to a single amdgpu/KFD kernel symbol —
the **AIS** (AMD Infinity Storage / hipFile GDS/P2PDMA) entry point — and exports
call counts, a per-process breakdown, and latency on a Prometheus `/metrics`
endpoint (default `:9489`).

It is the AIS-path sibling of **hsa-snoop** (which probes
`kfd_ioctl_create_queue` and exports `hsa_*` on `:9488`): same
privileged / host-PID / tracefs runtime model, folded into the **same aai-day
image**, but implemented as a small `bpftrace` + Python wrapper rather than C++.

## The one thing you must set: `AIS_KFD_SYMBOL`

The exact AIS KFD kernel function is **not** derivable from this repo (hipFile is
cloned from `ROCm/rocm-systems` at image-build time), so the symbol is a runtime
input. Until it is set the exporter still serves, reporting
`ais_snoop_up=0{reason="no_symbol_configured"}`.

Discover candidate symbols on the GPU node:

```bash
sudo grep -E 'kfd|ais|hipfile|p2p|dmabuf' /sys/kernel/tracing/available_filter_functions
# or
grep -E 'kfd|ais' /proc/kallsyms
```

Then set `AIS_KFD_SYMBOL` (compose/`docker run` env) to the chosen function.

## Metrics

| Metric | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `ais_snoop_up` | gauge | `symbol`, `reason` | 1 while attached, else 0 (reason explains why) |
| `ais_kfd_calls_total` | counter | `symbol`, `comm` | cumulative kprobe hits; `comm=""` is the grand total |
| `ais_kfd_latency_seconds_sum` | counter | `symbol` | summed kprobe→kretprobe latency |
| `ais_kfd_latency_seconds_count` | counter | `symbol` | completed calls (kretprobe returns) |
| `ais_kfd_inflight` | gauge | `symbol` | calls entered but not yet returned |

Mean AIS-call latency:
`rate(ais_kfd_latency_seconds_sum[5m]) / rate(ais_kfd_latency_seconds_count[5m])`.

## Runtime requirements

- root + `--privileged` + `--pid host` (to see the vLLM/LMCache GPU processes)
- `bpftrace` on PATH (installed in the aai-day image); `/sys/kernel/debug` and
  `/sys/kernel/tracing` mounted; `/proc/kallsyms` readable
- kernel BTF (`/sys/kernel/btf/vmlinux`) is **not** required — the default probe
  program reads no function arguments. It is only needed if the program is later
  extended to decode args (e.g. a transfer size for an `ais_kfd_bytes_total`).

## Run standalone

```bash
docker run -d --name aai-day-ais-snoop --network host --pid host --privileged \
    --device /dev/kfd --device /dev/dri \
    -v /sys/kernel/debug:/sys/kernel/debug \
    -v /sys/kernel/tracing:/sys/kernel/tracing \
    -v /lib/modules:/lib/modules:ro -v /usr/src:/usr/src:ro \
    -e AIS_KFD_SYMBOL=<the_ais_kfd_symbol> \
    --entrypoint /usr/local/bin/ais-snoop rocm-aic-aai-day
```

Normally you don't run it by hand — the monitoring stack's `exporters` profile
(and the `.slurm/run-cliff.sbatch` docker-run fallback) launch it for you.
