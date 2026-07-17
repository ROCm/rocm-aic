<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# ais-snoop — AIS KFD kprobe Prometheus exporter

`ais-snoop` attaches a kprobe + kretprobe to the **AIS** (AMD Infinity Storage)
KFD ioctl handler in the `amdgpu-dkms` driver and exports call counts, a
per-process breakdown, and latency on a Prometheus `/metrics` endpoint
(default `:9489`).

AIS is the `AMDKFD_IOC_AIS_OP` ioctl (opcode `0x87`): it performs direct I/O
between a file on a PCI-P2PDMA-reachable NVMe and GPU VRAM (see
`amd/amdkfd/kfd_ais.c` and `kfd_chardev.c` in the amdgpu-dkms source). Its
in-kernel entry point is `kfd_ioctl_ais()`, which dispatches both
`KFD_IOC_AIS_READ` and `KFD_IOC_AIS_WRITE` and calls `kfd_ais_rw_file()` to run
the transfer — so a kprobe/kretprobe pair on `kfd_ioctl_ais` measures the full
AIS-call count and end-to-end latency.

It is the AIS-path sibling of **hsa-snoop** (which probes
`kfd_ioctl_create_queue` and exports `hsa_*` on `:9488`): same
privileged / host-PID / tracefs runtime model, folded into the **same AIC
image**, but implemented as a small `bpftrace` + Python wrapper rather than C++.

## Symbol: `AIS_KFD_SYMBOL` (default `kfd_ioctl_ais`)

The exporter defaults `AIS_KFD_SYMBOL` to `kfd_ioctl_ais`, the AIS ioctl handler
shipped by the amdgpu-dkms driver, so it works out of the box on a node whose
driver carries AIS. You only need to override it if a driver revision renames the
symbol.

If the chosen symbol isn't present in the running kernel (the amdgpu module isn't
loaded, or is too old to include AIS), the exporter still serves and reports
`ais_snoop_up=0{reason="symbol_not_in_kallsyms"}`. Setting `AIS_KFD_SYMBOL=""`
explicitly reports `ais_snoop_up=0{reason="no_symbol_configured"}`.

Confirm the symbol on a GPU node:

```bash
grep -w kfd_ioctl_ais /proc/kallsyms
# or, for the set of probeable AIS functions:
grep -E 'kfd_ais|kfd_ioctl_ais' /sys/kernel/tracing/available_filter_functions
```

## Metrics

| Metric | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `ais_snoop_up` | gauge | `symbol`, `reason` | 1 while attached, else 0 (reason explains why) |
| `ais_kfd_calls_total` | counter | `symbol`, `comm` | cumulative attempts; `comm=""` is the grand total |
| `ais_kfd_latency_seconds_sum` | summary | `symbol` | summed latency of **successful** calls |
| `ais_kfd_latency_seconds_count` | summary | `symbol` | **successful** calls (`retval >= 0`) |
| `ais_kfd_errors_total` | counter | `symbol`, `code` | failed calls by errno name; `code=""` is the grand total |
| `ais_kfd_inflight` | gauge | `symbol` | calls entered but not yet returned |

### Return-value gating

`kfd_ioctl_ais()` returns `0` on success or a negative errno. The kretprobe reads
that return value (cast to `int32`, since an `int`-returning function leaves the
upper 32 bits of the return register undefined) and splits the outcome:

- **success** (`retval >= 0`) → feeds `ais_kfd_latency_seconds_{sum,count}`, so
  fast-failing rejects (`-EINVAL` for an unaligned size, `-EBADF` for a bad fd,
  `-ENODEV` when there's no PCI-P2PDMA path) don't drag the mean latency down.
- **failure** (`retval < 0`) → increments `ais_kfd_errors_total{code=<errno>}`
  (e.g. `code="EINVAL"`); the numeric errno is used if the name is unknown.

Useful queries:

```promql
# mean latency of successful AIS calls
rate(ais_kfd_latency_seconds_sum[5m]) / rate(ais_kfd_latency_seconds_count[5m])
# AIS error rate (fraction of attempts that failed)
rate(ais_kfd_errors_total{code=""}[5m]) / rate(ais_kfd_calls_total{comm=""}[5m])
# which errno is dominating
topk(5, rate(ais_kfd_errors_total{code!=""}[5m]))
```

The latency pair is a Prometheus **summary** (no quantiles) so the series names
come out exactly as `ais_kfd_latency_seconds_sum` / `_count`.

## Runtime requirements

- root + `--privileged` + `--pid host` (to see the vLLM/LMCache GPU processes)
- `bpftrace` on PATH (installed in the AIC image); `/sys/kernel/debug` and
  `/sys/kernel/tracing` mounted; `/proc/kallsyms` readable
- kernel BTF (`/sys/kernel/btf/vmlinux`) is **not** required — the default probe
  program reads no function arguments. It is only needed if the program is later
  extended to decode args (e.g. a transfer size for an `ais_kfd_bytes_total`).

## Run standalone

```bash
docker run -d --name aic-ais-snoop --network host --pid host --privileged \
    --device /dev/kfd --device /dev/dri \
    -v /sys/kernel/debug:/sys/kernel/debug \
    -v /sys/kernel/tracing:/sys/kernel/tracing \
    -v /lib/modules:/lib/modules:ro -v /usr/src:/usr/src:ro \
    --entrypoint /usr/local/bin/ais-snoop rocm-aic
```

The kprobe target defaults to `kfd_ioctl_ais`; add `-e AIS_KFD_SYMBOL=<symbol>`
only to override it.

Normally you don't run it by hand — the monitoring stack's `exporters` profile
(and the `.slurm/run-cliff.sbatch` docker-run fallback) launch it for you.
