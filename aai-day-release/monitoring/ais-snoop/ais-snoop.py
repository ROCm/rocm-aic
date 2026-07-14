#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# ais-snoop -- a bpftrace-driven Prometheus exporter for the AIS (AMD Infinity
# Storage) KFD kernel path.
#
# AIS is the AMDKFD_IOC_AIS_OP ioctl (opcode 0x87) added by the amdgpu-dkms
# driver: it does direct I/O between a file on a PCI-P2PDMA-reachable NVMe and
# GPU VRAM (see amd/amdkfd/kfd_ais.c / kfd_chardev.c in the amdgpu-dkms source).
# Its in-kernel entry point is kfd_ioctl_ais(), which dispatches both the
# KFD_IOC_AIS_READ and KFD_IOC_AIS_WRITE operations and calls kfd_ais_rw_file()
# to run the transfer -- so a kprobe/kretprobe pair on kfd_ioctl_ais measures the
# full AIS-call count and end-to-end latency.
#
# ais-snoop attaches that kprobe + kretprobe and exports call counts, a
# per-process breakdown, and latency on a Prometheus /metrics endpoint
# (default :9489).  It is the AIS-path sibling of hsa-snoop (which probes
# kfd_ioctl_create_queue and exports hsa_* on :9488): same privileged/host-PID/
# tracefs runtime model, folded into the same aai-day image, but implemented as a
# small bpftrace + Python wrapper.
#
#   Symbol   : $AIS_KFD_SYMBOL (or --symbol), default kfd_ioctl_ais -- the AIS
#              ioctl handler in the amdgpu-dkms driver.  Override only if a driver
#              revision renames it.  A symbol absent from /proc/kallsyms (e.g. the
#              amdgpu module is not loaded, or is too old to carry AIS) reports
#              ais_snoop_up=0{reason="symbol_not_in_kallsyms"}; an empty symbol
#              reports reason="no_symbol_configured".
#   Port     : $AIS_SNOOP_PORT (or --prometheus-port), default 9489.
#   Interval : how often bpftrace flushes its maps to us (default 5s).
#
# Runtime requirements (see monitoring/docker-compose.monitoring.yml):
#   - root + --privileged + --pid host (see the vLLM/LMCache GPU processes)
#   - bpftrace on PATH; /sys/kernel/debug + /sys/kernel/tracing mounted
#   - /proc/kallsyms readable (symbol preflight)
#   - kernel BTF (/sys/kernel/btf/vmlinux) or matching headers are only needed if
#     the probe program is extended to read function arguments; the default
#     program uses none, so plain call-count + latency work without BTF.
#
# Metrics:
#   ais_snoop_up{symbol,reason}          1 while bpftrace is attached, else 0
#   ais_kfd_calls_total{symbol}          total kprobe hits / attempts (cumulative)
#   ais_kfd_calls_total{symbol,comm}     per-process attempts (cumulative)
#   ais_kfd_latency_seconds_sum{symbol}  summed latency of SUCCESSFUL calls
#   ais_kfd_latency_seconds_count{symbol} number of SUCCESSFUL calls (retval >= 0)
#   ais_kfd_errors_total{symbol}         total failed calls (retval < 0)
#   ais_kfd_errors_total{symbol,code}    per-errno failed calls (code=EINVAL, ...)
#   ais_kfd_inflight{symbol}             calls entered but not yet returned
#
# Return-value gating: kfd_ioctl_ais() returns 0 on success or a negative errno.
# Latency is accumulated for successes only (so fast-failing rejects don't skew
# the mean); failures are counted per errno in ais_kfd_errors_total instead.
#   error rate : rate(ais_kfd_errors_total{code=""}) / rate(ais_kfd_calls_total)
#   mean latency (successful calls): the latency sum/count pair is exported as a
#     Prometheus summary (no quantiles) -- the reliable way to source aggregates
#     from bpftrace -- so rate(ais_kfd_latency_seconds_sum) /
#     rate(ais_kfd_latency_seconds_count) gives the mean.

import argparse
import errno
import json
import os
import subprocess
import sys
import threading
import time

try:
    from prometheus_client import start_http_server
    from prometheus_client.core import (
        REGISTRY,
        CounterMetricFamily,
        GaugeMetricFamily,
        SummaryMetricFamily,
    )
except ImportError:  # pragma: no cover - image always ships prometheus_client
    sys.stderr.write(
        "ais-snoop: python prometheus_client not found "
        "(expected in the aai-day image via vLLM deps)\n"
    )
    raise


def log(msg: str) -> None:
    sys.stderr.write(f"[ais-snoop] {msg}\n")
    sys.stderr.flush()


# bpftrace program template.  Deliberately argument-free (no BTF/headers needed):
#   - @calls[comm]  per-process cumulative hit count (all attempts)
#   - @calls_total  cumulative hit count (all attempts)
#   - @start[tid]   entry timestamp, consumed by the kretprobe
#   - @lat_ns/@lat_count  latency sum + count for SUCCESSFUL calls only
#   - @errors_total cumulative count of failed calls (retval < 0)
#   - @errors[ret]  per-errno failed-call count, keyed by the negative retval
# The kretprobe gates on the return value: kfd_ioctl_ais() returns 0 on success
# or a negative errno.  retval is the raw return register, whose upper 32 bits are
# undefined for an int-returning function, so we cast to (int32) before testing
# its sign -- otherwise a success (0) can read as a bogus large value.  Latency is
# accumulated for successes only, so fast-failing rejects (bad fd, unaligned size)
# don't skew the mean; failures are counted per errno instead.
# Maps are NOT cleared: they carry cumulative totals, which map straight onto
# Prometheus counters.  The interval probe just flushes them to our parser.
BPFTRACE_PROGRAM = r"""
kprobe:{symbol}
{{
    @calls[comm] = count();
    @calls_total = count();
    @start[tid] = nsecs;
}}

kretprobe:{symbol}
/@start[tid]/
{{
    $d = nsecs - @start[tid];
    $ret = (int32)retval;
    if ($ret >= 0) {{
        @lat_ns = sum($d);
        @lat_count = count();
    }} else {{
        @errors_total = count();
        @errors[$ret] = count();
    }}
    delete(@start[tid]);
}}

interval:s:{interval}
{{
    print(@calls);
    print(@calls_total);
    print(@lat_ns);
    print(@lat_count);
    print(@errors_total);
    print(@errors);
}}

END
{{
    clear(@start);
}}
"""


class State:
    """Latest cumulative values parsed from bpftrace, guarded by a lock."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls_by_comm: dict[str, int] = {}
        self.calls_total = 0
        self.lat_ns = 0
        self.lat_count = 0
        self.errors_total = 0
        # keyed by positive errno (e.g. 22 for EINVAL) -> cumulative count
        self.errors_by_code: dict[int, int] = {}
        self.up = 0
        self.reason = "starting"

    def snapshot(self):
        with self.lock:
            return (
                dict(self.calls_by_comm),
                self.calls_total,
                self.lat_ns,
                self.lat_count,
                self.errors_total,
                dict(self.errors_by_code),
                self.up,
                self.reason,
            )


def _apply_map(state: State, name: str, data) -> None:
    """Merge one bpftrace map/scalar print into shared state."""
    with state.lock:
        if name == "@calls" and isinstance(data, dict):
            # {"comm": count, ...}; values may be int or {"count": int}.
            for comm, val in data.items():
                if isinstance(val, dict):
                    val = val.get("count", 0)
                state.calls_by_comm[str(comm)] = int(val)
        elif name == "@calls_total":
            state.calls_total = int(data)
        elif name == "@lat_ns":
            state.lat_ns = int(data)
        elif name == "@lat_count":
            state.lat_count = int(data)
        elif name == "@errors_total":
            state.errors_total = int(data)
        elif name == "@errors" and isinstance(data, dict):
            # {"<neg retval>": count, ...}; keys are the signed errno bpftrace saw.
            for key, val in data.items():
                if isinstance(val, dict):
                    val = val.get("count", 0)
                try:
                    raw = int(key)
                except (TypeError, ValueError):
                    continue
                # store under the positive errno (|retval|)
                state.errors_by_code[abs(raw)] = int(val)


def bpftrace_reader(state: State, symbol: str, interval: int) -> None:
    """Run bpftrace and stream its JSON output into `state` forever.

    On any exit/error we mark down with a reason and retry after a short sleep,
    so a transient failure (symbol briefly unavailable, module reload) recovers.
    """
    prog = BPFTRACE_PROGRAM.format(symbol=symbol, interval=interval)
    while True:
        # Re-preflight each loop: the symbol may appear/disappear with module load.
        if not symbol_in_kallsyms(symbol):
            with state.lock:
                state.up = 0
                state.reason = "symbol_not_in_kallsyms"
            time.sleep(interval)
            continue
        log(f"starting bpftrace on {symbol} (interval={interval}s)")
        try:
            proc = subprocess.Popen(
                ["bpftrace", "-f", "json", "-e", prog],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            with state.lock:
                state.up = 0
                state.reason = "bpftrace_not_found"
            log("bpftrace not found on PATH")
            time.sleep(interval)
            continue

        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = obj.get("type")
            if kind == "attached_probes":
                with state.lock:
                    state.up = 1
                    state.reason = "ok"
                log(f"attached ({obj.get('data')})")
            elif kind == "map":
                for name, data in obj.get("data", {}).items():
                    _apply_map(state, name, data)

        # stdout closed -> bpftrace exited.
        err = (proc.stderr.read() if proc.stderr else "") or ""
        rc = proc.wait()
        with state.lock:
            state.up = 0
            state.reason = f"bpftrace_exited_rc{rc}"
        log(f"bpftrace exited rc={rc}: {err.strip()[:500]}")
        time.sleep(interval)


def symbol_in_kallsyms(symbol: str) -> bool:
    if not symbol:
        return False
    try:
        with open("/proc/kallsyms", "r") as fh:
            for row in fh:
                # columns: <addr> <type> <name> [<module>]
                parts = row.split()
                if len(parts) >= 3 and parts[2] == symbol:
                    return True
    except OSError:
        # Can't read kallsyms -> don't block; let bpftrace be the judge.
        return True
    return False


class AisCollector:
    """Custom collector: reports bpftrace's cumulative maps on each scrape."""

    def __init__(self, state: State, symbol: str) -> None:
        self.state = state
        self.symbol = symbol or ""

    def collect(self):
        (by_comm, total, lat_ns, lat_count, errors_total,
         errors_by_code, up, reason) = self.state.snapshot()
        sym = self.symbol

        g_up = GaugeMetricFamily(
            "ais_snoop_up",
            "1 while the AIS KFD kprobe is attached, else 0",
            labels=["symbol", "reason"],
        )
        g_up.add_metric([sym, reason], up)
        yield g_up

        c_calls = CounterMetricFamily(
            "ais_kfd_calls",
            "Cumulative AIS KFD kprobe hits",
            labels=["symbol", "comm"],
        )
        # Per-process breakdown; comm="" carries the grand total so the metric is
        # useful even before any process is attributed.
        c_calls.add_metric([sym, ""], total)
        for comm, n in by_comm.items():
            c_calls.add_metric([sym, comm], n)
        yield c_calls

        # Summary (without quantiles) so prometheus_client emits the exact
        # `ais_kfd_latency_seconds_sum` / `ais_kfd_latency_seconds_count` series.
        # Emitting these as CounterMetricFamily would append a spurious `_total`
        # suffix (-> ais_kfd_latency_seconds_sum_total), breaking the documented
        # rate(sum)/rate(count) mean-latency query. Both are monotonic, so rate()
        # works on them exactly as it does on counters.
        s_lat = SummaryMetricFamily(
            "ais_kfd_latency_seconds",
            "kprobe->kretprobe latency for SUCCESSFUL AIS KFD calls (seconds)",
            labels=["symbol"],
        )
        s_lat.add_metric([sym], count_value=lat_count, sum_value=lat_ns / 1e9)
        yield s_lat

        # Failed calls (kfd_ioctl_ais returned < 0), broken down by errno name.
        # code="" carries the grand total so the series exists (at 0) even when
        # nothing has failed, making rate(ais_kfd_errors_total) alertable.
        c_err = CounterMetricFamily(
            "ais_kfd_errors",
            "Cumulative failed AIS KFD calls (negative return), by errno",
            labels=["symbol", "code"],
        )
        c_err.add_metric([sym, ""], errors_total)
        for code_num, n in errors_by_code.items():
            code = errno.errorcode.get(code_num, str(code_num))
            c_err.add_metric([sym, code], n)
        yield c_err

        # Returns = successes (lat_count) + failures (errors_total); anything
        # entered but not yet returned is still in flight.
        g_inflight = GaugeMetricFamily(
            "ais_kfd_inflight",
            "AIS KFD calls entered but not yet returned",
            labels=["symbol"],
        )
        g_inflight.add_metric([sym], max(0, total - lat_count - errors_total))
        yield g_inflight


def main() -> int:
    ap = argparse.ArgumentParser(
        description="bpftrace-driven Prometheus exporter for the AIS KFD path"
    )
    ap.add_argument(
        "--symbol",
        default=os.environ.get("AIS_KFD_SYMBOL", "") or "kfd_ioctl_ais",
        help="amdgpu/KFD kernel symbol to kprobe (env AIS_KFD_SYMBOL, "
             "default kfd_ioctl_ais -- the AIS ioctl handler in amdgpu-dkms)",
    )
    ap.add_argument(
        "--prometheus-port",
        type=int,
        default=int(os.environ.get("AIS_SNOOP_PORT", "9489")),
        help="Prometheus /metrics port (env AIS_SNOOP_PORT, default 9489)",
    )
    ap.add_argument(
        "--interval",
        type=int,
        default=int(os.environ.get("AIS_SNOOP_INTERVAL", "5")),
        help="bpftrace map-flush interval in seconds (default 5)",
    )
    args = ap.parse_args()

    state = State()
    REGISTRY.register(AisCollector(state, args.symbol))
    start_http_server(args.prometheus_port)
    log(f"serving /metrics on :{args.prometheus_port}")

    if not args.symbol:
        # Serve, but stay down with a clear reason until a symbol is configured.
        with state.lock:
            state.up = 0
            state.reason = "no_symbol_configured"
        log("no symbol configured; set AIS_KFD_SYMBOL / --symbol to the AIS KFD "
            "function (ais_snoop_up stays 0 until then)")
        while True:
            time.sleep(3600)

    # Reader loop runs in the foreground thread's stead; keep main alive.
    t = threading.Thread(
        target=bpftrace_reader,
        args=(state, args.symbol, args.interval),
        daemon=True,
    )
    t.start()
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
