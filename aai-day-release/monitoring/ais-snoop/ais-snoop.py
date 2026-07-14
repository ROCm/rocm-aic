#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# ais-snoop -- a bpftrace-driven Prometheus exporter for the AIS (AMD Infinity
# Storage / hipFile GDS/P2PDMA) KFD kernel path.
#
# It attaches a kprobe + kretprobe to a single amdgpu/KFD kernel symbol (the AIS
# entry point) and exports call counts, per-process breakdown, and latency on a
# Prometheus /metrics endpoint (default :9489).  It is the AIS-path sibling of
# hsa-snoop (which probes kfd_ioctl_create_queue and exports hsa_* on :9488):
# same privileged/host-PID/tracefs runtime model, folded into the same aai-day
# image, but implemented as a small bpftrace + Python wrapper.
#
#   Symbol   : $AIS_KFD_SYMBOL (or --symbol).  MUST be set to the real AIS KFD
#              function; with none configured the exporter still serves, reporting
#              ais_snoop_up=0{reason="no_symbol_configured"}.  A symbol absent
#              from /proc/kallsyms reports reason="symbol_not_in_kallsyms".
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
#   ais_kfd_calls_total{symbol}          total kprobe hits (cumulative)
#   ais_kfd_calls_total{symbol,comm}     per-process kprobe hits (cumulative)
#   ais_kfd_latency_seconds_sum{symbol}  summed kprobe->kretprobe latency
#   ais_kfd_latency_seconds_count{symbol}number of completed calls (returns)
#   ais_kfd_inflight{symbol}             calls entered but not yet returned
#
# rate(ais_kfd_latency_seconds_sum) / rate(ais_kfd_latency_seconds_count) gives
# the mean AIS-call latency (the summary-without-quantiles pattern -- reliable to
# source from bpftrace aggregates, unlike a reconstructed bucketed histogram).

import argparse
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
#   - @calls[comm]  per-process cumulative hit count
#   - @calls_total  cumulative hit count
#   - @start[tid]   entry timestamp, consumed by the kretprobe
#   - @lat_ns/@lat_count  cumulative latency sum + completed-call count
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
    @lat_ns = sum($d);
    @lat_count = count();
    delete(@start[tid]);
}}

interval:s:{interval}
{{
    print(@calls);
    print(@calls_total);
    print(@lat_ns);
    print(@lat_count);
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
        self.up = 0
        self.reason = "starting"

    def snapshot(self):
        with self.lock:
            return (
                dict(self.calls_by_comm),
                self.calls_total,
                self.lat_ns,
                self.lat_count,
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
        by_comm, total, lat_ns, lat_count, up, reason = self.state.snapshot()
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

        c_sum = CounterMetricFamily(
            "ais_kfd_latency_seconds_sum",
            "Cumulative kprobe->kretprobe latency for the AIS KFD call (seconds)",
            labels=["symbol"],
        )
        c_sum.add_metric([sym], lat_ns / 1e9)
        yield c_sum

        c_cnt = CounterMetricFamily(
            "ais_kfd_latency_seconds_count",
            "Number of completed AIS KFD calls (kretprobe returns)",
            labels=["symbol"],
        )
        c_cnt.add_metric([sym], lat_count)
        yield c_cnt

        g_inflight = GaugeMetricFamily(
            "ais_kfd_inflight",
            "AIS KFD calls entered but not yet returned",
            labels=["symbol"],
        )
        g_inflight.add_metric([sym], max(0, total - lat_count))
        yield g_inflight


def main() -> int:
    ap = argparse.ArgumentParser(
        description="bpftrace-driven Prometheus exporter for the AIS KFD path"
    )
    ap.add_argument(
        "--symbol",
        default=os.environ.get("AIS_KFD_SYMBOL", ""),
        help="amdgpu/KFD kernel symbol to kprobe (env AIS_KFD_SYMBOL)",
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
