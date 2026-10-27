#!/usr/bin/env python3
"""
emulate-bench.py — live load generator + metrics tracker for the AIC emulator.

Usage:
  python3 scripts/emulate-bench.py [--url URL] [--prom-url URL]
                                    [--concurrency N] [--total N] [--duration SECS]
                                    [--min-tokens N] [--max-tokens N]
                                    [--model NAME] [--interval N]

Defaults hit http://localhost:8010 with concurrency=5, 100 total requests.
Use --duration to run for a fixed time instead of a fixed request count
(--total is ignored when --duration is set).
TTFT and TPOT are read from Prometheus (--prom-url, default http://localhost:9292)
which has the real per-request histograms from vLLM's internal instrumentation.
E2E and throughput are measured client-side.
Prints a rolling stats table every --interval seconds and a final summary.
"""

import argparse
import statistics
import threading
import time
import urllib.request
import urllib.error
import urllib.parse
import json
import sys
import random

PROMPTS = [
    "The history of computing begins",
    "Explain how a transformer model works",
    "Once upon a time in a distant galaxy",
    "The quick brown fox jumped over",
    "What are the key differences between CPUs and GPUs",
    "In the beginning there was nothing and then",
    "Write a haiku about machine learning",
    "The AMD Instinct MI300X accelerator is designed for",
    "Summarize the key ideas behind attention mechanisms",
    "Deep learning has transformed the field of",
    "The capital of France is known for",
    "Quantum computing differs from classical computing because",
]

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url",         default="http://localhost:8010", help="vLLM base URL")
    p.add_argument("--prom-url",    default="http://localhost:9292", help="Prometheus base URL")
    p.add_argument("--model",       default="Qwen/Qwen3-8B",         help="Model name")
    p.add_argument("--concurrency", type=int, default=5,             help="Parallel requests in flight")
    p.add_argument("--total",       type=int, default=100,           help="Total requests to send")
    p.add_argument("--min-tokens",  type=int, default=20,            help="Min output tokens")
    p.add_argument("--max-tokens",  type=int, default=120,           help="Max output tokens")
    p.add_argument("--duration",    type=float, default=None,        help="Run for this many seconds instead of --total requests")
    p.add_argument("--interval",    type=float, default=5.0,         help="Stats print interval (seconds)")
    return p.parse_args()


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.e2es       = []
        self.gen_tokens = []
        self.errors = 0
        self.done   = 0
        self.start  = time.perf_counter()

    def record(self, e2e_ms, prompt_tok, gen_tok):
        with self.lock:
            self.e2es.append(e2e_ms)
            self.gen_tokens.append(gen_tok)
            self.done += 1

    def record_error(self):
        with self.lock:
            self.errors += 1
            self.done += 1

    def snapshot(self):
        with self.lock:
            return (
                list(self.e2es), list(self.gen_tokens),
                self.errors, self.done,
                time.perf_counter() - self.start
            )


# ---------------------------------------------------------------------------
# Prometheus helpers
# ---------------------------------------------------------------------------

def prom_query(prom_url, promql):
    """Run an instant query against Prometheus, return the result list."""
    url = prom_url.rstrip("/") + "/api/v1/query?" + urllib.parse.urlencode({"query": promql})
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        return data.get("data", {}).get("result", [])
    except Exception:
        return []


def prom_histogram_quantile(prom_url, metric_base, quantile):
    """
    Compute a quantile (0-1) from a Prometheus histogram using
    histogram_quantile() on the server side — avoids manual bucket math.
    Returns value in milliseconds (metric assumed to be in seconds).
    """
    promql = f"histogram_quantile({quantile}, rate({metric_base}_bucket[2m]))"
    results = prom_query(prom_url, promql)
    vals = [float(r["value"][1]) for r in results
            if r["value"][1] not in ("NaN", "+Inf", "-Inf", "Inf")]
    if not vals:
        return float("nan")
    return max(vals) * 1000  # seconds → ms


def prom_scalar(prom_url, promql):
    """Return the first scalar value from a PromQL instant query."""
    results = prom_query(prom_url, promql)
    for r in results:
        v = r["value"][1]
        if v not in ("NaN", "+Inf", "-Inf"):
            try:
                return float(v)
            except ValueError:
                pass
    return float("nan")


def pct(data, p):
    if not data:
        return float("nan")
    data = sorted(data)
    idx = int(len(data) * p / 100)
    return data[min(idx, len(data) - 1)]


def fmt(v, fmt_spec):
    if v != v:  # NaN check
        return f"{'n/a':>{fmt_spec.lstrip('>')[:2]}}"
    return format(v, fmt_spec)


def print_stats(stats, total, prom_url, prefix="", duration=None):
    e2es, gen_toks, errors, done, elapsed = stats.snapshot()
    n = len(e2es)

    if duration is not None:
        remaining = max(0.0, duration - elapsed)
        progress  = f"{elapsed:.0f}s / {duration:.0f}s ({remaining:.0f}s left)"
    else:
        progress  = f"{done}/{total} done"

    if n == 0:
        print(f"{prefix}  No completions yet ({progress}, {errors} errors)")
        return

    thpt = sum(gen_toks) / elapsed if elapsed > 0 else 0
    rps  = n / elapsed if elapsed > 0 else 0

    # TTFT and TPOT from Prometheus histograms (real vLLM instrumentation)
    ttft_mean = prom_scalar(prom_url,
        "rate(vllm:time_to_first_token_seconds_sum[2m]) / "
        "rate(vllm:time_to_first_token_seconds_count[2m])") * 1000
    ttft_p50 = prom_histogram_quantile(prom_url, "vllm:time_to_first_token_seconds", 0.50)
    ttft_p90 = prom_histogram_quantile(prom_url, "vllm:time_to_first_token_seconds", 0.90)
    ttft_p99 = prom_histogram_quantile(prom_url, "vllm:time_to_first_token_seconds", 0.99)

    tpot_mean = prom_scalar(prom_url,
        "rate(vllm:request_time_per_output_token_seconds_sum[2m]) / "
        "rate(vllm:request_time_per_output_token_seconds_count[2m])") * 1000
    tpot_p50 = prom_histogram_quantile(prom_url, "vllm:request_time_per_output_token_seconds", 0.50)
    tpot_p90 = prom_histogram_quantile(prom_url, "vllm:request_time_per_output_token_seconds", 0.90)
    tpot_p99 = prom_histogram_quantile(prom_url, "vllm:request_time_per_output_token_seconds", 0.99)

    src = "(prometheus)" if ttft_p50 == ttft_p50 else "(prometheus unavailable)"

    print(f"\n{prefix}── [{progress} | {errors} errors | {n} completions] ──")
    print(f"  {'Metric':<10}  {'mean':>8}  {'p50':>8}  {'p90':>8}  {'p99':>8}  {src}")
    print(f"  {'TTFT ms':<10}  {fmt(ttft_mean,'>8.1f')}  {fmt(ttft_p50,'>8.1f')}  {fmt(ttft_p90,'>8.1f')}  {fmt(ttft_p99,'>8.1f')}")
    print(f"  {'TPOT ms':<10}  {fmt(tpot_mean,'>8.2f')}  {fmt(tpot_p50,'>8.2f')}  {fmt(tpot_p90,'>8.2f')}  {fmt(tpot_p99,'>8.2f')}")
    print(f"  {'E2E ms':<10}  {statistics.mean(e2es):>8.0f}  {pct(e2es,50):>8.0f}  {pct(e2es,90):>8.0f}  {pct(e2es,99):>8.0f}  (client-side)")
    print(f"  Throughput : {thpt:.1f} tok/s gen   |   {rps:.2f} req/s")


def do_request(url, model, min_tok, max_tok, stats):
    prompt = random.choice(PROMPTS)
    max_tokens = random.randint(min_tok, max_tok)
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        url + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        e2e_ms = (time.perf_counter() - t0) * 1000
        usage = body.get("usage", {})
        prompt_tok = usage.get("prompt_tokens", 0)
        gen_tok    = usage.get("completion_tokens", 0)
        stats.record(e2e_ms, prompt_tok, gen_tok)
    except Exception as exc:
        stats.record_error()
        print(f"  [error] {exc}", file=sys.stderr)


def main():
    args = parse_args()
    stats = Stats()

    # Probe Prometheus availability once at startup
    test = prom_query(args.prom_url, "up")
    prom_ok = len(test) > 0
    if not prom_ok:
        print(f"WARNING: Prometheus not reachable at {args.prom_url} — "
              f"TTFT/TPOT will show n/a. Pass --prom-url to override.",
              file=sys.stderr)

    duration_mode = args.duration is not None
    stop_sending  = threading.Event()
    semaphore     = threading.Semaphore(args.concurrency)

    def worker():
        while not stop_sending.is_set():
            semaphore.acquire()
            if stop_sending.is_set():
                semaphore.release()
                return
            try:
                do_request(args.url, args.model, args.min_tokens, args.max_tokens, stats)
            finally:
                semaphore.release()
            if not duration_mode:
                return  # fixed-count mode: one request per thread

    stop_printer = threading.Event()
    limit_label  = f"{args.duration:.0f}s" if duration_mode else f"{args.total} requests"

    def printer():
        while not stop_printer.is_set():
            time.sleep(args.interval)
            _, _, _, done, elapsed = stats.snapshot()
            remaining = max(0, args.duration - elapsed) if duration_mode else args.total - done
            print_stats(stats, args.total if not duration_mode else None,
                        args.prom_url, duration=args.duration)
    t = threading.Thread(target=printer, daemon=True)
    t.start()

    print(f"Running for {limit_label} against {args.url}  "
          f"concurrency={args.concurrency}  "
          f"tokens=[{args.min_tokens},{args.max_tokens}]")
    print(f"Model: {args.model}  |  Prometheus: {args.prom_url} ({'ok' if prom_ok else 'unavailable'})")
    print(f"TTFT/TPOT source: {'Prometheus histograms' if prom_ok else 'n/a (no Prometheus)'}  |  "
          f"E2E: client-side wall clock")
    print(f"Stats every {args.interval:.0f}s  (Ctrl-C to stop early)\n")

    if duration_mode:
        # Spin up concurrency workers that loop until time is up
        threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
        for th in threads:
            th.start()
        time.sleep(args.duration)
        stop_sending.set()
        # Drain the semaphore so blocked workers can exit
        for _ in range(args.concurrency):
            semaphore.release()
    else:
        # Fixed-count: one thread per request, limited by semaphore
        threads = [threading.Thread(target=worker) for _ in range(args.total)]
        for th in threads:
            th.start()

    for th in threads:
        th.join()

    stop_printer.set()

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print_stats(stats, args.total if not duration_mode else None,
                args.prom_url, duration=args.duration)
    print()


if __name__ == "__main__":
    main()
