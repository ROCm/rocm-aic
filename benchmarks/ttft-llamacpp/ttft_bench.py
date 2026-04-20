#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""TTFT benchmark for llama-server with slot save/restore.

Subcommands:
    run     Execute the full benchmark sweep (cold / warm-tmpfs / warm-disk)
    report  Generate summary table, CSV, and charts from results
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

LOG_PREFIX = "[ttft_bench]"


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"{LOG_PREFIX} {ts} {msg}", flush=True)


# ── GPU detection ───────────────────────────────────────────────

def detect_gpu() -> str:
    """Detect GPU name via rocm_agent_enumerator and rocm-smi."""
    for cmd in [["rocm-smi", "--showproductname"], ["rocminfo"]]:
        try:
            out = subprocess.check_output(
                cmd, stderr=subprocess.DEVNULL, text=True, timeout=10)
            for line in out.splitlines():
                if any(k in line for k in ("Instinct", "Radeon", "gfx")):
                    name = line.split(":")[-1].strip() if ":" in line else line.strip()
                    if name:
                        return name
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    return "unknown"


# ── Disk IO ─────────────────────────────────────────────────────

def _get_fs_type(path: str) -> str:
    """Return the filesystem type for a given path (e.g., 'tmpfs', 'ext4')."""
    try:
        out = subprocess.check_output(
            ["df", "-T", path], text=True, stderr=subprocess.DEVNULL)
        return out.splitlines()[1].split()[1]
    except (subprocess.SubprocessError, IndexError):
        return ""


def _find_block_device(path: str) -> str | None:
    """Find the block device name for a given path.

    Returns None for virtual filesystems (tmpfs, devtmpfs).
    """
    fs_type = _get_fs_type(path)
    if fs_type in ("tmpfs", "devtmpfs", "ramfs"):
        return None
    try:
        out = subprocess.check_output(
            ["df", path], text=True, stderr=subprocess.DEVNULL)
        return out.splitlines()[1].split()[0]
    except (subprocess.SubprocessError, IndexError):
        return None


def read_proc_io(pid: int) -> tuple[int, int]:
    """Read (rchar, wchar) from /proc/{pid}/io.

    Uses rchar/wchar (total bytes read/written including page
    cache) rather than read_bytes/write_bytes (block device only).
    This captures slot file reads even when served from page cache,
    giving a meaningful comparison between tmpfs and disk tiers.
    """
    try:
        with open(f"/proc/{pid}/io") as fh:
            vals = {}
            for line in fh:
                key, val = line.strip().split(": ")
                vals[key] = int(val)
            return vals.get("rchar", 0), vals.get("wchar", 0)
    except (OSError, ValueError):
        return 0, 0


def bytes_to_mib(b: int) -> float:
    return round(b / (1024 * 1024), 2)


# ── Page cache ──────────────────────────────────────────────────

def drop_page_cache(target_dir: str | None = None) -> None:
    if shutil.which("vmtouch") and target_dir:
        r = subprocess.run(["vmtouch", "-e", target_dir],
                           capture_output=True)
        if r.returncode == 0:
            log(f"  page cache evicted (vmtouch) for {target_dir}")
            return
    r = subprocess.run(
        ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
        capture_output=True)
    if r.returncode == 0:
        log("  page cache dropped (global)")
    else:
        log("  WARNING: cannot drop page cache; "
            "disk results may reflect cached IO")


# ── Slot API ────────────────────────────────────────────────────

def slot_action(server_url: str, slot_id: int,
                action: str, filename: str = "") -> dict:
    url = f"{server_url}/slots/{slot_id}?action={action}"
    body = json.dumps({"filename": filename}).encode() if filename else b"{}"
    req = Request(url, data=body,
                  headers={"Content-Type": "application/json"},
                  method="POST")
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


# ── Corpus ──────────────────────────────────────────────────────

CORPUS_URLS = [
    "https://www.gutenberg.org/cache/epub/1184/pg1184.txt",
    "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
]


def ensure_corpus(path: Path) -> None:
    if path.exists():
        return
    log(f"downloading corpus to {path} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    with path.open("wb") as out:
        for url in CORPUS_URLS:
            log(f"  {url}")
            with urllib.request.urlopen(url) as resp:
                out.write(resp.read())
    log(f"corpus ready ({path.stat().st_size / 1024 / 1024:.1f} MiB)")


def build_prompt(corpus_path: Path, context_chars: int,
                 seed: int) -> str:
    raw = corpus_path.read_text(encoding="utf-8", errors="replace")
    rng = random.Random(seed)
    max_offset = max(0, len(raw) - context_chars)
    offset = rng.randint(0, max_offset) if max_offset > 0 else 0
    excerpt = raw[offset:offset + context_chars]
    return f"{excerpt}\n\nSummarize the above text in two sentences."


# ── Server lifecycle ────────────────────────────────────────────

class LlamaServer:
    def __init__(self, model: str, ctx_size: int = 16384,
                 n_gpu_layers: int = 99, slot_save_path: str = "/tmp/slots",
                 host: str = "0.0.0.0", port: int = 8080,
                 startup_timeout: int = 120):
        self.model = model
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.slot_save_path = slot_save_path
        self.host = host
        self.port = port
        self.url = f"http://localhost:{port}"
        self.startup_timeout = startup_timeout
        self.proc: subprocess.Popen | None = None
        self.startup_health_ms = 0
        self.startup_warmup_ms = 0
        self.startup_total_ms = 0

    def start(self) -> None:
        os.makedirs(self.slot_save_path, exist_ok=True)

        env = os.environ.copy()
        if not env.get("HIP_VISIBLE_DEVICES") and \
           not env.get("ROCR_VISIBLE_DEVICES"):
            env["HIP_VISIBLE_DEVICES"] = "0"

        cmd = [
            "llama-server",
            "--model", self.model,
            "--ctx-size", str(self.ctx_size),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--slot-save-path", self.slot_save_path,
            "--host", self.host,
            "--port", str(self.port),
        ]

        log(f"starting llama-server (port {self.port}) ...")
        t_start = time.monotonic()

        self.proc = subprocess.Popen(cmd, env=env)

        elapsed = 0
        while True:
            try:
                urlopen(f"{self.url}/health", timeout=2)
                break
            except Exception:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited unexpectedly")
            if elapsed >= self.startup_timeout:
                self.stop()
                raise RuntimeError(
                    f"server not ready within {self.startup_timeout}s")
            time.sleep(2)
            elapsed += 2

        t_healthy = time.monotonic()
        self.startup_health_ms = int((t_healthy - t_start) * 1000)

        try:
            req = Request(
                f"{self.url}/v1/chat/completions",
                data=json.dumps({
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1, "stream": False,
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST")
            urlopen(req, timeout=60)
        except Exception:
            pass

        t_warm = time.monotonic()
        self.startup_warmup_ms = int((t_warm - t_healthy) * 1000)
        self.startup_total_ms = int((t_warm - t_start) * 1000)

        log(f"llama-server ready (PID {self.proc.pid})")
        log(f"  startup: health={self.startup_health_ms}ms  "
            f"warmup={self.startup_warmup_ms}ms  "
            f"total={self.startup_total_ms}ms")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            log(f"stopping llama-server (PID {self.proc.pid}) ...")
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None

        tries = 0
        while tries < 15:
            try:
                urlopen(f"{self.url}/health", timeout=1)
                time.sleep(1)
                tries += 1
            except Exception:
                break
        log(f"llama-server stopped (port {self.port} free)")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


# ── TTFT measurement ───────────────────────────────────────────

def measure_ttft(server_url: str, model: str, prompt: str,
                 slot_id: int = 0) -> tuple[float, str]:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "dummy-key"),
        base_url=f"{server_url}/v1")

    start = time.perf_counter()
    ttft: float | None = None
    fragments: list[str] = []

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7, max_tokens=64, stream=True,
        extra_body={"id_slot": slot_id})

    for chunk in stream:
        choice = chunk.choices[0]
        text = choice.delta.content
        if text is None:
            text = getattr(choice.delta, "reasoning_content", None)
            if text is None:
                text = getattr(choice.delta, "reasoning", None)
        if text is not None:
            if ttft is None:
                ttft = time.perf_counter() - start
            fragments.append(text)

    if ttft is None:
        raise RuntimeError("Server returned no content tokens")
    return ttft, "".join(fragments)


# ── Single measurement ─────────────────────────────────────────

def run_single(server: LlamaServer, tag: str, prompt: str,
               model_name: str, gpu: str, seed: int,
               slot_id: int = 0,
               save_slot: str | None = None,
               restore_slot: str | None = None) -> dict:

    try:
        slot_action(server.url, slot_id, "erase")
    except Exception:
        pass

    if restore_slot:
        log(f"  restoring slot {slot_id} from {restore_slot}")
        slot_action(server.url, slot_id, "restore", restore_slot)

    srv_pid = server.proc.pid if server.proc else 0
    before_r, before_w = read_proc_io(srv_pid)
    ttft_s, reply = measure_ttft(server.url, model_name, prompt, slot_id)
    after_r, after_w = read_proc_io(srv_pid)

    ttft_ms = ttft_s * 1000.0
    disk_read_mib = bytes_to_mib(after_r - before_r)
    disk_write_mib = bytes_to_mib(after_w - before_w)

    log(f"  TTFT = {ttft_ms:.1f} ms  "
        f"IO: rd={disk_read_mib:.1f} MiB  wr={disk_write_mib:.1f} MiB")

    if save_slot:
        log(f"  saving slot {slot_id} to {save_slot}")
        slot_action(server.url, slot_id, "save", save_slot)

    return {
        "tag": tag,
        "model": model_name,
        "gpu": gpu,
        "context_chars": len(prompt),
        "seed": seed,
        "ttft_ms": round(ttft_ms, 2),
        "disk_read_mib": disk_read_mib,
        "disk_write_mib": disk_write_mib,
        "slot_path": server.slot_save_path,
        "slot_fs_type": _get_fs_type(server.slot_save_path),
        "slot_device": _find_block_device(server.slot_save_path) or "none",
        "startup_health_ms": server.startup_health_ms,
        "startup_warmup_ms": server.startup_warmup_ms,
        "startup_total_ms": server.startup_total_ms,
        "restored_from": restore_slot,
        "saved_to": save_slot,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Run subcommand ──────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> None:
    corpus_path = Path(args.corpus_file)
    ensure_corpus(corpus_path)

    gpu = args.gpu or detect_gpu()
    model_name = Path(args.model).name
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    slot_save_orig = args.slot_save_path
    slot_tmpfs = args.slot_tmpfs_path
    slot_disk = args.slot_disk_path

    log("=" * 50)
    log("  TTFT llama-server Benchmark")
    log("=" * 50)
    log(f"  MODEL          = {args.model}")
    log(f"  GPU            = {gpu}")
    log(f"  CONTEXT_CHARS  = {args.context_chars}")
    log(f"  REPEATS        = {args.repeats}")
    log(f"  SEED           = {args.seed}")
    log(f"  OUTPUT         = {output}")
    log("=" * 50)

    def append_record(rec: dict) -> None:
        with output.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    for cchars in args.context_chars:
        slot_file = f"cache-{cchars}.bin"
        prompt = build_prompt(corpus_path, cchars, args.seed)

        log(f"{'=' * 50}")
        log(f"  Context size: ~{cchars} chars ({len(prompt)} actual)")
        log(f"{'=' * 50}")

        # ── cold ────────────────────────────────────────
        log(f"--- Cold Runs ({cchars} chars) ---")
        for rep in range(1, args.repeats + 1):
            log(f"cold run {rep}/{args.repeats}")

            for d in [slot_save_orig, slot_tmpfs, slot_disk]:
                p = Path(d)
                if p.exists():
                    for f in p.glob("*.bin"):
                        f.unlink()

            server = LlamaServer(
                model=args.model, ctx_size=args.ctx_size,
                n_gpu_layers=args.n_gpu_layers,
                slot_save_path=slot_save_orig, port=args.port)
            with server:
                rec = run_single(
                    server, f"cold-{cchars}c-{rep}", prompt,
                    model_name, gpu, args.seed,
                    save_slot=slot_file)
                append_record(rec)

        src_slot = Path(slot_save_orig) / slot_file

        # ── warm-tmpfs ──────────────────────────────────
        log(f"--- Warm tmpfs ({cchars} chars) ---")

        # create a private tmpfs mount if possible
        if not os.path.ismount(slot_tmpfs):
            os.makedirs(slot_tmpfs, exist_ok=True)
            r = subprocess.run(
                ["sudo", "-n", "mount", "-t", "tmpfs",
                 "-o", "size=4G", "tmpfs", slot_tmpfs],
                capture_output=True)
            if r.returncode == 0:
                log(f"  mounted private tmpfs at {slot_tmpfs}")
            else:
                log(f"  WARNING: could not mount tmpfs at {slot_tmpfs}, "
                    f"using {slot_tmpfs} as-is")

        shutil.copy2(str(src_slot), f"{slot_tmpfs}/{slot_file}")

        tmpfs_dev = _find_block_device(slot_tmpfs)
        log(f"  tmpfs path: {slot_tmpfs}  (dev: {tmpfs_dev or 'none'})")

        server = LlamaServer(
            model=args.model, ctx_size=args.ctx_size,
            n_gpu_layers=args.n_gpu_layers,
            slot_save_path=slot_tmpfs, port=args.port)
        with server:
            for rep in range(1, args.repeats + 1):
                log(f"tmpfs warm run {rep}/{args.repeats}")
                try:
                    slot_action(server.url, 0, "erase")
                except Exception:
                    pass
                rec = run_single(
                    server, f"warm-tmpfs-{cchars}c-{rep}", prompt,
                    model_name, gpu, args.seed,
                    restore_slot=slot_file)
                append_record(rec)

        # ── warm-disk ───────────────────────────────────
        log(f"--- Warm disk ({cchars} chars) ---")
        os.makedirs(slot_disk, exist_ok=True)
        shutil.copy2(str(src_slot), f"{slot_disk}/{slot_file}")

        disk_dev = _find_block_device(slot_disk)
        log(f"  disk path: {slot_disk}  (dev: {disk_dev or 'none'})")

        server = LlamaServer(
            model=args.model, ctx_size=args.ctx_size,
            n_gpu_layers=args.n_gpu_layers,
            slot_save_path=slot_disk, port=args.port)
        with server:
            for rep in range(1, args.repeats + 1):
                log(f"disk warm run {rep}/{args.repeats}")
                drop_page_cache(slot_disk)
                try:
                    slot_action(server.url, 0, "erase")
                except Exception:
                    pass
                rec = run_single(
                    server, f"warm-disk-{cchars}c-{rep}", prompt,
                    model_name, gpu, args.seed,
                    restore_slot=slot_file)
                append_record(rec)

    log("--- Benchmark complete ---")
    log(f"Results: {output}")

    cmd_report_from_path(output, gpu)


# ── Report subcommand ───────────────────────────────────────────

def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def extract_ctx(tag: str) -> str | None:
    m = re.search(r"(\d+)c-", tag)
    return m.group(1) if m else None


def classify_phase(tag: str) -> str:
    if "tmpfs" in tag:
        return "warm-tmpfs"
    if "disk" in tag:
        return "warm-disk"
    if tag.startswith("cold"):
        return "cold"
    return "other"


def build_summary(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        ctx = extract_ctx(r["tag"])
        phase = classify_phase(r["tag"])
        if ctx:
            groups[(ctx, phase)].append(r)

    sizes = sorted(set(k[0] for k in groups), key=int)
    phases = ["cold", "warm-tmpfs", "warm-disk"]
    rows = []
    for sz in sizes:
        cold_recs = groups.get((sz, "cold"), [])
        cold_mean = mean([r["ttft_ms"] for r in cold_recs])
        for phase in phases:
            recs = groups.get((sz, phase), [])
            if not recs:
                continue
            ttfts = [r["ttft_ms"] for r in recs]
            m = mean(ttfts)
            speedup = cold_mean / m if m > 0 and phase != "cold" else 0
            rows.append({
                "ctx_chars": int(sz),
                "phase": phase,
                "n": len(ttfts),
                "mean_ms": round(m, 1),
                "min_ms": round(min(ttfts), 1),
                "max_ms": round(max(ttfts), 1),
                "speedup": round(speedup, 1) if speedup else None,
                "disk_read_mib": round(
                    mean([r.get("disk_read_mib", 0) for r in recs]), 1),
                "disk_write_mib": round(
                    mean([r.get("disk_write_mib", 0) for r in recs]), 1),
                "load_ms": round(
                    mean([r.get("startup_total_ms", 0) for r in recs])),
            })
    return rows


def print_table(rows: list[dict], gpu: str, model: str,
                records: list[dict] | None = None) -> None:
    print()
    print(f"GPU:   {gpu}")
    print(f"Model: {model}")
    if records:
        paths_seen: dict[str, tuple[str, str]] = {}
        for r in records:
            sp = r.get("slot_path", "")
            fs = r.get("slot_fs_type", "")
            dd = r.get("slot_device", "none")
            if sp and sp not in paths_seen:
                paths_seen[sp] = (fs, dd)
        if paths_seen:
            print("Storage:")
            for sp, (fs, dd) in paths_seen.items():
                fs_info = f"  ({fs})" if fs else ""
                dev_info = f"  (dev: {dd})"
                print(f"  {sp}{fs_info}{dev_info}")
    print()
    hdr = (f"{'ctx_chars':<10} {'phase':<14} {'n':>3} {'mean_ms':>10}"
           f" {'min_ms':>10} {'max_ms':>10} {'speedup':>8}"
           f" {'rd_MiB':>8} {'wr_MiB':>8} {'load_ms':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        sp = f"{r['speedup']:.1f}x" if r["speedup"] else ""
        print(f"{r['ctx_chars']:<10} {r['phase']:<14} {r['n']:>3}"
              f" {r['mean_ms']:>10.1f} {r['min_ms']:>10.1f}"
              f" {r['max_ms']:>10.1f} {sp:>8}"
              f" {r['disk_read_mib']:>8.1f} {r['disk_write_mib']:>8.1f}"
              f" {r['load_ms']:>9}")
    print()


def save_csv(rows: list[dict], path: Path,
             gpu: str, model: str) -> None:
    cols = ["ctx_chars", "phase", "n", "mean_ms", "min_ms", "max_ms",
            "speedup", "disk_read_mib", "disk_write_mib", "load_ms"]
    with path.open("w") as fh:
        fh.write(f"# gpu: {gpu}\n# model: {model}\n")
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    log(f"CSV saved: {path}")


def plot_ttft_bars(rows: list[dict], out: Path,
                   gpu: str, model: str) -> None:
    if not HAS_MPL:
        return
    sizes = sorted(set(r["ctx_chars"] for r in rows))
    phases = ["cold", "warm-tmpfs", "warm-disk"]
    colors = {"cold": "#e74c3c", "warm-tmpfs": "#2ecc71",
              "warm-disk": "#3498db"}
    labels = {"cold": "Cold (no cache)", "warm-tmpfs": "Warm (tmpfs/RAM)",
              "warm-disk": "Warm (disk)"}

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = range(len(sizes))
    width = 0.25
    for i, phase in enumerate(phases):
        vals = [next((r["mean_ms"] for r in rows
                      if r["ctx_chars"] == sz and r["phase"] == phase), 0)
                for sz in sizes]
        offset = (i - 1) * width
        bars = ax.bar([x + offset for x in x_pos], vals, width,
                      label=labels[phase], color=colors[phase])
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height(), f"{val:.0f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Context Size (chars)")
    ax.set_ylabel("Mean TTFT (ms)")
    ax.set_title(f"TTFT by Context Size and Cache Phase\n{gpu} | {model}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_yscale("log")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"TTFT bar chart saved: {out}")


def plot_speedup(rows: list[dict], out: Path,
                 gpu: str, model: str) -> None:
    if not HAS_MPL:
        return
    sizes = sorted(set(r["ctx_chars"] for r in rows))
    phases = ["warm-tmpfs", "warm-disk"]
    colors = {"warm-tmpfs": "#2ecc71", "warm-disk": "#3498db"}
    labels = {"warm-tmpfs": "tmpfs (RAM)", "warm-disk": "disk"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for phase in phases:
        vals = [next((r["speedup"] for r in rows
                      if r["ctx_chars"] == sz and r["phase"] == phase
                      and r["speedup"]), 0)
                for sz in sizes]
        ax.plot(sizes, vals, "o-", label=labels[phase],
                color=colors[phase], linewidth=2, markersize=8)
        for sz, val in zip(sizes, vals):
            if val > 0:
                ax.annotate(f"{val:.0f}x", (sz, val),
                            textcoords="offset points", xytext=(0, 10),
                            ha="center", fontsize=9)
    ax.set_xlabel("Context Size (chars)")
    ax.set_ylabel("Speedup vs Cold (x)")
    ax.set_title(f"Slot Restore Speedup vs Cold Prefill\n{gpu} | {model}")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"Speedup chart saved: {out}")


def plot_disk_io(rows: list[dict], out: Path,
                 gpu: str, model: str) -> None:
    if not HAS_MPL:
        return
    sizes = sorted(set(r["ctx_chars"] for r in rows))
    phases = ["cold", "warm-tmpfs", "warm-disk"]
    colors = {"cold": "#e74c3c", "warm-tmpfs": "#2ecc71",
              "warm-disk": "#3498db"}
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = range(len(sizes))
    width = 0.25
    for i, phase in enumerate(phases):
        vals = [next((r["disk_read_mib"] for r in rows
                      if r["ctx_chars"] == sz and r["phase"] == phase), 0)
                for sz in sizes]
        offset = (i - 1) * width
        ax.bar([x + offset for x in x_pos], vals, width,
               label=phase, color=colors[phase])
    ax.set_xlabel("Context Size (chars)")
    ax.set_ylabel("Disk Read (MiB)")
    ax.set_title(f"Disk Read IO by Phase\n{gpu} | {model}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"Disk IO chart saved: {out}")


def cmd_report_from_path(results_path: Path, gpu: str | None = None,
                         output_dir: Path | None = None) -> None:
    records = []
    with results_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        log("ERROR: no records found")
        return

    gpu = gpu or records[0].get("gpu") or detect_gpu()
    model = records[0].get("model", "unknown")
    out_dir = output_dir or results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_summary(records)
    print_table(rows, gpu, model, records)

    save_csv(rows, out_dir / "summary.csv", gpu, model)

    if HAS_MPL:
        plot_ttft_bars(rows, out_dir / "ttft_bars.png", gpu, model)
        plot_speedup(rows, out_dir / "speedup.png", gpu, model)
        plot_disk_io(rows, out_dir / "disk_io.png", gpu, model)
    else:
        log("WARNING: matplotlib not installed, skipping graphs")

    meta = {
        "gpu": gpu, "model": model,
        "n_records": len(records),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_file": str(results_path),
    }
    with (out_dir / "report_meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2)
    log(f"Metadata saved: {out_dir / 'report_meta.json'}")


def cmd_report(args: argparse.Namespace) -> None:
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(args.output_dir) if args.output_dir else None
    cmd_report_from_path(results_path, args.gpu, out_dir)


# ── CLI ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TTFT benchmark for llama-server")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── run ──
    p_run = sub.add_parser("run", help="Execute the benchmark sweep")
    p_run.add_argument("--model", required=True,
                       help="Path to GGUF model file")
    p_run.add_argument("--context-chars", type=int, nargs="+",
                       default=[400, 4000, 40000],
                       help="Context sizes in chars (default: 400 4000 40000)")
    p_run.add_argument("--repeats", type=int, default=10)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--ctx-size", type=int, default=16384)
    p_run.add_argument("--n-gpu-layers", type=int, default=99)
    p_run.add_argument("--port", type=int, default=8080)
    p_run.add_argument("--slot-save-path", default="/tmp/slots")
    p_run.add_argument("--slot-tmpfs-path",
                       default="/tmp/ttft-slots-tmpfs")
    p_run.add_argument("--slot-disk-path", default="/tmp/ttft-slots-disk")
    p_run.add_argument("--corpus-file",
                       default="corpus.txt")
    p_run.add_argument("--output", default="results.jsonl")
    p_run.add_argument("--gpu", default=None,
                       help="GPU name (auto-detected if omitted)")

    # ── report ──
    p_rep = sub.add_parser("report", help="Generate report from results")
    p_rep.add_argument("results", help="Path to results.jsonl")
    p_rep.add_argument("--output-dir", default=None)
    p_rep.add_argument("--gpu", default=None)

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
