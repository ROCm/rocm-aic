# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""KV Cache Cliff results → GitHub Pages HTML.

Reads one or more cliff run directories (each containing a results/ subdir with
CSVs and a plots/ subdir with PNGs produced by plot_cliff.py) and generates a
self-contained HTML performance-tracking page.

Each run directory is expected to follow the layout created by run-cliff.sbatch:

    logs/<job-id>/
        run-config.json                    (what each arm ran with; optional)
        results/
            cliff-vram_only-<stamp>.csv    (+ .json: run_cliff.py's arguments)
            cliff-kvd_v2-nvme-<stamp>.csv  (optional)
            cliff-kvd_v2-gds-<stamp>.csv   (optional)
        plots/
            nvme/
                cliff-throughput.png
                cliff-latency-p50.png
                cliff-latency-p95.png
                cliff-hit-rate.png
            gds/   (same layout, optional)
            vram/  (fallback when no kvd arms ran)

Usage:
    # Single run:
    python benchmarks/cliff_to_html.py \\
        --run-dir logs/12345 \\
        --sha abc1234 --run-date "2026-07-27" \\
        --output-dir cliff-page/

    # Multiple runs merged into one page:
    python benchmarks/cliff_to_html.py \\
        --run-dir logs/12345 logs/12346 \\
        --sha abc1234 --run-date "2026-07-27" \\
        --output-dir cliff-page/

    # Pull run metadata from a JSON manifest (written by spur-cliff-harvest.sh):
    python benchmarks/cliff_to_html.py \\
        --manifest cliff-manifest.json \\
        --output-dir cliff-page/
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _float_or_none(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _median_or_none(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def _summarise_arm(rows: list[dict]) -> dict[int, dict]:
    """Return {concurrency: medians of thr, p50, p95, l1_hit and ext_hit}.

    l1_hit is vLLM's GPU prefix cache, ext_hit LMCache's; either is None when the
    run did not scrape it (ext_hit always, for an arm without LMCache)."""
    by_c: dict[int, list] = {}
    for row in rows:
        try:
            c = int(row["concurrency"])
            thr = float(row["throughput_tok_s_total"])
            p50 = float(row["p50_latency_s"]) * 1000
            p95 = float(row["p95_latency_s"]) * 1000
        except (KeyError, ValueError):
            continue
        by_c.setdefault(c, []).append((
            thr, p50, p95,
            _float_or_none(row.get("l1_hit_rate_pct")),
            _float_or_none(row.get("ext_hit_rate_pct")),
        ))
    result: dict[int, dict] = {}
    for c, vals in sorted(by_c.items()):
        result[c] = {
            "thr": statistics.median(v[0] for v in vals),
            "p50": statistics.median(v[1] for v in vals),
            "p95": statistics.median(v[2] for v in vals),
            "l1_hit": _median_or_none([v[3] for v in vals]),
            "ext_hit": _median_or_none([v[4] for v in vals]),
        }
    return result


# Keyed by the arm part of the CSV name (cliff-<key>-<stamp>.csv), not the CSV's
# arm column: both kvd backends write arm=kvd_v2, so only the name separates them.
_ARM_LABELS = {
    "vram_only":   "VRAM only (no AIC)",
    "vram_dram":   "VRAM + DRAM",
    "kvd_v2-nvme": "AIC NVMe (DRAM L1 + NVMe L2 via LMCache)",
    "kvd_v2-gds":  "AIC GDS (hipFile NVMe slab via LMCache)",
    "kvd_v2":      "AIC (NVMe/NFS via LMCache)",
    "kvbench":     "KVBench (no vLLM)",
}
_ARM_ORDER = tuple(_ARM_LABELS)

_ARM_COLORS = {
    "vram_only":   "#c0392b",
    "vram_dram":   "#e67e22",
    "kvd_v2-nvme": "#2980b9",
    "kvd_v2-gds":  "#8e44ad",
    "kvd_v2":      "#2980b9",
    "kvbench":     "#16a085",
}

# plots/<backend>/ subdirs run-cliff.sbatch writes; vram only when no kvd arm ran.
_CHART_BACKENDS = ("nvme", "gds", "vram")
_STAMP_RE = re.compile(r"^cliff-(?P<key>.+)-\d{8}-\d{6}$")


def _peak_thr(summary: dict[int, dict]) -> float:
    return max((v["thr"] for v in summary.values()), default=0.0)


def _cliff_concurrency(summary: dict[int, dict]) -> int | None:
    """Return concurrency where throughput first drops >10% from its peak."""
    peak = 0.0
    for c in sorted(summary):
        y = summary[c]["thr"]
        if y > peak:
            peak = y
        elif peak > 0 and (peak - y) / peak > 0.10:
            return c
    return None


# ---------------------------------------------------------------------------
# PNG → base64 data URI
# ---------------------------------------------------------------------------

def _img_src(path: Path) -> str | None:
    if not path.is_file():
        return None
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError as exc:
        print(f"WARN: ignoring unreadable {path}: {exc}", file=sys.stderr)
        return None


def _discover_run(run_dir: Path) -> dict[str, Any]:
    """Scan a single run directory and return its data dict."""
    results_dir = run_dir / "results"
    plots_dir = run_dir / "plots"

    arms: dict[str, dict[int, dict]] = {}
    # run_cliff.py's arguments for each arm, from the JSON beside its CSV.
    bench: dict[str, dict] = {}
    for csv_path in sorted(results_dir.glob("cliff-*.csv")) if results_dir.is_dir() else []:
        rows = _load_csv(csv_path)
        m = _STAMP_RE.match(csv_path.stem)
        # A name without the run stamp falls back to the CSV's own arm column.
        key = m.group("key") if m else next(
            (r.get("arm", "").strip() for r in rows if r.get("arm", "").strip()), "")
        if key and key not in arms:
            arms[key] = _summarise_arm(rows)
            args = _load_json(csv_path.with_suffix(".json"))
            if isinstance(args, dict):
                bench[key] = args

    # One chart set per backend run-cliff.sbatch plotted.
    charts: dict[str, dict[str, str | None]] = {}
    for sub in _CHART_BACKENDS:
        sub_dir = plots_dir / sub
        if sub_dir.is_dir():
            charts[sub] = {
                "throughput": _img_src(sub_dir / "cliff-throughput.png"),
                "p50": _img_src(sub_dir / "cliff-latency-p50.png"),
                "p95": _img_src(sub_dir / "cliff-latency-p95.png"),
                "hit": _img_src(sub_dir / "cliff-hit-rate.png"),
            }

    # What each arm ran with, as run-cliff.sbatch read it back from the containers.
    config = _load_json(run_dir / "run-config.json")
    if not isinstance(config, dict):
        config = {}

    return {"arms": arms, "charts": charts, "bench": bench, "config": config}


# ---------------------------------------------------------------------------
# Run configuration (run-config.json + run_cliff.py's arguments)
# ---------------------------------------------------------------------------
# Every value shown comes from the run's own record -- the argv vLLM and the
# LMCache server were started with, the GPU vLLM saw, the arguments the sweep
# ran with -- so the page describes what ran rather than restating defaults.

_GIB = 1 << 30
_MIB = 1 << 20

# Components of the image tag, <aic-version>-<name><version>-...
# (docker/scripts/aic-image-tag.sh), in display form.
_TAG_COMPONENTS = {
    "rocm": "ROCm",
    "pytorch": "PyTorch",
    "vllm": "vLLM",
    "aiter": "AITER",
    "fa": "FlashAttention",
    "lmcache": "LMCache",
    "nixl": "NIXL",
}
_TAG_PART_RE = re.compile(r"^(?P<name>[a-z]+)(?P<version>\d[\w.+]*)$")


def _flag_values(argv: list[str], name: str) -> list[str]:
    """Every value given for --name in argv, as `--name v` or `--name=v`."""
    values = []
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            values.append(argv[i + 1])
        elif arg.startswith(name + "="):
            values.append(arg.split("=", 1)[1])
    return values


def _flag(argv: list[str], name: str) -> str | None:
    """The effective value of --name (the last one given), else None."""
    values = _flag_values(argv, name)
    return values[-1] if values else None


def _fmt_gib(nbytes: float) -> str:
    s = f"{nbytes / _GIB:,.1f}"
    return f"{s[:-2] if s.endswith('.0') else s} GiB"


def _vllm_argv(arm_cfg: dict) -> list[str]:
    return list((arm_cfg.get("vllm") or {}).get("args") or [])


def _stack_versions(image_ref: str) -> str | None:
    """'ROCm 7.14.1 · vLLM 0.29.0 · ...' from an AIC image tag, else None."""
    parts = image_ref.rpartition(":")[2].split("-")
    out = []
    for part in parts[1:]:
        m = _TAG_PART_RE.match(part)
        if not m:
            return None
        if m["name"] in _TAG_COMPONENTS:
            out.append(f"{_TAG_COMPONENTS[m['name']]} {m['version']}")
    return " · ".join(out) or None


def _storage_desc(storage: dict, path: str) -> str | None:
    """The host storage behind container path `path`, e.g. '8× local NVMe (xfs)'."""
    mounts = [d for d in storage if path == d or path.startswith(d.rstrip("/") + "/")]
    if not mounts:
        return None
    info = storage[max(mounts, key=len)] or {}
    fstype = info.get("fstype") or "unknown fs"
    try:
        nvme = int(info.get("nvme_disks") or 0)
    except ValueError:
        nvme = 0
    if nvme > 1:
        return f"{nvme}× local NVMe ({fstype})"
    if nvme == 1:
        return f"local NVMe ({fstype})"
    return f"{info.get('source') or 'unknown device'} ({fstype})"


def _on(where: str | None) -> str:
    return f", on {where}" if where else ""


def _lmcache_tiers(lmcache: dict) -> tuple[str, str]:
    """(L1, L2) descriptions from the LMCache server's argv and config file."""
    process = lmcache.get("process") or {}
    argv = list(process.get("argv") or [])
    storage = lmcache.get("storage") or {}
    if not argv:
        return "Not recorded", "Not recorded"

    size = _flag(argv, "--l1-size-gb")
    size_str = f"{size} GiB" if size else "size not recorded"
    gds_path = _flag(argv, "--gds-l1-path")
    if gds_path:
        l1 = f"hipFile slab, {size_str}{_on(_storage_desc(storage, gds_path))}"
    else:
        l1 = f"Pinned host DRAM, {size_str}"

    l2 = []
    for raw in _flag_values(argv, "--l2-adapter"):
        try:
            adapter = json.loads(raw)
        except ValueError:
            l2.append(raw)
            continue
        params = adapter.get("backend_params") or {}
        io = ("O_DIRECT" if str(params.get("use_direct_io")).lower() == "true"
              else "buffered I/O")
        where = _on(_storage_desc(storage, params.get("file_path", "")))
        backend = adapter.get("backend", "?")
        if adapter.get("type") == "nixl_store":
            pool = int(adapter.get("pool_size") or 0)
            file_size = params.get("file_size")
            if file_size:
                fs = int(file_size)
                cap = f"{pool:,} × {fs / _MIB:g} MiB files = {_fmt_gib(pool * fs)}"
            else:
                cap = f"{pool:,} files"
            l2.append(f"NIXL {backend}: {cap}, {io}{where}")
        elif adapter.get("type") == "nixl_store_dynamic":
            l2.append(f"NIXL {backend}, one file per chunk, up to "
                      f"{params.get('max_capacity_gb', '?')} GB, {io}{where}")
        else:
            l2.append(raw)

    # LMCache's own local-disk tier comes from its config file, not the argv.
    text = process.get("config_file") or ""
    disk = re.search(r'^local_disk:\s*"?file://([^"\s]+)', text, re.M)
    if disk:
        cap = re.search(r"^max_local_disk_size:\s*(\S+)", text, re.M)
        odirect = re.search(r"^\s*use_odirect:\s*(\S+)", text, re.M)
        io = "O_DIRECT" if odirect and odirect[1].lower() == "true" else "buffered I/O"
        l2.append(f"LMCache local disk, up to {cap[1] if cap else '?'} GB, {io}"
                  f"{_on(_storage_desc(storage, disk[1]))}")

    return l1, "; ".join(l2) or "None"


def _merged(values: dict[str, str | None]) -> str | None:
    """One value when every arm agrees, else each arm's own."""
    present = {arm: v for arm, v in values.items() if v}
    if not present:
        return None
    if len(set(present.values())) == 1:
        return next(iter(present.values()))
    return "; ".join(f"{_ARM_LABELS.get(arm, arm)}: {v}" for arm, v in present.items())


def _ordered(keys) -> list[str]:
    return sorted(keys, key=lambda k: (_ARM_ORDER.index(k) if k in _ARM_ORDER else 99, k))


def _model_and_tp(run: dict[str, Any]) -> tuple[str | None, str | None]:
    arms = (run.get("config") or {}).get("arms") or {}
    argvs = {k: _vllm_argv(arms[k]) for k in _ordered(arms)}
    model = (_merged({k: _flag(a, "--model") for k, a in argvs.items()})
             or _merged({k: b.get("model") for k, b in (run.get("bench") or {}).items()}))
    # vLLM's own default, when the argv leaves it out.
    tp = _merged({k: _flag(a, "--tensor-parallel-size") or "1" for k, a in argvs.items()})
    return model, tp


def _run_headline(run: dict[str, Any]) -> str:
    """'<model> · <GPU> · TP=<n>' for the page header, or '' if not recorded."""
    model, tp = _model_and_tp(run)
    gpu = ((run.get("config") or {}).get("gpu") or {}).get("name")
    return " · ".join(p for p in (model, gpu, f"TP={tp}" if tp else None) if p)


def _spec_rows(run: dict[str, Any]) -> list[tuple[str, str]]:
    """(item, value) rows that describe the run as a whole."""
    config = run.get("config") or {}
    arms = config.get("arms") or {}
    argvs = {k: _vllm_argv(arms[k]) for k in _ordered(arms)}
    bench = {k: run["bench"][k] for k in _ordered(run.get("bench") or {})}
    rows: list[tuple[str, str]] = []

    def add(item: str, value: str | None) -> None:
        if value:
            rows.append((item, value))

    gpu = config.get("gpu") or {}
    try:
        vram = float(gpu["vram_bytes"]) if gpu.get("vram_bytes") else None
    except (TypeError, ValueError):
        vram = None
    if gpu.get("name"):
        arch = f" ({gpu['arch']})" if gpu.get("arch") else ""
        add("GPU", f"{gpu['name']}{arch}" + (f", {_fmt_gib(vram)} VRAM" if vram else ""))
    model, tp = _model_and_tp(run)
    if tp:
        add("Tensor parallelism",
            f"TP={tp} ({tp} GPU{'' if tp == '1' else 's'})" if tp.isdigit() else tp)
    add("Model", model)

    gpus = "the GPU's" if tp in (None, "1") else "each GPU's"

    def budget(argv: list[str]) -> str | None:
        util = _flag(argv, "--gpu-memory-utilization")
        if not util:
            return None
        if vram:
            return (f"{_fmt_gib(float(util) * vram)} of {gpus} {_fmt_gib(vram)} "
                    f"(--gpu-memory-utilization {util}), shared by the weights, "
                    "activations and KV cache")
        return f"--gpu-memory-utilization {util} of {gpus} VRAM"
    add("vLLM VRAM budget", _merged({k: budget(a) for k, a in argvs.items()}))

    def engine(argv: list[str]) -> str | None:
        parts = []
        if _flag(argv, "--max-model-len"):
            parts.append(f"max model length {int(_flag(argv, '--max-model-len')):,}")
        if _flag(argv, "--kv-cache-dtype"):
            parts.append(f"{_flag(argv, '--kv-cache-dtype')} KV cache")
        if _flag(argv, "--block-size"):
            parts.append(f"{_flag(argv, '--block-size')}-token blocks")
        if _flag(argv, "--attention-backend"):
            parts.append(f"{_flag(argv, '--attention-backend')} attention")
        return " · ".join(parts) or None
    add("vLLM engine", _merged({k: engine(a) for k, a in argvs.items()}))

    add("Software", _merged({
        k: _stack_versions((arms[k].get("vllm") or {}).get("image") or "") for k in argvs}))

    images: list[str] = []
    for key in argvs:
        for svc in ("vllm", "lmcache"):
            info = arms[key].get(svc) or {}
            if info.get("image"):
                image_id = (info.get("image_id") or "").split(":")[-1][:12]
                image = info["image"] + (f" (image {image_id})" if image_id else "")
                if image not in images:
                    images.append(image)
    add("Images", ", ".join(images))

    def prompt(b: dict) -> str | None:
        try:
            isl, shared = int(b["isl"]), int(b["shared_prefix_tokens"])
        except (KeyError, TypeError, ValueError):
            return None
        reuse = {"per_client": "that each client reuses on every request",
                 "shared": "shared by every client"}.get(
                     b.get("prefix_mode"), f"(prefix mode {b.get('prefix_mode')})")
        return (f"{isl:,} tokens per request: {shared:,} prefix tokens {reuse}, "
                f"plus {isl - shared:,} unique tokens")
    add("Prompt", _merged({k: prompt(b) for k, b in bench.items()}))

    def output(b: dict) -> str | None:
        n = b.get("max_tokens")
        return None if n is None else f"{n} token{'' if n == 1 else 's'} per request"
    add("Output", _merged({k: output(b) for k, b in bench.items()}))
    add("Concurrency", _merged({
        k: ", ".join(c.strip() for c in str(b.get("concurrencies") or "").split(",") if c.strip())
        for k, b in bench.items()}))

    def iterations(b: dict) -> str | None:
        if b.get("iters") is None:
            return None
        timed = f"{b['iters']} timed per concurrency, median shown"
        if b.get("warmup_at_each_c"):
            waves = b.get("warmup_waves_per_c", 1)
            return (f"{timed}; {waves} warmup wave{'' if waves == 1 else 's'} "
                    "at each concurrency before timing")
        if b.get("warmup_iters"):
            return f"{timed}; {b['warmup_iters']} warmup iteration(s) at the lowest concurrency"
        return timed
    add("Iterations", _merged({k: iterations(b) for k, b in bench.items()}))
    return rows


def _tier_rows(run: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """(arm, GPU prefix cache, L1, L2) for every arm with a recorded config."""
    arms = (run.get("config") or {}).get("arms") or {}
    bench = run.get("bench") or {}
    rows = []
    for key in _ordered(arms):
        lmcache = arms[key].get("lmcache")
        if not lmcache:
            rows.append((key, "Only KV cache tier", "None", "None"))
            continue
        reset = (bench.get(key) or {}).get("reset_prefix_cache")
        gpu = {True: "Emptied before each timed iteration",
               False: "Kept between iterations"}.get(reset, "In front of LMCache")
        rows.append((key, gpu, *_lmcache_tiers(lmcache)))
    return rows


def _spec_html(run: dict[str, Any]) -> str:
    esc = html.escape
    rows = _spec_rows(run)
    tiers = _tier_rows(run)
    if not rows and not tiers:
        return '<p class="missing">Configuration not recorded for this run</p>'
    out = ""
    if rows:
        body = "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows)
        out += f'\n  <table class="spec"><tbody>{body}</tbody></table>'
    if tiers:
        body = "".join(
            f"<tr><td>{_arm_span(arm)}</td><td>{esc(gpu)}</td><td>{esc(l1)}</td>"
            f"<td>{esc(l2)}</td></tr>" for arm, gpu, l1, l2 in tiers)
        out += f"""
  <h3 class="sub">KV cache tiers</h3>
  <table class="tiers">
    <thead><tr><th>Arm</th><th>GPU prefix cache (vLLM)</th><th>LMCache L1</th><th>LMCache L2</th></tr></thead>
    <tbody>{body}</tbody>
  </table>"""
    return out


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --bg: #0d1117;
  --surface: #161b22;
  --border: #30363d;
  --text: #e6edf3;
  --muted: #8b949e;
  --accent: #2980b9;
  --green: #3fb950;
  --red: #f85149;
  --orange: #e67e22;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 14px; line-height: 1.5; padding: 24px; }
h1 { font-size: 24px; font-weight: 700; margin-bottom: 4px; }
h2 { font-size: 17px; font-weight: 600; margin: 32px 0 12px; color: var(--text); border-bottom: 1px solid var(--border); padding-bottom: 6px; }
.meta { color: var(--muted); font-size: 12px; margin-bottom: 24px; }
.meta a { color: var(--accent); text-decoration: none; }
.meta a:hover { text-decoration: underline; }
.kpi-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px 20px; min-width: 180px; }
.kpi-label { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 6px; }
.kpi-value { font-size: 26px; font-weight: 700; color: var(--text); }
.kpi-sub { font-size: 11px; color: var(--muted); margin-top: 2px; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 20px; margin-bottom: 32px; }
.chart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.chart-card h3 { font-size: 13px; font-weight: 600; color: var(--muted); margin-bottom: 12px; text-transform: uppercase; letter-spacing: .06em; }
.chart-card img { width: 100%; height: auto; border-radius: 4px; }
.chart-card .missing { color: var(--muted); font-size: 12px; font-style: italic; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
thead th { text-align: left; padding: 8px 12px; font-weight: 600; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; border-bottom: 1px solid var(--border); }
tbody tr { border-bottom: 1px solid var(--border); }
tbody tr:hover { background: var(--surface); }
tbody td { padding: 8px 12px; }
.badge-ok { color: var(--green); }
.badge-miss { color: var(--red); }
.badge-na { color: var(--muted); }
.arm-vram_only { color: #e05c4e; }
.arm-vram_dram { color: var(--orange); }
.arm-kvd_v2, .arm-kvd_v2-nvme { color: var(--accent); }
.arm-kvd_v2-gds { color: #b07cc6; }
.headline { font-size: 15px; color: var(--text); margin-bottom: 4px; }
h3.sub { font-size: 13px; font-weight: 600; color: var(--muted); margin: 20px 0 8px; text-transform: uppercase; letter-spacing: .06em; }
table.spec tbody th { text-align: left; vertical-align: top; white-space: nowrap; width: 1%; padding: 8px 24px 8px 12px; font-weight: 600; color: var(--muted); }
input#search { width: 100%; max-width: 400px; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; color: var(--text); padding: 8px 12px; font-size: 13px; margin-bottom: 12px; outline: none; }
input#search:focus { border-color: var(--accent); }
"""

_JS = """
document.getElementById('search').addEventListener('input', function() {
  const q = this.value.toLowerCase();
  document.querySelectorAll('#results-table tbody tr').forEach(function(tr) {
    tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
  });
});
"""


def _arm_span(arm: str) -> str:
    label = _ARM_LABELS.get(arm, arm)
    return f'<span class="arm-{arm}">{label}</span>'


def _fmt_thr(v: float) -> str:
    return f"{v:,.0f} tok/s"


def _fmt_lat(v: float) -> str:
    return f"{v:.0f} ms"


def _fmt_pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.1f}%"


def _build_html(
    runs: list[dict[str, Any]],
    page_sha: str,
    page_date: str,
    repo_url: str,
) -> str:

    # Most recent run's data for KPIs + charts
    latest = runs[-1] if runs else {}
    latest_arms = latest.get("arms", {})
    latest_charts = latest.get("charts", {})

    # --- KPI cards ---
    kpi_cards = ""
    for arm in _ARM_ORDER:
        if arm not in latest_arms:
            continue
        summary = latest_arms[arm]
        peak = _peak_thr(summary)
        cliff = _cliff_concurrency(summary)
        label = _ARM_LABELS.get(arm, arm)
        cliff_str = f"cliff @ c={cliff}" if cliff else "no cliff detected"
        kpi_cards += f"""
        <div class="kpi">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{peak:,.0f}</div>
          <div class="kpi-sub">tok/s peak &nbsp;·&nbsp; {cliff_str}</div>
        </div>"""

    if not kpi_cards:
        kpi_cards = '<div class="kpi"><div class="kpi-label">No data yet</div><div class="kpi-value">—</div></div>'

    # --- Chart cards: one row per backend, each against the vram_only baseline ---
    def _chart_card(title: str, src: str | None) -> str:
        inner = f'<img src="{src}" alt="{title}">' if src else '<p class="missing">Chart not available for this run</p>'
        return f'<div class="chart-card"><h3>{title}</h3>{inner}</div>'

    charts_html = ""
    for backend, imgs in latest_charts.items():
        charts_html += f"""
    <div class="charts">
      {_chart_card(f"Throughput vs Concurrency ({backend} backend)", imgs.get("throughput"))}
      {_chart_card(f"p50 Latency vs Concurrency ({backend} backend)", imgs.get("p50"))}
      {_chart_card(f"p95 Latency vs Concurrency ({backend} backend)", imgs.get("p95"))}
      {_chart_card(f"Cache Hit Rate vs Concurrency ({backend} backend)", imgs.get("hit"))}
    </div>"""
    if not charts_html:
        charts_html = '<p class="missing">No charts for this run</p>'

    # --- Results table (one row per arm per concurrency, latest run) ---
    rows_html = ""
    for arm in _ARM_ORDER:
        if arm not in latest_arms:
            continue
        summary = latest_arms[arm]
        cliff_c = _cliff_concurrency(summary)
        for c in sorted(summary):
            d = summary[c]
            at_cliff = cliff_c is not None and c == cliff_c
            cliff_marker = ' <span class="badge-miss" title="throughput cliff">⚡</span>' if at_cliff else ""
            rows_html += f"""<tr>
              <td>{page_date}</td>
              <td>{_arm_span(arm)}</td>
              <td>{c}</td>
              <td>{_fmt_thr(d['thr'])}{cliff_marker}</td>
              <td>{_fmt_lat(d['p50'])}</td>
              <td>{_fmt_lat(d['p95'])}</td>
              <td>{_fmt_pct(d.get('l1_hit'))}</td>
              <td>{_fmt_pct(d.get('ext_hit'))}</td>
            </tr>"""

    if not rows_html:
        rows_html = '<tr><td colspan="8" style="color:var(--muted);text-align:center;padding:24px">No results yet — run the nightly cliff workflow.</td></tr>'

    commit_link = f'<a href="{repo_url}/commit/{page_sha}">{page_sha[:7]}</a>' if repo_url else page_sha[:7]
    headline = _run_headline(latest) if latest else ""
    headline_html = f'<p class="headline">{html.escape(headline)}</p>' if headline else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AIC Cliff Performance — {page_date}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>AIC KV Cache Cliff — Performance Tracking</h1>
  {headline_html}
  <p class="meta">
    Last updated: {page_date} &nbsp;·&nbsp;
    Commit: {commit_link} &nbsp;·&nbsp;
    <a href="{repo_url}/actions/workflows/rocm-aic-self-hosted-nightly-cliff-test.yml">Nightly Cliff workflow</a>
  </p>

  <h2>Latest Run — Key Metrics</h2>
  <div class="kpi-row">{kpi_cards}</div>

  <h2>Test Configuration</h2>
  {_spec_html(latest)}

  <h2>Charts</h2>
  {charts_html}

  <h2>Raw Results</h2>
  <input id="search" type="search" placeholder="Filter results…" autocomplete="off">
  <table id="results-table">
    <thead>
      <tr>
        <th>Date</th>
        <th>Arm</th>
        <th>Concurrency</th>
        <th>Throughput</th>
        <th>p50 Latency</th>
        <th>p95 Latency</th>
        <th>GPU Cache Hit</th>
        <th>LMCache Hit</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>

  <script>{_JS}</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--run-dir", nargs="+", metavar="DIR",
        help="One or more run directories (each with results/ and plots/ subdirs)",
    )
    grp.add_argument(
        "--manifest", metavar="JSON",
        help="JSON manifest written by spur-cliff-harvest.sh "
             "(keys: run_dir, sha, run_date, repo_url)",
    )
    parser.add_argument("--sha", default="", help="Git SHA for this run (used in links)")
    parser.add_argument("--run-date", default="", help="ISO date string for this run")
    parser.add_argument(
        "--repo-url", default="https://github.com/ROCm/rocm-aic",
        help="GitHub repo base URL for commit/workflow links",
    )
    parser.add_argument(
        "--output-dir", default="cliff-page",
        help="Directory to write index.html into (default: cliff-page/)",
    )
    args = parser.parse_args()

    if args.manifest:
        with open(args.manifest) as fh:
            manifest = json.load(fh)
        run_dirs = [Path(manifest["run_dir"])]
        sha = manifest.get("sha", "")
        run_date = manifest.get("run_date", "")
        repo_url = manifest.get("repo_url", args.repo_url)
    else:
        run_dirs = [Path(d) for d in args.run_dir]
        sha = args.sha
        run_date = args.run_date
        repo_url = args.repo_url

    runs = []
    for d in run_dirs:
        if not d.is_dir():
            print(f"WARN: {d} is not a directory, skipping", file=sys.stderr)
            continue
        runs.append(_discover_run(d))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = _build_html(runs, sha, run_date, repo_url)
    out_path = out_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Written {out_path} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
