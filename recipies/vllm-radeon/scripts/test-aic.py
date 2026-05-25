#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""LMCache AIC prefill A/B harness (populate → cold → warm).

See ``python3 scripts/test-aic.py --help`` for CLI options and examples.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

SCRIPT = "test-aic"
DISK_BACKEND_HIPFILE = "GdsBackend"
DISK_BACKEND_POSIX = "RemoteBackend-fs"
SALT_PREFIX = "test-aic"

DEFAULT_GPU = 0
DEFAULT_SLUG = "war-and-peace"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_MAX_TOKENS = 32
DEFAULT_PAUSE_S = 2.0
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_LMCACHE_IO = "auto"
DEFAULT_ITERATIONS = 1

_CLI_EPILOG = """\
examples:
  # Default fixture (first chunk + question under data/war-and-peace/)
  python3 scripts/test-aic.py -o logs/test-aic.json

  # Explicit context; posix disk backend (RADEON_LMCACHE_IO=posix)
  python3 scripts/test-aic.py --lmcache-io posix \\
      --context data/war-and-peace/war-and-peace-10k.100270.txt \\
      -r my-run -o logs/my-run.json

  # Skip populate when NVMe already has this cache_salt
  python3 scripts/test-aic.py -r my-run --skip-populate

  # Five cold+warm cycles (populate once on iteration 1)
  python3 scripts/test-aic.py -n 5 -r bench -o logs/bench.json

  # Machine-readable JSON on stdout
  python3 scripts/test-aic.py --json -o logs/out.json

requirements:
  pip install 'openai>=1.40.0'   # this script only; full repo: pip install -r requirements.txt
  make run                          # VLLM_SERVER_DEV_MODE=1 for reset_prefix_cache
  LMCache image with cache_salt patch (#3008) for isolated NVMe keys

phases (same prompt, question, cache_salt=test-aic-<run-id>):
  populate  store KV to disk (GdsBackend or RemoteBackend-fs)
  cold      reset GPU prefix cache; bypass disk → full GPU prefill
  warm      reset GPU prefix cache; read KV from disk
"""


class _HelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    """Preserve newlines in epilog and show default= for each option."""


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _recipe_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_server_log(recipe_root: Path) -> Path | None:
    path = recipe_root / "logs" / "server.txt"
    return path if path.is_file() else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=SCRIPT,
        description=(
            "Run populate / cold / warm prefill A/B against a vLLM + LMCache server."
        ),
        formatter_class=_HelpFormatter,
        epilog=_CLI_EPILOG,
    )

    fixture = p.add_argument_group(
        "fixture",
        "Long-context input and cache_salt identity (LMCache #3008).",
    )
    fixture.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=DEFAULT_GPU,
        metavar="N",
        help=(
            "GPU index for default URLs: vLLM http://127.0.0.1:800{N}, "
            "LMCache worker http://127.0.0.1:699{N+1}."
        ),
    )
    fixture.add_argument(
        "-r",
        "--run-id",
        default="",
        metavar="ID",
        help=(
            "Suffix for cache_salt (test-aic-<ID>). Use a new ID for a fresh "
            "NVMe store; reuse to hit existing KV."
        ),
    )
    fixture.add_argument(
        "--slug",
        default=DEFAULT_SLUG,
        metavar="NAME",
        help=(
            "Book slug under data/<NAME>/ when --context or --question is omitted "
            "(requires make data)."
        ),
    )
    fixture.add_argument(
        "--context",
        type=Path,
        metavar="PATH",
        help="Context chunk file. Default: first data/<slug>/<slug>-*.txt.",
    )
    fixture.add_argument(
        "--question",
        metavar="TEXT",
        help="User question for the chat turn. Default: first entry in .questions.json.",
    )

    vllm = p.add_argument_group("vLLM", "OpenAI-compatible completion API.")
    vllm.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="ID",
        help="Model name passed to the chat completions API.",
    )
    vllm.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        metavar="N",
        help="Max completion tokens per phase (streaming, include_usage).",
    )
    vllm.add_argument(
        "--url",
        metavar="URL",
        help="vLLM base URL. Default: http://127.0.0.1:800<gpu> (see --gpu).",
    )
    vllm.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        metavar="SEC",
        help="HTTP timeout for vLLM and LMCache API calls.",
    )

    lmcache = p.add_argument_group(
        "LMCache",
        "Disk backend selection and cold-phase bypass target.",
    )
    lmcache.add_argument(
        "--lmcache-io",
        default=DEFAULT_LMCACHE_IO,
        choices=("auto", "hipfile", "posix"),
        metavar="MODE",
        help=(
            "Which disk backend to use for populate/cold/warm. "
            "'auto' reads GET /bypass/list (all_backends): GdsBackend (hipfile) "
            "or RemoteBackend-fs (posix). Match RADEON_LMCACHE_IO at make run."
        ),
    )

    phases = p.add_argument_group(
        "phases",
        "Control the three-phase sequence and GPU prefix cache resets.",
    )
    phases.add_argument(
        "--skip-populate",
        action="store_true",
        help="Omit populate; run only cold and warm (NVMe already has this salt).",
    )
    phases.add_argument(
        "--no-reset",
        action="store_true",
        help=(
            "Do not call POST /reset_prefix_cache before cold or warm "
            "(requires VLLM_SERVER_DEV_MODE=1 when enabled)."
        ),
    )
    phases.add_argument(
        "--pause",
        type=float,
        default=DEFAULT_PAUSE_S,
        metavar="SEC",
        help="Sleep between phases and between iterations.",
    )
    phases.add_argument(
        "-n",
        "--iterations",
        type=_positive_int,
        default=DEFAULT_ITERATIONS,
        metavar="N",
        help=(
            "Repeat the test N times. Populate runs only on iteration 1 "
            "(unless --skip-populate); each iteration runs cold then warm."
        ),
    )

    output = p.add_argument_group("output", "Stdout and JSON report file.")
    output.add_argument(
        "-o",
        "--report",
        type=Path,
        metavar="PATH",
        help="Write full JSON summary to PATH (always; not printed unless --json).",
    )
    output.add_argument(
        "--json",
        action="store_true",
        help="Also print JSON summary to stdout (default: human table only).",
    )

    logging_grp = p.add_argument_group(
        "logging",
        "Parse recipies/vllm-radeon/logs/server.txt for NVMe and hit-rate columns.",
    )
    log_exclusive = logging_grp.add_mutually_exclusive_group()
    log_exclusive.add_argument(
        "--server-log",
        type=Path,
        metavar="PATH",
        help=(
            "Engine log to tail per phase. Default: <recipe>/logs/server.txt "
            "if that file exists."
        ),
    )
    log_exclusive.add_argument(
        "--no-server-log",
        action="store_true",
        help="Do not parse any log; NVMe / LMCache hit columns stay empty.",
    )

    return p


def config_from_namespace(ns: argparse.Namespace, recipe_root: Path) -> RunConfig:
    run_id = (ns.run_id or uuid.uuid4().hex[:8]).strip()
    if ns.no_server_log:
        server_log: Path | None = None
    elif ns.server_log is not None:
        server_log = ns.server_log
    else:
        server_log = _default_server_log(recipe_root)

    return RunConfig(
        gpu=ns.gpu,
        run_id=run_id,
        slug=ns.slug,
        context_path=ns.context,
        question=ns.question,
        model=ns.model,
        max_tokens=ns.max_tokens,
        skip_populate=ns.skip_populate,
        reset_gpu=not ns.no_reset,
        pause_s=ns.pause,
        report=ns.report,
        timeout=ns.timeout,
        vllm_url=ns.url,
        server_log=server_log,
        lmcache_io=ns.lmcache_io,
        json_stdout=ns.json,
        iterations=ns.iterations,
    )


@dataclass
class RunConfig:
    gpu: int
    run_id: str
    slug: str
    context_path: Path | None
    question: str | None
    model: str
    max_tokens: int
    skip_populate: bool
    reset_gpu: bool
    pause_s: float
    report: Path | None
    timeout: float
    vllm_url: str | None
    server_log: Path | None
    lmcache_io: str
    json_stdout: bool
    iterations: int

    @property
    def cache_salt(self) -> str:
        return f"{SALT_PREFIX}-{self.run_id}"

    @property
    def vllm_base(self) -> str:
        if self.vllm_url:
            return self.vllm_url.rstrip("/")
        return f"http://127.0.0.1:800{self.gpu}"

    @property
    def lmcache_base(self) -> str:
        return f"http://127.0.0.1:{int(f'699{self.gpu}') + 1}"


@dataclass
class EngineLogSlice:
    """Parsed from server.txt lines written during one phase."""

    stored_tokens: int | None = None
    store_ms: float | None = None
    store_gbps: float | None = None
    retrieved_tokens: int | None = None
    retrieve_ms: float | None = None
    retrieve_gbps: float | None = None
    batched_get_s: float | None = None
    batched_get_mib: float | None = None
    need_to_load: int | None = None
    lmcache_hit_tokens: int | None = None
    external_prefix_hit_pct: float | None = None


@dataclass
class PhaseResult:
    phase: str
    wall_seconds: float
    ttft_seconds: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cache_salt: str
    gpu_prefix_reset: bool
    disk_backend: str
    lmcache_bypassed_backends: list[str]
    engine: EngineLogSlice = field(default_factory=EngineLogSlice)
    error: str | None = None


_STORE_RE = re.compile(
    r"Stored (\d+) out of total \d+ tokens\. size: [\d.]+ GB, "
    r"cost ([\d.]+) ms, throughput: ([\d.]+) GB/s"
)
_RETRIEVE_RE = re.compile(
    r"Retrieved (\d+) out of (\d+) required tokens.*?cost ([\d.]+) ms, "
    r"throughput: ([\d.]+) GB/s"
)
_NEED_LOAD_RE = re.compile(
    r"Total tokens (\d+), .* LMCache hit tokens: (\d+), need to load: (\d+)"
)
_BATCH_GET_RE = re.compile(r"batched_get_blocking: ([\d.]+)s \| ([\d.]+)MiB")
_EXTERNAL_HIT_RE = re.compile(r"External prefix cache hit rate: ([\d.]+)%")


def _http(
    method: str,
    url: str,
    timeout: float,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, Any]:
    hdrs = {"Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        status = e.code
    if not raw.strip():
        return status, None
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def reset_gpu_prefix(vllm_base: str, timeout: float) -> None:
    url = f"{vllm_base.rstrip('/')}/reset_prefix_cache"
    status, body = _http("POST", url, timeout)
    if status != 200:
        raise RuntimeError(
            f"reset_prefix_cache HTTP {status}: {body!r} "
            "(need VLLM_SERVER_DEV_MODE=1; make run)"
        )


def storage_mode_get(base: str, timeout: float) -> str | None:
    status, payload = _http("GET", f"{base}/storage/mode", timeout)
    if status == 404:
        return None
    if status != 200:
        raise RuntimeError(f"storage/mode GET HTTP {status}: {payload!r}")
    mode = payload.get("mode")
    return str(mode) if mode else None


def storage_mode_set(
    base: str,
    mode: str,
    timeout: float,
    *,
    gds_path: str | None = None,
    fs_base_path: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"mode": mode}
    if gds_path:
        body["gds_path"] = gds_path
    if fs_base_path:
        body["fs_base_path"] = fs_base_path
    status, payload = _http(
        "POST",
        f"{base}/storage/mode",
        timeout,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    if status == 404:
        raise RuntimeError(
            "POST /storage/mode unavailable (rebuild image with "
            "lmcache-storage-mode-switch.patch)"
        )
    if status != 200:
        raise RuntimeError(f"storage/mode POST HTTP {status}: {payload!r}")
    return payload


def lmcache_all_backends(base: str, timeout: float) -> list[str]:
    status, payload = _http("GET", f"{base}/bypass/list", timeout)
    if status != 200 or not isinstance(payload, dict):
        raise RuntimeError(f"bypass/list HTTP {status}: {payload!r}")
    names = payload.get("all_backends")
    if not isinstance(names, list):
        raise RuntimeError(f"unexpected bypass/list: {payload!r}")
    return [str(x) for x in names]


def resolve_disk_backend(base: str, timeout: float, lmcache_io: str) -> str:
    """Return the LMCache disk backend name to bypass for the cold phase."""
    mode = lmcache_io.strip().lower()
    if mode in ("gds", "hipfile", "hip"):
        want = DISK_BACKEND_HIPFILE
    elif mode in ("posix", "mmap", "non-gds", "fs"):
        want = DISK_BACKEND_POSIX
    elif mode == "auto":
        want = None
    else:
        raise ValueError(
            f"--lmcache-io must be auto, hipfile, or posix (got {lmcache_io!r})",
        )

    available = lmcache_all_backends(base, timeout)
    has_hipfile = DISK_BACKEND_HIPFILE in available
    has_posix = DISK_BACKEND_POSIX in available

    if want is not None:
        if want not in available:
            raise RuntimeError(
                f"disk backend {want!r} not in LMCache backends {available!r}; "
                f"start server with matching RADEON_LMCACHE_IO or use --lmcache-io auto",
            )
        return want

    if has_hipfile:
        return DISK_BACKEND_HIPFILE
    if has_posix:
        return DISK_BACKEND_POSIX
    raise RuntimeError(
        f"no disk backend in LMCache backends {available!r} "
        f"(expected {DISK_BACKEND_HIPFILE} or {DISK_BACKEND_POSIX})",
    )


def bypass_list(base: str, timeout: float) -> list[str]:
    status, payload = _http("GET", f"{base}/bypass/list", timeout)
    if status != 200 or not isinstance(payload, dict):
        raise RuntimeError(f"bypass/list HTTP {status}: {payload!r}")
    bypassed = payload.get("bypassed_backends")
    if not isinstance(bypassed, list):
        raise RuntimeError(f"unexpected bypass/list: {payload!r}")
    return [str(x) for x in bypassed]


def bypass_set(base: str, backend: str, on: bool, timeout: float) -> None:
    action = "add" if on else "remove"
    q = urllib.parse.urlencode({"backend_name": backend})
    status, payload = _http("PUT", f"{base}/bypass/{action}?{q}", timeout)
    if status != 200:
        raise RuntimeError(f"bypass/{action} HTTP {status}: {payload!r}")


def default_fixture(recipe_root: Path, slug: str) -> tuple[Path, str]:
    book_dir = recipe_root / "data" / slug
    chunks = sorted(book_dir.glob(f"{slug}-*.txt"))
    if not chunks:
        raise FileNotFoundError(f"no chunks under {book_dir}; run 'make data'")
    questions_path = book_dir / f"{slug}.questions.json"
    if not questions_path.is_file():
        raise FileNotFoundError(f"missing {questions_path}; run 'make data'")
    data = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"no questions in {questions_path}")
    return chunks[0], str(questions[0])


def chat_messages(context: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Answer using only the provided context."},
        {
            "role": "user",
            "content": f"Context:\n\n{context}\n\nQuestion: {question}",
        },
    ]


class ServerLogTail:
    """Read only new bytes appended since the last slice (for per-phase parsing)."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._offset = 0

    def read_slice(self) -> str:
        if self.path is None or not self.path.is_file():
            return ""
        with self.path.open("rb") as f:
            f.seek(self._offset)
            data = f.read().decode("utf-8", errors="replace")
            self._offset = f.tell()
        return data


def parse_engine_log_slice(text: str) -> EngineLogSlice:
    out = EngineLogSlice()
    stores = _STORE_RE.findall(text)
    if stores:
        tokens = sum(int(t) for t, _, _ in stores)
        ms = sum(float(m) for _, m, _ in stores)
        gbps_vals = [float(g) for _, _, g in stores]
        out.stored_tokens = tokens
        out.store_ms = ms
        out.store_gbps = sum(gbps_vals) / len(gbps_vals)
    m = _RETRIEVE_RE.search(text)
    if m:
        out.retrieved_tokens = int(m.group(1))
        out.retrieve_ms = float(m.group(3))
        out.retrieve_gbps = float(m.group(4))
    m = _BATCH_GET_RE.search(text)
    if m:
        out.batched_get_s = float(m.group(1))
        out.batched_get_mib = float(m.group(2))
    need = list(_NEED_LOAD_RE.finditer(text))
    if need:
        last = need[-1]
        out.lmcache_hit_tokens = int(last.group(2))
        out.need_to_load = int(last.group(3))
    hits = _EXTERNAL_HIT_RE.findall(text)
    if hits:
        out.external_prefix_hit_pct = float(hits[-1])
    return out


def stream_completion(
    client: OpenAI,
    cfg: RunConfig,
    messages: list[dict[str, str]],
) -> tuple[float, float | None, int | None, int | None, str | None]:
    t0 = time.perf_counter()
    ttft: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    try:
        stream = client.chat.completions.create(
            model=cfg.model,
            messages=messages,
            max_tokens=cfg.max_tokens,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"cache_salt": cfg.cache_salt},
        )
        for chunk in stream:
            if chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
    except Exception as e:
        return time.perf_counter() - t0, ttft, prompt_tokens, completion_tokens, str(e)
    return time.perf_counter() - t0, ttft, prompt_tokens, completion_tokens, None


def _fmt_num(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def _tok_per_s(tokens: int | None, seconds: float | None) -> float | None:
    if tokens is None or seconds is None or seconds <= 0:
        return None
    return tokens / seconds


def build_metrics_table(
    results: list[PhaseResult],
    max_tokens: int,
) -> list[dict[str, Any]]:
    """One row per phase with client + engine (NVMe / AIC) metrics."""
    rows: list[dict[str, Any]] = []
    for pr in results:
        prompt = pr.prompt_tokens
        completion = pr.completion_tokens if pr.completion_tokens is not None else max_tokens
        ttft = pr.ttft_seconds
        decode_s = (pr.wall_seconds - ttft) if ttft is not None else None
        eng = pr.engine
        nvme_ms = eng.retrieve_ms if pr.phase == "warm" else eng.store_ms
        nvme_gbps = eng.retrieve_gbps if pr.phase == "warm" else eng.store_gbps
        nvme_op = (
            "retrieve"
            if pr.phase == "warm"
            else ("store" if eng.stored_tokens else "—")
        )
        token_hit_pct: float | None = None
        if prompt and eng.lmcache_hit_tokens is not None and prompt > 0:
            token_hit_pct = 100.0 * eng.lmcache_hit_tokens / prompt
        rows.append(
            {
                "phase": pr.phase,
                "ttft_s": ttft,
                "wall_s": pr.wall_seconds,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "prefill_tok_per_s": _tok_per_s(prompt, ttft),
                "decode_tok_per_s": _tok_per_s(completion, decode_s),
                "e2e_tok_per_s": _tok_per_s(
                    (prompt or 0) + completion,
                    pr.wall_seconds,
                )
                if prompt
                else None,
                "nvme_op": nvme_op,
                "nvme_ms": nvme_ms,
                "nvme_gbps": nvme_gbps,
                "nvme_kv_tokens": eng.retrieved_tokens or eng.stored_tokens,
                "need_to_load": eng.need_to_load,
                "lmcache_hit_tokens": eng.lmcache_hit_tokens,
                "lmcache_token_hit_pct": token_hit_pct,
                "external_prefix_hit_pct": eng.external_prefix_hit_pct,
                "gpu_prefix_reset": pr.gpu_prefix_reset,
                "disk_bypassed": pr.disk_backend in pr.lmcache_bypassed_backends,
            }
        )
    return rows


def format_metrics_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "phase",
        "TTFT(s)",
        "wall(s)",
        "prefill tok/s",
        "decode tok/s",
        "e2e tok/s",
        "NVMe",
        "NVMe ms",
        "GB/s",
        "need load",
        "LM hit%",
        "ext hit%",
    ]
    body: list[list[str]] = []
    for r in rows:
        body.append(
            [
                str(r["phase"]),
                _fmt_num(r.get("ttft_s"), 3),
                _fmt_num(r.get("wall_s"), 3),
                _fmt_num(r.get("prefill_tok_per_s"), 0),
                _fmt_num(r.get("decode_tok_per_s"), 0),
                _fmt_num(r.get("e2e_tok_per_s"), 0),
                str(r.get("nvme_op") or "—"),
                _fmt_num(r.get("nvme_ms"), 1),
                _fmt_num(r.get("nvme_gbps"), 2),
                str(r["need_to_load"]) if r.get("need_to_load") is not None else "—",
                _fmt_num(r.get("lmcache_token_hit_pct"), 1),
                _fmt_num(r.get("external_prefix_hit_pct"), 1),
            ]
        )
    widths = [max(len(headers[i]), *(len(row[i]) for row in body)) for i in range(len(headers))]

    def line(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    sep = line(["-" * widths[i] for i in range(len(headers))])
    out = [line(headers), sep]
    out.extend(line(row) for row in body)
    return "\n".join(out)


def _mean_optional(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_metrics_tables(
    tables: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Mean numeric columns per phase across iteration tables."""
    if not tables:
        return []
    if len(tables) == 1:
        return list(tables[0])

    by_phase: dict[str, list[dict[str, Any]]] = {}
    for table in tables:
        for row in table:
            by_phase.setdefault(str(row["phase"]), []).append(row)

    numeric_keys = (
        "ttft_s",
        "wall_s",
        "prefill_tok_per_s",
        "decode_tok_per_s",
        "e2e_tok_per_s",
        "nvme_ms",
        "nvme_gbps",
        "lmcache_token_hit_pct",
        "external_prefix_hit_pct",
    )
    agg: list[dict[str, Any]] = []
    for phase in ("populate", "cold", "warm"):
        rows = by_phase.get(phase, [])
        if not rows:
            continue
        n = len(rows)
        out: dict[str, Any] = {"phase": f"{phase} (mean n={n})"}
        for key in numeric_keys:
            vals = [float(r[key]) for r in rows if r.get(key) is not None]
            out[key] = _mean_optional(vals)
        sample = rows[0]
        out["nvme_op"] = sample.get("nvme_op")
        out["prompt_tokens"] = sample.get("prompt_tokens")
        out["completion_tokens"] = sample.get("completion_tokens")
        agg.append(out)
    return agg


def format_iteration_tables(
    tables: list[list[dict[str, Any]]],
    aggregate: list[dict[str, Any]],
) -> str:
    if len(tables) == 1:
        return format_metrics_table(tables[0])
    parts: list[str] = []
    total = len(tables)
    for i, table in enumerate(tables, start=1):
        parts.append(f"--- iteration {i}/{total} ---")
        parts.append(format_metrics_table(table))
    if aggregate:
        parts.append(f"--- aggregate (mean over {total} iterations) ---")
        parts.append(format_metrics_table(aggregate))
    return "\n\n".join(parts)


def combine_iteration_summaries(
    cfg: RunConfig,
    runs: list[dict[str, Any]],
    aggregate_table: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(runs) == 1:
        return runs[0]

    cold_ttft = [
        r["delta"]["ttft_seconds_cold_minus_warm"]
        for r in runs
        if r.get("delta", {}).get("ttft_seconds_cold_minus_warm") is not None
    ]
    cold_wall = [
        r["delta"]["wall_seconds_cold_minus_warm"]
        for r in runs
        if r.get("delta", {}).get("wall_seconds_cold_minus_warm") is not None
    ]
    base = dict(runs[0])
    base["iterations"] = cfg.iterations
    base["runs"] = runs
    base["aggregate_metrics_table"] = aggregate_table
    base["aggregate_delta"] = {
        "ttft_seconds_cold_minus_warm_mean": _mean_optional(cold_ttft),
        "wall_seconds_cold_minus_warm_mean": _mean_optional(cold_wall),
    }
    return base


def parse_args(argv: list[str] | None = None) -> RunConfig:
    return config_from_namespace(build_parser().parse_args(argv), _recipe_root())


def run_ab(
    cfg: RunConfig,
    recipe_root: Path,
    server_log: Path | None,
    *,
    iteration: int = 1,
    total_iterations: int = 1,
) -> tuple[list[PhaseResult], dict[str, Any], list[dict[str, Any]]]:
    context_path = cfg.context_path
    question = cfg.question
    if context_path is None or question is None:
        default_ctx, default_q = default_fixture(recipe_root, cfg.slug)
        context_path = context_path or default_ctx
        question = question or default_q
    if not context_path.is_file():
        raise FileNotFoundError(f"context not found: {context_path}")

    context = context_path.read_text(encoding="utf-8", errors="replace")
    messages = chat_messages(context, question)

    client = OpenAI(
        base_url=f"{cfg.vllm_base}/v1",
        api_key="EMPTY",
        timeout=cfg.timeout,
    )

    iter_label = (
        f" iteration={iteration}/{total_iterations}"
        if total_iterations > 1
        else ""
    )
    print(
        f"{SCRIPT}: context={context_path} ({len(context)} bytes) "
        f"salt={cfg.cache_salt!r} reset={cfg.reset_gpu}{iter_label}",
        file=sys.stderr,
    )
    disk_backend = resolve_disk_backend(
        cfg.lmcache_base, cfg.timeout, cfg.lmcache_io,
    )
    print(
        f"{SCRIPT}: vllm={cfg.vllm_base} lmcache={cfg.lmcache_base} "
        f"disk_backend={disk_backend}",
        file=sys.stderr,
    )

    results: list[PhaseResult] = []
    log_tail = ServerLogTail(server_log)
    phases: list[tuple[str, bool, bool]] = []
    skip_populate = cfg.skip_populate or iteration > 1
    if not skip_populate:
        phases.append(("populate", False, False))
    phases.extend(
        [
            ("cold", True, True),
            ("warm", False, True),
        ]
    )

    try:
        for name, bypass_gds, reset_before in phases:
            print(f"{SCRIPT}: phase={name}", file=sys.stderr)
            did_reset = False
            if reset_before and cfg.reset_gpu:
                reset_gpu_prefix(cfg.vllm_base, cfg.timeout)
                did_reset = True

            if bypass_gds:
                bypass_set(cfg.lmcache_base, disk_backend, True, cfg.timeout)
            elif disk_backend in bypass_list(cfg.lmcache_base, cfg.timeout):
                bypass_set(cfg.lmcache_base, disk_backend, False, cfg.timeout)

            bypassed = bypass_list(cfg.lmcache_base, cfg.timeout)
            wall, ttft, prompt_tok, completion_tok, err = stream_completion(
                client, cfg, messages
            )
            engine = parse_engine_log_slice(log_tail.read_slice())
            results.append(
                PhaseResult(
                    phase=name,
                    wall_seconds=wall,
                    ttft_seconds=ttft,
                    prompt_tokens=prompt_tok,
                    completion_tokens=completion_tok,
                    cache_salt=cfg.cache_salt,
                    gpu_prefix_reset=did_reset,
                    disk_backend=disk_backend,
                    lmcache_bypassed_backends=bypassed,
                    engine=engine,
                    error=err,
                )
            )
            if err:
                print(f"{SCRIPT}: error in {name}: {err[:2000]}", file=sys.stderr)
                raise RuntimeError(f"phase {name} failed: {err}")

            if cfg.pause_s > 0:
                time.sleep(cfg.pause_s)
    finally:
        try:
            bypass_set(cfg.lmcache_base, disk_backend, False, cfg.timeout)
        except RuntimeError:
            pass

    cold = next(x for x in results if x.phase == "cold")
    warm = next(x for x in results if x.phase == "warm")
    metrics_table = build_metrics_table(results, cfg.max_tokens)
    summary: dict[str, Any] = {
        "iteration": iteration,
        "iterations_total": total_iterations,
        "run_id": cfg.run_id,
        "cache_salt": cfg.cache_salt,
        "context_file": str(context_path.resolve()),
        "question": question,
        "model": cfg.model,
        "vllm_base": cfg.vllm_base,
        "lmcache_base": cfg.lmcache_base,
        "disk_backend": disk_backend,
        "lmcache_io": cfg.lmcache_io,
        "server_log": str(server_log) if server_log else None,
        "gpu_prefix_reset_before_cold_and_warm": cfg.reset_gpu,
        "metrics_table": metrics_table,
        "phases": [asdict(x) for x in results],
        "delta": {
            "ttft_seconds_cold_minus_warm": (
                (cold.ttft_seconds - warm.ttft_seconds)
                if cold.ttft_seconds is not None and warm.ttft_seconds is not None
                else None
            ),
            "wall_seconds_cold_minus_warm": cold.wall_seconds - warm.wall_seconds,
        },
    }
    return results, summary, metrics_table


def main(argv: list[str] | None = None) -> int:
    recipe_root = _recipe_root()
    try:
        cfg = parse_args(argv)
        if cfg.iterations < 1:
            raise ValueError("--iterations must be >= 1")
        if cfg.server_log:
            print(f"{SCRIPT}: server_log={cfg.server_log}", file=sys.stderr)
        if cfg.iterations > 1:
            print(f"{SCRIPT}: iterations={cfg.iterations}", file=sys.stderr)

        run_summaries: list[dict[str, Any]] = []
        metrics_tables: list[list[dict[str, Any]]] = []
        for iteration in range(1, cfg.iterations + 1):
            if cfg.iterations > 1:
                print(
                    f"{SCRIPT}: --- iteration {iteration}/{cfg.iterations} ---",
                    file=sys.stderr,
                )
            _, summary, metrics_table = run_ab(
                cfg,
                recipe_root,
                cfg.server_log,
                iteration=iteration,
                total_iterations=cfg.iterations,
            )
            run_summaries.append(summary)
            metrics_tables.append(metrics_table)
            if iteration < cfg.iterations and cfg.pause_s > 0:
                time.sleep(cfg.pause_s)

        aggregate_table = aggregate_metrics_tables(metrics_tables)
        summary = combine_iteration_summaries(cfg, run_summaries, aggregate_table)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"{SCRIPT}: {e}", file=sys.stderr)
        return 1

    out = json.dumps(summary, indent=2)
    if cfg.json_stdout:
        print(out)
    table_text = format_iteration_tables(metrics_tables, aggregate_table)
    print(f"{SCRIPT} summary:\n{table_text}")
    if cfg.report:
        cfg.report.parent.mkdir(parents=True, exist_ok=True)
        cfg.report.write_text(out + "\n", encoding="utf-8")
        print(f"{SCRIPT}: wrote {cfg.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
