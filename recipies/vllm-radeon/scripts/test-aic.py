#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""LMCache AIC prefill A/B: populate NVMe, cold GPU compute, warm NVMe retrieve.

Requires: pip install -r requirements.txt from repo root (openai), LMCache #3008
(cache_salt),
VLLM_SERVER_DEV_MODE=1 (make run) for POST /reset_prefix_cache.

Phases (same context, question, cache_salt):
  populate — store KV to GdsBackend
  cold     — reset GPU prefix, bypass GdsBackend (GPU prefill)
  warm     — reset GPU prefix, retrieve from NVMe
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
LMCACHE_BACKEND = "GdsBackend"
SALT_PREFIX = "test-aic"


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


def _http(method: str, url: str, timeout: float) -> tuple[int, Any]:
    req = urllib.request.Request(url, method=method, headers={"Accept": "application/json"})
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
                "gds_bypassed": LMCACHE_BACKEND in pr.lmcache_bypassed_backends,
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


def parse_args(argv: list[str] | None = None) -> RunConfig:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-g", "--gpu", type=int, default=0, help="vLLM :800{gpu}, LMCache :699{gpu}+1")
    p.add_argument("-r", "--run-id", default="", help="cache_salt suffix (default: random 8 hex)")
    p.add_argument("--slug", default="war-and-peace", help="data/<slug>/ fixture")
    p.add_argument("--context", type=Path, help="context chunk (default: first in slug dir)")
    p.add_argument("--question", help="question (default: first in .questions.json)")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--skip-populate", action="store_true", help="NVMe already has this salt")
    p.add_argument("--no-reset", action="store_true", help="skip reset_prefix_cache before cold/warm")
    p.add_argument("--pause", type=float, default=2.0, help="seconds between phases")
    p.add_argument("-o", "--report", type=Path, help="write JSON summary")
    p.add_argument("--url", help="vLLM base URL (default: http://127.0.0.1:800{gpu})")
    p.add_argument(
        "--server-log",
        type=Path,
        help="vLLM server.txt for NVMe/AIC columns (default: <recipe>/logs/server.txt)",
    )
    p.add_argument(
        "--no-server-log",
        action="store_true",
        help="do not parse engine log for NVMe / external hit rate",
    )
    ns = p.parse_args(argv)
    run_id = (ns.run_id or uuid.uuid4().hex[:8]).strip()
    recipe_root = Path(__file__).resolve().parents[1]
    if ns.no_server_log:
        server_log: Path | None = None
    elif ns.server_log is not None:
        server_log = ns.server_log
    else:
        default_log = recipe_root / "logs" / "server.txt"
        server_log = default_log if default_log.is_file() else None
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
        timeout=600.0,
        vllm_url=ns.url,
        server_log=server_log,
    )


def run_ab(
    cfg: RunConfig,
    recipe_root: Path,
    server_log: Path | None,
) -> tuple[list[PhaseResult], dict[str, Any]]:
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

    print(
        f"{SCRIPT}: context={context_path} ({len(context)} bytes) "
        f"salt={cfg.cache_salt!r} reset={cfg.reset_gpu}",
        file=sys.stderr,
    )
    print(
        f"{SCRIPT}: vllm={cfg.vllm_base} lmcache={cfg.lmcache_base}",
        file=sys.stderr,
    )

    results: list[PhaseResult] = []
    log_tail = ServerLogTail(server_log)
    phases: list[tuple[str, bool, bool]] = []
    if not cfg.skip_populate:
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
                bypass_set(cfg.lmcache_base, LMCACHE_BACKEND, True, cfg.timeout)
            elif LMCACHE_BACKEND in bypass_list(cfg.lmcache_base, cfg.timeout):
                bypass_set(cfg.lmcache_base, LMCACHE_BACKEND, False, cfg.timeout)

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
            bypass_set(cfg.lmcache_base, LMCACHE_BACKEND, False, cfg.timeout)
        except RuntimeError:
            pass

    cold = next(x for x in results if x.phase == "cold")
    warm = next(x for x in results if x.phase == "warm")
    metrics_table = build_metrics_table(results, cfg.max_tokens)
    summary: dict[str, Any] = {
        "run_id": cfg.run_id,
        "cache_salt": cfg.cache_salt,
        "context_file": str(context_path.resolve()),
        "question": question,
        "model": cfg.model,
        "vllm_base": cfg.vllm_base,
        "lmcache_base": cfg.lmcache_base,
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
    recipe_root = Path(__file__).resolve().parents[1]
    try:
        cfg = parse_args(argv)
        if cfg.server_log:
            print(f"{SCRIPT}: server_log={cfg.server_log}", file=sys.stderr)
        _, summary, metrics_table = run_ab(cfg, recipe_root, cfg.server_log)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"{SCRIPT}: {e}", file=sys.stderr)
        return 1

    out = json.dumps(summary, indent=2)
    print(out)
    table_text = format_metrics_table(metrics_table)
    print(f"\n{SCRIPT} summary:\n{table_text}")
    if cfg.report:
        cfg.report.parent.mkdir(parents=True, exist_ok=True)
        cfg.report.write_text(out + "\n", encoding="utf-8")
        print(f"{SCRIPT}: wrote {cfg.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
