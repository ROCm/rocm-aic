#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Replay SemiAnalysis CC agent traces against an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from token_fill import count_messages, pad_messages_to_in  # noqa: E402


def _parse_sse_line(line: str) -> dict | None:
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return None
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        return None
    return chunk if isinstance(chunk, dict) else None


def stream_completion(
    base_url: str, payload: dict, api_key: str | None
) -> tuple[int, dict[str, Any], float, float | None]:
    body = dict(payload)
    body["stream"] = True
    body.setdefault("stream_options", {"include_usage": True})

    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    t0 = time.perf_counter()
    ttft: float | None = None
    content_parts: list[str] = []
    usage: dict | None = None
    http_status = 0

    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            http_status = resp.status
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace")
                chunk = _parse_sse_line(line)
                if not chunk:
                    continue
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice0 = choices[0]
                if not isinstance(choice0, dict):
                    continue
                delta = choice0.get("delta")
                if isinstance(delta, dict):
                    piece = delta.get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        content_parts.append(piece)
    except urllib.error.HTTPError as exc:
        http_status = exc.code
        err_body = exc.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(err_body)
        except json.JSONDecodeError:
            err_json = {"error": err_body}
        wall = time.perf_counter() - t0
        return http_status, {
            "http_status": http_status,
            "client_wall_time_seconds": wall,
            "client_ttft_seconds": ttft,
            "error": err_json,
            "content": "".join(content_parts),
        }, wall, ttft
    except urllib.error.URLError as exc:
        wall = time.perf_counter() - t0
        return 0, {
            "http_status": 0,
            "client_wall_time_seconds": wall,
            "client_ttft_seconds": None,
            "error": {"message": str(exc)},
            "content": "",
        }, wall, None

    wall = time.perf_counter() - t0
    return http_status, {
        "http_status": http_status,
        "client_wall_time_seconds": wall,
        "client_ttft_seconds": ttft,
        "content": "".join(content_parts),
        "usage": usage,
    }, wall, ttft


def fetch_models_body(base_url: str) -> dict[str, Any] | None:
    url = base_url.rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        return None
    return body if isinstance(body, dict) else None


def fetch_served_models(base_url: str) -> list[str] | None:
    body = fetch_models_body(base_url)
    if body is None:
        return None
    data = body.get("data")
    if not isinstance(data, list):
        return None
    ids: list[str] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.append(item["id"])
    return ids


def fetch_model_max_context(base_url: str, model: str) -> int | None:
    body = fetch_models_body(base_url)
    if body is None:
        return None
    data = body.get("data")
    if not isinstance(data, list):
        return None
    for item in data:
        if not isinstance(item, dict) or item.get("id") != model:
            continue
        raw = item.get("max_model_len")
        if isinstance(raw, int) and raw > 0:
            return raw
    return None


def validate_model(base_url: str, model: str) -> bool:
    body = fetch_models_body(base_url)
    if body is None:
        print(
            f"error: server not reachable at {base_url.rstrip('/')}/v1/models",
            file=sys.stderr,
        )
        return False
    served = fetch_served_models(base_url)
    if served is None or model not in served:
        print(f"error: MODEL={model!r} is not served at {base_url}", file=sys.stderr)
        print(f"  served models: {', '.join(served) if served else '(none)'}", file=sys.stderr)
        return False
    max_ctx = fetch_model_max_context(base_url, model)
    if max_ctx is not None:
        print(f"  server max_context: {max_ctx} tokens", file=sys.stderr)
    return True


def completion_budget(trace_out: int, max_tokens_cap: int) -> int:
    if trace_out > 0:
        return min(trace_out, max_tokens_cap)
    return max_tokens_cap


def exceeds_max_context(
    trace_in: int, trace_out: int, max_tokens_cap: int, max_context: int | None
) -> bool:
    if max_context is None:
        return False
    return trace_in + completion_budget(trace_out, max_tokens_cap) > max_context


def load_traces(path: Path) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                traces.append(obj)
    return traces


def iter_model_requests(requests: list[Any]) -> Iterator[tuple[dict[str, Any], bool]]:
    """Yield (request, is_subagent_inner) in trace order."""
    for req in requests:
        if not isinstance(req, dict):
            continue
        req_type = req.get("type")
        if req_type == "subagent":
            inner = req.get("requests")
            if isinstance(inner, list):
                for item in inner:
                    if isinstance(item, dict) and item.get("type") in ("s", "n"):
                        yield item, True
        elif req_type in ("s", "n"):
            yield req, False


def replay_request(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    req: dict[str, Any],
    max_tokens_cap: int,
    max_context: int | None,
    skip_oversized: bool,
    api_key: str | None,
    dry_run: bool,
    honor_think_time: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_in = int(req.get("in") or 0)
    trace_out = int(req.get("out") or 0)
    max_tokens = completion_budget(trace_out, max_tokens_cap)

    row: dict[str, Any] = {
        "trace_request_type": req.get("type"),
        "trace_model": req.get("model"),
        "trace_in": trace_in,
        "trace_out": trace_out,
        "trace_hash_ids": req.get("hash_ids"),
        "trace_t_seconds": req.get("t"),
        "model": model,
    }

    if skip_oversized and exceeds_max_context(
        trace_in, trace_out, max_tokens_cap, max_context
    ):
        row.update(
            {
                "skipped": True,
                "skip_reason": "exceeds_max_context",
                "server_max_context": max_context,
                "required_tokens": trace_in + max_tokens,
            }
        )
        return messages, row

    payload_messages = pad_messages_to_in(messages, trace_in)
    approx_in = count_messages(payload_messages)
    row["approx_prompt_tokens"] = approx_in

    if honor_think_time:
        think = req.get("think_time")
        if isinstance(think, (int, float)) and think > 0:
            time.sleep(float(think))

    if dry_run:
        row.update(
            {
                "http_status": 200,
                "client_ttft_seconds": float(req.get("ttft") or 0.0),
                "client_wall_time_seconds": float(req.get("api_time") or 0.0),
                "dry_run": True,
            }
        )
        next_messages = list(payload_messages)
        next_messages.append(
            {"role": "assistant", "content": f"[dry-run output len={trace_out}]"}
        )
        return next_messages, row

    payload = {
        "model": model,
        "messages": payload_messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    status, result, _wall, _ttft = stream_completion(base_url, payload, api_key)
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    row.update(
        {
            "http_status": status,
            "client_ttft_seconds": result.get("client_ttft_seconds"),
            "client_wall_time_seconds": result.get("client_wall_time_seconds"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "assistant_content_len": len(result.get("content") or ""),
        }
    )
    if "error" in result:
        row["error"] = result["error"]

    next_messages = list(payload_messages)
    content = result.get("content") or ""
    if content:
        next_messages.append({"role": "assistant", "content": content})
    return next_messages, row


def replay_trace(
    *,
    base_url: str,
    model: str,
    trace: dict[str, Any],
    max_tokens_cap: int,
    max_context: int | None,
    skip_oversized: bool,
    max_requests: int | None,
    api_key: str | None,
    dry_run: bool,
    honor_think_time: bool,
) -> list[dict[str, Any]]:
    trace_id = str(trace.get("id") or "")
    requests = trace.get("requests")
    if not isinstance(requests, list):
        requests = []

    rows: list[dict[str, Any]] = []
    main_messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a coding assistant. Reply concisely.",
        }
    ]
    sub_messages: list[dict[str, Any]] = []

    for idx, (req, is_subagent) in enumerate(iter_model_requests(requests)):
        if max_requests is not None and idx >= max_requests:
            break

        if is_subagent:
            if not sub_messages:
                sub_messages = [
                    {
                        "role": "system",
                        "content": "You are a sub-agent coding assistant.",
                    }
                ]
            sub_messages, row = replay_request(
                base_url=base_url,
                model=model,
                messages=sub_messages,
                req=req,
                max_tokens_cap=max_tokens_cap,
                max_context=max_context,
                skip_oversized=skip_oversized,
                api_key=api_key,
                dry_run=dry_run,
                honor_think_time=honor_think_time,
            )
        else:
            main_messages, row = replay_request(
                base_url=base_url,
                model=model,
                messages=main_messages,
                req=req,
                max_tokens_cap=max_tokens_cap,
                max_context=max_context,
                skip_oversized=skip_oversized,
                api_key=api_key,
                dry_run=dry_run,
                honor_think_time=honor_think_time,
            )
            sub_messages = []

        row["trace_id"] = trace_id
        row["request_index"] = idx
        row["is_subagent_inner"] = is_subagent
        rows.append(row)

        if row.get("skipped"):
            continue

        status = row.get("http_status")
        if not dry_run and (status is None or int(status) < 200 or int(status) >= 300):
            break

    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True, help="Server base URL (no /v1 suffix)")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--traces-file", type=Path, help="Default: <data-root>/traces.jsonl")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--trace-id", help="Single trace id")
    p.add_argument("--seed", type=int, help="Random trace selection seed")
    p.add_argument("--count", type=int, default=1, help="Random traces when --trace-id unset")
    p.add_argument("--max-tokens", type=int, default=512, help="Cap completion tokens")
    p.add_argument(
        "--max-requests",
        type=int,
        help="Cap model requests replayed per trace",
    )
    p.add_argument("--api-key", default="dummy-key")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-o", "--output", type=Path, help="Write JSONL results")
    p.add_argument("--skip-health-check", action="store_true")
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Verify BASE_URL and MODEL, then exit",
    )
    p.add_argument(
        "--honor-think-time",
        action="store_true",
        help="Sleep trace think_time between requests",
    )
    p.add_argument(
        "--max-context",
        type=int,
        help="Server context limit; default: read max_model_len from /v1/models",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail on oversized requests instead of skipping them",
    )
    args = p.parse_args()

    skip_oversized = not args.strict

    if args.check_only:
        if args.dry_run:
            print("error: --check-only cannot be used with --dry-run", file=sys.stderr)
            return 1
        return 0 if validate_model(args.url, args.model) else 1

    data_root = args.data_root.resolve()
    traces_path = args.traces_file or (data_root / "traces.jsonl")
    if not traces_path.is_file():
        print(f"error: missing {traces_path}", file=sys.stderr)
        return 1

    traces = load_traces(traces_path)
    if not traces:
        print(f"error: no traces in {traces_path}", file=sys.stderr)
        return 1

    if not args.dry_run and not args.skip_health_check:
        if not validate_model(args.url, args.model):
            return 1

    max_context = args.max_context
    if max_context is None and not args.dry_run:
        max_context = fetch_model_max_context(args.url, args.model)
    if skip_oversized and max_context is not None:
        print(
            f"replay: max_context={max_context} (oversized trace requests skipped)",
            file=sys.stderr,
        )
    elif skip_oversized and max_context is None:
        print(
            "replay: warning: max_context unknown; oversized requests may 400",
            file=sys.stderr,
        )

    if args.trace_id:
        selected = [t for t in traces if str(t.get("id")) == args.trace_id]
        if not selected:
            print(f"error: trace id not found: {args.trace_id}", file=sys.stderr)
            return 1
    else:
        rng = random.Random(args.seed)
        n = min(args.count, len(traces))
        selected = rng.sample(traces, n)

    all_rows: list[dict[str, Any]] = []
    for trace in selected:
        rows = replay_trace(
            base_url=args.url,
            model=args.model,
            trace=trace,
            max_tokens_cap=args.max_tokens,
            max_context=max_context,
            skip_oversized=skip_oversized,
            max_requests=args.max_requests,
            api_key=args.api_key,
            dry_run=args.dry_run,
            honor_think_time=args.honor_think_time,
        )
        all_rows.extend(rows)

    skipped = sum(1 for r in all_rows if r.get("skipped"))
    if skipped:
        print(f"replay: skipped {skipped} request(s) over max_context", file=sys.stderr)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            for row in all_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(all_rows)} rows -> {args.output}", file=sys.stderr)
    else:
        for row in all_rows:
            print(json.dumps(row, ensure_ascii=False))

    bad = [
        r
        for r in all_rows
        if not r.get("skipped")
        and r.get("http_status") not in (200, None)
        and not r.get("dry_run")
    ]
    if bad:
        for row in bad[:3]:
            err = row.get("error")
            task_id = row.get("trace_id")
            step = row.get("request_index")
            status = row.get("http_status")
            if isinstance(err, dict):
                msg = err.get("message") or err.get("error") or json.dumps(err)
            else:
                msg = str(err) if err is not None else f"HTTP {status}"
            print(f"error: trace {task_id} request {step}: {msg}", file=sys.stderr)
        if len(bad) > 3:
            print(f"error: {len(bad) - 3} more failed request(s)", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
