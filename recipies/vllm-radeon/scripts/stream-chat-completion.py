#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Stream a vLLM chat completion; write response JSON plus client timings."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def _load_payload(args: argparse.Namespace) -> dict:
    if args.data_file:
        raw = args.data_file.read()
    elif args.data:
        raw = args.data
    else:
        raise SystemExit("error: --data or --data-file required")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit("error: payload must be a JSON object")
    return payload


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
    base_url: str, payload: dict
) -> tuple[int, dict, float, float | None]:
    body = dict(payload)
    body["stream"] = True
    body.setdefault("stream_options", {"include_usage": True})

    url = base_url.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    ttft: float | None = None
    content_parts: list[str] = []
    completion: dict | None = None
    usage: dict | None = None

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
                if choice0.get("finish_reason") is not None:
                    completion = chunk
    except urllib.error.HTTPError as exc:
        http_status = exc.code
        err_body = exc.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(err_body)
        except json.JSONDecodeError:
            err_json = {"error": err_body}
        wall = time.perf_counter() - t0
        out = {
            "http_status": http_status,
            "client_wall_time_seconds": wall,
            "client_ttft_seconds": ttft,
            "error": err_json,
        }
        return http_status, out, wall, ttft

    wall = time.perf_counter() - t0
    if completion is None:
        completion = {
            "id": None,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(content_parts)},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
    elif content_parts:
        msg = completion.get("choices", [{}])[0].get("message", {})
        if isinstance(msg, dict) and not msg.get("content"):
            msg["content"] = "".join(content_parts)
    if usage and isinstance(completion, dict):
        completion["usage"] = usage
    completion["http_status"] = http_status
    completion["client_wall_time_seconds"] = wall
    completion["client_ttft_seconds"] = ttft
    return http_status, completion, wall, ttft


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True, help="Server base URL (no /v1 suffix)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--data", help="JSON request body")
    g.add_argument("--data-file", type=argparse.FileType("r", encoding="utf-8"))
    ap.add_argument("-o", "--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    payload = _load_payload(args)
    try:
        status, doc, _wall, _ttft = stream_completion(args.url, payload)
    except urllib.error.URLError as exc:
        print(f"error: request failed: {exc}", file=sys.stderr)
        return 1

    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")

    if status < 200 or status >= 300:
        print(f"HTTP {status} (client_wall_time_seconds={doc.get('client_wall_time_seconds')})",
              file=sys.stderr)
        if "error" in doc:
            json.dump(doc["error"], sys.stderr, indent=2)
            sys.stderr.write("\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
