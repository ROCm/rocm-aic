#!/usr/bin/env python3
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Probe an OpenAI-compatible vLLM HTTP endpoint."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="API root without /v1 suffix",
    )
    args = parser.parse_args()

    base = args.url.rstrip("/")
    req_url = f"{base}/v1/models"
    try:
        with urllib.request.urlopen(req_url, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        print(f"ERROR: cannot reach {req_url}: {exc}", file=sys.stderr)
        return 1

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        print(f"ERROR: non-JSON response from {req_url}", file=sys.stderr)
        return 1

    if not isinstance(payload, dict) or "data" not in payload:
        print(f"ERROR: unexpected payload from {req_url}", file=sys.stderr)
        return 1

    print(f"OK: {req_url} ({len(payload.get('data', []))} model(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
