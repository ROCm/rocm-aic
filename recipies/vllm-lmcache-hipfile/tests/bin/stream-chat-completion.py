#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Mock stream-chat-completion for run-long CI (no live vLLM server)."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--data")
    g.add_argument("--data-file", type=argparse.FileType("r", encoding="utf-8"))
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    if args.data_file is not None:
        args.data_file.read()

    doc = {
        "id": "ci-mock",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
        "http_status": 200,
        "client_wall_time_seconds": 0.001,
        "client_ttft_seconds": 0.0005,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
