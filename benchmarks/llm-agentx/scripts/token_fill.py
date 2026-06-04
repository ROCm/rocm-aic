#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Token counting and filler text for CC trace replay (o200k_base proxy)."""

from __future__ import annotations

import functools
from typing import Any

try:
    import tiktoken
except ImportError as exc:
    raise SystemExit("error: tiktoken required (pip install tiktoken)") from exc

_FILLER_UNIT = " benchmark"


@functools.lru_cache(maxsize=1)
def _encoding():
    return tiktoken.get_encoding("o200k_base")


def count_text(text: str) -> int:
    return len(_encoding().encode(text))


def count_messages(messages: list[dict[str, Any]]) -> int:
    """Approximate chat prompt tokens for OpenAI-style messages."""
    total = 3
    for msg in messages:
        role = str(msg.get("role") or "")
        content = msg.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append(str(part.get("text") or ""))
            text = "\n".join(parts)
        else:
            text = str(content or "")
        total += count_text(role) + count_text(text) + 4
    return total


def make_filler(target_tokens: int) -> str:
    if target_tokens <= 0:
        return ""
    unit_tokens = max(count_text(_FILLER_UNIT), 1)
    repeats = max(1, (target_tokens // unit_tokens) + 1)
    text = _FILLER_UNIT * repeats
    tokens = _encoding().encode(text)
    if len(tokens) >= target_tokens:
        return _encoding().decode(tokens[:target_tokens])
    return text


def pad_messages_to_in(
    messages: list[dict[str, Any]],
    target_in: int,
) -> list[dict[str, Any]]:
    """Grow the last user turn (or add one) until messages reach target_in."""
    if target_in <= 0:
        return list(messages)

    out = [dict(m) for m in messages]
    current = count_messages(out)
    if current >= target_in:
        return out

    need = target_in - current
    filler = make_filler(need)

    if out and out[-1].get("role") == "user":
        prev = str(out[-1].get("content") or "")
        out[-1] = {"role": "user", "content": prev + filler}
    else:
        out.append({"role": "user", "content": filler})

    while count_messages(out) < target_in:
        extra = make_filler(target_in - count_messages(out))
        last = out[-1]
        if last.get("role") == "user":
            last["content"] = str(last.get("content") or "") + extra
        else:
            out.append({"role": "user", "content": extra})

    return out
