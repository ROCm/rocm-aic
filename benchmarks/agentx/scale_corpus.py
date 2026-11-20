#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Write a token-scaled copy of a Weka trace corpus as a local weka-trace directory.

The public AgentX corpus needs a large context window (its smallest trace
peaks above 44k tokens), so it cannot be replayed on a small-context model.
This divides every token count of every trace by an integer divisor and writes
one `<trace id>.json` per trace, the layout `aiperf profile
--custom-dataset-type weka-trace --input-file <dir>` reads.

What is kept: the request order, the recorded timing (timestamps, think and
API times), the subagent fan-out, and the prefix-reuse shape. `hash_ids` name
one block of `block_size` tokens each, so they are cut to the number of blocks
the scaled input needs; a turn's blocks stay a prefix of the next turn's.

What is not kept: the token counts, and with them the context sizes, the KV
cache footprint and the load on the server. A scaled corpus is NOT the AgentX
benchmark and its results are not comparable with published AgentX or
InferenceX numbers.

Runs inside the aic-agentx image (it needs `aiperf` and `datasets`), as
benchmarks/agentx/scaled_corpus.sh does:

    python3 scale_corpus.py --dataset semianalysisai/cc-traces-weka-062126 \\
        --divisor 8 --output /out/cc-traces-weka-062126-scaled-8

The output depends only on the dataset revision and the divisor.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

# aiperf has no public API for either: the models are the corpus schema, and
# the peak function is the rule --max-context-length applies, so the summary
# below counts exactly the traces aiperf would keep.
from aiperf.dataset.loader.weka_trace import _trace_peak_context_length
from aiperf.dataset.loader.weka_trace_models import WekaSubagentEntry, WekaTrace
from datasets import load_dataset

# Not *.json: aiperf loads every *.json in the directory as a trace.
MANIFEST = "SCALED-CORPUS.txt"
CAVEAT = (
    "Token counts are divided by {divisor}: this is an AgentX-shaped workload, NOT the AgentX "
    "benchmark. Results are not comparable with published AgentX or InferenceX numbers."
)


def _scale_request(request, divisor: int, block_size: int) -> None:
    request.input_length = max(1, request.input_length // divisor)
    if request.output_length:
        request.output_length = max(1, request.output_length // divisor)
    blocks = max(1, math.ceil(request.input_length / block_size))
    request.hash_ids = request.hash_ids[:blocks]


def scale_trace(trace: WekaTrace, divisor: int) -> int:
    """Scale one trace in place; return its request count."""
    count = 0
    trace.tool_tokens //= divisor
    trace.system_tokens //= divisor
    for entry in trace.requests:
        if isinstance(entry, WekaSubagentEntry):
            entry.tool_tokens //= divisor
            entry.system_tokens //= divisor
            for request in entry.requests:
                _scale_request(request, divisor, trace.block_size)
                count += 1
        else:
            _scale_request(entry, divisor, trace.block_size)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset repo of Weka traces")
    parser.add_argument("--revision", default=None, help="dataset revision (default: the repo's current one)")
    parser.add_argument("--divisor", required=True, type=int, help="divide every token count by this integer")
    parser.add_argument("--output", required=True, type=Path, help="directory to write; must be empty or absent")
    parser.add_argument(
        "--limits",
        default="8192,16384,32768,65536,131072",
        help="context limits to count fitting traces for (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.divisor < 2:
        parser.error("--divisor must be an integer >= 2")
    limits = sorted(int(limit) for limit in args.limits.split(",") if limit)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error(f"{args.output} is not empty")

    dataset = load_dataset(args.dataset, split="train", revision=args.revision)
    # `datasets` keeps a Hub dataset under a directory named after its commit.
    commits = {match for item in dataset.cache_files for match in re.findall(r"/([0-9a-f]{40})/", item["filename"])}
    revision = args.revision or (commits.pop() if len(commits) == 1 else "unknown")

    peaks: list[int] = []
    requests = 0
    seen: set[str] = set()
    for row in dataset:
        trace = WekaTrace.model_validate(row)
        if trace.id in seen:
            raise SystemExit(f"duplicate trace id {trace.id!r} in {args.dataset}")
        seen.add(trace.id)
        requests += scale_trace(trace, args.divisor)
        peaks.append(_trace_peak_context_length(trace))
        text = json.dumps(trace.model_dump(by_alias=True, exclude_none=True), sort_keys=True, separators=(",", ":"))
        (args.output / f"{trace.id}.json").write_text(text + "\n", encoding="utf-8")

    if not peaks:
        raise SystemExit(f"{args.dataset} holds no traces")
    peaks.sort()
    lines = [
        CAVEAT.format(divisor=args.divisor),
        "",
        f"source dataset: {args.dataset}",
        f"source revision: {revision}",
        f"divisor: {args.divisor}",
        f"traces: {len(peaks)}",
        f"requests: {requests}",
        f"peak context (prompt + output tokens) min / median / max: {peaks[0]} / {peaks[len(peaks) // 2]} / {peaks[-1]}",
    ]
    lines += [f"traces that fit a context of {limit}: {sum(peak <= limit for peak in peaks)}" for limit in limits]
    # Written last: its presence marks the directory as complete.
    (args.output / MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
