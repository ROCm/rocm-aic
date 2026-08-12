# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Score one endpoint and print the result as JSON.

The two serving arms share a container name, a port, and the GPU, so they
cannot be up at the same time. The driver therefore scores the VRAM-only arm
with this helper, tears it down, brings up the tiered arm, and hands the
recorded number to pytest via ``AIC_ACCURACY_BASELINE_SCORE``.

Uses exactly the same ``_score`` path as the test module, so the two arms are
scored identically — a separate implementation here would be a way for the
differential to drift.

    python -m score_endpoint http://172.18.0.4:8000/v1 --out baseline.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from test_accuracy import LIMIT, MODEL, _score  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="base URL of the arm, e.g. http://IP:8000/v1")
    parser.add_argument("--out", help="write the JSON result here as well as stdout")
    args = parser.parse_args()

    score = _score(args.url.rstrip("/"))
    payload = {"url": args.url, "model": MODEL, "limit": LIMIT, "score": score}
    text = json.dumps(payload)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
