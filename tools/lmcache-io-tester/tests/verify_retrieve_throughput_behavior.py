#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Verify retrieve-only conventions and ``--rate`` throttling.

Run from ``tools/lmcache-io-tester`` (same as CI).

1. ``run-this.sh`` lines mentioning ``retrieve-only`` or ``lookup-only`` must
   not include ``--rate`` (unthrottled max throughput) or ``--fs-odirect``.
2. ``lmcache-sim run`` with ``--rate`` reports lower ``throughput_ops_per_sec``
   than unthrottled retrieve for the same pattern.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent


def _parse_json_stdout(stdout: str) -> dict:
    """Parse the first top-level JSON object (``json.dumps(..., indent=2)``)."""
    i = stdout.find("{")
    if i < 0:
        raise ValueError(f"no JSON object in stdout: {stdout[:500]!r}")
    obj, _ = json.JSONDecoder().raw_decode(stdout[i:])
    return obj


def _run_sim(
    argv: list[str],
    *,
    cwd: Path,
    py: str,
    expect_json: bool = False,
) -> dict:
    proc = subprocess.run(
        [py, "-m", "src.lmcache-sim", *argv],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.stderr.write(proc.stdout)
        raise RuntimeError(
            f"lmcache-sim failed ({proc.returncode}): "
            f"{' '.join(argv)}"
        )
    out = proc.stdout
    if expect_json:
        if not out.strip():
            raise RuntimeError("empty stdout from lmcache-sim json run")
        return _parse_json_stdout(out)
    return {}


def _check_run_this_sh(root: Path) -> None:
    run_this = root / "run-this.sh"
    if not run_this.is_file():
        return
    for i, line in enumerate(
        run_this.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if "retrieve-only" not in line and "lookup-only" not in line:
            continue
        if line.strip().startswith("#"):
            continue
        if "--rate" in line:
            raise AssertionError(
                f"{run_this}:{i}: lookup/retrieve lines should omit "
                f"--rate for max throughput; found: {line.strip()!r}"
            )
        if "--fs-odirect" in line:
            raise AssertionError(
                f"{run_this}:{i}: lookup/retrieve lines should omit "
                f"--fs-odirect; found: {line.strip()!r}"
            )


def _check_rate_caps_throughput(root: Path, py: str) -> None:
    tmp = Path(
        tempfile.mkdtemp(
            prefix="lmcache-io-verify-",
            dir=os.environ.get("TMPDIR") or None,
        )
    )
    cfg = str(root / "configs" / "lmcache-config.yml")
    base = [
        "run",
        "--storage-type",
        "filesystem",
        "--storage-path",
        str(tmp),
        "--device",
        "cpu",
        "--config",
        cfg,
    ]
    _run_sim(
        base
        + [
            "--pattern",
            "store-only",
            "--num-operations",
            "48",
        ],
        cwd=root,
        py=py,
        expect_json=False,
    )
    fast = _run_sim(
        base
        + [
            "--pattern",
            "retrieve-only",
            "--num-operations",
            "96",
            "--output-format",
            "json",
        ],
        cwd=root,
        py=py,
        expect_json=True,
    )
    slow = _run_sim(
        base
        + [
            "--pattern",
            "retrieve-only",
            "--num-operations",
            "40",
            "--rate",
            "20",
            "--output-format",
            "json",
        ],
        cwd=root,
        py=py,
        expect_json=True,
    )
    tp_fast = float(fast["throughput_ops_per_sec"])
    tp_slow = float(slow["throughput_ops_per_sec"])
    if tp_slow > 35.0:
        raise AssertionError(
            f"expected --rate 20 to cap throughput (~20 IOPS); "
            f"got {tp_slow:.2f}"
        )
    if tp_fast <= tp_slow * 1.15:
        raise AssertionError(
            f"expected unthrottled throughput ({tp_fast:.2f}) "
            f"well above throttled ({tp_slow:.2f})"
        )


def main() -> int:
    root = _repo_root()
    py = sys.executable
    _check_run_this_sh(root)
    _check_rate_caps_throughput(root, py)
    print(
        "verify_retrieve_throughput_behavior: OK "
        "(run-this.sh conventions + --rate cap)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
