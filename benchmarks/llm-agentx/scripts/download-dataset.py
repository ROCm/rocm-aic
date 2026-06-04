#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Download SemiAnalysis CC traces from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit("error: PyYAML required (pip install PyYAML)") from exc

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:
    raise SystemExit("error: huggingface_hub required") from exc


def load_hf_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"error: invalid HF config: {path}")
    return data


def resolve_hf_cache(output: Path, override: Path | None) -> Path:
    if override is not None:
        base = override
    elif os.environ.get("AGENTX_HF_HOME"):
        base = Path(os.environ["AGENTX_HF_HOME"])
    elif os.environ.get("HF_HOME"):
        base = Path(os.environ["HF_HOME"])
    else:
        base = output / ".hf-cache"
    cache_dir = (base / "hub").resolve()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise SystemExit(
            f"error: cannot create HF cache {cache_dir}: {exc}\n"
            "  fix: export AGENTX_HF_HOME=$PWD/../../data/cc-traces/.hf-cache"
        ) from exc
    return cache_dir


def count_traces(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "cc-traces-hf.yaml",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="Destination root (default: repo data/cc-traces)",
    )
    p.add_argument(
        "--hf-cache",
        type=Path,
        help="HF hub cache root (default: <output>/.hf-cache/hub)",
    )
    args = p.parse_args()

    cfg = load_hf_config(args.config.resolve())
    repo_id = str(cfg["repo_id"])
    revision = str(cfg.get("revision") or "main")
    trace_file = str(cfg.get("trace_file") or "traces.jsonl")

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    output = args.output or (repo_root / "data" / "cc-traces")
    output = output.resolve()
    hf_cache = resolve_hf_cache(output, args.hf_cache)

    print(f"download-dataset: {repo_id} @ {revision} -> {output}")
    print(f"  hf cache: {hf_cache}")

    local = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=trace_file,
        revision=revision,
        cache_dir=str(hf_cache),
    )
    dest = output / trace_file
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local, dest)
    print(f"  {trace_file} -> {dest}")

    meta = {
        "repo_id": repo_id,
        "revision": revision,
        "trace_file": trace_file,
        "trace_count": count_traces(dest),
    }
    meta_path = output / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"  traces: {meta['trace_count']} -> {meta_path}")
    print(f"done: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
