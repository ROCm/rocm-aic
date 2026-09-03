#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Validate the emulator profile packs committed under profiles/.

Run from CI (see .github/workflows/aic-patches.yml) with the patched LLM-Emu
plugin installed, so the packs are checked against the emulator's OWN loader and
oracle rather than a re-implementation of the schema that can drift from it.

Checks, each of which corresponds to a way we have actually broken this:

  1. Every pack loads through vllm_emulator.profile.loader (schema valid) and
     an oracle can be constructed from it and asked for a latency.
  2. Every pack has a sibling <pack>.capture.txt -- a pack whose provenance is
     unknown cannot be trusted or reproduced.
  3. Every pack is self-describing: GPU identity, the KV pool measured on that
     GPU, and the serving configuration it is only valid for.
  4. The pack that docker-compose.yml defaults to actually exists here.  The
     first version of the emulate service pointed at /profiles/profile.json,
     which never existed; the hook then silently stayed disabled and vLLM tried
     a real forward pass.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROFILES = REPO / "profiles"
COMPOSE = REPO / "docker" / "docker-compose.yml"

# Fields that make a pack self-describing.  Missing ones are not fatal on their
# own (older packs predate them) but they are reported, and the default pack is
# held to the full bar.
GPU_FIELDS = ("gpu_name", "gpu_memory_bytes", "gpu_sm_count",
              "available_kv_cache_bytes")
CONFIG_FIELDS = ("max_num_batched_tokens", "max_num_seqs",
                 "enable_prefix_caching", "enable_chunked_prefill",
                 "kv_cache_dtype", "async_scheduling", "vllm_version")

failures: list[str] = []
warnings: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"  FAIL {msg}")


def warn(msg: str) -> None:
    warnings.append(msg)
    print(f"  WARN {msg}")


def compose_default_pack() -> str | None:
    """The pack path the emulate compose service falls back to."""
    text = COMPOSE.read_text(encoding="utf-8")
    m = re.search(r"VLLM_EMULATOR_PROFILE_PACK=\$\{VLLM_EMULATOR_PROFILE_PACK:-([^}]+)\}",
                  text)
    return m.group(1) if m else None


def check_pack(path: Path) -> None:
    print(f"== {path.name}")
    from vllm_emulator.oracle import create_oracle_from_profile_pack
    from vllm_emulator.profile.loader import load_profile_pack

    try:
        pack = load_profile_pack(path)
    except Exception as exc:  # noqa: BLE001 - report any loader rejection
        fail(f"{path.name}: does not load: {exc}")
        return

    gpu_model = pack.get("gpu_model")
    if not gpu_model or gpu_model == "unknown":
        fail(f"{path.name}: gpu_model is {gpu_model!r}")
    if not pack.get("model_name"):
        fail(f"{path.name}: model_name is missing")

    cells = {k: len(pack.get(k, []))
             for k in ("step_cycle_2d_distribution", "step_cycle_3d_distribution")}
    if not cells["step_cycle_2d_distribution"]:
        fail(f"{path.name}: no step_cycle_2d_distribution cells")
    samples = sum(c["num_samples"] for c in pack.get("step_cycle_2d_distribution", []))

    model_config = pack.get("model_config", {})
    gpu = model_config.get("gpu", {})
    missing_gpu = [f for f in GPU_FIELDS if not gpu.get(f)]
    missing_cfg = [f for f in CONFIG_FIELDS if f not in model_config]

    if not path.with_suffix(".capture.txt").exists():
        fail(f"{path.name}: no sibling {path.stem}.capture.txt (provenance)")

    # The oracle is the real consumer: make sure it can be built and queried.
    try:
        oracle = create_oracle_from_profile_pack(pack)
        latency = oracle.estimate_step_latency_us(
            16, has_prefill=False, num_requests=4, num_new_reqs=0, sum_kv=4096)
    except Exception as exc:  # noqa: BLE001
        fail(f"{path.name}: oracle rejected the pack: {exc}")
        return
    if not latency or latency <= 0:
        fail(f"{path.name}: oracle returned {latency} us for a populated query")

    print(f"  gpu={gpu_model} model={pack.get('model_name')} "
          f"cells={cells['step_cycle_2d_distribution']} "
          f"(3d {cells['step_cycle_3d_distribution']}) samples={samples} "
          f"query->{latency:.0f}us")
    if missing_gpu:
        warn(f"{path.name}: GPU metadata missing {missing_gpu} "
             f"(a pack without available_kv_cache_bytes makes the emulator "
             f"ESTIMATE the KV pool, moving its admission point)")
    if missing_cfg:
        warn(f"{path.name}: serving config missing {missing_cfg} "
             f"(re-capture with a current tracer to make the pack self-describing)")
    return


def main() -> int:
    packs = sorted(PROFILES.glob("*.json"))
    if not packs:
        print(f"FAIL: no profile packs in {PROFILES}")
        return 1
    for pack in packs:
        check_pack(pack)

    print("== compose default")
    default = compose_default_pack()
    if not default:
        fail("could not find VLLM_EMULATOR_PROFILE_PACK default in docker-compose.yml")
    else:
        name = Path(default).name
        if (PROFILES / name).exists():
            print(f"  OK  compose defaults to {default}, which is shipped in profiles/")
        else:
            fail(f"compose defaults to {default}, but profiles/{name} does not exist")

    print()
    if failures:
        print(f"{len(failures)} failure(s), {len(warnings)} warning(s)")
        return 1
    print(f"All {len(packs)} profile pack(s) OK ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
