# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Regression tests for the nightly cliff page generator."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import cliff_to_html

_FIELDS = ["arm", "concurrency", "iter", "throughput_tok_s_total",
           "p50_latency_s", "p95_latency_s", "l1_hit_rate_pct", "ext_hit_rate_pct"]
_STAMP = "20261109-233656"
_TAG = ("0.1.0-rocm7.14.1-pytorch2.13-vllm0.29.0-aiter0.1.22.post1"
        "-fa2.8.3.post1-lmcache0.5.6-nixl1.4.1-hsasnoop1.2.0")


def _write_csv(path: Path, arm: str, thr: float, l1: str, ext: str) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_FIELDS)
        w.writerow([arm, 1, 0, thr, 0.5, 0.5, l1, ext])
    # run_cliff.py's arguments, as it writes them beside the CSV.
    path.with_suffix(".json").write_text(json.dumps({
        "arm": arm, "model": "openai/gpt-oss-120b", "isl": 20000,
        "shared_prefix_tokens": 18000, "max_tokens": 1, "prefix_mode": "per_client",
        "concurrencies": "1,2,250", "iters": 3, "warmup_at_each_c": True,
        "warmup_waves_per_c": 2, "reset_prefix_cache": arm != "vram_only",
    }))


def _vllm(util: str) -> dict:
    return {"image": f"aic-vllm:{_TAG}", "image_id": "sha256:ee34b164750f547a",
            "args": ["--model", "openai/gpt-oss-120b", "--tensor-parallel-size", "1",
                     "--gpu-memory-utilization", util, "--max-model-len", "32768",
                     "--kv-cache-dtype", "fp8"]}


def _lmcache(*args: str) -> dict:
    nvme = {"fstype": "xfs", "source": "/dev/mapper/vg", "nvme_disks": "8"}
    return {"image": f"aic-lmcache:{_TAG}", "image_id": "sha256:5dbc823baf66",
            "process": {"argv": ["/usr/bin/python3", "/usr/local/bin/lmcache", "server",
                                 *args], "config_file": None},
            "storage": {"/data/nvme": nvme, "/data/slab": nvme}}


_POSIX_L2 = json.dumps({
    "type": "nixl_store", "backend": "POSIX", "pool_size": 32768,
    "backend_params": {"file_path": "/data/nvme/lmcache", "use_direct_io": "false",
                       "file_size": "33554432"}})

# run-config.json the way run-cliff.sbatch records it.
_RUN_CONFIG = {
    "gpu": {"name": "AMD Instinct MI355X", "arch": "gfx950", "vram_bytes": 309220868096},
    "arms": {
        "vram_only": {"vllm": _vllm("0.40")},
        "kvd_v2-nvme": {"vllm": _vllm("0.40"),
                        "lmcache": _lmcache("--l1-size-gb", "160", "--l2-adapter", _POSIX_L2)},
        "kvd_v2-gds": {"vllm": _vllm("0.40"),
                       "lmcache": _lmcache("--l1-size-gb", "256", "--gds-l1-path", "/data/slab")},
    },
}


def _make_run(root: Path, config: dict | None = _RUN_CONFIG) -> Path:
    """A run dir laid out the way run-cliff.sbatch leaves it, all three arms."""
    results = root / "results"
    results.mkdir(parents=True)
    _write_csv(results / f"cliff-vram_only-{_STAMP}.csv", "vram_only", 25366.0, "29.6", "")
    # Both kvd arms carry arm=kvd_v2; only the file name tells them apart.
    _write_csv(results / f"cliff-kvd_v2-nvme-{_STAMP}.csv", "kvd_v2", 100983.0, "0.0", "89.6")
    _write_csv(results / f"cliff-kvd_v2-gds-{_STAMP}.csv", "kvd_v2", 48264.0, "0.0", "89.6")
    for sub in ("nvme", "gds"):
        d = root / "plots" / sub
        d.mkdir(parents=True)
        for name in ("cliff-throughput.png", "cliff-latency-p50.png",
                     "cliff-latency-p95.png", "cliff-hit-rate.png"):
            (d / name).write_bytes(f"{sub}-{name}".encode())
    if config is not None:
        (root / "run-config.json").write_text(json.dumps(config))
    return root


class CliffToHtmlTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.run = cliff_to_html._discover_run(_make_run(Path(self._tmp.name)))
        self.html = cliff_to_html._build_html(
            [self.run], "35c701e58f53", "2026-11-09", "https://github.com/ROCm/rocm-aic")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_both_kvd_backends_are_kept_apart(self) -> None:
        self.assertEqual(self.run["arms"]["kvd_v2-nvme"][1]["thr"], 100983.0)
        self.assertEqual(self.run["arms"]["kvd_v2-gds"][1]["thr"], 48264.0)

    def test_page_reports_each_backend_under_its_own_label(self) -> None:
        nvme = cliff_to_html._ARM_LABELS["kvd_v2-nvme"]
        gds = cliff_to_html._ARM_LABELS["kvd_v2-gds"]
        self.assertNotEqual(nvme, gds)
        self.assertIn("100,983", self.html)
        self.assertIn("48,264", self.html)
        self.assertIn(nvme, self.html)
        self.assertIn(gds, self.html)

    def test_page_embeds_every_backend_chart_set(self) -> None:
        self.assertEqual(sorted(self.run["charts"]), ["gds", "nvme"])
        self.assertIn("gds backend", self.html)
        self.assertIn("nvme backend", self.html)

    def test_workflow_link_names_the_nightly_cliff_workflow(self) -> None:
        self.assertIn("actions/workflows/rocm-aic-self-hosted-nightly-cliff-test.yml",
                      self.html)

    def test_page_names_the_model_gpu_and_tp(self) -> None:
        self.assertIn("openai/gpt-oss-120b · AMD Instinct MI355X · TP=1", self.html)

    def test_spec_states_the_vram_vllm_may_use(self) -> None:
        rows = dict(cliff_to_html._spec_rows(self.run))
        self.assertEqual(rows["GPU"], "AMD Instinct MI355X (gfx950), 288 GiB VRAM")
        self.assertEqual(rows["Tensor parallelism"], "TP=1 (1 GPU)")
        self.assertTrue(rows["vLLM VRAM budget"].startswith(
            "115.2 GiB of the GPU's 288 GiB (--gpu-memory-utilization 0.40)"))
        self.assertIn("vLLM 0.29.0", rows["Software"])
        self.assertIn("1 token per request", rows["Output"])

    def test_spec_follows_the_recorded_settings(self) -> None:
        # Nothing on the page is a restated default: change the run, the page follows.
        config = json.loads(json.dumps(_RUN_CONFIG))
        for arm in config["arms"].values():
            arm["vllm"]["args"][arm["vllm"]["args"].index("0.40")] = "0.90"
        with tempfile.TemporaryDirectory() as tmp:
            run = cliff_to_html._discover_run(_make_run(Path(tmp), config))
        rows = dict(cliff_to_html._spec_rows(run))
        self.assertTrue(rows["vLLM VRAM budget"].startswith("259.2 GiB of the GPU's 288 GiB"))

    def test_tiers_come_from_the_lmcache_server_argv(self) -> None:
        tiers = {arm: rest for arm, *rest in cliff_to_html._tier_rows(self.run)}
        self.assertEqual(tiers["vram_only"], ["Only KV cache tier", "None", "None"])
        gpu, l1, l2 = tiers["kvd_v2-nvme"]
        self.assertEqual(gpu, "Emptied before each timed iteration")
        self.assertEqual(l1, "Pinned host DRAM, 160 GiB")
        self.assertEqual(l2, "NIXL POSIX: 32,768 × 32 MiB files = 1,024 GiB, "
                             "buffered I/O, on 8× local NVMe (xfs)")
        self.assertEqual(tiers["kvd_v2-gds"][1:],
                         ["hipFile slab, 256 GiB, on 8× local NVMe (xfs)", "None"])

    def test_results_table_and_charts_show_hit_rates(self) -> None:
        self.assertIn("<th>GPU Cache Hit</th>", self.html)
        self.assertIn("29.6%", self.html)
        self.assertIn("89.6%", self.html)
        self.assertIn("Cache Hit Rate vs Concurrency (nvme backend)", self.html)

    def test_run_without_a_record_says_so(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run(root, config=None)
            for sidecar in (root / "results").glob("*.json"):
                sidecar.unlink()
            run = cliff_to_html._discover_run(root)
        page = cliff_to_html._build_html([run], "abc1234", "2026-11-09", "")
        self.assertIn("Configuration not recorded for this run", page)


if __name__ == "__main__":
    unittest.main()
