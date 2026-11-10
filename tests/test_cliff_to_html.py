# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Regression tests for the nightly cliff page generator."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import cliff_to_html

_FIELDS = ["arm", "concurrency", "iter", "throughput_tok_s_total",
           "p50_latency_s", "p95_latency_s"]
_STAMP = "20261109-233656"


def _write_csv(path: Path, arm: str, thr: float) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_FIELDS)
        w.writerow([arm, 1, 0, thr, 0.5, 0.5])


def _make_run(root: Path) -> Path:
    """A run dir laid out the way run-cliff.sbatch leaves it, all three arms."""
    results = root / "results"
    results.mkdir(parents=True)
    _write_csv(results / f"cliff-vram_only-{_STAMP}.csv", "vram_only", 25366.0)
    # Both kvd arms carry arm=kvd_v2; only the file name tells them apart.
    _write_csv(results / f"cliff-kvd_v2-nvme-{_STAMP}.csv", "kvd_v2", 100983.0)
    _write_csv(results / f"cliff-kvd_v2-gds-{_STAMP}.csv", "kvd_v2", 48264.0)
    for sub in ("nvme", "gds"):
        d = root / "plots" / sub
        d.mkdir(parents=True)
        for name in ("cliff-throughput.png", "cliff-latency-p50.png",
                     "cliff-latency-p95.png"):
            (d / name).write_bytes(f"{sub}-{name}".encode())
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


if __name__ == "__main__":
    unittest.main()
