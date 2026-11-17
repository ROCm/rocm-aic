# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""docker/docker-compose.yml is the single source of container-runtime defaults.

Covers docker/scripts/aic-compose-default.sh (the reader for callers that need
a compose default without the compose plugin) and the layers that must not
hold their own copy of a compose default.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE = REPO_ROOT / "docker" / "docker-compose.yml"
HELPER = REPO_ROOT / "docker" / "scripts" / "aic-compose-default.sh"

# Every layer that used to repeat a compose default.
CONSUMERS = [
    REPO_ROOT / "Makefile",
    *sorted((REPO_ROOT / "mk").glob("*.mk")),
    *sorted((REPO_ROOT / ".slurm").glob("*.s*")),
    REPO_ROOT / "monitoring" / "monitoring-lib.sh",
]

IMAGE_PINS = {
    "AIC_PROM_IMAGE": "prom/prometheus",
    "AIC_GRAFANA_IMAGE": "grafana/grafana",
    "AIC_NODE_EXPORTER_IMAGE": "quay.io/prometheus/node-exporter",
    "AIC_AMDGPU_EXPORTER_IMAGE": "rocm/device-metrics-exporter",
}


def _helper(var: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(HELPER), var], capture_output=True, text=True, check=False)


def _code(path: Path) -> str:
    """The file without its comment lines."""
    return "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")
    )


class ComposeDefaultHelperTest(unittest.TestCase):
    def test_prints_the_compose_default(self) -> None:
        text = COMPOSE.read_text(encoding="utf-8")
        for var, repo in IMAGE_PINS.items():
            with self.subTest(var=var):
                result = _helper(var)
                self.assertEqual(result.returncode, 0, result.stderr)
                pin = result.stdout.strip()
                self.assertTrue(pin.startswith(f"{repo}:"), pin)
                self.assertIn(f"${{{var}:-{pin}}}", text)

    def test_fails_without_a_plain_default(self) -> None:
        for var in ("AIC_NO_SUCH_VARIABLE", "AIC_ROCR_VISIBLE"):
            with self.subTest(var=var):
                result = _helper(var)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")

    def test_rejects_a_non_variable_name(self) -> None:
        self.assertEqual(_helper("a b").returncode, 2)


class SingleOwnerTest(unittest.TestCase):
    def test_approved_compose_defaults(self) -> None:
        self.assertEqual(_helper("LMCACHE_MAX_GPU_WORKERS").stdout.strip(), "4")
        self.assertEqual(_helper("LMCACHE_NIXL_POSIX_SLOT_SIZE").stdout.strip(), "16777216")

    def test_no_consumer_repeats_an_image_pin(self) -> None:
        for path in CONSUMERS:
            code = _code(path)
            for repo in IMAGE_PINS.values():
                with self.subTest(path=path.name, image=repo):
                    self.assertNotRegex(code, re.escape(repo) + r":v?\d")

    def test_no_consumer_defaults_the_posix_slot_size(self) -> None:
        for path in CONSUMERS:
            with self.subTest(path=path.name):
                self.assertNotRegex(_code(path), r"LMCACHE_NIXL_POSIX_SLOT_SIZE\s*(\?=|:?=|:-)\s*\d")

    def test_gpu_workers_is_only_set_as_an_explicit_preset(self) -> None:
        # The prometheus-dump job's single-client preset is the one exception.
        allowed = {".slurm/run-build-distribute.sh": ["export LMCACHE_MAX_GPU_WORKERS=1"]}
        for path in CONSUMERS:
            rel = path.relative_to(REPO_ROOT).as_posix()
            found = re.findall(r"^.*LMCACHE_MAX_GPU_WORKERS\s*(?:\?=|:?=|:-)\s*\d.*$", _code(path), re.M)
            with self.subTest(path=rel):
                self.assertEqual([line.strip() for line in found], allowed.get(rel, []))


if __name__ == "__main__":
    unittest.main()
