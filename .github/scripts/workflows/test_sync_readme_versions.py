#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Fixture tests for sync-readme-versions.py."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_NAME = "sync-readme-versions.py"


class ReadmeSyncTest(unittest.TestCase):
    # Maps each ARG name to its owning Dockerfile (relative to docker/).
    _ARG_DOCKERFILE_REL: dict[str, str] = {
        "ROCM_VERSION": "base/Dockerfile",
        "ROCM_BASE_IMAGE": "base/Dockerfile",
        "PYTORCH_BRANCH": "base/Dockerfile",
        "VLLM_REF": "vllm/Dockerfile",
        "AITER_REF": "lmcache/Dockerfile",
        "LMCACHE_REF": "lmcache/Dockerfile",
        "NIXL_REF": "lmcache/Dockerfile",
        "HSA_SNOOP_REF": "lmcache/Dockerfile",
    }

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / ".github" / "scripts" / "workflows").mkdir(parents=True)
        for subdir in ("base", "vllm", "lmcache"):
            (self.root / "docker" / subdir).mkdir(parents=True)
        shutil.copy2(Path(__file__).with_name(SCRIPT_NAME), self.script)
        for subdir, src_name in (
            ("base", "base/Dockerfile"),
            ("vllm", "vllm/Dockerfile"),
            ("lmcache", "lmcache/Dockerfile"),
        ):
            shutil.copy2(
                REPO_ROOT / "docker" / src_name,
                self.root / "docker" / subdir / "Dockerfile",
            )
        shutil.copy2(REPO_ROOT / "README.md", self.readme)
        shutil.copytree(REPO_ROOT / "patches", self.root / "patches")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @property
    def script(self) -> Path:
        return self.root / ".github" / "scripts" / "workflows" / SCRIPT_NAME

    def _dockerfile_for_arg(self, name: str) -> Path:
        rel = self._ARG_DOCKERFILE_REL.get(name, "lmcache/Dockerfile")
        return self.root / "docker" / rel

    @property
    def dockerfile(self) -> Path:
        # Backward compat: tests that call self.dockerfile directly get lmcache/Dockerfile
        # (which holds most ARGs).  Tests that need a specific Dockerfile use _dockerfile_for_arg.
        return self.root / "docker" / "lmcache" / "Dockerfile"

    @property
    def readme(self) -> Path:
        return self.root / "README.md"

    def run_sync(self, mode: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(self.script), mode],
            cwd=self.root,
            check=False,
            capture_output=True,
            text=True,
        )

    def replace_dockerfile(self, old: str, new: str, name: str = "") -> None:
        df = self._dockerfile_for_arg(name) if name else self.dockerfile
        text = df.read_text(encoding="utf-8")
        self.assertIn(old, text)
        df.write_text(text.replace(old, new, 1), encoding="utf-8")

    def docker_arg(self, name: str) -> tuple[str, str]:
        df = self._dockerfile_for_arg(name)
        pattern = re.compile(
            rf"^[ \t]*(?i:ARG)[ \t]+{re.escape(name)}=(.*)$",
            re.MULTILINE,
        )
        matches = list(pattern.finditer(df.read_text(encoding="utf-8")))
        self.assertEqual(len(matches), 1, f"fixture must have exactly one ARG {name}=")
        return matches[0].group(0), matches[0].group(1).strip()

    def set_docker_arg(self, name: str, value: str, *, keyword: str = "ARG") -> None:
        line, _ = self.docker_arg(name)
        df = self._dockerfile_for_arg(name)
        text = df.read_text(encoding="utf-8")
        self.assertIn(line, text)
        df.write_text(text.replace(line, f"{keyword} {name}={value}", 1), encoding="utf-8")

    def replace_readme(self, old: str, new: str) -> None:
        text = self.readme.read_text(encoding="utf-8")
        self.assertIn(old, text)
        self.readme.write_text(text.replace(old, new, 1), encoding="utf-8")

    def readme_badge_line(self, name: str) -> str:
        pattern = re.compile(rf"^\[!\[{re.escape(name)}\].*$", re.MULTILINE)
        matches = pattern.findall(self.readme.read_text(encoding="utf-8"))
        self.assertEqual(len(matches), 1, f"fixture must have exactly one {name} badge")
        return matches[0]

    def patch_names(self, component: str) -> list[str]:
        directory = self.root / "patches" / component
        return sorted(
            path.name
            for path in directory.iterdir()
            if not path.name.startswith(".") and path.name.endswith(".patch") and path.is_file()
        )

    def add_unique_patch(self, component: str, preferred_name: str) -> str:
        directory = self.root / "patches" / component
        name = preferred_name
        while (directory / name).exists():
            name = f"fixture-{name}"
        (directory / name).touch()
        return name

    def readme_hash(self) -> str:
        return hashlib.sha256(self.readme.read_bytes()).hexdigest()

    def assert_write_fails_without_change(self, expected: str) -> None:
        before = self.readme_hash()
        result = self.run_sync("--write")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(expected, result.stderr)
        self.assertEqual(self.readme_hash(), before)

    def test_current_tree_is_in_sync(self) -> None:
        before = self.readme_hash()
        result = self.run_sync("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("README component metadata is in sync", result.stdout)
        self.assertEqual(self.readme_hash(), before)

    def test_stale_readme_prints_diff_and_repair_command(self) -> None:
        _, current = self.docker_arg("VLLM_REF")
        stale_ref = f"{current}-fixture-stale"
        self.set_docker_arg("VLLM_REF", stale_ref)
        before = self.readme_hash()
        result = self.run_sync("--check")
        self.assertEqual(result.returncode, 1)
        self.assertIn("vLLM badge, vLLM row", result.stderr)
        self.assertIn("README.md (committed)", result.stdout)
        self.assertIn(stale_ref, result.stdout)
        self.assertIn(
            "python3 .github/scripts/workflows/sync-readme-versions.py --write",
            result.stderr,
        )
        self.assertEqual(self.readme_hash(), before)

    def test_write_repairs_all_versions_and_is_idempotent(self) -> None:
        new_values = {
            "ROCM_VERSION": "98.76.54",
            "PYTORCH_BRANCH": "release/98.76",
            "VLLM_REF": "v98.76.54-fixture",
            "AITER_REF": "v98.76.58-fixture",
            "LMCACHE_REF": "v98.76.55-fixture",
            "NIXL_REF": "v98.76.56-fixture",
            "HSA_SNOOP_REF": "v98.76.57-fixture",
        }
        for name, value in new_values.items():
            _, current = self.docker_arg(name)
            if current == value:
                value = value.replace("98", "97", 1)
                new_values[name] = value
            self.set_docker_arg(name, value)

        first = self.run_sync("--write")
        self.assertEqual(first.returncode, 0, first.stderr)
        rendered = self.readme.read_text(encoding="utf-8")
        self.assertIn(f"ROCm-{new_values['ROCM_VERSION']}-green.svg", rendered)
        self.assertIn(
            f"[![PyTorch](https://img.shields.io/badge/PyTorch-release%2F98.76-ee4c2c.svg)]"
            "(https://github.com/ROCm/pytorch/tree/release/98.76)",
            rendered,
        )
        self.assertIn(
            f"`{new_values['VLLM_REF']}` + {len(self.patch_names('vllm'))} AMD patches",
            rendered,
        )
        self.assertIn(
            f"[![AITER](https://img.shields.io/badge/AITER-v98.76.58--fixture-blue.svg)]"
            f"(https://github.com/ROCm/aiter/tree/{new_values['AITER_REF']})",
            rendered,
        )
        for name in ("PYTORCH_BRANCH", "AITER_REF", "LMCACHE_REF", "NIXL_REF", "HSA_SNOOP_REF"):
            self.assertIn(f"`{new_values[name]}`", rendered)
        rocm_short = ".".join(new_values["ROCM_VERSION"].split(".")[:2])
        self.assertIn(f"ROCm {rocm_short} base image | GA in ROCm 7.14", rendered)
        first_hash = self.readme_hash()

        check = self.run_sync("--check")
        second = self.run_sync("--write")
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(self.readme_hash(), first_hash)

    def test_duplicate_missing_and_empty_args_fail_without_write(self) -> None:
        line, value = self.docker_arg("VLLM_REF")
        vllm_df = self._dockerfile_for_arg("VLLM_REF")
        cases = (
            (
                line,
                f"{line}\nARG VLLM_REF={value}-duplicate",
                "found 2 default(s)",
            ),
            (line, "", "found 0 default(s)"),
            (line, "ARG VLLM_REF=", "0 non-empty"),
        )
        for old, new, expected in cases:
            with self.subTest(expected=expected):
                original = vllm_df.read_text(encoding="utf-8")
                self.replace_dockerfile(old, new, name="VLLM_REF")
                self.assert_write_fails_without_change(expected)
                vllm_df.write_text(original, encoding="utf-8")

    def test_arg_keyword_is_case_insensitive(self) -> None:
        _, value = self.docker_arg("VLLM_REF")
        self.set_docker_arg("VLLM_REF", value, keyword="arg")
        result = self.run_sync("--check")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_arg_continuation_is_rejected_without_write(self) -> None:
        line, _ = self.docker_arg("VLLM_REF")
        self.replace_dockerfile(line, "ARG VLLM_REF=fixture.\\\ncontinued", name="VLLM_REF")
        self.assert_write_fails_without_change("must use a single-line default")

    def test_unresolved_base_image_variable_is_rejected(self) -> None:
        self.set_docker_arg("ROCM_BASE_IMAGE", "example.invalid/${UNRESOLVED}:tag")
        self.assert_write_fails_without_change("contains an unresolved variable")

    def test_unsupported_base_image_contract_is_rejected(self) -> None:
        self.set_docker_arg(
            "ROCM_BASE_IMAGE",
            "rocm/dev-ubuntu-22.04:${ROCM_VERSION}-full",
        )
        self.assert_write_fails_without_change(
            "must match the README Ubuntu 24.04/Python 3.12 contract"
        )

    def test_ref_badge_value_is_escaped(self) -> None:
        cases = (
            (
                "PYTORCH_BRANCH",
                "PyTorch",
                "ee4c2c",
                "https://github.com/ROCm/pytorch/tree/",
                "release/rocm-test_branch",
                "release%2Frocm--test__branch",
            ),
            (
                "VLLM_REF",
                "vLLM",
                "blue",
                "https://github.com/vllm-project/vllm",
                "release/rocm-test_branch-fixture",
                "release%2Frocm--test__branch--fixture",
            ),
            (
                "AITER_REF",
                "AITER",
                "blue",
                "https://github.com/ROCm/aiter/tree/",
                "release/rocm-test_branch-fixture",
                "release%2Frocm--test__branch--fixture",
            ),
        )
        for arg_name, badge_name, color, base_url, ref, escaped_ref in cases:
            with self.subTest(arg_name=arg_name):
                _, current = self.docker_arg(arg_name)
                if current == ref:
                    ref = f"{ref}-alt"
                    escaped_ref = f"{escaped_ref}--alt"
                self.set_docker_arg(arg_name, ref)
                result = self.run_sync("--write")
                self.assertEqual(result.returncode, 0, result.stderr)
                rendered = self.readme.read_text(encoding="utf-8")
                expected_url = f"{base_url}{ref}" if base_url.endswith("/") else base_url
                self.assertIn(
                    f"[![{badge_name}](https://img.shields.io/badge/{badge_name}-{escaped_ref}-{color}.svg)]"
                    f"({expected_url})",
                    rendered,
                )
                if arg_name == "VLLM_REF":
                    self.assertIn(f"`{ref}` + {len(self.patch_names('vllm'))} AMD patches", rendered)
                else:
                    self.assertIn(f"`{ref}`", rendered)
                self.assertEqual(self.run_sync("--check").returncode, 0)

    def test_patch_metadata_is_derived(self) -> None:
        _, vllm_ref = self.docker_arg("VLLM_REF")
        _, lmcache_ref = self.docker_arg("LMCACHE_REF")
        vllm_patch = self.add_unique_patch("vllm", "zz-readme-sync-fixture.patch")
        lmcache_patch = self.add_unique_patch("lmcache", "zz-readme-sync-fixture.patch")
        nixl_patch = self.add_unique_patch("nixl", "zz-readme-sync-fixture.patch")
        result = self.run_sync("--write")
        self.assertEqual(result.returncode, 0, result.stderr)
        rendered = self.readme.read_text(encoding="utf-8")
        self.assertIn(
            "| vLLM | `github.com/vllm-project/vllm` (source build) | "
            f"`{vllm_ref}` + {len(self.patch_names('vllm'))} AMD patches |",
            rendered,
        )
        self.assertIn(
            "| LMCache | `LMCache/LMCache` (upstream) | "
            f"`{lmcache_ref}` + {len(self.patch_names('lmcache'))} AMD patches |",
            rendered,
        )
        nixl_summary = " + ".join(f"`{name}`" for name in self.patch_names("nixl"))
        self.assertIn(nixl_summary, rendered)
        self.assertIn(vllm_patch, self.patch_names("vllm"))
        self.assertIn(lmcache_patch, self.patch_names("lmcache"))
        self.assertIn(nixl_patch, self.patch_names("nixl"))

    def test_required_patch_set_cannot_be_empty(self) -> None:
        for patch in (self.root / "patches" / "lmcache").glob("*.patch"):
            patch.unlink()
        self.assert_write_fails_without_change("expected at least one .patch file")

    def test_missing_and_duplicate_readme_fields_fail_with_hint(self) -> None:
        badge = self.readme_badge_line("NIXL")
        cases = ((badge, "", "found 0"), (badge, f"{badge}\n{badge}", "found 2"))
        for old, new, expected in cases:
            with self.subTest(expected=expected):
                original = self.readme.read_text(encoding="utf-8")
                self.replace_readme(old, new)
                before = self.readme_hash()
                result = self.run_sync("--write")
                self.assertEqual(result.returncode, 1)
                self.assertIn(expected, result.stderr)
                self.assertIn("Restore the named README badge", result.stderr)
                self.assertIn(
                    "python3 .github/scripts/workflows/sync-readme-versions.py --write",
                    result.stderr,
                )
                self.assertEqual(self.readme_hash(), before)
                self.readme.write_text(original, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
