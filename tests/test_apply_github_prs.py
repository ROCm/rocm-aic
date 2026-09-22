# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Regression tests for docker/scripts/apply-github-prs.sh."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "docker" / "scripts" / "apply-github-prs.sh"


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


class ApplyGithubPrsTest(unittest.TestCase):
    def test_default_origin_applies_and_duplicate_entry_skips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            upstream = temp / "upstream.git"
            _git("init", "--bare", str(upstream))

            seed = temp / "seed"
            _git("clone", f"file://{upstream}", str(seed))
            _git("config", "user.name", "Test User", cwd=seed)
            _git("config", "user.email", "test@example.com", cwd=seed)
            (seed / "file.txt").write_text("base\n", encoding="utf-8")
            _git("add", "file.txt", cwd=seed)
            _git("commit", "-m", "base", cwd=seed)
            _git("branch", "-M", "main", cwd=seed)
            _git("tag", "v0.5.5", cwd=seed)
            _git("push", "origin", "main", "--tags", cwd=seed)

            prwork = temp / "prwork"
            _git("clone", f"file://{upstream}", str(prwork))
            _git("config", "user.name", "Test User", cwd=prwork)
            _git("config", "user.email", "test@example.com", cwd=prwork)
            _git("checkout", "-b", "feature", "origin/main", cwd=prwork)
            with (prwork / "file.txt").open("a", encoding="utf-8") as handle:
                handle.write("pr1\n")
            _git("commit", "-am", "pr commit 1", cwd=prwork)
            with (prwork / "file.txt").open("a", encoding="utf-8") as handle:
                handle.write("pr2\n")
            _git("commit", "-am", "pr commit 2", cwd=prwork)
            _git("push", "origin", "HEAD:refs/heads/pr-42", cwd=prwork)
            pr_head = _git("rev-parse", "HEAD", cwd=prwork)
            _git("--git-dir", str(upstream), "update-ref", "refs/pull/42/head", pr_head)

            buildrepo = temp / "buildrepo"
            _git("clone", "--branch", "v0.5.5", "--depth", "1", f"file://{upstream}", str(buildrepo))
            manifest = temp / "lmcache.pull-requests"
            manifest.write_text("42\n\n# duplicate\n42\n", encoding="utf-8")

            subprocess.run(
                [str(SCRIPT), str(buildrepo), str(manifest), "v0.5.5"],
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertEqual(
                (buildrepo / "file.txt").read_text(encoding="utf-8"),
                "base\npr1\npr2\n",
            )

    def test_inline_remote_url_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            repo = temp / "repo"
            _git("init", str(repo))
            _git("config", "user.name", "Test User", cwd=repo)
            _git("config", "user.email", "test@example.com", cwd=repo)
            (repo / "file.txt").write_text("base\n", encoding="utf-8")
            _git("add", "file.txt", cwd=repo)
            _git("commit", "-m", "base", cwd=repo)
            _git("remote", "add", "origin", "https://github.com/LMCache/LMCache.git", cwd=repo)

            manifest = temp / "lmcache.pull-requests"
            manifest.write_text(
                "origin=https://github.com/example/other.git 42\n", encoding="utf-8"
            )

            result = subprocess.run(
                [str(SCRIPT), str(repo), str(manifest), "v0.5.5"],
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already points to", result.stderr)

    def test_non_github_inline_remote_url_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            repo = temp / "repo"
            _git("init", str(repo))
            _git("config", "user.name", "Test User", cwd=repo)
            _git("config", "user.email", "test@example.com", cwd=repo)
            (repo / "file.txt").write_text("base\n", encoding="utf-8")
            _git("add", "file.txt", cwd=repo)
            _git("commit", "-m", "base", cwd=repo)

            manifest = temp / "lmcache.pull-requests"
            manifest.write_text("fork=file:///tmp/example.git 42\n", encoding="utf-8")

            result = subprocess.run(
                [str(SCRIPT), str(repo), str(manifest), "v0.5.5"],
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("supported GitHub URL", result.stderr)


if __name__ == "__main__":
    unittest.main()
