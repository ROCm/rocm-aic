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
    def _make_upstream_with_base(self, temp: Path) -> tuple[Path, str]:
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
        return upstream, "v0.5.5"

    def test_default_origin_applies_and_duplicate_entry_skips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            upstream, base_ref = self._make_upstream_with_base(temp)

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
                [str(SCRIPT), str(buildrepo), str(manifest), base_ref],
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

    def test_inline_github_remote_url_is_added_and_used(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            upstream, base_ref = self._make_upstream_with_base(temp)

            prwork = temp / "prwork"
            _git("clone", f"file://{upstream}", str(prwork))
            _git("config", "user.name", "Test User", cwd=prwork)
            _git("config", "user.email", "test@example.com", cwd=prwork)
            _git("checkout", "-b", "feature", "origin/main", cwd=prwork)
            with (prwork / "file.txt").open("a", encoding="utf-8") as handle:
                handle.write("inline-remote\n")
            _git("commit", "-am", "inline remote commit", cwd=prwork)
            _git("push", "origin", "HEAD:refs/heads/pr-77", cwd=prwork)
            pr_head = _git("rev-parse", "HEAD", cwd=prwork)
            _git("--git-dir", str(upstream), "update-ref", "refs/pull/77/head", pr_head)

            buildrepo = temp / "buildrepo"
            _git("clone", "--branch", "v0.5.5", "--depth", "1", f"file://{upstream}", str(buildrepo))
            github_url = "https://github.com/example/lmcache-fork.git"
            _git(
                "config",
                f'url.file://{upstream}.insteadOf',
                github_url,
                cwd=buildrepo,
            )

            manifest = temp / "lmcache.pull-requests"
            manifest.write_text(f"fork={github_url} 77\n", encoding="utf-8")

            subprocess.run(
                [str(SCRIPT), str(buildrepo), str(manifest), base_ref],
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertEqual(
                (buildrepo / "file.txt").read_text(encoding="utf-8"),
                "base\ninline-remote\n",
            )

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

    def test_stacked_pr_applies_dependency_commits_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            upstream, base_ref = self._make_upstream_with_base(temp)

            prwork = temp / "prwork"
            _git("clone", f"file://{upstream}", str(prwork))
            _git("config", "user.name", "Test User", cwd=prwork)
            _git("config", "user.email", "test@example.com", cwd=prwork)
            _git("checkout", "-b", "pr-42", "origin/main", cwd=prwork)
            with (prwork / "file.txt").open("a", encoding="utf-8") as handle:
                handle.write("dep-a\n")
            _git("commit", "-am", "dep commit a", cwd=prwork)
            with (prwork / "file.txt").open("a", encoding="utf-8") as handle:
                handle.write("dep-b\n")
            _git("commit", "-am", "dep commit b", cwd=prwork)
            _git("push", "origin", "HEAD:refs/heads/pr-42", cwd=prwork)
            pr42_head = _git("rev-parse", "HEAD", cwd=prwork)
            _git("--git-dir", str(upstream), "update-ref", "refs/pull/42/head", pr42_head)

            _git("checkout", "-b", "pr-43", "HEAD", cwd=prwork)
            with (prwork / "file.txt").open("a", encoding="utf-8") as handle:
                handle.write("top\n")
            _git("commit", "-am", "top commit", cwd=prwork)
            _git("push", "origin", "HEAD:refs/heads/pr-43", cwd=prwork)
            pr43_head = _git("rev-parse", "HEAD", cwd=prwork)
            _git("--git-dir", str(upstream), "update-ref", "refs/pull/43/head", pr43_head)

            buildrepo = temp / "buildrepo"
            _git("clone", "--branch", "v0.5.5", "--depth", "1", f"file://{upstream}", str(buildrepo))
            manifest = temp / "lmcache.pull-requests"
            manifest.write_text("43\n", encoding="utf-8")

            subprocess.run(
                [str(SCRIPT), str(buildrepo), str(manifest), base_ref],
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertEqual(
                (buildrepo / "file.txt").read_text(encoding="utf-8"),
                "base\ndep-a\ndep-b\ntop\n",
            )

    def test_stacked_pr_after_dependency_entry_only_applies_missing_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            upstream, base_ref = self._make_upstream_with_base(temp)

            prwork = temp / "prwork"
            _git("clone", f"file://{upstream}", str(prwork))
            _git("config", "user.name", "Test User", cwd=prwork)
            _git("config", "user.email", "test@example.com", cwd=prwork)
            _git("checkout", "-b", "pr-42", "origin/main", cwd=prwork)
            with (prwork / "file.txt").open("a", encoding="utf-8") as handle:
                handle.write("dep\n")
            _git("commit", "-am", "dep commit", cwd=prwork)
            _git("push", "origin", "HEAD:refs/heads/pr-42", cwd=prwork)
            pr42_head = _git("rev-parse", "HEAD", cwd=prwork)
            _git("--git-dir", str(upstream), "update-ref", "refs/pull/42/head", pr42_head)

            _git("checkout", "-b", "pr-43", "HEAD", cwd=prwork)
            with (prwork / "file.txt").open("a", encoding="utf-8") as handle:
                handle.write("child\n")
            _git("commit", "-am", "child commit", cwd=prwork)
            _git("push", "origin", "HEAD:refs/heads/pr-43", cwd=prwork)
            pr43_head = _git("rev-parse", "HEAD", cwd=prwork)
            _git("--git-dir", str(upstream), "update-ref", "refs/pull/43/head", pr43_head)

            buildrepo = temp / "buildrepo"
            _git("clone", "--branch", "v0.5.5", "--depth", "1", f"file://{upstream}", str(buildrepo))
            manifest = temp / "lmcache.pull-requests"
            manifest.write_text("42\n43\n", encoding="utf-8")

            subprocess.run(
                [str(SCRIPT), str(buildrepo), str(manifest), base_ref],
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertEqual(
                (buildrepo / "file.txt").read_text(encoding="utf-8"),
                "base\ndep\nchild\n",
            )


if __name__ == "__main__":
    unittest.main()
