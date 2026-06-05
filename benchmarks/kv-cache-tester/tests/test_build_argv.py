#!/usr/bin/env python3
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_argv import build_argv, load_profile  # noqa: E402


class BuildArgvTests(unittest.TestCase):
    def test_trace_replay_blog_profile(self) -> None:
        cfg = Path(__file__).resolve().parent.parent / "configs" / "trace-replay-blog.yaml"
        profile = load_profile(cfg)
        argv = build_argv(
            profile,
            api_endpoint="http://127.0.0.1:8000",
            upstream_root=Path("/tmp/kv-cache-tester"),
            output_dir=Path("/tmp/out"),
        )
        self.assertEqual(argv[0], "/tmp/kv-cache-tester/trace_replay_tester.py")
        self.assertIn("--api-endpoint", argv)
        self.assertIn("http://127.0.0.1:8000", argv)
        self.assertIn("--trace-directory", argv)
        self.assertIn("traces", argv)
        self.assertIn("--recycle", argv)
        self.assertIn("--seed", argv)
        self.assertIn("42", argv)


if __name__ == "__main__":
    unittest.main()
