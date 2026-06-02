#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Backward-compatible wrapper for summarize-recipe-job.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.argv[0] = str(_here / "summarize-recipe-job.py")
runpy.run_path(str(_here / "summarize-recipe-job.py"), run_name="__main__")
