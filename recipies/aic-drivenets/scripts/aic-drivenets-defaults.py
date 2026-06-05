# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CONTAINER_DATA_DIR = "/data"
DEFAULT_CONTAINER_SERVER_LOG_DIR = "/var/log/aic-drivenets"


def container_data_root() -> Path:
    """In-container LMCache data root (**`ADE_CONTAINER_DATA_DIR`**)."""
    v = os.environ.get("ADE_CONTAINER_DATA_DIR", "").strip()
    return Path(v) if v else Path(DEFAULT_CONTAINER_DATA_DIR)


def container_server_log_dir() -> Path:
    """Directory for **`vllm-server`** tee (**`server.txt`**)."""
    v = os.environ.get("ADE_SERVER_LOG_DIR", "").strip()
    return Path(v) if v else Path(DEFAULT_CONTAINER_SERVER_LOG_DIR)
