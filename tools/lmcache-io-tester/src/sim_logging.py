# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""LMCache / PyTorch noise control for the IO tester CLI."""
import logging
import os
import sys
import warnings
from typing import Set

_HANDLER_IDS_WITH_FILTER: Set[int] = set()


def lmcache_verbose_enabled() -> bool:
    return (
        os.environ.get(
            "LMCACHE_SIM_VERBOSE_LMCACHE", ""
        ).strip().lower()
        in ("1", "true", "yes")
        or "--verbose-lmcache" in sys.argv
    )


class _DropLmcacheInfo(logging.Filter):
    """Hide INFO/DEBUG from ``lmcache.*`` loggers unless verbose."""

    def filter(self, record: logging.LogRecord) -> bool:
        if lmcache_verbose_enabled():
            return True
        name = record.name
        if name == "lmcache" or name.startswith(
            "lmcache."
        ):
            return record.levelno >= logging.WARNING
        return True


_FILTER = _DropLmcacheInfo()


def _attach_filter(handler: logging.Handler) -> None:
    hid = id(handler)
    if hid in _HANDLER_IDS_WITH_FILTER:
        return
    handler.addFilter(_FILTER)
    _HANDLER_IDS_WITH_FILTER.add(hid)


def suppress_lmcache_info() -> None:
    """Drop LMCache library INFO/DEBUG on stderr."""
    if lmcache_verbose_enabled():
        return
    warn_level = logging.WARNING
    reg = logging.root.manager.loggerDict
    for name in list(reg.keys()):
        if name == "lmcache" or name.startswith(
            "lmcache."
        ):
            logging.getLogger(name).setLevel(warn_level)
    root = logging.getLogger()
    for h in root.handlers:
        _attach_filter(h)
    for name in list(reg.keys()):
        if name == "lmcache" or name.startswith(
            "lmcache."
        ):
            obj = reg.get(name)
            if isinstance(obj, logging.Logger):
                for h in obj.handlers:
                    _attach_filter(h)


def suppress_torch_cuda_warning() -> None:
    """Hide PyTorch CUDA driver probe warning for CPU-only sim."""
    if lmcache_verbose_enabled():
        return
    warnings.filterwarnings(
        "ignore",
        message=r".*CUDA initialization.*",
        category=UserWarning,
    )


def configure_lmcache_env_defaults() -> None:
    """Environment defaults before LMCache imports.

    - ``LMCACHE_LOG_LEVEL``: LMCache ``init_logger()`` reads this (library
      default INFO). Default WARNING unless verbose.
    - ``PYTHONHASHSEED``: LMCache falls back to Python ``hash`` when vLLM is
      not installed; a fixed seed avoids its PYTHONHASHSEED warnings and
      keeps chunk keys stable. Override with ``PYTHONHASHSEED=`` in the
      environment if you need a different seed.
    """
    if lmcache_verbose_enabled():
        return
    os.environ.setdefault(
        "LMCACHE_LOG_LEVEL", "WARNING"
    )
    os.environ.setdefault("PYTHONHASHSEED", "0")


def configure_default_at_import() -> None:
    """Run when the CLI module loads (before LMCache import)."""
    configure_lmcache_env_defaults()
    suppress_torch_cuda_warning()
    suppress_lmcache_info()
