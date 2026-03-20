# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Utility functions for the plotting framework.

Standalone version with no external dependencies beyond standard library.
"""

from typing import Any, Callable, Iterable, TypeVar


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for use in file paths."""
    return filename.replace("/", "_").replace("..", "__").strip("'").strip('"')


T = TypeVar('T')
K = TypeVar('K')


def full_groupby(
    iterable: Iterable[T],
    key: Callable[[T], K],
) -> list[tuple[K, list[T]]]:
    """
    Group elements of an iterable by a key function.

    Similar to itertools.groupby but doesn't require pre-sorting
    and returns all groups as a list.

    Args:
        iterable: Items to group
        key: Function that extracts the grouping key from each item

    Returns:
        List of (key, group_items) tuples
    """
    groups: dict[K, list[T]] = {}

    for item in iterable:
        k = key(item)
        if k not in groups:
            groups[k] = []
        groups[k].append(item)

    return list(groups.items())
