# SPDX-License-Identifier: Apache-2.0
"""
CuPy-backed paged memory allocator for the NIXL GPU buffer path.

Drop-in replacement for PagedTensorMemoryAllocator when
LMCACHE_NIXL_USE_CUPY=1.  Uses cupy.ndarray page slices instead of
torch.Tensor slices so that the NIXL MP connector works without loading
the ROCm GPU-compute libraries (librocblas, libMIOpen, etc.).

The allocator exposes the same external interface as
PagedTensorMemoryAllocator (buffer_ptr, allocate(), free()) so the NIXL
backend does not need to distinguish between the two after construction.
"""

from collections import deque
from typing import List, Optional, Union

import cupy
import torch

from lmcache.utils import get_size_bytes
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
    logger,
)


class CuPyMemoryObj(TensorMemoryObj):
    """
    Page handle backed by a cupy.ndarray slice instead of torch.Tensor.

    Inherits TensorMemoryObj so isinstance() checks and metadata handling
    remain valid.  Only data_ptr() is overridden; the NIXL backend never
    reads raw_data bytes directly — all transfers are done by NIXL's
    C++ DMA engine via the registered pointer.
    """

    def __init__(
        self,
        raw_data: "cupy.ndarray",
        metadata: "MemoryObjMetadata",
        parent_allocator: "PagedCuPyMemoryAllocator",
    ) -> None:
        # Bypass TensorMemoryObj.__init__ (expects torch.Tensor).
        self.raw_data = raw_data
        self.metadata = metadata
        self._parent_allocator = parent_allocator
        self.monitor = None

    def data_ptr(self) -> int:  # type: ignore[override]
        return int(self.raw_data.data.ptr)


class PagedCuPyMemoryAllocator(MemoryAllocatorInterface):
    """
    Paged allocator backed by a single cupy.ndarray GPU buffer.

    Provides the same external interface as PagedTensorMemoryAllocator:
      buffer_ptr: int        — raw GPU pointer to the buffer start
      allocate(...)          — returns CuPyMemoryObj | None
      free(mem_obj)          — returns the page to the free pool
    """

    def __init__(
        self,
        buffer: "cupy.ndarray",
        shapes: "list[torch.Size]",
        dtypes: "list[torch.dtype]",
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> None:
        self.buffer = buffer.view(cupy.uint8).ravel()
        self.buffer_size = int(self.buffer.nbytes)
        self.buffer_ptr = int(self.buffer.data.ptr)

        self.shapes = shapes
        self.dtypes = dtypes
        self.fmt = fmt
        self.align_bytes = get_size_bytes(shapes, dtypes)

        assert self.buffer_size % self.align_bytes == 0, (
            f"Buffer size {self.buffer_size} is not a multiple of "
            f"align_bytes {self.align_bytes}."
        )

        n_pages = self.buffer_size // self.align_bytes
        self.free_blocks: deque[CuPyMemoryObj] = deque()

        for idx in range(n_pages):
            start = idx * self.align_bytes
            page = self.buffer[start : start + self.align_bytes]
            meta = MemoryObjMetadata(
                shapes[0], dtypes[0], idx, self.align_bytes, 1, 0, fmt,
                shapes=shapes, dtypes=dtypes,
            )
            self.free_blocks.append(CuPyMemoryObj(page, meta, self))

        self.num_active_allocations = 0
        logger.info(
            "PagedCuPyMemoryAllocator: %d pages × %d B (ptr=0x%x)",
            n_pages, self.align_bytes, self.buffer_ptr,
        )

    def allocate(self, shapes, dtypes, fmt=MemoryFormat.KV_2LTD,
                 allocator_type=None) -> Optional[CuPyMemoryObj]:
        if not self.free_blocks:
            return None
        self.num_active_allocations += 1
        return self.free_blocks.popleft()

    def free(self, mem_obj: CuPyMemoryObj) -> None:
        self.free_blocks.append(mem_obj)
        self.num_active_allocations -= 1

    def get_free_size(self) -> int:
        return len(self.free_blocks) * self.align_bytes

    def get_heap_size(self) -> int:
        return self.buffer_size
