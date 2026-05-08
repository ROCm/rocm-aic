"""Direct LMCache engine manager for in-process cache operations.

Replaces the subprocess + HTTP API approach with direct calls
to the LMCache engine's store/retrieve/lookup/clear methods.
"""
import uuid
from typing import Optional, List, Union

import torch
from lmcache.v1.cache_engine import (
    LMCacheEngine,
    LMCacheEngineBuilder,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.gpu_connector import CreateGPUConnector
from lmcache.utils import (
    EngineType,
    mock_up_broadcast_fn,
    mock_up_broadcast_object_fn,
)

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
}


class EngineManager:
    """Manages an in-process LMCacheEngine instance.

    Provides simplified store/retrieve/lookup/clear wrappers
    that accept token ID lists directly, bypassing the HTTP
    API and its sequential-only tokens_mock limitation.
    """

    DTYPE_BYTE_SIZES = {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.uint8: 1,
    }

    def __init__(self):
        self.engine: Optional[LMCacheEngine] = None
        self.instance_id: Optional[str] = None
        self._kv_shape: Optional[tuple] = None
        self._kv_dtype: Optional[torch.dtype] = None

    def create_engine(
        self,
        config_path: str,
        model_name: str = "lmcache_model",
        kv_shape: str = "2,2,256,4,16",
        kv_dtype: str = "float16",
        worker_id: int = 0,
        world_size: int = 1,
        use_mla: bool = False,
    ) -> LMCacheEngine:
        """Create and initialize an LMCacheEngine in-process.

        Args:
            config_path: Path to YAML config file
            model_name: Model name for cache key identity
            kv_shape: Comma-separated KV cache shape string
            kv_dtype: KV cache dtype string
            worker_id: Worker ID
            world_size: Total number of workers
            use_mla: Enable Multi-Level Attention

        Returns:
            The initialized LMCacheEngine instance
        """
        shape_tuple = tuple(
            int(x.strip()) for x in kv_shape.split(",")
        )
        torch_dtype = DTYPE_MAP.get(kv_dtype, torch.float16)
        self._kv_shape = shape_tuple
        self._kv_dtype = torch_dtype

        config = LMCacheEngineConfig.from_file(config_path)
        config.validate()

        self.instance_id = (
            config.lmcache_instance_id
            or f"sim_{uuid.uuid4().hex[:8]}"
        )

        metadata = LMCacheMetadata(
            model_name=model_name,
            world_size=world_size,
            local_world_size=world_size,
            worker_id=worker_id,
            local_worker_id=worker_id,
            kv_dtype=torch_dtype,
            kv_shape=shape_tuple,
            use_mla=use_mla,
            role="worker",
        )

        gpu_connector = CreateGPUConnector(
            config, metadata, EngineType.MOCK
        )

        self.engine = LMCacheEngineBuilder.get_or_create(
            instance_id=self.instance_id,
            config=config,
            metadata=metadata,
            gpu_connector=gpu_connector,
            broadcast_fn=mock_up_broadcast_fn,
            broadcast_object_fn=mock_up_broadcast_object_fn,
        )
        self.engine.post_init()
        return self.engine

    @property
    def bytes_per_chunk(self) -> int:
        """KV cache bytes per chunk, derived from the
        engine's kv_shape and dtype."""
        if self._kv_shape is None or self._kv_dtype is None:
            return 0
        total_elements = 1
        for dim in self._kv_shape:
            total_elements *= dim
        elem_size = self.DTYPE_BYTE_SIZES.get(
            self._kv_dtype, 2
        )
        return total_elements * elem_size

    @property
    def bytes_per_token(self) -> int:
        """KV cache bytes per token (bytes_per_chunk
        divided by the chunk_size dimension)."""
        if self._kv_shape is None or self._kv_dtype is None:
            return 0
        # kv_shape = (layers, 2, chunk_size, heads, head_dim)
        chunk_size = self._kv_shape[2]
        if chunk_size == 0:
            return 0
        return self.bytes_per_chunk // chunk_size

    def store(
        self,
        token_ids: Union[List[int], torch.Tensor],
        req_id: Optional[str] = None,
    ) -> None:
        """Store KV cache for the given token IDs.

        Args:
            token_ids: List of token IDs (arbitrary, not
                       necessarily sequential)
            req_id: Optional request identifier for
                    LMCache logging
        """
        if self.engine is None:
            raise RuntimeError(
                "Engine not created. "
                "Call create_engine() first."
            )
        slot_mapping = torch.arange(
            len(token_ids), dtype=torch.long
        )
        kwargs: dict = {
            "tokens": token_ids,
            "slot_mapping": slot_mapping,
            "kvcaches": None,
        }
        if req_id is not None:
            kwargs["req_id"] = req_id
        self.engine.store(**kwargs)

    def retrieve(
        self,
        token_ids: Union[List[int], torch.Tensor],
        req_id: Optional[str] = None,
    ) -> int:
        """Retrieve KV cache for the given token IDs.

        Args:
            token_ids: List of token IDs
            req_id: Optional request identifier for
                    LMCache logging

        Returns:
            Number of tokens successfully retrieved
        """
        if self.engine is None:
            raise RuntimeError(
                "Engine not created. "
                "Call create_engine() first."
            )
        slot_mapping = torch.arange(
            len(token_ids), dtype=torch.long
        )
        kwargs: dict = {
            "tokens": token_ids,
            "slot_mapping": slot_mapping,
            "kvcaches": None,
        }
        if req_id is not None:
            kwargs["req_id"] = req_id
        mask = self.engine.retrieve(**kwargs)
        return int(mask.sum().item())

    def lookup(
        self,
        token_ids: Union[List[int], torch.Tensor],
    ) -> int:
        """Check how many prefix tokens are cached.

        Args:
            token_ids: List of token IDs

        Returns:
            Number of prefix tokens found in cache
        """
        if self.engine is None:
            raise RuntimeError(
                "Engine not created. Call create_engine() first."
            )
        return self.engine.lookup(tokens=token_ids)

    def clear(
        self,
        token_ids: Optional[
            Union[List[int], torch.Tensor]
        ] = None,
    ) -> int:
        """Clear cache entries.

        Args:
            token_ids: Specific tokens to clear, or None to
                       clear all entries

        Returns:
            Number of entries cleared
        """
        if self.engine is None:
            raise RuntimeError(
                "Engine not created. Call create_engine() first."
            )
        return self.engine.clear(tokens=token_ids)

    def freeze(self, enabled: bool) -> None:
        """Enable/disable freeze mode (no stores, retrieve
        only)."""
        if self.engine is None:
            raise RuntimeError(
                "Engine not created. Call create_engine() first."
            )
        self.engine.freeze(enabled)

    def set_hot_cache(self, enabled: bool) -> None:
        """Enable/disable the CPU hot cache layer."""
        if self.engine is None:
            raise RuntimeError(
                "Engine not created. Call create_engine() first."
            )
        self.engine.set_hot_cache(enabled)

    def is_healthy(self) -> bool:
        """Check engine health."""
        if self.engine is None:
            return False
        return self.engine.is_healthy()

    def close(self) -> None:
        """Close the engine and free resources."""
        if self.engine is not None and self.instance_id:
            try:
                LMCacheEngineBuilder.destroy(self.instance_id)
            except Exception:
                pass
            self.engine = None
            self.instance_id = None
