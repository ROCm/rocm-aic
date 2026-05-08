"""Workload patterns for LMCache simulation."""
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tokenizer_interface import TokenizerWrapper

try:
    from .base import BaseWorkload, WorkloadMetrics
except ImportError:
    from workloads.base import BaseWorkload, WorkloadMetrics


def _tokenize_input(
    tokenizer: Optional[Any],
    text_input: Optional[str],
) -> Optional[list]:
    """Tokenize text input, handling both file paths and
    inline text.

    Args:
        tokenizer: TokenizerWrapper instance
        text_input: File path or inline text

    Returns:
        List of token IDs, or None on failure
    """
    if not text_input or not tokenizer:
        return None
    try:
        from pathlib import Path

        text_path = Path(text_input)
        if text_path.exists():
            return tokenizer.tokenize_text_file(text_path)
        return tokenizer.tokenize(text_input)
    except Exception as e:
        print(
            f"Warning: Failed to tokenize text input: {e}",
            file=sys.stderr,
        )
        return None


def _build_token_ids(
    tokenized_text: Optional[list],
    start_idx: int,
    chunk_size: int,
    fallback_start: int,
) -> list:
    """Build a list of token IDs for an engine operation.

    When tokenized text is available, slices actual token
    IDs from it. Otherwise falls back to a sequential range.

    Args:
        tokenized_text: Full tokenized text or None
        start_idx: Starting index into tokenized_text
        chunk_size: Number of tokens per chunk
        fallback_start: Start value for fallback range

    Returns:
        List of token IDs
    """
    if tokenized_text and len(tokenized_text) >= 2:
        end_idx = min(
            start_idx + chunk_size, len(tokenized_text)
        )
        if end_idx <= start_idx:
            end_idx = min(
                start_idx + 1, len(tokenized_text)
            )
        return tokenized_text[start_idx:end_idx]

    return list(range(fallback_start, fallback_start + chunk_size))


class RandomWorkload(BaseWorkload):
    """Random workload pattern."""

    def __init__(
        self,
        engine: Any = None,
        key_range: int = 10000,
        value_size: int = 1024,
        tokenizer: Optional[Any] = None,
        text_input: Optional[str] = None,
    ):
        super().__init__(engine)
        self.key_range = key_range
        self.value_size = value_size
        self.tokenizer = tokenizer
        self.text_input = text_input
        self._tokenized_text = _tokenize_input(
            tokenizer, text_input
        )

    def generate_operation(self) -> Dict[str, Any]:
        """Generate random operation."""
        key = f"key_{random.randint(0, self.key_range)}"
        op_type = random.choice(["store", "retrieve"])
        return {"type": op_type, "key": key}

    def execute_operation(
        self, operation: Dict[str, Any]
    ) -> bool:
        """Execute operation via direct engine call."""
        try:
            chunk_size = 256
            key_hash = abs(hash(operation["key"]))

            if (
                self._tokenized_text
                and len(self._tokenized_text) >= 2
            ):
                max_start = max(
                    0,
                    len(self._tokenized_text) - chunk_size,
                )
                start_idx = key_hash % max(1, max_start + 1)
            else:
                start_idx = 0

            token_ids = _build_token_ids(
                self._tokenized_text,
                start_idx,
                chunk_size,
                fallback_start=key_hash % 10000,
            )

            req_id = operation.get("key")
            bpt = getattr(
                self.engine, "bytes_per_token", 0
            )
            if operation["type"] == "store":
                self.engine.store(
                    token_ids, req_id=req_id
                )
                blocks = (
                    len(token_ids) // chunk_size
                ) or 1
                operation["kv_blocks"] = blocks
                operation["data_bytes"] = (
                    blocks * chunk_size * bpt
                )
                return True
            else:
                num_retrieved = self.engine.retrieve(
                    token_ids, req_id=req_id
                )
                cache_hit = num_retrieved > 0
                operation["cache_hit"] = cache_hit
                operation["kv_blocks"] = num_retrieved
                operation["data_bytes"] = (
                    num_retrieved * bpt
                )
                return True

        except Exception as e:
            if self.metrics.failed_operations < 3:
                print(
                    f"Engine operation failed: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
            return False


class SequentialWorkload(BaseWorkload):
    """Sequential workload pattern."""

    def __init__(
        self,
        engine: Any = None,
        start_key: int = 0,
        value_size: int = 1024,
        tokenizer: Optional[Any] = None,
        text_input: Optional[str] = None,
    ):
        super().__init__(engine)
        self.current_key = start_key
        self.value_size = value_size
        self.tokenizer = tokenizer
        self.text_input = text_input
        self._tokenized_text = _tokenize_input(
            tokenizer, text_input
        )
        self._token_offset = 0

    def generate_operation(self) -> Dict[str, Any]:
        """Generate sequential operation."""
        key = f"key_{self.current_key}"
        self.current_key += 1
        return {"type": "store", "key": key}

    def execute_operation(
        self, operation: Dict[str, Any]
    ) -> bool:
        """Execute operation via direct engine call."""
        try:
            chunk_size = 256

            if (
                self._tokenized_text
                and len(self._tokenized_text) >= 2
            ):
                start_idx = self._token_offset
                if start_idx >= len(self._tokenized_text):
                    start_idx = 0
                    self._token_offset = 0
            else:
                start_idx = 0

            token_ids = _build_token_ids(
                self._tokenized_text,
                start_idx,
                chunk_size,
                fallback_start=(
                    abs(hash(operation["key"])) % 10000
                ),
            )

            if self._tokenized_text:
                end_idx = min(
                    start_idx + chunk_size,
                    len(self._tokenized_text),
                )
                self._token_offset = (
                    end_idx
                    if end_idx < len(self._tokenized_text)
                    else 0
                )

            self.engine.store(
                token_ids,
                req_id=operation.get("key"),
            )
            blocks = (
                len(token_ids) // chunk_size
            ) or 1
            bpt = getattr(
                self.engine, "bytes_per_token", 0
            )
            operation["kv_blocks"] = blocks
            operation["data_bytes"] = (
                blocks * chunk_size * bpt
            )
            return True

        except Exception as e:
            if self.metrics.failed_operations < 3:
                print(
                    f"Engine operation failed: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
            return False


class BurstWorkload(BaseWorkload):
    """Burst workload pattern."""

    def __init__(
        self,
        engine: Any = None,
        key_range: int = 10000,
        value_size: int = 1024,
        burst_size: int = 100,
        burst_interval: float = 10.0,
        tokenizer: Optional[Any] = None,
        text_input: Optional[str] = None,
    ):
        super().__init__(engine)
        self.key_range = key_range
        self.value_size = value_size
        self.burst_size = burst_size
        self.burst_interval = burst_interval
        self.burst_count = 0
        self.last_burst_time = time.time()
        self.tokenizer = tokenizer
        self.text_input = text_input
        self._tokenized_text = _tokenize_input(
            tokenizer, text_input
        )

    def generate_operation(self) -> Dict[str, Any]:
        """Generate burst operation."""
        now = time.time()
        if (
            self.metrics.total_operations > 0
            and (now - self.last_burst_time)
            < self.burst_interval
            and self.burst_count >= self.burst_size
        ):
            time.sleep(
                self.burst_interval
                - (now - self.last_burst_time)
            )
            self.burst_count = 0
            self.last_burst_time = time.time()

        key = f"key_{random.randint(0, self.key_range)}"
        self.burst_count += 1
        return {"type": "store", "key": key}

    def execute_operation(
        self, operation: Dict[str, Any]
    ) -> bool:
        """Execute operation via direct engine call."""
        try:
            chunk_size = 256
            key_hash = abs(hash(operation["key"]))

            if (
                self._tokenized_text
                and len(self._tokenized_text) >= 2
            ):
                max_start = max(
                    0,
                    len(self._tokenized_text) - chunk_size,
                )
                start_idx = key_hash % max(1, max_start + 1)
            else:
                start_idx = 0

            token_ids = _build_token_ids(
                self._tokenized_text,
                start_idx,
                chunk_size,
                fallback_start=key_hash % 10000,
            )

            self.engine.store(
                token_ids,
                req_id=operation.get("key"),
            )
            blocks = (
                len(token_ids) // chunk_size
            ) or 1
            bpt = getattr(
                self.engine, "bytes_per_token", 0
            )
            operation["kv_blocks"] = blocks
            operation["data_bytes"] = (
                blocks * chunk_size * bpt
            )
            return True

        except Exception as e:
            if self.metrics.failed_operations < 3:
                print(
                    f"Engine operation failed: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
            return False


class SteadyStateWorkload(BaseWorkload):
    """Steady-state workload pattern."""

    def __init__(
        self,
        engine: Any = None,
        key_range: int = 1000,
        value_size: int = 1024,
        read_ratio: float = 0.8,
        tokenizer: Optional[Any] = None,
        text_input: Optional[str] = None,
    ):
        super().__init__(engine)
        self.key_range = key_range
        self.value_size = value_size
        self.read_ratio = read_ratio
        self.tokenizer = tokenizer
        self.text_input = text_input
        self._tokenized_text = _tokenize_input(
            tokenizer, text_input
        )
        self.populated_keys = set()

    def generate_operation(self) -> Dict[str, Any]:
        """Generate steady-state operation."""
        if random.random() < self.read_ratio:
            if self.populated_keys:
                key = random.choice(
                    list(self.populated_keys)
                )
            else:
                key = (
                    f"key_"
                    f"{random.randint(0, self.key_range)}"
                )
            op_type = "retrieve"
        else:
            key = (
                f"key_{random.randint(0, self.key_range)}"
            )
            self.populated_keys.add(key)
            op_type = "store"

        return {"type": op_type, "key": key}

    def execute_operation(
        self, operation: Dict[str, Any]
    ) -> bool:
        """Execute operation via direct engine call."""
        try:
            chunk_size = 256
            key_hash = abs(hash(operation["key"]))

            if (
                self._tokenized_text
                and len(self._tokenized_text) >= 2
            ):
                max_start = max(
                    0,
                    len(self._tokenized_text) - chunk_size,
                )
                start_idx = key_hash % max(1, max_start + 1)
            else:
                start_idx = 0

            token_ids = _build_token_ids(
                self._tokenized_text,
                start_idx,
                chunk_size,
                fallback_start=key_hash % 10000,
            )

            req_id = operation.get("key")
            bpt = getattr(
                self.engine, "bytes_per_token", 0
            )
            if operation["type"] == "store":
                self.engine.store(
                    token_ids, req_id=req_id
                )
                blocks = (
                    len(token_ids) // chunk_size
                ) or 1
                operation["kv_blocks"] = blocks
                operation["data_bytes"] = (
                    blocks * chunk_size * bpt
                )
                return True
            else:
                num_retrieved = self.engine.retrieve(
                    token_ids, req_id=req_id
                )
                cache_hit = num_retrieved > 0
                operation["cache_hit"] = cache_hit
                operation["kv_blocks"] = num_retrieved
                operation["data_bytes"] = (
                    num_retrieved * bpt
                )
                return True

        except Exception as e:
            if self.metrics.failed_operations < 3:
                print(
                    f"Engine operation failed: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
            return False


def _load_conversations(
    conversation_file: str,
    max_conversations: int = 0,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load conversations from a JSON file matching
    the LMCache conversation schema.

    Args:
        conversation_file: Path to JSON file
        max_conversations: Cap on conversations to
            load (0 = all)
        shuffle: Randomize conversation order
        seed: RNG seed for reproducible shuffle

    Returns:
        List of conversation dicts with 'id' and
        'turns'
    """
    file_path = Path(conversation_file)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Conversation file not found: "
            f"{conversation_file}"
        )
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    version = data.get("version", "")
    if version != "1.0":
        print(
            f"Warning: expected schema version "
            f"1.0, got '{version}'",
            file=sys.stderr,
        )

    conversations = data.get("conversations", [])
    if not conversations:
        raise ValueError(
            "No conversations found in "
            f"{conversation_file}"
        )

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(conversations)

    if max_conversations > 0:
        conversations = (
            conversations[:max_conversations]
        )

    total_turns = sum(
        len(c.get("turns", []))
        for c in conversations
    )
    print(
        f"Loaded {len(conversations)} conversations "
        f"({total_turns} turns)",
        file=sys.stderr,
    )
    return conversations


class ConversationWorkload(BaseWorkload):
    """Replay multi-turn conversations through the
    KV cache engine.

    Models real LLM prefix caching: each conversation
    builds a cumulative token context.  User turns
    trigger a retrieve (prefix cache lookup) and
    assistant turns trigger a store (cache the full
    context so far).

    Supports N concurrent conversation slots via the
    ``concurrency`` parameter.  Operations are drawn
    round-robin from the active slots so the cache
    sees interleaved access patterns that mirror a
    real inference server serving multiple users.
    """

    def __init__(
        self,
        engine: Any = None,
        conversation_file: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        concurrency: int = 1,
        max_conversations: int = 0,
        shuffle_conversations: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(engine)
        if not conversation_file:
            raise ValueError(
                "ConversationWorkload requires "
                "--conversation-file"
            )
        if not tokenizer:
            raise ValueError(
                "ConversationWorkload requires a "
                "tokenizer (use --tokenizer-mode "
                "text-to-tokens)"
            )
        self.tokenizer = tokenizer
        self._conversations = _load_conversations(
            conversation_file,
            max_conversations=max_conversations,
            shuffle=shuffle_conversations,
            seed=seed,
        )
        self._concurrency = max(1, concurrency)
        self._next_conv_idx = 0
        self._slot_cursor = 0
        self.conversations_completed = 0

        self._active_slots: List[
            List[Dict[str, Any]]
        ] = []
        for _ in range(self._concurrency):
            self._active_slots.append([])
            self._activate_slot(
                len(self._active_slots) - 1
            )

    def _activate_slot(self, slot_idx: int):
        """Fill *slot_idx* with the ops from the next
        conversation in the pool."""
        if self._next_conv_idx >= len(
            self._conversations
        ):
            self._next_conv_idx = 0

        conv = self._conversations[
            self._next_conv_idx
        ]
        self._next_conv_idx += 1

        ops: List[Dict[str, Any]] = []
        context_tokens: List[int] = []
        for turn in conv.get("turns", []):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if not content:
                continue

            turn_tokens = self.tokenizer.tokenize(
                content
            )
            context_tokens = (
                context_tokens + turn_tokens
            )

            if role == "user":
                ops.append({
                    "type": "retrieve",
                    "key": (
                        f"{conv['id']}_ctx_"
                        f"{len(context_tokens)}"
                    ),
                    "token_ids": list(
                        context_tokens
                    ),
                })
            elif role == "assistant":
                ops.append({
                    "type": "store",
                    "key": (
                        f"{conv['id']}_ctx_"
                        f"{len(context_tokens)}"
                    ),
                    "token_ids": list(
                        context_tokens
                    ),
                })

        self._active_slots[slot_idx] = ops

    def reset(self):
        """Reset state for a new pass over the
        conversation pool."""
        self._next_conv_idx = 0
        self._slot_cursor = 0
        self.conversations_completed = 0
        self._active_slots = []
        for _ in range(self._concurrency):
            self._active_slots.append([])
            self._activate_slot(
                len(self._active_slots) - 1
            )

    def generate_operation(self) -> Dict[str, Any]:
        """Round-robin across active slots, refilling
        exhausted slots from the conversation pool."""
        for _ in range(self._concurrency):
            idx = (
                self._slot_cursor % self._concurrency
            )
            self._slot_cursor += 1
            slot = self._active_slots[idx]
            if slot:
                op = slot.pop(0)
                if not slot:
                    self.conversations_completed += 1
                    self._activate_slot(idx)
                return op

        self._activate_slot(0)
        return self._active_slots[0].pop(0)

    def execute_operation(
        self, operation: Dict[str, Any]
    ) -> bool:
        """Execute a conversation-driven operation."""
        try:
            chunk_size = 256
            token_ids = operation.get(
                "token_ids", []
            )

            if not token_ids:
                return False

            req_id = operation.get("key")
            bpt = getattr(
                self.engine, "bytes_per_token", 0
            )
            if operation["type"] == "store":
                self.engine.store(
                    token_ids, req_id=req_id
                )
                blocks = max(
                    1, len(token_ids) // chunk_size
                )
                operation["kv_blocks"] = blocks
                operation["data_bytes"] = (
                    blocks * chunk_size * bpt
                )
                return True
            else:
                num_retrieved = self.engine.retrieve(
                    token_ids, req_id=req_id
                )
                cache_hit = num_retrieved > 0
                operation["cache_hit"] = cache_hit
                operation["kv_blocks"] = (
                    num_retrieved
                )
                operation["data_bytes"] = (
                    num_retrieved * bpt
                )
                return True

        except Exception as e:
            if self.metrics.failed_operations < 3:
                print(
                    f"Engine operation failed: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
            return False
