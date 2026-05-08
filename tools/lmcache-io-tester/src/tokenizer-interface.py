"""Tokenizer interface for Hugging Face models."""
from typing import Optional, List, Union
from pathlib import Path

try:
    from transformers import PreTrainedTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    PreTrainedTokenizer = None


class TokenizerWrapper:
    """Wrapper around Hugging Face tokenizer."""

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        mode: str = "vocab-only",
    ):
        """
        Initialize tokenizer wrapper.

        Args:
            tokenizer: Hugging Face tokenizer instance
            mode: "vocab-only" or "text-to-tokens"
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required. "
                "Install with: pip install transformers"
            )

        if mode not in ["vocab-only", "text-to-tokens"]:
            raise ValueError(
                f"Invalid mode: {mode}. "
                "Must be 'vocab-only' or 'text-to-tokens'"
            )

        self.tokenizer = tokenizer
        self.mode = mode

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        return self.tokenizer.vocab_size

    def tokenize(self, text: str) -> List[int]:
        """
        Convert text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        if self.mode == "vocab-only":
            raise RuntimeError(
                "Tokenizer is in vocab-only mode. "
                "Cannot perform tokenization."
            )

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        # Use encode to get token IDs
        return self.tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        if self.mode == "vocab-only":
            raise RuntimeError(
                "Tokenizer is in vocab-only mode. "
                "Cannot perform detokenization."
            )

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def get_token_range(
        self,
        text: str,
        start_pos: Optional[int] = None,
        end_pos: Optional[int] = None,
    ) -> tuple[int, int]:
        """
        Get token range for a text segment.

        Args:
            text: Input text
            start_pos: Start character position (optional)
            end_pos: End character position (optional)

        Returns:
            Tuple of (start_token_id, end_token_id)
        """
        if self.mode == "vocab-only":
            raise RuntimeError(
                "Tokenizer is in vocab-only mode. "
                "Cannot perform tokenization."
            )

        # Tokenize the full text
        tokens = self.tokenize(text)

        if not tokens:
            return (0, 0)

        # If positions specified, approximate token range
        if start_pos is not None or end_pos is not None:
            # Simple approximation: map character positions to token positions
            # This is approximate and may not be exact
            text_len = len(text)
            if start_pos is None:
                start_pos = 0
            if end_pos is None:
                end_pos = text_len

            # Approximate token positions based on character ratio
            start_token_idx = int(
                (start_pos / text_len) * len(tokens)
            ) if text_len > 0 else 0
            end_token_idx = int(
                (end_pos / text_len) * len(tokens)
            ) if text_len > 0 else len(tokens)

            start_token_idx = max(0, min(start_token_idx, len(tokens) - 1))
            end_token_idx = max(
                start_token_idx + 1, min(end_token_idx, len(tokens))
            )

            return (tokens[start_token_idx], tokens[end_token_idx - 1])

        # Return full range
        return (tokens[0], tokens[-1])

    def tokenize_text_file(self, file_path: Union[str, Path]) -> List[int]:
        """
        Tokenize text from a file.

        Args:
            file_path: Path to text file

        Returns:
            List of token IDs
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        return self.tokenize(text)

    def get_chunk_tokens(
        self,
        text: str,
        chunk_size: int = 256,
        start_offset: int = 0,
    ) -> tuple[int, int]:
        """
        Get token range for a chunk of text.

        Args:
            text: Input text
            chunk_size: Number of tokens in chunk
            start_offset: Starting token offset

        Returns:
            Tuple of (start_token_id, end_token_id)
        """
        if self.mode == "vocab-only":
            raise RuntimeError(
                "Tokenizer is in vocab-only mode. "
                "Cannot perform tokenization."
            )

        tokens = self.tokenize(text)

        if len(tokens) == 0:
            return (0, 0)

        # Clamp start_offset
        start_idx = max(0, min(start_offset, len(tokens) - 1))
        end_idx = min(start_idx + chunk_size, len(tokens))

        if start_idx >= len(tokens):
            return (tokens[-1], tokens[-1])

        return (tokens[start_idx], tokens[end_idx - 1])
