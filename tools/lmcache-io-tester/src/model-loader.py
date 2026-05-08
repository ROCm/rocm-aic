"""Model loader for Hugging Face models."""
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

try:
    from transformers import AutoTokenizer, AutoConfig
    from huggingface_hub import snapshot_download, login
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class KVCacheParams:
    """KV cache parameters extracted from model config."""

    num_layers: int
    num_heads: int
    hidden_size: int
    head_dim: int
    vocab_size: int
    dtype: str = "float16"


class KVCacheCalculator:
    """Calculate KV cache shape and size from model parameters."""

    @staticmethod
    def calculate_shape(
        num_layers: int,
        num_heads: int,
        head_dim: int,
        chunk_size: int,
    ) -> Tuple[int, ...]:
        """
        Calculate KV cache shape.

        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension per attention head
            chunk_size: Cache chunk size

        Returns:
            Shape tuple: [num_layers, 2, chunk_size, num_heads, head_dim]
        """
        return (num_layers, 2, chunk_size, num_heads, head_dim)

    @staticmethod
    def calculate_size(
        shape: Tuple[int, ...],
        dtype: str = "float16",
    ) -> int:
        """
        Calculate KV cache size in bytes.

        Args:
            shape: KV cache shape tuple
            dtype: Data type (float16, float32, bfloat16, uint8)

        Returns:
            Size in bytes
        """
        dtype_sizes = {
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
            "uint8": 1,
        }
        dtype_size = dtype_sizes.get(dtype, 2)

        total_elements = 1
        for dim in shape:
            total_elements *= dim

        return total_elements * dtype_size

    @staticmethod
    def get_dtype_from_config(config: Any) -> str:
        """
        Infer dtype from model config.

        Args:
            config: Model configuration object

        Returns:
            Dtype string
        """
        # Check common dtype attributes
        dtype_attrs = [
            "torch_dtype",
            "dtype",
            "model_type",
        ]

        for attr in dtype_attrs:
            if hasattr(config, attr):
                dtype_val = getattr(config, attr)
                if isinstance(dtype_val, str):
                    dtype_val = dtype_val.lower()
                    if "float16" in dtype_val or "half" in dtype_val:
                        return "float16"
                    elif "float32" in dtype_val or "float" in dtype_val:
                        return "float32"
                    elif "bfloat16" in dtype_val:
                        return "bfloat16"

        # Default to float16 for efficiency
        return "float16"


class ModelLoader:
    """Load models from Hugging Face and extract metadata."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_only: bool = False,
        token_file: Optional[str] = None,
    ):
        """
        Initialize model loader.

        Args:
            model_name: Hugging Face model identifier
            model_path: Local path to model (overrides model_name)
            cache_dir: Directory to cache models
            local_only: Only use local models, don't download
            token_file: Path to Hugging Face token file (optional)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and huggingface-hub are required. "
                "Install with: pip install transformers huggingface-hub"
            )

        self.model_name = model_name
        self.model_path = model_path
        self.cache_dir = cache_dir or os.path.expanduser(
            "~/.cache/huggingface"
        )
        self.local_only = local_only
        self.token_file = token_file

        self._tokenizer = None
        self._config = None
        self._kv_cache_params = None
        self._hf_token = None

        # Load token if provided or search for token files
        # This must be called before any HF API calls
        self._load_hf_token()

    def load_model(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> None:
        """
        Load model tokenizer and config.

        Args:
            model_name: Hugging Face model identifier
            model_path: Local path to model
        """
        if model_path:
            self.model_path = model_path
            self.model_name = None
        elif model_name:
            self.model_name = model_name
            self.model_path = None
        elif self.model_path:
            pass  # Use existing model_path
        elif self.model_name:
            pass  # Use existing model_name
        else:
            raise ValueError(
                "Either model_name or model_path must be provided"
            )

        # Determine model path
        if self.model_path:
            model_path_str = self.model_path
        else:
            if self.local_only:
                # Use snapshot_download with local_files_only
                try:
                    model_path_str = snapshot_download(
                        repo_id=self.model_name,
                        cache_dir=self.cache_dir,
                        local_files_only=True,
                    )
                except Exception:
                    raise FileNotFoundError(
                        f"Model {self.model_name} not found in cache. "
                        "Remove --local-only to download."
                    )
            else:
                # Download or use cached model
                try:
                    # Use token if available, or check environment
                    token = (
                        self._hf_token
                        if self._hf_token
                        else os.getenv("HF_TOKEN")
                        or os.getenv("HUGGING_FACE_HUB_TOKEN")
                        or None
                    )
                    model_path_str = snapshot_download(
                        repo_id=self.model_name,
                        cache_dir=self.cache_dir,
                        token=token,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load model {self.model_name}: {e}"
                    ) from e

        # Load tokenizer and config
        try:
            # Use token if available
            token = self._hf_token if self._hf_token else None
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path_str,
                trust_remote_code=True,
                token=token,
            )
            self._config = AutoConfig.from_pretrained(
                model_path_str,
                trust_remote_code=True,
                token=token,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer/config from {model_path_str}: {e}"
            ) from e

    def _load_hf_token(self) -> None:
        """Load Hugging Face token from file or environment."""
        # Check if token is already set in environment
        if os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"):
            self._hf_token = (
                os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            )
            return

        # Try to find token file
        token_file = self.token_file
        if not token_file:
            # Look for token files in current directory
            current_dir = Path.cwd()
            token_files = list(current_dir.glob("*.token"))
            token_files.extend(list(current_dir.glob(".huggingface*")))
            token_files.extend(
                list(current_dir.glob(".batesste-hugging-face-*.token"))
            )

            if token_files:
                # Use the first token file found
                token_file = str(token_files[0])

        if token_file:
            token_path = Path(token_file)
            # Resolve relative paths - try multiple approaches
            if not token_path.is_absolute():
                # First try as-is relative to current directory
                if not token_path.exists():
                    # Try with current directory prepended
                    token_path = Path.cwd() / token_path
                if not token_path.exists():
                    # Try just the filename in current directory
                    token_path = Path.cwd() / Path(token_file).name
            if token_path.exists():
                try:
                    with open(token_path, "r") as f:
                        token = f.read().strip()
                        if token:
                            self._hf_token = token
                            # Set environment variable for huggingface_hub
                            os.environ["HF_TOKEN"] = token
                            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
                            # Also try to login if login function is available
                            try:
                                if TRANSFORMERS_AVAILABLE:
                                    login(token=token, add_to_git_credential=False)
                            except Exception:
                                # Login might fail if already logged in, ignore
                                pass
                except Exception as e:
                    import sys
                    print(
                        f"Warning: Failed to load token from {token_path}: {e}",
                        file=sys.stderr,
                    )

    def _find_cached_model(self, model_name: str) -> Optional[str]:
        """Find model in cache directory."""
        cache_path = Path(self.cache_dir) / "hub"
        if not cache_path.exists():
            return None

        # Look for model in cache
        for repo_dir in cache_path.iterdir():
            if repo_dir.is_dir():
                # Check if this matches the model name
                repo_name = repo_dir.name
                if model_name.replace("/", "--") in repo_name:
                    return str(repo_dir)

        return None

    def get_tokenizer(self):
        """Get tokenizer instance."""
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._tokenizer

    def get_config(self):
        """Get model configuration."""
        if self._config is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._config

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._tokenizer.vocab_size

    def get_kv_cache_params(
        self,
        chunk_size: int = 256,
        dtype: Optional[str] = None,
    ) -> KVCacheParams:
        """
        Extract KV cache parameters from model config.

        Args:
            chunk_size: Cache chunk size
            dtype: Override dtype (optional)

        Returns:
            KVCacheParams object
        """
        if self._config is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self._kv_cache_params is not None:
            return self._kv_cache_params

        config = self._config

        # Extract parameters based on architecture
        num_layers = None
        num_heads = None
        hidden_size = None

        # Try common attribute names
        layer_attrs = [
            "num_hidden_layers",
            "n_layer",
            "num_layers",
            "n_layers",
        ]
        head_attrs = [
            "num_attention_heads",
            "n_head",
            "num_heads",
            "n_heads",
        ]
        hidden_attrs = [
            "hidden_size",
            "n_embd",
            "d_model",
            "model_dim",
        ]

        for attr in layer_attrs:
            if hasattr(config, attr):
                num_layers = getattr(config, attr)
                break

        for attr in head_attrs:
            if hasattr(config, attr):
                num_heads = getattr(config, attr)
                break

        for attr in hidden_attrs:
            if hasattr(config, attr):
                hidden_size = getattr(config, attr)
                break

        # Validate we got required parameters
        if num_layers is None:
            raise ValueError(
                f"Could not extract num_layers from config. "
                f"Available attributes: {dir(config)}"
            )
        if num_heads is None:
            raise ValueError(
                f"Could not extract num_heads from config. "
                f"Available attributes: {dir(config)}"
            )
        if hidden_size is None:
            raise ValueError(
                f"Could not extract hidden_size from config. "
                f"Available attributes: {dir(config)}"
            )

        # Calculate head dimension
        head_dim = hidden_size // num_heads

        # Get dtype
        if dtype is None:
            dtype = KVCacheCalculator.get_dtype_from_config(config)

        # Get vocab size
        vocab_size = self.get_vocab_size()

        self._kv_cache_params = KVCacheParams(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
            head_dim=head_dim,
            vocab_size=vocab_size,
            dtype=dtype,
        )

        return self._kv_cache_params

    def calculate_kv_shape(
        self,
        chunk_size: int = 256,
        dtype: Optional[str] = None,
    ) -> Tuple[int, ...]:
        """
        Calculate KV cache shape for this model.

        Args:
            chunk_size: Cache chunk size
            dtype: Override dtype (optional)

        Returns:
            Shape tuple
        """
        params = self.get_kv_cache_params(chunk_size=chunk_size, dtype=dtype)
        return KVCacheCalculator.calculate_shape(
            num_layers=params.num_layers,
            num_heads=params.num_heads,
            head_dim=params.head_dim,
            chunk_size=chunk_size,
        )

    def calculate_kv_size(
        self,
        chunk_size: int = 256,
        dtype: Optional[str] = None,
    ) -> int:
        """
        Calculate KV cache size in bytes for this model.

        Args:
            chunk_size: Cache chunk size
            dtype: Override dtype (optional)

        Returns:
            Size in bytes
        """
        shape = self.calculate_kv_shape(chunk_size=chunk_size, dtype=dtype)
        params = self.get_kv_cache_params(chunk_size=chunk_size, dtype=dtype)
        return KVCacheCalculator.calculate_size(shape, params.dtype)
