"""
SGLang serving engine adapter.

Handles SGLang-specific CLI formatting. Uses native SGLang argument names
directly - no translation needed.
"""

from typing import Dict, Any
from .base import ServingEngineBase


class SGLangEngine(ServingEngineBase):
    """
    SGLang serving engine adapter.

    Users specify SGLang args as they appear in SGLang documentation.
    SGLang uses simpler argument formats than vLLM (no complex JSON configs).
    """

    @property
    def name(self) -> str:
        return "sglang"

    @property
    def display_name(self) -> str:
        return "SGLang"

    @property
    def args_array_mode(self) -> str:
        """SGLang uses --key=value format in arrays."""
        return 'equals'

    def build_server_args(self, args: Dict[str, Any]) -> str:
        """
        Build SGLang CLI arguments from native SGLang argument names.

        Args:
            args: Dict with native SGLang argument names
                  e.g., {'max-running-requests': 1024, 'mem-fraction-static': 0.9}
                  Underscores are automatically converted to hyphens.

        Returns:
            Formatted CLI string
        """
        cli_parts = []

        for key, value in args.items():
            # Normalize key: underscores → hyphens (SGLang convention)
            normalized_key = key.replace("_", "-")

            # Format the argument using base class logic
            formatted = self.format_cli_arg(normalized_key, value)
            cli_parts.extend(formatted)

        # Join with line continuation
        return " \\\n            ".join(cli_parts)
