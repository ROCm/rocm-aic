"""
Base class for serving engine adapters.

This abstraction layer handles engine-specific CLI argument formatting.
Users specify native engine arguments as they appear in the engine's documentation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class ServingEngineBase(ABC):
    """
    Base class for serving engine adapters.

    Minimal abstraction focused solely on CLI argument formatting.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Engine identifier (lowercase).

        Returns:
            Engine name: 'vllm', 'sglang', etc.
        """
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """
        Human-readable engine name.

        Returns:
            Display name: 'vLLM', 'SGLang', etc.
        """
        pass

    @property
    def args_key(self) -> str:
        """
        Config key for engine-specific arguments in sweep configs.

        Returns:
            'vllm_args', 'sglang_args', etc.
        """
        return f"{self.name}_args"

    def format_cli_arg(self, key: str, value: Any) -> List[str]:
        """
        Format a single argument for CLI.

        Args:
            key: Native engine argument name
            value: Argument value

        Returns:
            List of CLI tokens

        Default behavior (override for special cases):
        - Booleans: ['--flag'] if True, [] if False
        - None: []
        - Others: ['--key', 'value']
        """
        if value is None:
            return []

        if isinstance(value, bool):
            return [f"--{key}"] if value else []

        return [f"--{key}", str(value)]

    @property
    def args_array_mode(self) -> str:
        """
        Mode for formatting args array.

        Returns:
            'separate': --key <value> (two array elements)
            'equals': --key=<value> (single array element)
        """
        return 'separate'  # Default mode

    def build_server_args_array(self, args: Dict[str, Any]) -> List[str]:
        """
        Convert engine-specific args dict to CLI arguments array.

        Args:
            args: Dictionary with native engine argument names

        Returns:
            List of CLI argument strings (suitable for YAML array format)
        """
        cli_parts = []
        mode = self.args_array_mode

        for key, value in args.items():
            # Normalize key: underscores → hyphens
            normalized_key = key.replace("_", "-")

            # Handle None values
            if value is None:
                continue

            # Handle boolean flags
            if isinstance(value, bool):
                if value:
                    # Boolean flags are quoted as YAML strings
                    cli_parts.append(f'"--{normalized_key}"')
                continue

            # Handle regular key-value arguments based on mode
            if mode == 'equals':
                # Mode 2: --key=value (single element), quoted as YAML string
                cli_parts.append(f'"--{normalized_key}={value}"')
            else:
                # Mode 1: --key <value> (two elements), each quoted as YAML string
                cli_parts.extend([f'"--{normalized_key}"', f'"{value}"'])

        return cli_parts

    @abstractmethod
    def build_server_args(self, args: Dict[str, Any]) -> str:
        """
        Convert engine-specific args dict to CLI arguments string.

        Args:
            args: Dictionary with native engine argument names

        Returns:
            Formatted CLI arguments string
        """
        pass
