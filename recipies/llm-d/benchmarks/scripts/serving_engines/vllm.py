"""
vLLM serving engine adapter.

Handles vLLM-specific CLI formatting, including special cases like
kv-transfer-config JSON formatting.
"""

from typing import Dict, Any, List
import json
from .base import ServingEngineBase


class VLLMEngine(ServingEngineBase):
    """vLLM serving engine adapter."""

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def display_name(self) -> str:
        return "vLLM"

    @property
    def args_array_mode(self) -> str:
        """vLLM uses --key=value format in arrays."""
        return 'equals'

    def format_cli_arg(self, key: str, value: Any) -> List[str]:
        """
        Format vLLM argument with special handling for kv-transfer-config.

        vLLM uses native argument names, but kv-transfer-config needs
        special JSON formatting.
        """
        # Special handling for kv-transfer-config (accepts dict or string)
        if key == "kv-transfer-config":
            return self._format_kv_transfer_config(value)

        # Use base class formatting for everything else
        return super().format_cli_arg(key, value)

    def _expand_kv_connector(self, connector_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expand kv_connector shorthand to full kv-transfer-config JSON.

        Converts shorthand like:
            {'type': 'lmcache', 'role': 'kv_both'}
        To vLLM's expected format:
            {'kv_connector': 'LMCacheConnectorV1', 'kv_role': 'kv_both'}

        Args:
            connector_config: Shorthand connector config dict

        Returns:
            Full kv-transfer-config dict for vLLM
        """
        if not connector_config:
            return {}

        # Handle raw JSON pass-through
        if "raw_json" in connector_config:
            return json.loads(connector_config["raw_json"])

        connector_type = connector_config.get("type")
        role = connector_config.get("role", "kv_both")

        if connector_type == "offloading":
            # Offloading connector
            cpu_bytes = connector_config.get("cpu_bytes")
            kv_config = {
                "kv_connector": "OffloadingConnector",
                "kv_role": role,
                "kv_connector_extra_config": {
                    "cpu_bytes_to_use": cpu_bytes
                }
            }
        elif connector_type == "lmcache":
            # LMCache connector
            kv_config = {
                "kv_connector": "LMCacheConnectorV1",
                "kv_role": role
            }
        else:
            raise ValueError(f"Unknown kv_connector type: {connector_type}")

        return kv_config

    def _format_kv_transfer_config(self, value: Any) -> List[str]:
        """
        Format kv-transfer-config argument.

        Accepts either:
        1. A string (raw JSON) - pass through
        2. A dict - convert to JSON string
        """
        if value is None:
            return []

        if isinstance(value, str):
            # Raw JSON string - use as-is
            json_str = value
        elif isinstance(value, dict):
            # Dict - convert to JSON
            json_str = json.dumps(value)
        else:
            raise ValueError(
                f"kv-transfer-config must be string or dict, got {type(value)}"
            )

        return ["--kv-transfer-config", f"'{json_str}'"]

    def build_server_args_array(self, args: Dict[str, Any]) -> List[str]:
        """
        Convert vLLM args dict to CLI arguments array.

        Overrides base class to handle:
        - kv_connector shorthand expansion to kv-transfer-config
        - kv-transfer-config JSON formatting

        Args:
            args: Dictionary with native vLLM argument names

        Returns:
            List of CLI argument strings (suitable for YAML array format)
        """
        # First, expand kv_connector shorthand if present
        processed_args = args.copy()
        if "kv_connector" in processed_args or "kv-connector" in processed_args:
            kv_connector = processed_args.pop("kv_connector", None) or processed_args.pop("kv-connector", None)
            if kv_connector:
                # Expand to full kv-transfer-config
                expanded = self._expand_kv_connector(kv_connector)
                processed_args["kv-transfer-config"] = expanded

        cli_parts = []

        for key, value in processed_args.items():
            # Normalize key: underscores → hyphens
            normalized_key = key.replace("_", "-")

            # Handle None values
            if value is None:
                continue

            # Special handling for kv-transfer-config (needs JSON formatting)
            if normalized_key == "kv-transfer-config":
                if isinstance(value, str):
                    json_str = value
                elif isinstance(value, dict):
                    json_str = json.dumps(value)
                else:
                    raise ValueError(
                        f"kv-transfer-config must be string or dict, got {type(value)}"
                    )
                # Use single quotes around the whole arg for YAML compatibility
                # Format: '--kv-transfer-config={"json":"here"}'
                cli_parts.append(f"'--{normalized_key}={json_str}'")
                continue

            # Handle boolean flags
            if isinstance(value, bool):
                if value:
                    cli_parts.append(f"--{normalized_key}")
                continue

            # Handle regular key-value arguments (--key=value format)
            cli_parts.append(f"--{normalized_key}={value}")

        return cli_parts

    def build_server_args(self, args: Dict[str, Any]) -> str:
        """
        Build vLLM CLI arguments from native vLLM argument names.

        Args:
            args: Dict with native vLLM argument names
                  e.g., {'max-num-seq': 1024, 'gpu-memory-utilization': 0.9}
                  Underscores are automatically converted to hyphens.

        Returns:
            Formatted CLI string
        """
        # First, expand kv_connector shorthand if present
        processed_args = args.copy()
        if "kv_connector" in processed_args or "kv-connector" in processed_args:
            kv_connector = processed_args.pop("kv_connector", None) or processed_args.pop("kv-connector", None)
            if kv_connector:
                # Expand to full kv-transfer-config
                expanded = self._expand_kv_connector(kv_connector)
                processed_args["kv-transfer-config"] = expanded

        cli_parts = []

        for key, value in processed_args.items():
            # Normalize key: underscores → hyphens (vLLM convention)
            normalized_key = key.replace("_", "-")

            # Format the argument
            formatted = self.format_cli_arg(normalized_key, value)
            cli_parts.extend(formatted)

        # Join with line continuation for readability
        return " \\\n            ".join(cli_parts)
