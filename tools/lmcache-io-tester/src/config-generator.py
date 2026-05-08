"""Configuration generator for LMCache YAML config files."""
import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from jinja2 import Template


class ConfigGenerator:
    """Generates LMCache configuration files."""

    DEFAULT_CONFIG = {
        "chunk_size": 256,
        "local_cpu": False,
        "max_local_cpu_size": 5.0,
        "remote_serde": "naive",
        "blocking_timeout_secs": 10,
        "cache_policy": "LRU",
    }

    def __init__(self):
        self.template_dir = Path(__file__).parent / "templates"

    def generate_filesystem_config(
        self,
        storage_path: str,
        chunk_size: int = 256,
        local_cpu: bool = False,
        max_local_cpu_size: float = 5.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate configuration for filesystem storage.

        Args:
            storage_path: Path to storage directory
            chunk_size: KV cache chunk size
            local_cpu: Enable CPU caching
            max_local_cpu_size: Max CPU cache size in GB
            **kwargs: Additional configuration options

        Returns:
            Configuration dictionary
        """
        config = self.DEFAULT_CONFIG.copy()
        config.update(
            {
                "chunk_size": chunk_size,
                "local_cpu": local_cpu,
                "max_local_cpu_size": max_local_cpu_size,
                "remote_url": f"fs://host:0{storage_path}/",
            }
        )
        config.update(kwargs)
        return config

    def generate_block_device_config(
        self,
        mount_point: str,
        chunk_size: int = 256,
        local_cpu: bool = False,
        max_local_cpu_size: float = 5.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate configuration for block device storage.

        Args:
            mount_point: Mount point path
            chunk_size: KV cache chunk size
            local_cpu: Enable CPU caching
            max_local_cpu_size: Max CPU cache size in GB
            **kwargs: Additional configuration options

        Returns:
            Configuration dictionary
        """
        config = self.DEFAULT_CONFIG.copy()
        config.update(
            {
                "chunk_size": chunk_size,
                "local_cpu": local_cpu,
                "max_local_cpu_size": max_local_cpu_size,
                "remote_url": f"fs://host:0{mount_point}/",
            }
        )
        config.update(kwargs)
        return config

    def generate_gds_config(
        self,
        gds_path: str,
        chunk_size: int = 256,
        local_cpu: bool = False,
        cufile_buffer_size: int = 8192,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate configuration for GDS (GPU Direct Storage) backend.

        Args:
            gds_path: Path to GDS storage
            chunk_size: KV cache chunk size
            local_cpu: Enable CPU caching
            cufile_buffer_size: CuFile buffer size in MiB
            **kwargs: Additional configuration options

        Returns:
            Configuration dictionary
        """
        config = self.DEFAULT_CONFIG.copy()
        config.update(
            {
                "chunk_size": chunk_size,
                "local_cpu": local_cpu,
                "gds_path": gds_path,
                "cufile_buffer_size": cufile_buffer_size,
            }
        )
        config.update(kwargs)
        return config

    def save_config(self, config: Dict[str, Any], output_path: str):
        """
        Save configuration to YAML file.

        Args:
            config: Configuration dictionary
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Configuration file path

        Returns:
            Configuration dictionary
        """
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def generate_from_template(
        self,
        template_name: str,
        output_path: str,
        **template_vars,
    ):
        """
        Generate config from Jinja2 template.

        Args:
            template_name: Template file name
            output_path: Output file path
            **template_vars: Template variables
        """
        template_path = self.template_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {template_path}"
            )

        with open(template_path, "r") as f:
            template = Template(f.read())

        output = template.render(**template_vars)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            f.write(output)
