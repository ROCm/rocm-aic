"""Configuration generator for LMCache YAML config files."""
import json
from pathlib import Path
from typing import Any, Dict

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

        Uses ``remote_storage_plugins: [fs]`` and
        ``extra_config.remote_storage_plugin.fs.base_path`` so LMCache does
        not emit the legacy ``remote_url`` deprecation warning.

        Args:
            storage_path: Path to storage directory
            chunk_size: KV cache chunk size
            local_cpu: Enable CPU caching
            max_local_cpu_size: Max CPU cache size in GB
            **kwargs: Additional configuration options

        Returns:
            Configuration dictionary
        """
        extra: Dict[str, Any] = {
            "remote_storage_plugin.fs.base_path": (
                Path(storage_path).resolve().as_posix().rstrip("/") + "/"
            ),
        }
        config = self.DEFAULT_CONFIG.copy()
        config.update(
            {
                "chunk_size": chunk_size,
                "local_cpu": local_cpu,
                "max_local_cpu_size": max_local_cpu_size,
                "remote_storage_plugins": ["fs"],
                "extra_config": extra,
            }
        )
        config.update(kwargs)
        return config

    def generate_local_disk_config(
        self,
        storage_path: str,
        chunk_size: int = 256,
        local_cpu: bool = True,
        max_local_cpu_size: float = 5.0,
        max_local_disk_size: float = 5.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """LMCache local_disk tier using file:// URLs (POSIX disk offload)."""
        resolved = Path(storage_path).resolve()
        file_url = f"file://{resolved.as_posix()}/"
        config = self.DEFAULT_CONFIG.copy()
        config.update(
            {
                "chunk_size": chunk_size,
                "local_cpu": local_cpu,
                "max_local_cpu_size": max_local_cpu_size,
                "local_disk": file_url,
                "max_local_disk_size": max_local_disk_size,
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

        Uses the same ``fs`` remote plugin layout as
        :meth:`generate_filesystem_config` (see that docstring).

        Args:
            mount_point: Mount point path
            chunk_size: KV cache chunk size
            local_cpu: Enable CPU caching
            max_local_cpu_size: Max CPU cache size in GB
            **kwargs: Additional configuration options

        Returns:
            Configuration dictionary
        """
        extra: Dict[str, Any] = {
            "remote_storage_plugin.fs.base_path": (
                Path(mount_point).resolve().as_posix().rstrip("/") + "/"
            ),
        }
        config = self.DEFAULT_CONFIG.copy()
        config.update(
            {
                "chunk_size": chunk_size,
                "local_cpu": local_cpu,
                "max_local_cpu_size": max_local_cpu_size,
                "remote_storage_plugins": ["fs"],
                "extra_config": extra,
            }
        )
        config.update(kwargs)
        return config

    def generate_gds_config(
        self,
        gds_path: str,
        chunk_size: int = 256,
        local_cpu: bool = False,
        gds_buffer_size: int = 8192,
        gds_backend: str = "cufile",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate configuration for GDS (GPU Direct Storage) backend.

        Args:
            gds_path: Path to GDS storage
            chunk_size: KV cache chunk size
            local_cpu: Enable CPU caching
            gds_buffer_size: GDS buffer size in MiB
            gds_backend: GDS library ("cufile" or "hipfile")
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
                "gds_buffer_size": gds_buffer_size,
                "gds_backend": gds_backend,
            }
        )
        config.update(kwargs)
        return config

    def generate_redis_config(
        self,
        remote_url: str,
        chunk_size: int = 256,
        local_cpu: bool = False,
        max_local_cpu_size: float = 5.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """LMCache config with Redis (or Sentinel) remote_url."""
        config = self.DEFAULT_CONFIG.copy()
        config.update(
            {
                "chunk_size": chunk_size,
                "local_cpu": local_cpu,
                "max_local_cpu_size": max_local_cpu_size,
                "remote_url": remote_url,
            }
        )
        config.update(kwargs)
        return config

    def generate_s3_config(
        self,
        remote_url: str,
        s3_region: str,
        chunk_size: int = 256,
        local_cpu: bool = False,
        max_local_cpu_size: float = 5.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """LMCache config for S3 or S3-compatible object storage."""
        extra = {
            "s3_region": s3_region,
            "s3_num_io_threads": 64,
            "save_chunk_meta": False,
        }
        kw_ex = kwargs.pop("extra_config", None)
        if isinstance(kw_ex, dict):
            extra.update(kw_ex)
        config = self.DEFAULT_CONFIG.copy()
        config.update(
            {
                "chunk_size": chunk_size,
                "local_cpu": local_cpu,
                "max_local_cpu_size": max_local_cpu_size,
                "save_unfull_chunk": False,
                "remote_url": remote_url,
                "extra_config": extra,
            }
        )
        config.update(kwargs)
        return config

    def generate_remote_config(
        self,
        remote_url: str,
        chunk_size: int = 256,
        local_cpu: bool = False,
        max_local_cpu_size: float = 5.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generic remote_url-only config (lm://, mooncakestore://, …)."""
        config = self.DEFAULT_CONFIG.copy()
        config.update(
            {
                "chunk_size": chunk_size,
                "local_cpu": local_cpu,
                "max_local_cpu_size": max_local_cpu_size,
                "remote_url": remote_url,
            }
        )
        config.update(kwargs)
        return config

    #: LMCache ``extra_config`` keys for POSIX O_DIRECT on the ``fs``
    #: remote connector (filesystem and block-device backends). Chunk meta
    #: must be off for block-aligned direct I/O in upstream LMCache.
    FS_ODIRECT_EXTRA_CONFIG: Dict[str, Any] = {
        "fs_connector_use_odirect": True,
        "save_chunk_meta": False,
    }

    def apply_fs_odirect_extra_config(
        self,
        base: Dict[str, Any],
    ) -> None:
        """Merge :attr:`FS_ODIRECT_EXTRA_CONFIG` into ``base['extra_config']``."""
        self.merge_config_fragment(
            base,
            {"extra_config": dict(self.FS_ODIRECT_EXTRA_CONFIG)},
        )

    @staticmethod
    def merge_config_fragment(
        base: Dict[str, Any],
        fragment: Dict[str, Any],
    ) -> None:
        """Deep-merge fragment into base (nested extra_config)."""
        for key, val in fragment.items():
            if key == "extra_config" and isinstance(val, dict):
                existing = base.get("extra_config")
                if isinstance(existing, dict):
                    merged = existing.copy()
                    merged.update(val)
                    base["extra_config"] = merged
                else:
                    base["extra_config"] = dict(val)
            else:
                base[key] = val

    def merge_from_path(
        self,
        base: Dict[str, Any],
        path: str,
    ) -> None:
        """Load YAML or JSON object from path and merge into base."""
        p = Path(path)
        with open(p, encoding="utf-8") as f:
            if p.suffix.lower() in (".yaml", ".yml"):
                fragment = yaml.safe_load(f)
            else:
                fragment = json.load(f)
        if not isinstance(fragment, dict):
            raise ValueError(
                "extra-config file must contain a YAML/JSON object "
                "at the root"
            )
        self.merge_config_fragment(base, fragment)

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
