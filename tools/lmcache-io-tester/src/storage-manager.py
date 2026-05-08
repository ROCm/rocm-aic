"""Storage manager for handling filesystem and block device setup."""
import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse


_DEFAULT_SCHEME_PORTS = {
    "redis": 6379,
    "lm": 65432,
    "http": 80,
    "https": 443,
    "mooncakestore": 50051,
    "infinistore": 12345,
}


class StorageManager:
    """Manages storage setup for filesystem and block devices."""

    def __init__(self):
        self.mount_points: dict[str, str] = {}
        self.temp_dirs: list[str] = []

    def probe_tcp(
        self,
        host: str,
        port: int,
        timeout: float = 2.0,
    ) -> Tuple[bool, str]:
        """Try TCP connect to host:port."""
        try:
            with socket.create_connection(
                (host, port), timeout=timeout
            ):
                return True, f"tcp ok {host}:{port}"
        except OSError as e:
            return False, f"tcp {host}:{port}: {e}"

    def probe_remote_url(
        self,
        url: str,
        timeout: float = 2.0,
    ) -> Tuple[bool, str]:
        """Best-effort TCP reachability for LMCache remote_url.

        Skips ``s3://`` (use AWS checks separately). Handles
        ``redis-sentinel://h1:26379,h2:26379`` by probing the
        first endpoint only.
        """
        raw = url.strip()
        lower = raw.lower()
        if lower.startswith("s3://"):
            return True, "probe skipped for s3://"
        if lower.startswith("redis-sentinel://"):
            body = raw.split("://", 1)[1]
            first = body.split(",")[0].strip()
            if ":" in first:
                host, _, ps = first.rpartition(":")
                try:
                    port = int(ps)
                except ValueError:
                    return False, f"bad sentinel endpoint {first!r}"
            else:
                host, port = first, 26379
            return self.probe_tcp(host, port, timeout)
        parsed = urlparse(raw)
        scheme = (parsed.scheme or "").lower()
        host = parsed.hostname
        if not host:
            return False, "remote_url has no host"
        port = parsed.port
        if port is None:
            port = _DEFAULT_SCHEME_PORTS.get(scheme)
        if port is None:
            return (
                False,
                f"cannot infer TCP port for scheme {scheme!r}; "
                "include :port in remote_url",
            )
        return self.probe_tcp(host, port, timeout)

    def validate_filesystem_path(self, path: str) -> Tuple[bool, str]:
        """
        Validate that a filesystem path exists and is writable.

        Args:
            path: Path to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        path_obj = Path(path)
        if not path_obj.exists():
            return False, f"Path does not exist: {path}"
        if not path_obj.is_dir():
            return False, f"Path is not a directory: {path}"
        if not os.access(path, os.W_OK):
            return False, f"Path is not writable: {path}"
        return True, ""

    def validate_block_device(self, device: str) -> Tuple[bool, str]:
        """
        Validate that a block device exists.

        Args:
            device: Block device path (e.g., /dev/nvme0n1)

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not os.path.exists(device):
            return False, f"Block device does not exist: {device}"
        if not os.path.isblock(device):
            return False, f"Path is not a block device: {device}"
        return True, ""

    def create_filesystem(
        self, device: str, filesystem: str = "ext4"
    ) -> Tuple[bool, str]:
        """
        Create a filesystem on a block device.

        Args:
            device: Block device path
            filesystem: Filesystem type (default: ext4)

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Check if device is already mounted
            result = subprocess.run(
                ["findmnt", "-n", "-o", "TARGET", device],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                return False, f"Device {device} is already mounted"

            # Create filesystem
            subprocess.run(
                ["mkfs", "-t", filesystem, device],
                check=True,
                capture_output=True,
            )
            return True, ""
        except subprocess.CalledProcessError as e:
            return False, f"Failed to create filesystem: {e.stderr.decode()}"
        except FileNotFoundError:
            return False, "mkfs command not found"

    def mount_block_device(
        self,
        device: str,
        mount_point: Optional[str] = None,
        create_fs: bool = False,
        filesystem: str = "ext4",
    ) -> Tuple[Optional[str], str]:
        """
        Mount a block device to a mount point.

        Args:
            device: Block device path
            mount_point: Mount point path (creates temp dir if None)
            create_fs: Whether to create filesystem if needed
            filesystem: Filesystem type if creating

        Returns:
            Tuple of (mount_point_path, error_message)
        """
        # Validate device
        is_valid, error = self.validate_block_device(device)
        if not is_valid:
            return None, error

        # Create mount point if not provided
        if mount_point is None:
            temp_dir = tempfile.mkdtemp(prefix="lmcache_")
            mount_point = temp_dir
            self.temp_dirs.append(temp_dir)
        else:
            Path(mount_point).mkdir(parents=True, exist_ok=True)

        # Check if already mounted
        if device in self.mount_points:
            return self.mount_points[device], ""

        # Check if device has filesystem
        try:
            result = subprocess.run(
                ["blkid", device],
                capture_output=True,
                text=True,
            )
            has_fs = result.returncode == 0 and result.stdout.strip()
        except FileNotFoundError:
            # blkid not available, try mounting anyway
            has_fs = False

        # Create filesystem if needed
        if not has_fs and create_fs:
            success, error = self.create_filesystem(device, filesystem)
            if not success:
                return None, error

        # Mount the device
        try:
            subprocess.run(
                ["mount", device, mount_point],
                check=True,
                capture_output=True,
            )
            self.mount_points[device] = mount_point
            return mount_point, ""
        except subprocess.CalledProcessError as e:
            return None, f"Failed to mount device: {e.stderr.decode()}"
        except PermissionError:
            return None, "Permission denied. Run with sudo or as root."

    def unmount_block_device(self, device: str) -> Tuple[bool, str]:
        """
        Unmount a block device.

        Args:
            device: Block device path

        Returns:
            Tuple of (success, error_message)
        """
        if device not in self.mount_points:
            return False, f"Device {device} is not mounted by this manager"

        mount_point = self.mount_points[device]
        try:
            subprocess.run(
                ["umount", mount_point],
                check=True,
                capture_output=True,
            )
            del self.mount_points[device]

            # Cleanup temp directory if we created it
            if mount_point in self.temp_dirs:
                shutil.rmtree(mount_point, ignore_errors=True)
                self.temp_dirs.remove(mount_point)

            return True, ""
        except subprocess.CalledProcessError as e:
            return False, f"Failed to unmount: {e.stderr.decode()}"

    def cleanup(self):
        """Cleanup all mounts and temporary directories."""
        devices = list(self.mount_points.keys())
        for device in devices:
            self.unmount_block_device(device)

        # Cleanup any remaining temp dirs
        for temp_dir in self.temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
        self.temp_dirs.clear()
