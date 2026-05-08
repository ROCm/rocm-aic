"""Process manager for LMCache standalone."""
import os
import sys
import subprocess
import signal
import time
from typing import Optional, Dict, Any
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None


class ProcessManager:
    """Manages LMCache standalone process lifecycle."""

    PROCESS_NAME = "lmcache-standalone"

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.config_path: Optional[str] = None

    def process_exists(self) -> bool:
        """
        Check if LMCache process is running.

        Returns:
            True if process exists
        """
        return self.process is not None and self.process.poll() is None

    def get_process(self):
        """
        Get process object.

        Returns:
            Process object or None
        """
        if self.process_exists():
            return self.process
        return None

    def start_process(
        self,
        config_path: str,
        storage_path: str,
        device: str = "cpu",
        model_name: str = "lmcache_model",
        worker_id: int = 0,
        world_size: int = 1,
        kv_dtype: str = "float16",
        kv_shape: str = "2,2,256,4,16",
        use_mla: bool = False,
        internal_api_server_enabled: bool = True,
        internal_api_server_port: int = 6999,
        environment: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Start LMCache standalone process.

        Args:
            config_path: Path to config file
            storage_path: Storage path
            device: Device to run on (cpu, cuda, xpu)
            model_name: Model name for cache identification
            worker_id: Worker ID
            world_size: Total workers
            kv_dtype: KV cache data type
            kv_shape: KV cache shape
            use_mla: Enable Multi-Level Attention
            internal_api_server_enabled: Enable internal API server
            internal_api_server_port: Internal API server port
            environment: Additional environment variables

        Returns:
            Process ID
        """
        # Check if process already exists
        if self.process_exists():
            print(f"Process {self.PROCESS_NAME} is already running")
            return self.process.pid

        # Prepare environment
        env = os.environ.copy()
        env["LMCACHE_CONFIG_FILE"] = config_path
        env["LMCACHE_USE_EXPERIMENTAL"] = "True"
        if internal_api_server_enabled:
            env["LMCACHE_INTERNAL_API_SERVER_ENABLED"] = "True"
            env["LMCACHE_INTERNAL_API_SERVER_PORT_START"] = str(
                internal_api_server_port
            )
        if environment:
            env.update(environment)

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "lmcache.v1.standalone",
            "--config",
            config_path,
            "--model_name",
            model_name,
            "--worker_id",
            str(worker_id),
            "--world_size",
            str(world_size),
            "--device",
            device,
            "--kv_dtype",
            kv_dtype,
            "--kv_shape",
            kv_shape,
        ]

        if use_mla:
            cmd.append("--use_mla")

        # Start process
        try:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,  # Create new process group
            )
            self.config_path = config_path
            print(
                f"Started LMCache process {self.PROCESS_NAME} "
                f"(PID: {self.process.pid})"
            )
            return self.process.pid
        except Exception as e:
            raise RuntimeError(f"Failed to start process: {e}") from e

    def stop_process(self, timeout: int = 10) -> bool:
        """
        Stop process.

        Args:
            timeout: Stop timeout in seconds

        Returns:
            True if stopped successfully
        """
        if not self.process_exists():
            print(f"Process {self.PROCESS_NAME} not running")
            return False

        try:
            # Try graceful shutdown with SIGTERM
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

            # Wait for process to terminate
            try:
                self.process.wait(timeout=timeout)
                print(f"Stopped process {self.PROCESS_NAME}")
                self.process = None
                return True
            except subprocess.TimeoutExpired:
                # Force kill if timeout
                print(f"Process did not stop gracefully, forcing kill...")
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait()
                self.process = None
                print(f"Force stopped process {self.PROCESS_NAME}")
                return True
        except ProcessLookupError:
            # Process already terminated
            self.process = None
            return True
        except Exception as e:
            print(f"Error stopping process: {e}")
            return False

    def get_process_status(self) -> Optional[Dict[str, Any]]:
        """
        Get process status.

        Returns:
            Status dictionary or None
        """
        if not self.process_exists():
            return None

        return {
            "pid": self.process.pid,
            "name": self.PROCESS_NAME,
            "status": "running" if self.process.poll() is None else "stopped",
            "config": self.config_path,
        }

    def get_process_logs(
        self, tail: int = 100, follow: bool = False
    ) -> str:
        """
        Get process logs.

        Args:
            tail: Number of lines to tail
            follow: Follow log output

        Returns:
            Log output
        """
        if not self.process_exists():
            return ""

        # Note: stdout/stderr are captured, so we can't easily tail them
        # This would require more complex logging setup
        return "Logs are captured in process stdout/stderr"

    def wait_for_process(
        self, timeout: int = 30, check_interval: float = 1.0, api_port: int = 6999
    ) -> bool:
        """
        Wait for process to be ready.

        Args:
            timeout: Timeout in seconds
            check_interval: Check interval in seconds
            api_port: API server port to check

        Returns:
            True if process is ready
        """
        import requests

        start_time = time.time()

        while time.time() - start_time < timeout:
            if not self.process_exists():
                time.sleep(check_interval)
                continue

            # Check if API server is responding
            if requests is not None:
                try:
                    # Try /metrics endpoint which is available on all components
                    response = requests.get(
                        f"http://localhost:{api_port}/metrics", timeout=1
                    )
                    if response.status_code == 200:
                        return True
                except requests.RequestException:
                    pass
            else:
                # Fallback: just check if process exists
                return True

            time.sleep(check_interval)

        return False


# Import sys for sys.executable
import sys
