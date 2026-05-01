"""
Centralized health monitoring for benchmark sweep pods.

Monitors multiple namespaces concurrently with a single background thread,
detecting failures early through pod status, events, and log analysis.
"""

import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum


class FailureCategory(Enum):
    """Categories of deployment failures."""
    OOM = "oom"
    CRASH = "crash"
    IMAGE_PULL = "image_pull"
    CONFIG_ERROR = "config_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class FailurePhase(Enum):
    """Phase where failure occurred."""
    MODEL_LOAD = "model_load"
    DEPLOYMENT = "deployment"
    BENCHMARK = "benchmark"
    UNKNOWN = "unknown"


@dataclass
class FailureInfo:
    """Detailed failure information."""
    category: str  # FailureCategory value
    phase: str     # FailurePhase value
    trigger: str   # What detected it: "pod_status", "pod_event", "log_pattern", "restart"
    message: str
    pod_name: Optional[str] = None
    timestamp: float = 0.0
    diagnostics: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class NamespaceMonitorState:
    """State tracking for a monitored namespace."""
    namespace: str
    run_id: int
    start_time: float
    last_check_time: float
    error_start_time: Optional[float]
    failure_info: Optional[FailureInfo]


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_calls_per_second: int):
        self.min_interval = 1.0 / max_calls_per_second if max_calls_per_second > 0 else 0
        self.last_call_time = 0
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """Wait if necessary to respect rate limit."""
        if self.min_interval == 0:
            return

        with self.lock:
            now = time.time()
            time_since_last = now - self.last_call_time

            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)

            self.last_call_time = time.time()


class CentralizedHealthMonitor:
    """Single monitor that tracks multiple namespaces concurrently.

    Thread-safe design for parallel sweep execution.
    """

    def __init__(self, check_interval: int = 15, aggressive_timeout: int = 60,
                 api_rate_limit: int = 10):
        """
        Initialize the centralized health monitor.

        Args:
            check_interval: Seconds between health checks
            aggressive_timeout: Seconds of persistent errors before failing
            api_rate_limit: Max Kubernetes API calls per second
        """
        self.check_interval = check_interval
        self.aggressive_timeout = aggressive_timeout

        # Thread-safe tracking of monitored namespaces
        self.monitored_namespaces: Dict[str, NamespaceMonitorState] = {}
        self.lock = threading.Lock()

        # Single background thread for all monitoring
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Rate limiting for k8s API calls
        self.api_rate_limiter = RateLimiter(max_calls_per_second=api_rate_limit)

    def start(self):
        """Start the centralized monitor thread."""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.stop_event.clear()
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name="CentralizedHealthMonitor",
                daemon=True
            )
            self.monitor_thread.start()

    def register_namespace(self, namespace: str, run_id: int) -> None:
        """Register a namespace to be monitored."""
        with self.lock:
            self.monitored_namespaces[namespace] = NamespaceMonitorState(
                namespace=namespace,
                run_id=run_id,
                start_time=time.time(),
                last_check_time=0,
                error_start_time=None,
                failure_info=None
            )
        print(f"  🔍 [Run {run_id:03d}] Started health monitoring for namespace: {namespace}")

    def unregister_namespace(self, namespace: str) -> None:
        """Unregister a namespace (called when run completes)."""
        with self.lock:
            if namespace in self.monitored_namespaces:
                run_id = self.monitored_namespaces[namespace].run_id
                del self.monitored_namespaces[namespace]
                print(f"  🔍 [Run {run_id:03d}] Stopped monitoring namespace: {namespace}")

    def check_namespace_health(self, namespace: str) -> Optional[FailureInfo]:
        """Check if a specific namespace has detected failure.

        Thread-safe check that can be called from worker threads.

        Returns:
            FailureInfo if failure detected, None otherwise
        """
        with self.lock:
            if namespace in self.monitored_namespaces:
                state = self.monitored_namespaces[namespace]
                return state.failure_info
        return None

    def get_monitored_count(self) -> int:
        """Get count of currently monitored namespaces."""
        with self.lock:
            return len(self.monitored_namespaces)

    def stop(self):
        """Stop the monitor thread."""
        self.stop_event.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

    def _monitoring_loop(self):
        """Main monitoring loop - checks all registered namespaces."""
        while not self.stop_event.is_set():
            # Get snapshot of namespaces to check
            with self.lock:
                namespaces_to_check = list(self.monitored_namespaces.items())

            # Check each namespace (outside the lock to avoid blocking)
            for ns_name, ns_state in namespaces_to_check:
                if self.stop_event.is_set():
                    break

                # Rate limit API calls
                self.api_rate_limiter.wait_if_needed()

                try:
                    # Perform health check
                    failure = self._check_namespace(ns_name, ns_state)

                    # Update state
                    with self.lock:
                        if ns_name not in self.monitored_namespaces:
                            continue  # Namespace was unregistered

                        state = self.monitored_namespaces[ns_name]
                        state.last_check_time = time.time()

                        if failure:
                            # Track persistent errors
                            if state.error_start_time is None:
                                state.error_start_time = time.time()
                                print(f"  ⚠️  [Run {state.run_id:03d}] Health issue detected: {failure.message}")

                            # Fail if error persists
                            error_duration = time.time() - state.error_start_time
                            if error_duration > self.aggressive_timeout:
                                state.failure_info = failure
                                print(f"  ❌ [Run {state.run_id:03d}] Health check FAILED after {error_duration:.0f}s")
                                print(f"     Category: {failure.category}, Phase: {failure.phase}")
                                print(f"     Message: {failure.message}")
                        else:
                            # Clear error state if recovered
                            if state.error_start_time is not None:
                                print(f"  ✅ [Run {state.run_id:03d}] Health issue resolved")
                            state.error_start_time = None

                except Exception as e:
                    print(f"  ⚠️  Monitor error for {ns_name}: {e}")

            # Wait before next check cycle
            self.stop_event.wait(self.check_interval)

    def _check_namespace(self, namespace: str, state: NamespaceMonitorState) -> Optional[FailureInfo]:
        """Check health of a single namespace.

        Returns FailureInfo if failure detected, None otherwise.
        """
        # Get pods with model serving label
        pods = self._get_pods_by_label(namespace, "llm-d.ai/role=decode")

        if not pods:
            # No pods yet - might still be creating
            # Only fail if we've been waiting too long
            if time.time() - state.start_time > 120:  # 2 minutes
                return FailureInfo(
                    category=FailureCategory.DEPLOYMENT.value,
                    phase=FailurePhase.DEPLOYMENT.value,
                    trigger="pod_status",
                    message="No model serving pods found after 2 minutes",
                    timestamp=time.time()
                )
            return None

        for pod in pods:
            # A. Pod status check
            failure = self._check_pod_status(pod, state.run_id)
            if failure:
                return failure

            # B. Pod events check
            failure = self._check_pod_events(namespace, pod['name'], state.run_id)
            if failure:
                return failure

            # C. Log pattern check (only if pod is running)
            if pod['phase'] == "Running":
                failure = self._check_container_logs(namespace, pod['name'], state.run_id)
                if failure:
                    return failure

        return None

    def _get_pods_by_label(self, namespace: str, label_selector: str) -> List[Dict]:
        """Get pods by label selector.

        Returns list of pod dicts with name, phase, and status info.
        """
        try:
            cmd = [
                "kubectl", "get", "pods",
                "-n", namespace,
                "-l", label_selector,
                "-o", "json"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return []

            data = json.loads(result.stdout)
            pods = []

            for item in data.get('items', []):
                pod_info = {
                    'name': item['metadata']['name'],
                    'phase': item['status'].get('phase', 'Unknown'),
                    'container_statuses': item['status'].get('containerStatuses', []),
                    'raw': item
                }
                pods.append(pod_info)

            return pods

        except Exception as e:
            print(f"    Warning: Failed to get pods in {namespace}: {e}")
            return []

    def _check_pod_status(self, pod: Dict, run_id: int) -> Optional[FailureInfo]:
        """Check pod phase and container statuses.

        Returns FailureInfo if failure detected.
        """
        pod_name = pod['name']
        phase = pod['phase']

        # Check for bad pod phases
        if phase in ['Failed', 'Unknown']:
            return FailureInfo(
                category=FailureCategory.CRASH.value,
                phase=FailurePhase.DEPLOYMENT.value,
                trigger="pod_status",
                message=f"Pod in {phase} state",
                pod_name=pod_name,
                timestamp=time.time()
            )

        # Check container statuses
        for container_status in pod['container_statuses']:
            # Check waiting state
            waiting = container_status.get('state', {}).get('waiting')
            if waiting:
                reason = waiting.get('reason', '')

                # CrashLoopBackOff
                if reason == 'CrashLoopBackOff':
                    return FailureInfo(
                        category=FailureCategory.CRASH.value,
                        phase=FailurePhase.DEPLOYMENT.value,
                        trigger="pod_status",
                        message=f"Container in CrashLoopBackOff: {waiting.get('message', '')}",
                        pod_name=pod_name,
                        timestamp=time.time()
                    )

                # ImagePullBackOff
                if reason in ['ImagePullBackOff', 'ErrImagePull']:
                    return FailureInfo(
                        category=FailureCategory.IMAGE_PULL.value,
                        phase=FailurePhase.DEPLOYMENT.value,
                        trigger="pod_status",
                        message=f"Image pull failed: {waiting.get('message', '')}",
                        pod_name=pod_name,
                        timestamp=time.time()
                    )

                # Other container errors
                if reason in ['CreateContainerError', 'CreateContainerConfigError']:
                    return FailureInfo(
                        category=FailureCategory.CONFIG_ERROR.value,
                        phase=FailurePhase.DEPLOYMENT.value,
                        trigger="pod_status",
                        message=f"Container creation failed: {waiting.get('message', '')}",
                        pod_name=pod_name,
                        timestamp=time.time()
                    )

            # Check terminated state
            terminated = container_status.get('state', {}).get('terminated')
            if terminated:
                reason = terminated.get('reason', '')
                if reason == 'OOMKilled':
                    return FailureInfo(
                        category=FailureCategory.OOM.value,
                        phase=FailurePhase.DEPLOYMENT.value,
                        trigger="pod_status",
                        message=f"Container OOMKilled: {terminated.get('message', '')}",
                        pod_name=pod_name,
                        timestamp=time.time()
                    )

            # Check restart count (NO restarts allowed)
            restart_count = container_status.get('restartCount', 0)
            if restart_count > 0:
                return FailureInfo(
                    category=FailureCategory.CRASH.value,
                    phase=FailurePhase.DEPLOYMENT.value,
                    trigger="restart",
                    message=f"Container restarted {restart_count} time(s) - not allowed in benchmark runs",
                    pod_name=pod_name,
                    timestamp=time.time()
                )

        return None

    def _check_pod_events(self, namespace: str, pod_name: str, run_id: int) -> Optional[FailureInfo]:
        """Check for error/warning events in last 60 seconds.

        Returns FailureInfo if critical events detected.
        """
        try:
            cmd = [
                "kubectl", "get", "events",
                "-n", namespace,
                "--field-selector", f"involvedObject.name={pod_name}",
                "-o", "json"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return None

            data = json.loads(result.stdout)
            now = time.time()

            for event in data.get('items', []):
                # Check if event is recent (last 60 seconds)
                last_timestamp = event.get('lastTimestamp') or event.get('eventTime')
                if not last_timestamp:
                    continue

                # Parse timestamp (simplified - assumes recent events)
                # Full parsing would use dateutil, but we'll check type == Warning

                if event.get('type') == 'Warning':
                    reason = event.get('reason', '')
                    message = event.get('message', '')

                    # Critical warnings
                    if reason in ['FailedScheduling', 'FailedMount', 'BackOff', 'Unhealthy']:
                        return FailureInfo(
                            category=FailureCategory.CONFIG_ERROR.value,
                            phase=FailurePhase.DEPLOYMENT.value,
                            trigger="pod_event",
                            message=f"Pod event: {reason} - {message}",
                            pod_name=pod_name,
                            timestamp=time.time()
                        )

            return None

        except Exception as e:
            # Don't fail monitoring due to event check errors
            return None

    def _check_container_logs(self, namespace: str, pod_name: str, run_id: int) -> Optional[FailureInfo]:
        """Scan recent logs for error patterns.

        Returns FailureInfo if critical patterns detected.
        """
        try:
            # Get last 100 lines of logs
            cmd = [
                "kubectl", "logs",
                "-n", namespace,
                pod_name,
                "--tail=100"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                # Log retrieval failed - might be too early, not an error
                return None

            logs = result.stdout

            # Check for critical patterns (from failure_patterns.py will be imported)
            from failure_patterns import ERROR_PATTERNS

            for pattern_info in ERROR_PATTERNS:
                pattern = pattern_info['pattern']
                category = pattern_info['category']
                phase = pattern_info['phase']

                match = re.search(pattern, logs, re.IGNORECASE | re.MULTILINE)
                if match:
                    # Extract context around the match
                    context = self._extract_log_context(logs, match.start(), context_lines=5)

                    return FailureInfo(
                        category=category,
                        phase=phase,
                        trigger="log_pattern",
                        message=f"Error pattern detected: {pattern_info['description']}",
                        pod_name=pod_name,
                        timestamp=time.time(),
                        diagnostics={'log_context': context}
                    )

            return None

        except Exception as e:
            # Don't fail monitoring due to log check errors
            return None

    def _extract_log_context(self, logs: str, match_position: int, context_lines: int = 5) -> str:
        """Extract context around a log match."""
        lines = logs[:match_position].split('\n')
        start_line = max(0, len(lines) - context_lines)

        # Get lines around match
        all_lines = logs.split('\n')
        match_line_num = len(lines) - 1
        end_line = min(len(all_lines), match_line_num + context_lines + 1)

        context_lines_list = all_lines[start_line:end_line]
        return '\n'.join(context_lines_list[-10:])  # Limit to 10 lines max

    def collect_diagnostics(self, namespace: str) -> Dict:
        """Collect detailed diagnostics for a namespace.

        Called when a failure is detected to gather debugging info.
        """
        diagnostics = {
            'namespace': namespace,
            'timestamp': time.time(),
            'pods': [],
            'events': None,
            'describe': None
        }

        try:
            # Get pod details
            pods = self._get_pods_by_label(namespace, "llm-d.ai/role=decode")
            for pod in pods:
                pod_diag = {
                    'name': pod['name'],
                    'phase': pod['phase'],
                    'container_statuses': pod['container_statuses']
                }

                # Get pod describe output
                try:
                    cmd = ["kubectl", "describe", "pod", pod['name'], "-n", namespace]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    pod_diag['describe'] = result.stdout
                except Exception:
                    pass

                # Get recent logs
                try:
                    cmd = ["kubectl", "logs", pod['name'], "-n", namespace, "--tail=-1"]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    pod_diag['logs'] = result.stdout
                except Exception:
                    pass

                diagnostics['pods'].append(pod_diag)

            # Get events
            try:
                cmd = ["kubectl", "get", "events", "-n", namespace, "-o", "json"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    diagnostics['events'] = json.loads(result.stdout)
            except Exception:
                pass

        except Exception as e:
            diagnostics['error'] = str(e)

        return diagnostics


class DeploymentHealthCheckFailure(Exception):
    """Exception raised when health check detects a deployment failure."""

    def __init__(self, message: str, failure_info: FailureInfo):
        super().__init__(message)
        self.failure_info = failure_info
