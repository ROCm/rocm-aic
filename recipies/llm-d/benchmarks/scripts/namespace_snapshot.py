#!/usr/bin/env python3
"""
Namespace snapshot module for capturing comprehensive Kubernetes diagnostics.

Provides an extensible step-based architecture for collecting pod logs,
pod descriptions, events, and other diagnostic information from namespaces.
"""

import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class SnapshotResult:
    """Result from executing a snapshot collection step."""
    step_name: str
    status: str  # "success", "partial", "failed"
    files_created: List[Path] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'step_name': self.step_name,
            'status': self.status,
            'files_created': [str(f) for f in self.files_created],
            'error_message': self.error_message,
            'metadata': self.metadata
        }


class SnapshotStep(ABC):
    """Abstract base class for snapshot collection steps."""

    @property
    @abstractmethod
    def step_name(self) -> str:
        """Name of this snapshot step."""
        pass

    @abstractmethod
    def collect(self, namespace: str, output_dir: Path,
                label_selector: str) -> SnapshotResult:
        """
        Execute this snapshot collection step.

        Args:
            namespace: Kubernetes namespace to collect from
            output_dir: Directory where files should be saved
            label_selector: Label selector for filtering pods

        Returns:
            SnapshotResult with status, files created, and any errors
        """
        pass


class CollectPodLogsStep(SnapshotStep):
    """Collect pod logs including previous logs if containers restarted."""

    @property
    def step_name(self) -> str:
        return "collect_pod_logs"

    def collect(self, namespace: str, output_dir: Path,
                label_selector: str) -> SnapshotResult:
        """
        Collect current and previous pod logs for all containers.

        Discovers all pods matching the label selector and collects:
        - Current logs for each pod
        - Previous logs if containers have restarted
        - Per-container logs for multi-container pods
        """
        files_created = []
        errors = []
        pods_processed = 0
        restarts_detected = 0

        try:
            # Get pods with JSON format to check restart count
            result = subprocess.run([
                "kubectl", "get", "pods",
                "-n", namespace,
                "-l", label_selector,
                "-o", "json"
            ], capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                # Fallback: Try simple pod name list
                result = subprocess.run([
                    "kubectl", "get", "pods",
                    "-n", namespace,
                    "-l", label_selector,
                    "-o", "jsonpath={.items[*].metadata.name}"
                ], capture_output=True, text=True, timeout=30)

                if result.returncode != 0:
                    return SnapshotResult(
                        step_name=self.step_name,
                        status="failed",
                        error_message=f"Failed to get pods: {result.stderr}",
                        metadata={}
                    )

                # Simple fallback: just get current logs
                pod_names = result.stdout.strip().split()
                for pod_name in pod_names:
                    try:
                        log_file = output_dir / f"{pod_name}.log"
                        with open(log_file, 'w') as f:
                            subprocess.run([
                                "kubectl", "logs", pod_name, "-n", namespace
                            ], stdout=f, stderr=subprocess.STDOUT, timeout=60)
                        files_created.append(log_file)
                        pods_processed += 1
                    except Exception as e:
                        errors.append(f"Pod {pod_name}: {str(e)}")

            else:
                # Full processing with JSON data
                try:
                    pods_data = json.loads(result.stdout)
                except json.JSONDecodeError as e:
                    return SnapshotResult(
                        step_name=self.step_name,
                        status="failed",
                        error_message=f"Failed to parse pod JSON: {str(e)}",
                        metadata={}
                    )

                for pod in pods_data.get('items', []):
                    pod_name = pod['metadata']['name']

                    try:
                        # Check for container restarts
                        has_restarts = False
                        container_statuses = pod.get('status', {}).get('containerStatuses', [])

                        for container_status in container_statuses:
                            restart_count = container_status.get('restartCount', 0)
                            if restart_count > 0:
                                has_restarts = True
                                restarts_detected += 1
                                break

                        # Collect current logs
                        log_file = output_dir / f"{pod_name}.log"
                        with open(log_file, 'w') as f:
                            subprocess.run([
                                "kubectl", "logs", pod_name, "-n", namespace
                            ], stdout=f, stderr=subprocess.STDOUT, timeout=60)
                        files_created.append(log_file)

                        # Collect previous logs if restarted
                        if has_restarts:
                            prev_log_file = output_dir / f"{pod_name}-previous.log"
                            prev_result = subprocess.run([
                                "kubectl", "logs", pod_name, "-n", namespace,
                                "--previous", "--tail=-1"
                            ], capture_output=True, text=True, timeout=60)

                            if prev_result.returncode == 0:
                                with open(prev_log_file, 'w') as f:
                                    f.write(prev_result.stdout)
                                files_created.append(prev_log_file)
                            else:
                                # Save error message
                                with open(prev_log_file, 'w') as f:
                                    f.write(f"Error retrieving previous logs:\n{prev_result.stderr}\n")
                                files_created.append(prev_log_file)

                        # Collect per-container logs if multi-container
                        if len(container_statuses) > 1:
                            for container_status in container_statuses:
                                container_name = container_status.get('name')
                                if container_name:
                                    container_log_file = output_dir / f"{pod_name}-{container_name}.log"
                                    with open(container_log_file, 'w') as f:
                                        subprocess.run([
                                            "kubectl", "logs", pod_name, "-n", namespace,
                                            "-c", container_name
                                        ], stdout=f, stderr=subprocess.STDOUT, timeout=60)
                                    files_created.append(container_log_file)

                        pods_processed += 1

                    except Exception as e:
                        errors.append(f"Pod {pod_name}: {str(e)}")

            # Determine status
            if errors and not files_created:
                status = "failed"
            elif errors:
                status = "partial"
            else:
                status = "success"

            return SnapshotResult(
                step_name=self.step_name,
                status=status,
                files_created=files_created,
                error_message="; ".join(errors) if errors else None,
                metadata={
                    'pods_processed': pods_processed,
                    'restarts_detected': restarts_detected,
                    'pods_failed': len(errors)
                }
            )

        except Exception as e:
            return SnapshotResult(
                step_name=self.step_name,
                status="failed",
                error_message=f"Unexpected error: {str(e)}",
                metadata={'pods_processed': pods_processed}
            )


class CollectPodDescribeStep(SnapshotStep):
    """Collect kubectl describe output for all pods."""

    @property
    def step_name(self) -> str:
        return "collect_pod_describe"

    def collect(self, namespace: str, output_dir: Path,
                label_selector: str) -> SnapshotResult:
        """
        Collect describe output for each pod.

        Runs kubectl describe pod for each pod and saves the output.
        """
        files_created = []
        errors = []
        pods_processed = 0

        try:
            # Get pod names
            result = subprocess.run([
                "kubectl", "get", "pods",
                "-n", namespace,
                "-l", label_selector,
                "-o", "jsonpath={.items[*].metadata.name}"
            ], capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                return SnapshotResult(
                    step_name=self.step_name,
                    status="failed",
                    error_message=f"Failed to get pods: {result.stderr}",
                    metadata={}
                )

            pod_names = result.stdout.strip().split()

            if not pod_names or (len(pod_names) == 1 and not pod_names[0]):
                return SnapshotResult(
                    step_name=self.step_name,
                    status="success",
                    error_message="No pods found",
                    metadata={'pods_processed': 0}
                )

            # Describe each pod
            for pod_name in pod_names:
                try:
                    describe_result = subprocess.run([
                        "kubectl", "describe", "pod", pod_name, "-n", namespace
                    ], capture_output=True, text=True, timeout=30)

                    if describe_result.returncode != 0:
                        errors.append(f"Pod {pod_name}: {describe_result.stderr}")
                        continue

                    describe_file = output_dir / f"{pod_name}-describe.txt"
                    with open(describe_file, 'w') as f:
                        f.write(describe_result.stdout)
                    files_created.append(describe_file)
                    pods_processed += 1

                except Exception as e:
                    errors.append(f"Pod {pod_name}: {str(e)}")

            # Determine status
            if errors and not files_created:
                status = "failed"
            elif errors:
                status = "partial"
            else:
                status = "success"

            return SnapshotResult(
                step_name=self.step_name,
                status=status,
                files_created=files_created,
                error_message="; ".join(errors) if errors else None,
                metadata={
                    'pods_processed': pods_processed,
                    'pods_failed': len(errors)
                }
            )

        except Exception as e:
            return SnapshotResult(
                step_name=self.step_name,
                status="failed",
                error_message=f"Unexpected error: {str(e)}",
                metadata={'pods_processed': pods_processed}
            )


class CollectNamespaceEventsStep(SnapshotStep):
    """Collect Kubernetes events for the namespace."""

    @property
    def step_name(self) -> str:
        return "collect_namespace_events"

    def collect(self, namespace: str, output_dir: Path,
                label_selector: str) -> SnapshotResult:
        """
        Collect namespace events.

        Runs kubectl get events and saves the YAML output.
        """
        try:
            result = subprocess.run([
                "kubectl", "get", "events",
                "-n", namespace,
                "-o", "yaml"
            ], capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                return SnapshotResult(
                    step_name=self.step_name,
                    status="failed",
                    error_message=f"Failed to get events: {result.stderr}",
                    metadata={}
                )

            events_file = output_dir / "namespace-events.yaml"
            with open(events_file, 'w') as f:
                f.write(result.stdout)

            return SnapshotResult(
                step_name=self.step_name,
                status="success",
                files_created=[events_file],
                metadata={'size_bytes': len(result.stdout)}
            )

        except Exception as e:
            return SnapshotResult(
                step_name=self.step_name,
                status="failed",
                error_message=f"Unexpected error: {str(e)}",
                metadata={}
            )


class NamespaceSnapshot:
    """Orchestrator for capturing comprehensive namespace diagnostics."""

    def __init__(self, steps: Optional[List[SnapshotStep]] = None):
        """
        Initialize snapshot orchestrator.

        Args:
            steps: List of snapshot steps to execute (None = use defaults)
        """
        self.steps = steps or self._get_default_steps()

    def _get_default_steps(self) -> List[SnapshotStep]:
        """Get default snapshot steps."""
        return [
            CollectPodLogsStep(),
            CollectPodDescribeStep(),
            CollectNamespaceEventsStep()
        ]

    def capture(self, namespace: str, output_dir: Path,
                label_selector: str = "llm-d.ai/role=decode") -> Dict[str, SnapshotResult]:
        """
        Execute all snapshot steps and save metadata.

        Args:
            namespace: Kubernetes namespace to snapshot
            output_dir: Directory where snapshot files should be saved
            label_selector: Label selector for filtering pods

        Returns:
            Dictionary mapping step names to their SnapshotResults
        """
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Execute each step
        results = {}
        for step in self.steps:
            result = step.collect(namespace, output_dir, label_selector)
            results[step.step_name] = result

        # Save metadata
        self._save_metadata(namespace, output_dir, label_selector, results)

        return results

    def _save_metadata(self, namespace: str, output_dir: Path,
                       label_selector: str, results: Dict[str, SnapshotResult]):
        """Save snapshot metadata to JSON file."""
        metadata = {
            'namespace': namespace,
            'timestamp': time.time(),
            'label_selector': label_selector,
            'steps': {
                step_name: result.to_dict()
                for step_name, result in results.items()
            }
        }

        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
