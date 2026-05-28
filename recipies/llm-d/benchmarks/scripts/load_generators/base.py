"""
Base class for load generation tools.
"""

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional


class LoadGeneratorBase(ABC):
    """Base class for all load generation tools."""

    # Exit code constants
    EXIT_CODE_TIMEOUT = 77

    # Failure reason constants
    FAILURE_TIMEOUT = "timeout"
    FAILURE_POD_ERROR = "pod_error"
    FAILURE_PARSING = "parsing_failed"

    def __init__(self, orchestrator):
        """
        Initialize load generator.

        Args:
            orchestrator: Reference to SweepOrchestrator instance for accessing utilities
        """
        self.orchestrator = orchestrator

    def execute_benchmark(
        self,
        cmd: List[str],
        run_dir: Path,
        run_label: str = "run1",
        image: Optional[str] = None,
        tool_name: str = "benchmark"
    ) -> Dict[str, Any]:
        """
        Execute benchmark subprocess with standardized error handling.

        This helper method provides consistent subprocess execution, output capture,
        and error classification for all load generators.

        Args:
            cmd: Command to execute as list of strings
            run_dir: Directory for saving output files
            run_label: Label for this run (used in output filenames)
            image: Container image name (for logging)
            tool_name: Human-readable tool name (for logging)

        Returns:
            Dictionary with:
                - exit_code: Process return code
                - failure_reason: Classified failure reason (None if success)
                - failure_details: Human-readable failure description (None if success)
                - stdout: Captured stdout
                - stderr: Captured stderr
                - runner_output_file: Path to saved output file
        """
        if image:
            print(f"  Running {tool_name} in pod (image: {image})...")
        else:
            print(f"  Running {tool_name}...")

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Save runner output
        output_file = run_dir / f"benchmark_runner_output_{run_label}.txt"
        with open(output_file, "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)

        # Determine failure reason based on exit code
        failure_reason = None
        failure_details = None

        if result.returncode == self.EXIT_CODE_TIMEOUT:
            failure_reason = self.FAILURE_TIMEOUT
            failure_details = "Benchmark pod timed out waiting for completion"
        elif result.returncode != 0:
            failure_reason = self.FAILURE_POD_ERROR
            failure_details = f"Benchmark pod exited with code {result.returncode}"

        return {
            "exit_code": result.returncode,
            "failure_reason": failure_reason,
            "failure_details": failure_details,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "runner_output_file": str(output_file),
        }

    @abstractmethod
    def run(self, config: Dict[str, Any], run_dir: Path,
            params: Dict[str, Any], namespace: str) -> Dict[str, Any]:
        """
        Run the load generation tool.

        Args:
            config: Load generation configuration from YAML
            run_dir: Directory for saving results
            params: Deployment parameters (model, vllm_args, etc.)
            namespace: Kubernetes namespace

        Returns:
            Dictionary with results including:
                - exit_code: Return code
                - tool: Tool name
                - Additional tool-specific results
        """
        pass

    @abstractmethod
    def build_args(self, benchmark_args: Dict[str, Any], model: str,
                   service_info: Dict[str, str], **kwargs) -> List[str]:
        """
        Build command-line arguments for the tool.

        Args:
            benchmark_args: Additional benchmark configuration
            model: Model name
            service_info: Service connection information (url, base_url, etc.)
            **kwargs: Additional tool-specific parameters

        Returns:
            List of command-line arguments
        """
        pass

    @abstractmethod
    def parse_metrics(self, output_file: Path) -> Dict[str, Any]:
        """
        Parse metrics from load generator output file.

        This method extracts performance metrics from the tool's output file
        and returns them in a structured format. Implementations must handle
        all exceptions gracefully and never raise errors.

        Args:
            output_file: Path to the output file to parse

        Returns:
            Dictionary containing:
                - metrics: Dict of parsed metrics (structure varies by tool)
                - parsing_status: "success", "partial", or "failed"
                - parsing_errors: Optional list of error messages (if parsing_status != "success")

        Implementation guidelines:
            - Return empty metrics dict with parsing_status="failed" if file doesn't exist
            - Catch all exceptions and return parsing_status="failed" with error details
            - Never raise exceptions - ensure graceful degradation
            - Return parsing_status="partial" if some but not all metrics were extracted
            - Include parsing_errors list for debugging when parsing fails/is partial
        """
        pass
