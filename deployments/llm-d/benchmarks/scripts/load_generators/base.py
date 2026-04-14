"""
Base class for load generation tools.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List


class LoadGeneratorBase(ABC):
    """Base class for all load generation tools."""

    def __init__(self, orchestrator):
        """
        Initialize load generator.

        Args:
            orchestrator: Reference to SweepOrchestrator instance for accessing utilities
        """
        self.orchestrator = orchestrator

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
