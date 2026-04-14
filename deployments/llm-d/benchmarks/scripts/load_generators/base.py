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
