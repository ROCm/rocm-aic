#!/usr/bin/env python3
"""
Shared state management for benchmark sweeps.

Provides data structures and utilities for reading/writing sweep state.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class RunStatus(Enum):
    """Status of a configuration run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RunState:
    """State information for a single configuration run."""
    run_id: int
    namespace: str
    parameters: Dict[str, Any]
    gpu_claim: int
    status: RunStatus
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None
    benchmark_results: Optional[Dict[str, Any]] = None
    failure_info: Optional[Dict[str, Any]] = None  # Detailed failure information

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['status'] = self.status.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunState':
        """Create from dictionary loaded from JSON."""
        data['status'] = RunStatus(data['status'])
        return cls(**data)


def read_state_file(state_file: Path) -> Dict[str, List[RunState]]:
    """
    Read state.json and parse into RunState objects.

    Args:
        state_file: Path to state.json file

    Returns:
        Dictionary with keys 'pending', 'running', 'completed', each containing
        a list of RunState objects

    Raises:
        FileNotFoundError: If state file doesn't exist
        ValueError: If state file is invalid
    """
    if not state_file.exists():
        raise FileNotFoundError(f"State file not found: {state_file}")

    with open(state_file) as f:
        state_data = json.load(f)

    return {
        'pending': [RunState.from_dict(r) for r in state_data.get('pending', [])],
        'running': [RunState.from_dict(r) for r in state_data.get('running', [])],
        'completed': [RunState.from_dict(r) for r in state_data.get('completed', [])]
    }


def write_state_file(state_file: Path, pending: List[RunState],
                     running: List[RunState], completed: List[RunState]):
    """
    Write state to state.json.

    Args:
        state_file: Path to state.json file
        pending: List of pending RunState objects
        running: List of running RunState objects
        completed: List of completed RunState objects
    """
    state = {
        'pending': [rs.to_dict() for rs in pending],
        'running': [rs.to_dict() for rs in running],
        'completed': [rs.to_dict() for rs in completed]
    }

    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def get_sweep_results_dir(sweep_dir_or_name: str) -> Path:
    """
    Resolve sweep directory path from various input formats.

    Supports:
    1. Full path to sweep directory
    2. Sweep directory name (looks in results/sweeps/)

    Args:
        sweep_dir_or_name: Sweep directory path or name

    Returns:
        Resolved Path to sweep directory

    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    import os

    sweep_path = Path(sweep_dir_or_name)

    # Case 1: Full path provided
    if sweep_path.exists() and sweep_path.is_dir():
        return sweep_path

    # Case 2: Directory name - look in configured results directory
    # Get base results directory from environment with default fallback
    base_results_dir = os.environ.get('SWEEP_RESULTS_DIR', 'results/sweeps')

    # Get script directory to locate results (if using relative default)
    script_dir = Path(__file__).parent.parent

    # Resolve base directory (handle both absolute and relative paths)
    if Path(base_results_dir).is_absolute():
        results_base = Path(base_results_dir)
    else:
        results_base = script_dir / base_results_dir

    results_dir = results_base / sweep_dir_or_name

    if results_dir.exists() and results_dir.is_dir():
        return results_dir

    # Not found
    raise FileNotFoundError(
        f"Sweep directory not found: {sweep_dir_or_name}\n"
        f"Tried:\n"
        f"  1. Direct path: {sweep_path.absolute()}\n"
        f"  2. In {base_results_dir}: {results_dir}"
    )
