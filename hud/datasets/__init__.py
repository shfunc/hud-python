"""HUD datasets module.

Provides data models, utilities, and execution functions for working with HUD datasets.
"""

# Data models
from .task import Task

# Utilities
from .utils import fetch_system_prompt_from_dataset, save_tasks

# Execution functions
from .execution import (
    calculate_optimal_workers,
    run_dataset,
    run_dataset_parallel,
    run_dataset_parallel_auto,
)

__all__ = [
    # Core data model
    "Task",
    # Utilities
    "fetch_system_prompt_from_dataset",
    "save_tasks",
    # Execution
    "run_dataset",
    "run_dataset_parallel",
    "run_dataset_parallel_auto",
    "calculate_optimal_workers",
]
