"""Dataset execution module."""

from __future__ import annotations

from .parallel import calculate_optimal_workers, run_dataset_parallel, run_dataset_parallel_manual
from .runner import run_dataset

__all__ = [
    "calculate_optimal_workers",
    "run_dataset",
    "run_dataset_parallel",
    "run_dataset_parallel_manual",
]
