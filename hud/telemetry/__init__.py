"""HUD Telemetry - User-facing APIs for tracing and job management.

This module provides the main telemetry APIs that users interact with:
- trace: Context manager for tracing code execution
- job: Context manager and utilities for job management  
- get_trace: Retrieve collected traces for replay/analysis
"""

from .trace import trace
from .job import job, create_job, Job
from .replay import get_trace, clear_trace

__all__ = [
    "trace",
    "job", 
    "create_job",
    "Job",
    "get_trace",
    "clear_trace",
]
