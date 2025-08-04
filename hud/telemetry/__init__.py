"""
HUD Telemetry module.

Provides context managers and utilities for capturing MCP telemetry data.
"""

from __future__ import annotations

# Main trace functions
from hud.telemetry._trace import init_telemetry, trace, trace_open
from hud.telemetry.context import flush_buffer, get_current_task_run_id
from hud.telemetry.exporter import flush
from hud.telemetry.job import job, get_current_job_id, get_current_job_name

__all__ = [
    # Management
    "flush",
    "flush_buffer",
    # Context management
    "get_current_task_run_id",
    "get_current_job_id",
    "get_current_job_name",
    # Management
    "init_telemetry",
    # Trace functions
    "trace",
    "trace_open",
    # Job context
    "job",
]
