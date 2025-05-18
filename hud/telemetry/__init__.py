"""
HUD telemetry module for capturing and reporting telemetry data from MCP calls.

This module provides functionality to trace MCP calls and export telemetry data
to the HUD platform for analysis.
"""

from hud.telemetry.context import get_current_task_run_id, set_current_task_run_id
from hud.telemetry.trace import init_telemetry, trace, async_trace, start_trace

__all__ = [
    "init_telemetry",
    "trace",
    "async_trace",
    "start_trace",
    "get_current_task_run_id",
    "set_current_task_run_id"
] 