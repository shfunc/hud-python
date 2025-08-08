"""HUD OpenTelemetry integration.

This package provides the internal OpenTelemetry implementation for HUD telemetry.
Users should interact with the telemetry APIs through hud.telemetry instead.

Internal Components:
- config: OpenTelemetry configuration and setup
- context: Trace context management and utilities
- processors: Span enrichment with HUD context
- exporters: Sending spans to HUD backend
- collector: In-memory span collection for replay
- instrumentation: Auto-instrumentation for agents and MCP
"""

from .config import configure_telemetry, shutdown_telemetry
from .context import trace, span_context, get_current_task_run_id, is_root_trace
from .collector import enable_trace_collection

__all__ = [
    # Configuration
    "configure_telemetry",
    "shutdown_telemetry",
    # Context management
    "trace",
    "span_context", 
    "get_current_task_run_id",
    "is_root_trace",
    # Collection
    "enable_trace_collection",
]