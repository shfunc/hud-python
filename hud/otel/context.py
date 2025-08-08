from __future__ import annotations

"""Context helper functions for HUD telemetry with OpenTelemetry.

This module provides utilities for managing span contexts within a trace.
"""

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from opentelemetry import baggage, context, trace

if TYPE_CHECKING:
    from collections.abc import Generator

# Import context functions from trace module
from hud.otel.trace import (
    current_task_run_id as _current_task_run_id_var,
    is_root_trace as _is_root_trace_var,
)

logger = logging.getLogger(__name__)

# Re-export for convenience
TASK_RUN_ID_KEY = "hud.task_run_id"
IS_ROOT_TRACE_KEY = "hud.is_root_trace"


def get_current_task_run_id() -> str | None:
    """Get current task_run_id from either contextvars or OTel baggage."""
    # First try OTel baggage
    task_run_id = baggage.get_baggage(TASK_RUN_ID_KEY)
    if task_run_id:
        return task_run_id
    
    # Fallback to contextvars
    return _current_task_run_id_var.get()


def is_root_trace() -> bool:
    """Check if current context is a root trace."""
    # First try OTel baggage
    is_root = baggage.get_baggage(IS_ROOT_TRACE_KEY)
    if is_root is not None:
        return is_root.lower() == "true"
    
    # Fallback to contextvars
    return _is_root_trace_var.get()


@contextmanager
def span_context(
    name: str,
    attributes: dict[str, Any] | None = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
) -> Generator[trace.Span, None, None]:
    """Create a child span within the current trace context.
    
    This is a simple wrapper around OpenTelemetry's span creation that
    ensures the span inherits the current HUD context (task_run_id, etc).
    
    Args:
        name: Name for the span
        attributes: Additional attributes to add to the span
        kind: OpenTelemetry span kind
        
    Example:
        with span_context("process_data", {"items": 100}) as span:
            # Process data...
            span.set_attribute("processed", True)
    """
    tracer = trace.get_tracer("hud-sdk")
    
    # Current task_run_id will be added by HudEnrichmentProcessor
    with tracer.start_as_current_span(
        name,
        attributes=attributes,
        kind=kind,
    ) as span:
        yield span