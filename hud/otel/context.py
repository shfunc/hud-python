"""OpenTelemetry context utilities for HUD telemetry.

This module provides internal utilities for managing OpenTelemetry contexts.
User-facing APIs are in hud.telemetry.
"""

from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional

from opentelemetry import baggage, context
from opentelemetry import trace as otel_trace
from opentelemetry.trace import Status, StatusCode

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import TracebackType

from hud.server import make_request
from hud.settings import settings
from hud.utils.async_utils import fire_and_forget

logger = logging.getLogger(__name__)

# Context variables for task tracking
current_task_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_task_run_id", default=None
)
is_root_trace_var: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "is_root_trace", default=False
)

# Keys for OpenTelemetry baggage
TASK_RUN_ID_KEY = "hud.task_run_id"
IS_ROOT_TRACE_KEY = "hud.is_root_trace"


def set_current_task_run_id(task_run_id: str | None) -> contextvars.Token:
    """Set the current task run ID."""
    return current_task_run_id.set(task_run_id)


def get_current_task_run_id() -> str | None:
    """Get current task_run_id from either contextvars or OTel baggage."""
    # First try OTel baggage
    task_run_id = baggage.get_baggage(TASK_RUN_ID_KEY)
    if task_run_id:
        return task_run_id
    
    # Fallback to contextvars
    return current_task_run_id.get()


def is_root_trace() -> bool:
    """Check if current context is a root trace."""
    # First try OTel baggage
    is_root = baggage.get_baggage(IS_ROOT_TRACE_KEY)
    if is_root is not None:
        return is_root.lower() == "true"
    
    # Fallback to contextvars
    return is_root_trace_var.get()


@contextmanager
def span_context(
    name: str,
    attributes: dict[str, Any] | None = None,
    kind: otel_trace.SpanKind = otel_trace.SpanKind.INTERNAL,
) -> Generator[otel_trace.Span, None, None]:
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
    tracer = otel_trace.get_tracer("hud-sdk")
    
    # Current task_run_id will be added by HudEnrichmentProcessor
    with tracer.start_as_current_span(
        name,
        attributes=attributes,
        kind=kind,
    ) as span:
        yield span


async def _update_task_status_async(
    task_run_id: str, 
    status: str,
    job_id: str | None = None,
    error_message: str | None = None,
) -> None:
    """Async task status update."""
    if not settings.telemetry_enabled:
        return
    
    try:
        data = {"status": status}
        if job_id:
            data["job_id"] = job_id
        if error_message:
            data["error_message"] = error_message
            
        await make_request(
            method="POST",
            url=f"{settings.base_url}/v2/task_runs/{task_run_id}/status",
            json=data,
            api_key=settings.api_key,
        )
        logger.debug(f"Updated task {task_run_id} status to {status}")
    except Exception as e:
        # Suppress warnings about interpreter shutdown
        if "interpreter shutdown" not in str(e):
            logger.warning(f"Failed to update task status: {e}")


def _fire_and_forget_status_update(
    task_run_id: str,
    status: str,
    job_id: str | None = None,
    error_message: str | None = None,
) -> None:
    """Fire and forget status update - works in any context including Jupyter."""
    fire_and_forget(
        _update_task_status_async(task_run_id, status, job_id, error_message),
        f"update task {task_run_id} status to {status}"
    )


class trace:
    """Internal OpenTelemetry trace context manager.
    
    This is the implementation class. Users should use hud.trace() instead.
    """
    
    def __init__(
        self,
        task_run_id: str,
        is_root: bool = True,
        span_name: str = "hud.task",
        attributes: dict[str, Any] | None = None,
        job_id: str | None = None,
    ):
        self.task_run_id = task_run_id
        self.job_id = job_id
        self.is_root = is_root
        self.span_name = span_name
        self.attributes = attributes or {}
        self._span: Optional[otel_trace.Span] = None
        self._otel_token: Optional[object] = None
        self._task_run_token = None
        self._root_token = None
        
    def __enter__(self) -> str:
        """Enter the trace context and return the task_run_id."""
        # Set context variables
        self._task_run_token = set_current_task_run_id(self.task_run_id)
        self._root_token = is_root_trace_var.set(self.is_root)
        
        # Set OpenTelemetry baggage for propagation
        ctx = baggage.set_baggage(TASK_RUN_ID_KEY, self.task_run_id)
        ctx = baggage.set_baggage(IS_ROOT_TRACE_KEY, str(self.is_root), context=ctx)
        if self.job_id:
            ctx = baggage.set_baggage("hud.job_id", self.job_id, context=ctx)
        self._otel_token = context.attach(ctx)
        
        # Start a span
        tracer = otel_trace.get_tracer("hud-sdk")
        span_attrs = {
            "hud.task_run_id": self.task_run_id,
            "hud.is_root_trace": self.is_root,
            **self.attributes,
        }
        if self.job_id:
            span_attrs["hud.job_id"] = self.job_id
        
        self._span = tracer.start_span(
            self.span_name,
            attributes=span_attrs,
        )
        self._span.__enter__()
        
        # Update task status to initializing if root
        if self.is_root:
            _fire_and_forget_status_update(self.task_run_id, "initializing", job_id=self.job_id)
        
        logger.debug("Started HUD trace context for task_run_id=%s", self.task_run_id)
        return self.task_run_id
        
    def __exit__(
        self, 
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None, 
        exc_tb: TracebackType | None
    ) -> None:
        """Exit the trace context."""
        # Update task status if root
        if self.is_root:
            if exc_type is not None:
                _fire_and_forget_status_update(
                    self.task_run_id, 
                    "error", 
                    job_id=self.job_id,
                    error_message=str(exc_val)
                )
            else:
                _fire_and_forget_status_update(
                    self.task_run_id, 
                    "completed", 
                    job_id=self.job_id
                )
        
        # End the span
        if self._span:
            if exc_type is not None:
                self._span.record_exception(exc_val)
                self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            else:
                self._span.set_status(Status(StatusCode.OK))
            self._span.__exit__(exc_type, exc_val, exc_tb)
        
        # Detach OpenTelemetry context
        if self._otel_token:
            context.detach(self._otel_token)
        
        # Reset context variables
        if self._task_run_token is not None:
            current_task_run_id.reset(self._task_run_token)  # type: ignore
        if self._root_token is not None:
            is_root_trace_var.reset(self._root_token)  # type: ignore
        
        logger.debug("Ended HUD trace context for task_run_id=%s", self.task_run_id)