"""Trace context manager for HUD with OpenTelemetry integration.

This module provides the core trace() context manager that applications use.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from opentelemetry import baggage, context
from opentelemetry import trace as otel_trace
from opentelemetry.trace import Status, StatusCode

if TYPE_CHECKING:
    from types import TracebackType

import contextvars

from hud.server import make_request
from hud.settings import settings

# Context variables for task tracking
current_task_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_task_run_id", default=None
)
is_root_trace: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "is_root_trace", default=False
)

def set_current_task_run_id(task_run_id: str | None) -> contextvars.Token:
    """Set the current task run ID."""
    return current_task_run_id.set(task_run_id)

logger = logging.getLogger(__name__)

# Keys for OpenTelemetry baggage
TASK_RUN_ID_KEY = "hud.task_run_id"
IS_ROOT_TRACE_KEY = "hud.is_root_trace"


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
    import asyncio
    
    coro = _update_task_status_async(task_run_id, status, job_id, error_message)
    
    try:
        # Try to get current event loop
        loop = asyncio.get_running_loop()
        # Schedule the coroutine
        task = loop.create_task(coro)
        # Add error handler to prevent unhandled exceptions
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
    except RuntimeError:
        # No running event loop (e.g., Jupyter without %autoawait)
        # Use the same pattern as the old telemetry system
        try:
            # Try to run in a thread as a fallback
            import threading
            import asyncio
            
            def run_in_thread():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(coro)
                
            thread = threading.Thread(target=run_in_thread, daemon=True)
            thread.start()
        except Exception as e:
            # If that fails too, just log and continue
            # Special case: suppress "cannot schedule new futures after interpreter shutdown"
            if "interpreter shutdown" not in str(e):
                logger.debug(f"Could not update task status - no event loop available: {e}")


class trace:
    """Context manager for HUD traces with OpenTelemetry integration.
    
    Usage:
        with trace(task_run_id="my-task") as run_id:
            print(f"Running {run_id}")
            # Your code here
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
        self._root_token = is_root_trace.set(self.is_root)
        
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
            is_root_trace.reset(self._root_token)  # type: ignore
        
        logger.debug("Ended HUD trace context for task_run_id=%s", self.task_run_id)
