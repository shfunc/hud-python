from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    ParamSpec,
    TypeVar,
)

from hud.telemetry import exporter
from hud.telemetry.context import (
    flush_buffer,
    get_current_task_run_id,
    is_root_trace,
    set_current_task_run_id,
)
from hud.telemetry.exporter import submit_to_worker_loop
from hud.telemetry.instrumentation.registry import registry

if TYPE_CHECKING:
    from collections.abc import Generator


logger = logging.getLogger("hud.telemetry")
T = TypeVar("T")
P = ParamSpec("P")

# Track whether telemetry has been initialized
_telemetry_initialized = False


def init_telemetry() -> None:
    """Initialize telemetry instrumentors and ensure worker is started if telemetry is active."""
    global _telemetry_initialized
    if _telemetry_initialized:
        return

    registry.install_all()
    logger.info("Telemetry initialized.")
    _telemetry_initialized = True


def _ensure_telemetry_initialized() -> None:
    """Ensure telemetry is initialized - called lazily by trace functions."""
    from hud.settings import settings

    if settings.telemetry_enabled and not _telemetry_initialized:
        init_telemetry()


@contextmanager
def trace_open(
    name: str | None = None,
    run_id: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Generator[str, None, None]:
    """
    Context manager for tracing a block of code.

    Args:
        name: Optional name for this trace, will be added to attributes.
        attributes: Optional dictionary of attributes to associate with this trace

    Returns:
        The generated task run ID (UUID string) used for this trace
    """
    # Lazy initialization - only initialize telemetry when trace() is actually called
    _ensure_telemetry_initialized()

    task_run_id = run_id or str(uuid.uuid4())

    logger.info("See your agent live at https://app.hud.so/trace/%s", task_run_id)

    local_attributes = attributes.copy() if attributes is not None else {}
    if name is not None:
        local_attributes["trace_name"] = name

    start_time = time.time()
    logger.debug("Starting trace %s (Name: %s)", task_run_id, name if name else "Unnamed")

    previous_task_id = get_current_task_run_id()
    was_root = is_root_trace.get()

    set_current_task_run_id(task_run_id)
    is_root = previous_task_id is None
    is_root_trace.set(is_root)

    try:
        yield task_run_id
    finally:
        end_time = time.time()
        duration = end_time - start_time
        local_attributes["duration_seconds"] = duration
        local_attributes["is_root_trace"] = is_root

        logger.debug("Finishing trace %s after %.2f seconds", task_run_id, duration)

        # Always flush the buffer for the current task
        mcp_calls = flush_buffer(export=True)
        logger.debug("Flushed %d MCP calls for trace %s", len(mcp_calls), task_run_id)

        # Submit the telemetry payload to the worker queue
        if is_root and mcp_calls:
            coro = exporter.export_telemetry(
                task_run_id=task_run_id,
                trace_attributes=local_attributes,
                mcp_calls=mcp_calls,
            )
            submit_to_worker_loop(coro)

        # Restore previous context
        set_current_task_run_id(previous_task_id)
        is_root_trace.set(was_root)

        # Log at the end
        if is_root:
            view_url = f"https://app.hud.so/trace/{task_run_id}"
            logger.info("View trace at %s", view_url)


@contextmanager
def trace(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Generator[str, None, None]:
    """
    Synchronous context manager that traces and blocks until telemetry is sent.

    This is the "worry-free" option when you want to ensure telemetry is
    sent immediately before continuing, rather than relying on background workers.

    Args:
        name: Optional name for this trace
        attributes: Optional attributes for the trace

    Returns:
        The generated task run ID (UUID string) used for this trace
    """
    with trace_open(name=name, attributes=attributes) as task_run_id:
        yield task_run_id

    # Ensure telemetry is flushed synchronously
    from hud import flush

    flush()


def trace_decorator(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Any:
    """
    Decorator for tracing functions.

    Can be used on both sync and async functions.
    """

    def decorator(func: Any) -> Any:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                func_name = name or f"{func.__module__}.{func.__name__}"
                with trace_open(name=func_name, attributes=attributes):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                func_name = name or f"{func.__module__}.{func.__name__}"
                with trace_open(name=func_name, attributes=attributes):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator
