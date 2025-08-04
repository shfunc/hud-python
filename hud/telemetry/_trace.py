from __future__ import annotations

# ruff: noqa: T201
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

from hud.telemetry.context import (
    flush_buffer,
    get_current_task_run_id,
    is_root_trace,
    set_current_task_run_id,
)
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


def _detect_agent_model() -> str | None:
    """
    Try to auto-detect agent model from parent frames.
    This is a best-effort approach and may not work in all cases.
    """
    import sys

    try:
        # Try different frame depths (2-3 typically covers most cases)
        for depth in range(2, 3):
            try:
                frame = sys._getframe(depth)
                # Check local variables for agent objects
                for var_value in frame.f_locals.values():
                    # Look for objects with model_name attribute
                    if hasattr(var_value, "model_name") and hasattr(var_value, "run"):
                        # Likely an agent object
                        model_name = getattr(var_value, "model_name", None)
                        if model_name:
                            logger.debug(
                                "Found agent with model_name in frame %d: %s", depth, model_name
                            )
                            return str(model_name)

                # Also check self in case we're in a method
                if "self" in frame.f_locals:
                    self_obj = frame.f_locals["self"]
                    if hasattr(self_obj, "model_name"):
                        model_name = getattr(self_obj, "model_name", None)
                        if model_name:
                            logger.debug(
                                "Found agent model_name in self at frame %d: %s", depth, model_name
                            )
                            return str(model_name)

            except (ValueError, AttributeError):
                # Frame doesn't exist at this depth or other issues
                continue

    except Exception as e:
        logger.debug("Agent model detection failed: %s", e)

    return None


def _print_trace_url(task_run_id: str) -> None:
    """Print the trace URL in a colorful box."""
    url = f"https://app.hud.so/trace/{task_run_id}"
    header = "🚀 See your agent live at:"

    # ANSI color codes
    DIM = "\033[90m"  # Dim/Gray for border (visible on both light and dark terminals)
    GOLD = "\033[33m"  # Gold/Yellow for URL
    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Calculate box width based on the longest line
    box_width = max(len(url), len(header)) + 6

    # Box drawing characters
    top_border = "╔" + "═" * (box_width - 2) + "╗"
    bottom_border = "╚" + "═" * (box_width - 2) + "╝"
    divider = "╟" + "─" * (box_width - 2) + "╢"

    # Center the content
    header_padding = (box_width - len(header) - 2) // 2
    url_padding = (box_width - len(url) - 2) // 2

    # Print the box
    print(f"\n{DIM}{top_border}{RESET}")
    print(
        f"{DIM}║{RESET}{' ' * header_padding}{header}{' ' * (box_width - len(header) - header_padding - 3)}{DIM}║{RESET}"  # noqa: E501
    )
    print(f"{DIM}{divider}{RESET}")
    print(
        f"{DIM}║{RESET}{' ' * url_padding}{BOLD}{GOLD}{url}{RESET}{' ' * (box_width - len(url) - url_padding - 2)}{DIM}║{RESET}"  # noqa: E501
    )
    print(f"{DIM}{bottom_border}{RESET}\n")


def _print_trace_complete_url(task_run_id: str) -> None:
    """Print the trace completion URL in a simple colorful format."""
    url = f"https://app.hud.so/trace/{task_run_id}"

    # ANSI color codes
    GREEN = "\033[92m"
    GOLD = "\033[33m"
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"

    print(f"\n{GREEN}✓ Trace complete!{RESET} {DIM}View at:{RESET} {BOLD}{GOLD}{url}{RESET}\n")


@contextmanager
def trace_open(
    name: str | None = None,
    agent_model: str | None = None,
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

    _print_trace_url(task_run_id)

    local_attributes = attributes.copy() if attributes is not None else {}
    if name is not None:
        local_attributes["trace_name"] = name

    # Auto-detect agent if not explicitly provided
    if agent_model is None:
        agent_model = _detect_agent_model()

    start_time = time.time()
    logger.debug("Starting trace %s (Name: %s)", task_run_id, name if name else "Unnamed")

    previous_task_id = get_current_task_run_id()
    was_root = is_root_trace.get()

    set_current_task_run_id(task_run_id)
    is_root = previous_task_id is None
    is_root_trace.set(is_root)

    # Update status to initializing for root traces
    if is_root:
        from hud.telemetry.exporter import (
            TaskRunStatus,
            submit_to_worker_loop,
            update_task_run_status,
        )
        from hud.telemetry.job import get_current_job_id

        # Include metadata in the initial status update
        initial_metadata = local_attributes.copy()
        initial_metadata["is_root_trace"] = is_root
        if agent_model:
            initial_metadata["agent_model"] = agent_model

        # Get job_id if we're in a job context
        job_id = get_current_job_id()

        coro = update_task_run_status(
            task_run_id, TaskRunStatus.INITIALIZING, metadata=initial_metadata, job_id=job_id
        )
        submit_to_worker_loop(coro)
        logger.debug("Updated task run %s status to INITIALIZING with metadata", task_run_id)

    error_occurred = False
    error_message = None

    try:
        yield task_run_id
    except Exception as e:
        error_occurred = True
        error_message = str(e)
        raise
    finally:
        end_time = time.time()
        duration = end_time - start_time
        local_attributes["duration_seconds"] = duration
        local_attributes["is_root_trace"] = is_root

        logger.debug("Finishing trace %s after %.2f seconds", task_run_id, duration)

        # Update status for root traces
        if is_root:
            from hud.telemetry.exporter import (
                TaskRunStatus,
                submit_to_worker_loop,
                update_task_run_status,
            )

            # Include final metadata with duration
            final_metadata = local_attributes.copy()

            if error_occurred:
                coro = update_task_run_status(
                    task_run_id, TaskRunStatus.ERROR, error_message, metadata=final_metadata
                )
                logger.debug("Updated task run %s status to ERROR: %s", task_run_id, error_message)
            else:
                coro = update_task_run_status(
                    task_run_id, TaskRunStatus.COMPLETED, metadata=final_metadata
                )
                logger.debug("Updated task run %s status to COMPLETED with metadata", task_run_id)

            # Wait for the status update to complete
            future = submit_to_worker_loop(coro)
            if future:
                try:
                    # Wait up to 5 seconds for the status update
                    import concurrent.futures

                    future.result(timeout=5.0)
                    logger.debug("Status update completed successfully")
                except concurrent.futures.TimeoutError:
                    logger.warning("Timeout waiting for status update to complete")
                except Exception as e:
                    logger.error("Error waiting for status update: %s", e)

        # Export any remaining records before flushing
        if is_root:
            from hud.telemetry.context import export_incremental

            export_incremental()

        # Always flush the buffer for the current task
        mcp_calls = flush_buffer(export=True)
        logger.debug("Flushed %d MCP calls for trace %s", len(mcp_calls), task_run_id)

        # Restore previous context
        set_current_task_run_id(previous_task_id)
        is_root_trace.set(was_root)

        # Log at the end
        if is_root:
            _print_trace_complete_url(task_run_id)


@contextmanager
def trace(
    name: str | None = None,
    agent_model: str | None = None,
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
    with trace_open(name=name, agent_model=agent_model, attributes=attributes) as task_run_id:
        yield task_run_id

    # Ensure telemetry is flushed synchronously
    from hud import flush

    flush()


def trace_decorator(
    name: str | None = None,
    agent_model: str | None = None,
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
                with trace_open(name=func_name, agent_model=agent_model, attributes=attributes):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                func_name = name or f"{func.__module__}.{func.__name__}"
                with trace_open(name=func_name, agent_model=agent_model, attributes=attributes):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator
