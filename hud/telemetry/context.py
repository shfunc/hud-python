from __future__ import annotations

import contextvars
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, TypeVar

from hud.telemetry.mcp_models import (
    BaseMCPCall,
    MCPNotificationCall,
    MCPRequestCall,
    MCPResponseCall,
    StatusType,
)

logger = logging.getLogger("hud.telemetry")

# Context variables for tracing
current_task_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_task_run_id", default=None
)
# Global dictionary for buffering, keyed by task_run_id
_GLOBAL_MCP_CALL_BUFFERS: defaultdict[str, list[BaseMCPCall]] = defaultdict(list)
# Track the last exported index for each task_run_id
_GLOBAL_EXPORT_INDICES: defaultdict[str, int] = defaultdict(int)
# Track whether we've seen a non-init request for each task_run_id
_GLOBAL_HAS_NON_INIT_REQUEST: defaultdict[str, bool] = defaultdict(bool)
is_root_trace: contextvars.ContextVar[bool] = contextvars.ContextVar("is_root_trace", default=False)

# Maximum buffer size before automatic flush
MAX_BUFFER_SIZE = 100

# Type variable for record factories
T = TypeVar("T", bound=BaseMCPCall)


def get_current_task_run_id() -> str | None:
    """Get the task_run_id for the current trace context."""
    return current_task_run_id.get()


def set_current_task_run_id(task_run_id: str | None) -> None:
    """Set the task_run_id for the current trace context."""
    current_task_run_id.set(task_run_id)


def buffer_mcp_call(record: BaseMCPCall | dict[str, Any]) -> None:
    """Buffer an MCP call record for the current trace."""
    task_run_id = get_current_task_run_id()

    if not task_run_id:
        logger.warning(
            "BUFFER_MCP_CALL: No task_run_id. Skipping buffer for %s", type(record).__name__
        )
        return

    # Ensure 'record' is a Pydantic model instance
    if isinstance(record, dict):
        try:
            record_model = BaseMCPCall.from_dict(record)
            record = record_model
        except Exception as e_conv:
            logger.exception("BUFFER_MCP_CALL: Failed to convert dict to BaseMCPCall: %s", e_conv)
            return

    _GLOBAL_MCP_CALL_BUFFERS[task_run_id].append(record)
    buffer_len = len(_GLOBAL_MCP_CALL_BUFFERS[task_run_id])

    if buffer_len >= MAX_BUFFER_SIZE:
        flush_buffer(export=True)


def export_incremental() -> list[BaseMCPCall]:
    """
    Export only new MCP calls since last export without clearing the buffer.

    Returns:
        The list of newly exported MCP calls
    """
    task_run_id = get_current_task_run_id()
    if not task_run_id or not is_root_trace.get():
        return []

    buffer = _GLOBAL_MCP_CALL_BUFFERS.get(task_run_id, [])
    last_exported_idx = _GLOBAL_EXPORT_INDICES.get(task_run_id, 0)

    # Get only the new records since last export
    new_records = buffer[last_exported_idx:]

    if new_records:
        # Update the export index
        _GLOBAL_EXPORT_INDICES[task_run_id] = len(buffer)

        # Trigger export
        from hud.telemetry import exporter
        from hud.telemetry.exporter import submit_to_worker_loop

        # Get current trace attributes if available
        attributes = {"incremental": True}

        coro = exporter.export_telemetry(
            task_run_id=task_run_id,
            trace_attributes=attributes,
            mcp_calls=new_records.copy(),  # Copy to avoid modification during export
        )
        submit_to_worker_loop(coro)

        logger.debug(
            "Incremental export: %d new MCP calls for trace %s", len(new_records), task_run_id
        )

    return new_records


def flush_buffer(export: bool = False) -> list[BaseMCPCall]:
    """
    Clear the MCP calls buffer and return its contents.

    Args:
        export: Whether to trigger export of this buffer

    Returns:
        The list of buffered MCP calls
    """
    task_run_id = get_current_task_run_id()
    if not task_run_id:
        logger.warning("FLUSH_BUFFER: No current task_run_id. Cannot flush.")
        return []

    buffer_for_task = _GLOBAL_MCP_CALL_BUFFERS.pop(task_run_id, [])
    # Clean up export index when buffer is flushed
    _GLOBAL_EXPORT_INDICES.pop(task_run_id, None)
    # Clean up non-init request tracking
    _GLOBAL_HAS_NON_INIT_REQUEST.pop(task_run_id, None)
    return buffer_for_task


def create_request_record(
    method: str, status: StatusType = StatusType.STARTED, **kwargs: Any
) -> MCPRequestCall:
    """Create and buffer a request record"""
    task_run_id = get_current_task_run_id()
    if not task_run_id:
        logger.warning("No active task_run_id, request record will not be created")
        raise ValueError("No active task_run_id")

    # Check if this is the first non-init request and update status
    if is_root_trace.get() and not _GLOBAL_HAS_NON_INIT_REQUEST[task_run_id]:
        # Common initialization method patterns
        init_methods = {"initialize", "session/new", "init", "setup", "connect"}
        method_lower = method.lower()

        # Check if this is NOT an initialization method
        if not any(init_pattern in method_lower for init_pattern in init_methods):
            _GLOBAL_HAS_NON_INIT_REQUEST[task_run_id] = True

            # Update status to running
            from hud.telemetry.exporter import (
                TaskRunStatus,
                submit_to_worker_loop,
                update_task_run_status,
            )

            coro = update_task_run_status(task_run_id, TaskRunStatus.RUNNING)
            submit_to_worker_loop(coro)
            logger.debug(
                "Updated task run %s status to RUNNING on first non-init request: %s",
                task_run_id,
                method,
            )

    record = MCPRequestCall(
        task_run_id=task_run_id,
        method=method,
        status=status,
        start_time=kwargs.pop("start_time", None) or datetime.now().timestamp(),
        **kwargs,
    )
    buffer_mcp_call(record)
    return record


def create_response_record(
    method: str, related_request_id: str | int | None = None, is_error: bool = False, **kwargs: Any
) -> MCPResponseCall:
    """Create and buffer a response record"""
    task_run_id = get_current_task_run_id()
    if not task_run_id:
        logger.warning("No active task_run_id, response record will not be created")
        raise ValueError("No active task_run_id")

    # Default to COMPLETED status if not provided
    if "status" not in kwargs:
        kwargs["status"] = StatusType.COMPLETED

    record = MCPResponseCall(
        task_run_id=task_run_id,
        method=method,
        related_request_id=related_request_id,
        is_error=is_error,
        **kwargs,
    )

    buffer_mcp_call(record)

    # Trigger incremental export when we receive a response
    export_incremental()

    return record


def create_notification_record(
    method: str, status: StatusType = StatusType.STARTED, **kwargs: Any
) -> MCPNotificationCall:
    """Create and buffer a notification record"""
    task_run_id = get_current_task_run_id()
    if not task_run_id:
        logger.warning("No active task_run_id, notification record will not be created")
        raise ValueError("No active task_run_id")

    record = MCPNotificationCall(
        task_run_id=task_run_id,
        method=method,
        status=status,
        start_time=kwargs.pop("start_time", None) or datetime.now().timestamp(),
        **kwargs,
    )
    buffer_mcp_call(record)
    return record
