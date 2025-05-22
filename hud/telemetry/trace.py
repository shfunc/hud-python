from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, TypeVar

from hud.telemetry.context import (
    flush_buffer,
    get_current_task_run_id,
    is_root_trace,
    set_current_task_run_id,
)
from hud.telemetry.exporter import export_telemetry
from hud.telemetry.instrumentation.registry import registry

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger("hud.telemetry")
T = TypeVar("T")

def init_telemetry() -> None:
    """Initialize telemetry instrumentors."""
    registry.install_all()

async def trace(
    task_run_id: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Async context manager for tracing an asynchronous block of code.
    
    Args:
        task_run_id: Optional ID for this task run, will be generated if not provided
        attributes: Optional dictionary of attributes to associate with this trace
        
    Returns:
        The task run ID used for this trace
    """
    # Generate a task_run_id if none provided
    if task_run_id is None:
        task_run_id = str(uuid.uuid4())
    
    # Default attributes
    if attributes is None:
        attributes = {}
    
    # Record trace start
    start_time = time.time()
    logger.debug("Starting trace %s", task_run_id)
    
    # Save previous context
    previous_task_id = get_current_task_run_id()
    was_root = is_root_trace.get()
    
    # Set new context
    set_current_task_run_id(task_run_id)
    is_root = previous_task_id is None
    is_root_trace.set(is_root)
    
    try:
        # Yield the task_run_id to the caller
        yield task_run_id
    finally:
        # Capture end time
        end_time = time.time()
        duration = end_time - start_time
        
        # Get any buffered MCP calls
        mcp_calls = flush_buffer()
        
        # Add our own metadata to the trace
        trace_attributes = {
            **attributes,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "is_root": is_root,
        }
        
        # Only export if this is a root trace and we have MCP calls
        if is_root: # and mcp_calls:
            # Export telemetry
            try:
                await export_telemetry(
                    task_run_id=task_run_id,
                    trace_attributes=trace_attributes,
                    mcp_calls=[call.model_dump() for call in mcp_calls]
                )
            except Exception as e:
                logger.warning("Failed to export telemetry: %s", e)
        
        # Restore previous context
        set_current_task_run_id(previous_task_id)
        is_root_trace.set(was_root)
        
        logger.debug(
            "Ended trace %s with %d MCP call(s)",
            task_run_id,
            len(mcp_calls)
        )
