from __future__ import annotations

import contextvars
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hud.telemetry")

# Context variables for tracing
current_task_run_id = contextvars.ContextVar("current_task_run_id", default=None)
mcp_calls_buffer = contextvars.ContextVar("mcp_calls_buffer", default=[])
is_root_trace = contextvars.ContextVar("is_root_trace", default=False)

# Maximum buffer size before automatic flush
MAX_BUFFER_SIZE = 100

def get_current_task_run_id() -> Optional[str]:
    """Get the task_run_id for the current trace context."""
    return current_task_run_id.get()

def set_current_task_run_id(task_run_id: Optional[str]) -> None:
    """Set the task_run_id for the current trace context."""
    # ContextVar.set doesn't allow None, so we need to handle this case
    # by using an empty string as a sentinel value for None
    current_task_run_id.set("" if task_run_id is None else task_run_id)

def buffer_mcp_call(call_data: Dict[str, Any]) -> None:
    """
    Add an MCP call to the buffer for the current trace.
    
    Args:
        call_data: Dictionary containing details of the MCP call
    """
    # Only buffer if we have an active trace
    task_run_id = get_current_task_run_id()
    if task_run_id is not None and task_run_id != "":
        buffer = mcp_calls_buffer.get()
        # Make a copy to avoid modifying objects outside our control
        buffer.append(call_data.copy() if isinstance(call_data, dict) else call_data)
        mcp_calls_buffer.set(buffer)
        
        # Auto-flush if buffer gets too large
        if len(buffer) >= MAX_BUFFER_SIZE:
            logger.debug(f"MCP calls buffer reached size {len(buffer)}, auto-flushing")
            flush_buffer(export=True)

def flush_buffer(export: bool = False) -> List[Dict[str, Any]]:
    """
    Clear the MCP calls buffer and return its contents.
    
    Args:
        export: Whether to trigger export of this buffer
        
    Returns:
        The list of buffered MCP calls
    """
    buffer = mcp_calls_buffer.get()
    # Reset buffer to empty list
    mcp_calls_buffer.set([])
    
    if export and buffer:
        # Handle export if requested - this branch is for future expansion
        # Currently the export is done in the trace context manager
        pass
        
    return buffer 