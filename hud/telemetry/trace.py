from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, AsyncGenerator, Callable, Dict, Generator, List, Optional, TypeVar, cast

from hud.telemetry.context import (
    buffer_mcp_call,
    flush_buffer,
    get_current_task_run_id,
    is_root_trace,
    set_current_task_run_id,
)
from hud.telemetry.exporter import export_telemetry
from hud.telemetry.instrumentation.registry import registry

logger = logging.getLogger("hud.telemetry")
T = TypeVar("T")

def init_telemetry() -> None:
    """Initialize telemetry instrumentors."""
    registry.install_all()

@contextmanager
def trace(
    task_run_id: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator[str, None, None]:
    """
    Context manager for tracing a synchronous block of code.
    
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
    logger.debug(f"Starting trace {task_run_id}")
    
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
        
        # Only export if this is a root trace or we have MCP calls
        if is_root and mcp_calls:
            # Try to export telemetry async
            try:
                # Run in executor to avoid blocking if we're in sync context
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(export_telemetry(
                        task_run_id=task_run_id,
                        trace_attributes=trace_attributes,
                        mcp_calls=mcp_calls
                    ))
                else:
                    logger.debug("No running event loop, skipping telemetry export")
            except Exception as e:
                logger.warning(f"Failed to schedule telemetry export: {e}")
        
        # Restore previous context
        set_current_task_run_id(previous_task_id)
        is_root_trace.set(was_root)
        
        logger.debug(f"Ended trace {task_run_id} with {len(mcp_calls)} MCP call(s)")

async def async_trace(
    task_run_id: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
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
    logger.debug(f"Starting async trace {task_run_id}")
    
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
        
        # Only export if this is a root trace or we have MCP calls
        if is_root and mcp_calls:
            # Export telemetry
            try:
                await export_telemetry(
                    task_run_id=task_run_id,
                    trace_attributes=trace_attributes,
                    mcp_calls=mcp_calls
                )
            except Exception as e:
                logger.warning(f"Failed to export telemetry: {e}")
        
        # Restore previous context
        set_current_task_run_id(previous_task_id)
        is_root_trace.set(was_root)
        
        logger.debug(f"Ended async trace {task_run_id} with {len(mcp_calls)} MCP call(s)")

# For backward compatibility - will be deprecated
start_trace = async_trace 