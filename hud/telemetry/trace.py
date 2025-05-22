from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeVar

from hud.telemetry.context import (
    flush_buffer,
    get_current_task_run_id,
    is_root_trace,
    set_current_task_run_id,
)
from hud.telemetry.exporter import export_telemetry as export_telemetry_coro
from hud.telemetry.exporter import submit_to_worker_loop
from hud.telemetry.instrumentation.registry import registry

if TYPE_CHECKING:
    from collections.abc import Generator

    from hud.telemetry.mcp_models import BaseMCPCall

logger = logging.getLogger("hud.telemetry")
T = TypeVar("T")

def init_telemetry() -> None:
    """Initialize telemetry instrumentors and ensure worker is started if telemetry is active."""
    registry.install_all()
    logger.info("HUD Telemetry initialized.")

@contextmanager
def trace(
    attributes: dict[str, Any] | None = None,
) -> Generator[str, None, None]:
    """
    Context manager for tracing a block of code.
    The task_run_id is always generated internally as a UUID.
    Telemetry export is handled by a background worker thread.
    
    Args:
        attributes: Optional dictionary of attributes to associate with this trace
        
    Returns:
        The generated task run ID (UUID string) used for this trace
    """
    task_run_id = str(uuid.uuid4())
    
    if attributes is None:
        attributes = {}
    
    start_time = time.time()
    logger.debug("Starting trace %s", task_run_id)
    
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
        
        mcp_calls: list[BaseMCPCall] = flush_buffer()
        
        trace_attributes = {
            **attributes,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "is_root": is_root,
        }
        
        if is_root and mcp_calls:
            try:
                coro_to_submit = export_telemetry_coro(
                    task_run_id=task_run_id,
                    trace_attributes=trace_attributes,
                    mcp_calls=mcp_calls
                )
                future = submit_to_worker_loop(coro_to_submit)
                if future:
                    logger.debug("Telemetry for trace %s submitted to background worker.", task_run_id)
                else:
                    logger.warning("Failed to submit telemetry for trace %s to background worker (loop not available).", task_run_id)
            except Exception as e:
                logger.warning("Failed to submit telemetry for trace %s: %s", task_run_id, e)
        
        set_current_task_run_id(previous_task_id)
        is_root_trace.set(was_root)
        
        logger.debug("Ended trace %s with %d MCP call(s)", task_run_id, len(mcp_calls))

        logger.info("[hud] View trace at https://app.hud.so/jobs/traces/%s", task_run_id)

