from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from hud.settings import settings

logger = logging.getLogger("hud.telemetry")

# Export queue and lock for async operations
_export_queue: List[Dict[str, Any]] = []
_export_lock = asyncio.Lock()
_export_task: Optional[asyncio.Task] = None

# Constants for export behavior
MAX_BATCH_SIZE = 50
EXPORT_INTERVAL = 5.0  # seconds

async def export_telemetry(
    task_run_id: str,
    trace_attributes: Dict[str, Any],
    mcp_calls: List[Dict[str, Any]]
) -> None:
    """
    Export telemetry data to the HUD telemetry service.
    
    Args:
        task_run_id: The task run ID associated with this telemetry
        trace_attributes: Attributes of the trace
        mcp_calls: List of MCP calls to export
    """
    if not mcp_calls:
        logger.debug(f"No MCP calls to export for task run {task_run_id}")
        return

    # Prepare the telemetry payload
    payload = {
        "task_run_id": task_run_id,
        "attributes": trace_attributes,
        "mcp_calls": mcp_calls,
        "timestamp": time.time()
    }
    
    # Queue the payload for async export
    await _queue_for_export(payload)

async def _queue_for_export(payload: Dict[str, Any]) -> None:
    """Add a payload to the export queue."""
    global _export_task
    
    async with _export_lock:
        _export_queue.append(payload)
        
        if _export_task is None or _export_task.done():
            _export_task = asyncio.create_task(_process_export_queue())
            logger.debug("Started telemetry export task")

async def _process_export_queue() -> None:
    """Process the export queue periodically, sending one trace at a time."""
    try:
        while True:
            payload_to_export: Optional[Dict[str, Any]] = None
            async with _export_lock:
                if not _export_queue:
                    logger.debug("Telemetry export queue empty, task will pause.")
                    _export_task = None # Allow task to complete and be restarted if new data comes
                    return 
                
                payload_to_export = _export_queue.pop(0)
            
            if payload_to_export:
                await _export_trace_payload(payload_to_export)
            
            await asyncio.sleep(EXPORT_INTERVAL)
            
    except asyncio.CancelledError:
        logger.info("Telemetry export task cancelled.")
        # When cancelled, _export_task.done() will be true. 
        # _queue_for_export will create a new task if needed.
        # Setting _export_task = None here could be problematic if cancellation happens
        # mid-operation in a way that _queue_for_export is called before this task fully exits.
        # It's generally safer to let the .done() check handle restart.
        raise
    except Exception as e:
        logger.error(f"Error processing telemetry export queue: {e}")
        # Similar to CancelledError, let the .done() check in _queue_for_export handle restart.
        # _export_task will be in a .done() state if an exception terminated it.

async def _export_trace_payload(payload: Dict[str, Any]) -> None:
    """Export a single trace payload to the HUD telemetry service."""
    if not settings.telemetry_enabled:
        logger.debug("Telemetry export skipped - telemetry not enabled")
        return

    task_run_id = payload.get("task_run_id")
    if not task_run_id:
        logger.warning("Payload missing task_run_id, skipping export")
        return

    # The payload itself is what we want to send (containing attributes and mcp_calls list)
    # The mcp_calls within the payload are already dumped dictionaries.
    data_to_send = {
        "attributes": payload.get("attributes", {}),
        "mcp_calls": payload.get("mcp_calls", [])
    }
    
    # Ensure mcp_calls is not empty if that's a requirement, or send as is. For now, send as is.
    # if not data_to_send["mcp_calls"]:
    #     logger.debug(f"No MCP calls in payload for task run {task_run_id}, skipping specific export if desired.")
    #     # Depending on backend, might not want to send empty mcp_calls list, or it's fine.
    
    telemetry_url = f"{settings.base_url}/v2/task_runs/{task_run_id}/telemetry-upload"
    
    try:
        async with httpx.AsyncClient() as client:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.api_key}"
            }
            
            logger.debug(f"Exporting telemetry for task run {task_run_id} to {telemetry_url}")
            response = await client.post(
                telemetry_url,
                json=data_to_send, # Send the structured attributes and mcp_calls
                headers=headers,
                timeout=30.0
            )
            
            if response.status_code >= 200 and response.status_code < 300:
                logger.debug(f"Successfully exported telemetry for task run {task_run_id}. Status: {response.status_code}")
            else:
                logger.warning(f"Failed to export telemetry for task run {task_run_id}: HTTP {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error exporting telemetry for task run {task_run_id}: {e}") 