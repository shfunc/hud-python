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
        
        # Start the export task if it doesn't exist or is done
        if _export_task is None or _export_task.done():
            _export_task = asyncio.create_task(_process_export_queue())
            logger.debug("Started telemetry export task")

async def _process_export_queue() -> None:
    """Process the export queue periodically."""
    try:
        while True:
            # Check if there's anything to export
            batch = []
            async with _export_lock:
                if not _export_queue:
                    return  # Queue empty, task done
                    
                # Take up to MAX_BATCH_SIZE items
                batch = _export_queue[:MAX_BATCH_SIZE]
                del _export_queue[:len(batch)]
            
            if batch:
                # Export the batch
                await _export_batch(batch)
            
            # Sleep to avoid spinning too fast
            await asyncio.sleep(EXPORT_INTERVAL)
            
            # If queue is empty after sleep, we can exit
            async with _export_lock:
                if not _export_queue:
                    return
    except Exception as e:
        logger.error(f"Error processing telemetry export queue: {e}")

async def _export_batch(batch: List[Dict[str, Any]]) -> None:
    """Export a batch of telemetry data to the HUD telemetry service."""
    if not settings.telemetry_enabled:
        logger.debug("Telemetry export skipped - telemetry not enabled")
        return

    telemetry_url = f"{settings.api_url}/telemetry"
    
    try:
        async with httpx.AsyncClient() as client:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.api_key}"
            }
            
            response = await client.post(
                telemetry_url,
                json=batch,
                headers=headers,
                timeout=30.0
            )
            
            if response.status_code == 200:
                logger.debug(f"Successfully exported {len(batch)} telemetry item(s)")
            else:
                logger.warning(f"Failed to export telemetry: HTTP {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error exporting telemetry: {e}") 