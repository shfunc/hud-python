from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from datetime import datetime, timezone # For ISO timestamp conversion
import json

import httpx

from hud.settings import settings
# Import BaseMCPCall and TrajectoryStep for type hinting and transformation
from hud.telemetry.mcp_models import BaseMCPCall, TrajectoryStep, MCPResponseCall # MCPResponseCall for isinstance check

logger = logging.getLogger("hud.telemetry")

# Export queue and lock for async operations
_export_queue: list[dict[str, Any]] = []
_export_lock = asyncio.Lock()
_export_task: asyncio.Task | None = None

# Constants for export behavior
MAX_BATCH_SIZE = 50
EXPORT_INTERVAL = 5.0  # seconds

async def export_telemetry(
    task_run_id: str,
    trace_attributes: dict[str, Any],
    mcp_calls: list[BaseMCPCall]  # Type hint is now list[BaseMCPCall]
) -> None:
    """
    Export telemetry data to the HUD telemetry service.
    
    Args:
        task_run_id: The task run ID associated with this telemetry
        trace_attributes: Attributes of the trace
        mcp_calls: List of MCP call Pydantic models to export
    """
    trajectory_steps_data: list[dict[str, Any]] = []
    for mcp_call_model in mcp_calls:
        action_data = mcp_call_model.model_dump()
        
        start_ts_iso = None
        end_ts_iso = None

        # Get start_time if available (e.g. on MCPRequestCall, MCPNotificationCall)
        actual_start_time_float = getattr(mcp_call_model, 'start_time', None)
        if actual_start_time_float:
            start_ts_iso = datetime.fromtimestamp(actual_start_time_float, timezone.utc).isoformat().replace("+00:00", "Z")
        
        # Use 'end_time' if available, otherwise fall back to 'timestamp' for the end_timestamp
        actual_end_time_float = getattr(mcp_call_model, 'end_time', None)
        effective_end_timestamp_float = actual_end_time_float if actual_end_time_float else mcp_call_model.timestamp

        if effective_end_timestamp_float:
            end_ts_iso = datetime.fromtimestamp(effective_end_timestamp_float, timezone.utc).isoformat().replace("+00:00", "Z")

        # For events that are more like points in time (e.g., a received response that doesn't have a separate start_time field)
        # set start_timestamp to be the same as end_timestamp if start_timestamp wasn't explicitly set.
        if end_ts_iso and not start_ts_iso:
            start_ts_iso = end_ts_iso

        step_metadata: dict[str, Any] = {
            "mcp_method": mcp_call_model.method,
            "mcp_status": mcp_call_model.status.value,
            "mcp_call_type_original": mcp_call_model.call_type,
        }
        if mcp_call_model.direction:
            step_metadata["mcp_direction"] = mcp_call_model.direction.value
        if mcp_call_model.message_id is not None:
            step_metadata["mcp_message_id"] = str(mcp_call_model.message_id) # Ensure string
        
        # Specific handling for MCPResponseCall fields in metadata
        if isinstance(mcp_call_model, MCPResponseCall):
            step_metadata["mcp_is_error"] = mcp_call_model.is_error # bool is fine for JSON Any
            if mcp_call_model.is_error:
                if mcp_call_model.error is not None:
                    step_metadata["mcp_error_details"] = str(mcp_call_model.error) # Ensure string
                if mcp_call_model.error_type is not None:
                    step_metadata["mcp_error_type"] = str(mcp_call_model.error_type) # Ensure string

        obs_text = None
        if isinstance(mcp_call_model, MCPResponseCall) and mcp_call_model.response_data:
            result_data = mcp_call_model.response_data.get('result')
            if result_data is not None:
                try:
                    obs_text = json.dumps(result_data)
                except (TypeError, OverflowError):
                    obs_text = str(result_data)
        
        trajectory_step = TrajectoryStep(
            type="mcp-step",
            actions=[action_data],
            start_timestamp=start_ts_iso,
            end_timestamp=end_ts_iso,
            metadata=step_metadata,
            observation_text=obs_text
        )
        trajectory_steps_data.append(trajectory_step.model_dump())

    payload_to_queue = {
        "task_run_id": task_run_id,
        "attributes": trace_attributes,
        "mcp_calls": trajectory_steps_data,
        "timestamp": time.time()
    }
    
    await _queue_for_export(payload_to_queue)

async def _queue_for_export(payload: dict[str, Any]) -> None:
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
            payload_to_export: dict[str, Any] | None = None
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

async def _export_trace_payload(payload: dict[str, Any]) -> None:
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
            
            logger.debug(
                "Exporting telemetry for task run %s to %s",
                task_run_id,
                telemetry_url,
            )
            response = await client.post(
                telemetry_url,
                json=data_to_send, # Send the structured attributes and mcp_calls
                headers=headers,
                timeout=30.0
            )
            
            if response.status_code >= 200 and response.status_code < 300:
                logger.debug(
                    "Successfully exported telemetry for task run %s. Status: %s",
                    task_run_id,
                    response.status_code,
                )
            else:
                logger.warning(
                    "Failed to export telemetry for task run %s: HTTP %s - %s",
                    task_run_id,
                    response.status_code,
                    response.text,
                )
    except Exception as e:
        logger.error("Error exporting telemetry for task run %s: %s", task_run_id, e)
