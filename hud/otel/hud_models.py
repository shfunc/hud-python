"""Typed models for HUD telemetry using MCP types.

We always use these typed models internally, then serialize to dicts
for the actual export (whether to HUD or generic OTel backends).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Optional, Union
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict
from mcp.types import ClientRequest, ServerResult

logger = logging.getLogger(__name__)

from hud.types import TraceStep as HudSpanAttributes


class HudSpan(BaseModel):
    """A telemetry span ready for export."""
    name: str
    trace_id: str = Field(pattern=r"^[0-9a-fA-F]{32}$")
    span_id: str = Field(pattern=r"^[0-9a-fA-F]{16}$")
    parent_span_id: Optional[str] = Field(None, pattern=r"^[0-9a-fA-F]{16}$")
    
    start_time: str  # ISO format
    end_time: str    # ISO format
    
    status_code: str  # "UNSET", "OK", "ERROR"
    status_message: Optional[str] = None
    
    attributes: HudSpanAttributes
    exceptions: Optional[List[Dict[str, Any]]] = None
    
    model_config = ConfigDict(extra="forbid")


def extract_span_attributes(attrs: Dict[str, Any], method_name: Optional[str] = None, span_name: Optional[str] = None) -> HudSpanAttributes:
    """Extract and parse span attributes into typed model.
    
    This handles:
    - Detecting span type (MCP vs Agent)
    - Renaming verbose OpenTelemetry semantic conventions
    - Parsing JSON strings to MCP types
    """
    # Start with core attributes - map to TraceStep field names
    result_attrs = {
        "task_run_id": attrs.get("hud.task_run_id"),  # TraceStep expects task_run_id, not hud.task_run_id
        "job_id": attrs.get("hud.job_id"),            # TraceStep expects job_id, not hud.job_id
        "type": attrs.get("span.kind", "CLIENT"),     # TraceStep expects type, not span.kind
    }
    
    # Determine span type based on presence of agent or MCP attributes
    if "agent_request" in attrs or "agent_response" in attrs or (span_name and span_name.startswith("agent.")):
        result_attrs["category"] = "agent"  # TraceStep expects category field
        # Check for agent span attributes
        if "agent_request" in attrs:
            agent_req = attrs["agent_request"]
            if isinstance(agent_req, str):
                try:
                    agent_req = json.loads(agent_req)
                except json.JSONDecodeError:
                    pass
            result_attrs["agent_request"] = agent_req
        if "agent_response" in attrs:
            agent_resp = attrs["agent_response"]
            if isinstance(agent_resp, str):
                try:
                    agent_resp = json.loads(agent_resp)
                except json.JSONDecodeError:
                    pass
            result_attrs["agent_response"] = agent_resp
    else:
        result_attrs["category"] = "mcp"  # TraceStep expects category field
        # Add method_name and request_id only if present (MCP spans)
        if method_name:
            result_attrs["method_name"] = method_name
        if "semconv_ai.mcp.request_id" in attrs:
            result_attrs["request_id"] = attrs.get("semconv_ai.mcp.request_id")
    
    # Parse input/output
    input_str = attrs.get("semconv_ai.traceloop.entity.input")
    output_str = attrs.get("semconv_ai.traceloop.entity.output")
    
    # Try to parse as MCP types (only for MCP spans)
    if result_attrs["category"] == "mcp":
        if input_str:
            try:
                input_data = json.loads(input_str) if isinstance(input_str, str) else input_str
                if isinstance(input_data, dict):
                    result_attrs["mcp_request"] = ClientRequest.model_validate(input_data)  # TraceStep expects mcp_request
            except Exception as e:
                logger.debug(f"Failed to parse request as MCP type: {e}")
        
        if output_str:
            try:
                output_data = json.loads(output_str) if isinstance(output_str, str) else output_str
                if isinstance(output_data, dict):
                    # Check for error first
                    if "error" in output_data:
                        result_attrs["mcp_error"] = True
                    else:
                        result_attrs["mcp_result"] = ServerResult.model_validate(output_data)  # TraceStep expects mcp_result
                        # Check for isError in the result
                        if hasattr(result_attrs["mcp_result"].root, "isError") and result_attrs["mcp_result"].root.isError:
                            result_attrs["mcp_error"] = True
            except Exception as e:
                logger.debug(f"Failed to parse result as MCP type: {e}")
    
    # Don't include the verbose attributes or ones we've already processed
    exclude_keys = {
        "hud.task_run_id", "hud.job_id", "span.kind",
        "semconv_ai.mcp.method_name", "semconv_ai.mcp.request_id",
        "semconv_ai.traceloop.entity.input", "semconv_ai.traceloop.entity.output",
        "agent_request", "agent_response", "agent.provider", "agent.model"
    }
    
    # Add any extra attributes
    for key, value in attrs.items():
        if key not in exclude_keys:
            result_attrs[key] = value
    
    return HudSpanAttributes(**result_attrs)