"""
Patches for MCP client library compatibility.

This module contains patches for the MCP client library to improve compatibility
and error handling with various MCP server implementations.
"""

from __future__ import annotations

import logging
from typing import Any

from jsonschema import ValidationError, validate

logger = logging.getLogger(__name__)


def patch_mcp_client_session() -> None:
    """
    Patch the MCP ClientSession to make output schema validation a warning instead of an error.

    This addresses the issue where tools with output schemas that don't return structured content
    cause runtime errors. This is particularly important for cloud environments where the server
    implementation might advertise output schemas but return text content instead.

    In deployment, these should be fixed by the server implementation. However, for debugging
    purposes, we allow this to continue execution. Instead of failing completely, we log a warning.
    """
    try:
        from mcp.client.session import ClientSession
        from mcp.types import CallToolResult

        # Store the original method
        original_call_tool = ClientSession.call_tool

        async def patched_call_tool(
            self: ClientSession,
            name: str,
            arguments: dict[str, Any] | None = None,
            read_timeout_seconds: Any | None = None,
        ) -> CallToolResult:
            """Patched call_tool that converts output schema errors to warnings."""

            # First, get the tool info to check if it has an output schema
            tool_info = None
            if hasattr(self, "_tools") and name in self._tools:
                tool_info = self._tools[name]

            # Call the original method to get the result
            result = await original_call_tool.__func__(self, name, arguments, read_timeout_seconds)

            # If the tool has an output schema but didn't return structured content,
            # log a warning instead of raising an error
            if tool_info and tool_info.outputSchema and result.structuredContent is None:
                logger.warning(
                    "Tool %s has an output schema but did not return structured content. "
                    "This is being allowed for testing, but the tool should be updated to either:\n"
                    "1. Return structured content matching the output schema, or\n"
                    "2. Remove the output schema from the tool definition.\n"
                    "Content received: %s",
                    name,
                    result.content
                )

            # If structured content is provided, still validate it
            if result.structuredContent is not None and tool_info and tool_info.outputSchema:
                try:
                    validate(result.structuredContent, tool_info.outputSchema)
                except ValidationError as e:
                    logger.error(
                        "Tool %s returned structured content that doesn't match its schema: %s",
                        name,
                        e
                    )
                    # Still return the result, but log the validation error

            return result

        # Replace the method
        ClientSession.call_tool = patched_call_tool
        logger.debug(
            "Successfully patched MCP ClientSession.call_tool for lenient output schema validation"
        )

    except ImportError as e:
        logger.warning("Could not import MCP client for patching: %s", e)
    except Exception as e:
        logger.error("Error applying MCP client patch: %s", e)
