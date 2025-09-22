"""LiteLLM MCP Agent implementation.

This agent implements the minimal 4-method surface described in the docs,
and behaves similarly to GenericOpenAIChatAgent — but routes calls through
LiteLLM so you can use any provider (OpenAI, Bedrock via LiteLLM, etc.).
"""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar, cast

import litellm
import mcp.types as types

import hud
from hud.types import AgentResponse, MCPToolCall, MCPToolResult
from hud.utils.hud_console import HUDConsole

from .base import MCPAgent

logger = logging.getLogger(__name__)

# Prefer LiteLLM's built-in MCP -> OpenAI tool transformer (handles Bedrock nuances)
try:
    from litellm.experimental_mcp_client.tools import (
        transform_mcp_tool_to_openai_tool,
    )
except Exception:  # pragma: no cover - optional dependency
    transform_mcp_tool_to_openai_tool = None  # type: ignore


class LiteAgent(MCPAgent):
    """
    MCP-enabled agent that routes LLM calls through LiteLLM.

    - Formatting is OpenAI chat-completions style (text + image_url blocks)
    - Tools use OpenAI "function" tool schema generated via LiteLLM
    - Tool calls are executed by MCPAgent.call_tools()
    """

    metadata: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        *,
        model_name: str = "gpt-4o-mini",
        completion_kwargs: dict[str, Any] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        """
        Args:
            model_name: Any LiteLLM-supported chat model (OpenAI, Azure, Bedrock, etc.)
            completion_kwargs: Extra kwargs forwarded to litellm.acompletion(...)
            **agent_kwargs: Base MCPAgent settings (mcp_client, allowed_tools, etc.)
        """
        super().__init__(**agent_kwargs)
        self.model_name = model_name
        self.completion_kwargs: dict[str, Any] = completion_kwargs or {}
        self.hud_console = HUDConsole(logger=logger)

    # -------------------------------------------------------------------------
    # 1) System messages
    # -------------------------------------------------------------------------
    async def get_system_messages(self) -> list[Any]:
        """Return a single system message for the chat.completions-style format."""
        return [{"role": "system", "content": self.system_prompt}]

    # -------------------------------------------------------------------------
    # 2) Get response (LLM turn)
    # -------------------------------------------------------------------------
    @hud.instrument(
        span_type="agent",
        record_args=False,
        record_result=True,
    )
    async def get_response(self, messages: list[Any]) -> AgentResponse:
        """Send messages to LiteLLM (chat.completions) and convert result."""
        tools = cast("list[dict[str, Any]]", self._openai_function_tools())

        # Avoid clobbering protected keys with completion_kwargs
        protected = {"model", "messages", "tools"}
        extra = {k: v for k, v in self.completion_kwargs.items() if k not in protected}

        try:
            resp = cast(
                Any,
                await litellm.acompletion(
                    model=self.model_name,
                    messages=messages,
                    tools=tools if tools else None,
                    **extra,
                ),
            )
        except Exception as e:
            self.hud_console.warning_log(f"LiteLLM error: {e}")
            return AgentResponse(
                content=f"Error getting response {e}",
                tool_calls=[],
                done=True,
                isError=True,
                raw=None,
            )

        choice = resp.choices[0]
        msg = choice.message

        # Record assistant message (content + tool_calls) to conversation
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if msg.content:
            assistant_msg["content"] = msg.content
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        # Convert tool calls into MCPToolCall objects
        tool_calls: list[MCPToolCall] = []
        if msg.tool_calls:
            tool_calls.extend(self._oai_to_mcp(tc) for tc in msg.tool_calls)

        # Only force-stop on token limit; otherwise MCPAgent stops when no tool calls
        done = choice.finish_reason == "length"
        if done:
            self.hud_console.info_log(f"Done decision: finish_reason={choice.finish_reason}")

        return AgentResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            done=done,
            raw=resp,
        )

    # -------------------------------------------------------------------------
    # 3) Initial blocks -> LLM message format
    # -------------------------------------------------------------------------
    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[Any]:
        """Convert MCP content blocks to OpenAI chat 'user' message with mixed content."""
        content: list[dict[str, Any]] = []
        for block in blocks:
            if isinstance(block, types.TextContent):
                content.append({"type": "text", "text": block.text})
            elif isinstance(block, types.ImageContent):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{block.mimeType};base64,{block.data}"},
                    }
                )
        return [{"role": "user", "content": content}]

    # -------------------------------------------------------------------------
    # 4) Tool results -> LLM message format
    # -------------------------------------------------------------------------
    async def format_tool_results(
        self,
        tool_calls: list[MCPToolCall],
        tool_results: list[MCPToolResult],
    ) -> list[Any]:
        """
        Render MCP tool results as OpenAI chat messages.

        - Emit a 'tool' message for each tool result (string content only)
        - If a screenshot/image is present, add a trailing 'user' message
          with an image_url so the model can see the new screen state.
        """
        rendered: list[dict[str, Any]] = []
        image_parts: list[dict[str, Any]] = []

        for call, res in zip(tool_calls, tool_results, strict=False):
            text_parts: list[str] = []

            items = res.content or []
            if not items and res.structuredContent:
                items = [res.structuredContent.get("result", res.content)]

            for item in items:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image":
                        mime = item.get("mimeType", "image/png")
                        data = item.get("data", "")
                        image_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{data}",
                                },
                            }
                        )
                elif isinstance(item, types.TextContent):
                    text_parts.append(item.text)
                elif isinstance(item, types.ImageContent):
                    image_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{item.mimeType};base64,{item.data}",
                            },
                        }
                    )

            text_content = (
                "".join(text_parts)
                if text_parts
                else ("Tool execution failed" if res.isError else "Tool executed successfully")
            )

            rendered.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": text_content,
                }
            )

        if image_parts:
            rendered.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Tool returned the following:"},
                        image_parts[-1],  # latest image is usually most relevant
                    ],
                }
            )

        return rendered

    # -------------------------------------------------------------------------
    # Helpers (kept minimal)
    # -------------------------------------------------------------------------
    @staticmethod
    def _oai_to_mcp(tool_call: Any) -> MCPToolCall:  # type: ignore[valid-type]
        """Convert an OpenAI tool_call to MCPToolCall."""
        args = {}
        try:
            args = json.loads(tool_call.function.arguments or "{}")
            if isinstance(args, list):
                args = args[0] if args else {}
            if not isinstance(args, dict):
                args = {}
        except json.JSONDecodeError:
            args = {}
        return MCPToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=args,
        )

    def _openai_function_tools(self) -> list[dict]:
        """
        Transform MCP tools to OpenAI 'function' tool specs.

        Uses LiteLLM's experimental transformer when available (preferred),
        which normalizes schemas for strict providers like Bedrock. Falls
        back to a simple dict if the transformer isn't importable.
        """
        tools: list[dict] = []
        mcp_tools = self.get_available_tools()  # Already filtered to exclude lifecycle tools

        for mcp_tool in mcp_tools:
            if transform_mcp_tool_to_openai_tool is not None:
                # Let LiteLLM return a ChatCompletionToolParam (or compatible)
                tools.append(transform_mcp_tool_to_openai_tool(mcp_tool))  # type: ignore[arg-type]
            else:  # Fallback: minimal OpenAI tool dict
                params = mcp_tool.inputSchema or {"type": "object", "properties": {}}
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": mcp_tool.name,
                            "description": mcp_tool.description or "",
                            "parameters": params,
                            "strict": False,
                        },
                    }
                )

        return tools
