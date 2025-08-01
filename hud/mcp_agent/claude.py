"""Claude MCP Agent implementation."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, cast

from anthropic import AsyncAnthropic, BadRequestError

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaCacheControlEphemeralParam,
        BetaImageBlockParam,
        BetaMessageParam,
        BetaTextBlockParam,
        BetaToolResultBlockParam,
    )

from hud.settings import settings

from .base import BaseMCPAgent

logger = logging.getLogger(__name__)


def base64_to_content_block(base64: str) -> BetaImageBlockParam:
    """Convert base64 image to Claude content block."""
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64},
    }


def text_to_content_block(text: str) -> BetaTextBlockParam:
    """Convert text to Claude content block."""
    return {"type": "text", "text": text}


def tool_use_content_block(
    tool_use_id: str, content: list[BetaTextBlockParam | BetaImageBlockParam]
) -> BetaToolResultBlockParam:
    """Create tool result content block."""
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


class ClaudeMCPAgent(BaseMCPAgent):
    """
    Claude agent that uses MCP servers for tool execution.

    This agent uses Claude's native tool calling capabilities but executes
    tools through MCP servers instead of direct implementation.
    """

    def __init__(
        self,
        model_client: AsyncAnthropic | None = None,
        model: str = "claude-3-7-sonnet-20250219",
        max_tokens: int = 4096,
        display_width_px: int = 1400,
        display_height_px: int = 850,
        use_computer_beta: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Claude MCP agent.

        Args:
            client: AsyncAnthropic client (created if not provided)
            model: Claude model to use
            max_tokens: Maximum tokens for response
            display_width_px: Display width for computer use tools
            display_height_px: Display height for computer use tools
            use_computer_beta: Whether to use computer-use beta features
            **kwargs: Additional arguments passed to BaseMCPAgent
        """
        super().__init__(**kwargs)

        # Initialize client if not provided
        if model_client is None:
            api_key = settings.anthropic_api_key
            if not api_key:
                raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY.")
            model_client = AsyncAnthropic(api_key=api_key)

        self.anthropic_client = model_client
        self.model = model
        self.max_tokens = max_tokens
        self.display_width_px = display_width_px
        self.display_height_px = display_height_px
        self.use_computer_beta = use_computer_beta

        # Track mapping from Claude tool names to MCP tool names
        self._claude_to_mcp_tool_map: dict[str, str] = {}

    async def initialize(self) -> None:
        """Initialize the agent and build tool mappings."""
        await super().initialize()
        # Build tool mappings after tools are discovered
        self._convert_tools_for_claude()

    async def create_initial_messages(
        self, prompt: str, screenshot: str | None
    ) -> list[BetaMessageParam]:
        """Create initial messages for Claude."""
        user_content: list[BetaImageBlockParam | BetaTextBlockParam] = []

        # Add prompt text
        user_content.append(text_to_content_block(prompt))

        # Add screenshot if available
        if screenshot:
            user_content.append(base64_to_content_block(screenshot))

        # Return initial user message
        return [
            cast(
                "BetaMessageParam",
                {
                    "role": "user",
                    "content": user_content,
                },
            )
        ]

    async def get_model_response(
        self, messages: list[BetaMessageParam], step: int
    ) -> dict[str, Any]:
        """Get response from Claude including any tool calls."""
        # Get Claude tools
        claude_tools = self._convert_tools_for_claude()

        # Make API call with retry for prompt length
        current_messages = messages.copy()

        while True:
            messages_cached = self._add_prompt_caching(current_messages)

            # Build create kwargs
            create_kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": self.get_system_prompt(),
                "messages": messages_cached,
                "tools": claude_tools,
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            }

            # Add beta features if using computer tools
            if self.use_computer_beta and any(
                t.get("type") == "computer_20250124" for t in claude_tools
            ):
                create_kwargs["betas"] = ["computer-use-2025-01-24"]

            try:
                response = await self.anthropic_client.beta.messages.create(**create_kwargs)
                break
            except BadRequestError as e:
                if e.message.startswith("prompt is too long"):
                    logger.warning("Prompt too long, truncating message history")
                    # Keep first message and last 20 messages
                    if len(current_messages) > 21:
                        current_messages = [current_messages[0]] + current_messages[-20:]
                    else:
                        raise
                else:
                    raise

        # Add assistant response to messages (for next step)
        messages.append(
            cast(
                "BetaMessageParam",
                {
                    "role": "assistant",
                    "content": response.content,
                },
            )
        )

        # Process response
        result = {
            "content": "",
            "tool_calls": [],
            "done": True,
            "raw_response": response.model_dump(),  # For debugging
        }

        # Extract text content and reasoning
        text_content = ""
        thinking_content = ""

        for block in response.content:
            if block.type == "tool_use":
                # Map Claude tool name back to MCP tool name
                mcp_tool_name = self._claude_to_mcp_tool_map.get(block.name, block.name)

                # Include the tool_use_id in the tool call for later reference
                result["tool_calls"].append(
                    {
                        "name": mcp_tool_name,  # Use MCP tool name for execution
                        "arguments": block.input,
                        "tool_use_id": block.id,  # Claude-specific metadata
                        "claude_name": block.name,  # Keep original Claude name for reference
                    }
                )
                result["done"] = False
            elif block.type == "text":
                text_content += block.text
            elif hasattr(block, "type") and block.type == "thinking":
                thinking_content += f"Thinking: {block.thinking}\n"

        # Combine text and thinking for final content
        if thinking_content:
            result["content"] = thinking_content + text_content
        else:
            result["content"] = text_content

        return result

    async def format_tool_results(
        self, processed_results: dict[str, Any], tool_calls: list[dict]
    ) -> list[BetaMessageParam]:
        """Format tool results into Claude messages."""
        # Build a mapping of tool_name to tool_use_id from the original calls
        tool_id_map = {}
        for tool_call in tool_calls:
            if "tool_use_id" in tool_call:
                tool_id_map[tool_call["name"]] = tool_call["tool_use_id"]

        # Process each tool result
        user_content = []

        for tool_name, content_blocks in processed_results["results"]:
            # Get the tool_use_id for this tool
            tool_use_id = tool_id_map.get(tool_name)
            if not tool_use_id:
                logger.warning("No tool_use_id found for %s", tool_name)
                continue

            # Convert content blocks to Claude format
            claude_blocks = []
            for block in content_blocks:
                if block["type"] == "text":
                    claude_blocks.append(text_to_content_block(block["text"]))
                elif block["type"] == "error":
                    claude_blocks.append(text_to_content_block(f"Error: {block['text']}"))
                elif block["type"] == "image":
                    claude_blocks.append(base64_to_content_block(block["data"]))

            # Add tool result
            user_content.append(tool_use_content_block(tool_use_id, claude_blocks))

        # Return as a user message containing all tool results
        return [
            cast(
                "BetaMessageParam",
                {
                    "role": "user",
                    "content": user_content,
                },
            )
        ]

    async def create_user_message(self, text: str) -> BetaMessageParam:
        """Create a user message in Claude's format."""
        return cast("BetaMessageParam", {"role": "user", "content": text})

    def _convert_tools_for_claude(self) -> list[dict]:
        """Convert MCP tools to Claude tool format."""
        claude_tools = []
        self._claude_to_mcp_tool_map = {}  # Reset mapping

        for tool in self._available_tools:
            # Special handling for computer use tools
            if tool.name in ["computer", "computer_anthropic", "anthropic_computer"]:
                # Use Claude's native computer use format with configurable dimensions
                claude_tool = {
                    "type": "computer_20250124",
                    "name": "computer",
                    "display_width_px": self.display_width_px,
                    "display_height_px": self.display_height_px,
                }
                # Map Claude's "computer" back to the actual MCP tool name
                self._claude_to_mcp_tool_map["computer"] = tool.name
            else:
                # Convert regular tools
                claude_tool = {
                    "name": tool.name,
                    "description": tool.description or f"Execute {tool.name}",
                    "input_schema": tool.inputSchema
                    or {
                        "type": "object",
                        "properties": {},
                    },
                }
                # Direct mapping for non-computer tools
                self._claude_to_mcp_tool_map[tool.name] = tool.name

            claude_tools.append(claude_tool)

        return claude_tools

    def _add_prompt_caching(self, messages: list[BetaMessageParam]) -> list[BetaMessageParam]:
        """Add prompt caching to messages."""
        messages_cached = copy.deepcopy(messages)

        # Mark last user message with cache control
        if messages_cached and messages_cached[-1].get("role") == "user":
            last_content = messages_cached[-1]["content"]
            if isinstance(last_content, list):
                for block in last_content:
                    if block.get("type") not in ["thinking", "redacted_thinking"]:
                        cache_control: BetaCacheControlEphemeralParam = {"type": "ephemeral"}
                        block["cache_control"] = cache_control  # type: ignore[reportGeneralTypeIssues]

        return messages_cached
