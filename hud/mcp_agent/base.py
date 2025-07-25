"""Base MCP Agent implementation."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

import mcp.types as types
from mcp_use import MCPClient

logger = logging.getLogger(__name__)


class BaseMCPAgent(ABC):
    """
    Base class for MCP-enabled agents.

    This class provides the foundation for agents that interact with MCP servers,
    handling tool discovery and filtering while leaving provider-specific
    implementation details to subclasses.
    """

    def __init__(
        self,
        client: MCPClient | None = None,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        initial_screenshot: bool = False,
        max_screenshot_history: int = 3,
        append_tool_system_prompt: bool = True,
        custom_system_prompt: str | None = None,
    ) -> None:
        """
        Initialize the base MCP agent.

        Args:
            client: MCPClient instance for server connections
            allowed_tools: List of tool names to allow (None = all tools)
            disallowed_tools: List of tool names to disallow
            initial_screenshot: Whether to capture screenshot before first prompt
            max_screenshot_history: Maximum number of screenshots to keep in context
            append_tool_system_prompt: Whether to append available tools to system prompt
            custom_system_prompt: Custom system prompt to use
        """
        self.client = client
        self.allowed_tools = allowed_tools
        self.disallowed_tools = disallowed_tools or []
        self.initial_screenshot = initial_screenshot
        self.max_screenshot_history = max_screenshot_history
        self.append_tool_system_prompt = append_tool_system_prompt
        self.custom_system_prompt = custom_system_prompt
        self._available_tools: list[types.Tool] = []
        self._tool_map: dict[str, tuple[str, types.Tool]] = {}
        self._sessions: dict[str, Any] = {}

        if client is None:
            self.client = MCPClient()

    async def initialize(self) -> None:
        """Initialize the agent and discover available tools."""
        # Get existing sessions or create new ones
        sessions = self.client.get_all_active_sessions()

        if not sessions:
            logger.info("No active sessions found, creating new ones...")
            sessions = await self.client.create_all_sessions()

        self._sessions = sessions

        # Discover tools from all servers
        self._available_tools = []
        self._tool_map = {}

        for server_name, session in sessions.items():
            try:
                # Ensure session is initialized
                if not hasattr(session, "connector") or not hasattr(
                    session.connector, "client_session"
                ):
                    await session.initialize()

                # Get tools from the session
                tools_result = await session.connector.client_session.list_tools()
                for tool in tools_result.tools:
                    # Apply filtering
                    if self.allowed_tools and tool.name not in self.allowed_tools:
                        continue
                    if tool.name in self.disallowed_tools:
                        continue

                    self._available_tools.append(tool)
                    # Store tool with server reference for execution
                    self._tool_map[tool.name] = (server_name, tool)

            except Exception as e:
                logger.error("Failed to list tools from server %s: %s", server_name, e)

        logger.info(
            "Agent initialized with %s tools: %s",
            len(self._available_tools),
            [t.name for t in self._available_tools],
        )

    def get_available_tools(self) -> list[types.Tool]:
        """Get list of available MCP tools after filtering."""
        return self._available_tools

    def get_tool_map(self) -> dict[str, tuple[str, types.Tool]]:
        """Get mapping of tool names to (server_name, tool) tuples."""
        return self._tool_map

    def get_sessions(self) -> dict[str, Any]:
        """Get active MCP sessions."""
        return self._sessions

    def get_tools_by_server(self) -> dict[str, list[types.Tool]]:
        """Get tools grouped by server name."""
        tools_by_server = {}
        for server_name, tool in self._tool_map.values():
            if server_name not in tools_by_server:
                tools_by_server[server_name] = []
            tools_by_server[server_name].append(tool)
        return tools_by_server

    def get_tools_by_connector(self) -> dict[Any, list[types.Tool]]:
        """Get tools grouped by connector instance."""
        tools_by_connector = {}
        for server_name, tool in self._tool_map.values():
            session = self._sessions[server_name]
            connector = session.connector

            if connector not in tools_by_connector:
                tools_by_connector[connector] = []
            tools_by_connector[connector].append(tool)
        return tools_by_connector

    def get_system_prompt(self) -> str:
        """Generate system prompt with optional tool information."""
        base_prompt = self.custom_system_prompt or "You are a helpful assistant."

        if self.append_tool_system_prompt and self._available_tools:
            tool_descriptions = []
            for tool in self._available_tools:
                desc = f"- {tool.name}: {tool.description}"
                if tool.inputSchema:
                    desc += f" (parameters: {tool.inputSchema})"
                tool_descriptions.append(desc)

            tools_prompt = "\n\nYou have access to the following tools:\n" + "\n".join(
                tool_descriptions
            )
            return base_prompt + tools_prompt

        return base_prompt

    async def call_tool(self, tool_call: dict[str, Any]) -> types.CallToolResult:
        """
        Call a tool through the MCP client.

        Args:
            tool_call: Dict with 'name' and optional 'arguments' keys

        Returns:
            The raw MCP CallToolResult
        """
        tool_name = tool_call.get("name")
        if not tool_name:
            raise ValueError("Tool call must have a 'name' field")

        tool_args = tool_call.get("arguments", {})

        if tool_name not in self._tool_map:
            raise ValueError(f"Tool '{tool_name}' not found or not allowed")

        server_name, tool = self._tool_map[tool_name]
        session = self.client.get_session(server_name)

        logger.info(
            "Calling tool '%s' on server '%s' with args: %s",
            tool_name,
            server_name,
            tool_args,
        )
        result = await session.connector.client_session.call_tool(tool_name, tool_args)

        # Log result for debugging
        if result.isError:
            logger.error("Tool '%s' returned error: %s", tool_name, result.content)
        else:
            logger.debug("Tool '%s' completed successfully", tool_name)

        return result

    def has_computer_tools(self) -> bool:
        """Check if any computer control tools are available."""
        computer_tools = {"computer", "computer_anthropic", "computer_openai", "screenshot"}
        return any(tool.name in computer_tools for tool in self._available_tools)

    def get_tool_schemas(self) -> list[dict]:
        """Get tool schemas in a format suitable for the model."""
        schemas = []
        for tool in self._available_tools:
            schema = {
                "name": tool.name,
                "description": tool.description,
            }
            if tool.inputSchema:
                schema["parameters"] = tool.inputSchema
            schemas.append(schema)
        return schemas

    async def capture_screenshot(self) -> str | None:
        """Capture a screenshot using available tools."""
        if not self.has_computer_tools():
            return None

        # Try different screenshot tools
        for tool_name in ["computer", "screenshot", "computer_anthropic", "computer_openai"]:
            if tool_name in self._tool_map:
                try:
                    # Different tools have different APIs
                    if tool_name == "computer_openai":
                        tool_call = {"name": tool_name, "arguments": {"type": "screenshot"}}
                    else:
                        tool_call = {"name": tool_name, "arguments": {"action": "screenshot"}}

                    result = await self.call_tool(tool_call)

                    # Extract screenshot from result
                    for content in result.content:
                        if isinstance(content, types.ImageContent):
                            logger.info("Captured screenshot")
                            return content.data

                except Exception as e:
                    logger.warning("Failed to capture screenshot with %s: %s", tool_name, e)

        return None

    def process_tool_results(self, tool_results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Process tool results into a standardized format.

        Returns a dict with:
        - text: Combined text output from all tools
        - screenshot: Latest screenshot if any tool returned one
        - errors: List of any errors encountered
        - results: List of (tool_name, content_blocks) tuples for provider-specific formatting
        """
        text_parts = []
        latest_screenshot = None
        errors = []
        results = []

        for tool_result in tool_results:
            tool_name = tool_result["tool_name"]
            content_blocks = []

            if tool_result.get("error"):
                error_msg = f"{tool_name}: {tool_result.get('error_message', 'Unknown error')}"
                errors.append(error_msg)
                text_parts.append(f"Error - {error_msg}")
                content_blocks.append(
                    {
                        "type": "error",
                        "text": tool_result.get("error_message", "Unknown error"),
                    }
                )
            else:
                result = tool_result["result"]
                if result.isError:
                    # Extract error from content
                    error_text = "Tool execution failed"
                    for content in result.content:
                        if isinstance(content, types.TextContent):
                            error_text = content.text
                            break
                    error_msg = f"{tool_name}: {error_text}"
                    errors.append(error_msg)
                    text_parts.append(f"Error - {error_msg}")
                    content_blocks.append(
                        {
                            "type": "error",
                            "text": error_text,
                        }
                    )
                else:
                    # Process success content
                    tool_output = []
                    for content in result.content:
                        if isinstance(content, types.TextContent):
                            tool_output.append(content.text)
                            content_blocks.append(
                                {
                                    "type": "text",
                                    "text": content.text,
                                }
                            )
                        elif isinstance(content, types.ImageContent):
                            # Keep the latest screenshot
                            latest_screenshot = content.data
                            content_blocks.append(
                                {
                                    "type": "image",
                                    "data": content.data,
                                }
                            )

                    if tool_output:
                        text_parts.append(f"{tool_name}: " + " ".join(tool_output))

            results.append((tool_name, content_blocks))

        return {
            "text": "\n".join(text_parts) if text_parts else "No output from tools",
            "screenshot": latest_screenshot,
            "errors": errors,
            "results": results,  # List of (tool_name, content_blocks) for provider-specific use
        }

    async def run(
        self, prompt: str, max_steps: int = 10, conversation_mode: bool = False
    ) -> str:
        """
        Run the agent with the given prompt.

        Args:
            prompt: The task to complete
            max_steps: Maximum number of steps
            conversation_mode: If True, continue even when model returns text without tool calls

        Returns:
            The final response or result
        """
        try:
            if not self._available_tools:
                await self.initialize()

            latest_screenshot = None
            if self.initial_screenshot:
                latest_screenshot = await self.capture_screenshot()

            messages = await self.create_initial_messages(prompt, latest_screenshot)

            step = 0
            while step < max_steps:
                step += 1
                logger.info("step %s/%s", step, max_steps)

                try:
                    response = await self.get_model_response(messages, step)

                    # Check if we should stop
                    if response.get("done", False) and not conversation_mode:
                        return response.get("content", "Task completed")

                    tool_calls = response.get("tool_calls", [])
                    if not tool_calls:
                        if conversation_mode:
                            # In conversation mode, if model responds without tools,
                            # show the response and get user input
                            model_response = response.get("content", "")
                            if model_response:
                                print(f"\n🤖 Agent: {model_response}")  # noqa: T201
                                user_input = input("\n👤 You: ").strip()
                                if user_input.lower() in ["exit", "quit", "bye"]:
                                    return "Conversation ended by user."
                                # Add user's response to the conversation
                                # This needs to be handled by subclass-specific format
                                user_message = await self.create_user_message(user_input)
                                messages.append(user_message)
                                continue
                            else:
                                # No content and no tools - something went wrong
                                return "No response generated"
                        else:
                            # In task mode, no tool calls means we're done
                            return response.get("content", "No response generated")

                    # Execute tool calls
                    tool_results = []
                    for tool_call in tool_calls:
                        if not tool_call.get("name"):
                            continue
                        try:
                            result = await self.call_tool(tool_call)
                            tool_results.append(
                                {
                                    "tool_name": tool_call["name"],
                                    "result": result,
                                    "error": False,
                                }
                            )
                        except Exception as e:
                            logger.error("Tool execution failed: %s", e)
                            tool_results.append(
                                {
                                    "tool_name": tool_call["name"],
                                    "error": True,
                                    "error_message": str(e),
                                }
                            )

                    # Process results
                    processed_results = self.process_tool_results(tool_results)

                    # Update screenshot if we got a new one
                    if processed_results["screenshot"]:
                        latest_screenshot = processed_results["screenshot"]

                    # Format tool results for the model
                    tool_messages = await self.format_tool_results(
                        processed_results,
                        response.get("tool_calls", []),
                    )
                    messages.extend(tool_messages)

                except Exception as e:
                    logger.error("Model call failed: %s", e)
                    return f"Error: {e}"

            return f"Maximum steps ({max_steps}) reached without completion"

        except KeyboardInterrupt:
            logger.info("Agent execution interrupted by user")
            return "Execution interrupted by user (Ctrl+C)"
        except asyncio.CancelledError:
            logger.info("Agent execution cancelled")
            return "Execution cancelled"

    @abstractmethod
    async def create_initial_messages(self, prompt: str, screenshot: str | None) -> list[Any]:
        """
        Create initial messages for the conversation.

        Args:
            prompt: The user's prompt
            screenshot: Optional initial screenshot

        Returns:
            List of messages in provider-specific format
        """

    @abstractmethod
    async def get_model_response(self, messages: list[Any], step: int) -> dict[str, Any]:
        """
        Get response from the model including any tool calls.

        Args:
            messages: List of messages in provider-specific format
            step: Current step number

        Returns:
            Dict with 'content', 'tool_calls', and 'done' keys
        """

    @abstractmethod
    async def format_tool_results(
        self, processed_results: dict[str, Any], tool_calls: list[dict[str, Any]]
    ) -> list[Any]:
        """
        Format tool results into messages for the model.

        Args:
            processed_results: Processed tool results from process_tool_results
            tool_calls: Original tool calls from the model

        Returns:
            List of formatted messages to append to conversation
        """
        raise NotImplementedError

    async def create_user_message(self, text: str) -> Any:
        """
        Create a user message in the format expected by the model.

        Default implementation for text-only messages.
        Subclasses can override for specific formats.

        Args:
            text: User's text input

        Returns:
            Formatted user message
        """
        return {"role": "user", "content": text}
