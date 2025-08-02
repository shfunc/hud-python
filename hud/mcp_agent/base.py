"""Base MCP Agent implementation."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import mcp.types as types
from mcp_use import MCPClient

if TYPE_CHECKING:
    from hud.task import Task

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
        lifecycle_tools: dict[str, str] | None = None,
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
            lifecycle_tools: Dict mapping lifecycle phases to tool names. Default:
                {
                    "setup": "setup",      # Setup phase tool
                    "evaluate": "evaluate"  # Evaluation phase tool
                }
        """
        self.client = client
        self.allowed_tools = allowed_tools
        self.disallowed_tools = disallowed_tools or []
        self.initial_screenshot = initial_screenshot
        self.max_screenshot_history = max_screenshot_history
        self.append_tool_system_prompt = append_tool_system_prompt
        self.custom_system_prompt = custom_system_prompt

        # Default lifecycle tool mapping
        default_lifecycle = {"setup": "setup", "evaluate": "evaluate"}
        self.lifecycle_tools = {**default_lifecycle, **(lifecycle_tools or {})}

        self._available_tools: list[types.Tool] = []
        self._tool_map: dict[str, tuple[str, types.Tool]] = {}
        self._sessions: dict[str, Any] = {}

        if client is None:
            self.client = MCPClient()

    async def initialize(self) -> None:
        """Initialize the agent and discover available tools."""
        # Get existing sessions or create new ones
        if self.client is None:
            raise ValueError("Client is not initialized")

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

                if session.connector.client_session is None:
                    raise ValueError("Client session is not initialized")

                tools_result = await session.connector.client_session.list_tools()

                # Log all tools before filtering
                logger.info(
                    "Tools from '%s' (pre-filter): %s",
                    server_name,
                    [tool.name for tool in tools_result.tools],
                )

                for tool in tools_result.tools:
                    # Always include lifecycle tools for framework use
                    is_lifecycle_tool = tool.name in self.lifecycle_tools.values()

                    # Apply filtering (but always allow lifecycle tools)
                    if not is_lifecycle_tool:
                        if self.allowed_tools and tool.name not in self.allowed_tools:
                            continue
                        if tool.name in self.disallowed_tools:
                            continue

                    self._available_tools.append(tool)
                    # Store tool with server reference for execution
                    self._tool_map[tool.name] = (server_name, tool)

            except Exception as e:
                logger.error("Failed to list tools from server %s: %s", server_name, e)

        # Separate lifecycle tools from regular tools for clearer logging
        lifecycle_tool_names = list(self.lifecycle_tools.values())
        regular_tools = [
            t.name for t in self._available_tools if t.name not in lifecycle_tool_names
        ]
        lifecycle_tools_found = [
            t.name for t in self._available_tools if t.name in lifecycle_tool_names
        ]

        logger.info(
            "Agent initialized with %s tools (%s regular, %s lifecycle)",
            len(self._available_tools),
            len(regular_tools),
            len(lifecycle_tools_found),
        )
        if regular_tools:
            logger.info("Regular tools: %s", regular_tools)
        if lifecycle_tools_found:
            logger.info("Lifecycle tools: %s", lifecycle_tools_found)

    def get_available_tools(self) -> list[types.Tool]:
        """Get list of available MCP tools for LLM use (excludes lifecycle tools)."""
        lifecycle_tool_names = list(self.lifecycle_tools.values())
        return [tool for tool in self._available_tools if tool.name not in lifecycle_tool_names]

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

        if self.client is None:
            raise ValueError("Client is not initialized")

        server_name, tool = self._tool_map[tool_name]
        session = self.client.get_session(server_name)

        logger.info(
            "Calling tool '%s' on server '%s' with args: %s",
            tool_name,
            server_name,
            tool_args,
        )
        if session.connector.client_session is None:
            raise ValueError("Client session is not initialized")

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
            # Filter out lifecycle tools from LLM conversation
            if tool.name in self.lifecycle_tools.values():
                continue

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
        for tool_name in [
            "computer",
            "screenshot",
            "computer_anthropic",
            "computer_openai",
            "anthropic_computer",
            "openai_computer",
        ]:
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
        self, prompt_or_task: str | Task, max_steps: int = 10, conversation_mode: bool = False
    ) -> dict[str, Any]:
        """
        Run the agent with the given prompt or task.

        Args:
            prompt_or_task: Either a string prompt for simple execution or a Task object
            max_steps: Maximum number of steps
            conversation_mode: If True, continue even when model returns text without tool calls

        Returns:
            For string prompts: The final response string
            For Task objects: Evaluation result dict with 'reward', 'done', 'info' keys
        """
        # Import here to avoid circular imports
        from hud.task import Task

        if not self._available_tools:
            await self.initialize()

        # Handle Task objects with full lifecycle
        if isinstance(prompt_or_task, Task):
            return await self._run_task(prompt_or_task, max_steps)

        # Handle simple string prompts (existing behavior)
        elif isinstance(prompt_or_task, str):
            return await self._run_prompt(prompt_or_task, max_steps, conversation_mode)

        else:
            raise TypeError(f"prompt_or_task must be str or Task, got {type(prompt_or_task)}")

    async def _run_task(self, task: Task, max_steps: int = 10) -> dict[str, Any]:
        """
        Execute a task with setup and evaluate phases.

        Args:
            task: Task object with prompt, setup, and evaluate configs
            max_steps: Maximum steps for task execution

        Returns:
            Evaluation result dict with 'reward', 'done', 'info' keys
        """
        try:
            # Setup phase
            if task.setup is not None:
                setup_tool = self.lifecycle_tools.get("setup", "setup")
                await self._call_tool_safe(setup_tool, task.setup)

            # Execute the task prompt
            await self._run_prompt(task.prompt, max_steps, conversation_mode=False)

            # Evaluate phase
            if task.evaluate is not None:
                evaluate_tool = self.lifecycle_tools.get("evaluate", "evaluate")
                eval_result = await self._call_tool_safe(evaluate_tool, task.evaluate)

                # Return evaluation result if it's properly formatted
                if (
                    isinstance(eval_result, dict)
                    and "reward" in eval_result
                    and "done" in eval_result
                ):
                    return eval_result
                elif isinstance(eval_result, dict) and "grade" in eval_result:
                    return {
                        "reward": eval_result.get("grade", 0.0),
                        "done": True,
                        "info": {
                            "error": eval_result.get("error"),
                            "logs": eval_result.get("logs", ""),
                            "original_result": eval_result,
                        },
                    }
                else:
                    # Fallback for invalid evaluation format
                    return {
                        "reward": 0.0,
                        "done": True,
                        "info": {"error": "Invalid evaluation result", "eval_result": eval_result},
                    }
            else:
                # No evaluation - assume success
                return {
                    "reward": 0.0,
                    "done": True,
                    "info": {"message": "Task completed (no evaluation specified)"},
                }

        except Exception as e:
            return {"reward": 0.0, "done": True, "info": {"error": str(e)}}

    async def _call_tool_safe(self, tool_name: str, arguments: Any) -> Any:
        """
        Safely call a tool and return its result.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool (config from task)

        Returns:
            Tool result or None if tool not available/failed
        """
        try:
            if tool_name in self._tool_map:
                tool_call = {"name": tool_name, "arguments": arguments}
                result = await self.call_tool(tool_call)

                if result.isError:
                    logger.error("Tool %s returned error: %s", tool_name, result.content)
                    return {"error": result.content}
                else:
                    # Extract content from MCP result
                    if hasattr(result, "content") and result.content:
                        if len(result.content) == 1:
                            content_item = result.content[0]
                            # Check if content_item is a text type
                            if hasattr(content_item, "text") and hasattr(content_item, "type"):
                                if getattr(content_item, "type", None) == "text":
                                    # Try to parse as JSON if it looks like structured data
                                    text = content_item.text  # type: ignore[reportAttributeAccessIssue]
                                    if text.strip().startswith("{") and text.strip().endswith("}"):
                                        try:
                                            import json

                                            return json.loads(text)
                                        except json.JSONDecodeError:
                                            return text
                                    return text
                            else:
                                return content_item
                        else:
                            return result.content
                    return result
            else:
                logger.warning("Tool %s not available", tool_name)
                return None
        except Exception as e:
            logger.error("Failed to call tool %s: %s", tool_name, e)
            return {"error": str(e)}

    async def _run_prompt(
        self,
        prompt: str,
        max_steps: int = 10,
        conversation_mode: bool = False,
    ) -> dict[str, Any]:
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

                    # Log the model's response
                    logger.info("Model response - Content: %s", response.get("content", ""))
                    logger.info(
                        "Model response - Tool calls: %s",
                        [tc.get("name") for tc in response.get("tool_calls", [])],
                    )
                    logger.info("Model response - Done: %s", response.get("done", False))

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
                                    return {
                                        "done": True,
                                        "reward": 0.0,
                                        "info": {"message": "Conversation ended by user."},
                                    }
                                # Add user's response to the conversation
                                # This needs to be handled by subclass-specific format
                                user_message = await self.create_user_message(user_input)
                                messages.append(user_message)
                                continue
                            else:
                                # No content and no tools - something went wrong
                                return {
                                    "done": False,
                                    "reward": 0.0,
                                    "info": {"message": "No response generated"},
                                }
                        else:
                            # In task mode, no tool calls means we're done
                            logger.info("In task mode with no tool calls - stopping execution")
                            logger.info(
                                "Final message: %s",
                                response.get("content", "No response generated"),
                            )
                            return {
                                "done": True,
                                "reward": 0.0,
                                "info": {
                                    "message": response.get("content", "No response generated"),
                                },
                            }

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
                    return {"done": False, "reward": 0.0, "info": {"message": f"Error: {e}"}}

            return {"done": True, "reward": 0.0, "info": {"message": "Task completed"}}

        except KeyboardInterrupt:
            logger.info("Agent execution interrupted by user")
            return {
                "done": False,
                "reward": 0.0,
                "info": {"message": "Execution interrupted by user (Ctrl+C)"},
            }
        except asyncio.CancelledError:
            logger.info("Agent execution cancelled")
            return {"done": False, "reward": 0.0, "info": {"message": "Execution cancelled"}}

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
