"""Base MCP Agent implementation."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import mcp.types as types
from mcp.types import CallToolRequestParams as MCPToolCall
from mcp.types import CallToolResult as MCPToolResult
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from hud.datasets import TaskConfig

    from .client import MCPClient


class ModelResponse(BaseModel):
    """Response from get_model_response method."""

    content: str | None = Field(default=None)
    tool_calls: list[MCPToolCall] = Field(default_factory=list)
    done: bool = Field(default=False)


class AgentResult(BaseModel):
    """Unified result from agent execution (task or prompt).

    Fields:
    - done: Whether execution is complete
    - reward: Numeric reward (mainly for task evaluation)
    - info: Additional metadata dict
    - content: Final text content from the agent
    - error: Error message if execution failed
    - messages: Full conversation history (populated in prompt mode)
    """

    done: bool = Field(default=True)
    reward: float = Field(default=0.0)
    info: dict[str, Any] = Field(default_factory=dict)
    content: str | None = Field(default=None)
    error: str | None = Field(default=None)
    messages: list[Any] = Field(default_factory=list)  # Full conversation history


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
        mcp_client: MCPClient | None = None,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        initial_screenshot: bool = False,
        max_screenshot_history: int = 3,
        append_tool_system_prompt: bool = True,
        custom_system_prompt: str | None = None,
        lifecycle_tools: list[str] | None = None,
    ) -> None:
        """
        Initialize the base MCP agent.

        Args:
            mcp_client: MCPClient instance for server connections
            allowed_tools: List of tool names to allow (None = all tools)
            disallowed_tools: List of tool names to disallow
            initial_screenshot: Whether to capture screenshot before first prompt
            max_screenshot_history: Maximum number of screenshots to keep in context
            append_tool_system_prompt: Whether to append available tools to system prompt
            custom_system_prompt: Custom system prompt to use
            lifecycle_tools: List of tool names to use for lifecycle tools
        """
        if not mcp_client:
            raise ValueError(
                "MCPClient is required. Please provide a configured MCPClient instance."
            )
        self.mcp_client = mcp_client
        self.allowed_tools = allowed_tools
        self.disallowed_tools = disallowed_tools or []
        self.initial_screenshot = initial_screenshot
        self.max_screenshot_history = max_screenshot_history
        self.append_tool_system_prompt = append_tool_system_prompt
        self.custom_system_prompt = custom_system_prompt

        self.lifecycle_tools = lifecycle_tools or []

        self.model_name = "test-agent"

        # Initialize these here so methods can be called before initialize()
        self._available_tools: list[types.Tool] = []
        self._tool_map: dict[str, tuple[str, types.Tool]] = {}
        self.screenshot_history: list[str] = []

    def _filter_tools(self) -> None:
        """Apply tool filtering based on allowed/disallowed lists."""
        # Get all tools from client
        tool_map = self.mcp_client.get_tool_map()

        # Filter tools
        self._available_tools = []
        self._tool_map = {}

        for tool_name, (server_name, tool) in tool_map.items():
            # Check if tool should be included
            if self.allowed_tools and tool_name not in self.allowed_tools:
                continue
            if tool_name in self.disallowed_tools:
                continue

            self._available_tools.append(tool)
            self._tool_map[tool_name] = (server_name, tool)

    async def initialize(self, task: str | TaskConfig | None = None) -> None:
        """Initialize the agent with task-specific configuration."""
        # If client wasn't initialized on construction, do it now
        if not self.mcp_client.get_sessions():
            await self.mcp_client.initialize()

        # If task is provided, add lifecycle tools
        from hud.datasets import TaskConfig

        if isinstance(task, TaskConfig):
            if task.setup_tool:
                self.lifecycle_tools.append(task.setup_tool.name)
            if task.evaluate_tool:
                self.lifecycle_tools.append(task.evaluate_tool.name)

        # Re-apply filtering with updated lifecycle tools
        self._filter_tools()

        logger.info(
            "Agent initialized with %d available tools (after filtering)",
            len(self._available_tools),
        )

    def get_available_tools(self) -> list[types.Tool]:
        """Get list of available MCP tools for LLM use (excludes lifecycle tools)."""
        lifecycle_tool_names = self.lifecycle_tools
        return [tool for tool in self._available_tools if tool.name not in lifecycle_tool_names]

    def get_tool_map(self) -> dict[str, tuple[str, types.Tool]]:
        """Get mapping of tool names to (server_name, tool) tuples."""
        return self._tool_map

    def get_sessions(self) -> dict[str, Any]:
        """Get active MCP sessions."""
        return self.mcp_client.get_sessions()

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
        sessions = self.mcp_client.get_sessions()
        for server_name, tool in self._tool_map.values():
            session = sessions[server_name]
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

    async def call_tool(self, tool_call: MCPToolCall) -> MCPToolResult:
        """
        Call a tool through the MCP client.

        Args:
            tool_call: Dict with 'name' and optional 'arguments' keys

        Returns:
            The raw MCPToolResult
        """
        tool_name = tool_call.name
        if not tool_name:
            raise ValueError("Tool call must have a 'name' field")

        tool_args = tool_call.arguments

        if tool_name not in self._tool_map and tool_name not in self.lifecycle_tools:
            raise ValueError(f"Tool '{tool_name}' not found or not allowed")

        if self.mcp_client is None:
            raise ValueError("Client is not initialized")

        # Use client's call_tool method which handles routing
        result = await self.mcp_client.call_tool(tool_name, tool_args)

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
            if tool.name in self.lifecycle_tools:
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
                        tool_call = MCPToolCall(name=tool_name, arguments={"type": "screenshot"})
                    else:
                        tool_call = MCPToolCall(name=tool_name, arguments={"action": "screenshot"})

                    result = await self.call_tool(tool_call)

                    # Extract screenshot from result
                    for content in result.content:
                        if isinstance(content, types.ImageContent):
                            logger.info("Captured screenshot")
                            return content.data

                except Exception as e:
                    logger.warning("Failed to capture screenshot with %s: %s", tool_name, e)

        return None

    def extract_latest_screenshot(self, tool_results: list[MCPToolResult]) -> str | None:
        """Extract the latest screenshot from tool results."""
        latest_screenshot = None
        for result in tool_results:
            if not result.isError:
                for content in result.content:
                    if isinstance(content, types.ImageContent):
                        latest_screenshot = content.data
        return latest_screenshot

    async def run(self, prompt_or_task: str | TaskConfig, max_steps: int = 10) -> AgentResult:
        """
        Run the agent with the given prompt or task.

        Args:
            prompt_or_task: Either a string prompt for simple execution or a Task object
            max_steps: Maximum number of steps

        Returns:
            AgentResult with appropriate fields populated based on execution type
        """
        # Import here to avoid circular imports
        from hud.datasets import TaskConfig

        if not self._available_tools:
            await self.initialize(prompt_or_task)

        # Handle Task objects with full lifecycle
        if isinstance(prompt_or_task, TaskConfig):
            return await self._run_task(prompt_or_task, max_steps)

        # Handle simple string prompts
        elif isinstance(prompt_or_task, str):
            return await self._run_prompt(prompt_or_task, max_steps)

        else:
            raise TypeError(f"prompt_or_task must be str or TaskConfig, got {type(prompt_or_task)}")

    async def _run_task(self, task: TaskConfig, max_steps: int = 10) -> AgentResult:
        """
        Execute a task with setup and evaluate phases.

        Args:
            task: Task object with prompt, setup, and evaluate configs
            max_steps: Maximum steps for task execution

        Returns:
            AgentResult with reward, done, and info fields
        """
        try:
            # Setup phase
            if task.setup_tool is not None:
                await self.call_tool(task.setup_tool)

            # Execute the task prompt
            prompt_result = await self._run_prompt(task.prompt, max_steps)

            # Evaluate phase
            if task.evaluate_tool is not None:
                eval_result = await self.call_tool(task.evaluate_tool)

                # Return evaluation result if it's properly formatted
                if (
                    isinstance(eval_result, MCPToolResult)
                    and eval_result.structuredContent is not None
                ):
                    return AgentResult(
                        reward=self._find_reward(eval_result),
                        done=True,
                        content=eval_result.structuredContent["content"],
                        messages=prompt_result.messages,
                    )
                else:
                    # Fallback for invalid evaluation format
                    return AgentResult(
                        reward=0.0,
                        done=True,
                        error="Invalid evaluation result",
                        info={"eval_result": eval_result},
                        messages=prompt_result.messages,
                    )
            else:
                # No evaluation - assume success
                return AgentResult(
                    reward=0.0,
                    done=True,
                    content=prompt_result.content,
                    messages=prompt_result.messages,
                )

        except Exception as e:
            return AgentResult(reward=0.0, done=True, error=str(e))

    def _find_reward(self, result: MCPToolResult) -> float:
        """Find the reward in the result.

        Agent accepts "reward", "grade", "score"

        If not found, return 0.0
        """
        accept_keys = ["reward", "grade", "score"]
        for key in accept_keys:
            if isinstance(result.structuredContent, dict) and key in result.structuredContent:
                return result.structuredContent[key]
        return 0.0

    def _format_error_result(self, error_message: str) -> MCPToolResult:
        return MCPToolResult(
            content=[types.TextContent(text=error_message, type="text")], isError=True
        )

    async def run_conversation(self, prompt: str, max_steps: int = 10) -> AgentResult:
        """
        Run the agent in interactive conversation mode.

        Args:
            prompt: The initial prompt to start the conversation
            max_steps: Maximum number of steps per turn

        Returns:
            AgentResult when conversation ends
        """
        try:
            latest_screenshot = None
            if self.initial_screenshot:
                latest_screenshot = await self.capture_screenshot()

            messages = await self.create_initial_messages(prompt, latest_screenshot)

            step = 0
            while step < max_steps:
                step += 1
                logger.info("Conversation step %s/%s", step, max_steps)

                try:
                    response = await self.get_model_response(messages)

                    # Log the model's response
                    logger.info("Model response - Content: %s", response.content)
                    logger.info(
                        "Model response - Tool calls: %s",
                        [tc.name for tc in response.tool_calls],
                    )

                    tool_calls = response.tool_calls
                    if not tool_calls:
                        # In conversation mode, if model responds without tools,
                        # show the response and get user input
                        model_response = response.content
                        if model_response:
                            print(f"\n🤖 Agent: {model_response}")  # noqa: T201
                            user_input = input("\n👤 You: ").strip()
                            if user_input.lower() in ["exit", "quit", "bye"]:
                                return AgentResult(done=True, reward=0.0, messages=messages)
                            # Add user's response to the conversation
                            user_message = await self.create_user_message(user_input)
                            messages.append(user_message)
                            continue
                        else:
                            # No content and no tools - something went wrong
                            return AgentResult(
                                done=False,
                                reward=0.0,
                                error="No response generated",
                                messages=messages,
                            )

                    # Execute tool calls
                    tool_results = []
                    for tool_call in tool_calls:
                        try:
                            result = await self.call_tool(tool_call)
                            tool_results.append(result)
                        except Exception as e:
                            logger.error("Tool execution failed: %s", e)
                            # Create error MCPToolResult
                            error_result = MCPToolResult(
                                content=[types.TextContent(text=str(e), type="text")], isError=True
                            )
                            tool_results.append(error_result)

                    # Format tool results for the model
                    tool_messages = await self.format_tool_results(tool_calls, tool_results)
                    messages.extend(tool_messages)

                except Exception as e:
                    logger.error("Model call failed: %s", e)
                    return AgentResult(done=False, reward=0.0, error=str(e), messages=messages)

            return AgentResult(done=True, reward=0.0, messages=messages)

        except KeyboardInterrupt:
            logger.info("Conversation interrupted by user")
            return AgentResult(
                done=False, reward=0.0, messages=messages if "messages" in locals() else []
            )
        except asyncio.CancelledError:
            logger.info("Conversation cancelled")
            return AgentResult(
                done=False, reward=0.0, messages=messages if "messages" in locals() else []
            )

    async def _run_prompt(self, prompt: str, max_steps: int = 10) -> AgentResult:
        """
        Run the agent with the given prompt in task mode.

        Args:
            prompt: The task to complete
            max_steps: Maximum number of steps

        Returns:
            AgentResult for task completion
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
                    response = await self.get_model_response(messages)

                    # Log the model's response
                    logger.info("Model response - Content: %s", response.content)
                    logger.info(
                        "Model response - Tool calls: %s",
                        [tc.name for tc in response.tool_calls],
                    )
                    logger.info("Model response - Done: %s", response.done)

                    # Check if we should stop
                    if response.done:
                        return AgentResult(
                            content=response.content, done=response.done, messages=messages
                        )

                    tool_calls = response.tool_calls
                    if not tool_calls:
                        # In task mode, no tool calls means we're done
                        logger.info("No tool calls - stopping execution")
                        logger.info(
                            "Final message: %s",
                            response.content,
                        )
                        return AgentResult(
                            done=True, reward=0.0, content=response.content, messages=messages
                        )

                    # Execute tool calls
                    tool_results = []
                    for tool_call in tool_calls:
                        try:
                            result = await self.call_tool(tool_call)
                            tool_results.append(result)
                        except Exception as e:
                            logger.error("Tool execution failed: %s", e)
                            # Create error MCPToolResult
                            error_result = MCPToolResult(
                                content=[types.TextContent(text=str(e), type="text")], isError=True
                            )
                            tool_results.append(error_result)

                    # Format tool results for the model
                    tool_messages = await self.format_tool_results(tool_calls, tool_results)
                    messages.extend(tool_messages)

                except Exception as e:
                    logger.error("Model call failed: %s", e)
                    return AgentResult(done=False, reward=0.0, error=str(e), messages=messages)

            return AgentResult(done=True, reward=0.0, messages=messages)

        except KeyboardInterrupt:
            logger.info("Agent execution interrupted by user")
            return AgentResult(done=False, reward=0.0, messages=messages)
        except asyncio.CancelledError:
            logger.info("Agent execution cancelled")
            return AgentResult(done=False, reward=0.0, messages=messages)

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
    async def get_model_response(self, messages: list[Any]) -> ModelResponse:
        """
        Get response from the model including any tool calls.

        Args:
            messages: List of messages in provider-specific format

        Returns:
            ModelResponse with content, tool_calls, and done fields
        """

    @abstractmethod
    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[Any]:
        """
        Format tool results into messages for the model.

        Args:
            tool_calls: List of MCPToolCall objects that were executed
            tool_results: List of MCPToolResult objects from tool execution

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
