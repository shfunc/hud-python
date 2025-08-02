"""Tests for BaseMCPAgent using simulated actions."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, create_autospec

import pytest
from mcp import types
from mcp_use import MCPClient

from hud.mcp_agent.base import BaseMCPAgent
from hud.task import Task
from hud.tools.executors.base import BaseExecutor


class MockMCPAgent(BaseMCPAgent):
    """Concrete implementation of BaseMCPAgent for testing."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.executor = BaseExecutor()  # Use simulated executor
        self._messages = []

    async def run(self, task: Task) -> list[dict[str, Any]]:
        """Mock run method."""
        return self._messages

    def create_initial_messages(self, prompt: str, screenshot: str | None = None) -> list[dict[str, Any]]:
        """Mock create initial messages."""
        messages = [{"role": "user", "content": prompt}]
        if screenshot:
            messages.append({"role": "assistant", "content": f"Screenshot: {screenshot}"})
        return messages

    def get_model_response(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Mock get model response."""
        return {"role": "assistant", "content": "Mock response"}

    def format_tool_results(
        self,
        results: list[tuple[str, Any]],
        screenshot: str | None = None,
        assistant_msg: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Mock format tool results."""
        formatted = []
        for tool_name, result in results:
            formatted.append({"role": "tool", "name": tool_name, "content": str(result)})
        if screenshot:
            formatted.append({"role": "screenshot", "content": screenshot})
        return formatted

    async def create_user_message(self, text: str) -> Any:
        """Mock create user message."""
        return {"role": "user", "content": text}

    async def get_model_response(self, messages: list[Any], step: int) -> dict[str, Any]:
        """Mock get model response."""
        return {"role": "assistant", "content": f"Response for step {step}"}


class TestBaseMCPAgent:
    """Tests for BaseMCPAgent with simulated actions."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        agent = MockMCPAgent()
        
        assert agent.client is not None
        assert agent.allowed_tools is None
        assert agent.disallowed_tools == []
        assert agent.initial_screenshot is False
        assert agent.max_screenshot_history == 3
        assert agent.append_tool_system_prompt is True
        assert agent.custom_system_prompt is None
        assert agent.lifecycle_tools == {"setup": "setup", "evaluate": "evaluate"}

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        client = MagicMock()
        agent = MockMCPAgent(
            client=client,
            allowed_tools=["tool1", "tool2"],
            disallowed_tools=["bad_tool"],
            initial_screenshot=True,
            max_screenshot_history=5,
            append_tool_system_prompt=False,
            custom_system_prompt="Custom prompt",
            lifecycle_tools={"setup": "custom_setup", "evaluate": "custom_eval"}
        )
        
        assert agent.client == client
        assert agent.allowed_tools == ["tool1", "tool2"]
        assert agent.disallowed_tools == ["bad_tool"]
        assert agent.initial_screenshot is True
        assert agent.max_screenshot_history == 5
        assert agent.append_tool_system_prompt is False
        assert agent.custom_system_prompt == "Custom prompt"
        assert agent.lifecycle_tools == {"setup": "custom_setup", "evaluate": "custom_eval"}

    @pytest.mark.asyncio
    async def test_initialize_no_client(self):
        """Test initialize fails without client."""
        agent = MockMCPAgent()
        agent.client = None
        
        with pytest.raises(ValueError, match="Client is not initialized"):
            await agent.initialize()

    @pytest.mark.asyncio
    async def test_initialize_with_sessions(self):
        """Test initialize with existing sessions."""
        agent = MockMCPAgent()
        
        # Create proper async mock for session
        mock_session = MagicMock()
        
        # Create async function for list_tools
        async def mock_list_tools():
            return types.ListToolsResult(
                tools=[
                    types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"}),
                    types.Tool(name="tool2", description="Tool 2", inputSchema={"type": "object"}),
                    types.Tool(name="setup", description="Setup tool", inputSchema={"type": "object"}),
                ]
            )
        
        mock_session.list_tools = mock_list_tools
        mock_session.connector = MagicMock()
        mock_session.connector.client_session = MagicMock()
        
        agent.client.get_all_active_sessions = MagicMock(return_value={"server1": mock_session})
        
        await agent.initialize()
        
        # Check available tools were populated
        tools = agent.get_available_tools()
        assert len(tools) == 3
        
        # Check tool map was populated
        tool_map = agent.get_tool_map()
        assert len(tool_map) == 3
        assert "tool1" in tool_map
        assert "tool2" in tool_map
        assert "setup" in tool_map

    @pytest.mark.asyncio
    async def test_initialize_with_filtering(self):
        """Test initialize with tool filtering."""
        agent = MockMCPAgent(allowed_tools=["tool1"], disallowed_tools=["tool3"])
        
        # Create proper async mock for session
        mock_session = MagicMock()
        
        async def mock_list_tools():
            return types.ListToolsResult(
                tools=[
                    types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"}),
                    types.Tool(name="tool2", description="Tool 2", inputSchema={"type": "object"}),
                    types.Tool(name="tool3", description="Tool 3", inputSchema={"type": "object"}),
                    types.Tool(name="setup", description="Setup", inputSchema={"type": "object"}),
                ]
            )
        
        mock_session.list_tools = mock_list_tools
        mock_session.connector = MagicMock()
        mock_session.connector.client_session = MagicMock()
        
        agent.client.get_all_active_sessions = MagicMock(return_value={"server1": mock_session})
        
        await agent.initialize()
        
        # Check filtering worked - should have tool1 and setup (lifecycle tool)
        tools = agent.get_available_tools()
        tool_names = [t.name for t in tools]
        assert len(tools) == 2
        assert "tool1" in tool_names
        assert "setup" in tool_names  # Lifecycle tool always included
        assert "tool2" not in tool_names  # Not in allowed list
        assert "tool3" not in tool_names  # In disallowed list

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """Test successful tool call."""
        agent = MockMCPAgent()
        
        # Initialize with a tool
        mock_session = MagicMock()
        
        async def mock_list_tools():
            return types.ListToolsResult(
                tools=[types.Tool(name="test_tool", description="Test", inputSchema={"type": "object"})]
            )
        
        mock_session.list_tools = mock_list_tools
        mock_session.connector = MagicMock()
        mock_session.connector.client_session = MagicMock()
        
        # Mock the call_tool method on the client session
        mock_result = types.CallToolResult(
            content=[types.TextContent(type="text", text="Tool result")],
            isError=False
        )
        
        async def mock_call_tool(name, args):
            return mock_result
        
        mock_session.connector.client_session.call_tool = mock_call_tool
        
        agent.client.get_all_active_sessions = MagicMock(return_value={"server1": mock_session})
        agent.client.get_session = MagicMock(return_value=mock_session)
        
        await agent.initialize()
        
        # Call the tool
        result = await agent.call_tool({"name": "test_tool", "arguments": {"param": "value"}})
        
        assert result == mock_result
        assert not result.isError

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self):
        """Test calling non-existent tool."""
        agent = MockMCPAgent()
        
        # Initialize without tools
        mock_session = MagicMock()
        
        async def mock_list_tools():
            return types.ListToolsResult(tools=[])
        
        mock_session.list_tools = mock_list_tools
        agent.client.get_all_active_sessions = MagicMock(return_value={"server1": mock_session})
        
        await agent.initialize()
        
        # Try to call unknown tool
        with pytest.raises(ValueError, match="Tool 'unknown_tool' not found"):
            await agent.call_tool({"name": "unknown_tool", "arguments": {}})

    @pytest.mark.asyncio
    async def test_call_tool_no_name(self):
        """Test calling tool without name."""
        agent = MockMCPAgent()
        
        with pytest.raises(ValueError, match="Tool call must have a 'name' field"):
            await agent.call_tool({"arguments": {}})

    def test_get_system_prompt_default(self):
        """Test get_system_prompt with default settings."""
        agent = MockMCPAgent()
        
        # Add some tools
        agent._available_tools = [
            types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"}),
            types.Tool(name="setup", description="Setup", inputSchema={"type": "object"}),
        ]
        
        prompt = agent.get_system_prompt()
        
        # Should include tool descriptions
        assert "tool1" in prompt
        assert "Tool 1" in prompt
        # Should not include lifecycle tools
        assert "setup" not in prompt

    def test_get_system_prompt_custom(self):
        """Test get_system_prompt with custom prompt."""
        agent = MockMCPAgent(
            custom_system_prompt="My custom prompt",
            append_tool_system_prompt=False
        )
        
        prompt = agent.get_system_prompt()
        assert prompt == "My custom prompt"

    def test_has_computer_tools(self):
        """Test checking for computer tools."""
        agent = MockMCPAgent()
        
        # No tools
        assert not agent.has_computer_tools()
        
        # With computer tool
        agent._available_tools = [
            types.Tool(name="computer", description="Computer", inputSchema={"type": "object"})
        ]
        assert agent.has_computer_tools()
        
        # With screenshot tool
        agent._available_tools = [
            types.Tool(name="screenshot", description="Screenshot", inputSchema={"type": "object"})
        ]
        assert agent.has_computer_tools()

    def test_get_tool_schemas(self):
        """Test getting tool schemas."""
        agent = MockMCPAgent()
        
        agent._available_tools = [
            types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"}),
            types.Tool(name="setup", description="Setup", inputSchema={"type": "object"}),
        ]
        
        schemas = agent.get_tool_schemas()
        
        # Should include non-lifecycle tools
        assert len(schemas) == 1
        assert schemas[0]["name"] == "tool1"

    @pytest.mark.asyncio
    async def test_capture_screenshot_no_tool(self):
        """Test screenshot capture without screenshot tool."""
        agent = MockMCPAgent()
        
        screenshot = await agent.capture_screenshot()
        assert screenshot is None

    @pytest.mark.asyncio
    async def test_capture_screenshot_with_tool(self):
        """Test screenshot capture with screenshot tool."""
        agent = MockMCPAgent()
        
        # Set up screenshot tool
        mock_session = MagicMock()
        
        async def mock_list_tools():
            return types.ListToolsResult(
                tools=[types.Tool(name="screenshot", description="Screenshot", inputSchema={"type": "object"})]
            )
        
        mock_session.list_tools = mock_list_tools
        mock_session.connector = MagicMock()
        mock_session.connector.client_session = MagicMock()
        
        # Mock screenshot result
        mock_result = types.CallToolResult(
            content=[types.ImageContent(type="image", data="base64imagedata", mimeType="image/png")],
            isError=False
        )
        
        async def mock_call_tool(name, args):
            return mock_result
        
        mock_session.connector.client_session.call_tool = mock_call_tool
        
        agent.client.get_all_active_sessions = MagicMock(return_value={"server1": mock_session})
        agent.client.get_session = MagicMock(return_value=mock_session)
        
        await agent.initialize()
        
        screenshot = await agent.capture_screenshot()
        assert screenshot == "base64imagedata"

    def test_process_tool_results_extracts_text(self):
        """Test processing tool results extracts text content."""
        agent = MockMCPAgent()
        
        tool_results = [
            {
                "role": "tool",
                "name": "test_tool",
                "content": [
                    {"type": "text", "text": "Result text"},
                    {"type": "image", "data": "imagedata", "mimeType": "image/png"}
                ]
            }
        ]
        
        processed = agent.process_tool_results(tool_results)
        
        assert "outputs" in processed
        assert "test_tool" in processed["outputs"]
        assert processed["outputs"]["test_tool"] == "Result text"

    def test_get_tools_by_server(self):
        """Test getting tools grouped by server."""
        agent = MockMCPAgent()
        
        # Set up tools from different servers
        tool1 = types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"})
        tool2 = types.Tool(name="tool2", description="Tool 2", inputSchema={"type": "object"})
        
        agent._available_tools = [tool1, tool2]
        agent._tool_map = {
            "tool1": ("server1", tool1),
            "tool2": ("server2", tool2),
        }
        
        tools_by_server = agent.get_tools_by_server()
        
        assert len(tools_by_server) == 2
        assert "server1" in tools_by_server
        assert "server2" in tools_by_server
        assert tools_by_server["server1"] == [tool1]
        assert tools_by_server["server2"] == [tool2]

    @pytest.mark.asyncio
    async def test_executor_integration(self):
        """Test integration with BaseExecutor for simulated actions."""
        agent = MockMCPAgent()
        
        # Test various executor actions
        click_result = await agent.executor.click(100, 200, take_screenshot=False)
        assert "[SIMULATED] Click at (100, 200)" in click_result.output
        
        type_result = await agent.executor.type("Test input", take_screenshot=False)
        assert "[SIMULATED] Type 'Test input'" in type_result.output
        
        scroll_result = await agent.executor.scroll(x=50, y=50, scroll_y=5, take_screenshot=False)
        assert "[SIMULATED] Scroll" in scroll_result.output
        
        # Test screenshot
        screenshot = await agent.executor.screenshot()
        assert isinstance(screenshot, str)
        assert screenshot.startswith("iVBORw0KGgo")  # PNG header