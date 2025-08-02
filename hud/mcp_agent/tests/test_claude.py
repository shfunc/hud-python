"""Tests for Claude MCP agent."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from mcp.types import Tool

from hud.mcp_agent.claude import ClaudeMCPAgent


class TestClaudeMCPAgent:
    """Tests for ClaudeMCPAgent."""

    def test_init_default(self):
        """Test default initialization."""
        agent = ClaudeMCPAgent()
        assert agent.model == "claude-3-5-sonnet-latest"
        assert agent.api_key is None
        assert agent.max_tokens == 4000
        assert agent.temperature == 0.0

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        agent = ClaudeMCPAgent(
            model="claude-3-opus-20240229", api_key="test-key", temperature=0.5, max_tokens=2000
        )
        assert agent.model == "claude-3-opus-20240229"
        assert agent.api_key == "test-key"
        assert agent.temperature == 0.5
        assert agent.max_tokens == 2000

    @pytest.mark.asyncio
    async def test_create_initial_messages(self):
        """Test creating initial messages."""
        agent = ClaudeMCPAgent()
        agent.custom_system_prompt = "You are a helpful assistant."
        agent._available_tools = []

        prompt = "Hello, how are you?"
        screenshot = None

        messages = await agent.create_initial_messages(prompt, screenshot)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == prompt

    @pytest.mark.asyncio
    async def test_create_initial_messages_with_screenshot(self):
        """Test creating initial messages with screenshot."""
        agent = ClaudeMCPAgent()
        agent.custom_system_prompt = "You are a helpful assistant."
        agent._available_tools = []

        prompt = "What do you see?"
        screenshot = "base64encodedimage"

        messages = await agent.create_initial_messages(prompt, screenshot)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert isinstance(messages[0]["content"], list)
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][0]["text"] == prompt
        assert messages[0]["content"][1]["type"] == "image"

    @pytest.mark.asyncio
    async def test_get_model_response_no_client(self):
        """Test get_model_response when client is not initialized."""
        agent = ClaudeMCPAgent(api_key="test-key")
        agent._client = None

        messages = [{"role": "user", "content": "Hello"}]

        # Mock Anthropic client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hi there!")]
        mock_response.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(return_value=mock_response)

        with patch("anthropic.Anthropic", return_value=mock_client):
            response = await agent.get_model_response(messages, 1)

        assert response["content"] == "Hi there!"
        assert response["done"] is True
        assert response["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_format_tool_results(self):
        """Test formatting tool results."""
        agent = ClaudeMCPAgent()

        processed_results = {
            "text": "Tool executed successfully",
            "screenshot": None,
            "errors": [],
            "results": [("test_tool", [{"type": "text", "text": "Success"}])],
        }

        tool_calls = [{"id": "tool_123", "name": "test_tool"}]

        messages = await agent.format_tool_results(processed_results, tool_calls)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert len(messages[0]["content"]) == 1
        assert messages[0]["content"][0]["type"] == "tool_result"
        assert messages[0]["content"][0]["tool_use_id"] == "tool_123"

    def test_get_tool_schemas(self):
        """Test getting tool schemas."""
        agent = ClaudeMCPAgent()

        # Create mock tools
        tool1 = Tool(
            name="test_tool",
            description="A test tool",
            inputSchema={"type": "object", "properties": {"param": {"type": "string"}}},
        )
        tool2 = Tool(name="setup", description="Setup tool", inputSchema={})

        agent._available_tools = [tool1, tool2]

        schemas = agent.get_tool_schemas()

        # Should exclude lifecycle tools like "setup"
        assert len(schemas) == 1
        assert schemas[0]["name"] == "test_tool"
        assert schemas[0]["description"] == "A test tool"
        assert "input_schema" in schemas[0]
