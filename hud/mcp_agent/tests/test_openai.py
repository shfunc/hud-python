"""Tests for OpenAI MCP agent."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from mcp.types import Tool

from hud.mcp_agent.openai import OpenAIMCPAgent


class TestOpenAIMCPAgent:
    """Tests for OpenAIMCPAgent."""

    def test_init_default(self):
        """Test default initialization."""
        agent = OpenAIMCPAgent()
        assert agent.model == "gpt-4o"
        assert agent.api_key is None
        assert agent.temperature == 0.0

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        agent = OpenAIMCPAgent(
            model="gpt-3.5-turbo",
            api_key="test-key",
            temperature=0.7,
            base_url="https://custom.api.com",
        )
        assert agent.model == "gpt-3.5-turbo"
        assert agent.api_key == "test-key"
        assert agent.temperature == 0.7
        assert agent.base_url == "https://custom.api.com"

    @pytest.mark.asyncio
    async def test_create_initial_messages(self):
        """Test creating initial messages."""
        agent = OpenAIMCPAgent()
        agent.custom_system_prompt = "You are a helpful assistant."
        agent._available_tools = []

        prompt = "Hello, how are you?"
        screenshot = None

        messages = await agent.create_initial_messages(prompt, screenshot)

        assert len(messages) == 2  # System + user message
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == prompt

    @pytest.mark.asyncio
    async def test_create_initial_messages_with_screenshot(self):
        """Test creating initial messages with screenshot."""
        agent = OpenAIMCPAgent()
        agent.custom_system_prompt = "You are a helpful assistant."
        agent._available_tools = []

        prompt = "What do you see?"
        screenshot = "base64encodedimage"

        messages = await agent.create_initial_messages(prompt, screenshot)

        assert len(messages) == 2  # System + user message
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert isinstance(messages[1]["content"], list)
        assert len(messages[1]["content"]) == 2
        assert messages[1]["content"][0]["type"] == "text"
        assert messages[1]["content"][0]["text"] == prompt
        assert messages[1]["content"][1]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_get_model_response_no_client(self):
        """Test get_model_response when client is not initialized."""
        agent = OpenAIMCPAgent(api_key="test-key")
        agent._client = None

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        # Mock OpenAI client response
        mock_choice = MagicMock()
        mock_choice.message.content = "Hi there!"
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)

        with patch("openai.OpenAI", return_value=mock_client):
            response = await agent.get_model_response(messages, 1)

        assert response["content"] == "Hi there!"
        assert response["done"] is True
        assert response["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_format_tool_results(self):
        """Test formatting tool results."""
        agent = OpenAIMCPAgent()

        processed_results = {
            "text": "Tool executed successfully",
            "screenshot": None,
            "errors": [],
            "results": [("test_tool", [{"type": "text", "text": "Success"}])],
        }

        tool_calls = [{"id": "call_123", "name": "test_tool"}]

        messages = await agent.format_tool_results(processed_results, tool_calls)

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_123"
        assert messages[0]["content"] == "Tool executed successfully"

    @pytest.mark.asyncio
    async def test_format_tool_results_with_screenshot(self):
        """Test formatting tool results with screenshot."""
        agent = OpenAIMCPAgent()

        processed_results = {
            "text": "Screenshot taken",
            "screenshot": "base64imagedata",
            "errors": [],
            "results": [("screenshot", [{"type": "image", "data": "base64imagedata"}])],
        }

        tool_calls = [{"id": "call_456", "name": "screenshot"}]

        messages = await agent.format_tool_results(processed_results, tool_calls)

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert isinstance(messages[0]["content"], list)
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][1]["type"] == "image_url"

    def test_get_tool_schemas(self):
        """Test getting tool schemas."""
        agent = OpenAIMCPAgent()

        # Create mock tools
        tool1 = Tool(
            name="test_tool",
            description="A test tool",
            inputSchema={"type": "object", "properties": {"param": {"type": "string"}}},
        )
        tool2 = Tool(name="evaluate", description="Evaluate tool", inputSchema={})

        agent._available_tools = [tool1, tool2]

        schemas = agent.get_tool_schemas()

        # Should exclude lifecycle tools like "evaluate"
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "test_tool"
        assert schemas[0]["function"]["description"] == "A test tool"
