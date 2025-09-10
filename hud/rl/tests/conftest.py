"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from hud.rl.config import Config
from hud.rl.types import Episode


@pytest.fixture
def test_config():
    """Create a test configuration."""
    config = Config()
    config.training.episodes_per_batch = 2
    config.training.max_training_steps = 1
    config.actor.parallel_episodes = 1
    config.actor.max_steps_per_episode = 10
    return config


@pytest.fixture
def mock_episode():
    """Create a mock episode for testing."""
    return Episode(
        task_id="test_task",
        terminal_reward=1.0,
        conversation_history=[
            {"role": "user", "content": "Test message"},
            {"role": "assistant", "content": "Test response"}
        ],
        tool_spec=[],
        info={},
        metadata={},
    )


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = AsyncMock()
    client.chat = AsyncMock()
    client.chat.completions = AsyncMock()
    client.chat.completions.create = AsyncMock()
    
    # Mock response
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = "Mock response"
    response.choices[0].message.tool_calls = []
    
    client.chat.completions.create.return_value = response
    
    return client


@pytest.fixture
def fixtures_dir():
    """Return the fixtures directory path."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_processor():
    """Create a mock HuggingFace processor."""
    processor = Mock()
    
    # Mock apply_chat_template
    processor.apply_chat_template = Mock(return_value="Mock template")
    
    # Mock processing
    mock_inputs = Mock()
    mock_inputs.input_ids = Mock()
    mock_inputs.input_ids.shape = [-1, 10]  # Mock shape
    processor.return_value = mock_inputs
    
    return processor