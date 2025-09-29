"""Tests for hud.cli.eval module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from hud.cli.eval import build_agent, eval_command, get_available_models, run_full_dataset, run_single_task
from hud.types import Task, Trace

class TestBuildAgent:
    """Test the build_agent function."""

    def test_builds_integration_test_agent(self) -> None:
        """
        Test building an integration test agent.
        """
        with patch("hud.agents.misc.integration_test_agent.IntegrationTestRunner") as mock_runner:
            mock_instance = Mock()
            mock_runner.return_value = mock_instance
            
            # Test with verbose=False
            result = build_agent("integration_test", verbose=False)
            
            mock_runner.assert_called_once_with(verbose=False)
            assert result == mock_instance

    def test_builds_claude_agent(self) -> None:
        """
        Test building a Claude agent with default model.
        """
        with patch("hud.agents.ClaudeAgent") as mock_runner:
            mock_instance = Mock()
            mock_runner.return_value = mock_instance
            
            # Test with verbose=False
            result = build_agent("claude", verbose=False)
            
            mock_runner.assert_called_once_with(
                model="claude-sonnet-4-20250514",
                verbose=False
            )
            assert result == mock_instance

    def test_builds_claude_agent_with_custom_model_and_allowed_tools(self) -> None:
        """
        Test building a Claude agent with custom model name and allowed tools.
        """
        with patch("hud.agents.ClaudeAgent") as mock_runner:
            mock_instance = Mock()
            mock_runner.return_value = mock_instance
            
            # Test with verbose=False
            result = build_agent(
                "claude",
                model="claude-sonnet-4-20250514",
                allowed_tools=["act"],
                verbose=True,
            )
            
            mock_runner.assert_called_once_with(
                model="claude-sonnet-4-20250514",
                allowed_tools=["act"],
                verbose=True,
            )
            assert result == mock_instance


class TestRunSingleTask:
    """Test the run_single_task function."""

    @pytest.mark.asyncio
    async def test_applies_agent_config_from_task(self) -> None:
        """Test that task.agent_config is applied during agent initialization."""
        mock_task = Task(
            prompt="Test",
            mcp_config={"local": {"url": "http://localhost:8765/mcp"}},
            agent_config={
                "system_prompt": "Custom instructions",
                "allowed_tools": ["tool1", "tool2"],
                "append_setup_output": False,
            }
        )
        mock_agent = AsyncMock(
            initialize=AsyncMock(),
            run=AsyncMock(return_value=Trace(reward=1.0, done=True))
        )
        
        with patch("hud.utils.tasks.load_tasks", return_value=[mock_task]), \
             patch("hud.agents.misc.integration_test_agent.IntegrationTestRunner", return_value=mock_agent), \
             patch("hud.cli.eval.find_environment_dir", return_value=None), \
             patch("hud.cli.eval.hud.trace"):
            await run_single_task("test.json", agent_type="integration_test", max_steps=10)
            
            # Verify agent.run was called with the task containing agent_config
            mock_agent.run.assert_called_once()
            called_task = mock_agent.run.call_args[0][0]
            assert called_task.agent_config == mock_task.agent_config

    @pytest.mark.asyncio
    async def test_runs_with_group_size_greater_than_one(self) -> None:
        """Test that group_size > 1 triggers run_tasks_grouped instead of agent.run."""
        mock_task = Task(prompt="Test", mcp_config={"local": {"url": "http://localhost:8765/mcp"}})
        
        with patch("hud.utils.tasks.load_tasks", return_value=[mock_task]), \
             patch("hud.cli.eval.run_tasks_grouped", new_callable=AsyncMock) as mock_grouped, \
             patch("hud.cli.eval.display_group_statistics"), \
             patch("hud.cli.eval.find_environment_dir", return_value=None), \
             patch("hud.cli.eval.hud.trace"):
            
            mock_grouped.return_value = [{"task": mock_task, "rewards": [1.0, 0.5]}]
            
            await run_single_task("test.json", agent_type="integration_test", group_size=3, max_steps=10)
            
            # Verify run_tasks_grouped was called with correct group_size
            mock_grouped.assert_called_once()
            assert mock_grouped.call_args.kwargs["group_size"] == 3
            assert mock_grouped.call_args.kwargs["max_steps"] == 10
