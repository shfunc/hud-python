"""Extended tests for dataset utilities to improve coverage."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.datasets import (
    TaskConfig,
    run_dataset,
    save_taskconfigs,
    to_taskconfigs,
)
from hud.types import MCPToolCall


class TestTaskConfigExtended:
    """Extended tests for TaskConfig functionality."""

    def test_taskconfig_with_all_fields(self):
        """Test TaskConfig with all possible fields."""
        setup_tool = MCPToolCall(name="setup", arguments={"board_size": 4})
        evaluate_tool = MCPToolCall(name="evaluate", arguments={"metric": "score"})

        task = TaskConfig(
            id="test-123",
            prompt="Play the game",
            mcp_config={
                "server": {"url": "http://localhost:8080"},
                "auth": {"token": "test-token"},
            },
            setup_tool=setup_tool,
            evaluate_tool=evaluate_tool,
            metadata={"experiment": "test1", "version": 2},
        )

        assert task.id == "test-123"
        assert task.prompt == "Play the game"
        assert task.setup_tool == setup_tool
        assert task.evaluate_tool == evaluate_tool
        assert task.metadata["experiment"] == "test1"
        assert task.metadata["version"] == 2

    def test_taskconfig_list_tools(self):
        """Test TaskConfig with list of tools."""
        setup_tools = [
            MCPToolCall(name="init", arguments={}),
            MCPToolCall(name="configure", arguments={"mode": "test"}),
        ]

        task = TaskConfig(
            prompt="Multi-setup task", mcp_config={"test": True}, setup_tool=setup_tools
        )

        assert isinstance(task.setup_tool, list)
        assert len(task.setup_tool) == 2
        assert task.setup_tool[0].name == "init"
        assert task.setup_tool[1].arguments["mode"] == "test"

    def test_env_var_complex_resolution(self):
        """Test complex environment variable scenarios."""
        os.environ["API_KEY"] = "sk-12345"
        os.environ["BASE_URL"] = "https://api.example.com"
        os.environ["EMPTY_VAR"] = ""

        try:
            # Mock get_current_task_run_id
            with patch("hud.datasets.get_current_task_run_id", return_value="run-789"):
                task = TaskConfig(
                    prompt="Complex env test",
                    mcp_config={
                        "auth": {
                            "bearer": "Bearer ${API_KEY}",
                            "empty": "${EMPTY_VAR}",
                            "missing": "${MISSING_VAR}",
                        },
                        "endpoints": ["${BASE_URL}/v1", "${BASE_URL}/v2", "${MISSING_URL}"],
                        "metadata": {"run_id": "${RUN_ID}", "combined": "${API_KEY}-${RUN_ID}"},
                    },
                )

                assert task.mcp_config["auth"]["bearer"] == "Bearer sk-12345"
                assert task.mcp_config["auth"]["empty"] == ""
                assert task.mcp_config["auth"]["missing"] == ""
                assert task.mcp_config["endpoints"][0] == "https://api.example.com/v1"
                assert task.mcp_config["endpoints"][1] == "https://api.example.com/v2"
                assert task.mcp_config["endpoints"][2] == ""
                assert task.mcp_config["metadata"]["run_id"] == "run-789"
                assert task.mcp_config["metadata"]["combined"] == "sk-12345-run-789"

        finally:
            del os.environ["API_KEY"]
            del os.environ["BASE_URL"]
            del os.environ["EMPTY_VAR"]

    def test_non_string_values_preserved(self):
        """Test that non-string values are preserved during env resolution."""
        task = TaskConfig(
            prompt="Test non-strings",
            mcp_config={
                "string": "${MISSING}",
                "number": 42,
                "boolean": True,
                "null": None,
                "nested": {"list": [1, 2, "${VAR}", 4], "dict": {"key": "${KEY}", "num": 123}},
            },
        )

        assert task.mcp_config["string"] == ""
        assert task.mcp_config["number"] == 42
        assert task.mcp_config["boolean"] is True
        assert task.mcp_config["null"] is None
        assert task.mcp_config["nested"]["list"] == [1, 2, "", 4]
        assert task.mcp_config["nested"]["dict"]["num"] == 123


class TestDatasetOperations:
    """Test dataset conversion and operations."""

    def test_to_taskconfigs_with_nulls(self):
        """Test handling of null/missing fields in dataset."""
        mock_dataset = [
            {
                "prompt": "Minimal task",
                "mcp_config": json.dumps({"basic": True}),
                # All optional fields missing
            },
            {
                "id": None,  # Explicit None
                "prompt": "Task with nulls",
                "mcp_config": json.dumps({"test": True}),
                "metadata": None,
                "setup_tool": None,
                "evaluate_tool": None,
            },
        ]

        tasks = to_taskconfigs(mock_dataset)

        assert len(tasks) == 2
        assert tasks[0].id is None
        assert tasks[0].metadata == {}
        assert tasks[0].setup_tool is None
        assert tasks[0].evaluate_tool is None

        # Second task should handle None values properly
        assert tasks[1].id is None
        assert tasks[1].metadata == {}

    def test_to_taskconfigs_invalid_json(self):
        """Test error handling for invalid JSON in dataset."""
        mock_dataset = [{"prompt": "Bad JSON task", "mcp_config": "not-valid-json"}]

        with pytest.raises(json.JSONDecodeError):
            to_taskconfigs(mock_dataset)

    def test_save_taskconfigs_empty_list(self):
        """Test saving empty task list."""
        with patch("hud.datasets.Dataset") as MockDataset:
            mock_instance = MagicMock()
            MockDataset.from_list.return_value = mock_instance

            save_taskconfigs([], "test-org/empty-dataset")

            MockDataset.from_list.assert_called_once_with([])
            mock_instance.push_to_hub.assert_called_once()

    def test_save_taskconfigs_mixed_rejection(self):
        """Test that mixing dicts and TaskConfig objects is rejected."""
        valid_dict = {"prompt": "Dict task", "mcp_config": {"test": True}}

        task_object = TaskConfig(prompt="Object task", mcp_config={"resolved": "${SOME_VAR}"})

        # First item is dict, second is object
        with pytest.raises(ValueError, match="Item 1 is a TaskConfig object"):
            save_taskconfigs([valid_dict, task_object], "test-org/mixed")


class TestRunDatasetExtended:
    """Extended tests for run_dataset functionality."""

    @pytest.mark.asyncio
    async def test_run_dataset_empty(self):
        """Test running empty dataset."""
        with patch("hud.datasets.MCPClient"), patch("hud.datasets.hud") as mock_hud:
            mock_job = MagicMock()
            mock_job.id = "job-empty"
            mock_hud.job.return_value.__enter__.return_value = mock_job

            results = await run_dataset(
                "empty_run",
                [],  # Empty task list
                MagicMock(),
            )

            assert results == []
            mock_hud.trace.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_dataset_with_metadata(self):
        """Test run_dataset with custom metadata."""
        mock_agent_class = MagicMock()
        mock_agent_instance = AsyncMock()
        mock_agent_class.return_value = mock_agent_instance
        mock_agent_instance.run.return_value = {"status": "complete"}

        tasks = [TaskConfig(prompt="Task 1", mcp_config={"url": "test1"})]

        custom_metadata = {
            "experiment_id": "exp-123",
            "tags": ["test", "v2"],
            "config": {"temperature": 0.7},
        }

        with patch("hud.datasets.MCPClient") as MockClient:
            with patch("hud.datasets.hud") as mock_hud:
                mock_job = MagicMock()
                mock_job.id = "job-meta"
                mock_hud.job.return_value.__enter__.return_value = mock_job
                mock_hud.trace.return_value.__enter__.return_value = "trace-id"

                mock_client = AsyncMock()
                MockClient.return_value = mock_client

                await run_dataset(
                    "metadata_run",
                    tasks,
                    mock_agent_class,
                    {"model": "test-model"},
                    metadata=custom_metadata,
                )

                # Verify job was created with merged metadata
                expected_metadata = {
                    "experiment_id": "exp-123",
                    "tags": ["test", "v2"],
                    "config": {"temperature": 0.7},
                    "agent_class": mock_agent_class.__name__,
                    "agent_config": {"model": "test-model"},
                }

                mock_hud.job.assert_called_once_with("metadata_run", metadata=expected_metadata)

    @pytest.mark.asyncio
    async def test_run_dataset_exception_handling(self):
        """Test exception handling during task execution."""
        mock_agent_class = MagicMock()
        mock_agent_instance = AsyncMock()
        mock_agent_class.return_value = mock_agent_instance

        # Make second task fail
        call_count = 0

        async def mock_run(task):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Task 2 failed")
            return {"result": f"success-{call_count}"}

        mock_agent_instance.run.side_effect = mock_run

        tasks = [TaskConfig(prompt=f"Task {i}", mcp_config={"url": f"test{i}"}) for i in range(3)]

        with patch("hud.datasets.MCPClient") as MockClient:
            with patch("hud.datasets.hud") as mock_hud:
                mock_job = MagicMock()
                mock_job.id = "job-error"
                mock_hud.job.return_value.__enter__.return_value = mock_job
                mock_hud.trace.return_value.__enter__.return_value = "trace-id"

                mock_client = AsyncMock()
                MockClient.return_value = mock_client

                # Should complete without raising
                results = await run_dataset("error_run", tasks, mock_agent_class)

                # First and third should succeed
                assert results[0] == {"result": "success-1"}
                assert results[2] == {"result": "success-3"}
                # Second result depends on implementation details

    @pytest.mark.asyncio
    async def test_run_dataset_client_cleanup(self):
        """Test that MCP clients are properly cleaned up."""
        mock_agent_class = MagicMock()
        mock_agent_instance = AsyncMock()
        mock_agent_class.return_value = mock_agent_instance
        mock_agent_instance.run.return_value = {"done": True}

        tasks = [TaskConfig(prompt=f"Task {i}", mcp_config={"url": f"test{i}"}) for i in range(3)]

        # Track client instances
        client_instances = []

        def create_client(**kwargs):
            client = AsyncMock()
            client_instances.append(client)
            return client

        with patch("hud.datasets.MCPClient", side_effect=create_client):
            with patch("hud.datasets.hud") as mock_hud:
                mock_job = MagicMock()
                mock_job.id = "job-cleanup"
                mock_hud.job.return_value.__enter__.return_value = mock_job
                mock_hud.trace.return_value.__enter__.return_value = "trace-id"

                await run_dataset("cleanup_run", tasks, mock_agent_class)

                # Verify all clients were created and closed
                assert len(client_instances) == 3
                for client in client_instances:
                    client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_dataset_no_mcp_config_warning(self):
        """Test warning when task has no mcp_config."""
        # Create a mock task with no mcp_config
        mock_task = MagicMock()
        mock_task.mcp_config = None

        mock_agent_class = MagicMock()

        with patch("hud.datasets.to_taskconfigs", return_value=[mock_task]):
            with patch("hud.datasets.hud") as mock_hud:
                with patch("hud.datasets.logger") as mock_logger:
                    mock_job = MagicMock()
                    mock_job.id = "job-no-config"
                    mock_hud.job.return_value.__enter__.return_value = mock_job
                    mock_hud.trace.return_value.__enter__.return_value = "trace-id"

                    results = await run_dataset(
                        "no_config_run",
                        [],  # Will be converted by to_taskconfigs
                        mock_agent_class,
                    )

                    # Should log warning
                    mock_logger.warning.assert_called_with("Task %d has no mcp_config defined", 0)

                    # Result should be None
                    assert results == [None]
