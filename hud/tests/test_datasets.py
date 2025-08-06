"""Tests for dataset utilities and TaskConfig serialization."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from hud.datasets import (
    TaskConfig,
    save_taskconfigs,
    to_taskconfigs,
)


class TestTaskConfigWithJSONStrings:
    """Test TaskConfig with JSON string serialization for HuggingFace."""

    def test_save_and_load_with_json_strings(self):
        """Test saving and loading TaskConfigs with JSON string fields."""
        # Create test data as dicts (not TaskConfig objects)
        taskconfig_dicts = [
            {
                "id": "task-1",
                "prompt": "Test task 1",
                "mcp_config": {"server": {"url": "test1"}},
                "metadata": {"test": True},
                "setup_tool": {"name": "setup", "arguments": {"arg": "value"}},
            },
            {
                "prompt": "Test task 2",
                "mcp_config": {"server": {"url": "test2"}},
                "evaluate_tool": {"name": "evaluate", "arguments": {}},
            },
        ]

        # Mock Dataset and load_dataset (imported inside functions)
        with (
            patch("datasets.Dataset") as MockDataset,
            patch("datasets.load_dataset") as mock_load_dataset,
        ):
            # Mock the Dataset.from_list and push_to_hub
            mock_dataset_instance = MagicMock()
            MockDataset.from_list.return_value = mock_dataset_instance

            # Save TaskConfigs
            save_taskconfigs(taskconfig_dicts, "test-org/test-dataset", private=True)

            # Verify Dataset.from_list was called with JSON strings
            MockDataset.from_list.assert_called_once()
            saved_data = MockDataset.from_list.call_args[0][0]

            # Check that complex fields were serialized as JSON strings
            assert isinstance(saved_data[0]["mcp_config"], str)
            assert isinstance(saved_data[0]["metadata"], str)
            assert isinstance(saved_data[0]["setup_tool"], str)

            # Verify push_to_hub was called
            mock_dataset_instance.push_to_hub.assert_called_once_with(
                "test-org/test-dataset", private=True
            )

            # Now test loading
            # Mock the dataset returned by load_dataset
            mock_dataset = [
                {
                    "id": "task-1",
                    "prompt": "Test task 1",
                    "mcp_config": json.dumps({"server": {"url": "test1"}}),
                    "metadata": json.dumps({"test": True}),
                    "setup_tool": json.dumps({"name": "setup", "arguments": {"arg": "value"}}),
                },
                {
                    "prompt": "Test task 2",
                    "mcp_config": json.dumps({"server": {"url": "test2"}}),
                    "evaluate_tool": json.dumps({"name": "evaluate", "arguments": {}}),
                },
            ]
            mock_load_dataset.return_value = mock_dataset

            # Load TaskConfigs using to_taskconfigs
            loaded_tasks = to_taskconfigs(mock_dataset)

            # Verify TaskConfigs were created correctly
            assert len(loaded_tasks) == 2
            assert loaded_tasks[0].id == "task-1"
            assert loaded_tasks[0].prompt == "Test task 1"
            assert loaded_tasks[0].mcp_config == {"server": {"url": "test1"}}
            assert loaded_tasks[0].metadata == {"test": True}
            assert loaded_tasks[0].setup_tool.name == "setup"

            assert loaded_tasks[1].prompt == "Test task 2"
            assert loaded_tasks[1].evaluate_tool.name == "evaluate"

    def test_env_var_resolution_with_json_strings(self):
        """Test that env vars are resolved when loading from JSON strings."""
        os.environ["TEST_KEY"] = "secret123"

        try:
            # Mock load_dataset to return JSON string fields
            with patch("datasets.load_dataset") as mock_load_dataset:
                mock_dataset = [
                    {
                        "prompt": "Test",
                        "mcp_config": json.dumps(
                            {
                                "auth": "${TEST_KEY}",
                                "url": "${MISSING_VAR}",
                            }
                        ),
                    }
                ]
                mock_load_dataset.return_value = mock_dataset

                # Load TaskConfigs using to_taskconfigs
                tasks = to_taskconfigs(mock_dataset)

                # Verify env vars were resolved
                assert tasks[0].mcp_config["auth"] == "secret123"
                assert tasks[0].mcp_config["url"] == ""  # Missing var becomes empty
        finally:
            del os.environ["TEST_KEY"]

    def test_optional_fields_handling(self):
        """Test handling of optional fields in JSON string format."""
        with patch("datasets.load_dataset") as mock_load_dataset:
            # Dataset with some fields missing
            mock_dataset = [
                {
                    "prompt": "Task without optional fields",
                    "mcp_config": json.dumps({"basic": "config"}),
                    # No id, metadata, setup_tool, or evaluate_tool
                },
                {
                    "id": "task-2",
                    "prompt": "Task with all fields",
                    "mcp_config": json.dumps({"full": "config"}),
                    "metadata": json.dumps({"meta": "data"}),
                    "setup_tool": json.dumps({"name": "setup", "arguments": {}}),
                    "evaluate_tool": json.dumps({"name": "eval", "arguments": {}}),
                },
            ]
            mock_load_dataset.return_value = mock_dataset

            # Load TaskConfigs using to_taskconfigs
            tasks = to_taskconfigs(mock_dataset)

            # First task should have defaults for missing fields
            assert tasks[0].id is None
            assert tasks[0].metadata == {}
            assert tasks[0].setup_tool is None
            assert tasks[0].evaluate_tool is None

            # Second task should have all fields
            assert tasks[1].id == "task-2"
            assert tasks[1].metadata == {"meta": "data"}
            assert tasks[1].setup_tool.name == "setup"
            assert tasks[1].evaluate_tool.name == "eval"

    def test_save_rejects_taskconfig_objects(self):
        """Test that save_taskconfigs rejects TaskConfig objects to prevent secret exposure."""
        import os

        os.environ["SECRET_KEY"] = "should-not-be-saved"

        try:
            # Create TaskConfig objects (which will resolve env vars)
            task_objects = [
                TaskConfig(
                    prompt="Test",
                    mcp_config={
                        "auth": "${SECRET_KEY}"
                    },  # Will be resolved to "should-not-be-saved"
                )
            ]

            # Try to save TaskConfig objects (should fail)
            with patch("datasets.Dataset") as MockDataset:
                mock_dataset_instance = MagicMock()
                MockDataset.from_list.return_value = mock_dataset_instance

                with pytest.raises(ValueError) as exc_info:
                    save_taskconfigs(task_objects, "test-org/test-dataset")

                assert "expects dictionaries, not TaskConfig objects" in str(exc_info.value)
                assert "resolved environment variables" in str(exc_info.value)

        finally:
            del os.environ["SECRET_KEY"]
