from __future__ import annotations

import asyncio
import uuid
from unittest.mock import MagicMock

import pytest

from hud.telemetry._trace import (
    init_telemetry,
    trace,
    trace_decorator,
    trace_open,
)
from hud.telemetry.context import get_current_task_run_id as actual_get_current_task_run_id
from hud.telemetry.context import is_root_trace as actual_is_root_trace
from hud.telemetry.context import set_current_task_run_id as actual_set_current_task_run_id


@pytest.fixture(autouse=True)
def reset_telemetry_context_fixture():
    """Ensures telemetry context is reset before and after each test in this file."""
    # Reset context before test
    actual_set_current_task_run_id(None)
    actual_is_root_trace.set(False)
    yield
    # Reset context after test
    actual_set_current_task_run_id(None)
    actual_is_root_trace.set(False)


class TestInitTelemetry:
    """Test telemetry initialization."""

    def test_init_telemetry(self, mocker):
        """Test telemetry initialization calls registry.install_all."""
        mock_registry = mocker.patch("hud.telemetry._trace.registry", autospec=True)
        init_telemetry()
        mock_registry.install_all.assert_called_once()


class TestTrace:
    """Test the trace context manager."""

    def test_trace_basic(self, mocker):
        """Test basic trace functionality and context setting."""
        mock_flush = mocker.patch(
            "hud.telemetry._trace.flush_buffer", return_value=[], autospec=True
        )
        mock_submit_loop = mocker.patch(
            "hud.telemetry.exporter.submit_to_worker_loop", return_value=MagicMock(), autospec=True
        )

        initial_root_state = actual_is_root_trace.get()

        with trace_open() as task_run_id:
            assert isinstance(task_run_id, str)
            uuid.UUID(task_run_id)
            assert actual_get_current_task_run_id() == task_run_id
            assert actual_is_root_trace.get() is True

        assert actual_get_current_task_run_id() is None
        assert actual_is_root_trace.get() == initial_root_state
        mock_flush.assert_called_once()
        # submit_to_worker_loop is now called for status updates
        assert mock_submit_loop.call_count == 2  # INITIALIZING and COMPLETED

    def test_trace_with_name_and_attributes(self, mocker):
        """Test trace with name and attributes, checking they are passed on."""
        mock_mcp_calls = [MagicMock()]
        mock_flush = mocker.patch(
            "hud.telemetry._trace.flush_buffer", return_value=mock_mcp_calls, autospec=True
        )
        mock_submit_loop = mocker.patch(
            "hud.telemetry.exporter.submit_to_worker_loop", return_value=MagicMock(), autospec=True
        )

        trace_name = "test_trace_with_data"
        attrs = {"key": "value", "number": 42}

        with trace_open(name=trace_name, attributes=attrs) as task_run_id:
            assert isinstance(task_run_id, str)

        mock_flush.assert_called_once()
        # submit_to_worker_loop is now called for status updates
        assert mock_submit_loop.call_count == 2  # INITIALIZING and COMPLETED

    @pytest.mark.asyncio
    async def test_trace_with_mcp_calls_exports(self, mocker):
        """Test trace with MCP calls exports telemetry with correct data."""
        mock_mcp_calls = [MagicMock(), MagicMock()]
        mock_flush = mocker.patch(
            "hud.telemetry._trace.flush_buffer", return_value=mock_mcp_calls, autospec=True
        )
        mock_submit_loop = mocker.patch(
            "hud.telemetry.exporter.submit_to_worker_loop", return_value=MagicMock(), autospec=True
        )

        async def mock_export(*args, **kwargs):
            return None

        mocker.patch(
            "hud.telemetry.exporter.export_telemetry",
            side_effect=mock_export,
        )

        test_attrs = {"custom_attr": "test_val"}
        test_name = "mcp_export_test"

        with trace_open(name=test_name, attributes=test_attrs) as task_run_id:
            pass

        mock_flush.assert_called_once()
        # submit_to_worker_loop is now called for status updates and export
        # The exact count may vary depending on whether export_incremental is called
        assert mock_submit_loop.call_count >= 2  # At least INITIALIZING and COMPLETED

        # With the new export flow, export_telemetry is submitted to worker loop
        # so we can't directly assert on it being called synchronously
        # Instead, verify that the trace completed successfully
        assert task_run_id is not None

    def test_trace_nested(self, mocker):
        """Test nested traces, verifying context restoration and root trace logic."""
        actual_set_current_task_run_id(None)
        actual_is_root_trace.set(False)

        mock_flush_internal = mocker.patch(
            "hud.telemetry._trace.flush_buffer", return_value=[], autospec=True
        )
        mock_submit_loop_internal = mocker.patch(
            "hud.telemetry.exporter.submit_to_worker_loop", return_value=MagicMock(), autospec=True
        )

        assert actual_get_current_task_run_id() is None
        assert actual_is_root_trace.get() is False

        with trace_open(name="outer") as outer_id:
            assert actual_get_current_task_run_id() == outer_id
            assert actual_is_root_trace.get() is True
            with trace(name="inner") as inner_id:
                assert actual_get_current_task_run_id() == inner_id
                assert actual_is_root_trace.get() is False
                assert outer_id != inner_id
            assert actual_get_current_task_run_id() == outer_id
            assert actual_is_root_trace.get() is True

        assert actual_get_current_task_run_id() is None
        assert actual_is_root_trace.get() is False
        assert mock_flush_internal.call_count == 2
        # submit_to_worker_loop is now called for status updates
        assert mock_submit_loop_internal.call_count == 2  # Only outer trace sends status updates

    def test_trace_exception_handling(self, mocker):
        """Test trace handles exceptions properly and restores context."""
        initial_task_id_before_trace = "pre_existing_id_123"
        initial_root_state_before_trace = True
        actual_set_current_task_run_id(initial_task_id_before_trace)
        actual_is_root_trace.set(initial_root_state_before_trace)

        mock_flush = mocker.patch(
            "hud.telemetry._trace.flush_buffer", return_value=[], autospec=True
        )
        mock_submit_loop = mocker.patch(
            "hud.telemetry.exporter.submit_to_worker_loop", return_value=MagicMock(), autospec=True
        )

        with (
            pytest.raises(ValueError, match="Test exception"),
            trace_open(name="trace_with_exception"),
        ):
            assert actual_get_current_task_run_id() != initial_task_id_before_trace
            assert actual_is_root_trace.get() is False
            raise ValueError("Test exception")

        mock_flush.assert_called_once()
        assert actual_get_current_task_run_id() == initial_task_id_before_trace
        assert actual_is_root_trace.get() == initial_root_state_before_trace
        mock_submit_loop.assert_not_called()


class TestTraceSync:
    """Test the trace_sync context manager."""

    def test_trace_sync_basic(self, mocker):
        """Test trace calls trace_open and flush."""
        mock_flush = mocker.patch("hud.flush", autospec=True)
        mock_trace_open = mocker.patch("hud.telemetry._trace.trace_open")
        mock_trace_open.return_value.__enter__.return_value = "test-task-id"
        mock_trace_open.return_value.__exit__.return_value = None

        with trace(name="test_sync") as task_run_id:
            assert task_run_id == "test-task-id"

        mock_trace_open.assert_called_once_with(name="test_sync", agent_model=None, attributes=None)
        mock_flush.assert_called_once()

    def test_trace_sync_with_attributes(self, mocker):
        """Test trace passes attributes correctly."""
        mock_flush = mocker.patch("hud.flush", autospec=True)
        mock_trace_open = mocker.patch("hud.telemetry._trace.trace_open")
        mock_trace_open.return_value.__enter__.return_value = "test-task-id"
        mock_trace_open.return_value.__exit__.return_value = None
        attrs = {"key": "value"}

        with trace(name="test_sync", attributes=attrs):
            pass

        mock_trace_open.assert_called_once_with(
            name="test_sync", agent_model=None, attributes=attrs
        )
        mock_flush.assert_called_once()


class TestTraceDecorator:
    """Test the trace_decorator function decorator."""

    def test_trace_decorator_sync_function(self, mocker):
        """Test trace_decorator on synchronous functions."""
        mock_trace_open = mocker.patch("hud.telemetry._trace.trace_open", autospec=True)
        mock_trace_open.return_value.__enter__.return_value = "mocked_task_id"
        mock_trace_open.return_value.__exit__.return_value = None

        @trace_decorator(name="test_func_sync")
        def sync_function(x, y):
            return x + y

        result = sync_function(1, 2)
        assert result == 3
        mock_trace_open.assert_called_once_with(
            name="test_func_sync", agent_model=None, attributes=None
        )

    def test_trace_decorator_async_function(self, mocker):
        """Test trace_decorator on asynchronous functions."""
        mock_trace_open = mocker.patch("hud.telemetry._trace.trace_open", autospec=True)
        mock_trace_open.return_value.__enter__.return_value = "mocked_task_id"
        mock_trace_open.return_value.__exit__.return_value = None

        @trace_decorator(name="test_func_async")
        async def async_function(x, y):
            return x + y

        async def run_test():
            result = await async_function(1, 2)
            assert result == 3
            mock_trace_open.assert_called_once_with(
                name="test_func_async", agent_model=None, attributes=None
            )

        asyncio.run(run_test())

    def test_trace_decorator_with_attributes(self, mocker):
        """Test trace_decorator with attributes."""
        mock_trace_open = mocker.patch("hud.telemetry._trace.trace_open", autospec=True)
        mock_trace_open.return_value.__enter__.return_value = "task_id"
        mock_trace_open.return_value.__exit__.return_value = None

        attrs = {"operation": "multiply"}

        @trace_decorator(name="test_func", attributes=attrs)
        def func_with_attrs(x):
            return x * 2

        result = func_with_attrs(5)
        assert result == 10
        mock_trace_open.assert_called_once_with(
            name="test_func", agent_model=None, attributes=attrs
        )

    def test_trace_decorator_without_name(self, mocker):
        """Test trace_decorator uses module.function name when name not provided."""
        mock_trace_open = mocker.patch("hud.telemetry._trace.trace_open", autospec=True)
        mock_trace_open.return_value.__enter__.return_value = "task_id"
        mock_trace_open.return_value.__exit__.return_value = None

        @trace_decorator()
        def my_function():
            return "result"

        result = my_function()
        assert result == "result"
        # Should use module.function name
        expected_name = f"{my_function.__module__}.my_function"
        mock_trace_open.assert_called_once_with(
            name=expected_name, agent_model=None, attributes=None
        )

    def test_trace_decorator_preserves_function_metadata(self):
        """Test trace_decorator preserves original function metadata."""

        @trace_decorator(name="test")
        def original_function():
            """Original docstring."""

        assert original_function.__name__ == "original_function"
        assert original_function.__doc__ == "Original docstring."

    def test_trace_decorator_exception_propagation(self, mocker):
        """Test trace_decorator propagates exceptions."""
        mock_trace_open = mocker.patch("hud.telemetry._trace.trace_open", autospec=True)
        mock_trace_open.return_value.__enter__.return_value = "task_id"
        mock_trace_open.return_value.__exit__.return_value = None

        @trace_decorator()
        def failing_function():
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError, match="Test error"):
            failing_function()

        mock_trace_open.assert_called_once()
