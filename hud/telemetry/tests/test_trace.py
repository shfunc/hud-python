from __future__ import annotations

import asyncio
import uuid
from unittest.mock import MagicMock, patch

import pytest

from hud.telemetry.trace import (
    init_telemetry,
    register_trace,
    trace,
)


class TestInitTelemetry:
    """Test telemetry initialization."""

    @patch("hud.telemetry.instrumentation.registry.registry")
    def test_init_telemetry(self, mock_registry_obj):
        """Test telemetry initialization calls registry.install_all."""
        init_telemetry()
        mock_registry_obj.install_all.assert_called_once()


class TestTrace:
    """Test the trace context manager."""

    @patch("hud.telemetry.exporter.submit_to_worker_loop")
    @patch("hud.telemetry.context.flush_buffer")
    @patch("hud.telemetry.context.set_current_task_run_id")
    @patch("hud.telemetry.context.get_current_task_run_id")
    def test_trace_basic(
        self, mock_get_ctx_task_id, mock_set_ctx_task_id, mock_ctx_flush, mock_submit_loop
    ):
        """Test basic trace functionality."""
        mock_get_ctx_task_id.return_value = None
        mock_ctx_flush.return_value = []
        mock_submit_loop.return_value = MagicMock()

        with trace() as task_run_id:
            assert isinstance(task_run_id, str)
            uuid.UUID(task_run_id)

        mock_set_ctx_task_id.assert_called()
        mock_ctx_flush.assert_called_once()

    @patch("hud.telemetry.exporter.submit_to_worker_loop")
    @patch("hud.telemetry.context.flush_buffer")
    @patch("hud.telemetry.context.set_current_task_run_id")
    @patch("hud.telemetry.context.get_current_task_run_id")
    def test_trace_with_name(
        self, mock_get_ctx_task_id, mock_set_ctx_task_id, mock_ctx_flush, mock_submit_loop
    ):
        """Test trace with name parameter."""
        mock_get_ctx_task_id.return_value = None
        mock_ctx_flush.return_value = []
        mock_submit_loop.return_value = MagicMock()

        with trace(name="test_trace") as task_run_id:
            assert isinstance(task_run_id, str)

        mock_ctx_flush.assert_called_once()

    @patch("hud.telemetry.exporter.submit_to_worker_loop")
    @patch("hud.telemetry.context.flush_buffer")
    @patch("hud.telemetry.context.set_current_task_run_id")
    @patch("hud.telemetry.context.get_current_task_run_id")
    def test_trace_with_attributes(
        self, mock_get_ctx_task_id, mock_set_ctx_task_id, mock_ctx_flush, mock_submit_loop
    ):
        """Test trace with attributes."""
        mock_get_ctx_task_id.return_value = None
        mock_ctx_flush.return_value = []
        mock_submit_loop.return_value = MagicMock()

        attrs = {"key": "value", "number": 42}
        with trace(attributes=attrs) as task_run_id:
            assert isinstance(task_run_id, str)

        mock_ctx_flush.assert_called_once()

    @patch("hud.telemetry.exporter.submit_to_worker_loop")
    @patch("hud.telemetry.context.flush_buffer")
    @patch("hud.telemetry.context.set_current_task_run_id")
    @patch("hud.telemetry.context.get_current_task_run_id")
    @patch("hud.telemetry.exporter.export_telemetry")
    def test_trace_with_mcp_calls(
        self,
        mock_export_telemetry_coro,
        mock_get_ctx_task_id,
        mock_set_ctx_task_id,
        mock_ctx_flush,
        mock_submit_loop,
    ):
        """Test trace with MCP calls exports telemetry."""
        mock_get_ctx_task_id.return_value = None
        mock_mcp_calls = [MagicMock(), MagicMock()]
        mock_ctx_flush.return_value = mock_mcp_calls
        mock_submit_loop.return_value = MagicMock()

        with trace():
            pass

        mock_submit_loop.assert_called_once()
        mock_export_telemetry_coro.assert_called_once()

    @patch("hud.telemetry.exporter.submit_to_worker_loop")
    @patch("hud.telemetry.context.flush_buffer")
    @patch("hud.telemetry.context.set_current_task_run_id")
    @patch("hud.telemetry.context.get_current_task_run_id")
    def test_trace_nested(
        self, mock_get_ctx_task_id, mock_set_ctx_task_id, mock_ctx_flush, mock_submit_loop
    ):
        """Test nested traces."""
        mock_get_ctx_task_id.side_effect = [None, "fake_outer_id", "fake_inner_id", "fake_outer_id"]

        from hud.telemetry.context import get_current_task_run_id as actual_get_current_task_run_id
        from hud.telemetry.context import set_current_task_run_id as actual_set_current_task_run_id

        actual_set_current_task_run_id(None)

        mock_ctx_flush.return_value = []
        mock_submit_loop.return_value = MagicMock()

        with trace(name="outer") as outer_id:
            assert actual_get_current_task_run_id() == outer_id
            with trace(name="inner") as inner_id:
                assert actual_get_current_task_run_id() == inner_id
                assert outer_id != inner_id
            assert actual_get_current_task_run_id() == outer_id

        assert actual_get_current_task_run_id() is None
        assert mock_ctx_flush.call_count == 2

    @patch("hud.telemetry.exporter.submit_to_worker_loop")
    @patch("hud.telemetry.context.flush_buffer")
    @patch("hud.telemetry.context.set_current_task_run_id")
    @patch("hud.telemetry.context.get_current_task_run_id")
    def test_trace_exception_handling(
        self, mock_get_ctx_task_id, mock_set_ctx_task_id, mock_ctx_flush, mock_submit_loop
    ):
        """Test trace handles exceptions properly."""
        mock_get_ctx_task_id.return_value = None
        mock_set_ctx_task_id.return_value = None
        mock_ctx_flush.return_value = []
        mock_submit_loop.return_value = MagicMock()

        with pytest.raises(ValueError), trace():
            raise ValueError("Test exception")

        mock_ctx_flush.assert_called_once()


class TestRegisterTrace:
    """Test the register_trace decorator."""

    @patch("hud.telemetry.trace.trace")
    def test_register_trace_sync_function(self, mock_trace_context_manager):
        """Test register_trace with synchronous function."""
        mock_trace_context_manager.return_value.__enter__ = MagicMock(return_value="task_id")
        mock_trace_context_manager.return_value.__exit__ = MagicMock(return_value=None)

        @register_trace(name="test_func")
        def sync_function(x, y):
            return x + y

        result = sync_function(1, 2)
        assert result == 3
        mock_trace_context_manager.assert_called_once_with(name="test_func", attributes=None)

    @patch("hud.telemetry.trace.trace")
    def test_register_trace_async_function(self, mock_trace_context_manager):
        """Test register_trace with asynchronous function."""
        mock_trace_context_manager.return_value.__enter__ = MagicMock(return_value="task_id")
        mock_trace_context_manager.return_value.__exit__ = MagicMock(return_value=None)

        @register_trace(name="test_async")
        async def async_function(x, y):
            return x + y

        async def run_test():
            result = await async_function(1, 2)
            assert result == 3
            mock_trace_context_manager.assert_called_once_with(name="test_async", attributes=None)

        asyncio.run(run_test())

    @patch("hud.telemetry.trace.trace")
    def test_register_trace_with_attributes(self, mock_trace_context_manager):
        """Test register_trace with attributes."""
        mock_trace_context_manager.return_value.__enter__ = MagicMock(return_value="task_id")
        mock_trace_context_manager.return_value.__exit__ = MagicMock(return_value=None)

        attrs = {"operation": "add"}

        @register_trace(name="test_func", attributes=attrs)
        def func_with_attrs(x):
            return x * 2

        result = func_with_attrs(5)
        assert result == 10
        mock_trace_context_manager.assert_called_once_with(name="test_func", attributes=attrs)

    @patch("hud.telemetry.trace.trace")
    def test_register_trace_without_name(self, mock_trace_context_manager):
        """Test register_trace uses function name when name not provided."""
        mock_trace_context_manager.return_value.__enter__ = MagicMock(return_value="task_id")
        mock_trace_context_manager.return_value.__exit__ = MagicMock(return_value=None)

        @register_trace()
        def my_function():
            return "result"

        result = my_function()
        assert result == "result"
        mock_trace_context_manager.assert_called_once_with(name="my_function", attributes=None)

    def test_register_trace_preserves_function_metadata(self):
        """Test register_trace preserves original function metadata."""
        @register_trace(name="test")
        def original_function():
            """Original docstring."""
            pass

        assert original_function.__name__ == "original_function"
        assert original_function.__doc__ == "Original docstring."

    @patch("hud.telemetry.trace.trace")
    def test_register_trace_exception_propagation(self, mock_trace_context_manager):
        """Test register_trace propagates exceptions."""
        mock_trace_context_manager.return_value.__enter__ = MagicMock(return_value="task_id")
        mock_trace_context_manager.return_value.__exit__ = MagicMock(return_value=None)

        @register_trace()
        def failing_function():
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError, match="Test error"):
            failing_function()

        mock_trace_context_manager.assert_called_once()
