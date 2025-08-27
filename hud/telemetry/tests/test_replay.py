"""Tests for telemetry replay functionality."""

from __future__ import annotations

from unittest.mock import patch

from hud.telemetry.replay import clear_trace, get_trace


class TestReplayAPI:
    """Tests for replay API functions."""

    def test_get_trace_calls_internal(self):
        """Test that get_trace calls the internal _get_trace function."""
        with patch("hud.telemetry.replay._get_trace") as mock_get:
            mock_get.return_value = None

            result = get_trace("test-task-id")

            mock_get.assert_called_once_with("test-task-id")
            assert result is None

    def test_clear_trace_calls_internal(self):
        """Test that clear_trace calls the internal _clear_trace function."""
        with patch("hud.telemetry.replay._clear_trace") as mock_clear:
            clear_trace("test-task-id")

            mock_clear.assert_called_once_with("test-task-id")

    def test_get_trace_with_data(self):
        """Test get_trace with mock data."""
        mock_trace = {"trace": [{"step": 1}], "task_run_id": "test-123"}

        with patch("hud.telemetry.replay._get_trace") as mock_get:
            mock_get.return_value = mock_trace

            result = get_trace("test-123")

            assert result == mock_trace
            mock_get.assert_called_once_with("test-123")
