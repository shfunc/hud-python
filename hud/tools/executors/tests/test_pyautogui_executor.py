"""Tests for PyAutoGUIExecutor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hud.tools.base import ToolResult
from hud.tools.executors.pyautogui import PyAutoGUIExecutor


class TestPyAutoGUIExecutor:
    """Test PyAutoGUIExecutor methods."""

    def test_is_available(self):
        """Test is_available method."""
        # Test when pyautogui can be imported
        with patch.dict("sys.modules", {"pyautogui": MagicMock()}):
            assert PyAutoGUIExecutor.is_available() is True

        # Test when pyautogui cannot be imported
        with (
            patch.dict("sys.modules", {"pyautogui": None}),
            patch("builtins.__import__", side_effect=ImportError),
        ):
            assert PyAutoGUIExecutor.is_available() is False

    @pytest.mark.asyncio
    async def test_screenshot_no_pyautogui(self):
        """Test screenshot when pyautogui is not available."""
        with patch.object(
            PyAutoGUIExecutor,
            "_ensure_pyautogui",
            side_effect=ImportError("No module named 'pyautogui'"),
        ):
            executor = PyAutoGUIExecutor()
            result = await executor.screenshot()

            assert isinstance(result, ToolResult)
            assert result.error == "PyAutoGUI not available: No module named 'pyautogui'"
            assert result.output is None
            assert result.base64_image is None

    @pytest.mark.asyncio
    async def test_click_no_pyautogui(self):
        """Test click when pyautogui is not available."""
        with patch.object(
            PyAutoGUIExecutor,
            "_ensure_pyautogui",
            side_effect=ImportError("No module named 'pyautogui'"),
        ):
            executor = PyAutoGUIExecutor()
            result = await executor.click(x=100, y=100)

            assert isinstance(result, ToolResult)
            assert result.error == "PyAutoGUI not available: No module named 'pyautogui'"

    @pytest.mark.asyncio
    async def test_type_text_no_pyautogui(self):
        """Test type when pyautogui is not available."""
        with patch.object(
            PyAutoGUIExecutor,
            "_ensure_pyautogui",
            side_effect=ImportError("No module named 'pyautogui'"),
        ):
            executor = PyAutoGUIExecutor()
            result = await executor.type(text="hello")

            assert isinstance(result, ToolResult)
            assert result.error == "PyAutoGUI not available: No module named 'pyautogui'"

    @pytest.mark.asyncio
    async def test_press_keys_no_pyautogui(self):
        """Test press when pyautogui is not available."""
        with patch.object(
            PyAutoGUIExecutor,
            "_ensure_pyautogui",
            side_effect=ImportError("No module named 'pyautogui'"),
        ):
            executor = PyAutoGUIExecutor()
            result = await executor.press(keys=["ctrl", "c"])

            assert isinstance(result, ToolResult)
            assert result.error == "PyAutoGUI not available: No module named 'pyautogui'"

    @pytest.mark.asyncio
    async def test_scroll_no_pyautogui(self):
        """Test scroll when pyautogui is not available."""
        with patch.object(
            PyAutoGUIExecutor,
            "_ensure_pyautogui",
            side_effect=ImportError("No module named 'pyautogui'"),
        ):
            executor = PyAutoGUIExecutor()
            result = await executor.scroll(x=100, y=100, scroll_y=5)

            assert isinstance(result, ToolResult)
            assert result.error == "PyAutoGUI not available: No module named 'pyautogui'"

    @pytest.mark.asyncio
    async def test_move_no_pyautogui(self):
        """Test move when pyautogui is not available."""
        with patch.object(
            PyAutoGUIExecutor,
            "_ensure_pyautogui",
            side_effect=ImportError("No module named 'pyautogui'"),
        ):
            executor = PyAutoGUIExecutor()
            result = await executor.move(x=100, y=100)

            assert isinstance(result, ToolResult)
            assert result.error == "PyAutoGUI not available: No module named 'pyautogui'"

    @pytest.mark.asyncio
    async def test_drag_no_pyautogui(self):
        """Test drag when pyautogui is not available."""
        with patch.object(
            PyAutoGUIExecutor,
            "_ensure_pyautogui",
            side_effect=ImportError("No module named 'pyautogui'"),
        ):
            executor = PyAutoGUIExecutor()
            result = await executor.drag(path=[(0, 0), (100, 100)])

            assert isinstance(result, ToolResult)
            assert result.error == "PyAutoGUI not available: No module named 'pyautogui'"

    @pytest.mark.asyncio
    async def test_wait(self):
        """Test wait method."""
        executor = PyAutoGUIExecutor()

        # Mock time.sleep
        with patch("asyncio.sleep") as mock_sleep:
            result = await executor.wait(time=100)

            assert isinstance(result, ToolResult)
            assert result.output == "Waited 100ms"
            mock_sleep.assert_called_once_with(0.1)  # 100ms = 0.1s

    @pytest.mark.asyncio
    async def test_position_no_pyautogui(self):
        """Test position when pyautogui is not available."""
        with patch.object(
            PyAutoGUIExecutor,
            "_ensure_pyautogui",
            side_effect=ImportError("No module named 'pyautogui'"),
        ):
            executor = PyAutoGUIExecutor()
            result = await executor.position()

            assert isinstance(result, ToolResult)
            assert result.error == "PyAutoGUI not available: No module named 'pyautogui'"

    def test_init_with_display_num(self):
        """Test initialization with display number."""
        # Should not raise
        executor = PyAutoGUIExecutor(display_num=0)
        assert hasattr(executor, "_pyautogui")
        assert executor._pyautogui is None  # Not loaded until needed
