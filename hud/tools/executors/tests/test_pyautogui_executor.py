"""Tests for PyAutoGUI executor."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from hud.tools.executors.pyautogui import PyAutoGUIExecutor, PYAUTOGUI_AVAILABLE
from hud.tools.base import ToolResult


class TestPyAutoGUIExecutor:
    """Tests for PyAutoGUIExecutor."""

    def test_is_available(self):
        """Test is_available method."""
        # The availability is determined by the module-level PYAUTOGUI_AVAILABLE
        assert PyAutoGUIExecutor.is_available() == PYAUTOGUI_AVAILABLE

    @pytest.mark.skipif(not PYAUTOGUI_AVAILABLE, reason="pyautogui not available")
    @pytest.mark.asyncio
    async def test_screenshot_with_pyautogui(self):
        """Test screenshot when pyautogui is available."""
        executor = PyAutoGUIExecutor()
        
        # Mock pyautogui screenshot
        with patch("pyautogui.screenshot") as mock_screenshot:
            mock_img = MagicMock()
            mock_img.save = MagicMock()
            mock_screenshot.return_value = mock_img
            
            result = await executor.screenshot()
            
            # screenshot() returns a base64 string, not a ToolResult
            assert isinstance(result, str)
            mock_screenshot.assert_called_once()

    @pytest.mark.skipif(not PYAUTOGUI_AVAILABLE, reason="pyautogui not available")
    @pytest.mark.asyncio
    async def test_click_with_pyautogui(self):
        """Test click when pyautogui is available."""
        executor = PyAutoGUIExecutor()
        
        with patch("pyautogui.click") as mock_click:
            result = await executor.click(100, 200, "left")
            
            assert isinstance(result, ToolResult)
            assert "Clicked" in result.output
            mock_click.assert_called_once_with(x=100, y=200, button="left")

    @pytest.mark.skipif(not PYAUTOGUI_AVAILABLE, reason="pyautogui not available")
    @pytest.mark.asyncio
    async def test_type_text_with_pyautogui(self):
        """Test type when pyautogui is available."""
        executor = PyAutoGUIExecutor()
        
        with patch("pyautogui.typewrite") as mock_type:
            result = await executor.type("Hello world")
            
            assert isinstance(result, ToolResult)
            assert "Typed" in result.output
            mock_type.assert_called_once_with("Hello world")

    @pytest.mark.skipif(not PYAUTOGUI_AVAILABLE, reason="pyautogui not available")
    @pytest.mark.asyncio
    async def test_press_keys_with_pyautogui(self):
        """Test press when pyautogui is available."""
        executor = PyAutoGUIExecutor()
        
        with patch("pyautogui.press") as mock_press:
            result = await executor.press(["ctrl", "a"])
            
            assert isinstance(result, ToolResult)
            assert "Pressed" in result.output
            mock_press.assert_called_once_with(["ctrl", "a"])

    @pytest.mark.skipif(not PYAUTOGUI_AVAILABLE, reason="pyautogui not available")
    @pytest.mark.asyncio
    async def test_scroll_with_pyautogui(self):
        """Test scroll when pyautogui is available."""
        executor = PyAutoGUIExecutor()
        
        with patch("pyautogui.scroll") as mock_scroll:
            result = await executor.scroll(100, 200, 5)
            
            assert isinstance(result, ToolResult)
            assert "Scrolled" in result.output
            mock_scroll.assert_called_once_with(5, x=100, y=200)

    @pytest.mark.skipif(not PYAUTOGUI_AVAILABLE, reason="pyautogui not available")
    @pytest.mark.asyncio
    async def test_move_with_pyautogui(self):
        """Test move when pyautogui is available."""
        executor = PyAutoGUIExecutor()
        
        with patch("pyautogui.moveTo") as mock_move:
            result = await executor.move(300, 400)
            
            assert isinstance(result, ToolResult)
            assert "Moved" in result.output
            mock_move.assert_called_once_with(300, 400)

    @pytest.mark.skipif(not PYAUTOGUI_AVAILABLE, reason="pyautogui not available")
    @pytest.mark.asyncio
    async def test_drag_with_pyautogui(self):
        """Test drag when pyautogui is available."""
        executor = PyAutoGUIExecutor()
        
        with patch("pyautogui.dragTo") as mock_drag:
            # drag method signature: drag(x: int, y: int, duration: float = 1.0, button: str = "left")
            result = await executor.drag(300, 400, 1.0)
            
            assert isinstance(result, ToolResult)
            assert "Dragged" in result.output
            mock_drag.assert_called_once_with(300, 400, duration=1.0, button="left")

    @pytest.mark.asyncio
    async def test_wait(self):
        """Test wait method."""
        executor = PyAutoGUIExecutor()
        
        # Mock asyncio.sleep
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await executor.wait(2.5)
            
            assert isinstance(result, ToolResult)
            assert "Waited" in result.output
            mock_sleep.assert_called_once_with(2.5)

    @pytest.mark.skipif(not PYAUTOGUI_AVAILABLE, reason="pyautogui not available")
    @pytest.mark.asyncio
    async def test_position_with_pyautogui(self):
        """Test position when pyautogui is available."""
        executor = PyAutoGUIExecutor()
        
        with patch("pyautogui.position") as mock_position:
            mock_position.return_value = (123, 456)
            result = await executor.position()
            
            assert isinstance(result, ToolResult)
            assert "Mouse position" in result.output
            assert "123" in result.output
            assert "456" in result.output
            mock_position.assert_called_once()

    def test_init_with_display_num(self):
        """Test initialization with display number."""
        # Should not raise
        executor = PyAutoGUIExecutor(display_num=0)
        assert executor.display_num == 0