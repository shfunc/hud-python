"""Tests for bash tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.tools.base import ToolResult
from hud.tools.bash import BashTool, ToolError, _BashSession


class TestBashSession:
    """Tests for _BashSession."""

    @pytest.mark.asyncio
    async def test_session_start(self):
        """Test starting a bash session."""
        session = _BashSession()
        assert session._started is False

        with patch("asyncio.create_subprocess_shell") as mock_create:
            mock_process = MagicMock()
            mock_create.return_value = mock_process

            await session.start()

            assert session._started is True
            assert session._process == mock_process
            mock_create.assert_called_once()

    def test_session_stop_not_started(self):
        """Test stopping a session that hasn't started."""
        session = _BashSession()

        with pytest.raises(ToolError) as exc_info:
            session.stop()

        assert "Session has not started" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_session_run_not_started(self):
        """Test running command on a session that hasn't started."""
        session = _BashSession()

        with pytest.raises(ToolError) as exc_info:
            await session.run("echo test")

        assert "Session has not started" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_session_run_success(self):
        """Test successful command execution."""
        session = _BashSession()
        session._started = True

        # Mock process
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readuntil = AsyncMock(return_value=b"Hello World\n<<exit>>\n")
        mock_process.stderr = MagicMock()
        mock_process.stderr.read = AsyncMock(return_value=b"")

        session._process = mock_process

        result = await session.run("echo Hello World")

        assert result.output == "Hello World\n"
        assert result.error == ""


class TestBashTool:
    """Tests for BashTool."""

    def test_bash_tool_init(self):
        """Test BashTool initialization."""
        tool = BashTool()
        assert tool._session is None

    @pytest.mark.asyncio
    async def test_call_with_command(self):
        """Test calling tool with a command."""
        tool = BashTool()

        # Mock session
        mock_session = MagicMock()
        mock_session.run = AsyncMock(return_value=ToolResult(output="test output"))

        # Mock _BashSession creation
        with patch("hud.tools.bash._BashSession") as mock_session_class:
            mock_session_class.return_value = mock_session
            mock_session.start = AsyncMock()

            result = await tool(command="echo test")

            assert isinstance(result, ToolResult)
            assert result.output == "test output"
            mock_session.start.assert_called_once()
            mock_session.run.assert_called_once_with("echo test")

    @pytest.mark.asyncio
    async def test_call_restart(self):
        """Test restarting the tool."""
        tool = BashTool()

        # Set up existing session
        old_session = MagicMock()
        old_session.stop = MagicMock()
        tool._session = old_session

        # Mock new session
        new_session = MagicMock()
        new_session.start = AsyncMock()

        with patch("hud.tools.bash._BashSession", return_value=new_session):
            result = await tool(restart=True)

            assert isinstance(result, ToolResult)
            assert result.system == "tool has been restarted."
            old_session.stop.assert_called_once()
            new_session.start.assert_called_once()
            assert tool._session == new_session

    @pytest.mark.asyncio
    async def test_call_no_command_error(self):
        """Test calling without command raises error."""
        tool = BashTool()

        with pytest.raises(ToolError) as exc_info:
            await tool()

        assert "no command provided" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_with_existing_session(self):
        """Test calling with an existing session."""
        tool = BashTool()

        # Set up existing session
        existing_session = MagicMock()
        existing_session.run = AsyncMock(return_value=ToolResult(output="result"))
        tool._session = existing_session

        result = await tool(command="ls")

        assert isinstance(result, ToolResult)
        assert result.output == "result"
        existing_session.run.assert_called_once_with("ls")
