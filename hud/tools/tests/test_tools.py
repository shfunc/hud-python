from __future__ import annotations

import inspect

import pytest

from hud.tools.bash import BashTool
from hud.tools.computer.hud import HudComputerTool
from hud.tools.edit import EditTool
from hud.tools.helper import register_instance_tool


@pytest.mark.asyncio
async def test_bash_tool_echo():
    tool = BashTool()
    result = await tool(command="echo hello")
    assert result.output is not None
    assert "hello" in result.output


@pytest.mark.asyncio
async def test_edit_tool_view(tmp_path):
    # Create a temporary file
    p = tmp_path / "sample.txt"
    p.write_text("Sample content\n")

    tool = EditTool()
    result = await tool(command="view", path=str(p))
    assert result.output is not None
    assert "Sample content" in result.output


@pytest.mark.asyncio
async def test_computer_tool_screenshot():
    comp = HudComputerTool()
    blocks = await comp(action="screenshot")
    assert any(getattr(b, "data", None) for b in blocks)


def test_register_instance_tool_signature():
    """Helper should expose same user-facing parameters (no *args/**kwargs)."""

    class Dummy:
        async def __call__(self, *, x: int, y: str) -> str:
            return f"{x}-{y}"

    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("test")
    fn = register_instance_tool(mcp, "dummy", Dummy())
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    assert [p.name for p in params] == ["x", "y"], "*args/**kwargs should be stripped"
