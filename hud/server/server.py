from __future__ import annotations

import asyncio
import os
import signal
import sys
from typing import TYPE_CHECKING, Any

import anyio
from fastmcp.server.server import FastMCP, Transport

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["HudServer"]

def _run_with_sigterm(coro_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Run *coro_fn* via anyio.run() and cancel on SIGTERM (POSIX)."""

    async def _runner() -> None:
        stop_evt: asyncio.Event | None = None
        if sys.platform != "win32" and os.getenv("FASTMCP_DISABLE_SIGTERM_HANDLER") != "1":
            if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
                loop = asyncio.get_running_loop()
                stop_evt = asyncio.Event()
                loop.add_signal_handler(signal.SIGTERM, stop_evt.set)

        async with anyio.create_task_group() as tg:
            tg.start_soon(coro_fn, *args, **kwargs)

            if stop_evt is not None:
                async def _watch() -> None:
                    await stop_evt.wait()
                    tg.cancel_scope.cancel()
                tg.start_soon(_watch)

    anyio.run(_runner)


class HudServer(FastMCP):
    """FastMCP wrapper with SIGTERM and convenient helpers."""

    def __init__(self, *, name: str | None = None, **fastmcp_kwargs: Any) -> None:
        super().__init__(name=name, **fastmcp_kwargs)
        self._initializer_fn: Callable | None = None

    # Initializer decorator
    def initialize(self, fn: Callable | None = None) -> Callable | None:
        def decorator(func: Callable) -> Callable:
            self._initializer_fn = func
            return func
        return decorator(fn) if fn else decorator

    # Run with optional initializer + SIGTERM
    def run(
        self,
        transport: Transport | None = None,
        show_banner: bool = True,
        **transport_kwargs: Any,
    ) -> None:
        if transport is None:
            transport = "stdio"

        if self._initializer_fn is not None:
            from hud.server.helper import mcp_intialize_wrapper
            mcp_intialize_wrapper(self._initializer_fn)

        async def _bootstrap() -> None:
            await self.run_async(transport=transport, show_banner=show_banner, **transport_kwargs)  # type: ignore[arg-type]

        _run_with_sigterm(_bootstrap)

    # Tool registration helper -- appends BaseTool to FastMCP
    def add_tool(self, obj: Any, **kwargs: Any) -> None:
        from hud.tools.base import BaseTool

        if isinstance(obj, BaseTool):
            super().add_tool(obj.mcp, **kwargs)
            return

        super().add_tool(obj, **kwargs)
