"""HUD server helpers."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import anyio
from fastmcp.server.server import FastMCP, Transport

from hud.server.low_level import LowLevelServerWithInit

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from mcp.shared.context import RequestContext

__all__ = ["MCPServer"]

logger = logging.getLogger(__name__)


def _run_with_sigterm(coro_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Run *coro_fn* via anyio.run() and cancel on SIGTERM or SIGINT (POSIX)."""

    async def _runner() -> None:
        stop_evt: asyncio.Event | None = None
        if sys.platform != "win32" and os.getenv("FASTMCP_DISABLE_SIGTERM_HANDLER") != "1":
            loop = asyncio.get_running_loop()
            stop_evt = asyncio.Event()

            # Handle both SIGTERM and SIGINT for graceful shutdown
            if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
                loop.add_signal_handler(signal.SIGTERM, stop_evt.set)
            if signal.getsignal(signal.SIGINT) is signal.SIG_DFL:
                loop.add_signal_handler(signal.SIGINT, stop_evt.set)

        async with anyio.create_task_group() as tg:
            tg.start_soon(coro_fn, *args, **kwargs)

            if stop_evt is not None:

                async def _watch() -> None:
                    logger.info("Waiting for SIGTERM or SIGINT")
                    if stop_evt is not None:
                        await stop_evt.wait()
                    logger.debug("Received shutdown signal, cancelling tasks...")
                    tg.cancel_scope.cancel()

                tg.start_soon(_watch)

    anyio.run(_runner)


class MCPServer(FastMCP):
    """FastMCP wrapper that adds helpful functionality for dockerized environments.
    This works with any MCP client, and adds just a few extra server-side features:
    1. SIGTERM handling for graceful shutdown in container runtimes.
    2. ``@MCPServer.initialize`` decorator that registers an async initializer
       executed during the MCP *initialize* request. The initializer function receives
       a single ``ctx`` parameter (RequestContext) from which you can access:
       - ``ctx.session``: The MCP ServerSession
       - ``ctx.meta.progressToken``: Token for progress notifications (if provided)
       - ``ctx.session.client_params.clientInfo``: Client information
    3. ``@MCPServer.shutdown`` decorator that registers a coroutine to run during
       server teardown, after all lifespan contexts have exited.
    4. Enhanced ``add_tool`` that accepts instances of
       :class:`hud.tools.base.BaseTool` which are classes that implement the
       FastMCP ``FunctionTool`` interface.
    """

    def __init__(self, *, name: str | None = None, **fastmcp_kwargs: Any) -> None:
        # Store shutdown function placeholder before super().__init__
        self._shutdown_fn: Callable | None = None

        # Inject custom lifespan if user did not supply one
        if "lifespan" not in fastmcp_kwargs:

            @asynccontextmanager
            async def _lifespan(_: Any) -> AsyncGenerator[dict[str, Any], None]:
                try:
                    yield {}
                finally:
                    if self._shutdown_fn is not None:
                        await self._shutdown_fn()

            fastmcp_kwargs["lifespan"] = _lifespan

        super().__init__(name=name, **fastmcp_kwargs)
        self._initializer_fn: Callable | None = None
        self._did_init = False

        # Replace FastMCP's low-level server with our version that supports
        # per-server initialization hooks
        def _run_init(ctx: RequestContext) -> Any:
            if self._initializer_fn is not None and not self._did_init:
                self._did_init = True
                return self._initializer_fn(ctx)
            return None

        # Save the old server's handlers before replacing it
        old_request_handlers = self._mcp_server.request_handlers
        old_notification_handlers = self._mcp_server.notification_handlers

        self._mcp_server = LowLevelServerWithInit(
            name=self.name,
            version=self.version,
            instructions=self.instructions,
            lifespan=self._mcp_server.lifespan,  # reuse the existing lifespan
            init_fn=_run_init,
        )

        # Copy handlers from the old server to the new one
        self._mcp_server.request_handlers = old_request_handlers
        self._mcp_server.notification_handlers = old_notification_handlers

    # Initializer decorator: runs on the initialize request
    # The decorated function receives a RequestContext object with access to:
    # - ctx.session: The MCP ServerSession
    # - ctx.meta.progressToken: Progress token (if provided by client)
    # - ctx.session.client_params.clientInfo: Client information
    def initialize(self, fn: Callable | None = None) -> Callable | None:
        def decorator(func: Callable) -> Callable:
            self._initializer_fn = func
            return func

        return decorator(fn) if fn else decorator

    # Shutdown decorator: runs after server stops
    # Supports dockerized SIGTERM handling
    def shutdown(self, fn: Callable | None = None) -> Callable | None:
        def decorator(func: Callable) -> Callable:
            self._shutdown_fn = func
            return func

        return decorator(fn) if fn else decorator

    # Run with SIGTERM handling and custom initialization
    def run(
        self,
        transport: Transport | None = None,
        show_banner: bool = True,
        **transport_kwargs: Any,
    ) -> None:
        if transport is None:
            transport = "stdio"

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
