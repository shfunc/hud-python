"""HUD server helpers."""

from __future__ import annotations

import asyncio
import contextlib
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

# Global flag to track if shutdown was triggered by SIGTERM
_sigterm_received = False


def _run_with_sigterm(coro_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Run *coro_fn* via anyio.run() and cancel on SIGTERM or SIGINT (POSIX)."""
    global _sigterm_received

    sys.stderr.flush()

    async def _runner() -> None:
        stop_evt: asyncio.Event | None = None
        if sys.platform != "win32" and os.getenv("FASTMCP_DISABLE_SIGTERM_HANDLER") != "1":
            loop = asyncio.get_running_loop()
            stop_evt = asyncio.Event()

            # Handle SIGTERM for production shutdown
            def handle_sigterm() -> None:
                global _sigterm_received
                _sigterm_received = True
                logger.info("Received SIGTERM signal, setting shutdown flag")
                stop_evt.set()

            # Handle SIGINT for hot-reload
            def handle_sigint() -> None:
                logger.info("Received SIGINT signal, triggering hot reload...")
                # Don't set _sigterm_received for SIGINT
                stop_evt.set()

            # Handle both SIGTERM and SIGINT for graceful shutdown
            # In Docker containers, we always want to register our handlers
            try:
                loop.add_signal_handler(signal.SIGTERM, handle_sigterm)
                logger.info("SIGTERM handler registered")
            except (ValueError, OSError) as e:
                logger.warning("Could not register SIGTERM handler: %s", e)

            try:
                loop.add_signal_handler(signal.SIGINT, handle_sigint)
                logger.info("SIGINT handler registered")
            except (ValueError, OSError) as e:
                logger.warning("Could not register SIGINT handler: %s", e)

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(coro_fn, *args, **kwargs)

                if stop_evt is not None:

                    async def _watch() -> None:
                        logger.info("Signal handler ready, waiting for SIGTERM or SIGINT")
                        if stop_evt is not None:
                            await stop_evt.wait()
                        logger.info("Shutdown signal received, initiating graceful shutdown...")
                        tg.cancel_scope.cancel()

                    tg.start_soon(_watch)
        except* asyncio.CancelledError:
            # This ensures the task group cleans up properly
            logger.info("Task group cancelled, cleaning up...")

    anyio.run(_runner)


class MCPServer(FastMCP):
    """FastMCP wrapper that adds helpful functionality for dockerized environments.
    This works with any MCP client, and adds just a few extra server-side features:
    1. SIGTERM handling for graceful shutdown in container runtimes.
       Note: SIGINT (Ctrl+C) is not handled, allowing normal hot reload behavior.
    2. ``@MCPServer.initialize`` decorator that registers an async initializer
       executed during the MCP *initialize* request. The initializer function receives
       a single ``ctx`` parameter (RequestContext) from which you can access:
       - ``ctx.session``: The MCP ServerSession
       - ``ctx.meta.progressToken``: Token for progress notifications (if provided)
       - ``ctx.session.client_params.clientInfo``: Client information
    3. ``@MCPServer.shutdown`` decorator that registers a coroutine to run during
       server teardown ONLY when SIGTERM is received (not on hot reload/SIGINT).
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
                global _sigterm_received
                try:
                    yield {}
                finally:
                    # Only call shutdown handler if SIGTERM was received
                    logger.info("Lifespan `finally` block reached. Checking for SIGTERM.")
                    # Force flush logs to ensure they're visible
                    sys.stderr.flush()

                    if self._shutdown_fn is not None and _sigterm_received:
                        logger.info("SIGTERM detected! Calling @mcp.shutdown handler...")
                        sys.stderr.flush()
                        try:
                            await self._shutdown_fn()
                            logger.info("@mcp.shutdown handler completed successfully.")
                            sys.stderr.flush()
                        except Exception as e:
                            logger.error("Error during @mcp.shutdown: %s", e)
                            sys.stderr.flush()
                        _sigterm_received = False
                    elif self._shutdown_fn is not None:
                        logger.info(
                            "No SIGTERM. This is a hot reload (SIGINT) or normal exit. Skipping @mcp.shutdown handler."  # noqa: E501
                        )
                        sys.stderr.flush()
                    else:
                        logger.info("No shutdown handler registered.")
                        sys.stderr.flush()

            fastmcp_kwargs["lifespan"] = _lifespan

        super().__init__(name=name, **fastmcp_kwargs)
        self._initializer_fn: Callable | None = None
        self._did_init = False
        self._replaced_server = False

    def _replace_with_init_server(self) -> None:
        """Replace the low-level server with init version when needed."""
        if self._replaced_server:
            return

        def _run_init(ctx: RequestContext | None = None) -> Any:
            if self._initializer_fn is not None and not self._did_init:
                self._did_init = True
                # Redirect stdout to stderr during initialization to prevent
                # any library prints from corrupting the MCP protocol
                with contextlib.redirect_stdout(sys.stderr):
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
        self._replaced_server = True

    # Initializer decorator: runs on the initialize request
    # The decorated function receives a RequestContext object with access to:
    # - ctx.session: The MCP ServerSession
    # - ctx.meta.progressToken: Progress token (if provided by client)
    # - ctx.session.client_params.clientInfo: Client information
    def initialize(self, fn: Callable | None = None) -> Callable | None:
        def decorator(func: Callable) -> Callable:
            self._initializer_fn = func
            # Only replace server when there's actually an init handler
            self._replace_with_init_server()
            return func

        return decorator(fn) if fn else decorator

    # Shutdown decorator: runs after server stops
    # Supports dockerized SIGTERM handling
    def shutdown(self, fn: Callable | None = None) -> Callable | None:
        """Register a shutdown handler that runs ONLY on SIGTERM.

        This handler will be called when the server receives a SIGTERM signal
        (e.g., during container shutdown). It will NOT be called on:
        - SIGINT (Ctrl+C or hot reload)
        - Normal client disconnects
        - Other graceful shutdowns

        This ensures that persistent resources (like browser sessions) are only
        cleaned up during actual termination, not during development hot reloads.
        """

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
