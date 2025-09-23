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
from starlette.responses import JSONResponse, Response

from hud.server.low_level import LowLevelServerWithInit

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from starlette.requests import Request

__all__ = ["MCPServer"]

logger = logging.getLogger(__name__)

# Global flag to track if shutdown was triggered by SIGTERM
_sigterm_received = False


def _run_with_sigterm(coro_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Run *coro_fn* via anyio.run() and cancel on SIGTERM or SIGINT (POSIX)."""
    global _sigterm_received

    sys.stderr.flush()

    # Check if we're already in an event loop
    try:
        loop = asyncio.get_running_loop()
        logger.warning(
            "HUD server is running in an existing event loop. "
            "SIGTERM handling may be limited. "
            "Consider using await hub.run_async() instead of hub.run() in async contexts."
        )

        task = loop.create_task(coro_fn(*args, **kwargs))

        # Try to handle SIGTERM if possible
        if sys.platform != "win32":

            def handle_sigterm(signum: Any, frame: Any) -> None:
                logger.info("SIGTERM received in async context, cancelling task...")
                loop.call_soon_threadsafe(task.cancel)

            signal.signal(signal.SIGTERM, handle_sigterm)

        return

    except RuntimeError:
        pass

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

                    if (
                        self._shutdown_fn is not None
                        and _sigterm_received
                        and not self._shutdown_has_run
                    ):
                        logger.info("SIGTERM detected! Calling @mcp.shutdown handler...")
                        sys.stderr.flush()
                        try:
                            await self._shutdown_fn()
                            logger.info("@mcp.shutdown handler completed successfully.")
                            sys.stderr.flush()
                        except Exception as e:
                            logger.error("Error during @mcp.shutdown: %s", e)
                            sys.stderr.flush()
                        finally:
                            self._shutdown_has_run = True
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
        self._shutdown_has_run = False  # Guard against double-execution of shutdown hook

    def _replace_with_init_server(self) -> None:
        """Replace the low-level server with init version when needed."""
        if self._replaced_server:
            return

        async def _run_init(ctx: object | None = None) -> None:
            """Run the user initializer exactly once, with stdout redirected."""
            if self._initializer_fn is not None and not self._did_init:
                self._did_init = True
                # Prevent stdout from polluting the MCP protocol on stdio/HTTP
                with contextlib.redirect_stdout(sys.stderr):
                    import inspect

                    fn = self._initializer_fn
                    sig = inspect.signature(fn)
                    params = sig.parameters

                    ctx_param = params.get("ctx") or params.get("_ctx")
                    if ctx_param is not None:
                        if ctx_param.kind == inspect.Parameter.KEYWORD_ONLY:
                            result = fn(**{ctx_param.name: ctx})
                        else:
                            result = fn(ctx)
                    else:
                        required_params = [
                            p
                            for p in params.values()
                            if p.default is inspect._empty
                            and p.kind
                            in (
                                inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                inspect.Parameter.KEYWORD_ONLY,
                            )
                        ]
                        if required_params:
                            param_list = ", ".join(p.name for p in required_params)
                            raise TypeError(
                                "Initializer must accept no args or a single `ctx` argument; "
                                f"received required parameters: {param_list}"
                            )
                        result = fn()
                    if inspect.isawaitable(result):
                        await result
                    return None
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

    async def run_async(
        self,
        transport: Transport | None = None,
        show_banner: bool = True,
        **transport_kwargs: Any,
    ) -> None:
        """Run the server with HUD enhancements."""
        if transport is None:
            transport = "stdio"

        # Register HTTP helpers for HTTP transport
        if transport in ("http", "sse"):
            self._register_hud_helpers()
            logger.info("Registered HUD helper endpoints at /hud/*")

        try:
            await super().run_async(
                transport=transport, show_banner=show_banner, **transport_kwargs
            )
        finally:
            # Fallback: ensure SIGTERM-triggered shutdown runs even when a custom
            # lifespan bypasses our default fastmcp shutdown path.
            global _sigterm_received
            if self._shutdown_fn is not None and _sigterm_received and not self._shutdown_has_run:
                try:
                    await self._shutdown_fn()
                except Exception as e:  # pragma: no cover - defensive logging
                    logger.error("Error during @mcp.shutdown (fallback): %s", e)
                finally:
                    self._shutdown_has_run = True
                    _sigterm_received = False

    # Tool registration helper -- appends BaseTool to FastMCP
    def add_tool(self, obj: Any, **kwargs: Any) -> None:
        from hud.tools.base import BaseTool

        if isinstance(obj, BaseTool):
            super().add_tool(obj.mcp, **kwargs)
            return

        super().add_tool(obj, **kwargs)

    # Override to keep original callables when used as a decorator
    def tool(self, name_or_fn: Any = None, **kwargs: Any) -> Any:  # type: ignore[override]
        """Register a tool but return the original function in decorator form.

        - Decorator usage (@mcp.tool, @mcp.tool("name"), @mcp.tool(name="name"))
          registers with FastMCP and returns the original function for composition.
        - Call-form (mcp.tool(fn, ...)) behaves the same but returns fn.
        """
        # Accept BaseTool / FastMCP Tool instances or callables in call-form
        if name_or_fn is not None and not isinstance(name_or_fn, str):
            try:
                from hud.tools.base import BaseTool  # lazy import
            except Exception:
                BaseTool = tuple()  # type: ignore[assignment]
            try:
                from fastmcp.tools.tool import Tool as _FastMcpTool
            except Exception:
                _FastMcpTool = tuple()  # type: ignore[assignment]

            # BaseTool instance → add underlying FunctionTool
            if isinstance(name_or_fn, BaseTool):
                super().add_tool(name_or_fn.mcp, **kwargs)
                return name_or_fn
            # FastMCP Tool/FunctionTool instance → add directly
            if isinstance(name_or_fn, _FastMcpTool):
                super().add_tool(name_or_fn, **kwargs)
                return name_or_fn
            # Callable function → register via FastMCP.tool and return original fn
            if callable(name_or_fn):
                super().tool(name_or_fn, **kwargs)
                return name_or_fn

        # Decorator form: get FastMCP's decorator, register, then return original fn
        base_decorator = super().tool(name_or_fn, **kwargs)

        def _wrapper(fn: Any) -> Any:
            base_decorator(fn)
            return fn

        return _wrapper

    def _register_hud_helpers(self) -> None:
        """Register HUD helper HTTP routes.

        This adds:
        - GET /hud - Overview of available endpoints
        - GET /hud/tools - List all registered tools with their schemas
        - GET /hud/resources - List all registered resources
        - GET /hud/prompts - List all registered prompts
        """

        @self.custom_route("/hud/tools", methods=["GET"])
        async def list_tools(request: Request) -> Response:
            """List all registered tools with their names, descriptions, and schemas."""
            tools = []
            # _tools is a mapping of tool_name -> FunctionTool/Tool instance
            for tool_key, tool in self._tool_manager._tools.items():
                tool_data = {"name": tool_key}
                try:
                    # Prefer converting to MCP model for consistent fields
                    mcp_tool = tool.to_mcp_tool()
                    tool_data["description"] = getattr(mcp_tool, "description", "")
                    if hasattr(mcp_tool, "inputSchema") and mcp_tool.inputSchema:
                        tool_data["input_schema"] = mcp_tool.inputSchema  # type: ignore[assignment]
                    if hasattr(mcp_tool, "outputSchema") and mcp_tool.outputSchema:
                        tool_data["output_schema"] = mcp_tool.outputSchema  # type: ignore[assignment]
                except Exception:
                    # Fallback to direct attributes on FunctionTool
                    tool_data["description"] = getattr(tool, "description", "")
                    params = getattr(tool, "parameters", None)
                    if params:
                        tool_data["input_schema"] = params
                tools.append(tool_data)

            return JSONResponse({"server": self.name, "tools": tools, "count": len(tools)})

        @self.custom_route("/hud/resources", methods=["GET"])
        async def list_resources(request: Request) -> Response:
            """List all registered resources."""
            resources = []
            for resource_key, resource in self._resource_manager._resources.items():
                resource_data = {
                    "uri": resource_key,
                    "name": resource.name,
                    "description": resource.description,
                    "mimeType": resource.mime_type,
                }
                resources.append(resource_data)

            return JSONResponse(
                {"server": self.name, "resources": resources, "count": len(resources)}
            )

        @self.custom_route("/hud/prompts", methods=["GET"])
        async def list_prompts(request: Request) -> Response:
            """List all registered prompts."""
            prompts = []
            for prompt_key, prompt in self._prompt_manager._prompts.items():
                prompt_data = {
                    "name": prompt_key,
                    "description": prompt.description,
                }
                # Check if it has arguments
                if hasattr(prompt, "arguments") and prompt.arguments:
                    prompt_data["arguments"] = [
                        {"name": arg.name, "description": arg.description, "required": arg.required}
                        for arg in prompt.arguments
                    ]
                prompts.append(prompt_data)

            return JSONResponse({"server": self.name, "prompts": prompts, "count": len(prompts)})

        @self.custom_route("/hud", methods=["GET"])
        async def hud_info(request: Request) -> Response:
            """Show available HUD helper endpoints."""
            base_url = str(request.base_url).rstrip("/")
            return JSONResponse(
                {
                    "name": "HUD MCP Development Helpers",
                    "server": self.name,
                    "endpoints": {
                        "tools": f"{base_url}/hud/tools",
                        "resources": f"{base_url}/hud/resources",
                        "prompts": f"{base_url}/hud/prompts",
                    },
                    "description": "These endpoints help you inspect your MCP server during development.",  # noqa: E501
                }
            )
