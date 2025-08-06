"""MCP Client wrapper with automatic initialization and debugging capabilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from mcp.shared.exceptions import McpError
from mcp_use.client import MCPClient as MCPUseClient
from pydantic import AnyUrl

if TYPE_CHECKING:
    from typing import Self

    from mcp import types
    from mcp_use.session import MCPSession as MCPUseSession

logger = logging.getLogger(__name__)


class MCPClient:
    """
    High-level MCP client wrapper that handles initialization, tool discovery,
    and provides debugging capabilities.
    """

    def __init__(
        self,
        mcp_config: dict[str, dict[str, Any]],
        verbose: bool = False,
        client_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the MCP client.

        Args:
            mcp_config: MCP server configuration dict (required)
            verbose: Enable verbose logging of server communications
            client_info: Optional client info for MCP initialization
        """
        self.verbose = verbose
        self.client_info = client_info or {"name": "hud-python", "version": "3.0.3"}

        # Create a logging callback that stores notifications
        async def logging_callback(params: types.LoggingMessage) -> None:
            """Handle logging notifications from MCP servers."""
            # Note: We don't have server name here, but we can work around this
            notification_data = {
                "timestamp": datetime.now().isoformat(),
                "method": "notifications/message",
                "params": {
                    "level": params.level,
                    "logger": getattr(params, "logger", None),
                    "data": params.data,
                },
            }
            self._notifications.append(notification_data)

            if self.verbose:
                logger.debug(
                    "Received notification: level=%s, logger=%s, data=%s",
                    params.level,
                    getattr(params, "logger", None),
                    params.data,
                )

        # Initialize mcp_use client with proper config and logging callback
        config = {"mcpServers": mcp_config}
        self._mcp_client = MCPUseClient.from_dict(config, logging_callback=logging_callback)

        self._sessions: dict[str, MCPUseSession] = {}
        self._available_tools: list[types.Tool] = []
        self._tool_map: dict[str, tuple[str, types.Tool]] = {}
        self._telemetry_data: dict[str, Any] = {}
        self._notifications: list[dict[str, Any]] = []  # Store received notifications

        # Set up verbose logging if requested
        if self.verbose:
            self._setup_verbose_logging()

    def _setup_verbose_logging(self) -> None:
        """Configure verbose logging for debugging."""
        # Set MCP-related loggers to DEBUG
        logging.getLogger("mcp").setLevel(logging.DEBUG)
        logging.getLogger("mcp_use").setLevel(logging.DEBUG)
        logging.getLogger("mcp.client.stdio").setLevel(logging.DEBUG)

        # Add handler for server communications
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

    async def initialize(self) -> None:
        """Perform async initialization tasks."""
        await self.create_sessions()
        await self.discover_tools()
        await self.fetch_telemetry()

    async def create_sessions(self) -> dict[str, MCPUseSession]:
        # Create all sessions at once
        try:
            self._sessions = await self._mcp_client.create_all_sessions()
        except McpError as e:
            # Protocol error - the server is reachable but rejecting our request
            logger.error("MCP protocol error: %s", e)
            logger.error("This typically means:")
            logger.error("- Invalid or missing initialization parameters")
            logger.error("- Incompatible protocol version")
            logger.error("- Server-side configuration issues")
            raise
        except Exception as e:
            # Transport or other errors
            logger.error("Failed to create sessions: %s", e)
            if self.verbose:
                logger.info("Check that the MCP server is running and accessible")
            raise

        # Log session details in verbose mode
        if self.verbose and self._sessions:
            for name, session in self._sessions.items():
                logger.debug("  - %s: %s", name, type(session).__name__)

        # Automatically set log level to info for all servers
        try:
            await self.set_log_level("info")
        except Exception as e:
            if self.verbose:
                logger.debug("Could not set default log level: %s", e)

        return self._sessions

    async def discover_tools(self) -> list[types.Tool]:
        """Discover all available tools from connected servers."""
        logger.info("Discovering available tools...")

        self._available_tools = []
        self._tool_map = {}

        for server_name, session in self._sessions.items():
            try:
                # Ensure session is initialized
                if not hasattr(session, "connector") or not hasattr(
                    session.connector, "client_session"
                ):
                    await session.initialize()

                if session.connector.client_session is None:
                    logger.warning("Client session not initialized for %s", server_name)
                    continue

                # List tools
                tools_result = await session.connector.client_session.list_tools()

                logger.info(
                    "Discovered %d tools from '%s': %s",
                    len(tools_result.tools),
                    server_name,
                    [tool.name for tool in tools_result.tools],
                )

                # Add to collections
                for tool in tools_result.tools:
                    self._available_tools.append(tool)
                    self._tool_map[tool.name] = (server_name, tool)

                # Log detailed tool info in verbose mode
                if self.verbose:
                    for tool in tools_result.tools:
                        description = tool.description or ""
                        logger.debug(
                            "  Tool '%s': %s",
                            tool.name,
                            description[:100] + "..." if len(description) > 100 else description,
                        )

            except Exception as e:
                logger.error("Error discovering tools from '%s': %s", server_name, e)
                if self.verbose:
                    logger.exception("Full error details:")

        logger.info("Total tools discovered: %d", len(self._available_tools))
        return self._available_tools

    async def fetch_telemetry(self) -> dict[str, Any]:
        """Fetch telemetry resource from all servers that provide it."""
        logger.info("Fetching telemetry resources...")

        for server_name, session in self._sessions.items():
            try:
                if not hasattr(session, "connector") or not hasattr(
                    session.connector, "client_session"
                ):
                    continue

                if session.connector.client_session is None:
                    continue

                # Try to read telemetry resource
                try:
                    result = await session.connector.client_session.read_resource(
                        AnyUrl("telemetry://live")
                    )
                    if result and result.contents and len(result.contents) > 0:
                        telemetry_data = json.loads(result.contents[0].text)  # type: ignore
                        self._telemetry_data[server_name] = telemetry_data

                        logger.info("📡 Telemetry data from server '%s':", server_name)
                        if "live_url" in telemetry_data:
                            logger.info("   🖥️  Live URL: %s", telemetry_data["live_url"])
                        if "status" in telemetry_data:
                            logger.info("   📊 Status: %s", telemetry_data["status"])
                        if "services" in telemetry_data:
                            logger.info("   📋 Services:")
                            for service, status in telemetry_data["services"].items():
                                status_icon = "✅" if status == "running" else "❌"
                                logger.info("      %s %s: %s", status_icon, service, status)

                        if self.verbose:
                            logger.debug(
                                "Full telemetry data:\n%s", json.dumps(telemetry_data, indent=2)
                            )

                except Exception as e:
                    # Resource might not exist, which is fine
                    if self.verbose:
                        logger.debug("No telemetry resource from '%s': %s", server_name, e)

            except Exception as e:
                logger.error("Error fetching telemetry from '%s': %s", server_name, e)

        return self._telemetry_data

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> types.CallToolResult:
        """
        Call a tool by name with the given arguments.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found
        """
        if tool_name not in self._tool_map:
            raise ValueError(f"Tool '{tool_name}' not found")

        server_name, tool = self._tool_map[tool_name]
        session = self._sessions[server_name]

        if self.verbose:
            logger.debug(
                "Calling tool '%s' on server '%s' with arguments: %s",
                tool_name,
                server_name,
                json.dumps(arguments, indent=2) if arguments else "None",
            )

        if session.connector.client_session is None:
            raise ValueError(f"Client session not initialized for {server_name}")

        result = await session.connector.client_session.call_tool(
            name=tool_name, arguments=arguments or {}
        )

        if self.verbose:
            logger.debug("Tool '%s' result: %s", tool_name, result)

        return result

    async def read_resource(self, uri: AnyUrl) -> types.ReadResourceResult | None:
        """
        Read a resource by URI from any server that provides it.

        Args:
            uri: Resource URI (e.g., "telemetry://live")

        Returns:
            Resource contents or None if not found
        """
        for server_name, session in self._sessions.items():
            try:
                if not hasattr(session, "connector") or not hasattr(
                    session.connector, "client_session"
                ):
                    continue

                if session.connector.client_session is None:
                    continue

                result = await session.connector.client_session.read_resource(uri)

                if self.verbose:
                    logger.debug(
                        "Successfully read resource '%s' from server '%s'", uri, server_name
                    )

                return result

            except Exception as e:
                if self.verbose:
                    logger.debug(
                        "Could not read resource '%s' from server '%s': %s", uri, server_name, e
                    )
                continue

        return None

    def get_available_tools(self) -> list[types.Tool]:
        """Get list of all available tools."""
        return self._available_tools

    def get_tool_map(self) -> dict[str, tuple[str, types.Tool]]:
        """Get mapping of tool names to (server_name, tool) tuples."""
        return self._tool_map

    def get_sessions(self) -> dict[str, MCPUseSession]:
        """Get active MCP sessions."""
        return self._sessions

    def get_telemetry_data(self) -> dict[str, Any]:
        """Get collected telemetry data from all servers."""
        return self._telemetry_data

    def get_all_active_sessions(self) -> dict[str, MCPUseSession]:
        """Get all active sessions (compatibility method)."""
        return self._sessions

    async def set_log_level(self, level: str, server_name: str | None = None) -> None:
        """
        Set the minimum log level for server notifications.

        Args:
            level: Log level (debug, info, notice, warning, error, critical, alert, emergency)
            server_name: Optional server name. If None, sets for all servers.
        """
        if server_name:
            if server_name not in self._sessions:
                raise ValueError(f"Server '{server_name}' not found")
            sessions = {server_name: self._sessions[server_name]}
        else:
            sessions = self._sessions

        for name, session in sessions.items():
            try:
                if session.connector.client_session:
                    await session.connector.client_session.send_request(
                        method="logging/setLevel", params={"level": level}
                    )
                    logger.info("Set log level to '%s' for server '%s'", level, name)
            except Exception as e:
                logger.error("Failed to set log level for '%s': %s", name, e)

    def get_notifications(self) -> list[dict[str, Any]]:
        """Get all received notifications."""
        return self._notifications.copy()

    def get_logs(self) -> str:
        """
        Get all notifications formatted as a log string.

        Returns:
            Formatted log string with timestamps, server names, and messages
        """
        logs = []
        for notif in self._notifications:
            timestamp = notif.get("timestamp", "")
            method = notif.get("method", "")
            params = notif.get("params", {})

            # Handle logging/message notifications
            if method == "notifications/message" and isinstance(params, dict):
                level = params.get("level", "info").upper()
                logger_name = params.get("logger", "")
                data = params.get("data", {})

                # Format the message
                message = f"[{timestamp}] [{level}]"
                if logger_name:
                    message += f" {logger_name}:"

                # Add data content
                if isinstance(data, dict):
                    # Try to get a message or error field first
                    if "message" in data:
                        message += f" {data['message']}"
                    elif "error" in data:
                        message += f" {data['error']}"
                    else:
                        # Format the entire data dict
                        message += f" {json.dumps(data)}"
                else:
                    message += f" {data}"

                logs.append(message)
            else:
                # Generic notification format
                logs.append(f"[{timestamp}] {method}: {json.dumps(params)}")

        return "\n".join(logs)

    async def close(self) -> None:
        """Close all active sessions."""
        await self._mcp_client.close_all_sessions()

        self._sessions = {}
        self._available_tools = []
        self._tool_map = {}

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Async context manager exit."""
        await self.close()
