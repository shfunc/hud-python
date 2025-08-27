"""Base protocol and implementation for HUD MCP clients."""

from __future__ import annotations

import json
import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, overload, runtime_checkable

from mcp.types import Implementation

from hud.types import MCPToolCall, MCPToolResult
from hud.utils.mcp import setup_hud_telemetry
from hud.version import __version__ as hud_version

if TYPE_CHECKING:
    import mcp.types as types

else:
    pass


logger = logging.getLogger(__name__)


@runtime_checkable
class AgentMCPClient(Protocol):
    """Minimal interface for MCP clients used by agents.

    Any custom client must implement this interface.

    Any custom agent can assume that this will be the interaction protocol.
    """

    _initialized: bool

    @property
    def mcp_config(self) -> dict[str, dict[str, Any]]:
        """Get the MCP config."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if client is connected and initialized."""
        ...

    async def initialize(self, mcp_config: dict[str, dict[str, Any]] | None = None) -> None:
        """Initialize the client."""
        ...

    async def list_tools(self) -> list[types.Tool]:
        """List all available tools."""
        ...

    async def call_tool(self, tool_call: MCPToolCall) -> MCPToolResult:
        """Execute a tool by name."""
        ...

    async def shutdown(self) -> None:
        """Shutdown the client."""
        ...


class BaseHUDClient(AgentMCPClient):
    """Base class with common HUD functionality that adds:
    - Connection management
    - Tool discovery
    - Telemetry fetching (hud environment-specific)
    - Logging
    - Strict tool output validation (optional)
    - Environment analysis (optional)

    Any custom client should inherit from this class, and implement:
    - _connect: Connect to the MCP server
    - list_tools: List all available tools
    - list_resources: List all available resources
    - call_tool: Execute a tool by name
    - read_resource: Read a resource by URI
    - _disconnect: Disconnect from the MCP server
    - any other MCP client methods
    """

    client_info = Implementation(name="hud-mcp", title="hud MCP Client", version=hud_version)

    def __init__(
        self,
        mcp_config: dict[str, dict[str, Any]] | None = None,
        verbose: bool = False,
        strict_validation: bool = False,
        auto_trace: bool = True,
    ) -> None:
        """
        Initialize base client.

        Args:
            mcp_config: MCP server configuration dict
            verbose: Enable verbose logging
            strict_validation: Enable strict tool output validation
        """
        self.verbose = verbose
        self._mcp_config = mcp_config
        self._strict_validation = strict_validation
        self._auto_trace = auto_trace
        self._auto_trace_cm: Any | None = None  # Store auto-created trace context manager

        self._initialized = False
        self._telemetry_data = {}  # Initialize telemetry data

        if self.verbose:
            self._setup_verbose_logging()

    async def initialize(self, mcp_config: dict[str, dict[str, Any]] | None = None) -> None:
        """Initialize connection and fetch tools."""
        if self._initialized:
            logger.warning(
                "Client already connected, if you want to reconnect or change the configuration, "
                "call shutdown() first. This is especially important if you are using an agent."
            )
            return

        self._mcp_config = mcp_config or self._mcp_config
        if self._mcp_config is None:
            raise ValueError(
                "An MCP server configuration is required"
                "Either pass it to the constructor or call initialize with a configuration"
            )

        self._auto_trace_cm = setup_hud_telemetry(self._mcp_config, auto_trace=self._auto_trace)

        logger.debug("Initializing MCP client...")

        try:
            # Subclasses implement connection
            await self._connect(self._mcp_config)
        except RuntimeError as e:
            # Re-raise authentication errors with clear message
            if "Authentication failed" in str(e):
                raise
            raise
        except Exception as e:
            # Check for authentication errors in the exception chain
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                # Check if connecting to HUD API
                for server_config in self._mcp_config.values():
                    url = server_config.get("url", "")
                    if "mcp.hud.so" in url:
                        raise RuntimeError(
                            "Authentication failed for HUD API. "
                            "Please ensure your HUD_API_KEY environment variable is set correctly. "
                            "You can get an API key at https://app.hud.so"
                        ) from e
                raise RuntimeError(
                    "Authentication failed (401 Unauthorized). "
                    "Please check your credentials or API key."
                ) from e
            raise

        # Common hud behavior - fetch telemetry
        await self._fetch_telemetry()

        self._initialized = True
        logger.info("Client initialized")

    async def shutdown(self) -> None:
        """Disconnect from the MCP server."""
        # Clean up auto-created trace if any
        if self._auto_trace_cm:
            try:
                self._auto_trace_cm.__exit__(None, None, None)
                logger.info("Closed auto-created trace")
            except Exception as e:
                logger.warning("Failed to close auto-created trace: %s", e)
            finally:
                self._auto_trace_cm = None

        # Disconnect from server
        if self._initialized:
            await self._disconnect()
            self._initialized = False
            logger.info("Client disconnected")
        else:
            logger.warning("Client is not running, cannot disconnect")

    @overload
    async def call_tool(self, tool_call: MCPToolCall, /) -> MCPToolResult: ...
    @overload
    async def call_tool(
        self,
        *,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult: ...

    async def call_tool(
        self,
        tool_call: MCPToolCall | None = None,
        *,
        name: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:
        if tool_call is not None:
            return await self._call_tool(tool_call)
        elif name is not None:
            return await self._call_tool(MCPToolCall(name=name, arguments=arguments))
        else:
            raise TypeError(
                "call_tool() requires either an MCPToolCall positional arg "
                "or keyword 'name' (and optional 'arguments')."
            )

    @abstractmethod
    async def _connect(self, mcp_config: dict[str, dict[str, Any]]) -> None:
        """Subclasses implement their connection logic."""
        raise NotImplementedError

    @abstractmethod
    async def list_tools(self) -> list[types.Tool]:
        """List all available tools."""
        raise NotImplementedError

    @abstractmethod
    async def list_resources(self) -> list[types.Resource]:
        """List all available resources."""
        raise NotImplementedError

    @abstractmethod
    async def _call_tool(self, tool_call: MCPToolCall) -> MCPToolResult:
        """Execute a tool by name."""
        raise NotImplementedError

    @abstractmethod
    async def read_resource(self, uri: str) -> types.ReadResourceResult | None:
        """Read a resource by URI."""
        raise NotImplementedError

    @abstractmethod
    async def _disconnect(self) -> None:
        """Subclasses implement their disconnection logic."""
        raise NotImplementedError

    @property
    def is_connected(self) -> bool:
        """Check if client is connected and initialized."""
        return self._initialized

    @property
    def mcp_config(self) -> dict[str, dict[str, Any]]:
        """Get the MCP config."""
        if self._mcp_config is None:
            raise ValueError("Please initialize the client with a valid MCP config")
        return self._mcp_config

    async def __aenter__(self: Any) -> Any:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Async context manager exit."""
        await self.shutdown()

    def _setup_verbose_logging(self) -> None:
        """Configure verbose logging for debugging."""
        logging.getLogger("mcp").setLevel(logging.DEBUG)
        logging.getLogger("fastmcp").setLevel(logging.DEBUG)

        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

    async def _fetch_telemetry(self) -> None:
        """Common telemetry fetching for all hud clients."""
        try:
            # Try to read telemetry resource directly
            result = await self.read_resource("telemetry://live")
            if result and result.contents:
                # Parse telemetry data
                telemetry_data = json.loads(result.contents[0].text)  # type: ignore
                self._telemetry_data = telemetry_data

                logger.info("📡 Telemetry data fetched:")
                if "live_url" in telemetry_data:
                    logger.info("   🖥️  Live URL: %s", telemetry_data["live_url"])
                if "cdp_url" in telemetry_data:
                    logger.info("   🦾  CDP URL: %s", telemetry_data["cdp_url"])
                if "status" in telemetry_data:
                    logger.info("   📊 Status: %s", telemetry_data["status"])
                if "services" in telemetry_data:
                    logger.debug("   📋 Services:")
                    for service, status in telemetry_data["services"].items():
                        status_icon = "✅" if status == "running" else "❌"
                        logger.debug("      %s %s: %s", status_icon, service, status)

                if self.verbose:
                    logger.debug("Full telemetry data:\n%s", json.dumps(telemetry_data, indent=2))
        except Exception as e:
            # Telemetry is optional
            if self.verbose:
                logger.debug("No telemetry available: %s", e)

    async def analyze_environment(self) -> dict[str, Any]:
        """Complete analysis of the MCP environment.

        Returns:
            Dictionary containing:
            - tools: All tools with full schemas
            - hub_tools: Hub structures with subtools
            - telemetry: Telemetry resources and data
            - resources: All available resources
            - metadata: Environment metadata
        """
        if not self._initialized:
            raise ValueError("Client must be initialized before analyzing the environment")

        analysis: dict[str, Any] = {
            "tools": [],
            "hub_tools": {},
            "telemetry": self._telemetry_data,
            "resources": [],
            "metadata": {
                "servers": list(self._mcp_config.keys()),  # type: ignore
                "initialized": self._initialized,
            },
        }

        # Get all tools with schemas
        tools = await self.list_tools()
        for tool in tools:
            tool_info = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            analysis["tools"].append(tool_info)

            # Check if this is a hub tool (like setup, evaluate)
            if (
                tool.description
                and "internal" in tool.description.lower()
                and "functions" in tool.description.lower()
            ):
                # This is likely a hub dispatcher tool
                hub_functions = await self.get_hub_tools(tool.name)
                if hub_functions:
                    analysis["hub_tools"][tool.name] = hub_functions

        # Get all resources
        try:
            resources = await self.list_resources()
            for resource in resources:
                resource_info = {
                    "uri": str(resource.uri),
                    "name": resource.name,
                    "description": resource.description,
                    "mime_type": getattr(resource, "mimeType", None),
                }
                analysis["resources"].append(resource_info)
        except Exception as e:
            if self.verbose:
                logger.debug("Could not list resources: %s", e)

        return analysis

    async def get_hub_tools(self, hub_name: str) -> list[str]:
        """Get all subtools for a specific hub (setup/evaluate).

        Args:
            hub_name: Name of the hub (e.g., "setup", "evaluate")

        Returns:
            List of available function names for the hub
        """
        try:
            # Read the hub's functions catalogue resource
            result = await self.read_resource(f"file:///{hub_name}/functions")
            if result and result.contents:
                # Parse the JSON list of function names
                import json

                functions = json.loads(result.contents[0].text)  # type: ignore
                return functions
        except Exception as e:
            if self.verbose:
                logger.debug("Could not read hub functions for '%s': %s", hub_name, e)
        return []
