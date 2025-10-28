"""Base protocol and implementation for HUD MCP clients."""

from __future__ import annotations

import json
import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, overload, runtime_checkable

from mcp.types import Implementation

from hud.shared.exceptions import HudAuthenticationError, HudException
from hud.types import MCPToolCall, MCPToolResult
from hud.utils.hud_console import HUDConsole
from hud.utils.mcp import setup_hud_telemetry
from hud.version import __version__ as hud_version

if TYPE_CHECKING:
    import mcp.types as types
logger = logging.getLogger(__name__)

hud_console = HUDConsole(logger=logger)


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
            hud_console.warning(
                "Client already connected, if you want to reconnect or change the configuration, "
                "call shutdown() first. This is especially important if you are using an agent."
            )
            return

        self._mcp_config = mcp_config or self._mcp_config
        if self._mcp_config is None:
            from hud.shared.exceptions import HudConfigError

            raise HudConfigError(
                "An MCP server configuration is required. "
                "Either pass it to the constructor or call initialize with a configuration"
            )

        self._auto_trace_cm = setup_hud_telemetry(self._mcp_config, auto_trace=self._auto_trace)

        hud_console.debug("Initializing MCP client...")

        try:
            # Check if API key is set for HUD API
            for server_config in self._mcp_config.values():
                url = server_config.get("url", "")
                headers = server_config.get("headers", {})
                if "mcp.hud.so" in url and len(headers.get("Authorization", "")) < 10:
                    raise HudAuthenticationError(
                        f'Sending authorization "{headers.get("Authorization", "")}", which may'
                        " be incomplete. Ensure HUD_API_KEY environment variable is set or send it"
                        " as a header. You can get an API key at https://hud.so"
                    )
            # Subclasses implement connection
            await self._connect(self._mcp_config)
        except HudException:
            raise
        except Exception as e:
            # Auto-converts to appropriate HUD exception type with hints
            raise HudException from e

        # Common hud behavior - fetch telemetry
        await self._fetch_telemetry()

        self._initialized = True

    async def shutdown(self) -> None:
        """Disconnect from the MCP server."""
        # Clean up auto-created trace if any
        if self._auto_trace_cm:
            try:
                self._auto_trace_cm.__exit__(None, None, None)
                hud_console.info("Closed auto-created trace")
            except Exception as e:
                hud_console.warning(f"Failed to close auto-created trace: {e}")
            finally:
                self._auto_trace_cm = None

        # Disconnect from server
        if self._initialized:
            await self._disconnect()
            self._initialized = False
            hud_console.info("Environment Shutdown completed")
        else:
            hud_console.debug("Client was not initialized, skipping disconnect")

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
            from hud.shared.exceptions import HudConfigError

            raise HudConfigError("Please initialize the client with a valid MCP config")
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

                if "live_url" in telemetry_data:
                    hud_console.info(f"   🖥️  Live URL: {telemetry_data['live_url']}")
                if "vnc_url" in telemetry_data:
                    hud_console.info(f"   🖥️  VNC URL: {telemetry_data['vnc_url']}")
                if "cdp_url" in telemetry_data:
                    hud_console.info(f"   🦾  CDP URL: {telemetry_data['cdp_url']}")
                if "status" in telemetry_data:
                    hud_console.debug(f"   📊 Status: {telemetry_data['status']}")
                if "services" in telemetry_data:
                    hud_console.debug("   📋 Services:")
                    for service, status in telemetry_data["services"].items():
                        status_icon = "✅" if status == "running" else "❌"
                        hud_console.debug(f"      {status_icon} {service}: {status}")

                if self.verbose:
                    hud_console.debug(
                        f"Full telemetry data:\n{json.dumps(telemetry_data, indent=2)}"
                    )
        except Exception as e:
            # Telemetry is optional
            if self.verbose:
                hud_console.debug(f"No telemetry available: {e}")

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
            from hud.shared.exceptions import HudClientError

            raise HudClientError("Client must be initialized before analyzing the environment")

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

        resources: list[types.Resource] = []
        try:
            resources = await self.list_resources()
        except Exception as e:
            if self.verbose:
                hud_console.debug(f"Could not list resources: {e}")
            resources = []

        # Get all tools with schemas
        tools = await self.list_tools()
        for tool in tools:
            tool_info = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }

            # Check if this is a hub tool (like setup, evaluate)
            if (
                tool.description
                and "internal" in tool.description.lower()
                and "functions" in tool.description.lower()
            ):
                # This is likely a hub dispatcher tool
                hub_functions = await self.get_hub_tools(tool.name, resources=resources)
                if hub_functions:
                    analysis["hub_tools"][tool.name] = hub_functions
                    tool_info["internal_tools"] = hub_functions

            analysis["tools"].append(tool_info)

        # Store resource metadata after tool analysis (tools may need raw objects)
        for resource in resources:
            resource_info = {
                "uri": str(resource.uri),
                "name": resource.name,
                "description": resource.description,
                "mime_type": getattr(resource, "mimeType", None),
            }
            analysis["resources"].append(resource_info)

        return analysis

    async def get_hub_tools(
        self, hub_name: str, *, resources: list[types.Resource] | None = None
    ) -> list[str]:
        """Get all subtools for a specific hub (setup/evaluate).

        Args:
            hub_name: Name of the hub (e.g., "setup", "evaluate")
            resources: Optional cache of resources to help discover catalogue URIs

        Returns:
            List of available function names for the hub
        """
        candidate_uris: list[str] = [
            f"file:///{hub_name}/functions",
            f"{hub_name}://functions",
            f"{hub_name}://functions.json",
        ]

        if resources:
            for resource in resources:
                try:
                    uri = str(resource.uri)
                except Exception:
                    continue
                resource_name = (resource.name or "").lower()
                hub_lower = hub_name.lower()
                if not uri:
                    continue
                # Prioritize exact matches
                if resource_name == f"{hub_lower} functions":
                    candidate_uris.insert(0, uri)
                elif hub_lower in resource_name and "function" in resource_name:
                    candidate_uris.append(uri)
                elif hub_lower in uri.lower() and "function" in uri.lower():
                    candidate_uris.append(uri)

        seen: set[str] = set()
        for uri in candidate_uris:
            if not uri or uri in seen:
                continue
            seen.add(uri)

            try:
                result = await self.read_resource(uri)
            except Exception as e:  # pragma: no cover - verbose logging only
                if self.verbose:
                    hud_console.debug(f"Could not read hub resource '{uri}': {e}")
                continue

            if result and result.contents:
                try:
                    import json

                    functions = json.loads(result.contents[0].text)  # type: ignore
                except json.JSONDecodeError:
                    if self.verbose:
                        hud_console.debug(f"Hub resource '{uri}' did not return JSON list")
                    continue

                if isinstance(functions, list):
                    # Ensure consistent ordering for determinism
                    return sorted({str(fn) for fn in functions})

        try:
            # Fallback for legacy catalogue location
            result = await self.read_resource(f"file:///{hub_name}/functions")
            if result and result.contents:
                import json

                functions = json.loads(result.contents[0].text)  # type: ignore
                if isinstance(functions, list):
                    return sorted({str(fn) for fn in functions})
        except Exception as e:  # pragma: no cover - verbose logging only
            if self.verbose:
                hud_console.debug(f"Could not read hub functions for '{hub_name}': {e}")
        return []
