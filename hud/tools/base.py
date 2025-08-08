from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from fastmcp import Context, FastMCP

from hud.tools.types import ContentBlock, EvaluationResult, SetupResult

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from fastmcp.tools import FunctionTool
    from fastmcp.tools.tool import ToolResult

# Basic result types for tools
BaseResult = list[ContentBlock] | EvaluationResult | SetupResult

class BaseTool(ABC):
    """
    Base helper class for all MCP tools to constrain their output.

    USAGE:
    All tools should inherit from this class and implement the __call__ method.
    Tools are registered with FastMCP using add_tool.

    FORMAT:
    Tools that return messages should return a list[ContentBlock].
    Tools that return miscallaneous content should return a pydantic model such as EvaluationResult.
    Both of these types of tools are processed via structuredContent.
    Any other type of tool will not be processed well by the client.
    """

    def __init__(
        self,
        env: Any = None,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            env: Optional, often stateful, context object that the tool operates on. Could be:
                - A game instance (e.g., Chess Board)
                - An executor (e.g., PyAutoGUIExecutor for computer control)
                - A browser/page instance (e.g., Playwright Page)
                - Any stateful resource the tool needs to interact with
            name: Tool name for MCP registration (auto-generated from class name if not provided)
            title: Human-readable display name for the tool (auto-generated from class name)
            description: Tool description (auto-generated from docstring if not provided)
        """
        self.env = env
        self.name = name or self.__class__.__name__.lower().replace("tool", "")
        self.title = title or self.__class__.__name__.replace("Tool", "").replace("_", " ").title()
        self.description = description or (self.__doc__.strip() if self.__doc__ else None)

        # Expose attributes FastMCP expects when registering an instance directly
        self.__name__ = self.name  # FastMCP uses fn.__name__ if name param omitted
        if self.description:
            self.__doc__ = self.description

    @abstractmethod
    async def __call__(self, **kwargs: Any) -> ToolResult:
        """Execute the tool. Often uses the context to perform an action.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            List of ContentBlock (TextContent, ImageContent, etc.) with the tool's output
        """
        raise NotImplementedError("Subclasses must implement __call__")

    def register(self, server: FastMCP, **meta: Any) -> BaseTool:
        """Register this tool on a FastMCP server and return self for chaining."""
        server.add_tool(self.mcp, **meta)
        return self

    @property
    def mcp(self) -> FunctionTool:
        """Get this tool as a FastMCP FunctionTool (cached).
        
        This allows even cleaner registration:
            server.add_tool(my_tool.mcp)
        """
        if not hasattr(self, "_mcp_tool"):
            from fastmcp.tools import FunctionTool
            
            self._mcp_tool = FunctionTool.from_function(
                self,
                name=self.name,
                title=self.title,
                description=self.description,
            )
        return self._mcp_tool


# Tag used to hide internal components
_INTERNAL_TAG = "internal"

class BaseHub(FastMCP):
    """A composition-friendly FastMCP server that hides its internals."""

    env: Any

    def __init__(
        self,
        name: str,
        *,
        env: Any | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        """Create a new BaseHub.

        Parameters
        ----------
        name:
            Public name. Also becomes the *dispatcher tool* name.
        env:
            Optional long-lived environment object. Stored on the server
            instance (``layer.env``) and therefore available to every request
            via ``ctx.fastmcp.env``.
        """

        # Naming scheme for hidden objects
        self._prefix_fn: Callable[[str], str] = lambda n: f"int_{n}"

        async def _lifespan(server: FastMCP) -> AsyncGenerator[None, Any]:
            if env is not None:
                server.env = env  # type: ignore[attr-defined]
            yield

        super().__init__(
            name=name,
            exclude_tags={_INTERNAL_TAG},
            lifespan=_lifespan,  # type: ignore[arg-type]
        )

        # Human-friendly metadata for dispatcher
        dispatcher_title = title or f"{name.title()} Dispatcher"
        dispatcher_desc = description or f"Call internal '{name}' functions"

        # Register dispatcher without internal tag so it's visible to clients
        @super().tool(name=name, title=dispatcher_title, description=dispatcher_desc)
        async def _dispatch(
            function: str,
            ctx: Context,
            args: dict | None = None,
        ) -> ToolResult:
            """Gateway to hidden tools.

            Parameters
            ----------
            function : str
                Internal function name *without* prefix.
            args : dict | None
                Arguments forwarded to the internal tool.
            ctx : Context
                Injected by FastMCP; can be the custom subclass.
            """

            # Use server private _call_tool to run the hidden function
            return await ctx.fastmcp._call_tool(self._prefix_fn(function), args or {})

        # ------------- catalogue resource for debugging ----------
        # Expose list of internal functions via read-only resource
        @self.resource(f"{name}://functions")
        async def _functions_catalogue() -> list[str]:
            tools = await self._tool_manager.list_tools()
            return [
                t.name.removeprefix(self._prefix_fn("") or "")
                for t in tools
                if _INTERNAL_TAG in t.tags
            ]

    def tool(self, name_or_fn: Any = None, /, **kwargs: Any) -> Callable[..., Any]:
        """Register an *internal* tool (hidden from clients)."""
        # Handle the name from either positional or keyword argument
        if isinstance(name_or_fn, str):
            # Called as @hub.tool("name")
            name = name_or_fn
        elif name_or_fn is None and "name" in kwargs:
            # Called as @hub.tool(name="name")
            name = kwargs.pop("name")
        else:
            # Called as @hub.tool or @hub.tool()
            name = None
            
        new_name = self._prefix_fn(name) if name is not None else None
        tags = {_INTERNAL_TAG} | kwargs.pop("tags", set())
        
        # Pass through correctly to parent
        if new_name is not None:
            return super().tool(new_name, **kwargs, tags=tags)
        else:
            return super().tool(**kwargs, tags=tags)

    resource = FastMCP.resource  # type: ignore[assignment]
    prompt = FastMCP.prompt      # type: ignore[assignment]

