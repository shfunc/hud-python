from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any

from mcp.types import ContentBlock, ImageContent, TextContent

if TYPE_CHECKING:
    from hud.tools.evaluate import EvaluationResult
    from hud.tools.setup import SetupResult


class BaseTool(ABC):
    """Base class for all MCP tools.

    All tools should inherit from this class and implement the __call__ method.
    Tools are registered with FastMCP using register_instance_tool.
    """

    def __init__(
        self,
        context: Any = None,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            context: Optional, often stateful, context object that the tool operates on. Could be:
                - A game instance (e.g., Chess Board)
                - An executor (e.g., PyAutoGUIExecutor for computer control)
                - A browser/page instance (e.g., Playwright Page)
                - Any stateful resource the tool needs to interact with
            name: Tool name for MCP registration (auto-generated from class name if not provided)
            title: Human-readable display name for the tool (auto-generated from class name)
            description: Tool description (auto-generated from docstring if not provided)
        """
        self.context = context
        self.name = name or self.__class__.__name__.lower().replace("tool", "")
        self.title = title
        self.description = description

    @abstractmethod
    async def __call__(self, **kwargs: Any) -> list[ContentBlock] | EvaluationResult | SetupResult:
        """Execute the tool. Often uses the context to perform an action.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            List of ContentBlock (TextContent, ImageContent, etc.) with the tool's output
        """
        raise NotImplementedError("Subclasses must implement __call__")

    def _to_content_blocks(self, result: ToolResult) -> list[ContentBlock]:
        """Helper method to convert ToolResult to content blocks.

        Subclasses can use this when they work with ToolResult internally.

        Args:
            result: ToolResult to convert

        Returns:
            List of ContentBlock
        """
        blocks: list[ContentBlock] = []

        if result.output:
            blocks.append(TextContent(text=result.output, type="text"))
        if result.error:
            blocks.append(TextContent(text=result.error, type="text"))
        if result.base64_image:
            blocks.append(
                ImageContent(data=result.base64_image, mimeType="image/png", type="image")
            )
        return blocks


@dataclass(kw_only=True, frozen=True)
class ToolResult:
    """Represents the intermediate result of a tool execution.

    Often useful for tools that need to return multiple types of content.
    """

    output: str | None = None
    error: str | None = None
    base64_image: str | None = None
    system: str | None = None

    def __bool__(self) -> bool:
        return any(getattr(self, field.name) for field in fields(self))

    def __add__(self, other: ToolResult) -> ToolResult:
        def combine_fields(
            field: str | None, other_field: str | None, concatenate: bool = True
        ) -> str | None:
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def replace(self, **kwargs: Any) -> ToolResult:
        """Returns a new ToolResult with the given fields replaced."""
        return replace(self, **kwargs)


class ToolError(Exception):
    """An error raised by a tool."""
