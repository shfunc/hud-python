from __future__ import annotations

import asyncio
import inspect
from functools import wraps
from typing import TYPE_CHECKING, Any, get_type_hints

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcp.server.fastmcp import FastMCP


def register_instance_tool(
    mcp: FastMCP,
    instance: Any,
    name: str | None = None,
    description: str | None = None,
    title: str | None = None,
) -> Callable[..., Any]:
    """Register ``instance.__call__`` as a FastMCP tool.

    Parameters
    ----------
    mcp:
        A :class:`mcp.server.fastmcp.FastMCP` instance.
    instance:
        Object with an ``async def __call__`` (or sync) implementing the tool.
        If the instance has a 'name' attribute, it will be used as the tool name.
        If the instance has a 'description' attribute, it will be used as the tool description.
        If the instance has a 'title' attribute, it will be used as the tool title.
    name:
        Optional public tool name. If not provided, uses instance.name if available.
    description:
        Optional description of what the tool does. If not provided, uses instance.description
        if available, or falls back to the docstring of instance.__call__.
    title:
        Optional human-readable title for the tool. If not provided, uses instance.title.
    """

    # If no name provided, try to get it from the instance
    if name is None and hasattr(instance, "name"):
        name = instance.name
    elif name is None:
        raise ValueError(
            "No tool name provided and instance has no 'name' attribute. "
            "Either provide a name parameter or ensure the instance has a 'name' attribute."
        )

    # If no description provided, try to get it from the instance
    if description is None and hasattr(instance, "description"):
        description = instance.description

    # If no title provided, try to get it from the instance
    if title is None and hasattr(instance, "title"):
        title = instance.title

    if inspect.isclass(instance):
        class_name = instance.__name__
        raise TypeError(
            f"register_instance_tool() expects an instance, but got class '{class_name}'. "
            f"Use: register_instance_tool(mcp, {class_name}()) "
            f"Not: register_instance_tool(mcp, {class_name})"
        )

    call_fn = instance.__call__
    sig = inspect.signature(call_fn)

    # Remove *args/**kwargs so Pydantic doesn't treat them as required fields
    from typing import Any as _Any

    filtered = [
        p.replace(kind=p.POSITIONAL_OR_KEYWORD, annotation=_Any)
        for p in sig.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]

    # Try to resolve the return annotation
    try:
        # Get the module where the instance's class is defined
        module = inspect.getmodule(instance.__class__)
        if module and sig.return_annotation != inspect.Signature.empty:
            # Try to get type hints which resolves forward references
            type_hints = get_type_hints(call_fn, globalns=module.__dict__)
            return_type = type_hints.get("return", sig.return_annotation)
        else:
            return_type = sig.return_annotation
    except Exception:
        # If we can't resolve it, just use the original annotation
        return_type = sig.return_annotation

    # Preserve the resolved return annotation for structured content
    public_sig = inspect.Signature(parameters=filtered, return_annotation=return_type)

    @wraps(call_fn)
    async def _wrapper(*args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        result = call_fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    _wrapper.__signature__ = public_sig  # type: ignore[attr-defined]

    return mcp.tool(name=name, description=description, title=title)(_wrapper)
