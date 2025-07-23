from __future__ import annotations

import asyncio
import inspect
from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcp.server.fastmcp import FastMCP


def register_instance_tool(mcp: FastMCP, name: str, instance: Any) -> Callable[..., Any]:
    """Register ``instance.__call__`` as a FastMCP tool.

    Parameters
    ----------
    mcp:
        A :class:`mcp.server.fastmcp.FastMCP` instance.
    name:
        Public tool name.
    instance:
        Object with an ``async def __call__`` (or sync) implementing the tool.
    """

    call_fn = instance.__call__
    sig = inspect.signature(call_fn)

    # Remove *args/**kwargs so Pydantic doesn't treat them as required fields
    # Also remove 'self' parameter for instance methods
    from typing import Any as _Any

    param_list = list(sig.parameters.values())
    filtered = []
    
    for i, p in enumerate(param_list):
        # Skip 'self' parameter (first parameter of instance methods)
        if i == 0 and p.name == "self":
            continue
        # Skip *args/**kwargs
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        # Keep other parameters but normalize their type annotations
        filtered.append(p.replace(kind=p.POSITIONAL_OR_KEYWORD, annotation=_Any))

    public_sig = inspect.Signature(parameters=filtered, return_annotation=_Any)

    @wraps(call_fn)
    async def _wrapper(*args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        result = call_fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    _wrapper.__signature__ = public_sig  # type: ignore[attr-defined]

    return mcp.tool(name=name)(_wrapper)
