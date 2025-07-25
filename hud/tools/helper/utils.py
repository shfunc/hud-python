"""Utilities for registering tools with MCP servers."""

import asyncio
import inspect
from functools import wraps
from typing import Any, Callable, Optional, get_type_hints
import sys

from mcp import types


def register_instance_tool(
    mcp: Any,
    name: str,
    instance: Any,
    description: Optional[str] = None,
) -> Callable[..., Any]:
    """
    Register an instance method as a tool.

    This function allows you to register a method from an instance (e.g., self.method)
    as a tool in FastMCP. The method's `self` parameter will be hidden from the tool's
    signature.

    Args:
        mcp: The FastMCP instance.
        name: The name to register the tool under.
        instance: The instance whose method to register.
        description: Optional description for the tool.

    Returns:
        The wrapped function.
    """
    if inspect.isclass(instance):
        class_name = instance.__name__
        raise TypeError(
            f"register_instance_tool() expects an instance, but got class '{class_name}'. "
            "Did you mean to pass an instance of the class instead?"
        )

    call_fn = instance.__call__

    sig = inspect.signature(call_fn)
    
    # Get the module where the function is defined
    module = inspect.getmodule(call_fn)
    globalns = {}
    
    if module:
        globalns = vars(module)
    else:
        # Fallback to the function's own globals
        globalns = getattr(call_fn, '__globals__', {})
    
    # Add common typing imports to the namespace
    import typing
    # Add all public attributes from typing module
    for attr in dir(typing):
        if not attr.startswith('_'):
            globalns[attr] = getattr(typing, attr)
    
    # Also add specific commonly used types
    globalns.update({
        'Any': typing.Any,
        'List': typing.List,
        'Dict': typing.Dict,
        'Optional': typing.Optional,
        'Union': typing.Union,
        'Literal': typing.Literal,
        'Tuple': typing.Tuple,
    })
    
    # Try to add MCP types if available
    try:
        from mcp.types import ImageContent, TextContent
        globalns['ImageContent'] = ImageContent
        globalns['TextContent'] = TextContent
    except ImportError:
        pass

    # Filter the signature to exclude 'self' and any VAR_POSITIONAL or VAR_KEYWORD
    filtered = [
        p.replace(kind=p.POSITIONAL_OR_KEYWORD)
        for p in sig.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) and p.name != 'self'
    ]

    # Create a new signature with the filtered parameters
    # Don't replace annotations with Any - keep them as-is
    public_sig = inspect.Signature(
        parameters=filtered,
        return_annotation=sig.return_annotation if sig.return_annotation != sig.empty else Any
    )

    @wraps(call_fn)
    async def _wrapper(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        result = call_fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    _wrapper.__signature__ = public_sig  # type: ignore[attr-defined]
    # Update the wrapper's globals to include our enhanced namespace
    # This is crucial for FastMCP/Pydantic to properly evaluate annotations
    if hasattr(_wrapper, '__globals__'):
        _wrapper.__globals__.update(globalns)
    
    # Also set annotations directly if possible
    if hasattr(call_fn, '__annotations__'):
        # Try to resolve annotations using get_type_hints
        try:
            resolved_hints = get_type_hints(call_fn, globalns=globalns, include_extras=True)
            # Create new annotations excluding 'self'
            new_annotations = {k: v for k, v in resolved_hints.items() if k != 'self'}
            _wrapper.__annotations__ = new_annotations
        except Exception:
            # If resolution fails, keep original string annotations
            original_annotations = getattr(call_fn, '__annotations__', {})
            _wrapper.__annotations__ = {k: v for k, v in original_annotations.items() if k != 'self'}

    # Register the tool with the MCP server
    return mcp.tool(name=name, description=description)(_wrapper)
