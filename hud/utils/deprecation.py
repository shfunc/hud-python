"""Deprecation utilities for HUD SDK."""

from __future__ import annotations

import functools
import logging
import warnings
from typing import TYPE_CHECKING, Any, TypeVar, cast

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable
T = TypeVar("T")


def deprecated(
    reason: str,
    *,
    version: str | None = None,
    replacement: str | None = None,
    removal_version: str | None = None,
) -> Callable[[T], T]:
    """
    Decorator to mark functions, methods, or classes as deprecated.

    Args:
        reason: Explanation of why this is deprecated
        version: Version when this was deprecated (e.g., "1.0.0")
        replacement: What to use instead
        removal_version: Version when this will be removed

    Example:
        @deprecated(
            reason="Use TaskConfig instead",
            replacement="hud.datasets.TaskConfig",
            version="0.3.0",
            removal_version="0.4.0"
        )
        class OldClass:
            pass
    """

    def decorator(obj: T) -> T:
        message_parts = [f"{obj.__module__}.{obj.__qualname__} is deprecated"]

        if version:
            message_parts.append(f"(deprecated since v{version})")

        message_parts.append(f": {reason}")

        if replacement:
            message_parts.append(f". Use {replacement} instead")

        if removal_version:
            message_parts.append(f". Will be removed in v{removal_version}")

        deprecation_message = " ".join(message_parts) + "."

        if isinstance(obj, type):
            # Handle class deprecation
            original_init = obj.__init__

            @functools.wraps(original_init)
            def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(deprecation_message, DeprecationWarning, stacklevel=2)
                logger.warning(deprecation_message)
                original_init(self, *args, **kwargs)

            obj.__init__ = new_init

            # Update docstring
            if obj.__doc__:
                obj.__doc__ = f"**DEPRECATED**: {deprecation_message}\n\n{obj.__doc__}"
            else:
                obj.__doc__ = f"**DEPRECATED**: {deprecation_message}"

        else:
            # Handle function/method deprecation
            func = cast("Callable[..., Any]", obj)

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                warnings.warn(deprecation_message, DeprecationWarning, stacklevel=2)
                logger.warning(deprecation_message)
                return func(*args, **kwargs)

            # Update docstring
            if wrapper.__doc__:
                wrapper.__doc__ = f"**DEPRECATED**: {deprecation_message}\n\n{wrapper.__doc__}"
            else:
                wrapper.__doc__ = f"**DEPRECATED**: {deprecation_message}"

            return cast("T", wrapper)

        return obj

    return decorator


def emit_deprecation_warning(
    message: str,
    category: type[Warning] = DeprecationWarning,
    stacklevel: int = 2,
) -> None:
    """
    Emit a deprecation warning with both warnings and logging.

    Args:
        message: The deprecation message
        category: Warning category (default: DeprecationWarning)
        stacklevel: Stack level for warning (default: 2)
    """
    warnings.warn(message, category, stacklevel=stacklevel)
    logger.warning(message)
