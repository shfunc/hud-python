from __future__ import annotations

from typing import Any

from .utils import register_instance_tool

__all__ = ["build_server", "register_instance_tool"]  # type: ignore


def __getattr__(name: str) -> Any:
    if name == "build_server":
        from .mcp_server import build_server as _bs

        globals()["build_server"] = _bs
        return _bs
    raise AttributeError(name)
