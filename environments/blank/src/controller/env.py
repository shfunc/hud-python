"""Minimal environment that persists across hot-reloads."""

from __future__ import annotations

import asyncio

from hud.server.context import run_context_server


class Environment:
    """Simple counter environment."""

    def __init__(self):
        self.count = 0

    def act(self):
        """Increment the counter."""
        self.count += 1
        return self.count

    def get_count(self):
        """Get current counter."""
        return self.count

    def reset(self):
        """Reset counter to zero."""
        self.count = 0


if __name__ == "__main__":
    asyncio.run(run_context_server(Environment(), sock_path="/tmp/hud_ctx.sock"))
