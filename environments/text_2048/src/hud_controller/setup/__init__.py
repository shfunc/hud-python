"""Setup module for 2048 environment."""

from hud.tools import SetupTool

# Create the setup tool instance
setup_tool = SetupTool(
    name="setup", title="Game Setup", description="Initialize or reset the 2048 game"
)

# Create decorator for registering to this tool
setup = setup_tool.register

# Import registry to trigger registration
from . import registry

__all__ = ["setup_tool"]
