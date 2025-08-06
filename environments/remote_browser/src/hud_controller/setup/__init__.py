"""Setup module for remote browser environment."""

from hud.tools import SetupTool

# Create the setup tool instance
setup_tool = SetupTool(
    name="setup",
    description="Setup the remote browser environment"
)

# Create decorator for registering to this tool
setup = setup_tool.register

# Import all setup modules to trigger registration
from . import navigate, cookies, load_html, interact, sheets

__all__ = ['setup_tool', 'setup']