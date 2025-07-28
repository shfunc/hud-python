# HUD Controller

MCP server implementation for the browser automation environment.

## Architecture

The controller manages:
- X11 display server (Xvfb)
- VNC server (x11vnc + websockify)
- Browser automation (via HudComputerTool)
- Dynamic app launching

## Key Files

- **__main__.py** - Entry point that routes to server or other commands
- **server.py** - FastMCP server with tool registration and initialization
- **services.py** - Service management (X11, VNC, browser, apps)
- **browser.py** - Browser control utilities

## Initialization Flow

1. `start.sh` starts Xvfb and waits for X11
2. `server.py` uses `@mcp_intialize_wrapper` for progress notifications
3. `initialize_environment()` runs during MCP initialization:
   - Starts core services via `ServiceManager`
   - Waits for X11 and VNC to be ready
   - Registers HudComputerTool after X11 is available
   - Launches browser and any requested apps
   - Sends progress notifications throughout

## Adding Tools

Tools can be added in two ways:

### 1. Decorator-based (loaded at module import)
```python
@mcp.tool()
async def my_tool(param: str, ctx: Context) -> str:
    """Tool that doesn't need X11"""
    return f"Result: {param}"
```

### 2. Instance-based (registered during initialization)
```python
# In initialize_environment()
from hud.tools import HudComputerTool
computer_tool = HudComputerTool(width=1024, height=768)
register_instance_tool(mcp, "computer", computer_tool)
```

## Service Manager

The `ServiceManager` class handles:
- Starting/stopping X11, VNC, and websockify
- Launching the browser with initial URL
- Dynamic app launching with port allocation
- Health checks and readiness waiting

## Environment Variables

- `DISPLAY=:1` - X11 display number
- `BROWSER_URL` - Initial browser URL
- `LAUNCH_APPS` - Comma-separated list of apps to launch

## Debugging

All logs go to stderr to keep stdout clean for JSON-RPC:
```python
logger = logging.getLogger(__name__)
logger.info("Debug message")  # Goes to stderr
```

Monitor logs during testing:
```bash
docker run -i --rm hud-browser 2>debug.log
``` 