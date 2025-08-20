import sys
import logging
import os
import json
import asyncio
from typing import Optional, Any
from pathlib import Path
from datetime import datetime

# Configure logging before imports to go to stderr
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

from fastmcp import Context  # for type annotations
from hud.server import MCPServer

from .services import ServiceManager
from .evaluators import evaluate as evaluate_hub
from .setup import setup as setup_hub
from .problems import ProblemRegistry
from .context import initialize_context, get_global_context

service_manager = ServiceManager()

# Create main server first so decorators can reference it
mcp = MCPServer(
    name="HUD Browser Environment",
    instructions="""
    This is a browser automation environment with full GUI access.
    Use the computer tool to interact with the browser and applications.
    You can also launch additional apps dynamically with launch_app.
    """,
)


# The decorator intercepts the MCP initialization to provide session and progress_token
@mcp.initialize
async def initialize_environment(session=None, progress_token=None):
    """Initialize the browser environment with clean startup sequence."""
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        f"initialize_environment called! session={session}, progress_token={progress_token}"
    )

    async def send_progress(progress: float, message: str):
        """Send progress notification through the session."""
        if progress_token and session:
            logger.info(f"Sending progress: {progress}% - {message}")
            await session.send_progress_notification(
                progress_token=progress_token, progress=progress, total=100, message=message
            )
        else:
            logger.info(f"No progress token, skipping: {progress}% - {message}")

    try:
        await send_progress(10, "Starting core services...")

        # Start ONLY core services (X11, VNC) - NO app launching
        await service_manager.start_services()
        await send_progress(20, "X11 server started")

        # Wait for X11 to be ready
        await service_manager.wait_for_x11()
        await send_progress(30, "X11 ready")

        # Start VNC and wait for it
        await service_manager.wait_for_vnc()
        vnc_message = {"message": "VNC server ready", "live_url": "http://localhost:8080/vnc.html"}
        await send_progress(50, json.dumps(vnc_message))

        # Initialize browser tools
        await send_progress(60, "Initializing browser tools...")
        from hud.tools import (
            HudComputerTool,
            PlaywrightTool,
            AnthropicComputerTool,
            OpenAIComputerTool,
        )

        # Store playwright tool instance for browser launch
        playwright_tool = PlaywrightTool()
        await playwright_tool._ensure_browser()
        await send_progress(70, "Browser ready...")

        # Create context and set on hubs
        global_context = initialize_context(service_manager, playwright_tool)
        setup_hub.env = global_context
        evaluate_hub.env = global_context

        # Try without proxy first to match text_2048 and remote_browser pattern
        mcp.mount(setup_hub)
        mcp.mount(evaluate_hub)
        logger.info("Mounted setup and evaluate hubs")

        await send_progress(80, "Tools registered...")

        # Register interaction tools
        mcp.add_tool(HudComputerTool())
        mcp.add_tool(AnthropicComputerTool())
        mcp.add_tool(OpenAIComputerTool())
        mcp.add_tool(playwright_tool)

        await send_progress(100, "Environment ready!")

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        if progress_token and session:
            await session.send_progress_notification(
                progress_token=progress_token,
                progress=0,
                total=100,
                message=f"Initialization failed: {str(e)}",
            )
        raise


# === MCP RESOURCES ===


@mcp.resource("problems://registry")
async def get_problems_registry_resource() -> str:
    """MCP resource containing all available problems."""
    return ProblemRegistry.to_json()


@mcp.resource("problems://{env}")
async def get_env_problems_resource(env: str) -> str:
    """MCP resource containing environment-specific problems."""
    env_problems = ProblemRegistry.get_problems_by_app(env)
    return json.dumps({"env": env, "problems": env_problems, "count": len(env_problems)}, indent=2)


@mcp.resource("schema://problem/{problem_name}")
async def get_problem_schema_resource(problem_name: str) -> str:
    """MCP resource containing detailed schema for a specific problem."""
    schema = ProblemRegistry.get_problem_schema(problem_name)
    return json.dumps(schema, indent=2)


@mcp.resource("telemetry://live")
async def get_telemetry_resource() -> str:
    """MCP resource containing telemetry data including VNC live_url."""
    telemetry_data = {
        "live_url": "http://localhost:8080/vnc.html",
        "display": os.getenv("DISPLAY", ":1"),
        "vnc_port": 8080,
        "websockify_port": 8080,
        "status": "ready",
        "timestamp": datetime.now().isoformat(),
        "services": {"x11": "running", "vnc": "running", "websockify": "running"},
    }
    return json.dumps(telemetry_data, indent=2)


# === APPLICATION TOOLS ===


@mcp.tool()
async def launch_app(ctx: Context, app_name: str) -> str:
    """Launch a specific application dynamically and navigate to it.

    Args:
        app_name: Name of the app to launch (e.g., 'todo', '2048')

    Returns:
        Success message with app URL
    """
    await ctx.info(f"Launching app: {app_name}")

    app_info = await service_manager.launch_app(app_name)
    app_url = app_info["url"]

    # Automatically navigate to the app after launching
    # Get the playwright tool from global context to navigate
    global_context = get_global_context()
    if global_context and global_context.playwright:
        try:
            await global_context.playwright.navigate(app_url)
            # Give the page a moment to fully load
            await asyncio.sleep(1)
            return f"Launched {app_name} at {app_url} and navigated to it"
        except Exception as e:
            logger.warning(f"Could not auto-navigate to app: {e}")

    return f"Launched {app_name} at {app_url}"


# API request tool (doesn't need X11)
@mcp.tool()
async def api_request(
    ctx: Context, url: str, method: str = "GET", data: dict | None = None
) -> dict:
    """Make HTTP API requests.

    Args:
        url: The URL to request
        method: HTTP method (GET, POST, etc.)
        data: Optional JSON data for POST/PUT requests

    Returns:
        Response data as dict
    """
    import httpx

    await ctx.debug(f"Making {method} request to {url}")

    async with httpx.AsyncClient() as client:
        response = await client.request(method, url, json=data)
        return {
            "status": response.status_code,
            "data": response.json()
            if response.headers.get("content-type", "").startswith("application/json")
            else response.text,
        }


@mcp.tool()
async def query_database(ctx: Context, query: str) -> list[dict]:
    """Execute a database query (mock implementation).

    Args:
        query: SQL query to execute

    Returns:
        Query results as list of dicts
    """
    await ctx.warning("This is a mock database query tool")

    # Mock implementation
    if "users" in query.lower():
        return [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
        ]
    return []


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def run():
        """Run the MCP server."""
        mcp.run(transport="stdio")

    app()
