"""MCP Development Proxy - Hot-reload environments with MCP over HTTP."""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
from pathlib import Path

import click
from fastmcp import FastMCP

from hud.utils.design import HUDDesign

from .utils.docker import get_docker_cmd, inject_supervisor
from .utils.environment import (
    build_environment,
    get_image_name,
    image_exists,
    update_pyproject_toml,
)

# Global design instance
design = HUDDesign()


def build_and_update(directory: str | Path, image_name: str, no_cache: bool = False) -> None:
    """Build Docker image and update pyproject.toml."""
    if not build_environment(directory, image_name, no_cache):
        raise click.Abort


def create_proxy_server(
    directory: str | Path,
    image_name: str,
    no_reload: bool = False,
    full_reload: bool = False,
    verbose: bool = False,
    docker_args: list[str] | None = None,
    interactive: bool = False,
) -> FastMCP:
    """Create an HTTP proxy server that forwards to Docker container with hot-reload."""
    src_path = Path(directory) / "src"

    # Get the original CMD from the image
    original_cmd = get_docker_cmd(image_name)
    if not original_cmd:
        design.warning(f"Could not extract CMD from {image_name}, using default")
        original_cmd = ["python", "-m", "hud_controller.server"]

    # Generate unique container name from image to avoid conflicts between multiple instances
    import os

    pid = str(os.getpid())[-6:]  # Last 6 digits of process ID for uniqueness
    base_name = image_name.replace(":", "-").replace("/", "-")
    container_name = f"{base_name}-{pid}"

    # Build the docker run command
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--name",
        container_name,
        "-v",
        f"{src_path.absolute()}:/app/src:rw",
        "-e",
        "PYTHONPATH=/app/src",
    ]

    # Add user-provided Docker arguments
    if docker_args:
        docker_cmd.extend(docker_args)

    # Disable hot-reload if interactive mode is enabled
    if interactive:
        no_reload = True

    # Validate reload options
    if no_reload and full_reload:
        design.warning("Cannot use --full-reload with --no-reload, ignoring --full-reload")
        full_reload = False

    if not no_reload and not full_reload:
        # Standard hot-reload: inject supervisor for server restart within container
        modified_cmd = inject_supervisor(original_cmd)
        docker_cmd.extend(["--entrypoint", modified_cmd[0]])
        docker_cmd.append(image_name)
        docker_cmd.extend(modified_cmd[1:])
    else:
        # No reload or full reload: use original CMD without supervisor
        # Note: Full reload logic (container restart) would be implemented here in the future
        docker_cmd.append(image_name)

    # Create configuration following MCPConfig schema
    config = {
        "mcpServers": {
            "default": {
                "command": docker_cmd[0],
                "args": docker_cmd[1:] if len(docker_cmd) > 1 else [],
                # transport defaults to stdio
            }
        }
    }

    # Debug output - only if verbose
    if verbose:
        if not no_reload and not full_reload:
            design.info("Mode: Hot-reload (server restart within container)")
            design.info("Watching: /app/src for changes")
        elif full_reload:
            design.info("Mode: Full reload (container restart on file changes)")
            design.info("Note: Full container restart not yet implemented, using no-reload mode")
        else:
            design.info("Mode: No reload")
            design.info("Container will run without hot-reload")
        design.command_example(f"docker logs -f {container_name}", "View container logs")

        # Show the full Docker command if there are environment variables
        if docker_args and any(arg == "-e" or arg.startswith("--env") for arg in docker_args):
            design.info("")
            design.info("Docker command with environment variables:")
            design.info(" ".join(docker_cmd))

    # Create the HTTP proxy server using config
    try:
        proxy = FastMCP.as_proxy(config, name=f"HUD Dev Proxy - {image_name}")
    except Exception as e:
        design.error(f"Failed to create proxy server: {e}")
        design.info("")
        design.info("💡 Tip: Run the following command to debug the container:")
        design.info(f"   hud debug {image_name}")
        raise

    return proxy


async def start_mcp_proxy(
    directory: str | Path,
    image_name: str,
    transport: str,
    port: int,
    no_reload: bool = False,
    full_reload: bool = False,
    verbose: bool = False,
    inspector: bool = False,
    no_logs: bool = False,
    interactive: bool = False,
    docker_args: list[str] | None = None,
) -> None:
    """Start the MCP development proxy server."""
    # Suppress FastMCP's verbose output FIRST
    import asyncio
    import logging
    import os
    import sys

    from .utils.logging import find_free_port

    # Always disable the banner - we have our own output
    os.environ["FASTMCP_DISABLE_BANNER"] = "1"

    # Configure logging BEFORE creating proxy
    if not verbose:
        # Create a filter to block the specific "Starting MCP server" message
        class _BlockStartingMCPFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return "Starting MCP server" not in record.getMessage()

        # Set environment variable for FastMCP logging
        os.environ["FASTMCP_LOG_LEVEL"] = "ERROR"
        os.environ["LOG_LEVEL"] = "ERROR"
        os.environ["UVICORN_LOG_LEVEL"] = "ERROR"
        # Suppress uvicorn's annoying shutdown messages
        os.environ["UVICORN_ACCESS_LOG"] = "0"

        # Configure logging to suppress INFO
        logging.basicConfig(level=logging.ERROR, force=True)

        # Set root logger to ERROR to suppress all INFO messages
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.ERROR)

        # Add filter to all handlers
        block_filter = _BlockStartingMCPFilter()
        for handler in root_logger.handlers:
            handler.addFilter(block_filter)

        # Also specifically suppress these loggers
        for logger_name in [
            "fastmcp",
            "fastmcp.server",
            "fastmcp.server.server",
            "FastMCP",
            "FastMCP.fastmcp.server.server",
            "mcp",
            "mcp.server",
            "mcp.server.lowlevel",
            "mcp.server.lowlevel.server",
            "uvicorn",
            "uvicorn.access",
            "uvicorn.error",
            "hud.server",
            "hud.server.server",
        ]:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.ERROR)
            # Add filter to this logger too
            logger.addFilter(block_filter)

        # Suppress deprecation warnings
        import warnings

        warnings.filterwarnings("ignore", category=DeprecationWarning)

    # CRITICAL: For stdio transport, ALL output must go to stderr
    if transport == "stdio":
        # Configure root logger to use stderr
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        stderr_handler = logging.StreamHandler(sys.stderr)
        root_logger.addHandler(stderr_handler)

    # Now check for src directory
    src_path = Path(directory) / "src"
    if not src_path.exists():
        design.error(f"Source directory not found: {src_path}")
        raise click.Abort

    # Extract container name from the proxy configuration (must match create_proxy_server naming)
    import os

    pid = str(os.getpid())[-6:]  # Last 6 digits of process ID for uniqueness
    base_name = image_name.replace(":", "-").replace("/", "-")
    container_name = f"{base_name}-{pid}"

    # Remove any existing container with the same name (silently)
    # Note: The proxy creates containers on-demand when clients connect
    try:  # noqa: SIM105
        subprocess.run(  # noqa: S603, ASYNC221
            ["docker", "rm", "-f", container_name],  # noqa: S607
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,  # Don't raise error if container doesn't exist
        )
    except Exception:  # noqa: S110
        pass

    if transport == "stdio":
        if verbose:
            design.info("Starting stdio proxy (each connection gets its own container)")
    else:
        # Find available port for HTTP
        actual_port = find_free_port(port)
        if actual_port is None:
            design.error(f"No available ports found starting from {port}")
            raise click.Abort

        if actual_port != port and verbose:
            design.warning(f"Port {port} in use, using port {actual_port} instead")

        # Launch MCP Inspector if requested
        if inspector:
            server_url = f"http://localhost:{actual_port}/mcp"

            # Function to launch inspector in background
            async def launch_inspector() -> None:
                """Launch MCP Inspector and capture its output to extract the URL."""
                # Wait for server to be ready
                await asyncio.sleep(3)

                try:
                    import platform
                    import urllib.parse

                    # Build the direct URL with query params to auto-connect
                    encoded_url = urllib.parse.quote(server_url)
                    inspector_url = (
                        f"http://localhost:6274/?transport=streamable-http&serverUrl={encoded_url}"
                    )

                    # Print inspector info cleanly
                    design.section_title("MCP Inspector")
                    design.link(inspector_url)

                    # Set environment to disable auth (for development only)
                    env = os.environ.copy()
                    env["DANGEROUSLY_OMIT_AUTH"] = "true"
                    env["MCP_AUTO_OPEN_ENABLED"] = "true"

                    # Launch inspector
                    cmd = ["npx", "--yes", "@modelcontextprotocol/inspector"]

                    # Run in background, suppressing output to avoid log interference
                    if platform.system() == "Windows":
                        subprocess.Popen(  # noqa: S602, ASYNC220
                            cmd,
                            env=env,
                            shell=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    else:
                        subprocess.Popen(  # noqa: S603, ASYNC220
                            cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )

                except (FileNotFoundError, Exception):
                    # Silently fail - inspector is optional
                    design.error("Failed to launch inspector")

            # Launch inspector asynchronously so it doesn't block
            asyncio.create_task(launch_inspector())  # noqa: RUF006

        # Launch interactive mode if requested
        if interactive:
            if transport != "http":
                design.warning("Interactive mode only works with HTTP transport")
            else:
                server_url = f"http://localhost:{actual_port}/mcp"

                # Function to launch interactive mode in a separate thread
                def launch_interactive_thread() -> None:
                    """Launch interactive testing mode in a separate thread."""
                    import time

                    # Wait for server to be ready
                    time.sleep(3)

                    try:
                        design.section_title("Interactive Mode")
                        design.info("Starting interactive testing mode...")
                        design.info("Press Ctrl+C in the interactive session to exit")

                        # Import and run interactive mode in a new event loop
                        from .utils.interactive import run_interactive_mode

                        # Create a new event loop for the thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(run_interactive_mode(server_url, verbose))
                        finally:
                            loop.close()

                    except Exception as e:
                        # Log error but don't crash the server
                        if verbose:
                            design.error(f"Interactive mode error: {e}")

                # Launch interactive mode in a separate thread
                import threading

                interactive_thread = threading.Thread(target=launch_interactive_thread, daemon=True)
                interactive_thread.start()

    # Function to stream Docker logs
    async def stream_docker_logs() -> None:
        """Stream Docker container logs asynchronously.

        Note: The Docker container is created on-demand when the first client connects.
        Any environment variables passed via -e flags are included when the container starts.
        """
        log_design = design

        # Always show waiting message
        log_design.info("")  # Empty line for spacing
        log_design.progress_message("⏳ Waiting for first client connection to start container...")
        log_design.info(f"📋 Looking for container: {container_name}")  # noqa: G004

        # Keep trying to stream logs - container is created on demand
        has_shown_started = False
        while True:
            # Check if container exists first (silently)
            check_result = await asyncio.create_subprocess_exec(
                "docker",
                "ps",
                "--format",
                "{{.Names}}",
                "--filter",
                f"name={container_name}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await check_result.communicate()

            # If container doesn't exist, wait and retry
            if container_name not in stdout.decode():
                await asyncio.sleep(1)
                continue

            # Container exists! Show success if first time
            if not has_shown_started:
                log_design.success("Container started! Streaming logs...")
                has_shown_started = True

            # Now stream the logs
            try:
                process = await asyncio.create_subprocess_exec(
                    "docker",
                    "logs",
                    "-f",
                    container_name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,  # Combine streams for simplicity
                )

                if process.stdout:
                    async for line in process.stdout:
                        decoded_line = line.decode().rstrip()
                        if not decoded_line:  # Skip empty lines
                            continue

                        # Skip docker daemon errors (these happen when container is removed)
                        if "Error response from daemon" in decoded_line:
                            continue

                        # Show all logs with gold formatting like hud debug
                        # Format all logs in gold/dim style like hud debug's stderr
                        # Use stdout console to avoid stderr redirection when not verbose
                        log_design._stdout_console.print(
                            f"[rgb(192,150,12)]■[/rgb(192,150,12)] {decoded_line}", highlight=False
                        )

                # Process ended - container might have been removed
                await process.wait()

                # Check if container still exists
                await asyncio.sleep(1)
                continue  # Loop back to check if container exists

            except Exception as e:
                # Some unexpected error - show it so we can debug
                log_design.warning(f"Failed to stream Docker logs: {e}")  # noqa: G004
                if verbose:
                    import traceback

                    log_design.warning(f"Traceback: {traceback.format_exc()}")  # noqa: G004
                await asyncio.sleep(1)

    # CRITICAL: Create proxy AFTER all logging setup to prevent it from resetting logging config
    # This is important because FastMCP might initialize loggers during creation
    proxy = create_proxy_server(
        directory, image_name, no_reload, full_reload, verbose, docker_args or [], interactive
    )

    # One more attempt to suppress the FastMCP server log
    if not verbose:
        # Re-apply the filter in case new handlers were created
        class BlockStartingMCPFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return "Starting MCP server" not in record.getMessage()

        block_filter = BlockStartingMCPFilter()

        # Apply to all loggers again - comprehensive list
        for logger_name in [
            "",  # root logger
            "fastmcp",
            "fastmcp.server",
            "fastmcp.server.server",
            "FastMCP",
            "FastMCP.fastmcp.server.server",
            "mcp",
            "mcp.server",
            "mcp.server.lowlevel",
            "mcp.server.lowlevel.server",
            "uvicorn",
            "uvicorn.access",
            "uvicorn.error",
            "hud.server",
            "hud.server.server",
        ]:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.ERROR)
            logger.addFilter(block_filter)
            for handler in logger.handlers:
                handler.addFilter(block_filter)

    try:
        # Start Docker logs streaming if enabled
        log_task = None
        if not no_logs:
            log_task = asyncio.create_task(stream_docker_logs())

        if transport == "stdio":
            # Run with stdio transport
            await proxy.run_async(
                transport="stdio", log_level="ERROR" if not verbose else "INFO", show_banner=False
            )
        else:
            # Run with HTTP transport
            # Temporarily redirect stderr to suppress uvicorn shutdown messages
            import contextlib
            import io

            if not verbose:
                # Create a dummy file to swallow unwanted stderr output
                with contextlib.redirect_stderr(io.StringIO()):
                    await proxy.run_async(
                        transport="http",
                        host="0.0.0.0",  # noqa: S104
                        port=actual_port,
                        path="/mcp",  # Serve at /mcp endpoint
                        log_level="ERROR",
                        show_banner=False,
                    )
            else:
                await proxy.run_async(
                    transport="http",
                    host="0.0.0.0",  # noqa: S104
                    port=actual_port,
                    path="/mcp",  # Serve at /mcp endpoint
                    log_level="INFO",
                    show_banner=False,
                )
    except (ConnectionError, OSError) as e:
        design.error(f"Failed to connect to Docker container: {e}")
        design.info("")
        design.info("💡 Tip: Run the following command to debug the container:")
        design.info(f"   hud debug {image_name}")
        design.info("")
        design.info("Common issues:")
        design.info("  • Container failed to start or crashed immediately")
        design.info("  • Server initialization failed")
        design.info("  • Port binding conflicts")
        raise
    except KeyboardInterrupt:
        design.info("\n👋 Shutting down...")

        # Show next steps tutorial
        if not interactive:  # Only show if not in interactive mode
            design.section_title("Next Steps")
            design.info("🏗️  Ready to test with real agents? Run:")
            design.info(f"    [cyan]hud build {directory}[/cyan]")
            design.info("")
            design.info("This will:")
            design.info("  1. Build your environment image")
            design.info("  2. Generate a hud.lock.yaml file")
            design.info("  3. Prepare it for testing with agents")
            design.info("")
            design.info("Then you can:")
            design.info("  • Test locally: [cyan]hud run <image>[/cyan]")
            design.info("  • Push to registry: [cyan]hud push --image <registry/name>[/cyan]")
    except Exception as e:
        # Suppress the graceful shutdown error and other FastMCP/uvicorn internal errors
        error_msg = str(e)
        if not any(
            x in error_msg
            for x in [
                "timeout graceful shutdown exceeded",
                "Cancel 0 running task(s)",
                "Application shutdown complete",
            ]
        ):
            design.error(f"Unexpected error: {e}")
    finally:
        # Cancel log streaming task if it exists
        if log_task and not log_task.done():
            log_task.cancel()
            try:
                await log_task
            except asyncio.CancelledError:
                contextlib.suppress(asyncio.CancelledError)


def run_mcp_dev_server(
    directory: str = ".",
    image: str | None = None,
    build: bool = False,
    no_cache: bool = False,
    transport: str = "http",
    port: int = 8765,
    no_reload: bool = False,
    full_reload: bool = False,
    verbose: bool = False,
    inspector: bool = False,
    no_logs: bool = False,
    interactive: bool = False,
    docker_args: list[str] | None = None,
) -> None:
    """Run MCP development server with hot-reload.

    This command starts a development proxy that:
    - Auto-detects or builds Docker images
    - Mounts local source code for hot-reload
    - Exposes an HTTP endpoint for MCP clients

    Examples:
        hud dev .                    # Auto-detect image from directory
        hud dev . --build            # Build image first
        hud dev . --image custom:tag # Use specific image
        hud dev . --no-cache         # Force clean rebuild
    """
    # Ensure directory exists
    if not Path(directory).exists():
        design.error(f"Directory not found: {directory}")
        raise click.Abort

    # No external dependencies needed for hot-reload anymore!

    # Resolve image name
    resolved_image, source = get_image_name(directory, image)

    # Update pyproject.toml with auto-generated name if needed
    if source == "auto":
        update_pyproject_toml(directory, resolved_image)

    # Build if requested
    if build or no_cache:
        build_and_update(directory, resolved_image, no_cache)

    # Check if image exists
    if not image_exists(resolved_image) and not build:
        if click.confirm(f"Image {resolved_image} not found. Build it now?"):
            build_and_update(directory, resolved_image)
        else:
            raise click.Abort

    # Generate server name from image
    server_name = resolved_image.split(":")[0] if ":" in resolved_image else resolved_image

    # For HTTP transport, find available port first
    actual_port = port
    if transport == "http":
        from .utils.logging import find_free_port

        actual_port = find_free_port(port)
        if actual_port is None:
            design.error(f"No available ports found starting from {port}")
            raise click.Abort
        if actual_port != port and verbose:
            design.warning(f"Port {port} in use, using port {actual_port}")

    # Create config
    if transport == "stdio":
        server_config = {"command": "hud", "args": ["dev", directory, "--transport", "stdio"]}
        # For stdio, include docker args in the command
        if docker_args:
            server_config["args"].extend(docker_args)
    else:
        server_config = {"url": f"http://localhost:{actual_port}/mcp"}
        # Note: Environment variables are passed to the Docker container via the proxy,
        # not included in the client configuration

    # For the deeplink, we only need the server config
    server_config_json = json.dumps(server_config, indent=2)
    config_base64 = base64.b64encode(server_config_json.encode()).decode()

    # Generate deeplink
    deeplink = (
        f"cursor://anysphere.cursor-deeplink/mcp/install?name={server_name}&config={config_base64}"
    )

    # Show header with gold border
    design.info("")  # Empty line before header
    design.header("HUD Development Server")

    # Always show the Docker image being used as the first thing after header
    design.section_title("Docker Image")
    if source == "cache":
        design.info(f"📦 {resolved_image}")
    elif source == "auto":
        design.info(f"🔧 {resolved_image} (auto-generated)")
    elif source == "override":
        design.info(f"🎯 {resolved_image} (specified)")
    else:
        design.info(f"🐳 {resolved_image}")

    design.progress_message(
        f"❗ If any issues arise, run `hud debug {resolved_image}` to debug the container"
    )

    # Show environment variables if provided
    if docker_args and any(arg == "-e" or arg.startswith("--env") for arg in docker_args):
        design.section_title("Environment Variables")
        design.info("The following environment variables will be passed to the Docker container:")
        i = 0
        while i < len(docker_args):
            if docker_args[i] == "-e" and i + 1 < len(docker_args):
                design.info(f"  • {docker_args[i + 1]}")
                i += 2
            elif docker_args[i].startswith("--env="):
                design.info(f"  • {docker_args[i][6:]}")
                i += 1
            elif docker_args[i] == "--env" and i + 1 < len(docker_args):
                design.info(f"  • {docker_args[i + 1]}")
                i += 2
            else:
                i += 1

    # Show hints about inspector and interactive mode
    if transport == "http":
        if not inspector and not interactive:
            design.progress_message("💡 Run with --inspector to launch MCP Inspector")
            design.progress_message("🧪 Run with --interactive for interactive testing mode")
        elif not inspector:
            design.progress_message("💡 Run with --inspector to launch MCP Inspector")
        elif not interactive:
            design.progress_message("🧪 Run with --interactive for interactive testing mode")

    # Disable logs and hot-reload if interactive mode is enabled
    if interactive:
        if not no_logs:
            design.warning("Docker logs disabled in interactive mode for better UI experience")
            no_logs = True
        if not no_reload:
            design.warning("Hot-reload disabled in interactive mode to prevent output interference")
            no_reload = True

    # Show configuration as JSON (just the server config, not wrapped)
    full_config = {}
    full_config[server_name] = server_config

    design.section_title("MCP Configuration (add this to any agent/client)")
    design.json_config(json.dumps(full_config, indent=2))

    # Show connection info
    design.section_title(
        "Connect to Cursor (be careful with multiple windows as that may interfere with the proxy)"
    )
    design.link(deeplink)
    design.info("")  # Empty line

    # Start the proxy (pass original port, start_mcp_proxy will find actual port again)
    try:
        asyncio.run(
            start_mcp_proxy(
                directory,
                resolved_image,
                transport,
                port,
                no_reload,
                full_reload,
                verbose,
                inspector,
                no_logs,
                interactive,
                docker_args or [],
            )
        )
    except Exception as e:
        design.error(f"Failed to start MCP server: {e}")
        design.info("")
        design.info("💡 Tip: Run the following command to debug the container:")
        design.info(f"   hud debug {resolved_image}")
        design.info("")
        design.info("This will help identify connection issues or initialization failures.")
        raise
