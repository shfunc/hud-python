"""MCP Development Proxy - Hot-reload environments with MCP over HTTP."""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
from pathlib import Path

import click
import toml
from fastmcp import FastMCP


def get_image_name(directory: str | Path, image_override: str | None = None) -> tuple[str, str]:
    """
    Resolve image name with source tracking.

        Returns:
        Tuple of (image_name, source) where source is "override", "cache", or "auto"
    """
    if image_override:
        return image_override, "override"
    
    # Check pyproject.toml
    pyproject_path = Path(directory) / "pyproject.toml"
    if pyproject_path.exists():
        try:
            with open(pyproject_path) as f:
                config = toml.load(f)
            if config.get("tool", {}).get("hud", {}).get("image"):
                return config["tool"]["hud"]["image"], "cache"
        except Exception:
            pass  # Fall through to auto-generation
    
    # Auto-generate with :dev tag
    dir_path = Path(directory).resolve()  # Get absolute path first
    dir_name = dir_path.name
    if not dir_name or dir_name == '.':
        # If we're in root or have empty name, use parent directory
        dir_name = dir_path.parent.name
    clean_name = dir_name.replace("_", "-")
    return f"hud-{clean_name}:dev", "auto"


def update_pyproject_toml(directory: str | Path, image_name: str, silent: bool = False) -> None:
    """Update pyproject.toml with image name."""
    pyproject_path = Path(directory) / "pyproject.toml"
    if pyproject_path.exists():
        try:
            with open(pyproject_path) as f:
                config = toml.load(f)
            
            # Ensure [tool.hud] exists
            if "tool" not in config:
                config["tool"] = {}
            if "hud" not in config["tool"]:
                config["tool"]["hud"] = {}
            
            # Update image name
            config["tool"]["hud"]["image"] = image_name
            
            # Write back
            with open(pyproject_path, "w") as f:
                toml.dump(config, f)
            
            if not silent:
                click.echo(f"✅ Updated pyproject.toml with image: {image_name}")
        except Exception as e:
            if not silent:
                click.echo(f"⚠️  Could not update pyproject.toml: {e}")


def build_and_update(directory: str | Path, image_name: str, no_cache: bool = False) -> None:
    """Build Docker image and update pyproject.toml."""
    from hud.utils.design import HUDDesign
    design = HUDDesign()
    
    build_cmd = ["docker", "build", "-t", image_name]
    if no_cache:
        build_cmd.append("--no-cache")
    build_cmd.append(str(directory))
    
    design.info(f"🔨 Building image: {image_name}{' (no cache)' if no_cache else ''}")
    design.info("")  # Empty line before Docker output
    
    # Just run Docker build directly - it has its own nice live display
    result = subprocess.run(build_cmd)
    
    if result.returncode == 0:
        design.info("")  # Empty line after Docker output
        design.success(f"Build successful! Image: {image_name}")
        # Update pyproject.toml (silently since we already showed success)
        update_pyproject_toml(directory, image_name, silent=True)
    else:
        design.error("Build failed!")
        raise click.Abort()


from .docker_utils import get_docker_cmd, inject_supervisor, image_exists


def create_proxy_server(
    directory: str | Path,
    image_name: str,
    no_reload: bool = False,
    verbose: bool = False,
    docker_args: list[str] | None = None,
    interactive: bool = False
) -> FastMCP:
    """Create an HTTP proxy server that forwards to Docker container with hot-reload."""
    src_path = Path(directory) / "src"
    
    # Get the original CMD from the image
    original_cmd = get_docker_cmd(image_name)
    if not original_cmd:
        click.echo(f"⚠️  Could not extract CMD from {image_name}, using default")
        original_cmd = ["python", "-m", "hud_controller.server"]
    
    # Generate container name from image
    container_name = f"{image_name.replace(':', '-').replace('/', '-')}"
    
    # Build the docker run command
    docker_cmd = [
        "docker", "run", "--rm", "-i",
        "--name", container_name,
        "-v", f"{src_path.absolute()}:/app/src:rw",
        "-e", "PYTHONPATH=/app/src",
    ]
    
    # Add user-provided Docker arguments
    if docker_args:
        docker_cmd.extend(docker_args)
    
    # Disable hot-reload if interactive mode is enabled
    if interactive:
        no_reload = True
    
    if not no_reload:
        # Inject our supervisor into the CMD
        modified_cmd = inject_supervisor(original_cmd)
        docker_cmd.extend(["--entrypoint", modified_cmd[0]])
        docker_cmd.append(image_name)
        docker_cmd.extend(modified_cmd[1:])
    else:
        # No reload - use original CMD
        docker_cmd.append(image_name)
    
    # Create configuration following MCPConfig schema
    config = {
        "mcpServers": {
            "default": {
                "command": docker_cmd[0],
                "args": docker_cmd[1:] if len(docker_cmd) > 1 else []
                # transport defaults to stdio
            }
        }
    }
    
    # Debug output - only if verbose
    if verbose:
        if not no_reload:
            click.echo(f"📁 Watching: /app/src for changes", err=True)
        else:
            click.echo(f"🔧 Container will run without hot-reload", err=True)
        click.echo(f"📊 docker logs -f {container_name}", err=True)
    
    # Create the HTTP proxy server using config
    proxy = FastMCP.as_proxy(
        config,
        name=f"HUD Dev Proxy - {image_name}"
    )
    
    return proxy


async def start_mcp_proxy(
    directory: str | Path,
    image_name: str,
    transport: str,
    port: int,
    no_reload: bool = False,
    verbose: bool = False,
    inspector: bool = False,
    no_logs: bool = False,
    interactive: bool = False,
    docker_args: list[str] | None = None
) -> None:
    """Start the MCP development proxy server."""
    # Suppress FastMCP's verbose output FIRST
    import logging
    import os
    import sys
    import subprocess
    import asyncio
    from .utils import find_free_port
    
    # Always disable the banner - we have our own output
    os.environ["FASTMCP_DISABLE_BANNER"] = "1"
    
    # Configure logging BEFORE creating proxy
    if not verbose:
        # Create a filter to block the specific "Starting MCP server" message
        class BlockStartingMCPFilter(logging.Filter):
            def filter(self, record):
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
        block_filter = BlockStartingMCPFilter()
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
            "hud.server.server"
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
        click.echo(f"❌ Source directory not found: {src_path}", err=(transport == "stdio"))
        raise click.Abort()
    
    # Extract container name from the proxy configuration
    container_name = f"{image_name.replace(':', '-').replace('/', '-')}"
    
    # Remove any existing container with the same name (silently)
    # Note: The proxy creates containers on-demand when clients connect
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False  # Don't raise error if container doesn't exist
        )
    except Exception:
        pass  # Ignore errors if container doesn't exist

    if transport == "stdio":
        if verbose:
            click.echo(f"🔌 Starting stdio proxy (each connection gets its own container)", err=True)
    else:
        # Find available port for HTTP
        actual_port = find_free_port(port)
        if actual_port is None:
            click.echo(f"❌ No available ports found starting from {port}")
            raise click.Abort()
        
        if actual_port != port and verbose:
            click.echo(f"⚠️  Port {port} in use, using port {actual_port} instead")
        
        # Launch MCP Inspector if requested
        if inspector:
            server_url = f"http://localhost:{actual_port}/mcp"
            
            # Function to launch inspector in background
            async def launch_inspector():
                """Launch MCP Inspector and capture its output to extract the URL."""
                # Wait for server to be ready
                await asyncio.sleep(3)
                
                try:
                    import platform
                    import urllib.parse
                    
                    # Build the direct URL with query params to auto-connect
                    encoded_url = urllib.parse.quote(server_url)
                    inspector_url = f"http://localhost:6274/?transport=streamable-http&serverUrl={encoded_url}"
                    
                    # Print inspector info cleanly
                    from hud.utils.design import HUDDesign
                    inspector_design = HUDDesign(stderr=(transport == "stdio"))
                    inspector_design.section_title("MCP Inspector")
                    inspector_design.link(inspector_url)
                    
                    # Set environment to disable auth (for development only)
                    env = os.environ.copy()
                    env["DANGEROUSLY_OMIT_AUTH"] = "true"
                    env["MCP_AUTO_OPEN_ENABLED"] = "true"
                    
                    # Launch inspector
                    cmd = ["npx", "--yes", "@modelcontextprotocol/inspector"]
                    
                    # Run in background, suppressing output to avoid log interference
                    if platform.system() == "Windows":
                        subprocess.Popen(cmd, env=env, shell=True, 
                                       stdout=subprocess.DEVNULL, 
                                       stderr=subprocess.DEVNULL)
                    else:
                        subprocess.Popen(cmd, env=env,
                                       stdout=subprocess.DEVNULL, 
                                       stderr=subprocess.DEVNULL)
                    

                    
                except (FileNotFoundError, Exception):
                    # Silently fail - inspector is optional
                    pass
            
            # Launch inspector asynchronously so it doesn't block
            asyncio.create_task(launch_inspector())
        
        # Launch interactive mode if requested
        if interactive:
            if transport != "http":
                from hud.utils.design import HUDDesign
                interactive_design = HUDDesign(stderr=True)
                interactive_design.warning("Interactive mode only works with HTTP transport")
            else:
                server_url = f"http://localhost:{actual_port}/mcp"
                
                # Function to launch interactive mode in a separate thread
                def launch_interactive_thread():
                    """Launch interactive testing mode in a separate thread."""
                    import time
                    import threading
                    
                    # Wait for server to be ready
                    time.sleep(3)
                    
                    try:
                        from hud.utils.design import HUDDesign
                        interactive_design = HUDDesign(stderr=(transport == "stdio"))
                        interactive_design.section_title("Interactive Mode")
                        interactive_design.info("Starting interactive testing mode...")
                        interactive_design.info("Press Ctrl+C in the interactive session to exit")
                        
                        # Import and run interactive mode in a new event loop
                        from .interactive import run_interactive_mode
                        
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
                            print(f"Interactive mode error: {e}", file=sys.stderr)
                
                # Launch interactive mode in a separate thread
                import threading
                interactive_thread = threading.Thread(target=launch_interactive_thread, daemon=True)
                interactive_thread.start()
    
    # Function to stream Docker logs
    async def stream_docker_logs():
        """Stream Docker container logs asynchronously."""
        # Import design system for consistent output
        from hud.utils.design import HUDDesign
        log_design = HUDDesign(stderr=(transport == "stdio"))
        
        # Always show waiting message
        log_design.info("")  # Empty line for spacing
        log_design.progress_message("⏳ Waiting for first client connection to start container...")
        
        # Keep trying to stream logs - container is created on demand
        has_shown_started = False
        while True:
            # Check if container exists first (silently)
            check_result = await asyncio.create_subprocess_exec(
                "docker", "ps", "--format", "{{.Names}}", "--filter", f"name={container_name}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL
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
                    "docker", "logs", "-f", container_name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT  # Combine streams for simplicity
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
                        log_design.console.print(f"[rgb(192,150,12)]■[/rgb(192,150,12)] {decoded_line}", highlight=False)
                
                # Process ended - container might have been removed
                await process.wait()
                
                # Check if container still exists
                await asyncio.sleep(1)
                continue  # Loop back to check if container exists

            except Exception as e:
                # Some unexpected error
                if verbose:
                    log_design.warning(f"Failed to stream logs: {e}")
                await asyncio.sleep(1)
        
    # CRITICAL: Create proxy AFTER all logging setup to prevent it from resetting logging config
    # This is important because FastMCP might initialize loggers during creation
    proxy = create_proxy_server(directory, image_name, no_reload, verbose, docker_args or [], interactive)
    
    # One more attempt to suppress the FastMCP server log
    if not verbose:
        # Re-apply the filter in case new handlers were created
        class BlockStartingMCPFilter(logging.Filter):
            def filter(self, record):
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
            "hud.server.server"
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
                transport="stdio",
                log_level="ERROR" if not verbose else "INFO",
                show_banner=False
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
                        host="0.0.0.0",
                        port=actual_port,
                        path="/mcp",  # Serve at /mcp endpoint
                        log_level="ERROR",
                        show_banner=False
                    )
            else:
                await proxy.run_async(
                    transport="http",
                    host="0.0.0.0",
                    port=actual_port,
                    path="/mcp",  # Serve at /mcp endpoint
                    log_level="INFO",
                    show_banner=False
                )
    except KeyboardInterrupt:
        from hud.utils.design import HUDDesign
        shutdown_design = HUDDesign(stderr=(transport == "stdio"))
        shutdown_design.info("\n👋 Shutting down...")
        
        # Show next steps tutorial
        if not interactive:  # Only show if not in interactive mode
            shutdown_design.section_title("Next Steps")
            shutdown_design.info("🏗️  Ready to test with real agents? Run:")
            shutdown_design.info(f"    [cyan]hud build {directory}[/cyan]")
            shutdown_design.info("")
            shutdown_design.info("This will:")
            shutdown_design.info("  1. Build your environment image")
            shutdown_design.info("  2. Generate a hud.lock.yaml file")
            shutdown_design.info("  3. Prepare it for testing with agents")
            shutdown_design.info("")
            shutdown_design.info("Then you can:")
            shutdown_design.info("  • Test locally: [cyan]hud run <image>[/cyan]")
            shutdown_design.info("  • Push to registry: [cyan]hud push --image <registry/name>[/cyan]")
    except Exception as e:
        # Suppress the graceful shutdown error and other FastMCP/uvicorn internal errors
        error_msg = str(e)
        if not any(x in error_msg for x in [
            "timeout graceful shutdown exceeded",
            "Cancel 0 running task(s)",
            "Application shutdown complete"
        ]):
            from hud.utils.design import HUDDesign
            shutdown_design = HUDDesign(stderr=(transport == "stdio"))
            shutdown_design.error(f"Unexpected error: {e}")
    finally:
        # Cancel log streaming task if it exists
        if log_task and not log_task.done():
            log_task.cancel()
            try:
                await log_task
            except asyncio.CancelledError:
                pass


def run_mcp_dev_server(
    directory: str = '.',
    image: str | None = None,
    build: bool = False,
    no_cache: bool = False,
    transport: str = 'http',
    port: int = 8765,
    no_reload: bool = False,
    verbose: bool = False,
    inspector: bool = False,
    no_logs: bool = False,
    interactive: bool = False,
    docker_args: list[str] | None = None
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
    from hud.utils.design import HUDDesign
    
    design = HUDDesign(stderr=(transport == "stdio"))
    
    # Ensure directory exists
    if not Path(directory).exists():
        design.error(f"Directory not found: {directory}")
        raise click.Abort()
        
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
            raise click.Abort()
    
    # Generate server name from image
    server_name = resolved_image.split(':')[0] if ':' in resolved_image else resolved_image
    
    # For HTTP transport, find available port first
    actual_port = port
    if transport == "http":
        from .utils import find_free_port
        actual_port = find_free_port(port)
        if actual_port is None:
            design.error(f"No available ports found starting from {port}")
            raise click.Abort()
        if actual_port != port and verbose:
            design.warning(f"Port {port} in use, using port {actual_port}")
    
    # Create config
    if transport == "stdio":
        server_config = {"command": "hud", "args": ["dev", directory, "--transport", "stdio"]}
    else:
        server_config = {"url": f"http://localhost:{actual_port}/mcp"}
    
    # For the deeplink, we only need the server config
    server_config_json = json.dumps(server_config, indent=2)
    config_base64 = base64.b64encode(server_config_json.encode()).decode()
    
    # Generate deeplink
    deeplink = f"cursor://anysphere.cursor-deeplink/mcp/install?name={server_name}&config={config_base64}"
    
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
    design.section_title("Connect to Cursor (be careful with multiple windows as that may interfere with the proxy)")
    design.link(deeplink)
    design.info("")  # Empty line
    
    # Start the proxy (pass original port, start_mcp_proxy will find actual port again)
    asyncio.run(start_mcp_proxy(directory, resolved_image, transport, port, no_reload, verbose, inspector, no_logs, interactive, docker_args or []))