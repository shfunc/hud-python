"""MCP Development Proxy - Hot-reload environments with MCP over HTTP."""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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


def update_pyproject_toml(directory: str | Path, image_name: str) -> None:
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
            
            click.echo(f"✅ Updated pyproject.toml with image: {image_name}")
        except Exception as e:
            click.echo(f"⚠️  Could not update pyproject.toml: {e}")


def build_and_update(directory: str | Path, image_name: str, no_cache: bool = False) -> None:
    """Build Docker image and update pyproject.toml."""
    build_cmd = ["docker", "build", "-t", image_name]
    if no_cache:
        build_cmd.append("--no-cache")
    build_cmd.append(str(directory))
    
    click.echo(f"🔨 Building image: {image_name}{' (no cache)' if no_cache else ''}")
    
    result = subprocess.run(build_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        click.echo("✅ Build successful!")
        # Update pyproject.toml
        update_pyproject_toml(directory, image_name)
    else:
        click.echo(f"❌ Build failed:\n{result.stderr}")
        raise click.Abort()


from .docker_utils import get_docker_cmd, inject_supervisor, image_exists


def create_proxy_server(
    directory: str | Path,
    image_name: str,
    no_reload: bool = False
) -> FastMCP:
    """Create an HTTP proxy server that forwards to Docker container with hot-reload."""
    src_path = Path(directory) / "src"
    
    # Get the original CMD from the image
    original_cmd = get_docker_cmd(image_name)
    if not original_cmd:
        click.echo(f"⚠️  Could not extract CMD from {image_name}, using default")
        original_cmd = ["python", "-m", "hud_controller.server"]
    
    # Build the docker run command
    docker_cmd = [
        "docker", "run", "--rm", "-i",
        "-v", f"{src_path.absolute()}:/app/src:rw",
        "-e", "PYTHONPATH=/app/src",
    ]
    
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
    
    # Create the HTTP proxy server using config
    proxy = FastMCP.as_proxy(
        config,
        name=f"HUD Dev Proxy - {image_name}"
    )
    
    return proxy


async def start_mcp_proxy(
    directory: str | Path,
    image_name: str,
    port: int,
    no_reload: bool = False,
    verbose: bool = False
) -> None:
    """Start the MCP development proxy server."""
    # Ensure src directory exists
    src_path = Path(directory) / "src"
    if not src_path.exists():
        click.echo(f"❌ Source directory not found: {src_path}")
        raise click.Abort()
    
    # Create the proxy server
    proxy = create_proxy_server(directory, image_name, no_reload)
    
    # Suppress FastMCP's verbose output in quiet mode
    import logging
    if not verbose:
        # Suppress uvicorn logs
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        # Suppress FastMCP banner by setting environment variable
        import os
        os.environ["FASTMCP_DISABLE_BANNER"] = "1"

    click.echo(f"🌐 Reloading proxy live, press Ctrl+C to stop")
    
    try:
        # Run the proxy with HTTP transport
        await proxy.run_async(
            transport="http",
            host="0.0.0.0",
            port=port,
            path="/mcp",  # Serve at /mcp endpoint
            log_level="warning" if not verbose else "info"
        )
    except KeyboardInterrupt:
        if not verbose:
            click.echo("\n👋 Shutting down...")
        else:
            click.echo("\n✅ MCP proxy stopped")


def run_mcp_dev_server(
    directory: str = '.',
    image: str | None = None,
    build: bool = False,
    no_cache: bool = False,
    port: int = 8765,
    no_reload: bool = False,
    verbose: bool = False
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
        click.echo(f"❌ Directory not found: {directory}")
        raise click.Abort()
        
    # No external dependencies needed for hot-reload anymore!
    
    # Resolve image name
    resolved_image, source = get_image_name(directory, image)
    
    if source == "cache":
        click.echo(f"📦 Using cached image from pyproject.toml: {resolved_image}")
    elif source == "auto":
        click.echo(f"🔧 Auto-generated image name: {resolved_image}")
        # Update pyproject.toml with auto-generated name
        update_pyproject_toml(directory, resolved_image)
    elif source == "override":
        click.echo(f"🎯 Using specified image: {resolved_image}")
    
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
    
    # Show config
    config = {"url": f"http://localhost:{port}/mcp"}
    config_json = json.dumps(config, indent=2)
    config_base64 = base64.b64encode(config_json.encode()).decode()
    
    click.echo(f'"{server_name}": {config_json}')
    
    # Generate deeplink
    deeplink = f"cursor://anysphere.cursor-deeplink/mcp/install?name={server_name}&config={config_base64}"
    click.echo(f"✨ Add to Cursor: {deeplink}")
    
    # Start the proxy
    asyncio.run(start_mcp_proxy(directory, resolved_image, port, no_reload, verbose))