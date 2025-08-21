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


def image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", image_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return result.returncode == 0


def create_proxy_server(
    directory: str | Path,
    image_name: str,
    no_reload: bool = False
) -> FastMCP:
    """Create an HTTP proxy server that forwards to stdio reloaderoo+docker."""
    # Build the command that will be proxied
    cmd = []
    if not no_reload:
        import shutil
        npx_cmd = shutil.which("npx")
        if not npx_cmd:
            # Fallback to npx if shutil.which fails (shouldn't happen if we got here)
            npx_cmd = "npx"
        cmd.extend([npx_cmd, "reloaderoo", "--"])
    
    src_path = Path(directory) / "src"
    cmd.extend([
        "docker", "run", "--rm", "-i",
        "-v", f"{src_path.absolute()}:/app/src:rw",
        "-e", "PYTHONPATH=/app/src",
        image_name
    ])
    
    # Create configuration following MCPConfig schema
    config = {
        "mcpServers": {
            "default": {
                "command": cmd[0],
                "args": cmd[1:] if len(cmd) > 1 else []
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
    no_reload: bool = False
) -> None:
    """Start the MCP development proxy server."""
    # Ensure src directory exists
    src_path = Path(directory) / "src"
    if not src_path.exists():
        click.echo(f"❌ Source directory not found: {src_path}")
        raise click.Abort()
    
    # Create the proxy server
    proxy = create_proxy_server(directory, image_name, no_reload)
    
    # Run the HTTP server
    click.echo(f"\n🌐 Starting HTTP proxy on port {port}...")
    click.echo(f"🔄 Files in {src_path} will trigger reload")
    click.echo(f"\n📡 Proxy is ready! Press Ctrl+C to stop.\n")
    
    try:
        # Run the proxy with HTTP transport
        await proxy.run_async(
            transport="http",
            host="0.0.0.0",
            port=port,
            path="/mcp"  # Serve at /mcp endpoint
        )
    except KeyboardInterrupt:
        click.echo("\n👋 Shutting down...")


def run_mcp_dev_server(
    directory: str = '.',
    image: str | None = None,
    build: bool = False,
    no_cache: bool = False,
    port: int = 8765,
    no_reload: bool = False
) -> None:
    """Run MCP development server with hot-reload.
    
    This command starts a development proxy that:
    - Auto-detects or builds Docker images
    - Mounts local source code for hot-reload
    - Exposes an HTTP endpoint for MCP clients
    
    Examples:
        hud mcp .                    # Auto-detect image from directory
        hud mcp . --build            # Build image first
        hud mcp . --image custom:tag # Use specific image
        hud mcp . --no-cache         # Force clean rebuild
    """
    # Ensure directory exists
    if not Path(directory).exists():
        click.echo(f"❌ Directory not found: {directory}")
        raise click.Abort()
        
    # Check if reloaderoo is available (unless --no-reload)
    if not no_reload:
        import shutil
        npx_cmd = shutil.which("npx")
        if not npx_cmd:
            click.echo("❌ npx not found. Install Node.js or use --no-reload")
            click.echo("💡 To install: https://nodejs.org/")
            click.echo("💡 Or use: hud mcp . --no-reload")
            raise click.Abort()
        
        # Verify it works
        result = subprocess.run([npx_cmd, "--version"], capture_output=True)
        if result.returncode != 0:
            click.echo("❌ npx found but not working properly")
            raise click.Abort()
    
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
    
    # Start the server
    click.echo(f"📁 Source: {directory}/src → /app/src")
    click.echo(f"🔄 Hot-reload: {'enabled (reloaderoo)' if not no_reload else 'disabled'}")
    
    # Show config
    config = {"url": f"http://localhost:{port}/mcp"}
    config_json = json.dumps(config, indent=2)
    config_base64 = base64.b64encode(config_json.encode()).decode()
    
    click.echo(f"\n✨ Add to Cursor:\n")
    click.echo(f'"{server_name}": {config_json}')
    
    # Generate deeplink
    deeplink = f"cursor://anysphere.cursor-deeplink/mcp/install?name={server_name}&config={config_base64}"
    click.echo(f"🔗 Quick install: {deeplink}\n")
    
    # Start the proxy
    asyncio.run(start_mcp_proxy(directory, resolved_image, port, no_reload))