"""Run Docker images as MCP servers."""

import asyncio
import subprocess
import sys
from pathlib import Path

import click
from fastmcp import FastMCP


def run_stdio_server(image: str, docker_args: list[str], verbose: bool) -> None:
    """Run Docker image as stdio MCP server (direct passthrough)."""
    # Build docker command
    docker_cmd = ["docker", "run", "--rm", "-i"] + docker_args + [image]
    
    if verbose:
        click.echo(f"🐳 Running: {' '.join(docker_cmd)}")
    
    # Run docker directly with stdio passthrough
    try:
        result = subprocess.run(docker_cmd, stdin=sys.stdin)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        click.echo("\n👋 Shutting down...")
        sys.exit(0)
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)


async def run_http_server(image: str, docker_args: list[str], port: int, verbose: bool) -> None:
    """Run Docker image as HTTP MCP server (proxy mode)."""
    from .utils import find_free_port
    
    # Find available port
    actual_port = find_free_port(port)
    if actual_port is None:
        click.echo(f"❌ No available ports found starting from {port}")
        return
    
    if actual_port != port:
        click.echo(f"⚠️  Port {port} in use, using port {actual_port} instead")
    
    # Generate container name
    container_name = f"run-{image.replace(':', '-').replace('/', '-')}"
    
    # Build docker command for stdio container
    docker_cmd = [
        "docker", "run", "--rm", "-i",
        "--name", container_name,
    ] + docker_args + [image]
    
    # Create MCP config for stdio transport
    config = {
        "mcpServers": {
            "default": {
                "command": docker_cmd[0],
                "args": docker_cmd[1:] if len(docker_cmd) > 1 else []
                # transport defaults to stdio
            }
        }
    }
    
    # Set up logging suppression
    import logging
    import os
    
    os.environ["FASTMCP_DISABLE_BANNER"] = "1"
    
    if not verbose:
        logging.getLogger("fastmcp").setLevel(logging.ERROR)
        logging.getLogger("mcp").setLevel(logging.ERROR)
        logging.getLogger("uvicorn").setLevel(logging.ERROR)
        logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
        logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
        
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Create HTTP proxy
    proxy = FastMCP.as_proxy(
        config,
        name=f"HUD Run - {image}"
    )
    
    click.echo(f"🌐 Starting HTTP proxy on port {actual_port}")
    click.echo(f"🔗 Server URL: http://localhost:{actual_port}/mcp")
    click.echo(f"📊 docker logs -f {container_name}")
    click.echo(f"⏹️  Press Ctrl+C to stop")
    
    try:
        await proxy.run_async(
            transport="http",
            host="0.0.0.0",
            port=actual_port,
            path="/mcp",
            log_level="error" if not verbose else "info"
        )
    except KeyboardInterrupt:
        click.echo("\n👋 Shutting down...")


def run_mcp_server(image: str, docker_args: list[str], transport: str, port: int, verbose: bool) -> None:
    """Run Docker image as MCP server with specified transport."""
    if transport == "stdio":
        run_stdio_server(image, docker_args, verbose)
    elif transport == "http":
        asyncio.run(run_http_server(image, docker_args, port, verbose))
    else:
        click.echo(f"❌ Unknown transport: {transport}")
        sys.exit(1)
