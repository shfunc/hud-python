"""Build HUD environments and generate lock files."""

import asyncio
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import typer
import yaml
from rich.console import Console
from rich.panel import Panel

from hud.clients import MCPClient
from hud.utils.design import HUDDesign
from hud.version import __version__ as hud_version

console = Console()


def get_docker_image_digest(image: str) -> Optional[str]:
    """Get the digest of a Docker image."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.RepoDigests}}", image],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse the output - it's in format [repo@sha256:digest]
        digests = result.stdout.strip()
        if digests and digests != "[]":
            # Extract the first digest
            digest_list = eval(digests)  # Safe since it's from docker
            if digest_list:
                # Return full image reference with digest
                return digest_list[0]
    except Exception:
        pass
    return None


def get_docker_image_id(image: str) -> Optional[str]:
    """Get the ID of a Docker image."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            check=True
        )
        image_id = result.stdout.strip()
        if image_id:
            return image_id
        return None
    except Exception as e:
        # Don't log here to avoid import issues
        return None


def extract_env_vars_from_dockerfile(dockerfile_path: Path) -> tuple[list[str], list[str]]:
    """Extract required and optional environment variables from Dockerfile."""
    required = []
    optional = []
    
    if not dockerfile_path.exists():
        return required, optional
    
    # Simple parsing - look for ENV directives
    # This is a basic implementation - could be enhanced
    content = dockerfile_path.read_text()
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("ENV "):
            # Basic parsing - this could be more sophisticated
            parts = line[4:].strip().split("=", 1)
            if len(parts) == 2 and not parts[1].strip():
                # No default value = required
                required.append(parts[0])
            elif len(parts) == 1:
                # No equals sign = required
                required.append(parts[0])
    
    return required, optional


async def analyze_mcp_environment(image: str, verbose: bool = False) -> dict[str, Any]:
    """Analyze an MCP environment to extract metadata."""
    design = HUDDesign()
    
    # Build Docker command to run the image
    docker_cmd = ["docker", "run", "--rm", "-i", image]
    
    # Create MCP config
    config = {
        "server": {
            "command": docker_cmd[0],
            "args": docker_cmd[1:] if len(docker_cmd) > 1 else []
        }
    }
    
    # Initialize client and measure timing
    start_time = time.time()
    client = MCPClient(mcp_config=config, verbose=verbose, auto_trace=False)
    initialized = False
    
    try:
        if verbose:
            design.info(f"Initializing MCP client with command: {' '.join(docker_cmd)}")
        
        await client.initialize()
        initialized = True
        initialize_ms = int((time.time() - start_time) * 1000)
        
        # Get tools
        tools = await client.list_tools()
        
        # Extract tool information
        tool_info = []
        for tool in tools:
            tool_dict = {
                "name": tool.name,
                "description": tool.description
            }
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                tool_dict["inputSchema"] = tool.inputSchema
            tool_info.append(tool_dict)
        
        return {
            "initializeMs": initialize_ms,
            "toolCount": len(tools),
            "tools": tool_info,
            "success": True
        }
    except Exception as e:
        import traceback
        error_msg = str(e)
        if verbose:
            design.error(f"Failed to analyze environment: {error_msg}")
            design.error(f"Traceback:\n{traceback.format_exc()}")
        
        # Common issues
        if "Connection reset" in error_msg or "EOF" in error_msg:
            design.warning("The MCP server may have crashed on startup. Check your server.py for errors.")
        elif "timeout" in error_msg:
            design.warning("The MCP server took too long to initialize. It might need more startup time.")
        
        return {
            "initializeMs": 0,
            "toolCount": 0,
            "tools": [],
            "success": False,
            "error": error_msg
        }
    finally:
        # Only shutdown if we successfully initialized
        if initialized:
            try:
                await client.shutdown()
            except Exception:
                # Ignore shutdown errors
                pass


def build_docker_image(directory: Path, tag: str, no_cache: bool = False, verbose: bool = False) -> bool:
    """Build a Docker image from a directory."""
    design = HUDDesign()
    
    # Check if Dockerfile exists
    dockerfile = directory / "Dockerfile"
    if not dockerfile.exists():
        design.error(f"No Dockerfile found in {directory}")
        return False
    
    # Build command
    cmd = ["docker", "build", "-t", tag]
    if no_cache:
        cmd.append("--no-cache")
    cmd.append(str(directory))
    
    # Always show build output
    design.info(f"Running: {' '.join(cmd)}")
    
    try:
        # Run with real-time output, handling encoding issues on Windows
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'  # Replace invalid chars instead of failing
        )
        
        # Stream output
        for line in process.stdout:
            print(line.rstrip())
        
        process.wait()
        
        if process.returncode != 0:
            return False
        return True
    except Exception as e:
        design.error(f"Build error: {e}")
        return False


def build_environment(
    directory: str = ".",
    tag: Optional[str] = None,
    no_cache: bool = False,
    verbose: bool = False
) -> None:
    """Build a HUD environment and generate lock file."""
    design = HUDDesign()
    design.header("HUD Environment Build")
    
    # Resolve directory
    env_dir = Path(directory).resolve()
    if not env_dir.exists():
        design.error(f"Directory not found: {directory}")
        raise typer.Exit(1)
    
    # Check for pyproject.toml
    pyproject_path = env_dir / "pyproject.toml"
    if not pyproject_path.exists():
        design.error(f"No pyproject.toml found in {directory}")
        raise typer.Exit(1)
    
    # Read pyproject.toml to get image name
    try:
        import toml
        pyproject = toml.load(pyproject_path)
        default_image = pyproject.get("tool", {}).get("hud", {}).get("image", None)
        if not default_image:
            # Generate default from directory name
            default_image = f"{env_dir.name}:dev"
    except Exception:
        default_image = f"{env_dir.name}:dev"
    
    # Use provided tag or default
    if not tag:
        tag = default_image
    
    # Build temporary image first
    temp_tag = f"hud-build-temp:{int(time.time())}"
    
    design.progress_message(f"Building Docker image: {temp_tag}")
    
    # Build the image
    if not build_docker_image(env_dir, temp_tag, no_cache, verbose):
        design.error("Docker build failed")
        raise typer.Exit(1)
    
    design.success(f"Built temporary image: {temp_tag}")
    
    # Analyze the environment
    design.progress_message("Analyzing MCP environment...")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        analysis = loop.run_until_complete(analyze_mcp_environment(temp_tag, verbose))
    finally:
        loop.close()
    
    if not analysis["success"]:
        design.error("Failed to analyze MCP environment")
        if "error" in analysis:
            design.error(f"Error: {analysis['error']}")
        
        # Provide helpful debugging tips
        design.section_title("Debugging Tips")
        console.print("1. Test your server directly:")
        console.print(f"   [cyan]docker run --rm -it {temp_tag}[/cyan]")
        console.print("   (Should see MCP initialization output)")
        console.print("")
        console.print("2. Check for common issues:")
        console.print("   - Server crashes on startup")
        console.print("   - Missing dependencies")
        console.print("   - Syntax errors in server.py")
        console.print("")
        console.print("3. Run with verbose mode:")
        console.print("   [cyan]hud build . --verbose[/cyan]")
        
        raise typer.Exit(1)
    
    design.success(f"Analyzed environment: {analysis['toolCount']} tools found")
    
    # Extract environment variables from Dockerfile
    dockerfile_path = env_dir / "Dockerfile"
    required_env, optional_env = extract_env_vars_from_dockerfile(dockerfile_path)
    
    # Create lock file content - minimal and useful
    lock_content = {
        "version": "1.0",  # Lock file format version
        "image": tag,  # Will be updated with ID/digest later
        "build": {
            "generatedAt": datetime.utcnow().isoformat() + "Z",
            "hudVersion": hud_version,
            "directory": str(env_dir.name)
        },
        "environment": {
            "initializeMs": analysis["initializeMs"],
            "toolCount": analysis["toolCount"],
        }
    }
    
    # Only add environment variables if they exist
    if required_env or optional_env:
        lock_content["environment"]["variables"] = {}
        if required_env:
            lock_content["environment"]["variables"]["required"] = required_env
        if optional_env:
            lock_content["environment"]["variables"]["optional"] = optional_env
    
    # Add tool summary (not full schemas to keep it concise)
    if analysis["tools"]:
        lock_content["tools"] = [
            {"name": tool["name"], "description": tool.get("description", "")}
            for tool in analysis["tools"]
        ]
    
    # Write lock file
    lock_path = env_dir / "hud.lock.yaml"
    with open(lock_path, "w") as f:
        yaml.dump(lock_content, f, default_flow_style=False, sort_keys=False)
    
    design.success(f"Created lock file: hud.lock.yaml")
    
    # Calculate lock file hash
    lock_content_str = yaml.dump(lock_content, default_flow_style=False, sort_keys=True)
    lock_hash = hashlib.sha256(lock_content_str.encode()).hexdigest()
    lock_size = len(lock_content_str)
    
    # Rebuild with label containing lock file hash
    design.progress_message(f"Rebuilding with lock file metadata...")
    
    # Build final image with label (uses cache from first build)
    label_cmd = [
        "docker", "build",
        "--label", f"org.hud.manifest.head={lock_hash}:{lock_size}",
        "-t", tag,
        str(env_dir)
    ]
    
    # Run rebuild with proper encoding
    process = subprocess.Popen(
        label_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    # Stream output if verbose
    if verbose:
        for line in process.stdout:
            print(line.rstrip())
    else:
        # Just consume output to avoid blocking
        process.stdout.read()
    
    process.wait()
    
    if process.returncode != 0:
        design.error("Failed to rebuild with label")
        raise typer.Exit(1)
    
    design.success(f"Built final image with lock file metadata")
    
    # NOW get the image ID after the final build
    image_id = get_docker_image_id(tag)
    if image_id:
        # For local builds, store the image ID
        # Docker IDs come as sha256:hash, we want tag@sha256:hash
        if image_id.startswith("sha256:"):
            lock_content["image"] = f"{tag}@{image_id}"
        else:
            lock_content["image"] = f"{tag}@sha256:{image_id}"
        
        # Update the lock file with the new image reference
        with open(lock_path, "w") as f:
            yaml.dump(lock_content, f, default_flow_style=False, sort_keys=False)
        
        design.success(f"Updated lock file with image ID")
    else:
        design.warning("Could not retrieve image ID for lock file")
    
    # Remove temp image after we're done
    subprocess.run(["docker", "rmi", temp_tag], capture_output=True)
    
    # Print summary
    design.section_title("Build Complete")
    
    # Show the actual image reference from the lock file
    final_image_ref = lock_content.get("image", tag)
    console.print(f"[green]✓[/green] Local image  : {final_image_ref}")
    console.print(f"[green]✓[/green] Lock file    : hud.lock.yaml")
    console.print(f"[green]✓[/green] Tools found  : {analysis['toolCount']}")
    
    design.section_title("Next Steps")
    console.print("Test locally:")
    console.print(f"  [cyan]hud dev[/cyan]           # Hot-reload development")
    console.print(f"  [cyan]hud run {tag}[/cyan]  # Run the built image")
    console.print("")
    console.print("Publish to registry:")
    console.print(f"  [cyan]hud push --image <registry>/{tag}[/cyan]")
    console.print("")
    console.print("The lock file can be used to reproduce this exact environment.")


def build_command(
    directory: str = typer.Argument(".", help="Environment directory to build"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Docker image tag (default: from pyproject.toml)"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Build without Docker cache"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Build a HUD environment and generate lock file."""
    build_environment(directory, tag, no_cache, verbose)
