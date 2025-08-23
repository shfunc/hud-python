"""Fast metadata analysis functions for hud analyze."""

from pathlib import Path
from typing import Optional

import requests
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from hud.settings import settings
from hud.utils.design import HUDDesign

console = Console()
design = HUDDesign()


def fetch_lock_from_registry(reference: str) -> Optional[dict]:
    """Fetch lock file from HUD registry."""
    try:
        # Reference should be org/name:tag format
        # If no tag specified, append :latest
        if "/" in reference and ":" not in reference:
            reference = f"{reference}:latest"
        
        registry_url = f"{settings.hud_telemetry_url.rstrip('/')}/registry/envs/{reference}"
        
        headers = {}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        
        response = requests.get(
            registry_url,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            # Parse the lock YAML from the response
            if "lock" in data:
                return yaml.safe_load(data["lock"])
            elif "lock_data" in data:
                return data["lock_data"]
            else:
                # Try to treat the whole response as lock data
                return data
        
        return None
    except Exception:
        return None


def check_local_cache(reference: str) -> Optional[dict]:
    """Check local cache for lock file."""
    # Extract digest if present
    if "@sha256:" in reference:
        digest = reference.split("@sha256:")[-1][:12]
    elif "/" in reference:
        # Try to find by name pattern
        cache_dir = Path.home() / ".hud" / "envs"
        if cache_dir.exists():
            # Look for any cached version of this image
            for env_dir in cache_dir.iterdir():
                if env_dir.is_dir():
                    lock_file = env_dir / "hud.lock.yaml"
                    if lock_file.exists():
                        with open(lock_file) as f:
                            lock_data = yaml.safe_load(f)
                        # Check if this matches our reference
                        if lock_data and "image" in lock_data:
                            image = lock_data["image"]
                            # Match by name (ignoring tag/digest)
                            ref_base = reference.split("@")[0].split(":")[0]
                            img_base = image.split("@")[0].split(":")[0]
                            if ref_base in img_base or img_base in ref_base:
                                return lock_data
        return None
    else:
        digest = "latest"
    
    # Check specific digest directory
    lock_file = Path.home() / ".hud" / "envs" / digest / "hud.lock.yaml"
    if lock_file.exists():
        with open(lock_file) as f:
            return yaml.safe_load(f)
    
    return None


async def analyze_from_metadata(reference: str, output_format: str, verbose: bool) -> None:
    """Analyze environment from cached or registry metadata."""
    import json
    from .analyze import display_interactive, display_markdown
    
    design.header("MCP Environment Analysis", icon="🔍")
    design.info(f"Looking up: {reference}")
    design.info("")
    
    lock_data = None
    source = None
    
    # 1. Check local cache first
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Checking local cache...", total=None)
        
        lock_data = check_local_cache(reference)
        if lock_data:
            progress.update(task, description="[green]✓ Found in local cache[/green]")
            source = "local"
        else:
            progress.update(task, description="[yellow]→ Not in cache, checking registry...[/yellow]")
            
            # 2. Try HUD registry
            # Parse reference to get org/name format
            if "/" in reference and "@" not in reference and ":" not in reference:
                # Already in org/name format
                registry_ref = reference
            elif "/" in reference:
                # Extract org/name from full reference
                parts = reference.split("/")
                if len(parts) >= 2:
                    # Handle docker.io/org/name or just org/name
                    if parts[0] in ["docker.io", "registry-1.docker.io", "index.docker.io"]:
                        # Remove registry prefix but keep tag
                        registry_ref = "/".join(parts[1:]).split("@")[0]
                    else:
                        # Keep org/name:tag format
                        registry_ref = "/".join(parts[:2]).split("@")[0]
                else:
                    registry_ref = reference
            else:
                registry_ref = reference
            
            if not settings.api_key:
                progress.update(task, description="[yellow]→ No API key (checking public registry)...[/yellow]")
            
            lock_data = fetch_lock_from_registry(registry_ref)
            if lock_data:
                progress.update(task, description="[green]✓ Found in HUD registry[/green]")
                source = "registry"
                
                # Save to local cache for next time
                if "@sha256:" in lock_data.get("image", ""):
                    digest = lock_data["image"].split("@sha256:")[-1][:12]
                else:
                    digest = "latest"
                
                cache_dir = Path.home() / ".hud" / "envs" / digest
                cache_dir.mkdir(parents=True, exist_ok=True)
                with open(cache_dir / "hud.lock.yaml", "w") as f:
                    yaml.dump(lock_data, f, default_flow_style=False, sort_keys=False)
            else:
                progress.update(task, description="[red]✗ Not found[/red]")
    
    if not lock_data:
        design.error("Environment metadata not found")
        console.print("\n[yellow]This environment hasn't been analyzed yet.[/yellow]")
        console.print("\nOptions:")
        console.print(f"  1. Pull it first: [cyan]hud pull {reference}[/cyan]")
        console.print(f"  2. Run live analysis: [cyan]hud analyze {reference} --live[/cyan]")
        if not settings.api_key:
            console.print(f"  3. Set HUD_API_KEY for private environments")
        return
    
    # Convert lock data to analysis format
    analysis = {
        "status": "metadata" if source == "local" else "registry",
        "source": source,
        "tools": [],
        "resources": [],
        "prompts": []
    }
    
    # Add basic info
    if "image" in lock_data:
        analysis["image"] = lock_data["image"]
    
    if "build" in lock_data:
        analysis["build_info"] = lock_data["build"]
    
    if "push" in lock_data:
        analysis["push_info"] = lock_data["push"]
    
    # Extract environment info
    if "environment" in lock_data:
        env = lock_data["environment"]
        if "initializeMs" in env:
            analysis["init_time"] = env["initializeMs"]
        if "toolCount" in env:
            analysis["tool_count"] = env["toolCount"]
        if "variables" in env:
            analysis["env_vars"] = env["variables"]
    
    # Extract tools
    if "tools" in lock_data:
        for tool in lock_data["tools"]:
            analysis["tools"].append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "inputSchema": tool.get("inputSchema", {}) if verbose else None
            })
    
    # Display results
    design.info("")
    if source == "local":
        design.dim_info("Source:", "Local cache")
    else:
        design.dim_info("Source:", "HUD registry")
    
    if "image" in analysis:
        design.dim_info("Image:", analysis["image"])
    
    design.info("")
    
    # Display results based on format
    if output_format == "json":
        console.print_json(json.dumps(analysis, indent=2))
    elif output_format == "markdown":
        display_markdown(analysis)
    else:  # interactive
        display_interactive(analysis)
