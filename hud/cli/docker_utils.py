"""Docker utilities for HUD CLI."""

import json
import subprocess


def get_docker_cmd(image: str) -> list[str] | None:
    """
    Extract the CMD from a Docker image.
    
    Args:
        image: Docker image name
        
    Returns:
        List of command parts or None if not found
    """
    try:
        result = subprocess.run(
            ["docker", "inspect", image],
            capture_output=True,
            text=True,
            check=True
        )
        
        inspect_data = json.loads(result.stdout)
        if inspect_data and len(inspect_data) > 0:
            config = inspect_data[0].get("Config", {})
            cmd = config.get("Cmd", [])
            return cmd if cmd else None
            
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        return None


def inject_supervisor(cmd: list[str]) -> list[str]:
    """
    Inject HUD dev supervisor into a Docker CMD.
    
    For shell commands, we inject before the last exec command.
    For direct commands, we wrap the entire command.
    
    Args:
        cmd: Original Docker CMD
        
    Returns:
        Modified CMD with supervisor injected
    """
    if not cmd:
        return cmd
    
    # Handle shell commands
    if cmd[0] in ["sh", "bash"] and len(cmd) >= 3 and cmd[1] == "-c":
        shell_cmd = cmd[2]
        
        # Look for 'exec' in the shell command
        if " exec " in shell_cmd:
            # Replace the last 'exec' with 'exec hud dev --'
            parts = shell_cmd.rsplit(" exec ", 1)
            if len(parts) == 2:
                new_shell_cmd = f"{parts[0]} exec hud dev -- {parts[1]}"
                return [cmd[0], cmd[1], new_shell_cmd]
        else:
            # No exec, wrap the whole command
            return [cmd[0], cmd[1], f"hud dev -- {shell_cmd}"]
    
    # Direct command - wrap it
    return ["hud", "dev", "--"] + cmd


def image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", image_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return result.returncode == 0
