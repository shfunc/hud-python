"""Docker utilities for HUD CLI."""

from __future__ import annotations

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
        result = subprocess.run(  # noqa: S603
            ["docker", "inspect", image],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )

        inspect_data = json.loads(result.stdout)
        if inspect_data and len(inspect_data) > 0 and isinstance(inspect_data[0], dict):
            config = inspect_data[0].get("Config", {})
            cmd = config.get("Cmd", [])
            return cmd if cmd else None

    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        return None


def inject_supervisor(cmd: list[str]) -> list[str]:
    """
    Inject watchfiles CLI supervisor into a Docker CMD.

    For shell commands, we inject before the last exec command.
    For direct commands, we wrap the entire command.

    Args:
        cmd: Original Docker CMD

    Returns:
        Modified CMD with watchfiles supervisor injected
    """
    if not cmd:
        return cmd

    # Handle shell commands that might have background processes
    if cmd[0] in ["sh", "bash"] and len(cmd) >= 3 and cmd[1] == "-c":
        shell_cmd = cmd[2]

        # Look for 'exec' in the shell command - this is the last command
        if " exec " in shell_cmd:
            # Replace only the exec'd command with watchfiles
            parts = shell_cmd.rsplit(" exec ", 1)
            if len(parts) == 2:
                # Extract the actual command after exec
                last_cmd = parts[1].strip()
                # Use watchfiles with logs redirected to stderr (which won't interfere with MCP on stdout)  # noqa: E501
                new_shell_cmd = f"{parts[0]} exec watchfiles --verbose '{last_cmd}' /app/src"
                return [cmd[0], cmd[1], new_shell_cmd]
        else:
            # No exec, the whole thing is the command
            return ["sh", "-c", f"watchfiles --verbose '{shell_cmd}' /app/src"]

    # Direct command - wrap with watchfiles
    watchfiles_cmd = " ".join(cmd)
    return ["sh", "-c", f"watchfiles --verbose '{watchfiles_cmd}' /app/src"]


def image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally."""
    result = subprocess.run(  # noqa: S603
        ["docker", "image", "inspect", image_name],  # noqa: S607
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def remove_container(container_name: str) -> bool:
    """Remove a Docker container by name.

    Args:
        container_name: Name of the container to remove

    Returns:
        True if successful or container doesn't exist, False on error
    """
    try:
        subprocess.run(  # noqa: S603
            ["docker", "rm", "-f", container_name],  # noqa: S607
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,  # Don't raise error if container doesn't exist
        )
        return True
    except Exception:
        return False


def generate_container_name(identifier: str, prefix: str = "hud") -> str:
    """Generate a safe container name from an identifier.

    Args:
        identifier: Image name or other identifier
        prefix: Prefix for the container name

    Returns:
        Safe container name with special characters replaced
    """
    # Replace special characters with hyphens
    safe_name = identifier.replace(":", "-").replace("/", "-").replace("\\", "-")
    return f"{prefix}-{safe_name}"
