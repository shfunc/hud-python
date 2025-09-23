from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
import yaml

from hud.cli.build import build_environment
from hud.cli.push import push_environment
from hud.cli.utils.docker import require_docker_running
from hud.cli.utils.environment import is_environment_directory
from hud.cli.utils.registry import extract_name_and_tag
from hud.utils.hud_console import hud_console
from hud.utils.tasks import load_tasks

if TYPE_CHECKING:
    from hud.types import Task


def _is_remote_url(url: str) -> bool:
    """Match the remote url."""
    # See if a url is a remote url
    return bool(re.match(r"^(https?:\/\/)?(www\.)?[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(\/\S*)?$", url))


def _validate_tasks(tasks: list[Task]) -> bool:
    """Validate the tasks file: return True if tasks already reference a remote MCP URL.

    A task is considered remote if any "url" field anywhere inside mcp_config
    is a valid remote URL (e.g., https://mcp.hud.so/v3/mcp).
    """

    def _has_remote_url(obj: Any) -> bool:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "url" and isinstance(v, str) and _is_remote_url(v):
                    return True
                if _has_remote_url(v):
                    return True
        elif isinstance(obj, list):
            for item in obj:
                if _has_remote_url(item):
                    return True
        return False

    for task in tasks:
        cfg = task.mcp_config or {}
        if not _has_remote_url(cfg):
            return False
    return True


def _find_environment_dir(tasks_path: Path) -> Path | None:
    """Find the environment directory related to a tasks file.

    Strategy:
    - Prefer a directory containing hud.lock.yaml
    - Fallback to a directory that looks like an environment (Dockerfile + pyproject.toml)
    - Search the tasks file directory, CWD, and a couple of parents
    """
    candidates: list[Path] = []
    cwd = Path.cwd()
    candidates.extend([tasks_path.parent, cwd])

    # Add parents (up to 2 levels for each)
    for base in list(candidates):
        p = base
        for _ in range(2):
            p = p.parent
            if p not in candidates:
                candidates.append(p)

    # Prefer those with hud.lock.yaml
    for d in candidates:
        if (d / "hud.lock.yaml").exists():
            return d

    # Otherwise, find a plausible environment dir
    for d in candidates:
        try:
            if is_environment_directory(d):
                return d
        except Exception as e:
            hud_console.debug(f"Skipping path {d}: {e}")
            continue

    return None


def _ensure_built(env_dir: Path) -> dict[str, Any]:
    """Ensure the environment is built and a lock file exists; return lock data."""
    lock_path = env_dir / "hud.lock.yaml"
    if not lock_path.exists():
        hud_console.warning("No hud.lock.yaml found. The environment hasn't been built.")
        if not hud_console.confirm("Build the environment now (runs 'hud build')?", default=True):
            raise typer.Exit(1)
        # Check Docker availability before attempting a build
        require_docker_running()
        # Run build (non-interactive). If Docker isn't running, this will raise and stop the flow.
        # Force linux/amd64 platform to ensure compatibility during RL flows.
        build_environment(str(env_dir), platform="linux/amd64")

    # Load lock file
    with open(lock_path) as f:
        lock_data = yaml.safe_load(f) or {}
    return lock_data


def _ensure_pushed(env_dir: Path, lock_data: dict[str, Any]) -> dict[str, Any]:
    """Ensure the environment is pushed to a registry; return updated lock data."""
    pushed = bool(lock_data.get("push"))
    if not pushed:
        hud_console.warning("Environment not pushed to a registry yet.")
        if not hud_console.confirm("Push to a registry now (runs 'hud push')?", default=True):
            raise typer.Exit(1)
        # Check Docker availability before attempting a push
        require_docker_running()

        # If Docker or login is not configured, the push function will fail and halt.
        push_environment(str(env_dir), yes=True)

        # Reload lock after push
        lock_path = env_dir / "hud.lock.yaml"
        with open(lock_path) as f:
            lock_data = yaml.safe_load(f) or {}

    return lock_data


def _derive_remote_image(lock_data: dict[str, Any]) -> str:
    """Derive org/name:tag from lock file for MCP header.

    Preference order:
    1) lock_data["push"]["image_with_tag"] if present
    2) Derive from lock_data["image"] (may be a digest; falls back to latest)
    """
    push_info = lock_data.get("push", {}) if isinstance(lock_data, dict) else {}

    # 1) Exact image_with_tag if present
    pushed_with_tag = str(push_info.get("image_with_tag", "")).strip()
    if pushed_with_tag:
        name, tag = extract_name_and_tag(pushed_with_tag)
        return f"{name}:{tag}"

    # Base name always comes from lock_data.image to preserve org/repo
    image_ref = str(lock_data.get("image", "")).strip()
    if not image_ref:
        raise typer.Exit(1)
    name, tag = extract_name_and_tag(image_ref)
    return f"{name}:{tag}"


def _extract_existing_images(tasks: list[Task]) -> set[str]:
    """Extract all Mcp-Image references from tasks."""
    images = set()
    
    def _extract_from_obj(obj: Any) -> None:
        if isinstance(obj, dict):
            # Check for Mcp-Image in headers
            if "headers" in obj and isinstance(obj["headers"], dict):
                mcp_image = obj["headers"].get("Mcp-Image")
                if mcp_image:
                    images.add(mcp_image)
            # Recursively check nested objects
            for v in obj.values():
                _extract_from_obj(v)
        elif isinstance(obj, list):
            for item in obj:
                _extract_from_obj(item)
    
    for task in tasks:
        if task.mcp_config:
            _extract_from_obj(task.mcp_config)
    
    return images


def convert_tasks_to_remote(tasks_file: str) -> str:
    """Convert a local tasks file to remote MCP tasks and return new filename.

    Steps:
    1) Find env dir; ensure built (hud.lock.yaml), otherwise build
    2) Ensure pushed to registry, otherwise push
    3) Check for outdated images in existing task configurations
    4) Create remote_[tasks].json with mcp_config pointing to mcp.hud.so and Mcp-Image
    5) Return the new tasks file path
    """
    tasks_path = Path(tasks_file).resolve()

    tasks = load_tasks(str(tasks_path))

    # Ensure HUD_API_KEY is available: prefer process env, else load from env_dir/.env
    from hud.settings import settings

    if not settings.api_key or not settings.api_key.strip():
        hud_console.error("HUD_API_KEY is not set")
        hud_console.info("Set it in your environment or run: hud set HUD_API_KEY=your-key-here")
        raise typer.Exit(1)

    # Check if tasks already have remote URLs
    already_remote = _validate_tasks(tasks)
    
    # Extract existing images from tasks
    existing_images = _extract_existing_images(tasks)
    
    # Load tasks (supports .json and .jsonl)
    if already_remote and not existing_images:
        # Tasks are remote but have no image references - just return as-is
        return str(tasks_path)

    # Locate environment
    env_dir = _find_environment_dir(tasks_path)
    if not env_dir:
        hud_console.error("Could not locate an environment directory (Dockerfile + pyproject.toml)")
        hud_console.hint("Ensure you're in or near your environment folder before running 'hud rl'")
        raise typer.Exit(1)

    # Ensure built and pushed
    lock_data = _ensure_built(env_dir)
    lock_data = _ensure_pushed(env_dir, lock_data)

    # Derive remote image name org/name:tag
    remote_image = _derive_remote_image(lock_data)
    
    # Check if existing images are outdated
    needs_update = False
    should_update_image = False
    if existing_images:
        # Check if any existing image differs from the latest
        for existing_img in existing_images:
            if existing_img != remote_image:
                hud_console.warning(f"Detected outdated image reference: {existing_img}")
                hud_console.info(f"Latest pushed image: {remote_image}")
                needs_update = True
                break
        
        if needs_update:
            if hud_console.confirm("Update task configuration with the latest image?", default=True):
                hud_console.info("Updating task configuration with latest image...")
                should_update_image = True
            else:
                # If user doesn't want to update, just return the original file
                if already_remote:
                    return str(tasks_path)
                # Otherwise, continue with conversion but keep old images
                remote_image = list(existing_images)[0]  # Use the first existing image
                
    # If tasks are already remote and we just need to update the image
    if already_remote and should_update_image:
        # Update image references in-place
        def _update_image_refs(obj: Any) -> Any:
            if isinstance(obj, dict):
                new_obj = {}
                for k, v in obj.items():
                    if k == "Mcp-Image" and isinstance(v, str) and v in existing_images:
                        new_obj[k] = remote_image
                    else:
                        new_obj[k] = _update_image_refs(v)
                return new_obj
            elif isinstance(obj, list):
                return [_update_image_refs(item) for item in obj]
            else:
                return obj
        
        # Update tasks with new image
        updated_tasks = []
        for task in tasks:
            task_dict = task.model_dump() if hasattr(task, "model_dump") else dict(task)
            if "mcp_config" in task_dict:
                task_dict["mcp_config"] = _update_image_refs(task_dict["mcp_config"])
            updated_tasks.append(task_dict)
        
        # Write updated file (preserve original format - check if it's .jsonl)
        if tasks_path.suffix == ".jsonl":
            with open(tasks_path, "w", encoding="utf-8") as f:
                for task in updated_tasks:
                    json.dump(task, f, ensure_ascii=False)
                    f.write("\n")
        else:
            with open(tasks_path, "w", encoding="utf-8") as f:
                json.dump(updated_tasks, f, ensure_ascii=False, indent=2)
                f.write("\n")
        
        hud_console.success(f"Updated {tasks_path.name} with latest image: {remote_image}")
        return str(tasks_path)

    # Helper to strip extra fields from tool calls
    def _simplify_tool_call(tool: Any) -> Any:
        def _one(x: Any) -> dict[str, Any]:
            try:
                data = x.model_dump() if hasattr(x, "model_dump") else dict(x)
            except Exception:
                try:
                    data = dict(x)
                except Exception:
                    return {}
            # Keep only name and arguments
            name = data.get("name")
            arguments = data.get("arguments", {})
            return {"name": name, "arguments": arguments}

        if tool is None:
            return None
        if isinstance(tool, list):
            return [_one(x) for x in tool]
        return _one(tool)

    # Convert to list[dict]
    tasks_payload: list[dict[str, Any]] = []
    for t in tasks:
        item: dict[str, Any] = {
            "prompt": t.prompt,
            "mcp_config": {
                "hud": {
                    "url": "https://mcp.hud.so/v3/mcp",
                    "headers": {
                        "Authorization": "Bearer ${HUD_API_KEY}",
                        "Mcp-Image": remote_image,
                    },
                }
            },
        }

        # Optional fields, omit Nones
        if t.setup_tool is not None:
            item["setup_tool"] = _simplify_tool_call(t.setup_tool)
        if t.evaluate_tool is not None:
            item["evaluate_tool"] = _simplify_tool_call(t.evaluate_tool)
        if t.agent_tools is not None:
            item["agent_tools"] = t.agent_tools
        if t.system_prompt is not None:
            item["system_prompt"] = t.system_prompt
        if t.metadata:
            item["metadata"] = t.metadata

        tasks_payload.append(item)

    # Write new file: remote_<name>.json (always JSON array)
    remote_name = f"remote_{tasks_path.stem}.json"
    remote_path = tasks_path.parent / remote_name
    with open(remote_path, "w", encoding="utf-8") as f:
        json.dump(tasks_payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    hud_console.success(f"Created remote tasks file: {remote_path.name}")
    hud_console.hint("Proceeding with RL training on the remote environment")

    return str(remote_path)
