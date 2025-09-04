"""HuggingFace dataset conversion command for HUD tasks."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer

from hud.cli.rl.utils import get_mcp_config_from_lock, read_lock_file, write_lock_file
from hud.utils.design import HUDDesign

design = HUDDesign()


def hf_command(
    tasks_file: Path | None = None,
    name: str | None = None,
    push: bool = True,
    private: bool = False,
    update_lock: bool = True,
    token: str | None = None,
) -> None:
    """📊 Convert tasks to HuggingFace dataset format.

    Automatically detects task files if not specified.
    Suggests dataset name based on environment if not provided.
    Converts a JSON file containing HUD tasks into a HuggingFace dataset
    and optionally pushes it to the Hub. Also updates hud.lock.yaml with
    the primary dataset reference.

    Examples:
        hud hf                      # Auto-detect tasks and suggest name
        hud hf tasks.json           # Use specific file, suggest name
        hud hf --name my-org/my-tasks  # Auto-detect tasks, use name
        hud hf tasks.json --name hud-evals/web-tasks --private
        hud hf tasks.json --name local-dataset --no-push
    """
    design.header("HuggingFace Dataset Converter", icon="📊")

    # Auto-detect task file if not provided
    if tasks_file is None:
        design.info("Looking for task files...")

        # Common task file patterns
        patterns = [
            "tasks.json",
            "task.json",
            "*_tasks.json",
            "eval*.json",
            "evaluation*.json",
        ]

        json_files = []
        for pattern in patterns:
            json_files.extend(Path(".").glob(pattern))

        # Remove duplicates and sort
        json_files = sorted(set(json_files))

        if not json_files:
            design.error("No task files found in current directory")
            design.info("Create a task JSON file (e.g., tasks.json) or specify the file path")
            raise typer.Exit(1)
        elif len(json_files) == 1:
            tasks_file = json_files[0]
            design.info(f"Found task file: {tasks_file}")
        else:
            # Multiple files found, let user choose
            design.info("Multiple task files found:")
            file_choice = design.select(
                "Select a task file to convert:",
                choices=[str(f) for f in json_files],
            )
            tasks_file = Path(file_choice)
            design.success(f"Selected: {tasks_file}")

    # Validate inputs
    if tasks_file and not tasks_file.exists():
        design.error(f"Tasks file not found: {tasks_file}")
        raise typer.Exit(1)

    # Suggest dataset name if not provided
    if name is None:
        design.info("Generating dataset name suggestion...")

        # Try to get HF username from environment or git config
        hf_username = None
        try:
            # Try HF token first
            from huggingface_hub import HfApi

            api = HfApi(token=token)
            user_info = api.whoami()
            hf_username = user_info.get("name", None)
        except Exception:
            # Try git config as fallback
            try:
                import subprocess

                result = subprocess.run(
                    ["git", "config", "user.name"],  # noqa: S607
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and result.stdout.strip():
                    hf_username = result.stdout.strip().lower().replace(" ", "-")
            except Exception:
                design.warning("Failed to get HF username from git config")

        # Get environment name from current directory or lock file
        env_name = Path.cwd().name

        # Try to get a better name from lock file
        lock_path = Path("hud.lock.yaml")
        if lock_path.exists():
            try:
                with open(lock_path) as f:
                    import yaml

                    lock_data = yaml.safe_load(f)
                    if "image" in lock_data:
                        # Extract name from image like "test:dev@sha256:..."
                        image_name = lock_data["image"].split(":")[0].split("/")[-1]
                        if image_name and image_name != "local":
                            env_name = image_name
            except Exception as e:
                design.warning(f"Failed to get HF username from lock file: {e}")

        # Generate suggestions
        suggestions = []
        if hf_username:
            suggestions.append(f"{hf_username}/{env_name}-tasks")
            suggestions.append(f"{hf_username}/{env_name}-dataset")
        suggestions.append(f"my-org/{env_name}-tasks")
        suggestions.append(f"hud-evals/{env_name}-tasks")

        # Let user choose or enter custom
        design.info("Dataset name suggestions:")
        suggestions.append("Enter custom name...")

        choice = design.select("Select or enter a dataset name:", choices=suggestions)

        if choice == "Enter custom name...":
            name = typer.prompt("Enter dataset name (e.g., 'my-org/my-dataset')")
        else:
            name = choice

        design.success(f"Using dataset name: {name}")

    # Validate dataset name format
    if push and name and "/" not in name:
        design.error("Dataset name must include organization (e.g., 'my-org/my-dataset')")
        design.info("For local-only datasets, use --no-push")
        raise typer.Exit(1)

    # Load tasks
    design.info(f"Loading tasks from: {tasks_file}")
    try:
        if tasks_file is None:
            raise ValueError("Tasks file is required")
        with open(tasks_file) as f:
            tasks_data = json.load(f)
    except json.JSONDecodeError as e:
        design.error(f"Invalid JSON file: {e}")
        raise typer.Exit(1) from e

    # Handle both single task and list of tasks
    if isinstance(tasks_data, dict):
        tasks = [tasks_data]
        design.info("Found 1 task")
    elif isinstance(tasks_data, list):
        tasks = tasks_data
        design.info(f"Found {len(tasks)} tasks")
    else:
        design.error("Tasks file must contain a JSON object or array")
        raise typer.Exit(1)

    # Validate task format
    valid_tasks = []
    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            design.warning(f"Skipping task {i}: not a JSON object")
            continue

        # Required fields
        if "prompt" not in task:
            design.warning(f"Skipping task {i}: missing 'prompt' field")
            continue

        if "evaluate_tool" not in task:
            design.warning(f"Skipping task {i}: missing 'evaluate_tool' field")
            continue

        # Add default values
        if "id" not in task:
            task["id"] = f"task-{i:04d}"

        if "mcp_config" not in task:
            # Try to infer from hud.lock.yaml
            mcp_config = get_mcp_config_from_lock()
            if mcp_config:
                task["mcp_config"] = mcp_config
            else:
                design.warning(f"Task {task['id']}: missing 'mcp_config' field")
                continue

        valid_tasks.append(task)

    if not valid_tasks:
        design.error("No valid tasks found")
        raise typer.Exit(1)

    design.success(f"Validated {len(valid_tasks)} tasks")

    # Check if dataset is suitable for training
    if len(valid_tasks) < 4:
        design.warning(
            f"Dataset has only {len(valid_tasks)} task(s). RL training typically requires at least 4 tasks."  # noqa: E501
        )
        use_for_training = design.select(
            "Will this dataset be used for RL training?",
            ["Yes, duplicate tasks to reach 4", "No, keep as is"],
        )

        if use_for_training == "Yes, duplicate tasks to reach 4":
            # Duplicate tasks to reach minimum of 4
            original_count = len(valid_tasks)
            while len(valid_tasks) < 4:
                for task in valid_tasks[:original_count]:
                    if len(valid_tasks) >= 4:
                        break
                    # Create a copy with modified ID
                    duplicated_task = task.copy()
                    duplicated_task["id"] = (
                        f"{task['id']}_dup{len(valid_tasks) - original_count + 1}"
                    )
                    valid_tasks.append(duplicated_task)

            design.info(f"Duplicated tasks: {original_count} → {len(valid_tasks)}")

    # Check if MCP configs should be converted to remote
    sample_mcp_config = valid_tasks[0].get("mcp_config", {})
    if isinstance(sample_mcp_config, str):
        sample_mcp_config = json.loads(sample_mcp_config)

    # Check config type by looking at all MCP server URLs
    config_type = "unknown"
    remote_image = None

    # Check all server configs (could be named anything, not just "hud")
    for server_config in sample_mcp_config.values():
        if isinstance(server_config, dict) and "url" in server_config:
            url = server_config.get("url", "")
            if "mcp.hud.so" in url:
                config_type = "remote"
                # Extract image from Mcp-Image header if present
                headers = server_config.get("headers", {})
                found_image = headers.get("Mcp-Image", "")
                if found_image:
                    remote_image = found_image
                    break
            else:
                # Any non-mcp.hud.so URL means local config
                config_type = "local"

    if config_type == "remote" and remote_image:
        design.info(f"Tasks already use remote MCP configs with image: {remote_image}")

    if config_type == "local":
        convert_to_remote = design.select(
            "Tasks use local MCP configs. Convert to remote configs for training?",
            ["Yes, convert to remote (requires public image)", "No, keep local configs"],
        )

        if convert_to_remote == "Yes, convert to remote (requires public image)":
            # Get the image name from lock file
            from hud.cli.rl.utils import get_image_from_lock

            image = get_image_from_lock()

            if not image:
                design.error("No image found in hud.lock.yaml")
                design.hint("Run 'hud build' first")
                raise typer.Exit(1)

            # Check if image contains registry prefix (indicates it's pushed)
            if "/" not in image or image.startswith("local/"):
                # Clean up image name for display (remove SHA if present)
                display_image = image.split("@")[0] if "@" in image else image
                design.warning(f"Image '{display_image}' appears to be local only")
                push_image = design.select(
                    "Would you like to push the image to make it publicly available?",
                    ["Yes, push image", "No, cancel"],
                )

                if push_image == "Yes, push image":
                    design.info("Running 'hud push' to publish image...")
                    # Import here to avoid circular imports
                    from hud.cli.push import push_command

                    # Run push command (it's synchronous)
                    push_command(directory=".", yes=True)
                    design.success("Image pushed successfully")

                    # Re-read the image name as it may have changed
                    image = get_image_from_lock()
                else:
                    design.info("Keeping local MCP configs")
                    convert_to_remote = None

            if convert_to_remote and image:
                # Convert all task configs to remote
                design.info(f"Converting MCP configs to use remote image: {image}")

                for task in valid_tasks:
                    # Create remote MCP config
                    remote_config = {
                        "hud": {
                            "url": "https://mcp.hud.so/v3/mcp",
                            "headers": {
                                "Authorization": "Bearer $HUD_API_KEY",
                                "Mcp-Image": image,
                            },
                        }
                    }
                    task["mcp_config"] = remote_config

                design.success("✓ Converted all tasks to use remote MCP configs")

    # Convert to HuggingFace format
    dataset_dict = {
        "id": [],
        "prompt": [],
        "mcp_config": [],
        "setup_tool": [],
        "evaluate_tool": [],
        "metadata": [],
    }

    for task in valid_tasks:
        dataset_dict["id"].append(task["id"])
        dataset_dict["prompt"].append(task["prompt"])
        dataset_dict["mcp_config"].append(json.dumps(task["mcp_config"]))
        dataset_dict["setup_tool"].append(json.dumps(task.get("setup_tool", {})))
        dataset_dict["evaluate_tool"].append(json.dumps(task["evaluate_tool"]))
        dataset_dict["metadata"].append(json.dumps(task.get("metadata", {})))

    # Push to HuggingFace Hub if requested
    if push:
        try:
            from datasets import Dataset
        except ImportError as e:
            design.error("datasets library not installed")
            design.info("Install with: pip install datasets")
            raise typer.Exit(1) from e

        design.info(f"Creating HuggingFace dataset: {name}")
        dataset = Dataset.from_dict(dataset_dict)

        # Set up HF token
        if token:
            import os

            os.environ["HF_TOKEN"] = token

        design.info(f"Pushing to Hub (private={private})...")
        try:
            if name is None:
                raise ValueError("Dataset name is required")
            dataset.push_to_hub(name, private=private)
            design.success(f"Dataset published: https://huggingface.co/datasets/{name}")
        except Exception as e:
            design.error(f"Failed to push to Hub: {e}")
            design.hint("Make sure you're logged in: huggingface-cli login")
            raise typer.Exit(1) from e
    else:
        # Save locally
        if name is None:
            raise ValueError("Dataset name is required")
        output_file = Path(f"{name.replace('/', '_')}_dataset.json")
        with open(output_file, "w") as f:
            json.dump(dataset_dict, f, indent=2)
        design.success(f"Dataset saved locally: {output_file}")

    # Update hud.lock.yaml if requested
    if update_lock:
        update_lock_file(name, len(valid_tasks))


def update_lock_file(dataset_name: str, task_count: int) -> None:
    """Update hud.lock.yaml with primary dataset reference."""
    # Load existing lock file or create new
    lock_data = read_lock_file()

    # Update dataset info
    lock_data["primary_dataset"] = {
        "name": dataset_name,
        "task_count": task_count,
        "updated_at": datetime.now().isoformat(),
    }

    # Write back
    if write_lock_file(lock_data):
        design.success(f"Updated hud.lock.yaml with dataset: {dataset_name}")
