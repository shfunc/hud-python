"""Main RL training command implementation."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import typer

from hud.settings import settings
from hud.utils.design import HUDDesign

from .pod import run_prime_training
from .utils import (
    detect_image_name,
    get_primary_dataset,
    validate_dataset_name,
)

design = HUDDesign()


def train_command_wrapper(
    model: str,
    dataset: str | None,
    config: Path | None,
    gpus: str,
    provider: str,
    output_dir: Path,
) -> None:
    """Wrapper to handle interactive prompts before entering async context."""
    # Pre-flight checks for required environment variables
    design.section_title("🔍 Pre-flight Checks")

    missing_vars = []

    # Check HUD API key
    if not settings.api_key:
        missing_vars.append("HUD_API_KEY")
    else:
        design.success("✓ HUD_API_KEY configured")

    # Check WANDB API key (optional but recommended)
    if not getattr(settings, "wandb_api_key", None):
        design.warning("⚠ WANDB_API_KEY not set (optional but recommended for training metrics)")
    else:
        design.success("✓ WANDB_API_KEY configured")

    # Check PRIME API key (required for remote training)
    if provider == "prime" and not getattr(settings, "prime_api_key", None):
        missing_vars.append("PRIME_API_KEY")
    elif provider == "prime":
        design.success("✓ PRIME_API_KEY configured")

    if missing_vars:
        design.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        design.info("")
        design.info("Set them using one of these methods:")
        design.info("1. Environment variables:")
        for var in missing_vars:
            design.command_example(f"export {var}=your-{var.lower().replace('_', '-')}")
        design.info("")
        design.info("2. Create a .env file in your project root:")
        design.command_example(
            "\n".join([f"{var}=your-{var.lower().replace('_', '-')}" for var in missing_vars]),
            "env",
        )
        raise typer.Exit(1)

    # Check for required components
    missing = check_requirements(config, dataset)

    # Auto-detect config if not specified and exactly one exists
    if not config and "config" not in missing:
        config_dir = Path("configs")
        if config_dir.exists():
            yaml_files = list(config_dir.glob("*.yaml"))
            if len(yaml_files) == 1:
                config = yaml_files[0]
                design.info(f"Using config: {config}")

    # Store user choice for pod creation
    auto_create_pod = None
    team_id = None

    if missing:
        # Handle interactive prompts here
        if "config" in missing:
            if missing["config"] == "multiple":
                # Select from multiple configs
                config_dir = Path("configs")
                yaml_files = list(config_dir.glob("*.yaml"))
                config_names = [f.name for f in yaml_files]
                selected_config = design.select(
                    "Multiple config files found. Select one:", config_names
                )
                config = config_dir / selected_config
            else:
                # No config found, offer to generate
                generate_config = design.select(
                    "No config file found. Would you like to generate one?",
                    ["Yes, generate config", "No, I'll create it manually"],
                )

                if generate_config == "Yes, generate config":
                    design.info("Running 'hud rl init' to generate config...")
                    design.info("")
                    # Import here to avoid circular imports
                    from .init import init_command_wrapper

                    init_command_wrapper(".", None, False, False)

                    # Look for generated config
                    config_dir = Path("configs")
                    if config_dir.exists():
                        yaml_files = list(config_dir.glob("*.yaml"))
                        if yaml_files:
                            config = yaml_files[0]
                            design.success(f"Using generated config: {config}")
                        else:
                            design.error("Config generation failed")
                            raise typer.Exit(1)
                else:
                    design.info("Please create a config file and try again")
                    raise typer.Exit(1)

        if "dataset" in missing:
            # Check if we have tasks.json
            tasks_file = Path("tasks.json")
            if tasks_file.exists():
                create_dataset = design.select(
                    "Found tasks.json. Would you like to upload it as a dataset?",
                    ["Yes, upload to HuggingFace", "No, I'll handle it manually"],
                )

                if create_dataset == "Yes, upload to HuggingFace":
                    dataset_name = typer.prompt("Enter dataset name (e.g., username/dataset-name)")

                    if not validate_dataset_name(dataset_name):
                        design.error("Invalid dataset name format. Expected: username/dataset-name")
                        raise typer.Exit(1)

                    design.info(f"Running 'hud hf tasks.json --name {dataset_name}'...")
                    design.info("")

                    # Run hf command
                    result = subprocess.run(  # noqa: S603
                        ["hud", "hf", "tasks.json", "--name", dataset_name],  # noqa: S607
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode == 0:
                        design.success("Dataset uploaded successfully")
                        dataset = dataset_name
                    else:
                        design.error("Failed to upload dataset")
                        if result.stderr:
                            design.error(result.stderr)
                        raise typer.Exit(1)
                else:
                    design.info("Please specify a dataset with --dataset")
                    raise typer.Exit(1)
            else:
                design.error("No dataset specified and no tasks.json found")
                design.info("Use --dataset to specify a HuggingFace dataset")
                raise typer.Exit(1)

    # Ask about pod creation for Prime training
    if provider == "prime":
        # Check if team ID is globally configured
        team_check = subprocess.run(
            ["prime", "config", "view"],  # noqa: S607
            capture_output=True,
            text=True,
        )

        has_global_team = False
        if team_check.returncode == 0:
            # Parse the table output - look for Team ID row
            for line in team_check.stdout.split("\n"):
                if "team id" in line.lower():
                    # Check if there's a value after the | separator
                    parts = line.split("|")
                    if len(parts) >= 2:
                        # Get the value part and check if it's not empty
                        value = parts[1].strip()
                        if value and value != "None":
                            has_global_team = True
                            design.info("Using globally configured team ID")
                            break

        if not has_global_team:
            # Only ask if no global team is configured
            auto_create_pod = design.select(
                "How would you like to create the Prime Intellect pod?",
                ["Personal account (automated)", "Team account (enter team ID)"],
            )

            # If team account selected, get the team ID
            if auto_create_pod == "Team account (enter team ID)":
                team_id = typer.prompt("Enter your team ID (e.g., team_abc123def456)")

                # Save it globally automatically
                subprocess.run(["prime", "config", "set-team-id", team_id])  # noqa: S603, S607
                design.success("Team ID saved globally")

                auto_create_pod = (
                    "Personal account (automated)"  # Treat as automated after getting team ID
                )

    # Now run the async command
    asyncio.run(
        train_command(
            model=model,
            dataset=dataset,
            config=config,
            gpus=gpus,
            provider=provider,
            output_dir=output_dir,
            auto_create_pod=auto_create_pod,
            team_id=team_id,
        )
    )


async def train_command(
    model: str,
    dataset: str | None,
    config: Path | None,
    gpus: str,
    provider: str,
    output_dir: Path,
    auto_create_pod: str | None = None,
    team_id: str | None = None,
) -> None:
    """Run RL training on HUD environments."""
    design.header("🤖 HUD RL Training")

    # Get environment image
    image = detect_image_name()
    if not image:
        design.error("No environment image found")
        design.hint("Run 'hud build' first or specify with 'hud rl init <image>'")
        raise typer.Exit(1)

    # Validate dataset has sufficient tasks for training
    dataset_size = None
    if dataset:
        design.info(f"Validating dataset: {dataset}")
        try:
            # Try to load dataset info from HuggingFace
            from datasets import load_dataset_builder

            ds_builder = load_dataset_builder(dataset)
            ds_info = ds_builder.info

            # Check split sizes
            train_size = ds_info.splits.get("train", None) if ds_info.splits else None
            if train_size and train_size.num_examples < 4:
                design.error(f"Dataset '{dataset}' has only {train_size.num_examples} tasks")
                design.info("RL training requires at least 4 tasks for proper batching")
                design.hint("Consider adding more tasks or duplicating existing ones")
                raise typer.Exit(1)
            elif train_size:
                dataset_size = train_size.num_examples
                design.success(f"✓ Dataset has {dataset_size} tasks")
        except Exception as e:
            # If we can't validate, warn but continue
            design.warning(f"Could not validate dataset size: {e}")
            design.info("Proceeding with training - ensure dataset has at least 4 tasks")

    # Use dataset from command or lock file
    if not dataset:
        dataset = get_primary_dataset()
        if dataset:
            design.info(f"Using dataset from lock file: {dataset}")

    # Display configuration
    design.section_title("📋 Training Configuration")
    design.json_config(
        json.dumps(
            {
                "Model": model,
                "Dataset": dataset,
                "Config": str(config) if config else None,
                "Environment": image,
                "GPUs": gpus,
                "Provider": provider,
                "Output": str(output_dir),
            },
            indent=2,
        )
    )

    if not config:
        design.error("No config file found")
        design.hint("Run 'hud rl init' to generate a config file")
        raise typer.Exit(1)

    if not dataset:
        design.error("No dataset found")
        design.hint("Run 'hud hf tasks.json' to create a dataset")
        raise typer.Exit(1)

    # Always run remote training
    await run_remote_training(
        model=model,
        dataset=dataset,
        config=config,
        gpus=gpus,
        provider=provider,
        output_dir=output_dir,
        image=image,
        auto_create_pod=auto_create_pod,
        team_id=team_id,
        dataset_size=dataset_size,
    )


def check_requirements(config: Path | None, dataset: str | None) -> dict[str, Any]:
    """Check if required components are present."""
    missing = {}

    # Check config
    if not config:
        config_dir = Path("configs")
        if config_dir.exists():
            yaml_files = list(config_dir.glob("*.yaml"))
            if not yaml_files:
                missing["config"] = "none"
            elif len(yaml_files) > 1:
                missing["config"] = "multiple"
            # If exactly one config, we'll use it
        else:
            missing["config"] = "none"

    # Check dataset
    if not dataset:
        # Check lock file for dataset
        primary_dataset = get_primary_dataset()
        if not primary_dataset:
            missing["dataset"] = "none"

    return missing


def generate_config_interactive() -> Path | None:
    """Generate config interactively and return the path."""
    from .init import init_command

    # Run init command
    asyncio.run(init_command(".", None, False, False))

    # Look for generated config
    config_dir = Path("configs")
    if config_dir.exists():
        yaml_files = list(config_dir.glob("*.yaml"))
        if yaml_files:
            return yaml_files[0]

    return None


def create_dataset_interactive() -> str | None:
    """Create dataset interactively and return the name."""
    # Check if tasks.json exists
    tasks_file = Path("tasks.json")
    if not tasks_file.exists():
        design.error("No tasks.json file found")
        return None

    # Prompt for dataset name
    dataset_name = typer.prompt("Enter HuggingFace dataset name (e.g., username/dataset-name)")

    if not validate_dataset_name(dataset_name):
        design.error("Invalid dataset name format")
        return None

    # Run hf command
    result = subprocess.run(  # noqa: S603
        ["hud", "hf", "tasks.json", "--name", dataset_name],  # noqa: S607
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        return dataset_name
    else:
        design.error("Failed to create dataset")
        if result.stderr:
            design.error(result.stderr)
        return None


async def run_remote_training(
    model: str,
    dataset: str,
    config: Path,
    gpus: str,
    provider: str,
    output_dir: Path,
    image: str,
    auto_create_pod: str | None = None,
    team_id: str | None = None,
    dataset_size: int | None = None,
) -> None:
    """Run training on remote infrastructure."""
    design.section_title("🚀 Remote Training")

    if provider == "prime":
        await run_prime_training(
            model, dataset, config, gpus, output_dir, image, auto_create_pod, team_id, dataset_size
        )
    else:
        design.error(f"Provider '{provider}' not yet supported")
        design.info("Currently supported: prime")
        raise typer.Exit(1)
