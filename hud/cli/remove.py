"""Remove HUD environments from local registry."""

from __future__ import annotations

import shutil

import typer

from hud.utils.design import HUDDesign

from .utils.registry import get_registry_dir, list_registry_entries, load_from_registry


def remove_environment(
    target: str,
    yes: bool = False,
    verbose: bool = False,
) -> None:
    """Remove an environment from the local registry."""
    design = HUDDesign()
    design.header("HUD Environment Removal")

    # Find the environment to remove
    found_entry = None
    found_digest = None

    # First check if target is a digest
    for digest, lock_file in list_registry_entries():
        if digest.startswith(target):
            found_entry = lock_file
            found_digest = digest
            break

    # If not found by digest, search by name
    if not found_entry:
        for digest, lock_file in list_registry_entries():
            try:
                lock_data = load_from_registry(digest)
                if lock_data and "image" in lock_data:
                    image = lock_data["image"]
                    # Extract name and tag
                    name = image.split("@")[0] if "@" in image else image
                    if "/" in name and (target in name or name.endswith(f"/{target}")):
                        found_entry = lock_file
                        found_digest = digest
                        break
            except Exception as e:
                design.error(f"Error loading lock file: {e}")
                continue

    if not found_entry:
        design.error(f"Environment not found: {target}")
        design.info("Use 'hud list' to see available environments")
        raise typer.Exit(1)

    # Load and display environment info
    try:
        if found_digest is None:
            raise ValueError("Found digest is None")
        lock_data = load_from_registry(found_digest)
        if lock_data:
            image = lock_data.get("image", "unknown")
            metadata = lock_data.get("metadata", {})
            description = metadata.get("description", "No description")

            design.section_title("Environment Details")
            design.status_item("Image", image)
            design.status_item("Digest", found_digest)
            design.status_item("Description", description)
            design.status_item("Location", str(found_entry.parent))
    except Exception as e:
        if verbose:
            design.warning(f"Could not read environment details: {e}")

    # Confirm deletion
    if not yes:
        design.info("")
        if not typer.confirm(f"Remove environment {found_digest}?"):
            design.info("Aborted")
            raise typer.Exit(0)

    # Remove the environment directory
    try:
        env_dir = found_entry.parent
        shutil.rmtree(env_dir)
        design.success(f"Removed environment: {found_digest}")

        # Check if the image is still available locally
        if lock_data:
            image = lock_data.get("image", "")
            if image:
                design.info("")
                design.info("Note: The Docker image may still exist locally.")
                design.info(f"To remove it, run: [cyan]docker rmi {image.split('@')[0]}[/cyan]")
    except Exception as e:
        design.error(f"Failed to remove environment: {e}")
        raise typer.Exit(1) from e


def remove_all_environments(
    yes: bool = False,
    verbose: bool = False,
) -> None:
    """Remove all environments from the local registry."""
    design = HUDDesign()
    design.header("Remove All HUD Environments")

    registry_dir = get_registry_dir()
    if not registry_dir.exists():
        design.info("No environments found in local registry.")
        return

    # Count environments
    entries = list(list_registry_entries())
    if not entries:
        design.info("No environments found in local registry.")
        return

    design.warning(f"This will remove {len(entries)} environment(s) from the local registry!")

    # List environments that will be removed
    design.section_title("Environments to Remove")
    for digest, _ in entries:
        try:
            lock_data = load_from_registry(digest)
            if lock_data:
                image = lock_data.get("image", "unknown")
                design.info(f"  • {digest[:12]} - {image}")
        except Exception:
            design.info(f"  • {digest[:12]}")

    # Confirm deletion
    if not yes:
        design.info("")
        if not typer.confirm("Remove ALL environments?", default=False):
            design.info("Aborted")
            raise typer.Exit(0)

    # Remove all environments
    removed = 0
    failed = 0

    for digest, lock_file in entries:
        try:
            env_dir = lock_file.parent
            shutil.rmtree(env_dir)
            removed += 1
            if verbose:
                design.success(f"Removed: {digest}")
        except Exception as e:
            failed += 1
            if verbose:
                design.error(f"Failed to remove {digest}: {e}")

    design.info("")
    if failed == 0:
        design.success(f"Successfully removed {removed} environment(s)")
    else:
        design.warning(f"Removed {removed} environment(s), failed to remove {failed}")

    design.info("")
    design.info("Note: Docker images may still exist locally.")
    design.info("To remove them, use: [cyan]docker image prune[/cyan]")


def remove_command(
    target: str | None = typer.Argument(
        None, help="Environment to remove (digest, name, or 'all' for all environments)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """🗑️ Remove HUD environments from local registry.

    Removes environment metadata from ~/.hud/envs/
    Note: This does not remove the Docker images.

    Examples:
        hud remove abc123              # Remove by digest
        hud remove text_2048           # Remove by name
        hud remove hudpython/test_init # Remove by full name
        hud remove all                 # Remove all environments
        hud remove all --yes           # Remove all without confirmation
    """
    if not target:
        design = HUDDesign()
        design.error("Please specify an environment to remove or 'all'")
        design.info("Use 'hud list' to see available environments")
        raise typer.Exit(1)

    if target.lower() == "all":
        remove_all_environments(yes, verbose)
    else:
        remove_environment(target, yes, verbose)
