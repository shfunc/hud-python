"""Push HUD environments to registry."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import click
import requests
import typer
import yaml
from rich.console import Console

from hud.settings import settings
from hud.utils.design import HUDDesign

console = Console()


def get_docker_username() -> str | None:
    """Get the current Docker username if logged in."""
    try:
        # Docker config locations
        config_paths = [
            Path.home() / ".docker" / "config.json",
            Path.home() / ".docker" / "plaintext-credentials.json",  # Alternative location
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)

                    # Look for auth entries
                    auths = config.get("auths", {})
                    for registry_url, auth_info in auths.items():
                        if (
                            any(
                                hub in registry_url
                                for hub in ["docker.io", "index.docker.io", "registry-1.docker.io"]
                            )
                            and "auth" in auth_info
                        ):
                            import base64

                            try:
                                decoded = base64.b64decode(auth_info["auth"]).decode()
                                username = decoded.split(":", 1)[0]
                                if username and username != "token":  # Skip token-based auth
                                    return username
                            except Exception:
                                click.echo("Failed to decode auth info", err=True)
                except Exception:
                    click.echo("Failed to get Docker username", err=True)

        # Alternative: Check credsStore/credHelpers
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)

                    # Check if using credential helpers
                    if "credsStore" in config:
                        # Try to get credentials from helper
                        helper = config["credsStore"]
                        try:
                            result = subprocess.run(  # noqa: S603
                                [f"docker-credential-{helper}", "list"],
                                capture_output=True,
                                text=True,
                            )
                            if result.returncode == 0:
                                creds = json.loads(result.stdout)
                                for url in creds:
                                    if "docker.io" in url:
                                        # Try to get the username
                                        get_result = subprocess.run(  # noqa: S603
                                            [f"docker-credential-{helper}", "get"],
                                            input=url,
                                            capture_output=True,
                                            text=True,
                                        )
                                        if get_result.returncode == 0:
                                            cred_data = json.loads(get_result.stdout)
                                            username = cred_data.get("Username", "")
                                            if username and username != "token":
                                                return username
                        except Exception:
                            click.echo("Failed to get Docker username", err=True)
                except Exception:
                    click.echo("Failed to get Docker username", err=True)
    except Exception:
        click.echo("Failed to get Docker username", err=True)
    return None


def get_docker_image_labels(image: str) -> dict:
    """Get labels from a Docker image."""
    try:
        result = subprocess.run(  # noqa: S603
            ["docker", "inspect", "--format", "{{json .Config.Labels}}", image],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout.strip()) or {}
    except Exception:
        click.echo("Failed to get Docker image labels", err=True)
        return {}


def push_environment(
    directory: str = ".",
    image: str | None = None,
    tag: str | None = None,
    sign: bool = False,
    yes: bool = False,
    verbose: bool = False,
) -> None:
    """Push HUD environment to registry."""
    design = HUDDesign()
    design.header("HUD Environment Push")

    # Find hud.lock.yaml in specified directory
    env_dir = Path(directory)
    lock_path = env_dir / "hud.lock.yaml"

    if not lock_path.exists():
        design.error(f"No hud.lock.yaml found in {directory}")
        design.info("Run 'hud build' first to generate a lock file")
        raise typer.Exit(1)

    # Check for API key first
    if not settings.api_key:
        design.error("No HUD API key found")
        console.print("\n[yellow]A HUD API key is required to push environments.[/yellow]")
        console.print("\nTo get started:")
        console.print("  1. Get your API key at: [link]https://hud.so/settings[/link]")
        console.print("  2. Set it: [cyan]export HUD_API_KEY=your-key-here[/cyan]")
        console.print("  3. Try again: [cyan]hud push[/cyan]\n")
        raise typer.Exit(1)

    # Load lock file
    with open(lock_path) as f:
        lock_data = yaml.safe_load(f)

    # Handle both old and new lock file formats
    local_image = lock_data.get("image", "")
    if not local_image and "build" in lock_data:
        # New format might have image elsewhere
        local_image = lock_data.get("image", "")

    # If no image specified, try to be smart
    if not image:
        # Check if user is logged in
        username = get_docker_username()
        if username:
            # Extract image name from lock file (handle @sha256:... format)
            base_image = local_image.split("@")[0] if "@" in local_image else local_image

            if ":" in base_image:
                base_name = base_image.split(":")[0]
                current_tag = base_image.split(":")[1]
            else:
                base_name = base_image
                current_tag = "latest"

            # Remove any existing registry prefix
            if "/" in base_name:
                base_name = base_name.split("/")[-1]

            # Use provided tag or default
            final_tag = tag if tag else current_tag

            # Suggest a registry image
            image = f"{username}/{base_name}:{final_tag}"
            design.info(f"Auto-detected Docker username: {username}")
            if tag:
                design.info(f"Using specified tag: {tag}")
            design.info(f"Will push to: {image}")

            if not yes and not typer.confirm(f"\nPush to {image}?"):
                design.info("Aborted.")
                raise typer.Exit(0)
        else:
            design.error(
                "Not logged in to Docker Hub. Please specify --image or run 'docker login'"
            )
            raise typer.Exit(1)
    elif tag:
        # Handle tag when image is provided
        if ":" in image:
            # Image already has a tag
            existing_tag = image.split(":")[-1]
            if existing_tag != tag:
                design.warning(f"Image already has tag '{existing_tag}', overriding with '{tag}'")
                image = image.rsplit(":", 1)[0] + f":{tag}"
            # else: tags match, no action needed
        else:
            # Image has no tag, append the specified one
            image = f"{image}:{tag}"
        design.info(f"Using specified tag: {tag}")
        design.info(f"Will push to: {image}")

    # Verify local image exists
    # Extract the tag part (before @sha256:...) for Docker operations
    local_tag = local_image.split("@")[0] if "@" in local_image else local_image

    # Verify the image exists locally
    try:
        subprocess.run(["docker", "inspect", local_tag], capture_output=True, check=True)  # noqa: S603, S607
    except subprocess.CalledProcessError:
        design.error(f"Local image not found: {local_tag}")
        design.info("Run 'hud build' first to create the image")
        raise typer.Exit(1)  # noqa: B904

    # Check if local image has the expected label
    labels = get_docker_image_labels(local_tag)
    expected_label = labels.get("org.hud.manifest.head", "")

    # Skip hash verification - the lock file may have been updated with digest after build
    if verbose and expected_label:
        design.info(f"Image label: {expected_label[:12]}...")

    # Tag the image for push
    design.progress_message(f"Tagging {local_tag} as {image}")
    subprocess.run(["docker", "tag", local_tag, image], check=True)  # noqa: S603, S607

    # Push the image
    design.progress_message(f"Pushing {image} to registry...")

    # Show push output
    process = subprocess.Popen(  # noqa: S603
        ["docker", "push", image],  # noqa: S607
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    for line in process.stdout or []:
        click.echo(line.rstrip(), err=True)

    process.wait()

    if process.returncode != 0:
        design.error("Push failed")
        raise typer.Exit(1)

    # Get the digest of the pushed image
    result = subprocess.run(  # noqa: S603
        ["docker", "inspect", "--format", "{{index .RepoDigests 0}}", image],  # noqa: S607
        capture_output=True,
        text=True,
    )

    if result.returncode == 0 and result.stdout.strip():
        pushed_digest = result.stdout.strip()
    else:
        pushed_digest = image

    # Success!
    design.success("Push complete!")

    # Show the final image reference
    console.print("\n[bold green]✓ Pushed image:[/bold green]")
    console.print(f"  [bold cyan]{pushed_digest}[/bold cyan]\n")

    # Update the lock file with registry information
    lock_data["image"] = pushed_digest

    # Add push information
    from datetime import datetime

    lock_data["push"] = {
        "source": local_image,
        "pushedAt": datetime.utcnow().isoformat() + "Z",
        "registry": pushed_digest.split("/")[0] if "/" in pushed_digest else "docker.io",
    }

    # Save updated lock file
    with open(lock_path, "w") as f:
        yaml.dump(lock_data, f, default_flow_style=False, sort_keys=False)

    console.print("[green]✓[/green] Updated lock file with registry image")

    # Upload lock file to HUD registry
    try:
        # Extract org/name:tag from the pushed image
        # e.g., "docker.io/hudpython/test_init:latest@sha256:..." -> "hudpython/test_init:latest"
        # e.g., "hudpython/test_init:v1.0" -> "hudpython/test_init:v1.0"
        registry_parts = pushed_digest.split("/")
        if len(registry_parts) >= 2:
            # Handle docker.io/org/name or just org/name
            if registry_parts[0] in ["docker.io", "registry-1.docker.io", "index.docker.io"]:
                # Remove registry prefix and get org/name:tag
                name_with_tag = "/".join(registry_parts[1:]).split("@")[0]
            else:
                # Just org/name:tag
                name_with_tag = "/".join(registry_parts[:2]).split("@")[0]

            # If no tag specified, use "latest"
            if ":" not in name_with_tag:
                name_with_tag = f"{name_with_tag}:latest"

            # Upload to HUD registry
            design.progress_message("Uploading metadata to HUD registry...")

            registry_url = f"{settings.hud_telemetry_url.rstrip('/')}/registry/envs/{name_with_tag}"

            # Prepare the payload
            payload = {
                "lock": yaml.dump(lock_data, default_flow_style=False, sort_keys=False),
                "digest": pushed_digest.split("@")[-1] if "@" in pushed_digest else "latest",
            }

            headers = {"Authorization": f"Bearer {settings.api_key}"}

            response = requests.post(registry_url, json=payload, headers=headers, timeout=10)

            if response.status_code in [200, 201]:
                design.success("Metadata uploaded to HUD registry")
                console.print(
                    f"  Others can now pull with: [cyan]hud pull {name_with_tag}[/cyan]\n"
                )
            else:
                design.warning(f"Could not upload to registry: {response.status_code}")
                if verbose:
                    design.info(f"Response: {response.text}")
                console.print("  Share [cyan]hud.lock.yaml[/cyan] manually\n")
        else:
            if verbose:
                design.info("Could not parse registry path for upload")
            console.print(
                "  Share [cyan]hud.lock.yaml[/cyan] to let others reproduce your exact environment\n"  # noqa: E501
            )
    except Exception as e:
        design.warning(f"Registry upload failed: {e}")
        console.print("  Share [cyan]hud.lock.yaml[/cyan] manually\n")

    # Show usage examples
    design.section_title("What's Next?")

    console.print("Test locally:")
    console.print(f"  [cyan]hud run {image}[/cyan]\n")

    console.print("Share environment:")
    console.print(
        "  Share the updated [cyan]hud.lock.yaml[/cyan] for others to reproduce your exact environment"  # noqa: E501
    )

    # TODO: Upload lock file to HUD registry
    if sign:
        design.warning("Signing not yet implemented")


def push_command(
    directory: str = ".",
    image: str | None = None,
    tag: str | None = None,
    sign: bool = False,
    yes: bool = False,
    verbose: bool = False,
) -> None:
    """Push HUD environment to registry."""
    push_environment(directory, image, tag, sign, yes, verbose)
