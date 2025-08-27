"""Push HUD environments to registry."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from urllib.parse import quote

import requests
import typer
import yaml

from hud.settings import settings
from hud.utils.design import HUDDesign


def _get_response_text(response: requests.Response) -> str:
    try:
        return response.json().get("detail", "No detail available")
    except Exception:
        return response.text


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
                            except Exception:  # noqa: S110
                                pass
                except Exception:  # noqa: S110
                    pass

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
                        except Exception:  # noqa: S110
                            pass
                except Exception:  # noqa: S110
                    pass
    except Exception:  # noqa: S110
        pass
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
        design.warning("A HUD API key is required to push environments.")
        design.info("\nTo get started:")
        design.info("1. Get your API key at: https://hud.so/settings")
        design.command_example("export HUD_API_KEY=your-key-here", "Set your API key")
        design.command_example("hud push", "Try again")
        design.info("")
        raise typer.Exit(1)

    # Load lock file
    with open(lock_path) as f:
        lock_data = yaml.safe_load(f)

    # Handle both old and new lock file formats
    local_image = lock_data.get("image", "")
    if not local_image and "build" in lock_data:
        # New format might have image elsewhere
        local_image = lock_data.get("image", "")

    # Get internal version from lock file
    internal_version = lock_data.get("build", {}).get("version", None)

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

            # Use provided tag, or internal version, or current tag as fallback
            if tag:
                final_tag = tag
                design.info(f"Using specified tag: {tag}")
            elif internal_version:
                final_tag = internal_version
                design.info(f"Using internal version from lock file: {internal_version}")
            else:
                final_tag = current_tag
                design.info(f"Using current tag: {current_tag}")

            # Suggest a registry image
            image = f"{username}/{base_name}:{final_tag}"
            design.info(f"Auto-detected Docker username: {username}")
            design.info(f"Will push to: {image}")

            if not yes and not typer.confirm(f"\nPush to {image}?"):
                design.info("Aborted.")
                raise typer.Exit(0)
        else:
            design.error(
                "Not logged in to Docker Hub. Please specify --image or run 'docker login'"
            )
            raise typer.Exit(1)
    elif tag or internal_version:
        # Handle tag when image is provided
        # Prefer explicit tag over internal version
        final_tag = tag if tag else internal_version

        if ":" in image:
            # Image already has a tag
            existing_tag = image.split(":")[-1]
            if existing_tag != final_tag:
                if tag:
                    design.warning(
                        f"Image already has tag '{existing_tag}', overriding with '{final_tag}'"
                    )
                else:
                    design.info(
                        f"Image has tag '{existing_tag}', but using internal version '{final_tag}'"
                    )
                image = image.rsplit(":", 1)[0] + f":{final_tag}"
            # else: tags match, no action needed
        else:
            # Image has no tag, append the appropriate one
            image = f"{image}:{final_tag}"

        if tag:
            design.info(f"Using specified tag: {tag}")
        else:
            design.info(f"Using internal version from lock file: {internal_version}")
        design.info(f"Will push to: {image}")

    # Verify local image exists
    # Extract the tag part (before @sha256:...) for Docker operations
    local_tag = local_image.split("@")[0] if "@" in local_image else local_image

    # Also check for version-tagged image if we have internal version
    version_tag = None
    if internal_version and ":" in local_tag:
        base_name = local_tag.split(":")[0]
        version_tag = f"{base_name}:{internal_version}"

    # Try to find the image - prefer version tag if it exists
    image_to_push = None
    if version_tag:
        try:
            subprocess.run(["docker", "inspect", version_tag], capture_output=True, check=True)  # noqa: S603, S607
            image_to_push = version_tag
            design.info(f"Found version-tagged image: {version_tag}")
        except subprocess.CalledProcessError:
            pass

    if not image_to_push:
        try:
            subprocess.run(["docker", "inspect", local_tag], capture_output=True, check=True)  # noqa: S603, S607
            image_to_push = local_tag
        except subprocess.CalledProcessError:
            design.error(f"Local image not found: {local_tag}")
            if version_tag:
                design.error(f"Also tried: {version_tag}")
            design.info("Run 'hud build' first to create the image")
            raise typer.Exit(1)  # noqa: B904

    # Check if local image has the expected label
    labels = get_docker_image_labels(image_to_push)
    expected_label = labels.get("org.hud.manifest.head", "")
    version_label = labels.get("org.hud.version", "")

    # Skip hash verification - the lock file may have been updated with digest after build
    if verbose:
        if expected_label:
            design.info(f"Image label: {expected_label[:12]}...")
        if version_label:
            design.info(f"Version label: {version_label}")

    # Tag the image for push
    design.progress_message(f"Tagging {image_to_push} as {image}")
    subprocess.run(["docker", "tag", image_to_push, image], check=True)  # noqa: S603, S607

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
        design.info(line.rstrip())

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
    design.section_title("Pushed Image")
    design.status_item("Registry", pushed_digest, primary=True)

    # Update the lock file with registry information
    lock_data["image"] = pushed_digest

    # Add push information
    from datetime import UTC, datetime

    lock_data["push"] = {
        "source": local_image,
        "pushedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "registry": pushed_digest.split("/")[0] if "/" in pushed_digest else "docker.io",
    }

    # Save updated lock file
    with open(lock_path, "w") as f:
        yaml.dump(lock_data, f, default_flow_style=False, sort_keys=False)

    design.success("Updated lock file with registry image")

    # Upload lock file to HUD registry
    try:
        # Extract org/name:tag from the pushed image
        # e.g., "docker.io/hudpython/test_init:latest@sha256:..." -> "hudpython/test_init:latest"
        # e.g., "hudpython/test_init:v1.0" -> "hudpython/test_init:v1.0"
        # Use the original image name for the registry path, not the digest
        # The digest might not contain the tag information
        registry_image = (
            image  # This is the image we tagged and pushed (e.g., hudpython/hud-text-2048:0.1.2)
        )

        # Remove any registry prefix for the HUD registry path
        registry_parts = registry_image.split("/")
        if len(registry_parts) >= 2:
            # Handle docker.io/org/name or just org/name
            if registry_parts[0] in [
                "docker.io",
                "registry-1.docker.io",
                "index.docker.io",
                "ghcr.io",
            ]:
                # Remove registry prefix
                name_with_tag = "/".join(registry_parts[1:])
            elif "." in registry_parts[0] or ":" in registry_parts[0]:
                # Likely a registry URL (has dots or port)
                name_with_tag = "/".join(registry_parts[1:])
            else:
                # No registry prefix, use as-is
                name_with_tag = registry_image
        else:
            name_with_tag = registry_image

        # The image variable already has the tag, no need to add :latest

        # Validate the image format
        if not name_with_tag:
            design.warning("Could not determine image name for registry upload")
            raise typer.Exit(0)

        # For HUD registry, we need org/name format
        if "/" not in name_with_tag:
            design.warning("Image name must include organization/namespace for HUD registry")
            design.info(f"Current format: {name_with_tag}")
            design.info("Expected format: org/name:tag (e.g., hudpython/myenv:v1.0)")
            design.info("\nYour Docker push succeeded - share hud.lock.yaml manually")
            raise typer.Exit(0)

        # Upload to HUD registry
        design.progress_message("Uploading metadata to HUD registry...")

        # URL-encode the path segments to handle special characters in tags
        url_safe_path = "/".join(quote(part, safe="") for part in name_with_tag.split("/"))
        registry_url = f"{settings.hud_telemetry_url.rstrip('/')}/registry/envs/{url_safe_path}"

        # Prepare the payload
        payload = {
            "lock": yaml.dump(lock_data, default_flow_style=False, sort_keys=False),
            "digest": pushed_digest.split("@")[-1] if "@" in pushed_digest else None,
        }

        headers = {"Authorization": f"Bearer {settings.api_key}"}

        response = requests.post(registry_url, json=payload, headers=headers, timeout=10)

        if response.status_code in [200, 201]:
            design.success("Metadata uploaded to HUD registry")
            design.info("Others can now pull with:")
            design.command_example(f"hud pull {name_with_tag}")
            design.info("")
        elif response.status_code == 401:
            design.error("Authentication failed")
            design.info("Check your HUD_API_KEY is valid")
            design.info("Get a new key at: https://hud.so/settings")
        elif response.status_code == 403:
            design.error("Permission denied")
            design.info("You may not have access to push to this namespace")
        elif response.status_code == 409:
            design.warning("This version already exists in the registry")
            design.info("Consider using a different tag if you want to update")
        else:
            design.warning(f"Could not upload to registry: {response.status_code}")
            design.warning(_get_response_text(response))
            design.info("Share hud.lock.yaml manually\n")
    except requests.exceptions.Timeout:
        design.warning("Registry upload timed out")
        design.info("The registry might be slow or unavailable")
        design.info("Your Docker push succeeded - share hud.lock.yaml manually")
    except requests.exceptions.ConnectionError:
        design.warning("Could not connect to HUD registry")
        design.info("Check your internet connection")
        design.info("Your Docker push succeeded - share hud.lock.yaml manually")
    except Exception as e:
        design.warning(f"Registry upload failed: {e}")
        design.info("Share hud.lock.yaml manually")

    # Show usage examples
    design.section_title("What's Next?")

    design.info("Test locally:")
    design.command_example(f"hud run {image}")
    design.info("")
    design.info("Share environment:")
    design.info("  Share the updated hud.lock.yaml for others to reproduce your exact environment")

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
