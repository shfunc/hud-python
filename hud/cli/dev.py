"""MCP Development Server - Hot-reload Python modules."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()


def show_dev_server_info(
    server_name: str,
    port: int,
    transport: str,
    inspector: bool,
    interactive: bool,
    env_dir: Path | None = None,
) -> str:
    """Show consistent server info for both Python and Docker modes.

    Returns the Cursor deeplink URL.
    """
    import base64
    import json

    # Generate Cursor deeplink
    server_config = {"url": f"http://localhost:{port}/mcp"}
    config_json = json.dumps(server_config, indent=2)
    config_base64 = base64.b64encode(config_json.encode()).decode()
    cursor_deeplink = (
        f"cursor://anysphere.cursor-deeplink/mcp/install?name={server_name}&config={config_base64}"
    )

    # Server section
    hud_console.section_title("Server")
    hud_console.info(f"{hud_console.sym.ITEM} {server_name}")
    if transport == "http":
        hud_console.info(f"{hud_console.sym.ITEM} http://localhost:{port}/mcp")
    else:
        hud_console.info(f"{hud_console.sym.ITEM} (stdio)")

    # Quick Links (only for HTTP mode)
    if transport == "http":
        hud_console.section_title("Quick Links")
        hud_console.info(f"{hud_console.sym.ITEM} Docs: http://localhost:{port}/docs")
        hud_console.info(f"{hud_console.sym.ITEM} Cursor: {cursor_deeplink}")

        # Check for VNC (browser environment)
        if env_dir and (env_dir / "environment" / "server.py").exists():
            try:
                content = (env_dir / "environment" / "server.py").read_text()
                if "x11vnc" in content.lower() or "vnc" in content.lower():
                    hud_console.info(f"{hud_console.sym.ITEM} VNC: http://localhost:8080/vnc.html")
            except Exception:  # noqa: S110
                pass

        # Inspector/Interactive status
        if inspector or interactive:
            hud_console.info("")
            if inspector:
                hud_console.info(f"{hud_console.sym.SUCCESS} Inspector launching...")
            if interactive:
                hud_console.info(f"{hud_console.sym.SUCCESS} Interactive mode enabled")

    hud_console.info("")
    hud_console.info(f"{hud_console.sym.SUCCESS} Hot-reload enabled")
    hud_console.info("")

    return cursor_deeplink


def auto_detect_module() -> tuple[str, Path | None] | tuple[None, None]:
    """Auto-detect MCP module in current directory.

    Looks for 'mcp' defined in either __init__.py or server.py.

    Returns:
        Tuple of (module_name, parent_dir_to_add_to_path) or (None, None)
    """
    cwd = Path.cwd()

    # First check __init__.py
    init_file = cwd / "__init__.py"
    if init_file.exists():
        try:
            content = init_file.read_text(encoding="utf-8")
            if "mcp" in content and ("= MCPServer" in content or "= FastMCP" in content):
                return (cwd.name, None)
        except Exception:  # noqa: S110
            pass

    # Then check main.py in current directory
    main_file = cwd / "main.py"
    if main_file.exists() and init_file.exists():
        try:
            content = main_file.read_text(encoding="utf-8")
            if "mcp" in content and ("= MCPServer" in content or "= FastMCP" in content):
                # Need to import as package.main, add parent to sys.path
                return (f"{cwd.name}.main", cwd.parent)
        except Exception:  # noqa: S110
            pass

    return (None, None)


def should_use_docker_mode(cwd: Path) -> bool:
    """Check if environment requires Docker mode (has Dockerfile in current dir)."""
    return (cwd / "Dockerfile").exists()


async def run_mcp_module(
    module_name: str,
    transport: str,
    port: int,
    verbose: bool,
    inspector: bool,
    interactive: bool,
) -> None:
    """Run an MCP module directly."""
    # Check if this is a reload (not first run)
    is_reload = os.environ.get("_HUD_DEV_RELOAD") == "1"

    # Configure logging
    if verbose:
        logging.basicConfig(
            stream=sys.stderr, level=logging.DEBUG, format="[%(levelname)s] %(message)s"
        )
    else:
        # Suppress tracebacks in logs unless verbose
        logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")

        # Suppress FastMCP's verbose error logging
        logging.getLogger("fastmcp.tools.tool_manager").setLevel(logging.WARNING)

        # On reload, suppress most startup logs
        if is_reload:
            logging.getLogger("hud.server.server").setLevel(logging.ERROR)
            logging.getLogger("mcp.server").setLevel(logging.ERROR)
            logging.getLogger("mcp.server.streamable_http_manager").setLevel(logging.ERROR)

            # Suppress deprecation warnings on reload
            import warnings

            warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Ensure proper directory is in sys.path based on module name
    cwd = Path.cwd()
    if "." in module_name:
        # For package.module imports (like server.server), add parent to sys.path
        parent = str(cwd.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
    else:
        # For simple module imports, add current directory
        cwd_str = str(cwd)
        if cwd_str not in sys.path:
            sys.path.insert(0, cwd_str)

    # Import the module
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        hud_console.error(f"Failed to import module '{module_name}'")
        hud_console.info(f"Error: {e}")
        hud_console.info("")
        hud_console.info("[bold cyan]Troubleshooting:[/bold cyan]")
        hud_console.info("  • Verify module exists and is importable")
        hud_console.info("  • Check for __init__.py in module directory")
        hud_console.info("  • Check for import errors in the module")
        if verbose:
            import traceback

            hud_console.info("")
            hud_console.info("[bold cyan]Full traceback:[/bold cyan]")
            hud_console.info(traceback.format_exc())
        sys.exit(1)

    # Look for 'mcp' attribute - check module __dict__ directly
    # Debug: print what's in the module
    if verbose:
        hud_console.info(f"Module attributes: {dir(module)}")
        module_dict = module.__dict__ if hasattr(module, "__dict__") else {}
        hud_console.info(f"Module __dict__ keys: {list(module_dict.keys())}")

    mcp_server = None

    # Try different ways to access the mcp variable
    if hasattr(module, "mcp"):
        mcp_server = module.mcp
    elif hasattr(module, "__dict__") and "mcp" in module.__dict__:
        mcp_server = module.__dict__["mcp"]

    if mcp_server is None:
        hud_console.error(f"Module '{module_name}' does not have 'mcp' defined")
        hud_console.info("")
        available = [k for k in dir(module) if not k.startswith("_")]
        hud_console.info(f"Available in module: {available}")
        hud_console.info("")
        hud_console.info("[bold cyan]Expected structure:[/bold cyan]")
        hud_console.info("  from hud.server import MCPServer")
        hud_console.info("  mcp = MCPServer(name='my-server')")
        raise AttributeError(f"Module '{module_name}' must define 'mcp'")

    # Only show full header on first run, brief message on reload
    if is_reload:
        hud_console.info(f"{hud_console.sym.SUCCESS} Reloaded")
        # Run server without showing full UI
    else:
        # Show full header on first run
        hud_console.info("")
        hud_console.header("HUD Development Server")

    # Show server info only on first run
    if not is_reload:
        show_dev_server_info(
            server_name=mcp_server.name or "mcp-server",
            port=port,
            transport=transport,
            inspector=inspector,
            interactive=interactive,
            env_dir=Path.cwd().parent if (Path.cwd().parent / "environment").exists() else None,
        )

    # Check if there's an environment backend and remind user to start it (first run only)
    if not is_reload:
        cwd = Path.cwd()
        env_dir = cwd.parent / "environment"
        if env_dir.exists() and (env_dir / "server.py").exists():
            hud_console.info("")
            hud_console.info(
                f"{hud_console.sym.FLOW} Don't forget to start the environment backend:"
            )
            hud_console.info("   cd ../environment && uvicorn server:app --reload")

        # Launch inspector if requested (first run only)
        if inspector and transport == "http":
            await launch_inspector(port)

        # Launch interactive mode if requested (first run only)
        if interactive and transport == "http":
            launch_interactive_thread(port, verbose)

        hud_console.info("")

    # Configure server options
    run_kwargs = {
        "transport": transport,
        "show_banner": False,
    }

    if transport == "http":
        run_kwargs["port"] = port
        run_kwargs["path"] = "/mcp"
        run_kwargs["host"] = "0.0.0.0"  # noqa: S104
        run_kwargs["log_level"] = "INFO" if verbose else "ERROR"

    # Run the server
    await mcp_server.run_async(**run_kwargs)


async def launch_inspector(port: int) -> None:
    """Launch MCP Inspector in background."""
    await asyncio.sleep(2)

    try:
        import platform
        import urllib.parse

        server_url = f"http://localhost:{port}/mcp"
        encoded_url = urllib.parse.quote(server_url)
        inspector_url = f"http://localhost:6274/?transport=streamable-http&serverUrl={encoded_url}"

        hud_console.section_title("MCP Inspector")
        hud_console.link(inspector_url)

        env = os.environ.copy()
        env["DANGEROUSLY_OMIT_AUTH"] = "true"
        env["MCP_AUTO_OPEN_ENABLED"] = "true"

        cmd = ["npx", "--yes", "@modelcontextprotocol/inspector"]

        if platform.system() == "Windows":
            subprocess.Popen(  # noqa: S602, ASYNC220
                cmd,
                env=env,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(  # noqa: S603, ASYNC220
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    except Exception as e:
        hud_console.error(f"Failed to launch inspector: {e}")


def launch_interactive_thread(port: int, verbose: bool) -> None:
    """Launch interactive testing mode in separate thread."""
    import time

    def run_interactive() -> None:
        time.sleep(2)

        try:
            hud_console.section_title("Interactive Mode")
            hud_console.info("Starting interactive testing mode...")

            from .utils.interactive import run_interactive_mode

            server_url = f"http://localhost:{port}/mcp"

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_interactive_mode(server_url, verbose))
            finally:
                loop.close()

        except Exception as e:
            if verbose:
                hud_console.error(f"Interactive mode error: {e}")

    interactive_thread = threading.Thread(target=run_interactive, daemon=True)
    interactive_thread.start()


def run_with_reload(
    module_name: str,
    watch_paths: list[str],
    transport: str,
    port: int,
    verbose: bool,
    inspector: bool,
    interactive: bool,
) -> None:
    """Run module with file watching and auto-reload."""
    try:
        import watchfiles
    except ImportError:
        hud_console.error("watchfiles required. Install: pip install watchfiles")
        sys.exit(1)

    # Resolve watch paths
    resolved_paths = []
    for path_str in watch_paths:
        path = Path(path_str).resolve()
        if path.is_file():
            resolved_paths.append(str(path.parent))
        else:
            resolved_paths.append(str(path))

    if verbose:
        hud_console.info(f"Watching: {', '.join(resolved_paths)}")

    import signal

    process = None
    stop_event = threading.Event()
    is_first_run = True

    def handle_signal(signum: int, frame: Any) -> None:
        if process:
            process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while True:
        cmd = [sys.executable, "-m", "hud", "dev", module_name, f"--port={port}"]

        if transport == "stdio":
            cmd.append("--stdio")

        if verbose:
            cmd.append("--verbose")
            hud_console.info(f"Starting: {' '.join(cmd)}")

        # Mark as reload after first run to suppress logs
        env = {**os.environ, "_HUD_DEV_CHILD": "1"}
        if not is_first_run:
            env["_HUD_DEV_RELOAD"] = "1"

        process = subprocess.Popen(  # noqa: S603
            cmd, env=env
        )

        is_first_run = False

        try:
            stop_event = threading.Event()

            def _wait_and_set(
                stop_event: threading.Event, process: subprocess.Popen[bytes]
            ) -> None:
                try:
                    if process is not None:
                        process.wait()
                finally:
                    stop_event.set()

            threading.Thread(target=_wait_and_set, args=(stop_event, process), daemon=True).start()

            for changes in watchfiles.watch(*resolved_paths, stop_event=stop_event):
                relevant_changes = [
                    (change_type, path)
                    for change_type, path in changes
                    if any(path.endswith(ext) for ext in [".py", ".json", ".toml", ".yaml"])
                    and "__pycache__" not in path
                    and not Path(path).name.startswith(".")
                ]

                if relevant_changes:
                    hud_console.flow("File changes detected, reloading...")
                    if verbose:
                        for change_type, path in relevant_changes:
                            hud_console.info(f"  {change_type}: {path}")

                    if process is not None:
                        process.terminate()
                    try:
                        if process is not None:
                            process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        if process is not None:
                            process.kill()
                            process.wait()

                    import time

                    time.sleep(0.1)
                    break

        except KeyboardInterrupt:
            if process:
                process.terminate()
                process.wait()
            break


def run_docker_dev_server(
    port: int, verbose: bool, inspector: bool, interactive: bool, docker_args: list[str]
) -> None:
    """Run MCP server in Docker with volume mounts, expose via local HTTP proxy."""
    import typer
    import yaml

    from hud.server import MCPServer

    cwd = Path.cwd()

    # Find environment directory (current or parent with hud.lock.yaml)
    env_dir = cwd
    lock_path = env_dir / "hud.lock.yaml"

    if not lock_path.exists():
        # Try parent directory
        if (cwd.parent / "hud.lock.yaml").exists():
            env_dir = cwd.parent
            lock_path = env_dir / "hud.lock.yaml"
        else:
            hud_console.error("No hud.lock.yaml found")
            hud_console.info("Run 'hud build' first to create an image")
            raise typer.Exit(1)

    # Load lock file to get image name
    try:
        with open(lock_path) as f:
            lock_data = yaml.safe_load(f)

        # Get image from new or legacy format
        images = lock_data.get("images", {})
        image_name = images.get("local") or images.get("full") or lock_data.get("image")

        if not image_name:
            hud_console.error("No image reference found in hud.lock.yaml")
            raise typer.Exit(1)

        # Strip digest if present
        if "@" in image_name:
            image_name = image_name.split("@")[0]

    except Exception as e:
        hud_console.error(f"Failed to read lock file: {e}")
        raise typer.Exit(1) from e

    # Generate unique container name
    pid = str(os.getpid())[-6:]
    base_name = image_name.replace(":", "-").replace("/", "-")
    container_name = f"{base_name}-dev-{pid}"

    # Build docker run command with volume mounts
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--name",
        container_name,
        # Mount both server and environment for hot-reload
        "-v",
        f"{env_dir.absolute()}/server:/app/server:rw",
        "-v",
        f"{env_dir.absolute()}/environment:/app/environment:rw",
        "-e",
        "PYTHONPATH=/app",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "HUD_DEV=1",
    ]

    # Load .env file if present
    env_file = env_dir / ".env"
    loaded_env_vars: dict[str, str] = {}
    if env_file.exists():
        try:
            from hud.cli.utils.config import parse_env_file

            env_contents = env_file.read_text(encoding="utf-8")
            loaded_env_vars = parse_env_file(env_contents)
            for key, value in loaded_env_vars.items():
                docker_cmd.extend(["-e", f"{key}={value}"])
            if verbose and loaded_env_vars:
                hud_console.info(f"Loaded {len(loaded_env_vars)} env var(s) from .env")
        except Exception as e:
            hud_console.warning(f"Failed to load .env file: {e}")

    # Add user-provided Docker arguments
    if docker_args:
        docker_cmd.extend(docker_args)

    # Append the image name
    docker_cmd.append(image_name)

    # Print startup info
    hud_console.header("HUD Development Mode (Docker)")

    if verbose:
        hud_console.section_title("Docker Command")
        hud_console.info(" ".join(docker_cmd))

    # Create MCP config pointing to the Docker container's stdio
    mcp_config = {
        "docker": {
            "command": docker_cmd[0],
            "args": docker_cmd[1:],
        }
    }

    # Show consistent server info
    show_dev_server_info(
        server_name=image_name,
        port=port,
        transport="http",  # Docker mode always uses HTTP proxy
        inspector=inspector,
        interactive=interactive,
        env_dir=env_dir,
    )

    # Suppress logs unless verbose
    if not verbose:
        logging.getLogger("fastmcp").setLevel(logging.ERROR)
        logging.getLogger("mcp").setLevel(logging.ERROR)
        logging.getLogger("uvicorn").setLevel(logging.ERROR)
        os.environ["FASTMCP_DISABLE_BANNER"] = "1"

    # Note about hot-reload behavior
    hud_console.dim_info(
        "",
        "Container restarts on file changes (mounted volumes), if changing tools run hud dev again",
    )
    hud_console.info("")

    # Create and run proxy with HUD helpers
    async def run_proxy() -> None:
        from fastmcp import FastMCP

        # Create FastMCP proxy to Docker stdio
        fastmcp_proxy = FastMCP.as_proxy(mcp_config, name="HUD Docker Dev Proxy")

        # Wrap in MCPServer to get /docs and REST wrappers
        proxy = MCPServer(name="HUD Docker Dev Proxy")

        # Import all tools from the FastMCP proxy
        await proxy.import_server(fastmcp_proxy)

        # Launch inspector if requested
        if inspector:
            await launch_inspector(port)

        # Launch interactive mode if requested
        if interactive:
            launch_interactive_thread(port, verbose)

        # Run proxy with HTTP transport
        await proxy.run_async(
            transport="http",
            host="0.0.0.0",  # noqa: S104
            port=port,
            path="/mcp",
            log_level="error" if not verbose else "info",
            show_banner=False,
        )

    try:
        asyncio.run(run_proxy())
    except KeyboardInterrupt:
        hud_console.info("\n\nStopping...")
        raise typer.Exit(0) from None


def run_mcp_dev_server(
    module: str | None,
    stdio: bool,
    port: int,
    verbose: bool,
    inspector: bool,
    interactive: bool,
    watch: list[str] | None,
    docker: bool = False,
    docker_args: list[str] | None = None,
) -> None:
    """Run MCP development server with hot-reload."""
    docker_args = docker_args or []
    cwd = Path.cwd()

    # Auto-detect Docker mode if Dockerfile present and no module specified
    if not docker and module is None and should_use_docker_mode(cwd):
        hud_console.note("Detected Dockerfile - using Docker mode with volume mounts")
        hud_console.dim_info("Tip", "Use 'hud dev --help' to see all options")
        hud_console.info("")
        run_docker_dev_server(port, verbose, inspector, interactive, docker_args)
        return

    # Route to Docker mode if explicitly requested
    if docker:
        run_docker_dev_server(port, verbose, inspector, interactive, docker_args)
        return

    transport = "stdio" if stdio else "http"

    # Auto-detect module if not provided
    if module is None:
        module, extra_path = auto_detect_module()
        if module is None:
            hud_console.error("Could not auto-detect MCP module in current directory")
            hud_console.info("")
            hud_console.info("[bold cyan]Expected:[/bold cyan]")
            hud_console.info("  • __init__.py file in current directory")
            hud_console.info("  • Module must define 'mcp' variable")
            hud_console.info("")
            hud_console.info("[bold cyan]Examples:[/bold cyan]")
            hud_console.info("  hud dev controller")
            hud_console.info("  cd controller && hud dev")
            hud_console.info("  hud dev --docker  # For Docker-based environments")
            hud_console.info("")
            import sys

            sys.exit(1)

        if verbose:
            hud_console.info(f"Auto-detected: {module}")
            if extra_path:
                hud_console.info(f"Adding to sys.path: {extra_path}")

        # Add extra path to sys.path if needed (for package imports)
        if extra_path:
            import sys

            sys.path.insert(0, str(extra_path))
        else:
            extra_path = None

    # Determine watch paths
    watch_paths = watch if watch else ["."]

    # Check if child process
    is_child = os.environ.get("_HUD_DEV_CHILD") == "1"

    if is_child:
        asyncio.run(run_mcp_module(module, transport, port, verbose, False, False))
    else:
        run_with_reload(module, watch_paths, transport, port, verbose, inspector, interactive)
