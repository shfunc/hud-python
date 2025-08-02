"""
Service management for the HUD Browser Environment.

Handles starting and monitoring X11, VNC, and application services.
"""

import asyncio
import subprocess
import os
import logging
from pathlib import Path
from typing import Optional, Dict, List
import socket

logger = logging.getLogger(__name__)


# ---- SERVICES ----
# All services that are needed for the environment to run
# (e.g. X11, VNC, apps, browser)
#
# This is a separate class from the server to make it easier to test
# and modify the services without having to restart the server.
#
# The server will launch the services and wait for them to be ready.
class ServiceManager:
    """Manages environment services (X11, VNC, apps)."""

    def __init__(self):
        self.x11_proc: Optional[subprocess.Popen] = None
        self.vnc_proc: Optional[subprocess.Popen] = None
        self.websockify_proc: Optional[subprocess.Popen] = None
        self._launched_apps: List[str] = []
        self._app_processes: Dict[str, subprocess.Popen] = {}
        self._app_ports: Dict[
            str, Dict[str, int]
        ] = {}  # Store frontend and backend ports for each app

    async def start_services(self):
        """Start all core services in parallel where possible."""
        # Check if X11 is already running (started by start.sh)
        if Path("/tmp/.X11-unix/X1").exists():
            logger.info("X11 display :1 already running (started by start.sh)")
        else:
            # Start Xvfb if not already running
            self.x11_proc = subprocess.Popen(
                ["Xvfb", ":1", "-screen", "0", "1920x1080x24"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            logger.info("Started Xvfb on display :1")

    async def wait_for_x11(self):
        """Wait for X11 display to be ready."""
        for i in range(100):  # 10 seconds max
            if Path("/tmp/.X11-unix/X1").exists():
                logger.info("X11 display :1 is ready")
                # Set DISPLAY for all subsequent processes
                os.environ["DISPLAY"] = ":1"
                return
            await asyncio.sleep(0.1)
        raise TimeoutError("X11 failed to start")

    async def wait_for_vnc(self):
        """Start VNC and websockify in parallel, then wait for both."""
        # Start x11vnc (depends on X11 which is already up)
        self.vnc_proc = subprocess.Popen(
            ["x11vnc", "-display", ":1", "-forever", "-shared", "-nopw"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env={**os.environ, "DISPLAY": ":1"},
        )
        logger.info("Started x11vnc")

        # Start websockify immediately (it will wait for VNC port)
        self.websockify_proc = subprocess.Popen(
            ["websockify", "--web", "/usr/share/novnc", "8080", "localhost:5900"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        logger.info("Started websockify on port 8080")

        # Wait for both services in parallel
        vnc_ready = asyncio.create_task(self._wait_for_port(5900, "VNC"))
        websockify_ready = asyncio.create_task(self._wait_for_port(8080, "websockify"))

        # Wait for both to be ready
        await asyncio.gather(vnc_ready, websockify_ready)
        logger.info("noVNC available at: http://localhost:8080/vnc.html")

    async def launch_apps(self):
        """Launch configured applications in parallel."""
        apps = os.getenv("LAUNCH_APPS", "").split(",")
        tasks = []

        for app in apps:
            app = app.strip()
            if app:
                tasks.append(self.launch_app(app))

        if tasks:
            # Launch all apps in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for app, result in zip(apps, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to launch app '{app.strip()}': {result}")

    async def launch_app(self, app_name: str) -> Dict[str, str]:
        """Launch a specific app dynamically.

        Args:
            app_name: Name of the app to launch

        Returns:
            Dict with app info including URL
        """
        app_path = Path(f"/app/apps/{app_name}")
        if not app_path.exists():
            raise ValueError(f"App '{app_name}' not found at {app_path}")

        # Check if app has a launch script
        launch_script = app_path / "launch.py"
        if not launch_script.exists():
            raise ValueError(f"App '{app_name}' missing launch.py")

        # For todo app, use specific ports
        if app_name == "todo":
            frontend_port = 3000
            backend_port = 5000
        else:
            # Find an available port for other apps
            frontend_port = self._get_next_port()
            backend_port = self._get_next_port()

        # Launch the app with both frontend and backend ports
        # Note: Can't use stdout/stderr pipes as stdio is reserved for MCP
        proc = subprocess.Popen(
            [
                "python3",
                str(launch_script),
                "--frontend-port",
                str(frontend_port),
                "--backend-port",
                str(backend_port),
            ],
            cwd=app_path,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,  # Must be DEVNULL to avoid MCP interference
            stderr=subprocess.DEVNULL,  # Must be DEVNULL to avoid MCP interference
            env={**os.environ, "DISPLAY": ":1"},
        )

        self._app_processes[app_name] = proc
        self._launched_apps.append(app_name)

        # Wait for frontend to be ready using HTTP health checks
        try:
            await self._wait_for_port(frontend_port, f"app '{app_name}' frontend", timeout=60)

            # Additional HTTP health check for web apps with faster timeout
            if app_name == "todo":
                await self._wait_for_http_health(f"http://localhost:{frontend_port}", timeout=10)

            logger.info(
                f"Launched app '{app_name}' - Frontend: {frontend_port}, Backend: {backend_port}"
            )

            # Store port information for later retrieval
            self._app_ports[app_name] = {"frontend": frontend_port, "backend": backend_port}
        except TimeoutError:
            # Check if process is still running
            if proc.poll() is not None:
                logger.error(f"App '{app_name}' process exited with code {proc.returncode}")
            else:
                logger.error(f"App '{app_name}' failed to become ready within timeout")
            raise

        return {"name": app_name, "url": f"http://localhost:{frontend_port}", "port": frontend_port}

    def get_app_port(self, app_name: str) -> int:
        """Get the port for a running app.

        Args:
            app_name: Name of the app (e.g., 'todo')

        Returns:
            Backend port number where the app is running (for API calls)

        Raises:
            ValueError: If app is not known or not running
        """
        # Check if app is running
        if app_name not in self._app_processes:
            raise ValueError(f"Could not find running app '{app_name}': app not launched")

        # Get the process and check if it's still running
        proc = self._app_processes[app_name]
        if proc.poll() is not None:
            raise ValueError(f"Could not find running app '{app_name}': process has exited")

        # Get stored port information
        if app_name not in self._app_ports:
            raise ValueError(
                f"Could not find port information for app '{app_name}': not properly launched"
            )

        # Return backend port (where APIs are served)
        return self._app_ports[app_name]["backend"]

    def get_app_frontend_port(self, app_name: str) -> int:
        """Get the frontend port for a running app.

        Args:
            app_name: Name of the app (e.g., 'todo')

        Returns:
            Frontend port number where the app UI is served

        Raises:
            ValueError: If app is not known or not running
        """
        # Check if app is running
        if app_name not in self._app_processes:
            raise ValueError(f"Could not find running app '{app_name}': app not launched")

        # Get stored port information
        if app_name not in self._app_ports:
            raise ValueError(
                f"Could not find port information for app '{app_name}': not properly launched"
            )

        # Return frontend port (where UI is served)
        return self._app_ports[app_name]["frontend"]

    async def shutdown(self):
        """Shutdown all services gracefully."""
        # Stop app processes
        for name, proc in self._app_processes.items():
            if proc.poll() is None:
                proc.terminate()
                await asyncio.sleep(1)
                if proc.poll() is None:
                    proc.kill()
                logger.info(f"Terminated app '{name}'")

        # Clear app tracking
        self._app_processes.clear()
        self._app_ports.clear()
        self._launched_apps.clear()

        # Stop services in reverse order
        for proc, name in [
            (self.websockify_proc, "websockify"),
            (self.vnc_proc, "x11vnc"),
            (self.x11_proc, "Xvfb"),
        ]:
            if proc and proc.poll() is None:
                proc.terminate()
                await asyncio.sleep(0.5)
                if proc.poll() is None:
                    proc.kill()
                logger.info(f"Stopped {name}")

    def _is_port_open(self, port: int) -> bool:
        """Check if a port is open."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)
        try:
            result = sock.connect_ex(("localhost", port))
            sock.close()
            return result == 0
        except:
            return False

    def _get_next_port(self) -> int:
        """Get next available port for apps."""
        # Start from 3000 and find first available
        base_port = 3010
        for offset in range(100):  # Support up to 100 apps
            port = base_port + offset
            if not self._is_port_open(port):
                return port
        raise RuntimeError("No available ports")

    async def _wait_for_http_health(self, url: str, timeout: int = 10):
        """Wait for HTTP endpoint to be healthy."""
        import httpx

        async with httpx.AsyncClient() as client:
            for i in range(timeout * 5):  # Check every 200ms instead of 500ms
                try:
                    response = await client.get(url, timeout=0.5)
                    if response.status_code < 500:  # Any non-server-error response is good
                        logger.info(f"HTTP health check passed for {url}")
                        return
                except (httpx.RequestError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(0.2)  # 200ms intervals for faster detection

        raise TimeoutError(f"HTTP health check failed for {url}")

    async def _wait_for_port(self, port: int, service_name: str = "service", timeout: int = 30):
        """Wait for a port to become available."""
        for i in range(timeout * 5):  # Check every 200ms instead of 100ms
            if self._is_port_open(port):
                logger.info(f"{service_name} is ready on port {port}")
                return
            await asyncio.sleep(0.2)  # 200ms intervals
        raise TimeoutError(f"Port {port} did not become available for {service_name}")
