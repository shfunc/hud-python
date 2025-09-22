"""Controller lifecycle hooks.

Provides a shutdown hook that:
- Requests the environment backend to shutdown
- Closes local tool/resources (Playwright, HTTP client)
"""

import contextlib
import logging
import sys

from controller import mcp, http_client
from controller.tools import playwright

logger = logging.getLogger(__name__)


async def shutdown_env() -> str:
    """Ask the environment backend to shutdown."""
    with contextlib.redirect_stdout(sys.stderr):
        try:
            resp = await http_client.post("/shutdown")
            if resp.status_code in (200, 204):
                logger.info("Requested environment shutdown")
                return "Environment shutdown requested"
            logger.warning("Environment /shutdown returned %s", resp.status_code)
            return f"Environment shutdown returned {resp.status_code}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to call /shutdown: %s", exc)
            return f"Failed to request environment shutdown: {exc}"


@mcp.shutdown
async def on_shutdown() -> None:
    """Graceful controller shutdown: remote then local cleanup."""
    with contextlib.redirect_stdout(sys.stderr):
        try:
            await shutdown_env()
        except Exception as exc:  # noqa: BLE001
            logger.warning("shutdown_env failed: %s", exc)

        # Close Playwright browser/session if present
        try:
            await playwright.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Playwright close failed: %s", exc)

        # Close shared HTTP client
        try:
            await http_client.aclose()
        except Exception as exc:  # noqa: BLE001
            logger.warning("HTTP client close failed: %s", exc)


