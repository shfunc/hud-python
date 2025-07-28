"""Browser automation module."""

import os
import asyncio
import logging
import sys
from playwright.async_api import async_playwright

# Configure logging to stderr to avoid stdio contamination
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] Browser: %(message)s", stream=sys.stderr
)

# ---- BROWSER ----
# The browser is launched by the ServiceManager
# and is used to navigate to the default URL
async def launch_browser(url: str):
    """Launch Playwright Chromium browser."""
    logging.info("Browser launcher starting...")

    # Ensure DISPLAY is set
    os.environ["DISPLAY"] = os.environ.get("DISPLAY", ":1")
    logging.info(f"Using DISPLAY: {os.environ['DISPLAY']}")

    # Default URL - can be overridden by environment variable
    default_url = url
    logging.info(f"Default URL: {default_url}")

    async with async_playwright() as p:
        # Launch Chromium
        browser = await p.chromium.launch(
            headless=False,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-blink-features=AutomationControlled",
                "--window-size=1920,1080",
                "--window-position=0,0",
                "--start-maximized",
                # Lightweight options
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-features=TranslateUI",
                "--disable-ipc-flooding-protection",
                "--disable-default-apps",
                "--no-first-run",
                "--disable-sync",
                "--no-default-browser-check",
            ],
        )

        # Create context and page
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            ignore_https_errors=True,
        )

        page = await context.new_page()

        # Navigate to default URL
        logging.info(f"Navigating to {default_url}")
        await page.goto(default_url)

        # Keep browser running
        logging.info("Browser launched successfully")

        # Keep the browser running indefinitely
        try:
            # Wait for the browser to be closed by the user or external signal
            while True:
                await asyncio.sleep(1)
                # Check if browser is still connected
                if browser.is_connected() is False:
                    break
        except asyncio.CancelledError:
            logging.info("Browser session cancelled")
        except Exception as e:
            logging.error(f"Browser error: {e}")
        finally:
            try:
                await browser.close()
                logging.info("Browser closed")
            except Exception as e:
                logging.error(f"Error closing browser: {e}")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.github.com"
    logging.info(f"Launching browser with URL: {url}")

    try:
        asyncio.run(launch_browser(url))
    except KeyboardInterrupt:
        logging.info("Browser interrupted by user")
    except Exception as e:
        logging.error(f"Browser launch failed: {e}")
        sys.exit(1)
