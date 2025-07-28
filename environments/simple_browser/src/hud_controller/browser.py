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
async def launch_browser(url: str = None):
    """Launch Playwright Chromium browser.

    Args:
        url: Optional URL to navigate to. If None, just launches browser without navigation.
    """
    logging.info("Browser launcher starting...")

    # Ensure DISPLAY is set
    os.environ["DISPLAY"] = os.environ.get("DISPLAY", ":1")
    logging.info(f"Using DISPLAY: {os.environ['DISPLAY']}")

    if url:
        logging.info(f"Will navigate to: {url}")
    else:
        logging.info("Launching browser without navigation")

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

        # Navigate only if URL is provided
        if url:
            logging.info(f"Navigating to {url}")
            await page.goto(url)
        else:
            # Navigate to a blank page
            await page.goto("about:blank")

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
    url = sys.argv[1] if len(sys.argv) > 1 else None
    if url:
        logging.info(f"Launching browser with URL: {url}")
    else:
        logging.info("Launching browser without navigation")

    try:
        asyncio.run(launch_browser(url))
    except KeyboardInterrupt:
        logging.info("Browser interrupted by user")
    except Exception as e:
        logging.error(f"Browser launch failed: {e}")
        sys.exit(1)
