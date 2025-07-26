"""Browser launcher module."""

import os
import asyncio
import time
import httpx
from playwright.async_api import async_playwright


async def wait_for_url(url: str, timeout: int = 30):
    """Wait for a URL to be ready."""
    print(f"Waiting for {url}...")
    start_time = time.time()
    
    async with httpx.AsyncClient() as client:
        while time.time() - start_time < timeout:
            try:
                response = await client.get(url, timeout=2)
                if response.status_code == 200:
                    print(f"{url} is ready!")
                    return True
            except:
                pass
            await asyncio.sleep(1)
    
    print(f"Warning: {url} did not become ready in time")
    return False


async def launch_browser():
    """Launch Playwright Chromium browser."""
    print("Browser launcher starting...")
    
    # Ensure DISPLAY is set
    os.environ['DISPLAY'] = os.environ.get('DISPLAY', ':1')
    print(f"Using DISPLAY: {os.environ['DISPLAY']}")
    
    # Default URL - can be overridden by environment variable
    default_url = os.environ.get('DEFAULT_URL', 'https://www.google.com')
    print(f"Default URL: {default_url}")
    
    async with async_playwright() as p:
        # Launch Chromium
        browser = await p.chromium.launch(
            headless=False,
            args=[
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process',
                '--disable-blink-features=AutomationControlled',
                '--window-size=1920,1080',
                '--window-position=0,0',
                '--start-maximized',
                # Lightweight options
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                '--disable-features=TranslateUI',
                '--disable-ipc-flooding-protection',
                '--disable-default-apps',
                '--no-first-run',
                '--disable-sync',
                '--no-default-browser-check',
            ]
        )
        
        # Create context and page
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            ignore_https_errors=True,
        )
        
        page = await context.new_page()
        
        # Navigate to default URL
        print(f"Navigating to {default_url}")
        await page.goto(default_url)
        
        # Keep browser running
        print("Browser launched successfully")
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            await browser.close()
            raise


def launch_browser_script():
    """Script entry point for launching browser."""
    asyncio.run(launch_browser()) 