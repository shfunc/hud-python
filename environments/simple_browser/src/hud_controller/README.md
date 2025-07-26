# Launch Scripts

This directory contains scripts that orchestrate the environment components.

## Scripts

- **browser.py** - Launches Playwright browser pointing to the app
- **mcp_tools.py** - MCP server with tools to interact with browser and backend
- **app_starter.py** - Starts the frontend and backend services

## How It Works

1. **app_starter.py** starts the Next.js frontend and FastAPI backend
2. **browser.py** waits for the app to be ready, then opens it in Playwright
3. **mcp_tools.py** exposes tools that can:
   - Control the browser (screenshots, clicks, etc.)
   - Make API calls to the backend
   - Query the SQLite database directly

## Adding Custom Tools

To add new tools, edit `mcp_tools.py`:

```python
# Example: Add a custom database query tool
@mcp.tool()
def query_database(sql: str) -> dict:
    """Execute a SQL query on the app database"""
    conn = sqlite3.connect('/app/backend/app.db')
    # ... implementation
```

The MCP server automatically exposes any registered tools. 