# HUD Controller Environment

A Docker environment that provides a browser with MCP tools and optional web app launching.

## Quick Start

```bash
# Start everything
docker-compose up -d

# View logs
docker-compose logs -f

# Stop everything
docker-compose down

## Configuration

The environment can be configured through environment variables or command-line arguments.

### Environment Variables

- `BROWSER_URL` - URL to navigate browser to (default: `https://google.com`)
- `LAUNCH_APPS` - Comma-separated list of apps to launch (e.g., `todo,calendar`)
- `WAIT_FOR_APPS` - Wait for apps to start before launching browser (default: `true`)
- `FRONTEND_PORT_START` - Starting port for frontend services (default: `3000`)
- `BACKEND_PORT_START` - Starting port for backend services (default: `5000`)
- `PORT_ALLOCATION` - Port allocation strategy: `auto` or `manual` (default: `auto`)

### Examples

#### Launch with a specific app:
```bash
docker run -e LAUNCH_APPS=todo -p 8080:8080 hud-browser
```

## Access Points

- **MCP Server**: http://localhost:8041/mcp
- **Debug View**: http://localhost:8080/vnc.html (watch the browser)

#### Command-line override:
```bash
# The container also accepts positional arguments
docker run -p 8080:8080 hud-browser todo http://localhost:3000
```

## Access Points

- **VNC Viewer**: http://localhost:8080/vnc.html
- **MCP Server**: http://localhost:8041/mcp
- **Apps**: Depend on configuration (default starts at port 3000)

## Available MCP Tools

- `computer` - Control the browser (click, type, screenshot, etc.)
- `api_request` - Make HTTP requests to backend APIs
- `query_database` - Execute SQL queries on app databases

## Port Allocation

When `PORT_ALLOCATION=auto` (default), apps are automatically assigned ports:
- First app: frontend=3000, backend=5000
- Second app: frontend=3010, backend=5010
- And so on...

When `PORT_ALLOCATION=manual`, apps use their default ports. 