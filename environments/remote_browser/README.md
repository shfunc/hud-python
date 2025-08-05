# HUD Remote Browser MCP Server

This MCP server provides browser automation capabilities using various remote browser providers.

## Running with Docker

The Docker image requires the `BROWSER_PROVIDER` environment variable to be set at runtime.

### Building the Image

```bash
docker build -t hud-remote-browser:latest .
```

### Running the Container

You must specify a browser provider when running the container:

```bash
# Using AnchorBrowser
docker run --rm -i \
  -e BROWSER_PROVIDER=anchorbrowser \
  -e ANCHOR_API_KEY=your-api-key \
  hud-remote-browser:latest

# Using BrowserBase
docker run --rm -i \
  -e BROWSER_PROVIDER=browserbase \
  -e BROWSERBASE_API_KEY=your-api-key \
  -e BROWSERBASE_PROJECT_ID=your-project-id \
  hud-remote-browser:latest
```

## Supported Browser Providers

- **anchorbrowser** - Requires `ANCHOR_API_KEY`
- **browserbase** - Requires `BROWSERBASE_API_KEY` and `BROWSERBASE_PROJECT_ID`
- **hyperbrowser** - Requires `HYPERBROWSER_API_KEY`

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `BROWSER_PROVIDER` | **Yes** | The browser provider to use |
| `LOG_LEVEL` | No | Logging level (default: INFO) |
| Provider-specific API keys | Yes* | Required based on the chosen provider |

## MCP Protocol

The server communicates via stdio using the MCP protocol. Example initialization:

```bash
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {...}}' | \
  docker run --rm -i -e BROWSER_PROVIDER=steel -e STEEL_API_KEY=... hud-remote-browser:latest
```

## Error Handling

If `BROWSER_PROVIDER` is not set, the server will fail with:
```
BROWSER_PROVIDER environment variable is required. Supported providers: anchorbrowser, steel, browserbase, hyperbrowser, kernel
```