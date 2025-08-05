# HUD Remote Browser MCP Server

This MCP server provides browser automation capabilities using various remote browser providers.

## Running with Docker

The Docker image supports both production and development modes using the same Dockerfile.

### Building the Image

```bash
# Production build (default)
docker build -t hud-remote-browser:latest .

# Development build (for hot-reload with volume mounts)
docker build --build-arg DEV_MODE=true -t hud-remote-browser:dev .
```

### Running in Production Mode

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

### Running in Development Mode (Hot Reload)

Development mode allows you to edit code locally and see changes immediately without rebuilding:

```bash
# Windows
docker run --rm -i ^
  -v "%cd%\src:/app/src:rw" ^
  -e BROWSER_PROVIDER=anchorbrowser ^
  -e ANCHOR_API_KEY=your-api-key ^
  -e PYTHONPATH=/app ^
  hud-remote-browser:dev

# Linux/Mac
docker run --rm -i \
  -v "$(pwd)/src:/app/src:rw" \
  -e BROWSER_PROVIDER=anchorbrowser \
  -e ANCHOR_API_KEY=your-api-key \
  -e PYTHONPATH=/app/src \
  hud-remote-browser:dev
```

The `-v` flag mounts your local `src/` directory into the container, allowing instant code changes.

## Supported Browser Providers

- **anchorbrowser** - Requires `ANCHOR_API_KEY`
- **browserbase** - Requires `BROWSERBASE_API_KEY` and `BROWSERBASE_PROJECT_ID`
- **hyperbrowser** - Requires `HYPERBROWSER_API_KEY`
- **steel** - Requires `STEEL_API_KEY`
- **kernel** - No additional requirements

## Environment Variables

### Core Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `BROWSER_PROVIDER` | **Yes** | The browser provider to use |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

### Provider-Specific Variables

| Provider | Required Variables |
|----------|-------------------|
| anchorbrowser | `ANCHOR_API_KEY` |
| browserbase | `BROWSERBASE_API_KEY`, `BROWSERBASE_PROJECT_ID` |
| hyperbrowser | `HYPERBROWSER_API_KEY` |
| steel | `STEEL_API_KEY` |

### Optional Browser Settings

| Variable | Description |
|----------|-------------|
| `HEADLESS` | Whether to run browser in headless mode |
| `DEFAULT_TIMEOUT` | Default timeout for browser operations |
| `WINDOW_WIDTH` | Browser window width |
| `WINDOW_HEIGHT` | Browser window height |
| `PROXY_URL` | HTTP proxy URL |

### Google Cloud Platform (GCP) Credentials

For Google Sheets functionality, you can provide GCP credentials in two formats:

**Option A: Single JSON String**
```bash
-e GCP_CREDENTIALS_JSON='{"type":"service_account","project_id":"...","private_key":"..."}'
```

**Option B: Individual Fields**
```bash
-e GCP_TYPE="service_account" \
-e GCP_PROJECT_ID="your-project-id" \
-e GCP_PRIVATE_KEY_ID="your-key-id" \
-e GCP_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n..." \
-e GCP_CLIENT_EMAIL="your-service-account@project.iam.gserviceaccount.com" \
-e GCP_CLIENT_ID="your-client-id" \
-e GCP_AUTH_URI="https://accounts.google.com/o/oauth2/auth" \
-e GCP_TOKEN_URI="https://oauth2.googleapis.com/token" \
-e GCP_AUTH_PROVIDER_X509_CERT_URL="https://www.googleapis.com/oauth2/v1/certs" \
-e GCP_CLIENT_X509_CERT_URL="https://www.googleapis.com/robot/v1/metadata/x509/..."
```

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